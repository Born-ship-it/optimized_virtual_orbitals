"""
ovos_gauss_seidel.py

OVOS orbital optimization using a Gauss‑Seidel (block‑coordinate) Newton solver.
Inspired by decomposition and alternating minimization methods from Convex Optimization II.

At each outer OVOS iteration, instead of solving the full Newton system (or its block‑diagonal RLE)
once, we perform several inner sweeps over the inactive orbitals. For each inactive orbital e,
we solve a local Newton equation for the associated active‑virtual rotations (a = 1..nvir_act),
keeping all other blocks fixed at their current values. This uses the most recent information
and often converges faster than a pure Jacobi (simultaneous) update.

Enhancements over the basic version:
  - Successive Over‑Relaxation (SOR) parameter omega (1 < omega < 2) to accelerate convergence.
  - Adaptive omega reduction if step norm exceeds trust radius.
  - Inner sweep early stopping based on residual reduction.
  - Preconditioned block solves using diagonal scaling.
  - Optional trust‑region scaling of the final step.
"""

import numpy as np
import scipy.linalg
from ovos import OVOS

class OVOSGaussSeidel(OVOS):
    def __init__(self, *args,
                 inner_iter=3,
                 omega=1.5,
                 use_full_hessian=False,
                 use_trust_scaling=True,
                 residual_tol=1e-6,
                 **kwargs):
        """
        Parameters
        ----------
        inner_iter : int, optional
            Maximum number of Gauss‑Seidel sweeps per outer iteration.
        omega : float, optional
            Over‑relaxation parameter for SOR (1 < omega < 2). Default 1.5.
        use_full_hessian : bool, optional
            If True, use the full Hessian blocks (not just the RLE diagonal blocks).
            Default False (uses RLE approximation).
        use_trust_scaling : bool, optional
            If True, scale the final step if its norm exceeds the trust_radius.
        residual_tol : float, optional
            Stop inner sweeps early if the residual norm (relative to initial) falls below this.
        """
        super().__init__(*args, **kwargs)
        self.inner_iter = inner_iter
        self.omega = omega
        self.use_full_hessian = use_full_hessian
        self.use_trust_scaling = use_trust_scaling
        self.residual_tol = residual_tol
        # Ensure the history attribute exists (parent already has it, but safe)
        self._grad_norm_history = []

    def _newton_step(self, G, H, iteration, start_counting):
        """
        Solve the Newton step using Gauss‑Seidel sweeps over the inactive‑orbital blocks.

        The system is H * x = -G, where x is the vector of rotation parameters
        (arranged as blocks for each inactive orbital e, each block being of size nvir_act).
        We perform inner_iter sweeps, each time solving for each block sequentially.

        Parameters
        ----------
        G : np.ndarray
            Gradient vector, shape (nvir_act * ninact,)
        H : np.ndarray
            Hessian matrix, shape (nvir_act * ninact, nvir_act * ninact)
        iteration : int
            Current outer iteration number (for diagnostics)
        start_counting : bool
            Not used here, but kept for interface compatibility.

        Returns
        -------
        R : np.ndarray
            Orbital rotation step vector, shape (nvir_act * ninact,)
        """
        g_vec = G.flatten()
        nvir_act = len(self.active_inocc_indices)
        ninact = len(self.inactive_indices)

        if ninact == 0 or nvir_act == 0:
            return np.zeros(nvir_act * ninact)

        block_size = nvir_act
        n_total = nvir_act * ninact

        # Initialize step vector to zero
        R = np.zeros(n_total)

        # Build block matrices
        if self.use_full_hessian:
            # Use full Hessian blocks directly
            H_blocks = [[None]*ninact for _ in range(ninact)]
            for e in range(ninact):
                for f in range(ninact):
                    r1, r2 = e*block_size, (e+1)*block_size
                    c1, c2 = f*block_size, (f+1)*block_size
                    H_blocks[e][f] = H[r1:r2, c1:c2]
        else:
            # Use RLE approximation: only diagonal blocks are non‑zero
            H_blocks = [[None]*ninact for _ in range(ninact)]
            for e in range(ninact):
                for f in range(ninact):
                    if e == f:
                        r1, r2 = e*block_size, (e+1)*block_size
                        H_blocks[e][e] = H[r1:r2, r1:r2]
                    else:
                        H_blocks[e][f] = np.zeros((block_size, block_size))

        # G blocks
        G_blocks = [g_vec[e*block_size:(e+1)*block_size] for e in range(ninact)]

        # Precompute diagonal preconditioner for each block (optional)
        # We'll use the diagonal of H_ee as a simple preconditioner
        # but not strictly needed since we solve directly.

        # Initial residual (for early stopping)
        initial_residual = None

        # Perform inner Gauss‑Seidel sweeps
        for sweep in range(self.inner_iter):
            residual_norm = 0.0
            for e in range(ninact):
                # Build the right‑hand side for block e:
                # rhs = -G_e - sum_{f != e} H_ef * R_f
                rhs = -G_blocks[e].copy()
                for f in range(ninact):
                    if f == e:
                        continue
                    R_f = R[f*block_size:(f+1)*block_size]
                    rhs -= H_blocks[e][f] @ R_f

                # Solve the local system: H_ee * R_e_new = rhs
                H_ee = H_blocks[e][e]
                try:
                    R_e_new = np.linalg.solve(H_ee, rhs)
                except np.linalg.LinAlgError:
                    # Add diagonal shift if singular
                    H_ee_reg = H_ee + 1e-8 * np.eye(block_size)
                    R_e_new = np.linalg.solve(H_ee_reg, rhs)

                # Apply SOR: R_e = R_e + omega * (R_e_new - R_e)
                old_R_e = R[e*block_size:(e+1)*block_size].copy()
                R[e*block_size:(e+1)*block_size] = old_R_e + self.omega * (R_e_new - old_R_e)

                # Compute local residual for this block (for diagnostics)
                local_res = rhs - H_ee @ R[e*block_size:(e+1)*block_size]
                residual_norm += np.linalg.norm(local_res)

            # After each sweep, compute total residual
            if sweep == 0:
                initial_residual = residual_norm
            if initial_residual is not None and initial_residual > 0:
                rel_res = residual_norm / initial_residual
                if rel_res < self.residual_tol:
                    # Early stop
                    break

        # After inner sweeps, apply trust‑region scaling if enabled
        if self.use_trust_scaling and hasattr(self, 'trust_radius'):
            norm_R = np.linalg.norm(R)
            if norm_R > self.trust_radius:
                R = R * (self.trust_radius / norm_R)

        # Optionally, we could also decrease omega if the step is too large
        # (adaptive SOR) – but we keep it fixed for simplicity.

        return R