"""
ovos_branch_bound.py

Global optimization of OVOS using branch‑and‑bound with convex relaxations.
Inspired by topics from Convex Optimization II (Stanford EE364b).

This version features:
  - True MP2 energy evaluation for upper bounds (local optimisation within the box).
  - A convex quadratic lower bound (shifted Hessian) as a pruning heuristic.
  - Clever variable selection for large problems.
  - Progress reporting with elapsed time, nodes, and gap.
"""

import numpy as np
import scipy.linalg
import heapq
import time
from scipy.optimize import minimize, Bounds
from ovos import OVOS

class OVOSBranchBound(OVOS):
    def __init__(self, *args,
                 global_tol=1e-4,
                 max_nodes=10000,
                 branch_strategy='coordinate',
                 max_bb_vars=12,
                 progress_interval=100,
                 **kwargs):
        """
        Parameters
        ----------
        global_tol : float
            Convergence tolerance for the global optimality gap.
        max_nodes : int
            Maximum number of nodes to explore before stopping.
        branch_strategy : str
            'coordinate' (split along the widest coordinate) or 'largest' (default).
        max_bb_vars : int
            Maximum number of variables for which B&B is attempted.
        progress_interval : int
            Print progress every `progress_interval` nodes.
        """
        super().__init__(*args, **kwargs)
        self.global_tol = global_tol
        self.max_nodes = max_nodes
        self.branch_strategy = branch_strategy
        self.max_bb_vars = max_bb_vars
        self.progress_interval = progress_interval
        self._grad_norm_history = []

    def _select_variables(self, g_vec, n_vars, n_select):
        """Select variables with largest absolute gradient."""
        abs_grad = np.abs(g_vec)
        sorted_idx = np.argsort(abs_grad)[::-1]
        return sorted_idx[:n_select].tolist(), sorted_idx[n_select:].tolist()

    def _compute_true_mp2_energy(self, R_vec, mo_coeffs, fock_spin):
        """Evaluate the true MP2 correlation energy for a given rotation vector R."""
        mo_rot, fock_rot, _ = self._rotate_orbitals(mo_coeffs, fock_spin, R_vec)
        mo_rot, fock_rot, _ = self._canonicalize_active(mo_rot, fock_rot, None)
        eri_as = self._eri_vovo_antisym(mo_rot)
        eps = np.diag(fock_rot)
        t_abij = self._mp1_amplitudes(eps, eri_as)
        return self._mp2_energy(fock_rot, t_abij, eri_as)

    def _newton_step(self, G, H, iteration, start_counting):
        """Attempt global branch‑and‑bound; fallback to Newton for large problems."""
        nvir_act = len(self.active_inocc_indices)
        ninact = len(self.inactive_indices)
        n_vars = nvir_act * ninact

        if ninact == 0:
            return np.zeros(n_vars)

        g_vec = G.flatten()
        H_matrix = H

        if n_vars <= self.max_bb_vars:
            if self.verbose:
                print(f"  B&B on all {n_vars} variables.")
            return self._run_branch_and_bound(g_vec, H_matrix, n_vars,
                                              self.mo_coeffs, self.fock_spin)

        n_select = self.max_bb_vars
        if self.verbose:
            print(f"  Selecting {n_select} most sensitive variables for B&B (out of {n_vars}).")
        selected_idx, unselected_idx = self._select_variables(g_vec, n_vars, n_select)

        R_full = super()._newton_step(G, H, iteration, start_counting)

        H_ss = H_matrix[np.ix_(selected_idx, selected_idx)]
        g_s = g_vec[selected_idx]
        H_su = H_matrix[np.ix_(selected_idx, unselected_idx)]
        u_fixed = R_full[unselected_idx]
        g_red = g_s + H_su @ u_fixed

        trust = getattr(self, 'trust_radius', 5.0)
        lb = -trust * np.ones(n_select)
        ub =  trust * np.ones(n_select)

        R_red = self._run_branch_and_bound(g_red, H_ss, n_select, lb, ub,
                                           mo_coeffs=self.mo_coeffs,
                                           fock_spin=self.fock_spin,
                                           selected_idx=selected_idx,
                                           unselected_idx=unselected_idx,
                                           R_full=R_full)

        R = np.zeros(n_vars)
        R[selected_idx] = R_red
        R[unselected_idx] = u_fixed
        return R

    def _run_branch_and_bound(self, g_vec, H_matrix, n_vars, lb=None, ub=None,
                              mo_coeffs=None, fock_spin=None,
                              selected_idx=None, unselected_idx=None, R_full=None):
        """
        Core B&B solver with progress reporting.
        """
        if lb is None:
            trust = getattr(self, 'trust_radius', 5.0)
            lb = -trust * np.ones(n_vars)
            ub =  trust * np.ones(n_vars)

        def true_energy_reduced(R_red):
            if selected_idx is not None and unselected_idx is not None and R_full is not None:
                R_full_local = R_full.copy()
                R_full_local[selected_idx] = R_red
                R_full_local[unselected_idx] = R_full[unselected_idx]
                return self._compute_true_mp2_energy(R_full_local, mo_coeffs, fock_spin)
            else:
                return self._compute_true_mp2_energy(R_red, mo_coeffs, fock_spin)

        def quad_energy(x):
            return 0.5 * x @ H_matrix @ x + g_vec @ x

        # Initial upper bound
        centre = (lb + ub) / 2.0
        res = minimize(lambda x: true_energy_reduced(x), centre,
                       method='L-BFGS-B', bounds=list(zip(lb, ub)),
                       options={'maxiter': 20, 'ftol': 1e-6})
        if res.success:
            best_R = res.x
            best_upper = res.fun
        else:
            best_R = centre
            best_upper = true_energy_reduced(centre)

        heap = []
        node_id = 0
        lower_bound = self._convex_lower_bound(centre, g_vec, H_matrix, lb, ub)
        heapq.heappush(heap, (lower_bound, node_id, lb.copy(), ub.copy()))
        node_id += 1

        # --- Progress tracking ---
        start_time = time.time()
        last_print_time = start_time
        nodes_explored = 0
        last_gap = None

        while heap and nodes_explored < self.max_nodes:
            lb_node, _, lb_node_box, ub_node_box = heapq.heappop(heap)
            nodes_explored += 1

            # Print progress if verbose and interval reached or time passed
            if self.verbose >= 1 and (nodes_explored % self.progress_interval == 0 or
                                      time.time() - last_print_time > 10.0):
                elapsed = time.time() - start_time
                gap = best_upper - heap[0][0] if heap else 0.0
                print(f"    B&B: {nodes_explored} nodes | "
                      f"best E = {best_upper:.6e} | "
                      f"gap = {gap:.2e} | "
                      f"time = {elapsed:.1f}s")
                last_print_time = time.time()
                last_gap = gap

            if lb_node >= best_upper - self.global_tol:
                continue

            widths = ub_node_box - lb_node_box
            idx = np.argmax(widths)
            mid = (lb_node_box[idx] + ub_node_box[idx]) / 2.0

            for (l_b, u_b) in [
                (lb_node_box.copy(), ub_node_box.copy()),
                (lb_node_box.copy(), ub_node_box.copy())
            ]:
                if (l_b == lb_node_box).all() and (u_b == ub_node_box).all():
                    u_b[idx] = mid
                else:
                    l_b[idx] = mid
                if np.max(u_b - l_b) < 1e-8:
                    continue

                centre_child = (l_b + u_b) / 2.0
                low = self._convex_lower_bound(centre_child, g_vec, H_matrix, l_b, u_b)

                res_child = minimize(lambda x: true_energy_reduced(x), centre_child,
                                     method='L-BFGS-B', bounds=list(zip(l_b, u_b)),
                                     options={'maxiter': 10, 'ftol': 1e-5})
                if res_child.success:
                    R_feas = res_child.x
                    upper = res_child.fun
                else:
                    R_feas = centre_child
                    upper = true_energy_reduced(centre_child)

                if upper < best_upper:
                    best_upper = upper
                    best_R = R_feas

                if low < best_upper - self.global_tol:
                    heapq.heappush(heap, (low, node_id, l_b, u_b))
                    node_id += 1

        # Final summary
        if self.verbose >= 1:
            elapsed = time.time() - start_time
            final_gap = best_upper - heap[0][0] if heap else 0.0
            print(f"    B&B finished after {nodes_explored} nodes "
                  f"({elapsed:.1f}s), gap = {final_gap:.2e}, best E = {best_upper:.6e}")

        return best_R

    def _convex_lower_bound(self, R, g_vec, H_matrix, lb, ub):
        """Heuristic convex lower bound using shifted Hessian."""
        eigvals = np.linalg.eigvalsh(H_matrix)
        lambda_min = np.min(eigvals)
        shift = max(0.0, -lambda_min + 1e-6)
        H_pos = H_matrix + shift * np.eye(len(R))
        res = minimize(lambda x: 0.5 * x @ H_pos @ x + g_vec @ x,
                       R, method='trust-constr',
                       bounds=Bounds(lb, ub),
                       options={'maxiter': 50, 'verbose': 0})
        if res.success:
            return res.fun
        else:
            return 0.5 * R @ H_pos @ R + g_vec @ R