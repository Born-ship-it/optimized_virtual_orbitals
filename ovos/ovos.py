"""
OVOS (Optimized Virtual Orbital Space) method

Minimizes the second-order correlation energy (MP2) by rotating virtual orbitals.

Spaces:
    Active occupied   (I,J): "Inactive"
    Active virtual    (A,B): "Active"
    Inactive virtual  (E,F): "Virtual"

Reference:
    L. Adamowicz & R. J. Bartlett (1987), J. Chem. Phys. 86, 6314
    https://pubs.aip.org/aip/jcp/article/86/11/6314/93345
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg
import pyscf
from pyscf import ao2mo

# Module metadata
import time
_LOAD_TIME = time.time()
_MODULE_VERSION = "dev_build"

# Limit OpenBLAS threads to avoid oversubscription in parallel runs
os.environ['NUMBA_THREADING_LAYER'] = 'omp'  # Use OpenMP (thread-safe)
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

np.set_printoptions(precision=4, suppress=False, linewidth=200)


class OVOS:
    """
    Minimizes the MP2 correlation energy via orbital rotations (OVOS algorithm).

    Parameters
    ----------
    mol : pyscf.gto.Mole
    scf : RHF or UHF object
    Fao : list of np.ndarray
        AO Fock matrices [alpha, beta].
    num_opt_virtual_orbs : int
        Number of optimized virtual spin orbitals (active virtuals).
    mo_coeff : list of np.ndarray
        Initial MO coefficients [alpha, beta].
    init_orbs : str, optional
        'RHF' or 'UHF' (unused, kept for compatibility).
    verbose : int, optional
        Verbosity level. 0 = silent, 1 = normal output.
    max_iter : int, optional
        Maximum number of OVOS iterations.
    conv_energy : float, optional
        Convergence threshold on absolute change in MP2 energy.
    conv_grad : float, optional
        Convergence threshold on gradient norm.
    keep_track_max : int, optional
        Number of consecutive converged iterations required to stop.
    lambda_init : float, optional
        Initial Levenberg‑Marquardt damping parameter.
    lambda_max : float, optional
        Maximum allowed damping parameter.
    trust_radius : float, optional
        Initial trust radius for step size control.
    hessian_reg : float, optional
        Regularization added to Hessian diagonal when singular.
    """

    def __init__(self, mol, scf, Fao, num_opt_virtual_orbs, mo_coeff,
                 init_orbs="RHF", verbose=1,
                 max_iter=1000, conv_energy=1e-8, conv_grad=1e-4,
                 keep_track_max=50,
                 lambda_init=1e-3, lambda_max=1e3,
                 trust_radius=5, hessian_reg=1e-8):

        self.verbose = verbose
        self.mol = mol
        self.num_opt_virtual_orbs = num_opt_virtual_orbs
        self.init_orbs = init_orbs
        self.mo_coeffs = mo_coeff
        self.Fao = Fao

        self.scf = scf
        self.e_rhf = scf.e_tot
        self.h_nuc = mol.energy_nuc()

        # Convergence parameters
        self.max_iter = max_iter
        self.conv_energy = conv_energy
        self.conv_grad = conv_grad
        self.t1_conv = conv_grad
        self.keep_track_max = keep_track_max

        # Optimization parameters
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.trust_radius = trust_radius
        self.hessian_reg = hessian_reg

        # AO integrals
        self.S = mol.intor('int1e_ovlp')
        self.hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        self.eri_4fold_ao = mol.intor('int2e_sph', aosym=1)

        # Orbital counts
        n_spatial = self.mo_coeffs[0].shape[0]
        self.tot_num_spin_orbs = 2 * n_spatial
        assert self.mo_coeffs[0].shape[0] == self.mo_coeffs[1].shape[0], \
            "Alpha and beta MO coeff matrices must have same number of rows"

        self.nelec = mol.nelec[0] + mol.nelec[1]

        # Index lists for subspaces
            # Occupied: 0 to nelec-1 (active)
        self.active_occ_indices = list(range(self.nelec))
            # Active virtual: next num_opt_virtual_orbs after occupied
        self.active_inocc_indices = list(range(
            self.active_occ_indices[-1] + 1,
            self.nelec + self.num_opt_virtual_orbs
        ))
            # Inactive virtual: the rest of the virtual orbitals
        self.virtual_inocc_indices = list(range(
            self.active_inocc_indices[-1] + 1,
            self.tot_num_spin_orbs
        ))
        self.inactive_indices = self.virtual_inocc_indices
        self.full_indices = list(range(self.tot_num_spin_orbs))

        assert self.tot_num_spin_orbs >= self.num_opt_virtual_orbs + self.nelec, \
            f"'num_opt_virtual_orbs' ({num_opt_virtual_orbs}) too large for basis set"

        if self.verbose:
            self._print()
            self._print("#### Active and inactive spaces ####")
            self._print("Total number of spin-orbitals:   ", self.tot_num_spin_orbs)
            self._print("Active occupied spin-orbitals:   ", self.active_occ_indices)
            self._print("Active unoccupied spin-orbitals: ", self.active_inocc_indices)
            self._print("Inactive unoccupied spin-orbs:   ", self.inactive_indices)
            self._print()

    # -------------------------------------------------------------------------
    # Print helper
    # -------------------------------------------------------------------------
    def _print(self, *args, **kwargs):
        """Print with CRLF line endings and immediate flush."""
        kwargs.setdefault('end', '\r\n')
        kwargs.setdefault('flush', True)
        print(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Spin‑orbital conversion helpers
    # -------------------------------------------------------------------------
    def _spatial_to_spin_energies(self, mo_energy_a: np.ndarray,
                                   mo_energy_b: np.ndarray) -> np.ndarray:
        """Interleave alpha/beta spatial MO energies into spin‑orbital order."""
        eps = np.empty(2 * len(mo_energy_a))
        eps[0::2] = mo_energy_a
        eps[1::2] = mo_energy_b
        return eps

    def _build_spin_fock(self, Fmo_a: np.ndarray, Fmo_b: np.ndarray) -> np.ndarray:
        """Build block‑diagonal spin‑orbital Fock matrix (even alpha, odd beta)."""
        n_spin = 2 * self.Fao[0].shape[0]
        Fspin = np.zeros((n_spin, n_spin))
        Fspin[0::2, 0::2] = Fmo_a
        Fspin[1::2, 1::2] = Fmo_b
        return Fspin

    # -------------------------------------------------------------------------
    # Integral transformation
    # -------------------------------------------------------------------------
    def _eri_vovo_antisym(self, mo_coeffs: List[np.ndarray]) -> np.ndarray:
        """
        Transform (vir,occ|vir,occ) AO integrals to MO basis, assemble spin‑orbital
        and antisymmetrize. Returns <ab||ij> in physicist notation (a,b,i,j).
        """
        nocc_a, nocc_b = self.mol.nelec
        C_occ_a = mo_coeffs[0][:, :nocc_a]
        C_occ_b = mo_coeffs[1][:, :nocc_b]
        C_vir_a = mo_coeffs[0][:, nocc_a:]
        C_vir_b = mo_coeffs[1][:, nocc_b:]
        nvir_a = C_vir_a.shape[1]
        nvir_b = C_vir_b.shape[1]

        def _ao2mo(C_left, C_row, C_right, C_col):
            # returns (left,row|right,col) in chemist notation
            return ao2mo.kernel(self.eri_4fold_ao,
                                [C_left, C_row, C_right, C_col],
                                compact=False)

        # Chemist blocks (vir,occ|vir,occ)
        vovo_aaaa = _ao2mo(C_vir_a, C_occ_a, C_vir_a, C_occ_a).reshape(
            nvir_a, nocc_a, nvir_a, nocc_a)
        vovo_bbbb = _ao2mo(C_vir_b, C_occ_b, C_vir_b, C_occ_b).reshape(
            nvir_b, nocc_b, nvir_b, nocc_b)
        vovo_aabb = _ao2mo(C_vir_a, C_occ_a, C_vir_b, C_occ_b).reshape(
            nvir_a, nocc_a, nvir_b, nocc_b)

        nvir_spin = 2 * nvir_a
        nocc_spin = nocc_a + nocc_b

        # Assemble spin‑orbital chemist tensor
        vovo_spin = np.zeros((nvir_spin, nocc_spin, nvir_spin, nocc_spin))
        vovo_spin[0::2, 0::2, 0::2, 0::2] = vovo_aaaa
        vovo_spin[1::2, 1::2, 1::2, 1::2] = vovo_bbbb
        vovo_spin[0::2, 0::2, 1::2, 1::2] = vovo_aabb
        vovo_spin[1::2, 1::2, 0::2, 0::2] = vovo_aabb.transpose(2, 3, 0, 1)

        # Convert to physicist (a,i,b,j) -> (a,b,i,j) and antisymmetrize
        eri_phys = vovo_spin.transpose(0, 2, 1, 3)
        eri_as = eri_phys - eri_phys.transpose(0, 1, 3, 2)
        return eri_as

    # -------------------------------------------------------------------------
    # MP2 energy and amplitude computation
    # -------------------------------------------------------------------------
    def _mp1_amplitudes(self, fock_diag: np.ndarray,
                         eri_as: np.ndarray) -> np.ndarray:
        """
        Compute MP1 amplitudes t_ij^ab = -<ab||ij> / (f_aa + f_bb - f_ii - f_jj).
        Only the active occupied and active virtual subspaces are used.
        """
        occ_idx = self.active_occ_indices
        vir_idx = self.active_inocc_indices
        nvir_act = len(vir_idx)

        eps_occ = fock_diag[occ_idx]
        eps_vir = fock_diag[vir_idx]

        denom = (eps_vir[:, None, None, None] +
                 eps_vir[None, :, None, None] -
                 eps_occ[None, None, :, None] -
                 eps_occ[None, None, None, :])

        return -eri_as[:nvir_act, :nvir_act, :, :] / denom

    def _mp2_energy(self, fock_spin: np.ndarray, t_abij: np.ndarray,
                    eri_as: np.ndarray) -> float:
        """
        MP2 correlation energy using full virtual‑virtual Fock block to handle
        non‑canonical orbitals correctly.
        """
        occ_abs = np.array(self.active_occ_indices)
        vir_abs = np.array(self.active_inocc_indices)
        nocc = len(occ_abs)
        nvir = len(vir_abs)

        if nocc == 0 or nvir == 0:
            return 0.0

        a_loc, b_loc = np.triu_indices(nvir, k=1)          # a < b
        F_virt = fock_spin[np.ix_(vir_abs, vir_abs)]
        chunk_size = min(500, len(a_loc))
        n_pairs = len(a_loc)

        # Pre‑compute all (i,j) with i > j
        occ_pairs = [(i, j) for i in range(nocc) for j in range(i)]

        energy = 0.0
        for loc_i, loc_j in occ_pairs:
            abs_i = occ_abs[loc_i]
            abs_j = occ_abs[loc_j]
            eps_ij_sum = fock_spin[abs_i, abs_i] + fock_spin[abs_j, abs_j]

            t_ab = t_abij[a_loc, b_loc, loc_i, loc_j]       # shape (n_pairs,)
            g_ab = eri_as[a_loc, b_loc, loc_i, loc_j]       # shape (n_pairs,)

            # Diagonal part (2 * t · g)
            term2 = 2.0 * np.dot(t_ab, g_ab)

            # Off‑diagonal part
            term1 = 0.0
            for r0 in range(0, n_pairs, chunk_size):
                r1 = min(r0 + chunk_size, n_pairs)
                t_row = t_ab[r0:r1, None]
                ar, br = a_loc[r0:r1], b_loc[r0:r1]

                for c0 in range(0, n_pairs, chunk_size):
                    c1 = min(c0 + chunk_size, n_pairs)
                    t_col = t_ab[None, c0:c1]
                    ac, bc = a_loc[c0:c1], b_loc[c0:c1]

                    # Masks for index equalities
                    d_ac = ar[:, None] == ac[None, :]
                    d_bd = br[:, None] == bc[None, :]
                    d_ad = ar[:, None] == bc[None, :]
                    d_bc = br[:, None] == ac[None, :]

                    bracket = np.zeros((r1 - r0, c1 - c0))
                    if np.any(d_bd):
                        bracket += F_virt[ar[:, None], ac[None, :]] * d_bd
                    if np.any(d_ac):
                        bracket += F_virt[br[:, None], bc[None, :]] * d_ac
                    if np.any(d_bc):
                        bracket -= F_virt[ar[:, None], bc[None, :]] * d_bc
                    if np.any(d_ad):
                        bracket -= F_virt[br[:, None], ac[None, :]] * d_ad

                    mask_ac_bd = d_ac & d_bd
                    if np.any(mask_ac_bd):
                        bracket[mask_ac_bd] -= eps_ij_sum
                    mask_ad_bc = d_ad & d_bc
                    if np.any(mask_ad_bc):
                        bracket[mask_ad_bc] += eps_ij_sum

                    term1 += np.sum(t_row * bracket * t_col)

            energy += term1 + term2

        return energy

    def _compute_density(self, t_abij: np.ndarray) -> np.ndarray:
        """
        Virtual‑virtual density matrix D_ab = ∑_{i>j,c} t_ac^ij t_bc^ij.
        """
        nocc = t_abij.shape[2]
        i_idx, j_idx = np.tril_indices(nocc, k=-1)
        t_ij = t_abij[:, :, i_idx, j_idx]          # (nvir, nvir, n_pairs)
        D_ab = np.einsum('acp,bcp->ab', t_ij, t_ij, optimize=True)
        D_ab[np.abs(D_ab) < 1e-12] = 0.0
        return D_ab

    # -------------------------------------------------------------------------
    # Gradient and Hessian
    # -------------------------------------------------------------------------
    def _gradient(self, t_abij: np.ndarray, eri_as: np.ndarray,
                  D_ab: np.ndarray, fock_spin: np.ndarray) -> np.ndarray:
        """
        Gradient G[a,e] = 2∑_{i>j,b} t_ab^ij <eb||ij> + 2∑_b D_ab F_be
        """
        vir = np.array(self.active_inocc_indices)
        inactive = np.array(self.inactive_indices)
        nvir = len(vir)
        ninact = len(inactive)

        if ninact == 0:
            return np.zeros((nvir, 0))

        nocc = t_abij.shape[2]
        i_idx, j_idx = np.tril_indices(nocc, k=-1)

        # <eb||ij> for e in inactive, b in active, i>j
        eri_eb = eri_as[nvir:, :nvir, :, :][:, :, i_idx, j_idx]  # (ninact, nvir, n_pairs)
        t_ij = t_abij[:, :, i_idx, j_idx]                         # (nvir, nvir, n_pairs)

        G = 2.0 * np.einsum('abp,ebp->ae', t_ij, eri_eb, optimize=True)
        F_be = fock_spin[np.ix_(vir, inactive)]                   # (nvir, ninact)
        G += 2.0 * D_ab @ F_be
        return G

    def _hessian(self, t_abij: np.ndarray, eri_as: np.ndarray,
                 D_ab: np.ndarray, fock_spin: np.ndarray) -> np.ndarray:
        """
        Hessian H_{ae,bf} = 2∑_{i>j} t_ab^ij <ef||ij>
                          - [∑_{c,i>j} (t_ac^ij <bc||ij> + t_cb^ij <ca||ij>)] δ_ef
                          + D_ab (f_aa + f_bb) δ_ef
                          + D_ab f_ef (1 - δ_ef)
        """
        vir = np.array(self.active_inocc_indices)
        inactive = np.array(self.inactive_indices)
        nvir = len(vir)
        ninact = len(inactive)

        if ninact == 0:
            return np.zeros((0, 0))

        nocc = t_abij.shape[2]
        i_idx, j_idx = np.tril_indices(nocc, k=-1)

        fock_vir_diag = np.diag(fock_spin)[vir]                     # (nvir,)
        fock_inact = fock_spin[np.ix_(inactive, inactive)]          # (ninact, ninact)
        fock_inact_offdiag = fock_inact.copy()
        np.fill_diagonal(fock_inact_offdiag, 0.0)

        t_ij = t_abij[:, :, i_idx, j_idx]                            # (nvir, nvir, n_pairs)
        eri_ef = eri_as[nvir:, nvir:, :, :][:, :, i_idx, j_idx]      # (ninact, ninact, n_pairs)
        eri_ab = eri_as[:nvir, :nvir, :, :][:, :, i_idx, j_idx]      # (nvir, nvir, n_pairs)

        # Term 1: 2 t_ab^p <ef||ij>^p  -> reshape to (a,e,b,f)
        term1 = 2.0 * np.einsum('abp,efp->aebf', t_ij, eri_ef, optimize=True)
        H = term1.reshape(nvir * ninact, nvir * ninact)

        # Terms 2 & 3 (diagonal in e=f)
        term2 = -(
            np.einsum('acp,bcp->ab', t_ij, eri_ab, optimize=True) +
            np.einsum('cbp,cap->ab', t_ij, eri_ab, optimize=True)
        )
        term3 = D_ab * (fock_vir_diag[:, None] + fock_vir_diag[None, :])
        H += np.kron(term2 + term3, np.eye(ninact)).reshape(nvir * ninact, nvir * ninact)

        # Term 4 (off‑diagonal e≠f): D_ab f_ef
        H += np.kron(D_ab, fock_inact_offdiag).reshape(nvir * ninact, nvir * ninact)

        return H

    # -------------------------------------------------------------------------
    # Orbital rotation and canonicalization
    # -------------------------------------------------------------------------
    def _rotate_orbitals(self, mo_coeffs: List[np.ndarray],
                        fock_spin: np.ndarray,
                        R_vec: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Apply orbital rotation U = exp(R) where R is antisymmetric.
        
        Rotation only affects VIRTUAL orbitals (active + inactive).
        Occupied orbitals are never rotated.
        
        Key: Build R with correct absolute indexing so occupied block stays as identity.
        """
        n_occ = len(self.active_occ_indices)       # Occupied spin-orbitals
        nvir_act = len(self.active_inocc_indices)  # Active virtual spin-orbitals
        ninact = len(self.inactive_indices)        # Inactive virtual spin-orbitals
        n_spin = n_occ + nvir_act + ninact         # Total spin-orbitals
        
        # Reshape R_vec into local matrix form: R_local[a, e]
        R_local = R_vec.reshape(nvir_act, ninact)
        
        # ===== BUILD FULL ANTISYMMETRIC R MATRIX IN SPIN-ORBITAL SPACE =====
        # Occupied block (0:n_occ, 0:n_occ) stays ZERO (identity under exp)
        # Virtual block (n_occ:n_spin, n_occ:n_spin) contains the rotation
        R_full = np.zeros((n_spin, n_spin), dtype=np.float64)
        
        # Fill active-inactive coupling in the VIRTUAL SUBSPACE
        for a in range(nvir_act):
            for e in range(ninact):
                # Absolute spin-orbital indices
                abs_a = n_occ + a                           # Active virtual position
                abs_e = n_occ + nvir_act + e               # Inactive virtual position
                
                # Antisymmetric placement
                R_ae = R_local[a, e]
                R_full[abs_e, abs_a] =  R_ae
                R_full[abs_a, abs_e] = -R_ae
        
        # Verify antisymmetry
        assert np.allclose(R_full, -R_full.T, atol=1e-12), \
            f"R matrix not antisymmetric! Max violation: {np.max(np.abs(R_full + R_full.T))}"
        
        # ===== GENERATE UNITARY U = exp(R) =====
        U_spin = scipy.linalg.expm(R_full)
        
        # Verify unitarity
        assert np.allclose(U_spin.T @ U_spin, np.eye(n_spin), atol=1e-6), \
            "U†U ≠ I: Unitary transformation broken!"
        assert np.allclose(U_spin @ U_spin.T, np.eye(n_spin), atol=1e-6), \
            "UU† ≠ I: Unitary transformation broken!"
        
        # ===== VERIFY OCCUPIED BLOCK IS IDENTITY =====
        U_occ = U_spin[:n_occ, :n_occ]
        assert np.allclose(U_occ, np.eye(n_occ), atol=1e-6), \
            f"Occupied block not identity! Max deviation: {np.max(np.abs(U_occ - np.eye(n_occ)))}"
        
        # ===== APPLY UNITARY TO MO COEFFICIENTS =====
        # For unrestricted formalism, apply U separately to alpha and beta
        # U_spin has block structure: [U_alpha_block  at even indices]
        #                             [U_beta_block   at odd indices]
        U_alpha = U_spin[0::2, 0::2]
        U_beta = U_spin[1::2, 1::2]
        
        mo_rot = [
            mo_coeffs[0] @ U_alpha,
            mo_coeffs[1] @ U_beta
        ]
        
        # ===== TRANSFORM FOCK MATRIX =====
        fock_rot = U_spin.T @ fock_spin @ U_spin
        
        # Verify Hermiticity
        assert np.allclose(fock_rot, fock_rot.conj().T, atol=1e-8), \
            "Rotated Fock not Hermitian!"
        
        # ===== VERIFY OCCUPIED-VIRTUAL DECOUPLING =====
        # After rotation with identity in occupied block, 
        # occ-virt should remain zero
        n_occ_spin = n_occ
        n_virt_spin = n_spin - n_occ_spin
        
        occ_virt_block = fock_rot[:n_occ_spin, n_occ_spin:]
        occ_virt_norm = np.linalg.norm(occ_virt_block)
        
        assert occ_virt_norm < 1e-5, \
            f"ERROR: Occupied-virtual coupling appeared after rotation! Norm: {occ_virt_norm:.2e}\n" \
            f"This indicates the unitary was not applied correctly or R had occupied elements."
        
        return mo_rot, fock_rot, U_spin





    def _canonicalize_active(self, mo_coeffs: List[np.ndarray],
                            fock_spin: np.ndarray,
                            U_applied) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Canonicalize ONLY the ACTIVE VIRTUAL BLOCK after orbital rotation.
        
        Does NOT touch inactive orbitals or reorder them—this preserves the
        active-inactive separation that the gradient/Hessian optimization assumes.
        
        Inactive orbitals remain in their original order (or untouched).
        Only the ACTIVE virtuals are diagonalized to restore canonical form
        within that subspace.
        """
        n_occ_a, n_occ_b = self.mol.nelec
        nact_spatial = self.num_opt_virtual_orbs // 2  # Active virtual count
        
        # Build full spin transformation
        U_alpha_full = np.eye(fock_spin.shape[0] // 2)
        U_beta_full = np.eye(fock_spin.shape[0] // 2)
        
        # ===== ALPHA: Canonicalize ACTIVE virtual block only =====
        sl_a = slice(n_occ_a, n_occ_a + nact_spatial)
        if nact_spatial > 0:
            # Extract ONLY the active-active block from alpha Fock
            fock_alpha_spatial = fock_spin[0::2, 0::2]
            F_alpha_active = fock_alpha_spatial[sl_a, sl_a]
            
            # Diagonalize ONLY active-active (leaves active-inactive coupling as-is)
            evals_alpha_active, evecs_alpha_active = np.linalg.eigh(F_alpha_active)
            U_alpha_full[sl_a, sl_a] = evecs_alpha_active
            
            # if self.verbose:
            #     self._print(f"        [Canon Active] Alpha active virtuals: "
            #             f"min={evals_alpha_active[0]:+.6f}, max={evals_alpha_active[-1]:+.6f}")
        
        # ===== BETA: Canonicalize ACTIVE virtual block only =====
        sl_b = slice(n_occ_b, n_occ_b + nact_spatial)
        if nact_spatial > 0:
            # Extract ONLY the active-active block from beta Fock
            fock_beta_spatial = fock_spin[1::2, 1::2]
            F_beta_active = fock_beta_spatial[sl_b, sl_b]
            
            # Diagonalize ONLY active-active (leaves active-inactive coupling as-is)
            evals_beta_active, evecs_beta_active = np.linalg.eigh(F_beta_active)
            U_beta_full[sl_b, sl_b] = evecs_beta_active
            
            # if self.verbose:
            #     self._print(f"        [Canon Active] Beta active virtuals: "
            #             f"min={evals_beta_active[0]:+.6f}, max={evals_beta_active[-1]:+.6f}")
        
        # ===== BUILD SPIN-ORBITAL TRANSFORMATION =====
        U_canon_spin = np.zeros_like(fock_spin)
        U_canon_spin[0::2, 0::2] = U_alpha_full
        U_canon_spin[1::2, 1::2] = U_beta_full
        
        # ===== APPLY CANONICALIZATION =====
        mo_canon = [
            mo_coeffs[0] @ U_alpha_full,
            mo_coeffs[1] @ U_beta_full
        ]
        
        fock_canon = U_canon_spin.T @ fock_spin @ U_canon_spin
        
        # ===== VERIFICATION =====
        # Active blocks should now be diagonal
        if nact_spatial > 0:
            F_alpha_active_canon = fock_canon[0::2, 0::2][sl_a, sl_a]
            F_beta_active_canon = fock_canon[1::2, 1::2][sl_b, sl_b]
            
            off_diag_alpha = np.linalg.norm(F_alpha_active_canon - np.diag(np.diag(F_alpha_active_canon)))
            off_diag_beta = np.linalg.norm(F_beta_active_canon - np.diag(np.diag(F_beta_active_canon)))
            
            assert off_diag_alpha < 1e-6, \
                f"Alpha active block not diagonal (norm={off_diag_alpha:.2e})"
            assert off_diag_beta < 1e-6, \
                f"Beta active block not diagonal (norm={off_diag_beta:.2e})"
        
        # Eigenvalues must be preserved
        evals_before = np.sort(np.linalg.eigvalsh(fock_spin))
        evals_after = np.sort(np.linalg.eigvalsh(fock_canon))
        assert np.allclose(evals_before, evals_after, atol=1e-8), \
            "Canonicalization changed eigenvalues!"
        
        # Extract all orbital energies for diagnostics
        evals_alpha_all = np.diag(fock_canon)[0::2]
        evals_beta_all = np.diag(fock_canon)[1::2]
        
        return mo_canon, fock_canon, [evals_alpha_all, evals_beta_all]


    # -------------------------------------------------------------------------
    # Initial T1 amplitudes and norm
    # ------------------------------------------------------------------------
    def _mp1_t1_amplitudes(self, fock_diag: np.ndarray, eri_as: np.ndarray) -> np.ndarray:
        """
        Compute MP1 T1 amplitudes t_i^a = -<a||i> / (f_aa - f_ii).
        T1 should vanish at Brueckner orbitals.
        
        Parameters
        ----------
        fock_diag : np.ndarray
            Diagonal of spin-orbital Fock matrix
        eri_as : np.ndarray
            Antisymmetrized ERIs in spin-orbital basis
        
        Returns
        -------
        t_1 : np.ndarray
            Shape (nvir_act, nocc_act)
        """
        occ_idx = self.active_occ_indices
        vir_idx = self.active_inocc_indices
        
        if len(vir_idx) == 0 or len(occ_idx) == 0:
            return np.zeros((0, 0))
        
        eps_occ = fock_diag[occ_idx]
        eps_vir = fock_diag[vir_idx]
        nvir_act = len(vir_idx)
        nocc_act = len(occ_idx)
        
        # Denominator: f_aa - f_ii, shape (nvir_act, nocc_act)
        denom = eps_vir[:, None] - eps_occ[None, :]
        
        # Extract active subspace: (nvir_act, nvir_act, nocc_act, nocc_act)
        eri_active = eri_as[:nvir_act, :nvir_act, :nocc_act, :nocc_act]
        
        # Extract diagonal elements: a=b, i=j -> shape (nvir_act, nocc_act)
        t1_numerator = np.einsum('aaic->ai', eri_active)
        
        # T1 amplitudes: t_i^a = -<a||i> / (f_aa - f_ii)
        # Add small epsilon to avoid division by zero
        t1 = -t1_numerator / (denom + 1e-14)
        t1[np.abs(denom) < 1e-10] = 0.0  # Canonical orbitals have f_aa >> f_ii
        
        return t1

    def _t1_norm(self, t1: np.ndarray) -> float:
        """Compute Frobenius norm of T1 amplitudes."""
        return np.linalg.norm(t1.flatten())

    # -------------------------------------------------------------------------
    # Block-diagonal RLE approximation and regularization
    # -------------------------------------------------------------------------
    def _apply_block_diagonal_rle(self, H: np.ndarray, apply_regularization: bool = True) -> np.ndarray:
        """
        Apply block-diagonal RLE (Reduced Linear Equation) approximation to the Hessian.
        
        Per Adamowicz & Bartlett (1987): only keep diagonal blocks H_{ea,eb}
        where each block corresponds to rotations of ALL active orbitals (a,b) 
        against ONE inactive orbital (e).
        
        Block size: (len_active, len_active) for each inactive orbital.
        
        Parameters
        ----------
        H : np.ndarray
            Full Hessian matrix, shape (nvir_act * ninact, nvir_act * ninact)
        apply_regularization : bool, optional
            If True, regularize ill-conditioned blocks to ensure invertibility.
            Default: True
        
        Returns
        -------
        H_block_diag : np.ndarray
            Block-diagonal approximation of H, same shape as input.
            Off-diagonal blocks (e ≠ f) are zeroed out.
        """
        nvir_act = len(self.active_inocc_indices)
        ninact = len(self.inactive_indices)
        
        if ninact == 0 or H.size == 0:
            return H.copy()
        
        block_size = nvir_act
        H_block_diag = np.zeros_like(H)
        
        # Store diagnostics
        regularization_applied = []
        block_eigvals = []
        
        # Process each diagonal block (one per inactive orbital)
        for e_idx in range(ninact):
            start = e_idx * block_size
            end = (e_idx + 1) * block_size
            
            H_block_orig = H[start:end, start:end].copy()
            H_block = H_block_orig.copy()
            
            # Compute eigenvalue properties for diagnostics
            eigvals = np.linalg.eigvalsh(H_block_orig)
            min_eig = np.min(eigvals)
            max_eig_abs = np.max(np.abs(eigvals))
            det_val = np.linalg.det(H_block_orig)
            cond_H = np.linalg.cond(H_block_orig) if H_block_orig.size > 0 else np.inf
            
            block_eigvals.append((e_idx, min_eig, max_eig_abs, cond_H))

            # Inverse block
            # H_block = np.linalg.inv(H_block_orig) if H_block_orig.size > 0 else np.zeros_like(H_block_orig)
            
            H_block_diag[start:end, start:end] = H_block
        
        # Store diagnostics for optional inspection
        self._rle_diagnostics = {
            'block_eigvals': block_eigvals,
            'regularization_applied': regularization_applied,
            'num_blocks': ninact
        }
        
        return H_block_diag

    # -------------------------------------------------------------------------
    # Newton step solver 
    # -------------------------------------------------------------------------
    def _newton_step(self, G: np.ndarray, H: np.ndarray,
                 iteration: int, start_counting: bool) -> np.ndarray:
        """
        Solve H·R = -G using block-diagonal RLE approximation with adaptive regularization.
        
        Strategy:
        1. Apply block-diagonal RLE to extract only diagonal blocks (cheaper, structured)
        2. Regularize the full block-diagonal matrix for invertibility
        3. Solve with pure Newton; fallback to damping if solve fails
        
        Parameters
        ----------
        G : np.ndarray
            Gradient vector, shape (nvir_act * ninact,)
        H : np.ndarray
            Full Hessian matrix, shape (nvir_act * ninact, nvir_act * ninact)
        iteration : int
            Current iteration number (for diagnostics)
        start_counting : bool
            Whether we're in the convergence-tracking phase (unused here, kept for interface)
        
        Returns
        -------
        R : np.ndarray
            Orbital rotation step vector, same shape as G
        """
        g_vec = G.flatten()
        dim = len(g_vec)
        grad_norm = np.linalg.norm(g_vec)

        # ===== APPLY BLOCK-DIAGONAL RLE APPROXIMATION =====
        # Use block-diagonal approximation for better conditioning
            # Returns the inversed block-diagonal matrix
        H_block = self._apply_block_diagonal_rle(H, apply_regularization=False)
        
        cond_H_orig = np.linalg.cond(H) if H.size > 0 else np.inf
        cond_H_block = np.linalg.cond(H_block) if H_block.size > 0 else np.inf
        eigvals_orig = np.linalg.eigvalsh(H)
        eigvals_block = np.linalg.eigvalsh(H_block)
        n_neg_orig = np.sum(eigvals_orig < -1e-10)
        n_neg_block = np.sum(eigvals_block < -1e-10)

        # if self.verbose:
        #     self._print(f"        [RLE] Cond(H): {cond_H_orig:.2e} → {cond_H_block:.2e}, "
        #                f"Neg eigvals: {n_neg_orig} → {n_neg_block}")
        
        H = H_block
        
        # Track oscillation history
        if not hasattr(self, '_grad_norm_history'):
            self._grad_norm_history = []
        if not hasattr(self, '_damping_boost'):
            self._damping_boost = 1.0

        self._grad_norm_history.append(grad_norm)
        if len(self._grad_norm_history) > 20:
            self._grad_norm_history.pop(0)

        # ===== ADAPTIVE DAMPING BASED ON GRADIENT NORM =====
        try: 
            # Pure Newton 
            R = np.linalg.solve(H, -g_vec)
            # R = H @ (-g_vec)
            step_norm = np.linalg.norm(R)

        except np.linalg.LinAlgError:
            # Fallback: add damping
            self._print(f"        [RLE] Hessian solve failed at iteration {iteration}. Applying damping.")
            lambda_lm = self.lambda_init
            for attempt in range(10):
                try:
                    H_reg = H_block + lambda_lm * np.eye(dim)
                    R = np.linalg.solve(H_reg, -g_vec)
                    # R = H_reg @ (-g_vec)
                    step_norm = np.linalg.norm(R)
                    break
                except np.linalg.LinAlgError:
                    lambda_lm *= 10.0
            else:
                # Ultimate fallback
                scale = self.trust_radius / (grad_norm + 1e-14)
                return -scale * g_vec

        return R
                
    # -------------------------------------------------------------------------
    # Main driver
    # -------------------------------------------------------------------------
    def run(self, mo_coeffs: List[np.ndarray],
            fock_spin: Optional[np.ndarray] = None) -> dict:
        """
        Run the OVOS optimization loop.

        Parameters
        ----------
        mo_coeffs : list of np.ndarray
            Initial MO coefficients [alpha, beta].
        fock_spin : np.ndarray or None
            Initial spin‑orbital Fock matrix. If None, built from mo_coeffs.

        Returns
        -------
        dict with keys:
            energy_history : list of float
            converged_energy_history : list of float (only converged part)
            iteration_history : list of int
            final_mo_coeffs : list of np.ndarray
            final_fock_spin : np.ndarray
            stop_reasons : list of str
        """
        converged = False
        start_counting = False
        iter_count = 0
        keep_track = 0

        # Storage
        energy_hist = []
        iter_hist = []
        stop_reasons = []
        mo_hist = []
        fock_hist = []
        grad_norm_hist = []

        # Initial Fock if not provided
        if fock_spin is None:
            Fmo_a = mo_coeffs[0].T @ self.Fao[0] @ mo_coeffs[0]
            Fmo_b = mo_coeffs[1].T @ self.Fao[1] @ mo_coeffs[1]
            fock_spin = self._build_spin_fock(Fmo_a, Fmo_b)
            eig_a = scipy.linalg.eigh(Fmo_a, eigvals_only=True)
            eig_b = scipy.linalg.eigh(Fmo_b, eigvals_only=True)
            self.eps = self._spatial_to_spin_energies(np.real(eig_a), np.real(eig_b))
        else:
            self.eps = np.diag(fock_spin)

        while iter_count < self.max_iter:
            iter_count += 1

            prev_mo_coeffs = mo_coeffs

            if self.verbose:
                self._print()
                self._print(f"#### OVOS Iteration {iter_count} ####")

            # MP2 calculation
            eri_as = self._eri_vovo_antisym(mo_coeffs)
            t_abij = self._mp1_amplitudes(self.eps, eri_as)
            assert np.allclose(t_abij, t_abij.transpose(1, 0, 3, 2), atol=1e-10), \
                "t_abij antisymmetry broken"
            E_corr = self._mp2_energy(fock_spin, t_abij, eri_as)
            
            # Compute T1 norm amplitudes
            t1_abij = self._mp1_t1_amplitudes(self.eps, eri_as)
            t1_norm = self._t1_norm(t1_abij)

            # Track T1 norm history
            if not hasattr(self, '_t1_norm_history'):
                self._t1_norm_history = []
            self._t1_norm_history.append(t1_norm)

            # if self.verbose:
            #     nact = len(self.active_inocc_indices)
            #     ntot = nact + len(self.inactive_indices)
                # self._print(f"    [{nact}/{ntot}]: MP2 energy = {E_corr:.12f}, T1 norm = {t1_norm:.2e}")

            # Track energy difference, ...
            if not hasattr(self, '_step_history'):
                self._step_history = []
            if iter_count > 1:
                dE = E_corr - energy_hist[-1]
                self._step_history.append(dE)
                if len(self._step_history) > 10:
                    self._step_history.pop(0)

            energy_hist.append(E_corr)
            iter_hist.append(iter_count)
            mo_hist.append(mo_coeffs)
            fock_hist.append(fock_spin)

            if self.verbose:
                nact = len(self.active_inocc_indices)
                ntot = nact + len(self.inactive_indices)
                self._print(f"    [{nact}/{ntot}]: MP2 energy = {E_corr:.12f}")

            if iter_count == 1:
                stop_reasons.append("Initial")
                if len(self.virtual_inocc_indices) == 0:
                    break
            else:
                # Check convergence
                dE = abs(energy_hist[-1] - energy_hist[-2])
                self.dE = energy_hist[-1] - energy_hist[-2]

                D_ab = self._compute_density(t_abij)
                G = self._gradient(t_abij, eri_as, D_ab, fock_spin)
                grad_norm = np.linalg.norm(G.flatten())
                grad_norm_hist.append(grad_norm)
                if len(grad_norm_hist) > 5:
                    grad_norm_hist.pop(0)
                dgrad = abs(grad_norm_hist[-1] - grad_norm_hist[-2]) if len(grad_norm_hist) > 1 else None

                if self.verbose and dgrad is not None:
                    flag = "(energy increased!)" if self.dE > 0 else ""
                    self._print(f"            ΔE = {self.dE:.2e}  ‖grad‖ = {grad_norm:.2e}  ‖T1‖ = {t1_norm:.2e} {flag}")
                
                if (dE < self.conv_energy and grad_norm < self.conv_grad and t1_norm < self.t1_conv):
                    stop_reasons.append("Convergence")
                    start_counting = True
                else:
                    stop_reasons.append("Non‑converged")
                    start_counting = False
                    keep_track = 0

                if start_counting: # (dE < self.conv_energy and grad_norm < self.conv_grad) or dE < 1e-12:
                    keep_track += 1
                    if self.verbose:
                     # Make sure self.keep_track_max is not string
                        if isinstance(self.keep_track_max, str):
                            self._print(f"    Keep track: {keep_track}/{self.keep_track_max}")
                    if isinstance(self.keep_track_max, int):
                        if keep_track >= self.keep_track_max and iter_count > 100:
                            if self.verbose:
                                self._print(f"OVOS converged after {iter_count-keep_track} iterations")
                            # Trim the extra tracked steps
                            trim = self.keep_track_max - 1
                            energy_hist = energy_hist[:-trim]
                            iter_hist = iter_hist[:-trim]
                            mo_hist = mo_hist[:-trim]
                            fock_hist = fock_hist[:-trim]
                            stop_reasons = stop_reasons[:-trim]
                            converged = True
                            break
                    elif isinstance(self.keep_track_max, str):
                        if self.keep_track_max.lower() == "none":
                            # Let it run untill iter_count reaches max_iter, but we are in the converged state
                            if keep_track >= 1 and iter_count == 1000:
                                if self.verbose:
                                    self._print(f"OVOS converged after {iter_count-keep_track} iterations (keep_track_max=None)")
                                # Trim the extra tracked steps
                                trim = keep_track - 1
                                energy_hist = energy_hist[:-trim]
                                iter_hist = iter_hist[:-trim]
                                mo_hist = mo_hist[:-trim]
                                fock_hist = fock_hist[:-trim]
                                stop_reasons = stop_reasons[:-trim]
                                converged = True
                                break
                else:
                    keep_track = 0

            # Compute gradient and Hessian for Newton step
            D_ab = self._compute_density(t_abij)
            G = self._gradient(t_abij, eri_as, D_ab, fock_spin)
            H = self._hessian(t_abij, eri_as, D_ab, fock_spin)

            # Store current energy for use in _newton_step
            self.current_energy = E_corr
            self.fock_spin = fock_spin
            self.mo_coeffs = mo_coeffs

            # Solve Newton step
            R_vec = self._newton_step(G, H, iter_count, start_counting)

            # Apply rotation
            mo_coeffs, fock_spin, U_spin = self._rotate_orbitals(mo_coeffs, fock_spin, R_vec)

            # Re‑canonicalize the active virtual block
            mo_coeffs, fock_spin, evals = self._canonicalize_active(mo_coeffs, fock_spin, U_spin)

            # Update diagonal elements for next MP1 denominator (approximate, but ok)
            self.eps = np.diag(fock_spin)

            if iter_count >= self.max_iter:
                if self.verbose:
                    self._print(f"Reached maximum iterations ({self.max_iter})")
                break

        # --- Summary ---
        if self.verbose:
            self._print()
            self._print("#### OVOS Summary ####")
            self._print(f"Initial MP2 energy: {energy_hist[0]:.12f} Ha")
            self._print(f"Final MP2 energy:   {energy_hist[-1]:.12f} Ha")
            delta = energy_hist[-1] - energy_hist[0]
            self._print(f"Total change:       {delta:.12f} Ha")
            if delta < 0:
                self._print("OVOS lowered the correlation energy.")
            else:
                self._print("WARNING: OVOS increased the correlation energy.")
            # Unrestricted or restricted final orbitals
            if np.allclose(mo_coeffs[0], mo_coeffs[1], atol=1e-6):
                self._print("Final orbitals are effectively restricted.")
            else:
                self._print("Final orbitals are unrestricted.")
            self._print(f"Total iterations: {iter_count-keep_track} (converged: {converged})")
            # Difference in prev_mo_coeffs vs final mo_coeffs
            mo_diff_a = np.linalg.norm(mo_coeffs[0] - prev_mo_coeffs[0])
            mo_diff_b = np.linalg.norm(mo_coeffs[1] - prev_mo_coeffs[1])
            self._print(f"Change in MO coefficients (Frobenius norm): "
                    f"alpha: {mo_diff_a:.2e}, beta: {mo_diff_b:.2e}")
            self._print(f"Stopping reason: {stop_reasons[-1]}")

        # Select best result (lowest energy)
        best_idx = int(np.argmin(energy_hist)) 
        result = [
            energy_hist[best_idx],
            energy_hist[:best_idx+1],
            iter_hist[:best_idx+1],
            mo_hist[best_idx],
            fock_hist[best_idx],
            stop_reasons[best_idx]
        ]

        # if stop_reasons[best_idx] != "Initial":
        #     self._print(f"Final ||grad|| = {grad_norm:.2e}, ||T1|| = {t1_norm:.2e}")

        return result


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from pyscf import gto, scf

    # Water molecule, minimal basis
    mol = gto.Mole()
    mol.atom = 'O 0 0 0; H 0 0 1; H 0 1 0'
    mol.basis = '6-31G'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # RHF reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Initial data (RHF orbitals)
    Fao = [mf.get_fock(), mf.get_fock()]
    mo_coeffs = [mf.mo_coeff, mf.mo_coeff]

    # Create OVOS object and run
    ovos = OVOS(
        mol=mol,
        scf=mf,
        Fao=Fao,
        num_opt_virtual_orbs=2,      # active virtual spin‑orbitals
        mo_coeff=mo_coeffs,
        init_orbs="RHF",
        verbose=1,
        max_iter=1000,
        conv_energy=1e-8,
        conv_grad=1e-6,
        keep_track_max=50
    )
    E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)

    print("\nOptimization finished. Final MP2 energy =", E_corr, "Hartree (Stopping reason:", stop_reason, ")")