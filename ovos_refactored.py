"""
OVOS - Optimized Virtual Orbital Space

Minimizes the second-order correlation energy (MP2) using orbital rotations.

Implementation based on:
    L. Adamowicz & R. J. Bartlett (1987)
    "Optimized virtual orbital space for high-level correlated calculations"
    J. Chem. Phys. 86, 6314-6324
    DOI: 10.1063/1.452468

Author: Tobias (UCPH Master's Thesis)
"""

import os
import json
from typing import Tuple, Union, List, Optional
from decimal import Decimal

# Force single-threaded execution for reproducibility
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import scipy
import scipy.linalg
import pyscf
import pyscf.ao2mo

# NumPy print options
np.set_printoptions(precision=6, suppress=True, linewidth=200)


# =============================================================================
# MOLECULE AND BASIS SET DEFINITIONS
# =============================================================================

MOLECULES = {
    "H2": "H 0.0 0.0 0.0; H 0.0 0.0 0.74144",
    "LiH": "Li 0.0 0.0 0.0; H 0.0 0.0 1.595",
    "H2O": "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
    "CH2": "C 0.0 0.0 0.0; H 0.0 0.9350 0.5230; H 0.0 -0.9350 0.5230",
    "BH3": "B 0 0 0; H 0 0 1.19; H 0 1.03 -0.40; H 0 -1.03 -0.40",
    "N2": "N 0 0 0; N 0 0 1.10",
}

BASIS_SETS = [
    "STO-3G", "STO-6G", "3-21G", "6-31G", "DZP", "roosdz",
    "anoroosdz", "cc-pVDZ", "cc-pV5Z", "def2-QZVPP", "aug-cc-pV5Z", "ANO"
]


# =============================================================================
# OVOS CLASS
# =============================================================================

class OVOS:
    """
    Optimized Virtual Orbital Space (OVOS) algorithm.
    
    Minimizes the second-order correlation energy (MP2) using orbital rotations
    between active and inactive virtual orbital spaces.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object.
    num_opt_virtual_orbs : int
        Number of optimized virtual spin orbitals.
    init_orbs : str or np.ndarray
        Initial orbitals: "UHF" for fresh calculation, or array of MO coefficients.
    
    Attributes
    ----------
    active_occ_indices : list
        Occupied spin-orbital indices [0, nelec)
    active_inocc_indices : list
        Active virtual orbital indices [nelec, nelec + num_opt_virtual_orbs)
    inactive_indices : list
        Inactive virtual orbital indices [nelec + num_opt_virtual_orbs, tot_spin_orbs)
    """
    
    # Convergence parameters
    CONVERGENCE_THRESHOLD = 1e-8  # Hartree
    MAX_ITERATIONS = 2500
    
    def __init__(
        self, 
        mol: pyscf.gto.Mole, 
        num_opt_virtual_orbs: int, 
        init_orbs: Union[str, np.ndarray] = "UHF"
    ) -> None:
        
        self.mol = mol
        self.num_opt_virtual_orbs = num_opt_virtual_orbs
        
        # Run UHF calculation
        self.uhf = pyscf.scf.UHF(mol).run()
        self.e_uhf = self.uhf.e_tot
        self.h_nuc = mol.energy_nuc()
        
        # Set initial MO coefficients
        if isinstance(init_orbs, str) and init_orbs == "UHF":
            self.mo_coeffs = self.uhf.mo_coeff
        else:
            self.mo_coeffs = init_orbs
        
        # Reference MP2 calculation
        self.mp2_ref = self.uhf.MP2().run()
        self.mp2_ecorr_ref = self.mp2_ref.e_corr
        
        # AO integrals and matrices
        self.S = mol.intor('int1e_ovlp')
        self.Fao = self.uhf.get_fock()
        self.eri_ao = mol.intor('int2e_sph', aosym=1)
        
        # Orbital counts
        self.n_spatial = self.mo_coeffs[0].shape[1]
        self.n_spin = 2 * self.n_spatial
        self.nelec = mol.nelec[0] + mol.nelec[1]
        
        # Validate MO coefficients
        assert self.mo_coeffs[0].shape[1] == self.mo_coeffs[1].shape[1], \
            "Alpha and beta orbital counts must match"
        
        # Build orbital index spaces (NATURAL ordering)
        self._build_index_spaces()
        
        # Validate space sizes
        assert self.n_spin >= self.num_opt_virtual_orbs + self.nelec, \
            f"num_opt_virtual_orbs={num_opt_virtual_orbs} too large for system"
        
        # Print space information
        self._print_space_info()
    
    def _build_index_spaces(self) -> None:
        """Build index lists for active/inactive orbital spaces."""
        # Occupied: [0, nelec)
        self.active_occ_indices = list(range(self.nelec))
        
        # Active virtual: [nelec, nelec + num_opt_virtual_orbs)
        virt_start = self.nelec
        virt_end = self.nelec + self.num_opt_virtual_orbs
        self.active_inocc_indices = list(range(virt_start, virt_end))
        
        # Inactive virtual: [nelec + num_opt_virtual_orbs, n_spin)
        self.inactive_indices = list(range(virt_end, self.n_spin))
        
        # Full space
        self.full_indices = list(range(self.n_spin))
    
    def _print_space_info(self) -> None:
        """Print orbital space information."""
        print()
        print("=" * 50)
        print("OVOS Orbital Spaces")
        print("=" * 50)
        print(f"  Total spin-orbitals:     {self.n_spin}")
        print(f"  Occupied (I,J):          {self.active_occ_indices}")
        print(f"  Active virtual (A,B):    {self.active_inocc_indices}")
        print(f"  Inactive virtual (E,F):  {self.inactive_indices}")
        print("=" * 50)
        print()
    
    # =========================================================================
    # SPATIAL -> SPIN-ORBITAL CONVERSIONS
    # =========================================================================
    
    def _spatial_to_spin_eri(
        self, 
        eri_aaaa: np.ndarray, 
        eri_aabb: np.ndarray, 
        eri_bbbb: np.ndarray
    ) -> np.ndarray:
        """
        Convert spatial ERIs to spin-orbital ERIs using block assignment.
        
        Uses optimized NumPy slicing where possible, with loops for cross-spin blocks.
        """
        n = self.n_spatial
        eri_spin = np.zeros((self.n_spin,) * 4, dtype=np.float64)
        
        # Same-spin blocks: (αα|αα) and (ββ|ββ)
        eri_spin[0::2, 0::2, 0::2, 0::2] = eri_aaaa
        eri_spin[1::2, 1::2, 1::2, 1::2] = eri_bbbb
        
        # Mixed-spin blocks: (αα|ββ) and (ββ|αα)
        eri_spin[0::2, 0::2, 1::2, 1::2] = eri_aabb
        eri_spin[1::2, 1::2, 0::2, 0::2] = eri_aabb.transpose(2, 3, 0, 1)
        
        # Cross-spin blocks: (αβ|αβ), (βα|βα), etc. - MUST use full 4D loop
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        # (αβ|αβ) type blocks
                        eri_spin[2*p, 2*q+1, 2*r, 2*s+1] = eri_aabb[p, r, q, s]
                        eri_spin[2*p+1, 2*q, 2*r+1, 2*s] = eri_aabb[q, s, p, r]
        
        
        return eri_spin
    
    def _spatial_to_spin_fock(
        self, 
        Fmo_a: np.ndarray, 
        Fmo_b: np.ndarray
    ) -> np.ndarray:
        """
        Convert spatial Fock matrices to spin-orbital Fock matrix.
        
        Block-diagonal structure: [α, β, α, β, ...]
        """
        Fmo_spin = np.zeros((self.n_spin, self.n_spin), dtype=np.float64)
        Fmo_spin[0::2, 0::2] = Fmo_a  # Alpha block
        Fmo_spin[1::2, 1::2] = Fmo_b  # Beta block
        return Fmo_spin
    
    def _spatial_to_spin_mo_energy(
        self, 
        mo_energy_a: np.ndarray, 
        mo_energy_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert spatial MO energies to energy-sorted spin-orbital energies.
        
        Returns
        -------
        eps : ndarray
            Energy-sorted orbital energies
        orbspin : ndarray
            Spin label (0=alpha, 1=beta) for each energy-ordered orbital
        spatial_map : ndarray
            Spatial index for each energy-ordered position
        spin_orb_map : ndarray
            Natural spin-orbital index for each energy-ordered position
        """
        n = len(mo_energy_a)
        
        # Build (energy, spin, spatial_idx, spin_orb_idx) tuples
        spin_data = []
        for i in range(n):
            spin_data.append((mo_energy_a[i], 0, i, 2*i))      # Alpha
            spin_data.append((mo_energy_b[i], 1, i, 2*i + 1))  # Beta
        
        # Sort by energy
        spin_data.sort(key=lambda x: x[0])
        
        # Extract arrays
        eps = np.array([d[0] for d in spin_data])
        orbspin = np.array([d[1] for d in spin_data])
        spatial_map = np.array([d[2] for d in spin_data])
        spin_orb_map = np.array([d[3] for d in spin_data])
        
        return eps, orbspin, spatial_map, spin_orb_map
    
    def _spatial_to_spin_mo(
        self, 
        mo_coeffs: Union[List, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert spatial MO coefficients to spin-orbital form.
        
        Interleaves alpha/beta: [α₀, β₀, α₁, β₁, ...]
        
        Returns
        -------
        C_spin : ndarray
            Spin-orbital MO coefficients, shape (n_ao, n_spin)
        orb_map : ndarray
            Spatial index for each spin-orbital
        orbspin : ndarray
            Spin label for each spin-orbital
        """
        if isinstance(mo_coeffs, (list, tuple)):
            C_alpha, C_beta = mo_coeffs
        else:
            C_alpha, C_beta = mo_coeffs[0], mo_coeffs[1]
        
        n_ao, n_spatial = C_alpha.shape
        n_spin = 2 * n_spatial
        
        # Interleave columns
        C_spin = np.zeros((n_ao, n_spin), dtype=np.float64)
        C_spin[:, 0::2] = C_alpha
        C_spin[:, 1::2] = C_beta
        
        # Maps
        orb_map = np.array([i // 2 for i in range(n_spin)], dtype=int)
        orbspin = np.array([i % 2 for i in range(n_spin)], dtype=int)
        
        return C_spin, orb_map, orbspin
    
    def _spin_to_spatial_mo(
        self, 
        mo_coeffs_spin: np.ndarray, 
        orbspin: np.ndarray
    ) -> np.ndarray:
        """
        Convert spin-orbital MO coefficients back to spatial form.
        
        Returns array [C_alpha, C_beta].
        """
        n_ao = mo_coeffs_spin.shape[0]
        n_spin = mo_coeffs_spin.shape[1]
        n_spatial = n_spin // 2
        
        C_alpha = np.zeros((n_ao, n_spatial), dtype=np.float64)
        C_beta = np.zeros((n_ao, n_spatial), dtype=np.float64)
        
        alpha_cols = np.where(orbspin == 0)[0]
        beta_cols = np.where(orbspin == 1)[0]
        
        C_alpha = mo_coeffs_spin[:, alpha_cols]
        C_beta = mo_coeffs_spin[:, beta_cols]
        
        return np.array([C_alpha, C_beta])
    
    # =========================================================================
    # MP2 ENERGY CALCULATION
    # =========================================================================
    
    def MP2_energy(
        self, 
        mo_coeffs: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute MP2 correlation energy for unrestricted orbitals.
        
        Parameters
        ----------
        mo_coeffs : ndarray
            MO coefficients [alpha, beta]
        
        Returns
        -------
        E_corr : float
            MP2 correlation energy (J₂ functional)
        MP1_amplitudes : ndarray
            First-order amplitudes t_ij^ab
        eri_spin : ndarray
            Spin-orbital two-electron integrals (chemist notation)
        eri_phys : ndarray
            Spin-orbital integrals (physicist notation)
        eri_as : ndarray
            Antisymmetrized integrals
        Fmo_spin : ndarray
            Spin-orbital Fock matrix
        """
        # Build Fock matrices in MO basis
        Fmo_a = mo_coeffs[0].T @ self.Fao[0] @ mo_coeffs[0]
        Fmo_b = mo_coeffs[1].T @ self.Fao[1] @ mo_coeffs[1]
        Fmo_spin = self._spatial_to_spin_fock(Fmo_a, Fmo_b)
        
        # Get orbital energies via diagonalization
        eigval_a, _ = scipy.linalg.eigh(Fmo_a)
        eigval_b, _ = scipy.linalg.eigh(Fmo_b)
        mo_energy_a = np.real(eigval_a)
        mo_energy_b = np.real(eigval_b)
        
        # Convert to energy-sorted spin-orbital energies
        self.eps, self.orbspin_sorted, self.spatial_map_sorted, self.spin_orb_map_sorted = \
            self._spatial_to_spin_mo_energy(mo_energy_a, mo_energy_b)
        
        # Create index mappings: natural <-> energy ordering
        self.natural_to_energy = np.zeros(self.n_spin, dtype=int)
        for energy_idx, natural_idx in enumerate(self.spin_orb_map_sorted):
            self.natural_to_energy[natural_idx] = energy_idx
        
        # Convert index spaces to energy ordering
        self.active_occ_indices_energy = [self.natural_to_energy[i] for i in self.active_occ_indices]
        self.active_inocc_indices_energy = [self.natural_to_energy[i] for i in self.active_inocc_indices]
        self.inactive_indices_energy = [self.natural_to_energy[i] for i in self.inactive_indices]
        
        eps = self.eps
        
        # Transform 2e-integrals to MO basis
        eri_aaaa = pyscf.ao2mo.kernel(
            self.eri_ao, [mo_coeffs[0]]*4, compact=False
        ).reshape((self.n_spatial,)*4)
        
        eri_bbbb = pyscf.ao2mo.kernel(
            self.eri_ao, [mo_coeffs[1]]*4, compact=False
        ).reshape((self.n_spatial,)*4)
        
        eri_aabb = pyscf.ao2mo.kernel(
            self.eri_ao, [mo_coeffs[0], mo_coeffs[0], mo_coeffs[1], mo_coeffs[1]], compact=False
        ).reshape((self.n_spatial, self.n_spatial, self.n_spatial, self.n_spatial))
        
        # Convert to spin-orbital basis
        eri_spin = self._spatial_to_spin_eri(eri_aaaa, eri_aabb, eri_bbbb)
        
        # Physicist notation and antisymmetrize
        eri_phys = eri_spin.transpose(0, 2, 1, 3)
        eri_as = eri_phys - eri_phys.transpose(0, 1, 3, 2)
        
        # Compute MP1 amplitudes
        MP1_amplitudes = self._compute_mp1_amplitudes(eps, eri_as)
        
        # Compute J₂ (MP2 correlation energy)
        E_corr = self._compute_J2(eps, eri_as, MP1_amplitudes, Fmo_spin)
        
        return E_corr, MP1_amplitudes, eri_spin, eri_phys, eri_as, Fmo_spin
    
    def _compute_mp1_amplitudes(
        self, 
        eps: np.ndarray, 
        eri_as: np.ndarray
    ) -> np.ndarray:
        """
        Compute first-order MP amplitudes using vectorized operations.
        
        t_ij^ab = -<ab||ij> / (ε_a + ε_b - ε_i - ε_j)
        """
        occ = self.active_occ_indices
        virt = self.active_inocc_indices
        n_occ = len(occ)
        n_virt = len(virt)
        
        MP1_amplitudes = np.zeros((self.n_spin,)*4, dtype=np.float64)
        
        if n_occ == 0 or n_virt == 0:
            return MP1_amplitudes
        
        # Build energy denominator using broadcasting
        eps_occ = eps[occ]
        eps_virt = eps[virt]
        
        # denominator[a,b,i,j] = eps_virt[a] + eps_virt[b] - eps_occ[i] - eps_occ[j]
        denom = (
            eps_virt[:, None, None, None] +
            eps_virt[None, :, None, None] -
            eps_occ[None, None, :, None] -
            eps_occ[None, None, None, :]
        )
        
        # Extract integral block
        occ_arr = np.array(occ)
        virt_arr = np.array(virt)
        integral_block = eri_as[np.ix_(virt_arr, virt_arr, occ_arr, occ_arr)]
        
        # Compute amplitudes
        t_block = -integral_block / denom
        
        # Store in full array
        idx_a, idx_b, idx_i, idx_j = np.ix_(virt_arr, virt_arr, occ_arr, occ_arr)
        MP1_amplitudes[idx_a, idx_b, idx_i, idx_j] = t_block
        
        return MP1_amplitudes
    
    def _compute_J2(
        self, 
        eps: np.ndarray, 
        eri_as: np.ndarray, 
        t: np.ndarray, 
        Fmo_spin: np.ndarray
    ) -> float:
        """
        Compute J₂ functional (MP2 correlation energy).
        
        J₂ = Σ_{i>j} [term1 + term2]
        
        term1 = Σ_{a>b,c>d} t_ij^ab t_ij^cd [...bracket terms...]
        term2 = 2 Σ_{a>b} t_ij^ab <ab||ij>
        """
        occ = self.active_occ_indices
        virt = self.active_inocc_indices
        
        J2 = 0.0
        
        for i in occ:
            eps_i = eps[i]
            for j in occ:
                if j <= i:
                    continue
                eps_j = eps[j]
                
                # Term 1: Σ_{a>b,c>d} t_ij^ab t_ij^cd [brackets]
                term1 = 0.0
                for a in virt:
                    for b in virt:
                        if b <= a:
                            continue
                        t_ab = t[a, b, i, j]
                        
                        for c in virt:
                            for d in virt:
                                if d <= c:
                                    continue
                                t_cd = t[c, d, i, j]
                                
                                # Kronecker deltas
                                delta_ac = 1.0 if a == c else 0.0
                                delta_bd = 1.0 if b == d else 0.0
                                delta_ad = 1.0 if a == d else 0.0
                                delta_bc = 1.0 if b == c else 0.0
                                
                                # Fock matrix elements
                                f_ac = Fmo_spin[a, c]
                                f_bd = Fmo_spin[b, d]
                                f_ad = Fmo_spin[a, d]
                                f_bc = Fmo_spin[b, c]
                                
                                # Bracket terms from Eq. 5 in paper
                                bracket = (
                                    (f_ac * delta_bd - f_ad * delta_bc) +
                                    (f_bd * delta_ac - f_bc * delta_ad) -
                                    (eps_i + eps_j) * (delta_ac * delta_bd - delta_ad * delta_bc)
                                )
                                
                                term1 += t_ab * t_cd * bracket
                
                # Term 2: 2 Σ_{a>b} t_ij^ab <ab||ij>
                term2 = 0.0
                for a in virt:
                    for b in virt:
                        if b <= a:
                            continue
                        term2 += t[a, b, i, j] * eri_as[a, b, i, j]
                term2 *= 2.0
                
                J2 += term1 + term2
        
        return J2
    
    # =========================================================================
    # ORBITAL OPTIMIZATION
    # =========================================================================
    
    def orbital_optimization(
        self, 
        mo_coeffs: np.ndarray,
        MP1_amplitudes: np.ndarray,
        eri_spin: np.ndarray,
        eri_phys: np.ndarray,
        eri_as: np.ndarray,
        Fmo_spin: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Optimize orbitals via rotations (Steps v-viii).
        
        Uses Newton's method: ΔR = -H⁻¹ G
        
        Returns
        -------
        mo_coeffs_rot : ndarray
            Rotated MO coefficients
        use_RLE : bool
            Whether RLE method was used
        """
        # Compute D_ab density matrix
        D_ab = self._compute_D_ab(MP1_amplitudes)
        
        # Compute gradient G
        G = self._compute_gradient(MP1_amplitudes, eri_as, D_ab, Fmo_spin)
        
        # Compute Hessian H
        H = self._compute_hessian(MP1_amplitudes, eri_as, D_ab, Fmo_spin)
        
        # Solve Newton step
        try:
            H_inv = np.linalg.inv(H)
            R_flat = -G @ H_inv
        except np.linalg.LinAlgError:
            print("Warning: Hessian inversion failed, using pseudo-inverse")
            H_pinv = np.linalg.pinv(H)
            R_flat = -G @ H_pinv
        
        # Build antisymmetric rotation matrix
        n_active = len(self.active_inocc_indices)
        n_inactive = len(self.inactive_indices)
        R_matrix = np.zeros((self.n_spin, self.n_spin), dtype=np.float64)
        
        for idx_A, A in enumerate(self.active_inocc_indices):
            for idx_E, E in enumerate(self.inactive_indices):
                R_AE = R_flat[idx_A * n_inactive + idx_E]
                R_matrix[E, A] = R_AE
                R_matrix[A, E] = -R_AE  # Antisymmetry
        
        # Construct unitary transformation U = exp(R)
        U = scipy.linalg.expm(R_matrix)
        
        # Apply rotation in spin-orbital basis
        mo_coeffs_spin, orb_map, orbspin = self._spatial_to_spin_mo(mo_coeffs)
        mo_coeffs_spin_rot = mo_coeffs_spin @ U
        mo_coeffs_rot = self._spin_to_spatial_mo(mo_coeffs_spin_rot, orbspin)
        
        return mo_coeffs_rot, False
    
    def _compute_D_ab(self, t: np.ndarray) -> np.ndarray:
        """
        Compute virtual-virtual density matrix D_ab.
        
        D_ab = Σ_{i>j} Σ_c t_ij^ac t_ij^bc
        """
        occ = self.active_occ_indices
        virt = self.active_inocc_indices
        n_virt = len(virt)
        
        D_ab = np.zeros((n_virt, n_virt), dtype=np.float64)
        
        # Collect unique i>j pairs
        occ_pairs = [(i, j) for idx_i, i in enumerate(occ) 
                     for j in occ[idx_i+1:]]
        n_pairs = len(occ_pairs)
        
        if n_pairs == 0:
            return D_ab
        
        # Build t_unique[a, c, pair] array
        virt_arr = np.array(virt)
        t_unique = np.zeros((n_virt, n_virt, n_pairs), dtype=np.float64)
        
        for p, (i, j) in enumerate(occ_pairs):
            t_unique[:, :, p] = t[np.ix_(virt_arr, virt_arr, [i], [j])][:, :, 0, 0]
        
        # D_ab = Σ_p Σ_c t[a,c,p] * t[b,c,p]
        D_ab = np.einsum('acp,bcp->ab', t_unique, t_unique, optimize=True)
        
        return D_ab
    
    def _compute_gradient(
        self, 
        t: np.ndarray, 
        eri_as: np.ndarray, 
        D_ab: np.ndarray, 
        Fmo_spin: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient G_ea of J₂ functional.
        
        G_ea = 2 Σ_{i>j} Σ_b t_ij^ab <eb||ij> + 2 Σ_b D_ab f_be
        """
        occ = self.active_occ_indices
        virt = self.active_inocc_indices
        inactive = self.inactive_indices
        
        n_virt = len(virt)
        n_inactive = len(inactive)
        
        # Collect unique i>j pairs
        occ_pairs = [(i, j) for idx_i, i in enumerate(occ) 
                     for j in occ[idx_i+1:]]
        n_pairs = len(occ_pairs)
        
        G = np.zeros(n_virt * n_inactive, dtype=np.float64)
        
        if n_pairs == 0 or n_virt == 0 or n_inactive == 0:
            return G
        
        virt_arr = np.array(virt)
        inactive_arr = np.array(inactive)
        
        # Build t_pairs[a, b, p] and int_pairs[e, b, p]
        t_pairs = np.zeros((n_virt, n_virt, n_pairs), dtype=np.float64)
        int_pairs = np.zeros((n_inactive, n_virt, n_pairs), dtype=np.float64)
        
        for p, (i, j) in enumerate(occ_pairs):
            t_pairs[:, :, p] = t[np.ix_(virt_arr, virt_arr, [i], [j])][:, :, 0, 0]
            int_pairs[:, :, p] = eri_as[np.ix_(inactive_arr, virt_arr, [i], [j])][:, :, 0, 0]
        
        # Term 1: 2 * Σ_p Σ_b t[a,b,p] * int[e,b,p]
        # Result shape: (n_virt, n_inactive) -> transpose to (n_inactive, n_virt)
        term1_2d = 2.0 * np.einsum('abp,ebp->ae', t_pairs, int_pairs, optimize=True)
        
        # Term 2: 2 * Σ_b D[a,b] * F[b,e]
        F_block = Fmo_spin[np.ix_(virt_arr, inactive_arr)]
        term2_2d = 2.0 * D_ab @ F_block
        
        # Flatten: G[A * n_inactive + E]
        G_2d = term1_2d + term2_2d
        G = G_2d.flatten()
        
        return G

    def _compute_hessian(
        self, 
        t: np.ndarray, 
        eri_as: np.ndarray, 
        D_ab: np.ndarray, 
        Fmo_spin: np.ndarray
    ) -> np.ndarray:
        """
        Compute Hessian H_{ea,fb} of J₂ functional using optimized einsum.
        
        H = term1 + term2 + term3 + term4 (from Eq. 11b in paper)
        
        Term 1: 2 * Σ_{i>j} t_ij^{ab} <ef||ij>
        Term 2: - Σ_{i>j} Σ_c [t_ij^{ac} <bc||ij> + t_ij^{bc} <ac||ij>] * δ_EF
        Term 3: D_ab * (f_aa - f_bb) * δ_EF
        Term 4: D_ab * f_ef * (1 - δ_EF)
        """
        occ = np.array(self.active_occ_indices)
        virt = np.array(self.active_inocc_indices)
        inactive = np.array(self.inactive_indices)
        
        n_occ = len(occ)
        n_virt = len(virt)
        n_inactive = len(inactive)
        
        # Early return if no inactive orbitals
        if n_inactive == 0:
            return np.zeros((0, 0), dtype=np.float64)
        
        # Get I<J pairs (upper triangular indices give i<j)
        i_idx, j_idx = np.triu_indices(n_occ, k=1)
        n_pairs = len(i_idx)
        
        # Initialize Hessian in 4D format: H[A, E, B, F]
        H_4d = np.zeros((n_virt, n_inactive, n_virt, n_inactive), dtype=np.float64)
        
        # === PREPARE DATA ===
        # Extract t_pairs: t[A,B,I,J] for I<J pairs
        t_pairs = np.zeros((n_virt, n_virt, n_pairs), dtype=np.float64)
        for p in range(n_pairs):
            i = occ[i_idx[p]]
            j = occ[j_idx[p]]
            t_pairs[:, :, p] = t[np.ix_(virt, virt, [i], [j])].reshape(n_virt, n_virt)
        
        # === TERM 1: 2 * Σ_{i>j} t_ij^{ab} <ef||ij> ===
        # Extract integrals <E,F,I,J> for I<J pairs
        int_ef_pairs = np.zeros((n_inactive, n_inactive, n_pairs), dtype=np.float64)
        for p in range(n_pairs):
            i = occ[i_idx[p]]
            j = occ[j_idx[p]]
            int_ef_pairs[:, :, p] = eri_as[np.ix_(inactive, inactive, [i], [j])].reshape(n_inactive, n_inactive)
        
        # H[A,E,B,F] += 2 * Σ_p t[A,B,p] * int[E,F,p]
        term1_4d = 2.0 * np.einsum('abp,efp->aebf', t_pairs, int_ef_pairs, optimize=True)
        H_4d += term1_4d
        
        # === TERM 2: - Σ_{i>j} Σ_c [t_ij^{ac} <bc||ij> + t_ij^{bc} <ac||ij>] * δ_EF ===
        # Extract integrals <B,C,I,J> for I<J pairs
        int_bc_pairs = np.zeros((n_virt, n_virt, n_pairs), dtype=np.float64)
        for p in range(n_pairs):
            i = occ[i_idx[p]]
            j = occ[j_idx[p]]
            int_bc_pairs[:, :, p] = eri_as[np.ix_(virt, virt, [i], [j])].reshape(n_virt, n_virt)
        
        # Compute Σ_c t[A,C,p] * int[B,C,p]
        term2_part1 = np.einsum('acp,bcp->abp', t_pairs, int_bc_pairs, optimize=True)
        # Compute Σ_c t[B,C,p] * int[A,C,p]
        term2_part2 = np.einsum('bcp,acp->abp', t_pairs, int_bc_pairs, optimize=True)
        
        # Sum over p and apply negative sign
        term2_sum = -np.sum(term2_part1 + term2_part2, axis=2)  # shape: (n_virt, n_virt)
        
        # Apply δ_EF: add to H[A,E,B,E] for all A,B and each E
        for e in range(n_inactive):
            H_4d[:, e, :, e] += term2_sum
        
        # === TERM 3: D_ab * (f_aa - f_bb) * δ_EF ===
        f_diag_virt = np.array([Fmo_spin[a, a] for a in virt])
        f_diff = f_diag_virt[:, None] - f_diag_virt[None, :]  # shape: (n_virt, n_virt)
        term3 = D_ab * f_diff  # shape: (n_virt, n_virt)
        
        # Add to diagonal E,F blocks
        for e in range(n_inactive):
            H_4d[:, e, :, e] += term3
        
        # === TERM 4: D_ab * f_ef * (1 - δ_EF) ===
        # Extract f_ef for inactive orbitals
        f_inactive = Fmo_spin[np.ix_(inactive, inactive)]  # shape: (n_inactive, n_inactive)
        
        # Create term4_4d = D_ab[A,B] * f_inactive[E,F] for E != F
        term4_4d = np.einsum('ab,ef->aebf', D_ab, f_inactive, optimize=True)
        
        # Zero out diagonal E=F elements (1 - δ_EF means exclude diagonal)
        for e in range(n_inactive):
            term4_4d[:, e, :, e] = 0.0
        
        H_4d += term4_4d
        
        # Reshape to 2D: H[A*n_inactive + E, B*n_inactive + F]
        H = H_4d.reshape(n_virt * n_inactive, n_virt * n_inactive)
        
        return H
        
    # =========================================================================
    # MAIN OVOS LOOP
    # =========================================================================
    
    def run_ovos(
        self, 
        mo_coeffs: np.ndarray
    ) -> Tuple[List[float], List[int], np.ndarray]:
        """
        Run the OVOS optimization loop.
        
        Iterates between MP2 energy calculation and orbital optimization
        until convergence.
        
        Parameters
        ----------
        mo_coeffs : ndarray
            Initial MO coefficients [alpha, beta]
        
        Returns
        -------
        lst_E_corr : list
            Correlation energies at each iteration
        lst_iter_counts : list
            Iteration counts
        mo_coeffs : ndarray
            Final optimized MO coefficients
        """
        converged = False
        iter_count = 0
        lst_E_corr = []
        lst_iter_counts = []
        
        while not converged and iter_count < self.MAX_ITERATIONS:
            iter_count += 1
            print(f"OVOS Iteration {iter_count}")
            
            # Compute MP2 energy and amplitudes
            E_corr, MP1_amplitudes, eri_spin, eri_phys, eri_as, Fmo_spin = \
                self.MP2_energy(mo_coeffs)
            
            # Check convergence
            if iter_count > 1:
                delta_E = abs(E_corr - lst_E_corr[-1])
                if delta_E < self.CONVERGENCE_THRESHOLD:
                    converged = True
                    print(f"  Converged! ΔE = {delta_E:.2e}")
                    lst_E_corr.append(E_corr)
                    lst_iter_counts.append(iter_count)
                    break
            
            lst_E_corr.append(E_corr)
            lst_iter_counts.append(iter_count)
            
            print(f"  E_corr = {E_corr:.10f}")
            
            # Orbital optimization
            mo_coeffs, _ = self.orbital_optimization(
                mo_coeffs, MP1_amplitudes, eri_spin, eri_phys, eri_as, Fmo_spin
            )
        
        if not converged:
            print(f"Warning: OVOS did not converge in {self.MAX_ITERATIONS} iterations")
        
        # Print final summary
        print()
        print("=" * 50)
        print("OVOS Final Summary")
        print("=" * 50)
        print(f"  Total iterations:  {iter_count}")
        print(f"  Final E_corr:      {lst_E_corr[-1]:.10f}")
        print(f"  Reference MP2:     {self.mp2_ecorr_ref:.10f}")
        print(f"  Difference:        {lst_E_corr[-1] - self.mp2_ecorr_ref:.10f}")
        print("=" * 50)
        print()
        
        return lst_E_corr, lst_iter_counts, mo_coeffs


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_ovos_scan(
    molecule: str,
    basis: str,
    use_prev_orbs: bool = False,
    use_random_init: bool = False,
    n_random_attempts: int = 100,
    output_dir: str = "branch/data"
) -> List[Tuple[int, float, int]]:
    """
    Run OVOS for different numbers of optimized virtual orbitals.
    
    Parameters
    ----------
    molecule : str
        Molecule name (e.g., "H2", "CH2")
    basis : str
        Basis set name
    use_prev_orbs : bool
        Reuse optimized orbitals from previous run
    use_random_init : bool
        Use random unitary initialization
    n_random_attempts : int
        Number of random attempts if use_random_init=True
    output_dir : str
        Directory for output files
    
    Returns
    -------
    results : list of tuples
        (num_opt_virt, E_corr, n_iterations) for each run
    """
    # Validate inputs
    if molecule not in MOLECULES:
        raise ValueError(f"Unknown molecule: {molecule}. Available: {list(MOLECULES.keys())}")
    if basis not in BASIS_SETS:
        raise ValueError(f"Unknown basis: {basis}. Available: {BASIS_SETS}")
    
    # Initialize molecule
    mol = pyscf.M(atom=MOLECULES[molecule], basis=basis, unit="angstrom")
    uhf = pyscf.scf.UHF(mol).run()
    mp2_ref = uhf.MP2().run()
    
    # Get system info
    n_electrons = mol.nelec[0] + mol.nelec[1]
    n_spatial = uhf.mo_coeff[0].shape[1]
    max_virt = 2 * n_spatial - n_electrons
    
    print("=" * 60)
    print(f"OVOS Scan: {molecule} / {basis}")
    print("=" * 60)
    print(f"  Electrons:          {n_electrons}")
    print(f"  Spatial orbitals:   {n_spatial}")
    print(f"  Max virtual orbs:   {max_virt}")
    print(f"  Reference MP2:      {mp2_ref.e_corr:.10f}")
    print("=" * 60)
    print()
    
    # Results storage
    results = []
    mo_coeffs_prev = None
    
    # Scan over virtual orbital counts
    increment = 2  # Closed-shell increment
    num_virt = 0
    
    while num_virt < max_virt:
        num_virt += increment
        
        if num_virt > max_virt:
            break
        
        print(f"\n{'='*40}")
        print(f"Virtual orbitals: {num_virt}")
        print(f"{'='*40}")
        
        try:
            # Set initial orbitals
            if use_prev_orbs and mo_coeffs_prev is not None:
                init_orbs = mo_coeffs_prev
                print("  Using previous optimized orbitals")
            elif use_random_init:
                # Random initialization with best-of-N
                best_E = np.inf
                best_mo = None
                
                for attempt in range(n_random_attempts):
                    # Generate random unitary for virtual space
                    n_virt_spatial = n_spatial - n_electrons // 2
                    Q, _ = np.linalg.qr(np.random.randn(n_virt_spatial, n_virt_spatial))
                    
                    # Apply to virtual orbitals
                    mo_test = [uhf.mo_coeff[0].copy(), uhf.mo_coeff[1].copy()]
                    n_occ = n_electrons // 2
                    for spin in [0, 1]:
                        mo_test[spin][:, n_occ:] = mo_test[spin][:, n_occ:] @ Q
                    
                    # Run OVOS
                    ovos = OVOS(mol, num_virt, np.array(mo_test))
                    E_list, _, mo_out = ovos.run_ovos(np.array(mo_test))
                    
                    if E_list[-1] < best_E:
                        best_E = E_list[-1]
                        best_mo = mo_out
                
                mo_coeffs_prev = best_mo
                results.append((num_virt, best_E, -1))  # -1 = multiple attempts
                continue
            else:
                init_orbs = "UHF"
                mo_coeffs = uhf.mo_coeff
            
            # Run OVOS
            ovos = OVOS(mol, num_virt, init_orbs)
            E_list, iter_list, mo_coeffs_out = ovos.run_ovos(
                mo_coeffs if init_orbs == "UHF" else init_orbs
            )
            
            # Store results
            results.append((num_virt, E_list[-1], len(E_list)))
            mo_coeffs_prev = mo_coeffs_out
            
            print(f"  Final E_corr: {E_list[-1]:.10f}")
            print(f"  Iterations:   {len(E_list)}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((num_virt, np.nan, -1))
    
    # Print summary
    print("\n" + "=" * 60)
    print("OVOS Scan Complete")
    print("=" * 60)
    print(f"{'Virt Orbs':>10} {'E_corr':>18} {'Iterations':>12} {'% Recovery':>12}")
    print("-" * 60)
    
    for n_virt, E_corr, n_iter in results:
        if np.isnan(E_corr):
            print(f"{n_virt:>10} {'FAILED':>18} {'-':>12} {'-':>12}")
        else:
            recovery = 100.0 * E_corr / mp2_ref.e_corr if mp2_ref.e_corr != 0 else 0
            print(f"{n_virt:>10} {E_corr:>18.10f} {n_iter:>12} {recovery:>11.2f}%")
    
    print("-" * 60)
    print(f"{'Full MP2':>10} {mp2_ref.e_corr:>18.10f}")
    print("=" * 60)
    
    # Save results
    import os
    save_dir = f"{output_dir}/{molecule}/{basis}"
    os.makedirs(save_dir, exist_ok=True)
    
    suffix = ""
    if use_prev_orbs:
        suffix = "_prev"
    elif use_random_init:
        suffix = "_random"
    else:
        suffix = "_default"
    
    filename = f"{save_dir}/lst_MP2_different_virt_orbs{suffix}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")
    
    return results


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Configuration
    SELECT_MOLECULE = "H2"
    SELECT_BASIS = "6-31G"
    
    # Run options (only one should be True, or both False)
    USE_PREV_ORBS = False
    USE_RANDOM_INIT = False
    N_RANDOM_ATTEMPTS = 100
    
    # Execute scan
    results = run_ovos_scan(
        molecule=SELECT_MOLECULE,
        basis=SELECT_BASIS,
        use_prev_orbs=USE_PREV_ORBS,
        use_random_init=USE_RANDOM_INIT,
        n_random_attempts=N_RANDOM_ATTEMPTS,
    )
