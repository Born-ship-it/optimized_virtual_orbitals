"""
OVOS class

The OVOS algorithm minimizes the second-order correlation energy (MP2)
using orbital rotations.


Notes:
- VQE in SlowQaunt 
- Clean up this code!!! make new file for just OVOS!!!
- Finish dissociation plot w. .75 or .9% of virtual space optimized
"""

import os
# Force single-threaded
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from typing import Tuple

import numpy as np
import scipy
import pyscf

from decimal import Decimal


# Utilities for non-blocking keyboard input (for interactive stopping of optimization)
import sys
import select
import tty
import termios
import time

def is_key_pressed(timeout=0):
    """Return True if a key has been pressed on stdin within `timeout` seconds."""
    return select.select([sys.stdin], [], [], timeout) == ([sys.stdin], [], [])

def get_key():
    """Read a single character from stdin (non‑blocking, assumes raw mode is on)."""
    return sys.stdin.read(1)

class BreakOuterLoop(Exception):
    pass

def raw_print(*args, **kwargs):
    # Ensure each printed line ends with \r\n
    kwargs.setdefault('end', '\r\n')
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)

# For memory profiling (optional)
import tracemalloc

# Options:
np.set_printoptions(precision=4, suppress=False, linewidth=200)

class OVOS:
	"""
	The OVOS algorithm minimizes the second-order correlation energy (MP2) using orbital rotations. 

	Operate in 3 spaces:
	- Active occupied space (I,J indices)  -> "Inactive"
	- Active virtual space (A,B indices)   -> "Active"
	- Inactive virtual space (E,F indices) -> "Virtual"

	Implemenation is based on:
	[L. Adamowicz & R. J. Bartlett (1987)](https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level)

    Parameters
    ----------
    mol : pyscf.M
        PySCF molecule object.
    num_opt_virtual_orbs : int
        Number of optimized virtual spin orbitals.
    init_orbs : str, 
        Initial orbitals.
    """

	def __init__(self, mol: pyscf.gto.Mole, scf, Fao, num_opt_virtual_orbs: int, mo_coeff, init_orbs: str = "RHF", start_guess = "RHF") -> None:
		# Is it use_random_unitary_init?
		self.use_random_unitary_init = (start_guess == "random")

		# Store parameters
		self.mol = mol
		self.num_opt_virtual_orbs = num_opt_virtual_orbs
		self.init_orbs = init_orbs
		self.mo_coeffs = mo_coeff
		self.Fao = Fao

		# Perform initial Hartree-Fock calculation to get orbitals
		if init_orbs == "UHF":
			# Set up unrestricted Hartree-Fock calculation 
			self.uhf = scf
			self.e_rhf = self.uhf.e_tot
			self.h_nuc = mol.energy_nuc()
			self.scf = self.uhf

		if init_orbs == "RHF":
			# Set up restricted Hartree-Fock calculation 
			self.rhf = scf
			self.e_rhf = self.rhf.e_tot
			self.h_nuc = mol.energy_nuc()
			self.scf = self.rhf

		# MP2 calculation
		# self.MP2 = self.scf.MP2().run()
		# self.MP2_ecorr = self.MP2.e_corr

		# Overlap matrix check
		if not hasattr(self, 'S'):
			self.S = mol.intor('int1e_ovlp')
		
		# Integrals in AO basis
		if not hasattr(self, 'hcore_ao'):
			self.hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
		if not hasattr(self, 'eri_4fold_ao'):
			self.eri_4fold_ao = mol.intor('int2e_sph', aosym=1)

		# Number of orbitals
		n_spatial_orbs = int(self.mo_coeffs[0].shape[0])  # Number of spatial orbitals (same for alpha and beta)
		self.tot_num_spin_orbs = int(2*n_spatial_orbs)  # Total number of spin orbitals (alpha + beta)
			
			# Check that orbitals are unrestricted
		assert self.mo_coeffs[0].shape[0] == self.mo_coeffs[1].shape[0], "Number of alpha and beta orbitals must be equal for OVOS."
		
		# Number of electrons
		self.nelec = self.mol.nelec[0] + self.mol.nelec[1]

		# Build index lists of active and inactive spaces
			# I,J indices -> occupied spin orbitals
		self.active_occ_indices = [i for i in range(int(self.nelec))]
			# A, B indices -> inoccupied spin orbitals in active space
		self.active_inocc_indices = [i for i in range(self.active_occ_indices[-1]+1,int((self.num_opt_virtual_orbs+self.nelec)))]
			# E, F indices -> inoccipoed spin in virtual space
		self.virtual_inocc_indices = [i for i in range(self.active_inocc_indices[-1]+1,int((self.tot_num_spin_orbs)))]
			# actice + inactive space
		self.inactive_indices = [i for i in range(self.active_inocc_indices[-1]+1,int((self.tot_num_spin_orbs)))]
			# Full space
		self.full_indices = [i for i in range(int(self.tot_num_spin_orbs))]


		# Print information about the spaces
		if not self.use_random_unitary_init:
			raw_print()
			raw_print("#### Active and inactive spaces ####")
			raw_print("Total number of spin-orbitals: ", self.tot_num_spin_orbs)
			raw_print("Active occupied spin-orbitals: ", self.active_occ_indices)
			raw_print("Active unoccupied spin-orbitals: ", self.active_inocc_indices)
			raw_print("Inactive unoccupied spin-orbitals: ", self.inactive_indices)
			raw_print()

		# Checks and balances
			# Check that the number of optimized virtual orbitals is not too large
		assert self.tot_num_spin_orbs >= self.num_opt_virtual_orbs+self.nelec, "Your space 'num_opt_virtual_orbs' is too large"  



	# Methods for converting between spatial and spin-orbital representations
	def spatial_to_spin_mo_energy(self, mo_energy_a, mo_energy_b):
		"""
		Convert spatial MO energies to spin-orbital MO energies.
		Equivalent to the Fortran subroutine spatial_to_spin_MO_energy.
		
		Parameters:
		-----------
		mo_energy_a : np.ndarray
			Alpha MO energies, shape (n_spatial,)
		mo_energy_b : np.ndarray
			Beta MO energies, shape (n_spatial,)
			
		Returns:
		--------
		orbital_energies : np.ndarray
			Spin-orbital MO energies, shape (n_spin,)
				Where n_spin = 2 * n_spatial
			The spin-orbital MO energies are ordered such that the first half of the indices correspond to alpha spin and the second half correspond to beta spin.
				For example, if n_spatial = 3, then the spin-orbital indices would be:
					0: spatial orbital 0, alpha spin
					1: spatial orbital 0, beta spin
					2: spatial orbital 1, alpha spin
					3: spatial orbital 1, beta spin
					4: spatial orbital 2, alpha spin
					5: spatial orbital 2, beta spin
		"""
		n_spatial = len(mo_energy_a)

		# Create list of (energy, spin, spatial_index, spin_orbital_index)
		spin_data = []
		for i in range(n_spatial):
			# Alpha: spin-orbital index = 2*i
			spin_data.append((mo_energy_a[i], 0, i, 2*i))      # (energy, spin, spatial_idx, spin_orb_idx)
			# Beta: spin-orbital index = 2*i + 1
			spin_data.append((mo_energy_b[i], 1, i, 2*i + 1))  # (energy, spin, spatial_idx, spin_orb_idx)

		# Extract arrays
		orbital_energies = np.array([d[0] for d in spin_data])

		return orbital_energies

	def spatial_to_spin_fock_optimized(self, Fmo_a, Fmo_b):
		"""
		Convert spatial Fock matrices to spin-orbital Fock matrix.
		Parameters:
		-----------
		Fmo_a : np.ndarray
			Alpha Fock matrix in spatial orbital basis, shape (n_spatial, n_spatial)
		Fmo_b : np.ndarray
			Beta Fock matrix in spatial orbital basis, shape (n_spatial, n_spatial)
		Returns:
		--------
		Fmo_spin : np.ndarray
			Spin-orbital Fock matrix, shape (n_spin, n_spin)
				Where n_spin = 2 * n_spatial
			The spin-orbital Fock matrix is ordered such that the first half of the indices correspond to alpha spin and the second half correspond to beta spin.
				For example, if n_spatial = 3, then the spin-orbital indices would be:
					0: spatial orbital 0, alpha spin
					1: spatial orbital 0, beta spin
					2: spatial orbital 1, alpha spin
					3: spatial orbital 1, beta spin
					4: spatial orbital 2, alpha spin
					5: spatial orbital 2, beta spin
		"""
		n_spatial = self.Fao[0].shape[0]  # Number of spatial orbitals
		n_spin = 2 * n_spatial
		
		# Initialize spin Fock matrix
		Fmo_spin = np.zeros((n_spin, n_spin), dtype=np.float64)
		
		# Alpha block (even rows, even columns)
		Fmo_spin[0::2, 0::2] = Fmo_a
		
		# Beta block (odd rows, odd columns)
		Fmo_spin[1::2, 1::2] = Fmo_b
		
		return Fmo_spin
	


	# Main method to compute MP2 energy and amplitudes
	def MP2_energy(self, mo_coeffs, Fmo) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
		"""
		MP2 correlation energy for unrestricted orbitals in spin-orbital basis.
		Amplitudes are computed in spin-orbital basis as well.

		Parameters
		----------
		mo_coeffs : list of np.ndarray
			List containing alpha and beta MO coefficients, each of shape (n_ao, n_mo)
		Fmo : np.ndarray or None
			Fock matrix in spin-orbital basis, shape (n_spin, n_spin)
				If None, it will be computed from the spatial Fock matrices and converted to spin-orbital basis.
				If provided, it will be used directly without conversion. This allows us to use the rotated Fock matrix in subsequent iterations without having to convert from spatial to spin-orbital basis again.
					Where n_spin = 2 * n_spatial
				The spin-orbital Fock matrix is ordered such that the first half of the indices correspond to alpha spin and the second half correspond to beta spin.
					For example, if n_spatial = 3, then the spin-orbital indices would be:
						0: spatial orbital 0, alpha spin
						1: spatial orbital 0, beta spin
						2: spatial orbital 1, alpha spin
						3: spatial orbital 1, beta spin
						4: spatial orbital 2, alpha spin
						5: spatial orbital 2, beta spin

		Returns
		-------
		E_corr : float
			MP2 correlation energy.
		t1_amplitudes : ndarray
			First-order MP amplitudes.
		eri_spin : ndarray
			Spin-orbital two-electron integrals.
		Fmo_spin : ndarray
			Spin-orbital Fock matrix.
		"""

		# Fock matrix in MO basis 
		if not isinstance(Fmo, np.ndarray):
			Fmo_a = mo_coeffs[0].T @ self.Fao[0] @ mo_coeffs[0]
			Fmo_b = mo_coeffs[1].T @ self.Fao[1] @ mo_coeffs[1]

			# Sanity checks
				# Check that Fock matrices are finite
			assert np.all(np.isfinite(Fmo_a)), "Alpha Fock matrix contains non-finite values!"
			assert np.all(np.isfinite(Fmo_b)), "Beta Fock matrix contains non-finite values!"
				# Check that Fock matrices are not all zero
			assert np.count_nonzero(Fmo_a) > 0, "Alpha Fock matrix is all zero!"
			assert np.count_nonzero(Fmo_b) > 0, "Beta Fock matrix is all zero!"
				# Check that Fock matrices are Hermitian
			assert np.allclose(Fmo_a, Fmo_a.T.conj(), atol=1e-10), "Alpha Fock matrix is not Hermitian!"
			assert np.allclose(Fmo_b, Fmo_b.T.conj(), atol=1e-10), "Beta Fock matrix is not Hermitian!"

			# Convert to spin-orbital basis
			Fmo_spin_opt = self.spatial_to_spin_fock_optimized(Fmo_a, Fmo_b)

				# Use optimized version
			Fmo_spin = Fmo_spin_opt

			# After first iteration, we can use the rotated Fock matrix directly without conversion
		else:
			# Use rotated Fock matrix
			Fmo_spin = Fmo


		# Sanity checks
			# Check that Fock matrix is finite
		assert np.all(np.isfinite(Fmo_spin)), "Spin-orbital Fock matrix contains non-finite values!"
			# Check that Fock matrix is not all zero
		assert np.count_nonzero(Fmo_spin) > 0, "Spin-orbital Fock matrix is all zero!"
			# Check that Fock matrix is Hermitian
		assert np.allclose(Fmo_spin, Fmo_spin.T.conj(), atol=1e-10), "Spin-orbital Fock matrix is not Hermitian!"


		if not isinstance(Fmo, np.ndarray):
			# Get orbital energies
			eigval_a, eigvec_a = scipy.linalg.eigh(Fmo_a)  # Use eigh for Hermitian matrices
			eigval_b, eigvec_b = scipy.linalg.eigh(Fmo_b)  # Use eigh for Hermitian matrices

			# Already sorted by eigh
			mo_energy_a = np.real(eigval_a)
			mo_energy_b = np.real(eigval_b)

			# Convert to spin orbitals with proper sorting
			self.eps = self.spatial_to_spin_mo_energy(mo_energy_a, mo_energy_b)

			# Set for convenience
			eps = self.eps
		else:
			self.eps = np.diag(Fmo_spin)  # Approximate orbital energies from diagonal of Fock matrix in spin-orbital basis
			eps = self.eps

		# Sanity checks
			# Check that orbital energies are finite
		assert np.all(np.isfinite(eps)), "Spin-orbital MO energies contain non-finite values!"
			# Check that orbital energies are not all zero
		assert np.count_nonzero(eps) > 0, "Spin-orbital MO energies are all zero!"


		# i) Fock matrix in spin-orbital basis & Two-electron integrals in spin-orbital basis
			# OPTIMIZATION: Only transform MO columns we need.
				# All downstream eri_as accesses have last 2 indices = occupied.
				# First 2 indices are always virtual (active or inactive).
				# So we only need (virtual, virtual | occupied, occupied) blocks.

		# norb_alpha = mo_coeffs[0].shape[1]
		# norb_beta  = mo_coeffs[1].shape[1]
		nocc_a = self.mol.nelec[0]
		nocc_b = self.mol.nelec[1]

		C_vir_a = mo_coeffs[0][:, nocc_a:]   # all virtual alpha MOs
		C_vir_b = mo_coeffs[1][:, nocc_b:]   # all virtual beta MOs
		C_occ_a = mo_coeffs[0][:, :nocc_a]   # occupied alpha MOs
		C_occ_b = mo_coeffs[1][:, :nocc_b]   # occupied beta MOs

		nvir_a = C_vir_a.shape[1]
		nvir_b = C_vir_b.shape[1]

		#PySCF stores 2e integrals in chemists' notation: (ij|kl) = <ik|jl> in physicists' notation.
			# Partial AO→MO: only (virt, virt | occ, occ) blocks
				# (alpha alpha | alpha alpha) integrals
		# eri_aaaa = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[0], mo_coeffs[0], mo_coeffs[0], mo_coeffs[0]], compact=False)
				# (beta beta | beta beta) integrals
		# eri_bbbb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[1], mo_coeffs[1], mo_coeffs[1], mo_coeffs[1]], compact=False)
				# (alpha alpha | beta beta) integrals
		# eri_aabb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[0], mo_coeffs[0], mo_coeffs[1], mo_coeffs[1]], compact=False)
        		# (beta beta | alpha alpha) integrals - should be the transpose of (alpha alpha | beta beta)

		# We need <ab||ij> in physicist notation = (ai|bj) - (aj|bi) in chemist notation.
		# So the AO→MO blocks needed are (virt, occ | virt, occ), NOT (virt, virt | occ, occ).

		# (α_vir, α_occ | α_vir, α_occ)
		eri_vovo_aaaa = pyscf.ao2mo.kernel(
			self.eri_4fold_ao, [C_vir_a, C_occ_a, C_vir_a, C_occ_a], compact=False, 
		).reshape(nvir_a, nocc_a, nvir_a, nocc_a)

		# (β_vir, β_occ | β_vir, β_occ)
		eri_vovo_bbbb = pyscf.ao2mo.kernel(
			self.eri_4fold_ao, [C_vir_b, C_occ_b, C_vir_b, C_occ_b], compact=False
		).reshape(nvir_b, nocc_b, nvir_b, nocc_b)

		# (α_vir, α_occ | β_vir, β_occ)
		eri_vovo_aabb = pyscf.ao2mo.kernel(
			self.eri_4fold_ao, [C_vir_a, C_occ_a, C_vir_b, C_occ_b], compact=False
		).reshape(nvir_a, nocc_a, nvir_b, nocc_b)

		# (β_vir, β_occ | α_vir, α_occ)
		# eri_vovo_bbaa = pyscf.ao2mo.kernel(
		# 	self.eri_4fold_ao, [C_vir_b, C_occ_b, C_vir_a, C_occ_a], compact=False
		# ).reshape(nvir_b, nocc_b, nvir_a, nocc_a)


		    # reshape AO->MO (chemists' notation)
		# norb_alpha, norb_beta = mo_coeffs[0].shape[1], mo_coeffs[1].shape[1]
		# eri_aaaa = eri_aaaa.reshape((norb_alpha, norb_alpha, norb_alpha, norb_alpha))
		# eri_bbbb = eri_bbbb.reshape((norb_beta,  norb_beta,  norb_beta,  norb_beta))
		# eri_aabb = eri_aabb.reshape((norb_alpha, norb_alpha, norb_beta,  norb_beta))

		# assemble into spin-orbital (virt_spin, occ_spin, virt_spin, occ_spin) - chemist notation
		nvir_spin = 2 * nvir_a   # total virtual spin-orbitals
		nocc_spin = self.nelec   # total occupied spin-orbitals

		eri_vovo = np.zeros((nvir_spin, nocc_spin, nvir_spin, nocc_spin), dtype=np.float64)
		eri_vovo[0::2, 0::2, 0::2, 0::2] = eri_vovo_aaaa
		eri_vovo[1::2, 1::2, 1::2, 1::2] = eri_vovo_bbbb
		eri_vovo[0::2, 0::2, 1::2, 1::2] = eri_vovo_aabb
		eri_vovo[1::2, 1::2, 0::2, 0::2] = eri_vovo_aabb.transpose(2, 3, 0, 1)

		# Convert from chemist's to physicist's notation WITHIN the block:
		#   chemist: (a_vir, i_occ | b_vir, j_occ)  [indices: virt,occ,virt,occ]
		#   physicist: <a_vir, b_vir | i_occ, j_occ> = (a_vir, i_occ | b_vir, j_occ).transpose(0,2,1,3)
		#   Reorder: (virt, occ, virt, occ) -> (virt, virt, occ, occ)
		eri_phys_block = eri_vovo.transpose(0, 2, 1, 3)  # shape: (nvir_spin, nvir_spin, nocc_spin, nocc_spin)

		# Antisymmetrize in physicist's notation:
		#   <ab||ij> = <ab|ij> - <ab|ji>  =>  transpose last two indices
		eri_as = eri_phys_block - eri_phys_block.transpose(0, 1, 3, 2)

		# Clean up
		del eri_vovo, eri_phys_block
		del eri_vovo_aaaa, eri_vovo_bbbb, eri_vovo_aabb



		# ii) Compute MP1 amplitudes (spin-orbital)
			# Optimized computation of MP1 amplitudes only in active space
		def compute_mp1_amplitudes_block(self, eps, eri_as_block):
			"""
			Compute MP1 amplitudes in spin-orbital basis 
				using an optimized approach
				- avoids explicit 4-fold nested loops
				- reduces memory usage by processing only the active block of amplitudes.
			
			Parameters:
			-----------
			eps : np.ndarray
				Spin-orbital MO energies, shape (n_spin,)
			eri_as_block : np.ndarray
				Antisymmetrized two-electron integrals in spin-orbital basis,
				shape (nvir_spin, nvir_spin, nocc_spin, nocc_spin)
				where virtual indices are [0..nvir_spin) and occ indices are [0..nocc_spin).
				Active virtual orbitals are the first nvir_act of the virtual block.

			Returns:
			--------
			t_block : np.ndarray
				First-order MP amplitudes, shape (nvir_act, nvir_act, nocc, nocc)
				where nvir_act = len(self.active_inocc_indices), nocc = len(self.active_occ_indices).
				The amplitude t_ij^{ab} corresponds to the excitation of electrons from 
				occupied orbitals i,j to active virtual orbitals a,b, computed as:
				t_ij^{ab} = - <ab||ij> / (ε_a + ε_b - ε_i - ε_j)
			"""
			# Get active indices
			occ_idx = self.active_occ_indices
			vir_idx = self.active_inocc_indices
			nvir_act = len(vir_idx)
			nocc = len(occ_idx)
			
			# Pre-extract energies for active orbitals
			eps_occ = eps[occ_idx]
			eps_vir = eps[vir_idx]
			
			# Create energy denominator tensor using broadcasting
				# Denominator: ε_a + ε_b - ε_i - ε_j
			eps_vir_a = eps_vir.reshape(-1, 1, 1, 1)        # shape: (nvir_act, 1, 1, 1)
			eps_vir_b = eps_vir.reshape(1, -1, 1, 1)        # shape: (1, nvir_act, 1, 1)
			eps_occ_i = eps_occ.reshape(1, 1, -1, 1)        # shape: (1, 1, nocc, 1)
			eps_occ_j = eps_occ.reshape(1, 1, 1, -1)        # shape: (1, 1, 1, nocc)
			
			# Compute denominator tensor
			denominator = eps_vir_a + eps_vir_b - eps_occ_i - eps_occ_j

				# Check that denominator does not contain zeros to avoid division by zero
			assert not np.any(np.isclose(denominator, 0.0, atol=1e-10)), "Energy denominator contains values close to zero, which may lead to numerical instability in MP1 amplitude calculation!"
			
			# Extract the relevant block of antisymmetrized integrals
				# eri_as_block is (nvir_spin, nvir_spin, nocc_spin, nocc_spin)
				# Active virtual = first nvir_act rows/cols of virtual block
				# All occ indices are already [0..nocc_spin)
			integral_block = eri_as_block[:nvir_act, :nvir_act, :, :]
			
			# Compute MP1 amplitudes for active block only
			t_block = -integral_block / denominator
			
			# Return only the block — shape: (nvir_act, nvir_act, nocc, nocc)
			return t_block

		# Get MP1 amplitudes using block method (for testing)
		MP1_amplitudes = compute_mp1_amplitudes_block(self, eps, eri_as)

		# Sanity checks
			# Check that amplitudes are finite
		assert np.all(np.isfinite(MP1_amplitudes)), "MP1 amplitudes contain non-finite values!"
			# Check that amplitudes are not all zero
		assert np.count_nonzero(MP1_amplitudes) > 0, "MP1 amplitudes are all zero!"
			# Check amplitude antisymmetry: 
				# t_ij^{ab} = t_ji^{ba}
		assert np.allclose(MP1_amplitudes, MP1_amplitudes.transpose(1,0,3,2), atol=1e-10), "MP1 amplitudes do not satisfy antisymmetry t_ij^ab = t_ji^ba!"

        # iii) Compute MP2 correlation energy (spin-orbital indices)
		def compute_mp2_energy_optimized(self, eps, Fmo_spin, eri_as, MP1_amplitudes):
			"""
			Compute MP2 correlation energy using block-shaped arrays:
				eri_as has shape (nvir_spin, nvir_spin, nocc_spin, nocc_spin)
				MP1_amplitudes has shape (nvir_act, nvir_act, nocc, nocc)
			All with block-local indexing (0-based within each block).

			eps and Fmo_spin use absolute spin-orbital indices.
			
			NOTE: Uses the FULL Fock sub-matrix (not just diagonal) to correctly
			handle non-canonical orbitals after orbital rotations.
			"""
			# Absolute index lists
			occ_indices = self.active_occ_indices      # absolute: [0, 1, ..., nelec-1]
			virt_indices = self.active_inocc_indices    # absolute: [nelec, ..., nelec+num_opt-1]

			nocc = len(occ_indices)
			nvir = len(virt_indices)

			if nocc == 0 or nvir == 0:
				return 0.0

			# Absolute arrays (for eps / Fmo_spin lookups)
			occ_abs = np.array(occ_indices)
			vir_abs = np.array(virt_indices)

			J_2 = 0.0

			# All a>b pairs using LOCAL virtual indices 0..nvir-1
			a_loc, b_loc = np.triu_indices(nvir, k=1)
			n_ab_pairs = len(a_loc)

			# Build virtual Fock sub-matrix using the FULL block (not just diagonal)
			# This correctly handles non-canonical orbitals after rotations
			F_virt = Fmo_spin[np.ix_(vir_abs, vir_abs)]  # shape: (nvir, nvir) — FULL sub-matrix

			# Chunking to limit memory
			chunk_size = min(500, n_ab_pairs)

			# Loop over occupied i>j using LOCAL occupied indices
			for loc_i in range(nocc):
				abs_i = occ_abs[loc_i]
				f_ii = Fmo_spin[abs_i, abs_i]
				for loc_j in range(loc_i):
					abs_j = occ_abs[loc_j]
					f_jj = Fmo_spin[abs_j, abs_j]
					eps_ij_sum = f_ii + f_jj

					# MP1_amplitudes is (nvir_act, nvir_act, nocc, nocc) — local indices
					t_abij = MP1_amplitudes[a_loc, b_loc, loc_i, loc_j]  # shape: (n_ab_pairs,)

					# TERM 2: 2 * sum_{a>b} t * <ab|ij>
					integrals = eri_as[a_loc, b_loc, loc_i, loc_j]   # shape: (n_ab_pairs,)
					term2 = 2.0 * np.dot(t_abij, integrals)

					# TERM 1: double sum with bracket
					term1 = 0.0

					for start_row in range(0, n_ab_pairs, chunk_size):
						end_row = min(start_row + chunk_size, n_ab_pairs)
						chunk_rows = slice(start_row, end_row)

						t_row_chunk = t_abij[chunk_rows, None]  # (chunk_size, 1)
						a_loc_chunk = a_loc[chunk_rows]
						b_loc_chunk = b_loc[chunk_rows]

						for start_col in range(0, n_ab_pairs, chunk_size):
							end_col = min(start_col + chunk_size, n_ab_pairs)
							chunk_cols = slice(start_col, end_col)

							t_col_chunk = t_abij[None, chunk_cols]  # (1, chunk_size)
							a_loc_all = a_loc[chunk_cols]
							b_loc_all = b_loc[chunk_cols]

							# Masks using LOCAL indices (chunk_rows_size, chunk_cols_size)
							delta_ac = (a_loc_chunk[:, None] == a_loc_all[None, :])
							delta_bd = (b_loc_chunk[:, None] == b_loc_all[None, :])
							delta_ad = (a_loc_chunk[:, None] == b_loc_all[None, :])
							delta_bc = (b_loc_chunk[:, None] == a_loc_all[None, :])

							# Initialize bracket chunk
							bracket_chunk = np.zeros((end_row - start_row, end_col - start_col))

							# 1. f_ac δ_bd  — F_virt is now the FULL sub-matrix
							if np.any(delta_bd):
								f_ac = F_virt[a_loc_chunk[:, None], a_loc_all[None, :]]
								bracket_chunk += f_ac * delta_bd

							# 2. f_bd δ_ac
							if np.any(delta_ac):
								f_bd = F_virt[b_loc_chunk[:, None], b_loc_all[None, :]]
								bracket_chunk += f_bd * delta_ac

							# 3. -f_ad δ_bc
							if np.any(delta_bc):
								f_ad = F_virt[a_loc_chunk[:, None], b_loc_all[None, :]]
								bracket_chunk -= f_ad * delta_bc

							# 4. -f_bc δ_ad
							if np.any(delta_ad):
								f_bc = F_virt[b_loc_chunk[:, None], a_loc_all[None, :]]
								bracket_chunk -= f_bc * delta_ad

							# 5. - (ε_i + ε_j)(δ_ac δ_bd - δ_ad δ_bc)
							mask_ac_bd = delta_ac & delta_bd
							if np.any(mask_ac_bd):
								bracket_chunk[mask_ac_bd] -= eps_ij_sum

							mask_ad_bc = delta_ad & delta_bc
							if np.any(mask_ad_bc):
								bracket_chunk[mask_ad_bc] += eps_ij_sum

							# Accumulate chunk contribution
							term1 += np.sum(t_row_chunk * bracket_chunk * t_col_chunk)

					J_2 += term1 + term2

			return J_2
			
		def compute_mp2_energy_standard(self, t_block, eri_as_block):
			"""
			Compute MP2 correlation energy using the standard closed-form:
				J_2 = 1/4 * sum_{abij} t_ij^{ab} <ab||ij>
			
			This is equivalent to the full Hylleraas functional 
			(see derivation in copilot-instructions.md):
				J_2 = sum_{i>j} [ sum_{a>b} sum_{c>d} t^{ab}_{ij} t^{cd}_{ij} bracket 
						+ 2 sum_{a>b} t^{ab}_{ij} <ab||ij> ]
			because the bracket collapses via Kronecker deltas to 
			(eps_a + eps_b - eps_i - eps_j) * delta_{ac}*delta_{bd},
			which combined with Term 2 gives t * <ab||ij> with restricted summation,
			equaling 1/4 * sum_{abij} t * <ab||ij> with unrestricted summation.

			Parameters:
			-----------
			t_block : np.ndarray
				MP1 amplitudes, shape (nvir_act, nvir_act, nocc, nocc)
			eri_as_block : np.ndarray
				Antisymmetrized integrals, shape (nvir_spin, nvir_spin, nocc_spin, nocc_spin)
				Active virtual = first nvir_act indices.

			Returns:
			--------
			J_2 : float
				MP2 correlation energy.
			"""
			nvir_act = t_block.shape[0]
			
			# Extract the active-virtual block of integrals
			eri_act = eri_as_block[:nvir_act, :nvir_act, :, :]
			
			# Standard MP2: J_2 = 0.25 * sum_{abij} t^{ab}_{ij} * <ab||ij>
			J_2 = 0.25 * np.einsum('abij,abij->', t_block, eri_act, optimize=True)
			
			return J_2

		# Get MP2 energy using optimized method
		# J_2 = compute_mp2_energy_standard(self, MP1_amplitudes, eri_as)

		# Get MP2 energy using optimized method
		J_2 = compute_mp2_energy_optimized(self, eps, Fmo_spin, eri_as, MP1_amplitudes)
		self.E_current = J_2

		# Sanity check: optimized vs standard method
		# assert np.isclose(J_2, J_2_standard, atol=1e-10), f"Optimized MP2 energy {J_2} does not match standard MP2 energy {J_2_standard}!"

		# Print MP2 energy for current active space
			# Note: the denominator in the print statement is the total number of virtual orbitals in the active space,
				# which is the sum of active inocc and inactive indices. 
			# This gives a sense of how much of the virtual space is being included in the MP2 energy calculation.
		if not self.use_random_unitary_init:
			raw_print(f"    [{len(self.active_inocc_indices)}/{len(self.active_inocc_indices + self.inactive_indices)}]: MP2 (spin-orbital): ", J_2)

		# Sanity checks
			# Check that MP2 correlation energy is finite
		assert np.isfinite(J_2), "MP2 correlation energy is not finite!"
			# Check that MP2 correlation energy is not positive
		# assert J_2 <= 0.0, "MP2 correlation energy is not negative!"

		return J_2, MP1_amplitudes, eri_as, Fmo_spin	





	def apply_rotation(self, mo_coeffs, Fmo_rot, R_vec):
		"""
		Apply the orbital rotation defined by R_vec to the MO coefficients and Fock matrix.
		
		This is a HELPER method for testing orbital rotations during line search or
		Levenberg-Marquardt iteration in orbital_optimization().
		
		Parameters:
		-----------
		mo_coeffs : np.ndarray, shape (2, n_ao, n_spatial)
			Current MO coefficients [alpha, beta], each shape (n_ao, n_spatial)
		Fmo_rot : np.ndarray, shape (n_spin, n_spin)
			Fock matrix in the spin-orbital MO basis
		R_vec : np.ndarray, shape (nvir*ninact,)
			Vectorized orbital rotation parameters (only active-inactive rotations)
		
		Returns:
		--------
		mo_coeffs_rot : np.ndarray, shape (2, n_ao, n_spatial)
			New MO coefficients after rotation
		Fmo_rot_new : np.ndarray, shape (n_spin, n_spin)
			New Fock matrix in the rotated MO basis (NOT canonicalized)
		"""
		nvir = len(self.active_inocc_indices)
		ninact = len(self.inactive_indices)

		# Reshape R_vec to matrix form: (nvir, ninact)
		R_2d = R_vec.reshape(nvir, ninact)

		# Construct anti-symmetric R_matrix in (nvir + ninact) × (nvir + ninact) block
		# Indices: [0..nvir) = active virtual, [nvir..nvir+ninact) = inactive virtual
		R_matrix = np.zeros((nvir + ninact, nvir + ninact), dtype=np.float64)
		for i in range(nvir):
			for j in range(ninact):
				R_ae = R_2d[i, j]
				R_matrix[nvir + j, i] = R_ae      # Place R_ae at [inactive, active]
				R_matrix[i, nvir + j] = -R_ae     # Place -R_ae at [active, inactive]

		# Compute unitary rotation matrix U = exp(R) for the active+inactive block only
		U_sub = scipy.linalg.expm(R_matrix)

		# Expand U to full spin-orbital space
		# Full space = [occupied] + [active virtual] + [inactive virtual]
		n_occ = len(self.active_occ_indices)
		n_full = len(self.full_indices)
		U_full = np.eye(n_full, dtype=np.float64)
		
		# Place U_sub into the (active_virtual + inactive) block
		# Mapping: full_indices[n_occ:] correspond to local indices [0:nvir+ninact] in U_sub
		for i_full in range(n_occ, n_full):
			for j_full in range(n_occ, n_full):
				i_sub = i_full - n_occ
				j_sub = j_full - n_occ
				U_full[i_full, j_full] = U_sub[i_sub, j_sub]

		# Extract alpha and beta blocks from U_full (spin-orbital → spatial)
		# Convention: even indices = alpha, odd indices = beta
		U_alpha = U_full[0::2, 0::2]  # shape (n_spatial, n_spatial)
		U_beta = U_full[1::2, 1::2]   # shape (n_spatial, n_spatial)

		# Apply rotation to MO coefficients
		mo_coeffs_rot = np.array([
			mo_coeffs[0] @ U_alpha,
			mo_coeffs[1] @ U_beta
		], dtype=np.float64)

		# Apply rotation to Fock matrix (passive transformation: F' = U^T F U)
		Fmo_rot_new = U_full.T @ Fmo_rot @ U_full

		# Sanity checks
		assert np.allclose(U_full.T @ U_full, np.eye(n_full), atol=1e-6), \
			"Rotation matrix U is not unitary!"
		assert np.allclose(Fmo_rot_new, Fmo_rot_new.T, atol=1e-8), \
			"Rotated Fock matrix is not Hermitian!"

		return mo_coeffs_rot, Fmo_rot_new





	def orbital_optimization(self, mo_coeffs, MP1_amplitudes, eri_as, Fmo_spin) -> np.ndarray:

		"""
		Step (v-viii) of the OVOS algorithm: Orbital optimization via orbital rotations.
		
		- Compute gradient, first-order derivatives of the second-order Hylleraas functional, Equation 11a [L. Adamowicz & R. J. Bartlett (1987)]
		
		- Compute Hessiansecond-order derivatives of the second-order Hylleraas functional
		Equation 11b in [L. Adamowicz & R. J. Bartlett (1987)]

		- Use the Newton-Raphson method to minimize the second-order Hylleraas functional, Equations 14 in [L. Adamowicz & R. J. Bartlett (1987)]

		- Construct the unitary orbital rotation matrix U = exp(R), Equation 15 in [L. Adamowicz & R. J. Bartlett (1987)]

		First- and second-order derivatives of the second-order Hylleraas functional
		Equations 11a and 11b in https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level
		"""

		# Precompute D_ab as 2D array		
		def compute_D_ab_block(self, t_block):
			"""
			Compute virtual-virtual density matrix D_ab from MP1 amplitudes block.
			Only considers I<J pairs as in original code.
			
			Parameters:
			-----------
			t_block : np.ndarray
				MP1 amplitudes, shape (nvir_act, nvir_act, nocc, nocc)
			
			Returns:
			--------
			D_ab : np.ndarray
				Density matrix, shape (nvir_act, nvir_act)
			"""
			nocc = t_block.shape[2]
			
			# Get indices where I < J
			i_idx, j_idx = np.tril_indices(nocc, k=-1)
			
			# Extract only I<J pairs: shape (nvir_act, nvir_act, n_pairs)
			t_unique = t_block[:, :, i_idx, j_idx]
			
			# Compute D_ab = sum_{i>j,c} t^{ac}_{ij} * t^{bc}_{ij}
			D_ab = np.einsum('acp,bcp->ab', t_unique, t_unique, optimize=True)

			# For big active spaces, D_ab can have very small values that are effectively zero but not exactly zero due to numerical precision.
			# We can set a threshold to zero out very small values to improve numerical stability in the orbital optimization step.
			D_ab[np.abs(D_ab) < 1e-12] = 0.0
			
			return D_ab

		# Compute D_ab using optimized method
		D_ab = compute_D_ab_block(self, MP1_amplitudes)

		# Check symmetry
		sym_diff = np.max(np.abs(D_ab - D_ab.T))
		if sym_diff > 1e-10  and not self.use_random_unitary_init:
			raw_print("WARNING: D_ab is not symmetric!")
			# Print the non-symmetric parts
			diff_matrix = np.abs(D_ab - D_ab.T)
			idx = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
			raw_print(f"Max diff at ({idx[0]},{idx[1]}): D={D_ab[idx]:.6e}, D.T={D_ab.T[idx]:.6e}")

		# Sanity checks for D_ab
		assert np.count_nonzero(D_ab) > 0, "Density matrix D is all zeros!"
		assert np.all(np.isfinite(D_ab)), "Density matrix D contains non-finite values!"
		assert np.allclose(D_ab, D_ab.T, atol=1e-10), f"Density matrix D is not symmetric! Max diff: {np.max(np.abs(D_ab - D_ab.T))}"






		# i) Compute gradient G and Hessian H
			# ia) Compute gradient G		
		def compute_gradient_block(self, t_block, eri_as_block, D_ab, Fmo_spin):
			"""
			Compute gradient G of J_2 w.r.t. orbital rotations using block arrays.
			
			G_ae = 2 * sum_{i>j, b} t^{ab}_{ij} * <eb||ij>  +  2 * sum_b D_ab * F_be
			
			where a = active virtual (relative index 0..nvir_act-1)
					e = inactive virtual (relative index nvir_act..nvir_spin-1 in the block)
			
			Parameters:
			-----------
			t_block : np.ndarray, shape (nvir_act, nvir_act, nocc, nocc)
			eri_as_block : np.ndarray, shape (nvir_spin, nvir_spin, nocc_spin, nocc_spin)
			D_ab : np.ndarray, shape (nvir_act, nvir_act)
			Fmo_spin : np.ndarray, shape (n_spin, n_spin) - full Fock matrix
			
			Returns:
			--------
			G : np.ndarray, shape (nvir_act, ninactive)
			"""
			vir = np.array(self.active_inocc_indices)
			inactive = np.array(self.inactive_indices)
			
			nvir_act = len(vir)
			ninactive = len(inactive)
			nocc = t_block.shape[2]
			
			if ninactive == 0:
				return np.zeros((nvir_act, 0), dtype=np.float64)
			
			# Get I>J pairs
			i_idx, j_idx = np.tril_indices(nocc, k=-1)
			
			# Extract sub-blocks from eri_as_block (contiguous memory, no fancy indexing)
			# In eri_as_block, indices [0..nvir_act) = active virtual, [nvir_act..nvir_spin) = inactive
			# eri_as_block[e, b, i, j] where e=inactive, b=active virtual
			eri_eb_ij = eri_as_block[nvir_act:, :nvir_act, :, :]  # (ninactive, nvir_act, nocc, nocc)
			
			# Extract i>j pairs
			t_ij = t_block[:, :, i_idx, j_idx]         # (nvir_act, nvir_act, n_pairs)
			eri_ij = eri_eb_ij[:, :, i_idx, j_idx]      # (ninactive, nvir_act, n_pairs)
			
			# Term 1: G[a,e] = 2 * sum_{i>j, b} t[a,b,p] * eri[e,b,p]
			G = 2.0 * np.einsum('abp,ebp->ae', t_ij, eri_ij, optimize=True)
			
			# Term 2: G[a,e] += 2 * sum_b D[a,b] * F[b,e]
			# F_be: Fock matrix between active virtual (b) and inactive (e), in absolute indices
			F_be = Fmo_spin[np.ix_(vir, inactive)]  # (nvir_act, ninactive)
			G += 2.0 * D_ab @ F_be
			
			return G

		# Get gradient using block method
		G = compute_gradient_block(self, MP1_amplitudes, eri_as, D_ab, Fmo_spin)
	
		# Print diagnostics
			# Gradient
		# Norm of the gradient
		self.grad_norm = np.linalg.norm(G.flatten())
		if not self.use_random_unitary_init:
			raw_print(f"    Gradient norm: {self.grad_norm:.6e}")
		
			# Check if gradient has finite values
		assert np.all(np.isfinite(G)), "Gradient G contains non-finite values!"


		def compute_hessian_block(self, t_block, eri_as_block, D_ab, Fmo_spin):
			"""
			Compute Hessian H of J_2 w.r.t. orbital rotations using block arrays.
			Eliminates the inner Python for-loop over occupied pairs.
			
			H_{ae,bf} = 2 * sum_{i>j} t^{ab}_{ij} * <ef||ij>
						- sum_{i>j,c} [t^{ac}_{ij}*<bc||ij> + t^{cb}_{ij}*<ca||ij>] * delta_ef
						+ D_ab * (f_aa + f_bb) * delta_ef
						+ D_ab * f_ef * (1 - delta_ef)
			
			Parameters:
			-----------
			t_block : np.ndarray, shape (nvir_act, nvir_act, nocc, nocc)
			eri_as_block : np.ndarray, shape (nvir_spin, nvir_spin, nocc_spin, nocc_spin)
			D_ab : np.ndarray, shape (nvir_act, nvir_act)
			Fmo_spin : np.ndarray, shape (n_spin, n_spin) - full Fock matrix
			
			Returns:
			--------
			H : np.ndarray, shape (nvir_act * ninactive, nvir_act * ninactive)
			"""
			vir = np.array(self.active_inocc_indices)
			inactive = np.array(self.inactive_indices)
			
			nvir = len(vir)
			ninactive = len(inactive)
			nocc = t_block.shape[2]
			
			if ninactive == 0:
				return np.zeros((0, 0), dtype=np.float64)
			
			# Get I>J pairs
			i_idx, j_idx = np.tril_indices(nocc, k=-1)
			n_pairs = len(i_idx)
			
			H = np.zeros((nvir * ninactive, nvir * ninactive), dtype=np.float64)
			
			# === PREPARE COMMON DATA ===
			f_diag_vir = np.diag(Fmo_spin)[vir]          # (nvir,)
			f_inactive = Fmo_spin[np.ix_(inactive, inactive)]  # (ninactive, ninactive)
			
			# Zero diagonal for term4 (E != F condition)
			f_inactive_offdiag = f_inactive.copy()
			np.fill_diagonal(f_inactive_offdiag, 0.0)
			
			# === Extract i>j slices from block arrays (no Python loop!) ===
			# t_block[:, :, i_idx, j_idx] -> (nvir, nvir, n_pairs)
			t_ij = t_block[:, :, i_idx, j_idx]
			
			# eri_as_block[e, f, i, j] for inactive e,f -> (ninactive, ninactive, nocc, nocc)
			eri_ef_block = eri_as_block[nvir:, nvir:, :, :]
			eri_ef_ij = eri_ef_block[:, :, i_idx, j_idx]  # (ninactive, ninactive, n_pairs)
			
			# eri_as_block[a, b, i, j] for active a,b -> already t_block's index space
			eri_ab_block = eri_as_block[:nvir, :nvir, :, :]
			eri_ab_ij = eri_ab_block[:, :, i_idx, j_idx]   # (nvir, nvir, n_pairs)
			
			# === TERM 1: 2 * sum_{i>j} t^{ab}_{ij} * <ef||ij> ===
			# term1 = 2.0 * np.einsum('abp,efp->aebf', t_ij, eri_ef_ij, optimize=True)
			# H += term1.reshape(nvir * ninactive, nvir * ninactive)

			for a0 in range(0, nvir, nvir):
				a1 = nvir  # process all active virtuals at once (since they are contiguous in the block)
				# t_ij[a0:a1, :, :] shape: (blk, nvir, n_pairs)
				# Result shape: (blk, ninactive, nvir, ninactive)
				partial = 2.0 * np.einsum('abp,efp->aebf', t_ij[a0:a1], eri_ef_ij, optimize=True)
				H[a0*ninactive:(a1)*ninactive, :] += partial.reshape((a1-a0)*ninactive, nvir*ninactive)
			
			# === TERM 2: -sum_{i>j,c} [t^{ac}*eri^{bc} + t^{cb}*eri^{ca}] * delta_ef ===
			# Part 1: sum_{c,p} t[a,c,p] * eri[b,c,p]
			term2_part1 = np.einsum('acp,bcp->ab', t_ij, eri_ab_ij, optimize=True)
			# Part 2: sum_{c,p} t[c,b,p] * eri[c,a,p]
			term2_part2 = np.einsum('cbp,cap->ab', t_ij, eri_ab_ij, optimize=True)
			term2_sum = -(term2_part1 + term2_part2)  # (nvir, nvir)
			
			# === TERM 3: D_ab * (f_aa + f_bb) * delta_ef ===
			# The diagonal Fock second variation gives a SUM of orbital energies,
			# not a difference. The difference (f_aa - f_bb) was an error in the
			# derivation that produced an antisymmetric contribution to H.
			f_sum = f_diag_vir[:, None] + f_diag_vir[None, :] 
			term3 = D_ab * f_sum # (nvir, nvir)
			
			# Add terms 2 and 3 together for the diagonal blocks (e=f) in the Hessian
			combined_diag = term2_sum + term3  # (nvir, nvir)
				# This is equivalent to kron(combined_diag, I_ninactive):
			block_diag = np.kron(combined_diag, np.eye(ninactive))
			H += block_diag.reshape(nvir * ninactive, nvir * ninactive)

			# === TERM 4: D_ab * f_ef * (1 - delta_ef) ===
			# term4 = np.einsum('ab,ef->aebf', D_ab, f_inactive_offdiag, optimize=True)
				# This is equivalent to kron(f_inactive_offdiag, D_ab):
			term4 = np.kron(D_ab, f_inactive_offdiag)
				# Add to H
			H += term4.reshape(nvir * ninactive, nvir * ninactive)
			
			return H

		# Get Hessian using block method
		H = compute_hessian_block(self, MP1_amplitudes, eri_as, D_ab, Fmo_spin)

			# Hessian
		if not self.use_random_unitary_init:
			eigvals = np.linalg.eigvalsh(H)
			n_neg_eigval_H = np.sum(eigvals < 0)
			self.n_block_neg_eigval_H = 0
			det_H = np.linalg.det(H)
			cond_H = np.linalg.cond(H) if H.size > 0 else np.inf

			# Check eigenvalues
			raw_print(f"    Hessian norm: {np.linalg.norm(H):.6e}, (Neg. Eigval: {n_neg_eigval_H}/{len(eigvals)})")
			if np.any(eigvals < 0.0):
				raw_print("        WARNING: Hessian has negative eigenvalues!")				
				# Condition number
			raw_print(f"    Hessian condition number: {cond_H:.6e}, determinant: {det_H:.6e}, size: {H.shape}")

			# Sanity checks for H
			if np.count_nonzero(H) == 0:
				raw_print("        WARNING: Hessian is zero.")
			if not np.all(np.isfinite(H)):
				raw_print("        WARNING: Hessian H contains non-finite values!")
			if not np.allclose(H, H.T, atol=1e-10):
				H_symm_diff = np.max(np.abs(H - H.T))
				raw_print("        WARNING: Hessian H is not symmetric! Max symm diff: ", H_symm_diff)

		if False: # Apply Block-Diagonal Approximation to Hessian Inverse
			len_inac = len(self.inactive_indices)
			len_ac   = len(self.active_inocc_indices)
			H_block_diag = np.zeros_like(H)
			for i in range(len_inac):
				start = i * len_ac
				end   = (i + 1) * len_ac
				H_block_diag[start:end, start:end] = H[start:end, start:end]
			H = H_block_diag

		if self.iteration > 5:
			# Global damping parameter for regularization in Newton-Raphson step
			lambda_lm = 1e-4  # initial damping
			prev_E_trial = E_trial if 'E_trial' in locals() else None
			while True:
				try:
					H_reg = H + lambda_lm * np.eye(H.shape[0])
					R_vec = -np.linalg.solve(H_reg, G.flatten())
					# Test the step (e.g., using line search or simply check if it lowers energy)
					mo_trial, F_trial = self.apply_rotation(mo_coeffs, Fmo_spin, R_vec)
					E_trial, *_ = self.MP2_energy(mo_trial, F_trial)
					rho = (self.E_corr_ - E_trial) / (0.5 * R_vec @ H_reg @ R_vec - R_vec @ G.flatten())
					if rho > 0 and E_trial < self.E_corr_:
						lambda_lm = max(lambda_lm / 2, 1e-12)  # reduce damping
						break
					else:
						lambda_lm *= 2   # increase damping
						if lambda_lm > 1e4:
							break  # prevent infinite loop in case of numerical issues
				except np.linalg.LinAlgError:
					lambda_lm *= 2

			# Apply the accepted rotation
			R = R_vec
		else:
			R = -np.linalg.solve(H, G.flatten())
		







		# Step (vi): Use the Newton-Raphson method to minimize the second-order Hylleraas functional
			# Reduced Linear Equation method: R = -H^{-1} G		
		if False: # Set to True to always apply block-diagonal approximation				
			# Step (vi): Block-diagonal RLE approximation of Hessian inverse
			# Per Adamowicz & Bartlett (1987): invert only diagonal blocks H_{ea,eb}
			# Each block corresponds to rotations of ALL active orbitals (a,b) against ONE inactive orbital (e).
			# Block size: (len_ac, len_ac) for each inactive orbital e.

			len_inac = len(self.inactive_indices)
			len_ac   = len(self.active_inocc_indices) 
			len_occ  = len(self.active_occ_indices)

			cond_H_before  = np.linalg.cond(H) if H.size > 0 else np.inf
			norm_H_before  = np.linalg.norm(H)
			eigvals_before = np.linalg.eigvalsh(H)
			if not self.use_random_unitary_init:
				raw_print(f"         [Pre-RLE]  norm: {norm_H_before:.4e}, cond: {cond_H_before:.4e}, "
					f"Neg eigval: {np.sum(eigvals_before < 0)}/{len(eigvals_before)}")

			H_block_diag = np.zeros_like(H)
			H_block_inv_diag = np.zeros_like(H)
			for i in range(len_inac):
				start = i * len_ac
				end   = (i + 1) * len_ac

				H_block = H[start:end, start:end]
				H_block_diag[start:end, start:end] = H_block.copy()  # Store original block before regularization for diagnostics

				if True:
					# Apply block wise regularization to each block to ensure invertibility and good conditioning
					H_block_orig = H[start:end, start:end].copy()  # Original block before regularization
					eigvals  = np.linalg.eigvalsh(H_block_orig)
					min_eig  = np.min(eigvals)
					eig_ratio = min_eig / max(np.abs(eigvals)) if np.max(np.abs(eigvals)) > 0 else 0.0
					det_val  = np.linalg.det(H_block_orig)
					cond_H = np.linalg.cond(H_block_orig) if H_block_orig.size > 0 else np.inf

					# Check if block needs regularization
					# needs_regularization = (det_val == 0.0) or (min_eig < 0.0) 
					needs_regularization = (
						np.abs(det_val) < 1e-15 or    # Near-singular determinant threshold, 
						min_eig < -1e-10 or           # Negative eigenvalue threshold
						eig_ratio < -0.1 or           # Negative eigenvalue dominates over max eigenvalue
						cond_H > 1e6                  # Ill-conditioning threshold
					) if self.iteration > 10 else False  # Skip regularization for the first iteration to preserve the initial guess structure

					if needs_regularization:
						# if not self.use_random_unitary_init:
						# 	raw_print(f"              Block {i}: norm {np.linalg.norm(H_block_orig):.2e}, min eig {min_eig:.2e}, det {det_val:.2e} → "
						# 	f"{'Needs regularization' if needs_regularization else 'OK'}")

						# --- Regularize: find smallest shift s.t. min_eig >= 1e-2 and det != 0 ---
						# Strategy: H_reg = H_orig + lambda * I, where lambda = shift_scale * ||H_orig||
						H_block_reg = None
						shift_applied = None
						max_shift_exp = 14  # Safety cap: lambda <= 10^14 * ||H_orig||
						H_max = np.max(np.abs(H_block_orig))
						if H_max == 0.0:
							# Block is all zeros — cannot regularize, skip
							if not self.use_random_unitary_init:
								raw_print(f"              Block {i}: all-zero block, cannot regularize. Skipping.")
							H_block_reg = H_block_orig
							shift_applied = 0.0
						else:
							for exp in range(-24, max_shift_exp + 1):
								shift_scale = 10.0**exp
								lambda_reg  = shift_scale * H_max
								H_try       = H_block_orig + lambda_reg * np.eye(len_ac)

								eigvals_try = np.linalg.eigvalsh(H_try)
								min_eig_try = np.min(eigvals_try)
								eig_ratio_try = min_eig_try / max(np.abs(eigvals_try)) if np.max(np.abs(eigvals_try)) > 0 else 0.0
								det_try     = np.linalg.det(H_try)
								cond_H_try  = np.linalg.cond(H_try) if H_try.size > 0 else np.inf

								if (
									np.abs(det_try) != 0.0 and      # Near-singular determinant threshold
									min_eig_try > 0.0 and           # Negative eigenvalue threshold
									cond_H_try < 1e3                # Ill-conditioning threshold
								):
									H_block_reg  = H_try
									shift_applied = lambda_reg
									break

							if H_block_reg is None:
								# Regularization failed — use original and warn
								if not self.use_random_unitary_init:
									raw_print(f"              Block {i}: regularization failed (shift up to 10^{max_shift_exp} * ||H||). Using original block.")
								H_block_reg = H_block_orig
								shift_applied = 0.0
							# else:
							# 	if not self.use_random_unitary_init:
							# 		raw_print(f"              Block {i}: applied shift λ = {shift_applied:.2e} "
							# 			f"(min_eig {min_eig:.2e} → {np.min(np.linalg.eigvalsh(H_block_reg)):.2e})")
						
						H_block_diag[start:end, start:end] = H_block_reg
						H_block = H_block_reg

					else:
						# Block is well-conditioned: use as-is
						H_block_diag[start:end, start:end] = H_block_orig
						H_block = H_block_orig

				# Invert the regularized block and place in the corresponding block of the Hessian inverse
				try:
					H_block_inv = np.linalg.inv(H_block)
				except np.linalg.LinAlgError:
					if not self.use_random_unitary_init:
						raw_print(f"              Block {i}: inversion failed even after regularization. Using pseudo-inverse.")
					H_block_inv = np.linalg.pinv(H_block)
					
					# Store the pseudo-inverse in the Hessian inverse block
				H_block_inv_diag[start:end, start:end] = H_block_inv				
			
			H = H_block_diag

			cond_H_after  = np.linalg.cond(H) if H.size > 0 else np.inf
			norm_H_after  = np.linalg.norm(H)
			eigvals_after = np.linalg.eigvalsh(H)
			if not self.use_random_unitary_init:
				raw_print(f"         [Post-RLE] norm: {norm_H_after:.4e}, cond: {cond_H_after:.4e}, "
					f"Neg eigval: {np.sum(eigvals_after < 0)}/{len(eigvals_after)}")

			# assert H_block_inv_diag is close to np.linalg.inv(H) in the block-diagonal sense
			for i in range(len_inac):
				start = i * len_ac
				end   = (i + 1) * len_ac
				H_block = H[start:end, start:end]
				H_inv_block = H_block_inv_diag[start:end, start:end]
				if not np.allclose(H_block @ H_inv_block, np.eye(len_ac), atol=1e-6):
					raw_print(f"WARNING: Block {i} of Hessian inverse is not accurate! ||H*H_inv - I|| = {np.linalg.norm(H_block @ H_inv_block - np.eye(len_ac)):.2e}")


			# Newton-Raphson step: R = -H^{-1} G
			# Reshape G to vector form for multiplication
			G_vec = G.flatten()  				
			R = - H_block_inv_diag @ G_vec # Size:
		
		# Initialize R,
			# Misc
		vir = np.array(self.active_inocc_indices)
		inact = np.array(self.inactive_indices)
		nvir = len(vir)
		ninact = len(inact)
			# R is a vector of length nvir*ninact, reshape to matrix form for orbital rotations
		R_2d = R.reshape(nvir, ninact)
			# Matrix: nvir+ninact x nvir+ninact
		R_matrix = np.zeros((nvir+ninact, nvir+ninact), dtype=np.float64)
			# Place R_2d in the appropriate block of R_matrix corresponding to rotations between active virtuals and inactive orbitals
		R_vir = np.arange(nvir) # Local indices for active virtuals
		R_inact = np.arange(ninact) + nvir  # Local indices for inactive orbitals
			# Fill the anti-symmetric R_matrix with R_2d and its negative transpose
		for i, a in enumerate(R_vir):
			for j, e in enumerate(R_inact):
				R_ae = R_2d[i, j]
				R_matrix[e, a] = R_ae 			# Note the order of indices for correct placement
				R_matrix[a, e] = -R_ae			# Ensure anti-symmetry

			# Rotation matrix
		# Convergence check based on max element of R_matrix
		max_R_elem = np.max(np.abs(R_matrix))
		if not self.use_random_unitary_init:
			raw_print(f"    Rotation norm {np.linalg.norm(R_matrix):.6e}, (Max el.: {max_R_elem:.6e})")
		# Check that R is anti-symmetric
		diff_R = np.linalg.norm(R_matrix + R_matrix.T)
		assert diff_R < 1e-6, f"R_matrix is not anti-symmetric, ||R + R.T|| = {diff_R}"
		# Check that R_matrix has no NaN or Inf values
		assert np.all(np.isfinite(R_matrix)), "R_matrix contains NaN or Inf values!"


		# Step (vii): Construct the unitary orbital rotation matrix U = exp(R)
		def expm_antisymmetric(R):
			"""
			Compute U = exp(R) for an anti-symmetric matrix R using the
			formula from Adamowicz & Bartlett (1987), Eq. 15:

				U = X cosh(d) X^T + R X sinh(d) d^{-1} X^T

			where d^2 = X^T R^2 X  (eigendecomposition of R^2).

			For antisymmetric R, R^2 is symmetric negative-semidefinite,
			so its eigenvalues are <= 0: d^2_k = -theta_k^2.
			Then cosh(d_k) = cos(theta_k) and sinh(d_k)/d_k = sinc(theta_k).

			Parameters:
			-----------
			R_matrix : np.ndarray, shape (n, n)
				Anti-symmetric matrix (R = -R^T).

			Returns:
			--------
			U : np.ndarray, shape (n, n)
				Orthogonal matrix U = exp(R).
			"""

			# Diagonalize R^2
			eigenvalues, X = np.linalg.eigh(R @ R)
			
			# eigenvalues ≤ 0, so θ = √(-eigenvalues)
			theta = np.sqrt(np.maximum(-eigenvalues, 0))
			
			# Compute trigonometric functions
			cos_theta = np.diag(np.cos(theta))
			
			# sinc(θ) = sin(θ)/θ, with sinc(0) = 1
			sinc_vals = np.ones_like(theta)
			mask = np.abs(theta) > 1e-12
			sinc_vals[mask] = np.sin(theta[mask]) / theta[mask]
			sinc_theta = np.diag(sinc_vals)
			
			# Build U
			U = X @ cos_theta @ X.T + R @ X @ sinc_theta @ X.T

			# print(U.T @ U)

			return U

		# Unitary rotation matrix
			# Use eigendecomposition of anti-symmetric matrix for guaranteed unitarity
		# U_sub = expm_antisymmetric(R_matrix)
		U_sub = scipy.linalg.expm(R_matrix)

			# Expand U to full spin-orbital space
		n_full_space = len(self.full_indices)
		U_full = np.eye(n_full_space)
			# Here U is the rotation in the active virtual space, we need to embed it into the full space.
			# Full space indices: self.full_indices = self.active_inocc_indices + self.inactive_indices
			# We need to place U in the block corresponding to the active virtuals and inactive orbitals, and identity elsewhere.
		for i, idx_i in enumerate(self.full_indices):
			for j, idx_j in enumerate(self.full_indices):
				if idx_i > len(self.active_occ_indices) - 1 and idx_j > len(self.active_occ_indices) - 1:
					# Both indices are in the active virtual + inactive space, use U
					U_full[idx_i, idx_j] = U_sub[i - len(self.active_occ_indices), j - len(self.active_occ_indices)]
				else:
					# At least one index is in the occupied space, keep identity
					U_full[idx_i, idx_j] = 1.0 if idx_i == idx_j else 0.0

		U = U_full

		# Numerical checks on U
		assert np.allclose(U_sub.T @ U_sub, np.eye(len(U_sub)), atol=1e-6), "Unitary rotation matrix U is not unitary!"
			# Check U if U^T U = I 
		assert np.allclose(U.T @ U, np.eye(len(U)), atol=1e-6), "Unitary rotation matrix U is not unitary!"
			# Check is U has no NaN or Inf values
		assert np.all(np.isfinite(U)), "Unitary rotation matrix U contains NaN or Inf values!"
			# Check that U is not all zeros
		assert not np.allclose(U, 0), "Unitary rotation matrix U is all zeros!"

		# Check the active occupied area of U is close to identity
		for i in self.active_occ_indices:
			for j in self.active_occ_indices:
				expected = 1.0 if i == j else 0.0
				if abs(U[i, j] - expected) > 1e-6:
					if not self.use_random_unitary_init:
						raw_print(f"WARNING: U deviates from identity in active occupied block at ({i},{j}): U={U[i,j]:.6e}, expected={expected:.6e}")



		# Step (viii): Rotate the orbitals
			# Apply rotations directly to spatial C_alpha and C_beta

		# The spin-orbital U has block structure:
		# U[0::2, 0::2] acts on alpha orbitals
		# U[1::2, 1::2] acts on beta orbitals  
		# U[0::2, 1::2] ≈ 0 (no alpha-beta mixing in R_matrix by construction)

		U_alpha = U[0::2, 0::2]  # shape (n_spatial, n_spatial)
		U_beta  = U[1::2, 1::2]  # shape (n_spatial, n_spatial)

		mo_coeffs_rot = np.array([
			mo_coeffs[0] @ U_alpha,
			mo_coeffs[1] @ U_beta
			])

		# Rotate the Fock matrix in the MO basis as well
			# Paper: "The Fock matrix is diagonalized to generate new canonical active orbitals."

		Fmo_alpha = Fmo_spin[0::2, 0::2]   # (n_spatial, n_spatial)
		Fmo_beta  = Fmo_spin[1::2, 1::2]   # (n_spatial, n_spatial)

		# Rotate Fock matrix
		Fmo_alpha_rot = U_alpha.T @ Fmo_alpha @ U_alpha
		Fmo_beta_rot  = U_beta.T  @ Fmo_beta  @ U_beta

		# Diagonalize ONLY the active virtual block to get canonical active orbitals
		# Active virtual spin indices: self.active_inocc_indices = [nelec, nelec+nact)
		# In spatial indices (alpha block):
		nocc_spatial = self.mol.nelec[0]
		nact_spatial = self.num_opt_virtual_orbs // 2  # number of active spatial virtual orbs

		act_slice = slice(nocc_spatial, nocc_spatial + nact_spatial)

		# Alpha: diagonalize active virtual block
		F_act_alpha = Fmo_alpha_rot[act_slice, act_slice]
		eigvals_a, eigvecs_a = np.linalg.eigh(F_act_alpha)
		# Apply canonicalization rotation to active block of orbitals
		mo_coeffs_rot[0][:, act_slice] = mo_coeffs_rot[0][:, act_slice] @ eigvecs_a
		# Update rotated Fock: rotate the active block with eigvecs
		Fmo_alpha_rot[act_slice, :] = eigvecs_a.T @ Fmo_alpha_rot[act_slice, :]
		Fmo_alpha_rot[:, act_slice] = Fmo_alpha_rot[:, act_slice] @ eigvecs_a
		# After diagonalization, the active-active block should be diagonal
		# (off-diagonal ~0), and the diagonal = canonical orbital energies

		# Beta: same
		nocc_spatial_b = self.mol.nelec[1]
		act_slice_b = slice(nocc_spatial_b, nocc_spatial_b + nact_spatial)

		F_act_beta = Fmo_beta_rot[act_slice_b, act_slice_b]
		eigvals_b, eigvecs_b = np.linalg.eigh(F_act_beta)
		mo_coeffs_rot[1][:, act_slice_b] = mo_coeffs_rot[1][:, act_slice_b] @ eigvecs_b
		Fmo_beta_rot[act_slice_b, :] = eigvecs_b.T @ Fmo_beta_rot[act_slice_b, :]
		Fmo_beta_rot[:, act_slice_b] = Fmo_beta_rot[:, act_slice_b] @ eigvecs_b

		# Rebuild spin-orbital Fock matrix
		Fmo_rot = np.zeros_like(Fmo_spin)
		Fmo_rot[0::2, 0::2] = Fmo_alpha_rot
		Fmo_rot[1::2, 1::2] = Fmo_beta_rot

		# Sanity checks
		assert np.allclose(Fmo_rot, Fmo_rot.T, atol=1e-8), "Rotated+canonicalized Fock is not symmetric!"
		assert np.allclose(np.sort(np.linalg.eigvalsh(Fmo_rot)), np.sort(np.linalg.eigvalsh(Fmo_spin)), atol=1e-8), \
			"Eigenvalues changed after canonicalization — unitary property violated!"
	
		return mo_coeffs_rot, Fmo_rot



	def run_ovos(self,  mo_coeffs, Fmo_rot):
		"""
		Run the OVOS algorithm.
		"""

		converged = False
		max_iter = 10000
		iter_count = 0

		self.Fmo_rot = Fmo_rot  # Store for access in orbital optimization step
		self.mo_coeffs = mo_coeffs  # Store for access in orbital optimization step

		E_corr = None
		keep_track = 0
		keep_track_max = 50

		self.shift_e_nr = None  # To keep track of the exponent of the last shift applied to the Hessian for diagnostics
		self.shift_e_nr_block = np.full(len(self.inactive_indices), None)  # To keep track of the exponent of the last shift applied to each block of the Hessian for diagnostics

		self.rotation_norm_history = []  # To keep track of the norm of the orbital rotation matrix R at each iteration for diagnostics
		self.max_rotation_norm = 0.3  # Start conservative
		self.trust_radius_history = []
		self.damp_rot = False
		self.diff_E_corr_last = 1.0


		while iter_count < max_iter:
			# Print memory usage
				# If tracemalloc is available, print current memory usage in MB at each iteration to monitor for leaks or excessive usage
			if tracemalloc.is_tracing():
				current, peak = tracemalloc.get_traced_memory()
				raw_print(f"Current memory: {current / 1024 / 1024:.1f} MB | Peak memory: {peak / 1024 / 1024:.1f} MB")

			# Allow user to skip to next iteration by pressing 's' key (with a short timeout to avoid blocking)
				# Check if a key was pressed (with a very short timeout)
			if is_key_pressed(0.01):   # 10 ms timeout
				key = get_key()
				if key == 'i': 
					raw_print("\n[Skipping to next iteration...]")
					break
				if key == 's':
					raise BreakOuterLoop
				if key == '\x03':  # Ctrl-C to quit
					raw_print("\n[Exiting OVOS...]")
					sys.exit(0)

			# If reguælarization failed, break out of the loop to prevent divergence
			if hasattr(self, 'regularization_failed') and self.regularization_failed:
				raw_print("Regularization failed to produce a stable Hessian. Stopping OVOS to prevent divergence.")
				break

			# Increment iteration count
			iter_count += 1
			self.iteration = iter_count

			if not self.use_random_unitary_init:
				raw_print()
				raw_print("#### OVOS Iteration ", iter_count, " ####")

			# Step (iii-iv): Compute MP2 correlation energy and amplitudes
			if not self.use_random_unitary_init:
				raw_print(" Step (iii)-(iv): Compute MP2 corr. energy & amps.")
			E_corr, MP1_amplitudes, eri_as, Fmo_spin = self.MP2_energy(mo_coeffs = mo_coeffs, Fmo = Fmo_rot)
			self.E_corr_ = E_corr  # Store for access in orbital optimization step

			# Step (ix): check convergence and stability

			# =========================================================
			# CONVERGENCE CONTROL 
			# =========================================================
			
			# ---- First iteration: just store and step ----
			if iter_count == 1:
				lst_E_corr = [E_corr]
				lst_iter_counts = [iter_count]
				lst_stop_reason = ["Initial"]
				lst_mo_coeffs = [mo_coeffs]
				lst_Fmo_rot = [Fmo_rot]

				# # Only do first iteration when full space is used for orbital optimization (inoccupied virtual space is empty)
				if len(self.virtual_inocc_indices) == 0:
					break

				# Step (v)-(viii): Orbital optimization
				if not self.use_random_unitary_init:
					raw_print(" Step (v)-(viii): Orbital optimization")
				mo_coeffs, Fmo_rot = self.orbital_optimization(
					mo_coeffs=mo_coeffs,
					MP1_amplitudes=MP1_amplitudes,
					eri_as=eri_as,
					Fmo_spin=Fmo_spin)

			# ---- Subsequent iterations ----
			elif iter_count > 1:
				self.lst_E_corr = lst_E_corr  # Store for access in orbital optimization step

				# Check if correlation energy improved
				lst_E_corr.append(E_corr)
				lst_iter_counts.append(iter_count)
				lst_mo_coeffs.append(mo_coeffs)
				lst_Fmo_rot.append(Fmo_rot)

				# Step (v)-(viii): Orbital optimization
				if not self.use_random_unitary_init:
					raw_print(" Step (v)-(viii): Orbital optimization")
				mo_coeffs, Fmo_rot = self.orbital_optimization(
					mo_coeffs=mo_coeffs,
					MP1_amplitudes=MP1_amplitudes,
					eri_as=eri_as,
					Fmo_spin=Fmo_spin)

			# ---- Update convergence status ----
				if not self.use_random_unitary_init:
					raw_print(" Step (ix): Check convergence and stability")

				# Check if gradient norm is small enough for convergence
					# Gradient norm history, keep track of the last 2 values to check for convergence
				if iter_count == 2:
					lst_grad_norm = [self.grad_norm]
				else:
					lst_grad_norm.append(self.grad_norm)
					if len(lst_grad_norm) > 5:
						lst_grad_norm.pop(0)  # Keep only the last 5 values

				self.lst_grad_norm = lst_grad_norm  # Store for access in orbital optimization step
				diff_grad_norm = abs(lst_grad_norm[-1] - lst_grad_norm[-2]) if len(lst_grad_norm) > 1 else None

				# Check if correlation energy has improved significantly
				diff_E_corr = abs(lst_E_corr[-1] - lst_E_corr[-2])
				self.diff_E_corr = (lst_E_corr[-1] - lst_E_corr[-2])  # Store for access in orbital optimization step

					# Print convergence diagnostics
				if not self.use_random_unitary_init and diff_grad_norm is not None:
					pos_E_corr = "(Pos. ΔE_corr indicates energy increase!)" if self.diff_E_corr > 0 else ""
					raw_print(f"    ΔE_corr = {self.diff_E_corr:.2e} Hartree, Δ Gradient norm = {diff_grad_norm:.2e}     "+pos_E_corr)

				# Convergence criteria: correlation energy change below threshold AND small gradient norm
				if (diff_E_corr < 1e-8 and self.grad_norm < 1e-4) or diff_E_corr < 1e-12: #) or diff_E_corr < 1e-12:  # Convergence threshold
					converged = True
					lst_stop_reason.append("Convergence")

					keep_track += 1
					if keep_track >= keep_track_max + 1:  # Require 25 consecutive iterations below threshold to confirm convergence
						if not self.use_random_unitary_init:
							raw_print(f"OVOS converged with stable correlation energy for {keep_track} consecutive iterations.")

						# Now that we found a sure convergence, we can remove the last 25 points where the energy was still changing but below the threshold, and keep only the final converged point
						lst_E_corr = lst_E_corr[:-keep_track_max]
						lst_iter_counts = lst_iter_counts[:-keep_track_max]
						lst_mo_coeffs = lst_mo_coeffs[:-keep_track_max]
						lst_Fmo_rot = lst_Fmo_rot[:-keep_track_max]
						lst_stop_reason = lst_stop_reason[:-keep_track_max]

						break
				else:
					converged = False
					lst_stop_reason.append("Non-converged")
					
					keep_track = 0  # <--- IMPORTANT: Reset counter when not converged


				# set upper limit of iterations to prevent infinite loop in case of non-convergence
			if iter_count >= max_iter:
				if not self.use_random_unitary_init:
					raw_print(f"Reached upper limit of iterations ({max_iter}) without convergence. Stopping OVOS.")
				break

			

		# Which direction did we go?
		if converged and not self.use_random_unitary_init:
			final_E_corr = lst_E_corr[-1]
			initial_E_corr = lst_E_corr[0]
			raw_print()
			raw_print("#### OVOS Summary ####")
			raw_print(f"Initial MP2 correlation energy: {initial_E_corr:.10f} Hartree")
			raw_print(f"Final OVOS correlation energy: {final_E_corr:.10f} Hartree")
			raw_print(f"Total change in correlation energy: {final_E_corr - initial_E_corr:.10f} Hartree")
			if final_E_corr < initial_E_corr:
				raw_print("OVOS successfully lowered the correlation energy.")
			else:
				raw_print("WARNING: OVOS increased the correlation energy!")
			raw_print(f"Total number of iterations: {iter_count}")

		# Print information about the spaces
		if not self.use_random_unitary_init:
			raw_print()
			raw_print("#### Active and inactive spaces ####")
			raw_print("Total number of spin-orbitals: ", self.tot_num_spin_orbs)
			raw_print("Active occupied spin-orbitals: ", self.active_occ_indices)
			raw_print("Active unoccupied spin-orbitals: ", self.active_inocc_indices)
			raw_print("Inactive unoccupied spin-orbitals: ", self.inactive_indices)
			raw_print()
		
		# Check if OVOS converged
		if not converged:
			raw_print(f"  [{len(self.active_inocc_indices)}/{len(self.active_inocc_indices)+len(self.inactive_indices)}] OVOS did not converge within the maximum number of iterations.")

		# Return the results
		best_converged_idx = None
			# Get the full list of energies where the lst_stop_reason is "Convergence", and sort by energy to find the best converged point
		if lst_stop_reason.count("Convergence") > 0:
			lst_converged_points = [(idx, E) for idx, (E, reason) in enumerate(zip(lst_E_corr, lst_stop_reason)) if reason == "Convergence"]
			best_converged_idx, best_converged_E = min(lst_converged_points, key=lambda x: x[1])
		else:
			# If no converged points, get best energy from all iterations
			best_converged_idx = np.argmin(lst_E_corr)
			
			

				# If we found a converged point, truncate the lists to that point
		if best_converged_idx is not None:
			lst_E_corr = lst_E_corr[:best_converged_idx + 1]
			lst_iter_counts = lst_iter_counts[:best_converged_idx + 1]
			mo_coeffs = lst_mo_coeffs[best_converged_idx]
			Fmo_rot = lst_Fmo_rot[best_converged_idx]
			lst_stop_reason = lst_stop_reason[:best_converged_idx + 1]
		else:
			mo_coeffs = lst_mo_coeffs[-1]
			Fmo_rot = lst_Fmo_rot[-1]
		
		return lst_E_corr, lst_iter_counts, mo_coeffs, Fmo_rot, lst_stop_reason
	
# I want to run OVOS on CH2 w.
	# Article reference - CH2 !!!
	# (9s7p2d If,5s2p) 	-> [2, ..., 116]	-> E(SCF) = -38.89447 | E(UMP2) = ...      ,  E_corr = -0.182 683
	# ------------------------------------------------------------|-------------------------------------------
	# 6-31G 		 	-> [2, ..., 18 ]	-> E(SCF) = -38.84716 | E(UMP2) = -38.91172,  E_corr = -0.064 553
	# cc-pVDZ 		 	-> [2, ..., 40 ]	-> E(SCF) = -38.87219 | E(UMP2) = -38.98252,  E_corr = -0.110 327
	# DZP 		 		-> [2, ..., 42 ]	-> E(SCF) = -38.87497 | E(UMP2) = -38.99727,  E_corr = -0.122 306
	# roosdz 	 		-> [2, ..., 74 ]	-> E(SCF) = -38.88702 | E(UMP2) = -39.01329,  E_corr = -0.126 263
	# def2-QZVPP 		-> [2, ..., 226]	-> E(SCF) = -38.88865 | E(UMP2) = -39.05821,  E_corr = -0.169 552
	# ANO 		 		-> [2, ..., 334]	-> E(SCF) = -38.88763 | E(UMP2) = -39.07982,  E_corr = -0.192 186
	# cc-pV5Z 	 		-> [2, ..., 394]	-> E(SCF) = -38.88888 | E(UMP2) = -39.07190,  E_corr = -0.183 022 <-- !!!
	# aug-cc-pV5Z 		-> [2, ..., 566]	-> E(SCF) = -38.88897 | E(UMP2) = -39.07315,  E_corr = -0.184 171


"""
Setup OVOS calculation
"""
def setup_OVOS(select_atom, select_basis):
	# Custom minimal basis for testing
	# my_basis = {
	# 	'C': [
	# 		['s', (exp1_c_s, coeff1_c_s), (exp2_c_s, coeff2_c_s), ...],  # 9s primitives contracted
	# 		['p', (exp1_c_p, coeff1_c_p), (exp2_c_p, coeff2_c_p), ...],  # 7p primitives contracted
	# 		['d', (exp1_c_d, 1.0), (exp2_c_d, 1.0)],                     # 2d primitives uncontracted
	# 		['f', (exp_c_f, 1.0)]                                        # 1f primitive uncontracted
	# 	],
	# 	'H': [
	# 		['s', (exp1_h_s, coeff1_h_s), (exp2_h_s, coeff2_h_s), ...],  # 5s primitives contracted
	# 		['p', (exp1_h_p, 1.0), (exp2_h_p, 1.0)]                      # 2p primitives uncontracted
    # 	]
	# }

	# Molecule
	atom_choose_between = [
		"H .0 .0 .0; H .0 .0 0.74144",  # H2 bond length 0.74144 Angstrom
		"Li .0 .0 .0; H .0 .0 1.595",   # LiH bond length 1.595 Angstrom
		"O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;",  # H2O equilibrium geometry
		"C  0.0000  0.0000  0.0000; H  0.0000  0.8670  0.5040; H  0.0000 -0.8670  0.5040;", # CH2 
		"B 0 0 0; H 0 0 1.19; H 0 1.03 -0.40; H 0 -1.03 -0.40",  # BH3 equilibrium geometry
		"N 0 0 0; N 0 0 1.10", # N2 bond length 1.10 Angstrom
		"C 0 0 0; O 0 0 1.128", # CO bond length 1.128 Angstrom
		"H 0 0 0; F 0 0 0.917", # HF bond length 0.917 Angstrom
		"N 0 0 0; H 0 0 1.012; H 0 0.935 -0.262; H 0 -0.935 -0.262" # NH3 equilibrium geometry	
		]
	# Basis set
	basis_choose_between = [
		"STO-3G",
		"6-31G",
		"cc-pVDZ",
		"cc-pVTZ",
		"cc-pVQZ"
	]

	find_atom = {
		"H2": 0,
		"LiH": 1,
		"H2O": 2,
		"CH2": 3,
		"BH3": 4,
		"N2": 5,
		"CO": 6,
		"HF": 7,
		"NH3": 8
	}	

	find_basis = {
		"STO-3G": 0,
		"6-31G": 1,
		"cc-pVDZ": 2,
		"cc-pVTZ": 3
	}

	atom, basis = (atom_choose_between[find_atom[select_atom]], select_basis)

	# Print start message
	raw_print(" Running OVOS on ", atom, " with basis set ", basis)
	print("")

	# Get number of electrons and full space size in molecular orbitals
	unit = "angstrom" # angstrom or bohr
		# Initialize molecule and UHF
	mol = pyscf.M(atom=atom,
			    basis=basis,
				unit=unit,
				spin=0,  				  # Closed-shell molecule
				charge=0,				  # symmetry=False 		      # Disable symmetry for OVOS
				symmetry=False
				)
		# Set symmetry
	# mol.symmetry = False  # Disable symmetry for OVOS
	rhf = mol.RHF().run()
		# Number of electrons
	num_electrons = mol.nelec[0] + mol.nelec[1]
		# Full space size in molecular orbitals
	full_space_size = int(rhf.mo_coeff.shape[1])
		# MP2 correlation energy for the full space
	MP2_RHF = rhf.MP2().run()
			# Set reference MP2
	MP2 = MP2_RHF

	return mol, rhf, num_electrons, full_space_size, MP2


"""
Run OVOS for different numbers of optimized virtual orbitals
"""
def get_OVOS_data(num_opt_virtual_orbs_current, retry_count, start_guess, select_atom, select_basis):
	# Start guess:
	if start_guess == "RHF":
		use_RHF_init = True
		use_prev_virt_orbs = False
		use_random_unitary_init = False
	elif start_guess == "prev":
		use_RHF_init = False
		use_prev_virt_orbs = True
		use_random_unitary_init = False
	elif start_guess == "random":
		use_RHF_init = False
		use_prev_virt_orbs = False
		use_random_unitary_init = True

	# Total attempts for each num_opt_virtual_orbs_current
	if select_basis == "6-31G":
		attempts_total = 1000
	elif select_basis == "cc-pVDZ":
		attempts_total = 100

	if use_random_unitary_init == True:
		try_best_of = True
	else:
		try_best_of = False

		# Loop over different numbers of optimized virtual orbitals
	# List of MP2 correlation energies for different numbers of optimized virtual orbitals
	lst_E_corr_virt_orbs = [[],[],[],[],[],[]]  # [[E_corr_list], [num_opt_virtual_orbs_list], [iterations_till_convergence_list], [Unr. SCF check list], [mo_coeffs_final], [lst_stop_reason]]
	lst_MP2_virt_orbs = []  # [(num_opt_virtual_orbs, E_corr, iterations_till_convergence), ...]

	# List of error messages for failed runs
	lst_error_messages = []
	# List of message for priting later
	lst_message = []

	# Retry bounds
	max_retries = 1
	retry_count = 0

	# Set maximum number of optimized virtual orbitals to test
		# Denoted in molecular orbitals (not spin orbitals)
	max_opt_virtual_orbs = full_space_size*2 - num_electrons
		# Set statrting number of optimized virtual orbitals and increment
	num_opt_virtual_orbs_current = 0  # Start with number of occupied orbitals
		# Incremenet by 2 for closed shell molecules
	increment = 2


	# Specific num_opt_virtual_orbs
	# num_opt_virtual_orbs_current_specific = np.array([0.114, 0.171, 0.314, 0.5, 0.657, 1]) * max_opt_virtual_orbs
	# print("Initial specific numbers of optimized virtual orbitals to test (before rounding): ", num_opt_virtual_orbs_current_specific)
	# 	# Int and round to nearest even number
	# num_opt_virtual_orbs_current_specific = [int(round(x / 2) * 2) for x in num_opt_virtual_orbs_current_specific]
	# print("Specific numbers of optimized virtual orbitals to test: ", num_opt_virtual_orbs_current_specific)
	# num_opt_increment = 0

		# For specifc run....
	# num_opt_virtual_orbs_current = 28 - increment  # Start with number of occupied orbitals
	# one_num_opt = True  # If True, only run OVOS for one number of optimized virtual orbitals (num_opt_virtual_orbs_current), otherwise loop through increments until max_opt_virtual_orbs
	# while one_num_opt == True:
	# 	one_num_opt = False

	while num_opt_virtual_orbs_current < max_opt_virtual_orbs:
	# while num_opt_increment < len(num_opt_virtual_orbs_current_specific):
		# Increment num_opt_virtual_orbs until OVOS converges successfully
		num_opt_virtual_orbs_current += increment 
		# num_opt_virtual_orbs_current = num_opt_virtual_orbs_current_specific[num_opt_increment]
		# num_opt_increment += 1

		lst_E_corr = None  # Reset lst_E_corr for each run

		raw_print("")
		raw_print("#### OVOS with ", num_opt_virtual_orbs_current, " out of ", max_opt_virtual_orbs," optimized virtual orbitals (Retry count: ", retry_count,") ####")

		try:

			if try_best_of == False: # Single run of OVOS

				# Get RHF orbitals
				if use_RHF_init == True:
					if use_prev_virt_orbs == False or use_random_unitary_init == False and 'mo_coeffs' not in locals() and 'Fmo_rot' not in locals():
						# Only get RHF orbitals if not using previous or random init
							# The if state will prevent re-getting RHF orbitals if using previous or random init
						mo_coeffs = np.array([rhf.mo_coeff, rhf.mo_coeff]) # Start with RHF orbitals for both alpha and beta spins
						Fmo_rot = None # Fock matrix in the MO basis will be constructed later based on the initial orbitals
						Fao_get = rhf.get_fock() # Get Fock matrix in AO basis for later use in orbital optimization
						Fao = np.array([Fao_get, Fao_get]) # Use the same Fock matrix for both spins as starting point
					init_orbs = "RHF"

				if use_prev_virt_orbs == True:
					if 'mo_coeffs' not in locals() and 'Fmo_rot' not in locals():
						mo_coeffs = np.array([rhf.mo_coeff, rhf.mo_coeff]) # Start with RHF orbitals for both alpha and beta spins
						Fmo_rot = None # Fock matrix in the MO basis will be constructed later based on the initial orbitals
						Fao_get = rhf.get_fock() # Get Fock matrix in AO basis for later use in orbital optimization
						Fao = np.array([Fao_get, Fao_get]) # Use the same Fock matrix for both spins as starting point
					
					# Use previously optimized orbitals as starting guess
					mo_coeffs = mo_coeffs.copy() 
					Fmo_rot = Fmo_rot  
					init_orbs = "RHF"

					raw_print("    Using previously optimized orbitals as starting guess.")				

					# Run OVOS
				lst_E_corr, lst_iter_counts, mo_coeffs, Fmo_rot, lst_stop_reason = OVOS(mol=mol, scf=rhf, Fao=Fao, num_opt_virtual_orbs=num_opt_virtual_orbs_current, init_orbs=init_orbs, mo_coeff=mo_coeffs).run_ovos(mo_coeffs=mo_coeffs, Fmo_rot=Fmo_rot)

			if try_best_of == True: # Multiple runs of OVOS with different random initializations
				raw_print("Using random unitary rotated RHF virtual orbitals as starting guess.")

				# Try multiple random initializations and pick the best result
				attempt = 0
				best_E_corr = None
				best_lst_E_corr = None
				best_lst_iter_counts = None
				best_mo_coeffs = None

				eri_4fold_ao = mol.intor('int2e_sph', aosym=1)
				S = mol.intor('int1e_ovlp')
				hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

				# Keep track if we ended up at the same best result multiple times, which could indicate a local minimum
				best_result_count = 0

				# If at full space update the attempts_total, as we can not optimize here
				if num_opt_virtual_orbs_current == max_opt_virtual_orbs:
					# Scale up attempts_total
					attempts_total *= 10
					raw_print(f"Optimizing in the full virtual space, increasing total attempts to {attempts_total} to ensure we find the best result.")

				try:
					while attempt < attempts_total:
						attempt += 1
						raw_print("")
						raw_print("---- Attempt ", attempt, " of ", attempts_total, " ----")

						# Get new random unitary rotation for virtual orbitals
							# Get RHF orbitals
						mo_coeffs_rhf = np.array([rhf.mo_coeff, rhf.mo_coeff]) # Start with RHF orbitals for both alpha and beta spins
						Fao_rhf = np.array([rhf.get_fock(), rhf.get_fock()]) # Get Fock matrix in AO basis for later use in orbital optimization

						# Apply random unitary rotation to virtual orbitals only
							# Number of occupied orbitals
						num_occupied_orbitals = num_electrons // 2
							# Total number of spatial orbitals
						total_spatial_orbitals = mo_coeffs_rhf[0].shape[1]
							# Number of virtual orbitals
						num_virtual_orbitals = total_spatial_orbitals - num_occupied_orbitals

							# Generate random unitary matrix for virtual orbitals
						rand_matrix = np.random.rand(num_virtual_orbitals, num_virtual_orbitals)
						Q, R = np.linalg.qr(rand_matrix)  # QR decomposition to get unitary matrix

							# Rotate virtual orbitals for alpha and beta spins
						mo_coeffs = [np.copy(mo_coeffs_rhf[0]), np.copy(mo_coeffs_rhf[1])]  # Deep copy to avoid modifying original
						for spin in [0, 1]:
							# Extract occupied and virtual parts
							C_occ = mo_coeffs_rhf[spin][:, :num_occupied_orbitals]
							C_virt = mo_coeffs_rhf[spin][:, num_occupied_orbitals:]

							# Rotate virtual orbitals
							C_virt_rot = C_virt @ Q

							# Combine back
							mo_coeffs[spin] = np.hstack((C_occ, C_virt_rot))
						
							# Note: The occupied-occupied and occupied-virtual blocks of the Fock matrix remain unchanged, only the virtual-virtual block is rotated.

						mo_coeffs = np.array(mo_coeffs)
						Fmo_rot = None # Fock matrix in the MO basis will be constructed later based on the initial orbitals
						Fao = Fao_rhf  # Use the same Fock matrix for both spins as starting point
						init_orbs = "RHF"

						# Run OVOS
						ovos_obj = OVOS(mol=mol, scf=rhf, Fao=Fao, num_opt_virtual_orbs=num_opt_virtual_orbs_current, init_orbs=init_orbs, mo_coeff=mo_coeffs, start_guess=start_guess)
							# Pass, type: ignore
						ovos_obj.eri_4fold_ao = eri_4fold_ao  # Pass the 4-fold AO integrals to avoid recomputation
						ovos_obj.S = S  # Pass the overlap matrix
						ovos_obj.hcore_ao = hcore_ao  # Pass the core Hamiltonian in AO basis

						# Run OVOS with the current random unitary rotated orbitals
						lst_E_corr_attempt, lst_iter_counts_attempt, mo_coeffs_attempt, Fmo_rot_attempts, lst_stop_reason_attempts = ovos_obj.run_ovos(mo_coeffs=mo_coeffs, Fmo_rot=Fmo_rot)

						# --- Evaluate this attempt ---
						E_this = lst_E_corr_attempt[-1]
							# Print the result of this attempt
						raw_print(f"    [{num_opt_virtual_orbs_current}/{max_opt_virtual_orbs}] MP2: {E_this} @ {len(lst_E_corr_attempt)} iterations")
						
						# Discard this attempt if it did not converge (or falsely converged...)
							# False converged if the last five are not converged points
						# if lst_stop_reason_attempts[-5:].count("Convergence") < 5 and len(lst_stop_reason_attempts) > 5:
						# 	print(f"Attempt {attempt} did not show stable convergence in the last 5 iterations. Discarding this attempt.")
						# 	best_result_count = 0  # Reset count for this new best result
						# 	continue

						# Update best result
						if best_E_corr is None or E_this < best_E_corr:
							best_E_corr = E_this
							best_lst_E_corr = lst_E_corr_attempt
							best_lst_iter_counts = lst_iter_counts_attempt
							best_mo_coeffs = mo_coeffs_attempt
							best_Fmo_rot = Fmo_rot_attempts
							best_lst_stop_reason = lst_stop_reason_attempts
							best_result_count = 0  # Reset count for this new best result
							raw_print(f"    New best result found: {best_E_corr} Hartree with {num_opt_virtual_orbs_current} optimized virtual orbitals at attempt {attempt}.")

						# Check if we got the same best result again, which could indicate a local minimum
						if 1 < attempt <= 125:  # Only start checking after the first attempt
							if best_E_corr is not None and abs(E_this - best_E_corr) < 1e-4:
								best_result_count += 1
						# After 125 attempts, we can loosen the criterion for being the same best result
						if attempt > 125:  # Only start checking after the first attempt
							if abs(E_this - best_E_corr) < 1e-3:
								best_result_count += 1
						# After 250 attempts, we can loosen the criterion even more
						elif attempt > 250:
							if abs(E_this - best_E_corr) < 1e-2:
								best_result_count += 1
						# After 500 attempts, we can loosen the criterion even more
						elif attempt > 500:
							if abs(E_this - best_E_corr) < 1e-1:
								best_result_count += 1

						if best_result_count > 25 and attempt > 5:  # If we got the same best result more than 25 times, we can stop trying more random initializations
							raw_print(f"Got the same best correlation energy {best_E_corr} for {best_result_count-1} attempts (with further loosened criterion), which could indicate a local minimum. Stopping further attempts.")
							break
				except BreakOuterLoop:
					pass

				# Use the best result from all attempts
				lst_E_corr = best_lst_E_corr
				lst_iter_counts = best_lst_iter_counts
				mo_coeffs = best_mo_coeffs
				Fmo_rot = best_Fmo_rot
				lst_stop_reason = best_lst_stop_reason

			# Check alpha/beta are the same for a tolerance - Done after orbital optimization
			diff_alpha_beta = np.max(np.abs(mo_coeffs[0] - mo_coeffs[1]))
			if diff_alpha_beta > 1e-4 and not use_random_unitary_init:
				raw_print("Warning: OVOS with ", num_opt_virtual_orbs_current, " optimized vorbs resulted in different alpha and beta orbitals (max diff: ", diff_alpha_beta, ").")
					# Store message
				lst_message.append(f"OVOS w. {num_opt_virtual_orbs_current} optimized vorbs resulted in different alpha and beta orbitals (max diff: {diff_alpha_beta}). Here largest alpha {np.max(mo_coeffs[0])} and beta {np.max(mo_coeffs[1])} orbital coeffs.")
					# Append True-False flag to lst_E_corr_virt_orbs
				lst_E_corr_virt_orbs[3].append("True")
			else:
				lst_E_corr_virt_orbs[3].append("False")

			# run_OVOS converged to a positive MP2 correlation energy
			if lst_E_corr[-1] > 0 and not use_random_unitary_init:
				raw_print("Warning: OVOS with ", num_opt_virtual_orbs_current, " optimized virtual orbitals converged to a positive MP2 correlation energy.")

			# Store results
			lst_MP2_virt_orbs.append((num_opt_virtual_orbs_current, lst_E_corr[-1], lst_iter_counts[-1], lst_stop_reason[-1]))
			lst_E_corr_virt_orbs[0].append(lst_E_corr)
			lst_E_corr_virt_orbs[1].append(num_opt_virtual_orbs_current)
			lst_E_corr_virt_orbs[2].append(lst_iter_counts)
			# ...
			lst_E_corr_virt_orbs[4].append(mo_coeffs.tolist())
			lst_E_corr_virt_orbs[5].append(lst_stop_reason)

			# Reset retry count on success
			retry_count = 0
				

		# Catch errors during OVOS
		except AssertionError as e:
			raw_print(f"Error during OVOS with {num_opt_virtual_orbs_current} optimized virtual orbitals: {e}")
			raw_print("Rerunning with the same number of virtual orbitals.")

			# Add error message to list

				# Get results if available
			lst_error_messages.append((num_opt_virtual_orbs_current, str(e), lst_iter_counts[-1] if lst_iter_counts is not None else 0))

			retry_count += 1
			if retry_count >= max_retries:
				raw_print(f"Maximum retries reached for {num_opt_virtual_orbs_current} optimized virtual orbitals. Skipping to next.")
				retry_count = 0
				continue

			num_opt_virtual_orbs_current -= increment  # Decrement to retry the same number
			continue



	# Print summary of the run
	raw_print("Number of electrons: ", num_electrons)
	raw_print("Full space size in molecular orbitals: ", full_space_size)
	raw_print("Maximum number of optimized virtual orbitals tested: ", max_opt_virtual_orbs)
	raw_print("Total OVOS runs completed: ", len(lst_MP2_virt_orbs))
	raw_print("")

	# Print the final MP2 correlation energy after all OVOS and amount of iterations till convergence
	for num_opt_virtual_orbs_current, E_corr, iter_, lst_stop_reason in lst_MP2_virt_orbs:
		raw_print("MP2 correlation energy, for ", num_opt_virtual_orbs_current, f" optimized virtual orbitals: ", '%.5E' % Decimal(E_corr),f" ({(E_corr/MP2.e_corr)*100:.4}%)"+" @ ", iter_, " iterations till convergence (",lst_stop_reason,")")
	raw_print("MP2 correlation energy, for full space: ", '%.5E' % Decimal(MP2.e_corr), "| Difference:", '%.5E' % Decimal(MP2.e_corr - lst_MP2_virt_orbs[-1][1]))
	raw_print("")

	# Print if the check of alpha and beta orbitals were the same
	for msg in lst_message:
		raw_print(msg)
	raw_print("")
	

	# Print what methods were used
	if use_prev_virt_orbs == True:
		raw_print("Previously optimized virtual orbitals were used as starting guess for each OVOS run.")
	if use_random_unitary_init == True:
		raw_print("Random unitary rotations of UHF virtual orbitals were used as starting guess for each OVOS run.")
	raw_print("")

	# Print error messages summary
	if len(lst_error_messages) > 0:
		raw_print("#### Error messages summary ####")
		for num_opt_virtual_orbs_current, error_msg, iter_ in lst_error_messages:
			raw_print("  OVOS w. ", num_opt_virtual_orbs_current, " optimized vorbs failed at iteration ", iter_ ," w. error: ", error_msg)
		raw_print("")


	# Save data to JSON files
	import json

	str_name = ""

	if use_prev_virt_orbs == True:
		str_name = "different_virt_orbs_prev" # !!!!
	if use_random_unitary_init == True:
		str_name = "different_virt_orbs_random" # !!!!
	if use_prev_virt_orbs == False and use_random_unitary_init == False:
		str_name = "different_virt_orbs" # !!!!

	if use_RHF_init == True:
		str_name += "_RHF_init" # !!!!

	str_atom = select_atom
	str_basis = select_basis

	raw_print("Saving data to branch/data/"+str_atom+"/"+str_basis+"/...")

	# Save MP2 correlation energy convergence data
	with open("branch/data/"+str_atom+"/"+str_basis+"/lst_MP2_"+str_name+".json", "w") as f:
		json.dump(lst_E_corr_virt_orbs, f, indent=2)

	raw_print("Data saved to branch/data/"+str_atom+"/"+str_basis+"/lst_MP2_"+str_name+".json")


"""
Get data
"""

import sys

class Tee:
    """Write to multiple streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

if False: # Done with OVOS runs. Comment out this line to run more or move on to the next part of the code.
	# Save original terminal settings
	old_settings = termios.tcgetattr(sys.stdin)
	try:
		# Put terminal in raw mode (so we get keys instantly)
		tty.setraw(sys.stdin.fileno())
		
		for basis_set in ["6-31G"]: 				#   	"6-31G" | Yet: "cc-pVDZ", ... 
			for molecule in ["H2O", "CO", "HF", "NH3"]: 				#       "H2O", "CO", "HF", "NH3"
													#        (16)  (22)  (12)  (20)
					
				for start_guess in ["random"]: # "RHF", "prev", "random" | Yet: "random"
						# Create a log file for this molecule and basis set
					with open(f"branch/data/{molecule}/{basis_set}/OVOS_{molecule}_{basis_set}_"+start_guess+"_output.txt", "w") as f:
						sys.stdout = Tee(sys.__stdout__, f)
						try:
							raw_print("")
							raw_print("==========================================================")
							raw_print("Running OVOS for molecule: ", molecule, " with basis set: ", basis_set)
							raw_print("==========================================================")
							raw_print("")

							mol, rhf, num_electrons, full_space_size, MP2 = setup_OVOS(molecule, basis_set)
							get_OVOS_data(num_opt_virtual_orbs_current=0, retry_count=0, start_guess=start_guess, select_atom=molecule, select_basis=basis_set)
						finally:
							sys.stdout = sys.__stdout__
	finally:
		# Restore terminal settings no matter what
		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

"""
The oscillations are a property of the Newton-Raphson method applied to a non-quadratic functional. 
The Hessian gives a quadratic approximation that can overshoot when far from the minimum, causing the energy to temporarily increase. 
This is normal and expected behavior for Newton's method without line search or trust region damping.

Negative eigenvalues in Hessian can also be level-shifted to stabilize convergence, but this can slow down progress.
"""		
if False: # Done with dissociation curve.
	# Save results to JSON
	import json

	class NumpyEncoder(json.JSONEncoder):
		"""Custom JSON encoder that handles numpy scalars, arrays, and None."""
		def default(self, obj):
			if isinstance(obj, np.integer):
				return int(obj)
			if isinstance(obj, np.floating):
				return float(obj)
			if isinstance(obj, np.ndarray):
				return obj.tolist()
			return super().default(obj)

	def to_python_scalar(val):
		"""Recursively convert numpy scalars/arrays to native Python types."""
		if val is None:
			return None
		if isinstance(val, np.ndarray):
			return val.tolist()
		if isinstance(val, (np.integer,)):
			return int(val)
		if isinstance(val, (np.floating,)):
			return float(val)
		if isinstance(val, list):
			return [to_python_scalar(v) for v in val]
		if isinstance(val, tuple):
			return [to_python_scalar(v) for v in val]
		return val

	# OOMP2 implementation based on PySCF's MP2, which is based on the original OOMP2 paper 
	# by Lee and Head-Gordon (https://pubs.acs.org/doi/10.1021/acs.jpca.5b07881) and the 
	# recent developments in PySCF (https://pubs.aip.org/aip/jcp/article/153/2/024109/1061482)
	# OOMP2: Orbital-Optimized MP2, where the orbitals are optimized to minimize the MP2 energy.

	class ROOMP2(object):
		"""
		Restricted OOMP2 solver.
		Used as fcisolver in pyscf.mcscf.CASSCF(rhf, ...).
		
		CASSCF passes h1, h2 already in the MO basis.
		We build a fake RHF object and run RMP2 on it.
		This always stays restricted — cannot break spin symmetry.
		"""
		def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
			if isinstance(nelec, (int, np.integer)):
				na = nelec // 2
				nb = nelec // 2
				
			else:
				na, nb = int(nelec[0]), int(nelec[1])
			
			n_elec = na + nb

			fakemol = pyscf.gto.M(verbose=0)
			fakemol.nelectron = n_elec

			fake_hf = pyscf.scf.RHF(fakemol)
			fake_hf._eri = h2
			fake_hf.get_hcore = lambda *args: h1
			fake_hf.get_ovlp = lambda *args: np.eye(norb)

			fake_hf.mo_coeff = np.eye(norb)
			fake_hf.mo_occ = np.zeros(norb)
			fake_hf.mo_occ[:n_elec // 2] = 2

			self.mp2 = pyscf.mp.MP2(fake_hf)
			self.mp2.verbose = 0

			e_corr, t2 = self.mp2.kernel()
			e_tot = self.mp2.e_tot + ecore
			return e_tot, t2

		def make_rdm12(self, t2, norb, nelec):
			dm1 = self.mp2.make_rdm1(t2)
			dm2 = self.mp2.make_rdm2(t2)
			if isinstance(dm2, (tuple, list)):
				dm2 = sum(dm2)
			return dm1, dm2


	class UOOMP2(object):
		"""
		Unrestricted OOMP2 solver.
		Used as fcisolver in pyscf.mcscf.UCASSCF(uhf, ...).

		UCASSCF passes:
			h1  = (h1_alpha, h1_beta)          — 1e integrals in MO basis, per spin
			h2  = (h2_aa, h2_ab, h2_bb)        — 2e integrals in MO basis, per spin block
			nelec = (nalpha, nbeta)
		"""
		def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
			if isinstance(nelec, (int, np.integer)):
				na = nelec // 2
				nb = nelec // 2
			else:
				na, nb = int(nelec[0]), int(nelec[1])

			n_elec = na + nb

			h1a, h1b = h1
			h2aa, h2ab, h2bb = h2

			fakemol = pyscf.gto.M(verbose=0)
			fakemol.nelectron = n_elec
			fakemol.spin = na - nb + 1  	# Just needs to be nonzero to trigger UHF, but we won't actually use the orbitals from this SCF object
			fakemol.symmetry = False  		# Disable symmetry

			fake_uhf = pyscf.scf.UHF(fakemol)
			fake_uhf.get_hcore = lambda *args: (h1a + h1b) * 0.5
			fake_uhf.get_ovlp  = lambda *args: np.eye(norb)
			fake_uhf.mo_coeff  = np.array([np.eye(norb), np.eye(norb)])
			fake_uhf.mo_occ    = np.zeros((2, norb))
			fake_uhf.mo_occ[0, :na] = 1
			fake_uhf.mo_occ[1, :nb] = 1
			fake_uhf.mo_energy = np.array([np.diag(h1a), np.diag(h1b)])
			fake_uhf._eri      = pyscf.ao2mo.restore(4, h2aa, norb)

			# Compute the HF energy from the MO integrals passed by UCASSCF.
			# UCASSCF passes h1, h2 already in the MO basis with the current orbitals.
			# The HF energy = sum of occupied orbital energies (1e part)
			#               + 2e Coulomb/exchange from occupied-occupied block.
			# Simplest: use the trace of h1 over occupied orbitals (1e contribution)
			# plus the 2e contribution from the occupied-occupied block of h2.
			#
			# For UHF:
			#   E_HF = sum_i^{occ_a} h1a[i,i]
			#        + sum_i^{occ_b} h1b[i,i]
			#        + 0.5 * sum_{ij}^{occ_a} (h2aa[i,i,j,j] - h2aa[i,j,j,i])
			#        + 0.5 * sum_{ij}^{occ_b} (h2bb[i,i,j,j] - h2bb[i,j,j,i])
			#        + sum_{i}^{occ_a} sum_{j}^{occ_b} h2ab[i,i,j,j]
			oa = slice(0, na)
			ob = slice(0, nb)

			e_1e  = np.einsum('ii->', h1a[oa, oa]) + np.einsum('ii->', h1b[ob, ob])

			e_2e_aa = 0.5 * (  np.einsum('iijj->', h2aa[oa, oa, oa, oa])
								- np.einsum('ijji->', h2aa[oa, oa, oa, oa]))
			e_2e_bb = 0.5 * (  np.einsum('iijj->', h2bb[ob, ob, ob, ob])
								- np.einsum('ijji->', h2bb[ob, ob, ob, ob]))
			e_2e_ab = np.einsum('iijj->', h2ab[oa, oa, ob, ob])

			e_hf = e_1e + e_2e_aa + e_2e_bb + e_2e_ab

			self.mp2 = pyscf.mp.UMP2(fake_uhf)
			self.mp2.verbose = 0

			nocc_a, nocc_b = na, nb
			nvir_a = norb - nocc_a
			nvir_b = norb - nocc_b

			class _ERIS:
				pass

			eris = _ERIS()
			eris.mo_energy = [np.diag(h1a), np.diag(h1b)]

			va = slice(nocc_a, norb)
			vb = slice(nocc_b, norb)

			eris.ovov = h2aa[oa, va, oa, va].reshape(nocc_a*nvir_a, nocc_a*nvir_a)
			eris.ovOV = h2ab[oa, va, ob, vb].reshape(nocc_a*nvir_a, nocc_b*nvir_b)
			eris.OVOV = h2bb[ob, vb, ob, vb].reshape(nocc_b*nvir_b, nocc_b*nvir_b)

			e_corr, t2 = pyscf.mp.ump2.kernel(
				self.mp2,
				mo_energy=eris.mo_energy,
				mo_coeff=fake_uhf.mo_coeff,
				eris=eris
			)

			self.mp2.e_corr = e_corr
			self.t2 = t2

			# e_tot = HF energy (from h1/h2 in MO basis)
			#       + MP2 correlation energy
			#       + ecore (nuclear repulsion, passed by UCASSCF)
			e_tot = e_corr + e_hf + ecore
			return e_tot, t2

		def make_rdm1s(self, t2, norb, nelec):
			"""
			Return SEPARATE (dm1a, dm1b) — one 2D array per spin.
			UCASSCF.gen_g_hop calls fcasdm1s() which dispatches here.
			Shape of each: (norb, norb).
			"""
			dm1 = self.mp2.make_rdm1(t2)
			if isinstance(dm1, (tuple, list)) and len(dm1) == 2:
				return dm1[0], dm1[1]
			return dm1 * 0.5, dm1 * 0.5

		def make_rdm1(self, t2, norb, nelec):
			dm1a, dm1b = self.make_rdm1s(t2, norb, nelec)
			return dm1a + dm1b

		def make_rdm12s(self, t2, norb, nelec):
			"""
			Return ((dm1a, dm1b), (dm2aa, dm2ab, dm2bb)) for UCASSCF.

			UCASSCF.gen_g_hop unpacks as:
				casdm1s = dm1s        -> (dm1a, dm1b), each shape (ncas, ncas)
				casdm2s = dm2s        -> (dm2aa, dm2ab, dm2bb), each shape (ncas,)*4

			pyscf.mp.UMP2.make_rdm2 returns a tuple of 4 arrays:
				(d2aa, d2ab, d2ba, d2bb)
			We need to combine d2ab + d2ba into a single d2ab = (d2ab + d2ba) / 2
			to match the (aa, ab, bb) triplet UCASSCF expects.
			"""
			dm1s = self.make_rdm1s(t2, norb, nelec)  # (dm1a, dm1b)

			dm2_raw = self.mp2.make_rdm2(t2)

			if isinstance(dm2_raw, (tuple, list)) and len(dm2_raw) == 4:
				# UMP2 returns (d2aa, d2ab, d2ba, d2bb)
				d2aa, d2ab, d2ba, d2bb = dm2_raw
				# UCASSCF expects 3-tuple: (d2aa, d2ab, d2bb)
				# where d2ab represents the alpha-beta block
				d2ab_sym = (d2ab + d2ba.transpose(2, 3, 0, 1)) * 0.5
				dm2s = (d2aa, d2ab_sym, d2bb)
			elif isinstance(dm2_raw, (tuple, list)) and len(dm2_raw) == 3:
				# Already in (d2aa, d2ab, d2bb) form
				dm2s = dm2_raw
			else:
				# Fallback: spin-summed 2-RDM, split equally
				dm2s = (dm2_raw * 0.25, dm2_raw * 0.5, dm2_raw * 0.25)

			return dm1s, dm2s



		# The parallelized version of the above code...
	from multiprocessing import Pool
	import functools

	def compute_geometry(geom, distances, mol_setup_params, ovos_params, worker_id=None, existing_data=None):
		"""
		Worker function: compute OVOS for a single geometry.
		Runs in a separate process.
		"""

		# Identify geometry index based on distance
		geom_dist = float(geom.split()[-1])
		geom_idx = np.where(np.isclose(distances, geom_dist))[0][0] 
		geom_idx_int = int(geom_idx)  # Convert to integer index for printing

		# # for debugging only do the first geometry
		# if geom_idx_int > 0:
		# 	return None

			# Skip the computations already computed
		existing_data = existing_data[geom_idx_int] if existing_data is not None else None
		rhf_e_tot = existing_data[1][0] if existing_data is not None else None
		uhf_e_tot = existing_data[1][1] if existing_data is not None else None
		MP2_e_corr_rhf = existing_data[2][0] if existing_data is not None else None
		MP2_e_corr_uhf = existing_data[2][1] if existing_data is not None else None
		ooMP2_e_tot_rhf = existing_data[3][0] if existing_data is not None else None
		ooMP2_e_tot_uhf = existing_data[3][1] if existing_data is not None else None
		E_corr_rhf_50 = existing_data[4][0] if existing_data is not None else None
		# E_corr_rhf_75 = existing_data[4][1] if existing_data is not None else None
		# E_corr_rhf_90 = existing_data[4][2] if existing_data is not None else None
			
			# Force recomputation 
		E_corr_rhf_75 = None
		E_corr_rhf_90 = None

		# Reconstruct molecule from geometry string
		mol = pyscf.M(
			atom=geom,
			basis=mol_setup_params['basis'],
			unit="angstrom",
			verbose=0
		)
		
		# RHF
		rhf = pyscf.scf.RHF(mol)
		rhf.verbose = 0
		rhf_e_tot = rhf.kernel()
		
		# UHF 
			# with custom guess based on placing atomic densities on the two Li atoms
		# 1. Run UHF on a single Li atom (doublet)
		atom_li = pyscf.gto.M(
			atom='Li 0 0 0',
			basis=mol_setup_params['basis'],   # <-- use molecular basis
			spin=1,
			verbose=0,
			symmetry=False
		)
		mf_li = pyscf.scf.UHF(atom_li)
		mf_li.kernel()
		dm_li_alpha, dm_li_beta = mf_li.make_rdm1()   # atomic alpha and beta densities

		# 2. Build the molecular density matrices by placing atomic blocks
		per_ao = mol.nao_nr() // mol.natm          # should be 14 for cc-pVDZ
		idx0 = slice(0, per_ao)                     # first atom AOs
		idx1 = slice(per_ao, 2*per_ao)              # second atom AOs

			# Build density matrices
		n_ao = mol.nao_nr()
		dm_alpha = np.zeros((n_ao, n_ao))
		dm_beta  = np.zeros((n_ao, n_ao))

		# Atom1 (first Li) gets alpha; Atom2 gets beta  (for alpha DM)
		dm_alpha[idx0, idx0] = dm_li_alpha
		dm_alpha[idx1, idx1] = dm_li_beta

		# Atom1 gets beta; Atom2 gets alpha  (for beta DM)
		dm_beta[idx0, idx0] = dm_li_beta
		dm_beta[idx1, idx1] = dm_li_alpha

		# 3. Run UHF with this guess
		uhf = pyscf.scf.UHF(mol)
		uhf.verbose = 0
		uhf_e_tot = uhf.kernel(dm0=(dm_alpha, dm_beta))

		# MP2
		if MP2_e_corr_rhf is None or MP2_e_corr_uhf is None:
			MP2_e_corr_rhf = pyscf.mp.RMP2(rhf).kernel()[0] + rhf_e_tot if MP2_e_corr_rhf is None else MP2_e_corr_rhf
			MP2_e_corr_uhf = pyscf.mp.UMP2(uhf).kernel()[0] + uhf_e_tot if MP2_e_corr_uhf is None else MP2_e_corr_uhf

		# OOMP2
		# Two separate calculations:
		# 1) Restricted OOMP2 via CASSCF(rhf) + ROOMP2  — always stays restricted
		# 2) Unrestricted OOMP2 via UCASSCF(uhf) + UOOMP2 — can break spin symmetry

		if ooMP2_e_tot_rhf is None:
			# === Restricted OOMP2 ===
			# Uses CASSCF with one set of spatial MOs. Cannot break symmetry.
			# At equilibrium this is the correct OOMP2 energy.
			nmo = rhf.mo_coeff.shape[1]
			mc_rhf = pyscf.mcscf.CASSCF(rhf, nmo, mol.nelectron)
			mc_rhf.verbose = 0
			mc_rhf.fcisolver = ROOMP2()
			mc_rhf.conv_tol = 1e-6
				# Enable internal rotation optimization to allow the orbitals to rotate and break symmetry if it lowers the energy, even though the overall method is still restricted.
					# This allows the method to find a lower-energy solution if it exists, even though the orbitals themselves will still be constrained to be the same for alpha and beta spins.
			mc_rhf.internal_rotation = True
			# mo_coeff_avg = np.array([(uhf.mo_coeff[0] + uhf.mo_coeff[1])//2])[0]  # CASSCF expects a list of MO coeffs per spin, even for restricted
			ooMP2_e_tot_rhf = mc_rhf.kernel()[0] 


		if ooMP2_e_tot_uhf is None:
			# === Unrestricted OOMP2 ===
			# Uses UCASSCF with separate alpha/beta MO sets.
			# Starts from UHF orbitals which are already broken-symmetry at dissociation.
			# The UMP2 energy functional allows alpha != beta → symmetry breaking.
			nmo_uhf = uhf.mo_coeff[0].shape[1]
			mc_uhf = pyscf.mcscf.UCASSCF(uhf, nmo_uhf, (mol.nelec[0], mol.nelec[1]))
			mc_uhf.verbose = 0
			mc_uhf.fcisolver = UOOMP2()
			ooMP2_e_tot_uhf = mc_uhf.kernel()[0]
			
			# 	# After UHF convergence, print spin contamination diagnostic
			# raw_print(f"UHF <S²> = {uhf.spin_square()[0]:.4f} (expected 0.0 for singlet, ~0.75 for dissociated)")
			# raw_print(f"UHF converged: {uhf.converged}")

		# s2 = uhf.spin_square()[0]
		# threshold_s2 = 0.01  # Adjust as needed to detect significant spin contamination
		# if s2 - 0.75 > threshold_s2:
		# 	ooMP2_e_tot_uhf = ooMP2_e_tot_uhf
		# else:
		# 	ooMP2_e_tot_uhf = ooMP2_e_tot_rhf

		# OVOS @ 50% of full space
		num_electrons = mol.nelec[0] + mol.nelec[1]
		full_space_size = int(rhf.mo_coeff.shape[1])
		max_opt_virtual_orbs = full_space_size * 2 - num_electrons
		num_opt_virtual_orbs_current_50 = int(round(0.50 * max_opt_virtual_orbs / 2) * 2)
		num_opt_virtual_orbs_current_75 = int(round(0.75 * max_opt_virtual_orbs / 2) * 2)
		num_opt_virtual_orbs_current_90 = int(round(0.90 * max_opt_virtual_orbs / 2) * 2)
			# Setup initial guess for OVOS
		Fmo_rot = None
				# RHF
		mo_coeffs_rhf = np.array([rhf.mo_coeff, rhf.mo_coeff])
		Fao_get = rhf.get_fock()
		Fao_rhf = np.array([Fao_get, Fao_get])

		import time
			# Index of geometry being processed (for debugging)
				# Get index from last number in str geom, which is in the format "Li .0 .0 .0: Li .0 .0 {distance}"
		# geom_idx += 1
		# time.sleep(5*geom_idx)  # Simulate some delay to allow print statements from different processes to interleave less

		# Print progress
		raw_print("")
		raw_print("==========================================================")
		raw_print(f"Processing geometry: {geom} (Distance: {geom_dist:.2f} Å, Index: {geom_idx}/{len(distances)})")
		raw_print(f"RHF total energy: {rhf_e_tot:.4f} [Spin cont.: {rhf.spin_square()[0]:.4f}]")
		raw_print(f"UHF total energy: {uhf_e_tot:.4f} [Spin cont.: {uhf.spin_square()[0]:.4f}]")
		raw_print(f"MP2 correlation energy:   {MP2_e_corr_rhf:.4f}, {MP2_e_corr_uhf:.4f}")
		raw_print(f"OOMP2 correlation energy: {ooMP2_e_tot_rhf:.4f}, {ooMP2_e_tot_uhf:.4f}")
		raw_print(f"Running OVOS with {num_opt_virtual_orbs_current_50} and {num_opt_virtual_orbs_current_75} optimized virtual orbitals...")

		# geom_idx_inv = len(distances) - geom_idx + 1
		# time.sleep(5*geom_idx_inv)  # Simulate some delay to allow print statements from different processes to interleave less

		# # Suppress output from worker processes to avoid cluttering the terminal
		# from io import StringIO
		# old_stdout = sys.stdout
		# sys.stdout = StringIO()

		try:
				# Run OVOS 
					# with RHF orbitals as starting guess
			try:
				E_corr_rhf_50 = OVOS(
					mol=mol,
					scf=rhf,
					Fao=Fao_rhf,
					num_opt_virtual_orbs=num_opt_virtual_orbs_current_50,
					init_orbs="RHF",
					mo_coeff=mo_coeffs_rhf
				).run_ovos(mo_coeffs=mo_coeffs_rhf, Fmo_rot=Fmo_rot)[0][-1] + rhf_e_tot if E_corr_rhf_50 is None else E_corr_rhf_50
			except Exception as e:
				E_corr_rhf_50 = None

			try:
				E_corr_rhf_75 = OVOS(
					mol=mol,
					scf=rhf,
					Fao=Fao_rhf,
					num_opt_virtual_orbs=num_opt_virtual_orbs_current_75,
					init_orbs="RHF",
					mo_coeff=mo_coeffs_rhf
				).run_ovos(mo_coeffs=mo_coeffs_rhf, Fmo_rot=Fmo_rot)[0][-1] + rhf_e_tot if E_corr_rhf_75 is None else E_corr_rhf_75
			except Exception as e:
				E_corr_rhf_75 = None

			try:
				E_corr_rhf_90 = OVOS(
					mol=mol,
					scf=rhf,
					Fao=Fao_rhf,
					num_opt_virtual_orbs=num_opt_virtual_orbs_current_90,
					init_orbs="RHF",
					mo_coeff=mo_coeffs_rhf
				).run_ovos(mo_coeffs=mo_coeffs_rhf, Fmo_rot=Fmo_rot)[0][-1] + rhf_e_tot if E_corr_rhf_90 is None else E_corr_rhf_90
			except Exception as e:
				E_corr_rhf_90 = None

					# Store results		
			return {
				'geom': geom,
				'e_tot': [rhf_e_tot, uhf_e_tot],
				'MP2_e_corr': [MP2_e_corr_rhf, MP2_e_corr_uhf],
				'ooMP2_e_corr': [ooMP2_e_tot_rhf, ooMP2_e_tot_uhf],
				'OVOS_e_corr': [E_corr_rhf_50, E_corr_rhf_75, E_corr_rhf_90],
				'success': True,
				'error': None
			}

		except Exception as e:
			return {
				'geom': geom,
				'e_tot': [rhf_e_tot, uhf_e_tot],
				'MP2_e_corr': [MP2_e_corr_rhf, MP2_e_corr_uhf],
				'ooMP2_e_corr': [ooMP2_e_tot_rhf, ooMP2_e_tot_uhf],
				'OVOS_e_corr': [E_corr_rhf_50, E_corr_rhf_75, E_corr_rhf_90],
				'success': False,
				'error': str(e)
			}

		# finally:
		# 	sys.stdout = old_stdout  # Restore original stdout so we can print results from the main process


	# In your main dissociation curve code, replace the sequential loop:
	import json
	if False:  # Parallelized dissociation curve
		tracemalloc.start()
		
		# Windows only - get total RAM
		total_memory_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
		total_memory_gb = total_memory_bytes / (1024**3)
		print(f"Total memory: {total_memory_gb:.2f} GB")
			# For total_memory_gb round down to nearest integer and leave 1 GB free to avoid crashing the system
		total_memory_gb = int(total_memory_gb) - 1
		print(f"Total memory for Python process: {total_memory_gb} GB")
		set_OVOS_memory_mb = 100
		print(f"Memory peak for OVOS runs: {set_OVOS_memory_mb} MB (MAX)") # Found 65 MB peak for Li2 cc-pVDZ with 50% optimized virtual orbitals, so 100 MB is a safe upper bound for all runs.
			# Ratio of memory used by Python process to total memory
		memory_ratio = set_OVOS_memory_mb / (total_memory_gb * 1024)  # Convert total memory to MB for ratio calculation
		memory_ratio = min(memory_ratio, 1.0)  # Cap the ratio at 1.0 (100%) to avoid misleading output if set_OVOS_memory_mb exceeds total memory
		print(f"Estimated memory usage ratio: {memory_ratio:.4f} (100 MB / {total_memory_gb * 1024:.0f} MB)")
			# This set OVOS memory is for 1 core CPU
			# If we were to parallelize OVOS, we would need to multiply set_OVOS_memory_mb by the number of cores used to get the total memory usage for OVOS runs.
				# Check if we can do a set amount of cores in parallel without exceeding total memory
		num_cores = os.cpu_count()
		print(f"Number of CPU cores available: {num_cores}")
		max_parallel_cores = int(total_memory_gb * 1024 / set_OVOS_memory_mb)  # Maximum number of cores we could use in parallel without exceeding total memory
		print(f"Maximum number of cores that could be used in parallel for OVOS runs without exceeding total memory: {max_parallel_cores}")
		if max_parallel_cores < num_cores:
			print(f"Warning: Using all {num_cores} cores in parallel for OVOS runs could exceed total memory. Consider using up to {max_parallel_cores} cores for parallel OVOS runs to stay within memory limits.")
			# If maximum number of cores is more than 10 set it to 10 to be safe, as we are not sure about the memory usage of other parts of the code and to avoid overwhelming the system
		max_parallel_cores = min(max_parallel_cores, 10)
		
		# Define Li2 geometries (in Angstrom)
		distances = np.arange(2.5, 6.1, 0.1)  # Total points: 36
		geometries = [f"Li .0 .0 .0; Li .0 .0 {d:.1f}" for d in distances]
		
		# Setup parameters for worker function
		mol_setup_params = {
			'basis': "cc-pVDZ"
		}
		
		# Find what data we already have from previous runs to avoid recomputing data points whitin each geometry
		with open(f"branch/data/Li2/cc-pVDZ/dissociation_Li2_cc-pVDZ_RHF_parallel.json", "r") as f:
			existing_data = json.load(f)

		ovos_params = {}  # Add any other params needed
		
		# Create a partial function for the worker (bind parameters)	
		worker_fn = functools.partial(
			compute_geometry,
			distances=distances,
			mol_setup_params=mol_setup_params,
			ovos_params=ovos_params,
			worker_id=None,  # This will be set by the Pool.map to the index of the geometry
			existing_data=existing_data  # Pass existing data to worker function to skip already computed geometries
		)
		
		# Run in parallel
		print(f"Running {len(geometries)} geometries with {max_parallel_cores} processes...")
				# Use multiprocessing Pool to run worker_fn on each geometry in parallel
		with Pool(processes=max_parallel_cores) as pool:
			results = pool.map(worker_fn, geometries)
		print()

		# Collect results
		results_list = []
		for result in results:
			if result['success']:
				results_list.append((
					result['geom'],
					result['e_tot'],
					result['MP2_e_corr'],
					result['ooMP2_e_corr'],
					result['OVOS_e_corr'],
					None  # placeholder for lst_E_corr_virt_orbs
				))
				print(f"✓ {result['geom']}: OVOS converged")
			else:
				print(f"✗ {result['geom']}: Error - {result['error']}")
		
		# Save results to JSON
		import json
		with open(f"branch/data/Li2/cc-pVDZ/dissociation_Li2_cc-pVDZ_RHF_parallel.json", "w") as f:
			json.dump(
				results_list,
				f,
				indent=2,
				cls=NumpyEncoder
				)
		
		print(f"Data saved. Processed {len(results_list)}/{len(geometries)} geometries successfully.")
		
		current, peak = tracemalloc.get_traced_memory()
		print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
		tracemalloc.stop()















# Save miscellaneous data about the molecule and basis set
if True: # Done...
	class OOMP2(object):
		"""
		Restricted OOMP2 solver.
		Used as fcisolver in pyscf.mcscf.CASSCF(rhf, ...).
		
		CASSCF passes h1, h2 already in the MO basis.
		We build a fake RHF object and run RMP2 on it.
		This always stays restricted — cannot break spin symmetry.
		"""
		def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
			if isinstance(nelec, (int, np.integer)):
				na = nelec // 2
				nb = nelec // 2
				
			else:
				na, nb = int(nelec[0]), int(nelec[1])
			
			n_elec = na + nb

			fakemol = pyscf.gto.M(verbose=0)
			fakemol.nelectron = n_elec

			fake_hf = pyscf.scf.RHF(fakemol)
			fake_hf._eri = h2
			fake_hf.get_hcore = lambda *args: h1
			fake_hf.get_ovlp = lambda *args: np.eye(norb)

			fake_hf.mo_coeff = np.eye(norb)
			fake_hf.mo_occ = np.zeros(norb)
			fake_hf.mo_occ[:n_elec // 2] = 2

			self.mp2 = pyscf.mp.MP2(fake_hf)
			self.mp2.verbose = 0

			e_corr, t2 = self.mp2.kernel()
			e_tot = self.mp2.e_tot + ecore
			return e_tot, t2

		def make_rdm12(self, t2, norb, nelec):
			dm1 = self.mp2.make_rdm1(t2)
			dm2 = self.mp2.make_rdm2(t2)
			if isinstance(dm2, (tuple, list)):
				dm2 = sum(dm2)
			return dm1, dm2

	for basis_set in ["6-31G", "cc-pVDZ"]: # Do: "6-31G", "cc-pVDZ", ...
		for molecule in ["H2O","HF", "CO", "NH3"]: # Do: "CO", "H2O", "HF", "NH3"
			raw_print("#### Miscellaneous data about the molecule and basis set ####")
			raw_print("Molecule: ", molecule)
			raw_print("Basis set: ", basis_set)
			raw_print()

			mol, rhf, num_electrons, full_space_size, MP2 = setup_OVOS(molecule, basis_set)
			active_space_size = full_space_size - num_electrons//2 + 1

			raw_print()
			raw_print("Number of electrons: ", num_electrons)
			raw_print("Full space size in molecular orbitals: ", full_space_size)
			raw_print()

			# Check which starting should be used for the FCI/CASCI reference calculation
				# Costum full-space random MO for FCI/CASCI reference
					# From .json file corresponding to the atom and basis set, which was generated during the OVOS runs
			import json
			with open(f"branch/data/{molecule}/{basis_set}/lst_MP2_different_virt_orbs_RHF_init.json", "r") as f:
				data = json.load(f)
					# Get the full-space RHF MO coeffs from the last OVOS run with RHF initialization
				RHF_mo = np.array(data[4][-1])[0]
				RHF_E_corr = data[0][-1][-1]
					
			my_custom_mos = RHF_mo

				# Get RHF MP2 correlation energy for full space reference
			MP2_e_corr = rhf.MP2().run().e_corr
			raw_print()

			####OOMP2 (Full space)####
			# Put in the active space all orbitals of the system
			mc = pyscf.mcscf.CASSCF(rhf, mol.nao, mol.nelectron)
			mc.fcisolver = OOMP2()
			# Internal rotation inside the active space needs to be enabled
			mc.internal_rotation = True
			#mc.kernel()
			ooMP2_e_tot, ooMP2_e_cas, ooMP2_ci, ooMP2_mo_coeff, ooMP2_mo_energy = mc.kernel()
			raw_print()

				# Run FCI
			if basis_set == "6-31G" and molecule in ["HF", "H2O"]:  # FCI is only feasible for the smaller basis set and smaller molecules
				try:
					# FCI is very expensive for larger basis sets, so we only run it for the
					cisolver = pyscf.fci.FCI(mol, my_custom_mos)
					cisolver_e_tot = cisolver.kernel()[0]
					raw_print('E(FCI) = %.12f' % cisolver_e_tot, "E_corr = %.12f" % (cisolver_e_tot - rhf.e_tot))
				except Exception as e:
					cisolver_e_tot = None
					raw_print("FCI calculation failed: ", str(e))
			else:
				cisolver_e_tot = None
				raw_print("FCI calculation skipped for larger basis set due to computational cost.")

			# Save data to JSON file
			import json

			data = {
				"num_electrons": num_electrons,
				"full_space_size": full_space_size,
				"active_space_size": active_space_size,
				"MP2_e_corr": MP2_e_corr,
				"FCI_e_corr": cisolver_e_tot - rhf.e_tot if cisolver_e_tot is not None else None,
				"OOMP2_e_corr": ooMP2_e_tot - rhf.e_tot
			}

			with open(f"branch/data/{molecule}/{basis_set}/molecule_data.json", "w") as f:
				json.dump(data, f, indent=2)
			raw_print(f"Miscellaneous data saved to branch/data/{molecule}/{basis_set}/molecule_data.json")
			raw_print()


"""
There are two separate causes for negative eigenvalues in your Hessian, and neither one is a bug — they're a fundamental property of the 
J
2
J 
2
​
  functional and the Newton-Raphson method:

1. The 
J
2
J 
2
​
  functional is not convex everywhere
Your functional 
J
2
(
R
)
J 
2
​
 (R) is the second-order Hylleraas functional — a quartic function of the rotation parameters 
R
a
e
R 
ae
​
  (because the amplitudes 
t
t depend on the orbitals, and the energy is quadratic in 
t
t). A quartic function can have saddle points and local maxima, meaning the Hessian 
∂
2
J
2
/
∂
R
a
e
∂
R
b
f
∂ 
2
 J 
2
​
 /∂R 
ae
​
 ∂R 
bf
​
  is not guaranteed to be positive-definite away from a minimum.

  Negative eigenvalues mean the current point in orbital-rotation space is a saddle point with respect to some rotation directions — the functional curves upward (less negative) along those directions. This is especially common:

At the starting point (RHF orbitals), where the active/inactive partition is arbitrary and some rotations genuinely increase 
J
2
J 
2
​
 
With more virtual orbitals (8+), because there are more redundant rotation degrees of freedom and more chances for the functional landscape to be non-convex
2. Near-degenerate or degenerate orbital pairs
When two virtual orbitals 
a
a and 
b
b have similar Fock matrix eigenvalues (
f
a
a
≈
f
b
b
f 
aa
​
 ≈f 
bb
​
 ), the Hessian contribution from Term 3 (
D
a
b
(
f
a
a
+
f
b
b
)
D 
ab
​
 (f 
aa
​
 +f 
bb
​
 )) and the amplitude-dependent terms can become small or negative. For CO with 16/22 virtual orbitals, the active space includes orbitals that are nearly degenerate with some inactive orbitals, creating persistent negative curvature directions.

Why 16/22 is especially bad
With 16 out of 22 virtual orbitals active, you only have 6 inactive orbitals (3 spatial pairs). The Hessian is 96×96 with 6 negative eigenvalues — exactly the 6 inactive orbitals' worth of "wrong-curvature" directions. This suggests the optimal rotation for those 6 inactive directions is essentially zero (they're already well-placed), but the Newton step tries to follow the negative curvature and overshoots, causing the slow drift you see in the output.

Level-shifting (level_shift_threshold = 0.01) makes the Hessian eigenvalues at least 0.01, which means 
∣
R
i
∣
≤
∣
G
i
∣
/
0.01
∣R 
i
​
 ∣≤∣G 
i
​
 ∣/0.01. For a gradient norm of ~0.1 (typical for 30/42 early iterations), that gives rotation elements of order 10 — even worse. Level-shifting prevents following negative curvature directions but does nothing to limit step size.

 
Why This Works Physically
The negative eigenvalues in the Hessian correspond to rotation directions where the MP2 energy increases (becomes less negative). The Newton step naively follows these directions, suggesting rotations that mix active and inactive orbitals in a way that worsens the energy.

With level-shifting:

Negative curvature directions are converted to positive curvature with a large eigenvalue, giving a small step in the gradient descent direction — this nudges the orbitals in the correct (energy-lowering) direction
Positive curvature directions are barely affected (the shift is small relative to the existing positive eigenvalues)
The result is a hybrid Newton/steepest-descent step that:

Takes efficient Newton steps along well-conditioned directions
Takes cautious gradient-descent steps along saddle-point directions
This is exactly what algorithms like Augmented Hessian or Trust Region Newton methods do in geometry optimization — the same physics applies to orbital optimization.

"""