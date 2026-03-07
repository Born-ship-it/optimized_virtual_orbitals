"""
OVOS class

The OVOS algorithm minimizes the second-order correlation energy (MP2)
using orbital rotations.
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








		# Step (vi): Use the Newton-Raphson method to minimize the second-order Hylleraas functional
			# Reduced Linear Equation method
		if True: # Set to True to always apply block-diagonal approximation				
			# Modify Hessian to only inverse the block-diagonal part corresponding H_{ea,eb}
			len_inac = len(self.inactive_indices)
			len_ac   = len(self.active_inocc_indices)

			# Hessian has shape (len_ac*len_inac, len_ac*len_inac)
			# We want to extract the block-diagonal part corresponding to rotations between all active orbitals and each inactive orbital, which is a block of size (len_ac, len_ac) for each inactive orbital.
			# This corresponds to the indices in the Hessian that are multiples of len_ac, i.e. H_{ea,eb} where e are inactive orbitals
			H_block_diag = np.zeros_like(H)
			for i in range(len_inac):
				start = i * len_ac
				end = (i + 1) * len_ac
				H_block_diag[start:end, start:end] = H[start:end, start:end]
			
			# Set H to the block-diagonal approximation
			H = H_block_diag

		# # Only once save the block diagonal approximation to a file
		# 	if not os.path.exists("H_block_diag.csv"):
		# 		np.savetxt("H_block_diag.csv", H_block_diag, fmt='%.6e', delimiter=',',
        #    header='Block‑diagonal Hessian (rows/cols: inactive×active rotations)')

		# Correct determinant and condition number for block-diagonal approximation
			# Singularity: any single eigenvalue is ~0
			# Ill-conditioning: ratio of largest to smallest eigenvalue is large
		if True: # Set to True to always apply block-diagonal approximation
			eigvals_block = np.linalg.eigvalsh(H)
			n_neg_eigval_block = np.sum(eigvals_block < 0)
			det_H_block = np.linalg.det(H)
			cond_H_block = np.linalg.cond(H) if H.size > 0 else np.inf
			if not self.use_random_unitary_init:
				raw_print(f"    Block-diag Hessian cond.: {cond_H_block:.6e}, det.: {det_H_block:.6e}, Neg. Eigval: {n_neg_eigval_block}/{len(eigvals_block)}")

				# Sanity checks for block-diagonal H
			if not self.use_random_unitary_init:
				if cond_H_block > 1e12:
					raw_print("        WARNING: Block-diagonal Hessian is ill-conditioned!")
				if det_H_block == 0.0:
					raw_print("        WARNING: Block-diagonal Hessian is singular (zero determinant)!")
				if np.any(eigvals_block < 0.0):
					raw_print("        WARNING: Block-diagonal Hessian has negative eigenvalues!")

			if np.any(eigvals_block < 0.0):
				if not self.use_random_unitary_init:
					raw_print("                 Applying level-shift to block-diagonal Hessian to address negative eigenvalues.")
				# Level-shift negative eigenvalues to zero to ensure positive semi-definite Hessian for stable Newton-Raphson updates
				eigvals_block_clipped = np.where(eigvals_block < 0.0, 1e-6, eigvals_block)  # Shift negative eigenvalues to a small positive value
				H = (H @ np.diag(eigvals_block_clipped / eigvals_block))  # Scale H to have the clipped eigenvalues, preserving eigenvectors
				det_H_block = np.linalg.det(H)
				cond_H_block = np.linalg.cond(H) if H.size > 0 else np.inf
				if not self.use_random_unitary_init:
					raw_print(f"                     Block-diag Hessian cond.: {cond_H_block:.6e}, det.: {det_H_block:.6e}")

				# Correct if singular or ill-conditioned
					# Singularity: any single eigenvalue is ~0
					# Ill-conditioning: ratio of largest to smallest eigenvalue is large
			eigvals_block = np.linalg.eigvalsh(H)
			min_eig_block = np.min(eigvals_block)
			if cond_H_block > 1e12 or min_eig_block < 1e-2: #
				if not self.use_random_unitary_init:
					raw_print("                 Applying Tikhonov regularization to block-diagonal Hessian.")
				# Find which shift to apply: we can use a small fraction of the largest absolute eigenvalue as a regularization parameter
				shift_e = 0
					# Regularize until the Hessian is no longer singular or ill-conditioned, or until we have tried a reasonable number of shifts
				while cond_H_block > 1e12 or min_eig_block < 1e-2: # det_H_block == 0.0:
						# This fraction can be found by...
					shift_e += 1
					if self.shift_e_nr is None:
						shift = 10**(-12) * 10**shift_e  # Start with a small shift
					else:
						shift = 10**(self.shift_e_nr) * 10**shift_e  # Start with a small shift based on the previous shift applied
					reg_lambda = shift * np.max(np.abs(H))  # Regularization parameter, can be tuned
					H += reg_lambda * np.eye(H.shape[0])
						# Recompute determinant, condition number, and minimum eigenvalue after regularization
					eigvals_block = np.linalg.eigvalsh(H)
					min_eig_block = np.min(eigvals_block)
					det_H_block = np.linalg.det(H)
					cond_H_block = np.linalg.cond(H)

						# I should break out out of the loop if det_H_block is infinite or NaN, which indicates that the regularization is not working and we are likely applying too large of a shift, which can lead to divergence in the orbital optimization.
					if np.isnan(det_H_block) or np.isinf(det_H_block):
						if not self.use_random_unitary_init:
							raw_print("                     WARNING: Regularization failed to produce a finite determinant! Stopping regularization.")
							# Set an arg to break futher out
						self.regularization_failed = True
						break
					        
					# if shift_e > 12:  # Safety: don't shift more than 1e6 * max_diag
					# 	if not self.use_random_unitary_init:
					# 		raw_print("                     WARNING: Max regularization shift reached!")
					# 	break

					# Get the 1e-"number" shift that was applied for diagnostics
					self.shift_e_nr = (int(str(shift).lower().split('e')[1])-2) if 'e' in str(shift).lower() else None

				if not self.use_random_unitary_init:
					raw_print(f"                     Block-diag Hessian cond.: {cond_H_block:.6e}, det.: {det_H_block:.6e}")

					

		if True: # Set to True to always apply SVD 
			# Apply SVD truncation to the Hessian to improve stability
			U, S, Vh = np.linalg.svd(H)
			svd_threshold = 1e-3 * np.max(S)		  # Threshold for singular values, can be tuned
			S_truncated = np.where(S > svd_threshold, S, svd_threshold)
			H_svd_inv = (Vh.T / S_truncated) @ U.T
			R = - G.flatten() @ H_svd_inv
		elif False: # Set to True to always apply pseudo-inverse
			H_pinv = np.linalg.pinv(H, rcond=1e-2)
			R = - G.flatten() @ H_pinv
		else:
			R = -np.linalg.solve(H, G.flatten())  # Solve H R = -G for R using a stable linear solver
			
		
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

		E_corr = None
		keep_track = 0
		keep_track_max = 25

		self.shift_e_nr = None  # To keep track of the exponent of the last shift applied to the Hessian for diagnostics

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

				diff_grad_norm = abs(lst_grad_norm[-1] - lst_grad_norm[-2]) if len(lst_grad_norm) > 1 else None

				# Check if correlation energy has improved significantly
				diff_E_corr = abs(lst_E_corr[-1] - lst_E_corr[-2])

					# Print convergence diagnostics
				if not self.use_random_unitary_init and diff_grad_norm is not None:
					raw_print(f"    ΔE_corr = {diff_E_corr:.2e} Hartree, Δ Gradient norm = {diff_grad_norm:.2e}")

				# Convergence criteria: correlation energy change below threshold AND small gradient norm
				if (diff_E_corr < 1e-8 and self.grad_norm < 1e-6) or diff_E_corr < 1e-9: #) or diff_E_corr < 1e-12:  # Convergence threshold
					converged = True
					lst_stop_reason.append("Convergence")
					
					keep_track += 1
					if keep_track >= keep_track_max + 1:  # Require 25 consecutive iterations below threshold to confirm convergence
						if not self.use_random_unitary_init:
							raw_print("OVOS converged with stable correlation energy for 25 consecutive iterations.")

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

						if best_result_count > 250 and attempt > 5:  # If we got the same best result more than 25 times, we can stop trying more random initializations
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
		
		for basis_set in ["cc-pVDZ"]: 		#   	"6-31G" | Yet: "cc-pVDZ", ... 
			for molecule in ["H2O"]: 				#       "H2O", "CO", "HF", "NH3"
													#        (16)  (22)  (12)  (20)
				sys.stdout = Tee(sys.__stdout__, f)
				try:
					raw_print("")
					raw_print("==========================================================")
					raw_print("Running OVOS for molecule: ", molecule, " with basis set: ", basis_set)
					raw_print("==========================================================")
					raw_print("")

					mol, rhf, num_electrons, full_space_size, MP2 = setup_OVOS(molecule, basis_set)
					
					for start_guess in ["random"]: # "RHF", "prev", "random" | Yet: "random"
						with open(f"branch/data/{molecule}/{basis_set}/OVOS_{molecule}_{basis_set}_"+start_guess+"_output.txt", "w") as f:
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


# OOMP2 implementation based on PySCF's MP2, which is based on the original OOMP2 paper by Lee and Head-Gordon (https://pubs.acs.org/doi/10.1021/acs.jpca.5b07881) and the recent developments in PySCF (https://pubs.aip.org/aip/jcp/article/153/2/024109/1061482/Recent-developments-in-the-PySCF-program-package)
# OOMP2: Orbital-Optimized MP2, where the orbitals are optimized to minimize the MP2 energy.

# https://pubs.aip.org/aip/jcp/article/153/2/024109/1061482/Recent-developments-in-the-PySCF-program-package
class OOMP2(object):
    def kernel(self, h1, h2, norb, nelec, cio=None, ecore=0, **kwargs):
        # Kernel takes the set of integrals from the current set of orbitals
        fakemol = pyscf.M(verbose=0)
        fakemol.nelectron = sum(nelec)
        fake_hf = fakemol.RHF()
        fake_hf._eri = h2
        fake_hf.get_hcore = lambda *args: h1
        fake_hf.get_ovlp = lambda *args: np.eye(norb)
        
        # Build an SCF object fake_hf without SCF iterations to perform MP2
        fake_hf.mo_coeff = np.eye(norb)
        fake_hf.mo_occ = np.zeros(norb)
        fake_hf.mo_occ[:fakemol.nelectron//2] = 2
        self.mp2 = fake_hf.MP2().run()
        return self.mp2.e_tot + ecore, self.mp2.t2
    
    def make_rdm12(self, t2, norb, nelec):
        dm1 = self.mp2.make_rdm1(t2)
        dm2 = self.mp2.make_rdm2(t2)
        return dm1, dm2

# I want to create a plot which shows Li_2 dissociation 
# curve for MP2 using restricted and unrestricted
# orbitals and for OOMP2 with a cc-pVDZ basis.

# Setup molecule and basis with distance dependent geometry
	# Run RHF, UHF, MP2, and OOMP2 for each geometry
	# Plot the results
if False: # Done with OVOS runs. Comment out this line to run more or move on to the next part of the code.
	# Start tracing
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
	

	# Save original terminal settings
	old_settings = termios.tcgetattr(sys.stdin)


	molecule = "Li2"
	basis_set = "cc-pVDZ"
	start_guess = "RHF"

	try:
		with open(f"branch/data/{molecule}/{basis_set}/dissociation_{molecule}_{basis_set}_"+start_guess+"_output.txt", "w") as f:
			sys.stdout = Tee(sys.__stdout__, f)
			try:
				# Define Li2 geometries (in Angstrom)
				distances = np.linspace(2.5, 6, 10)  # From 1.0 to 5.0 Angstrom with 10 points
				geometries = [f"Li .0 .0 .0; Li .0 .0 {d}" for d in distances]

				# Store results
				results = []

				for geom in geometries:
					# Allow user to skip to next iteration by pressing 's' key (with a short timeout to avoid blocking)
						# Check if a key was pressed (with a very short timeout)
					if is_key_pressed(0.01):   # 10 ms timeout
						key = get_key()
						if key == '\x03':  # Ctrl-C to quit
							raw_print("\n[Exiting OVOS...]")
							sys.exit(0)

					# Molecule
					mol = pyscf.M(
						atom=geom, 
						basis="cc-pVDZ",
						unit="angstrom", 
						charge=0, 
						spin=0,
						symmetry=False
						)
					
					raw_print("")
					raw_print("==========================================================")
					raw_print("\nRunning calculations for geometry: ", geom)
					raw_print("")

					# RHF
					rhf = mol.RHF().run()
					rhf_e_tot = rhf.e_tot
					raw_print("RHF total energy: ", rhf_e_tot)

					# MP2
					MP2_e_corr = rhf.MP2().run().e_corr
					raw_print("MP2 correlation energy: ", MP2_e_corr)
					

					# OOMP2
						# Put in the active space all orbitals of the system
					mc = pyscf.mcscf.CASSCF(rhf, mol.nao, mol.nelectron)
					mc.fcisolver = OOMP2()
						# Internal rotation inside the active space needs to be enabled
					mc.internal_rotation = True
						# Run OOMP2
					ooMP2_e_tot, ooMP2_e_cas, ooMP2_ci, ooMP2_mo_coeff, ooMP2_mo_energy = mc.kernel()
					ooMP2_e_corr = ooMP2_e_tot - rhf.e_tot
					raw_print("OOMP2 correlation energy: ", ooMP2_e_corr)
					raw_print("")

					# OVOS @ 50% of full space optimized virtual orbitals
					try:
							# Miscellaneous data about the molecule and basis set
								# Number of electrons
						num_electrons = mol.nelec[0] + mol.nelec[1]
								# Full space size in molecular orbitals
						full_space_size = int(rhf.mo_coeff.shape[1])
							
							# List of MP2 correlation energies for different numbers of optimized virtual orbitals
						lst_E_corr_virt_orbs = [[],[],[],[],[],[]]  # [[E_corr_list], [num_opt_virtual_orbs_list], [iterations_till_convergence_list], [Unr. SCF check list], [mo_coeffs_final], [lst_stop_reason]]
						lst_MP2_virt_orbs = []  # [(num_opt_virtual_orbs, E_corr, iterations_till_convergence), ...]

							# List of error messages for failed runs
						lst_error_messages = []
							# List of message for priting later
						lst_message = []

							# Set maximum number of optimized virtual orbitals to test
								# Denoted in molecular orbitals (not spin orbitals)
						max_opt_virtual_orbs = full_space_size*2 - num_electrons
								# Set statrting number of optimized virtual orbitals and increment
						num_opt_virtual_orbs_current = 0.5 * max_opt_virtual_orbs  # Start with number of occupied orbitals
								# Ensure num_opt_virtual_orbs_current is an even integer for closed-shell molecules
						num_opt_virtual_orbs_current = int(round(num_opt_virtual_orbs_current / 2) * 2)

							# Run OVOS
								# The if state will prevent re-getting RHF orbitals if using previous or random init
						mo_coeffs = np.array([rhf.mo_coeff, rhf.mo_coeff]) 		# Start with RHF orbitals for both alpha and beta spins
						Fmo_rot = None 											# Fock matrix in the MO basis will be constructed later based on the initial orbitals
						Fao_get = rhf.get_fock() 								# Get Fock matrix in AO basis for later use in orbital optimization
						Fao = np.array([Fao_get, Fao_get]) 						# Use the same Fock matrix for both spins as starting point
						init_orbs = "RHF"

								# Run OVOS with RHF orbitals as starting guess
						lst_E_corr, lst_iter_counts, mo_coeffs, Fmo_rot, lst_stop_reason = OVOS(mol=mol, scf=rhf, Fao=Fao, num_opt_virtual_orbs=num_opt_virtual_orbs_current, init_orbs=init_orbs, mo_coeff=mo_coeffs).run_ovos(mo_coeffs=mo_coeffs, Fmo_rot=Fmo_rot)

								# Check alpha/beta are the same for a tolerance - Done after orbital optimization
						diff_alpha_beta = np.max(np.abs(mo_coeffs[0] - mo_coeffs[1]))
						if diff_alpha_beta > 1e-4:
							raw_print("Warning: OVOS with ", num_opt_virtual_orbs_current, " optimized vorbs resulted in different alpha and beta orbitals (max diff: ", diff_alpha_beta, ").")
								# Store message
							lst_message.append(f"OVOS w. {num_opt_virtual_orbs_current} optimized vorbs resulted in different alpha and beta orbitals (max diff: {diff_alpha_beta}). Here largest alpha {np.max(mo_coeffs[0])} and beta {np.max(mo_coeffs[1])} orbital coeffs.")
								# Append True-False flag to lst_E_corr_virt_orbs
							lst_E_corr_virt_orbs[3].append("True")
						else:
							lst_E_corr_virt_orbs[3].append("False")

								# Store OVOS results
						lst_MP2_virt_orbs.append((num_opt_virtual_orbs_current, lst_E_corr[-1], lst_iter_counts[-1], lst_stop_reason[-1]))
						lst_E_corr_virt_orbs[0].append(lst_E_corr)
						lst_E_corr_virt_orbs[1].append(num_opt_virtual_orbs_current)
						lst_E_corr_virt_orbs[2].append(lst_iter_counts)
						lst_E_corr_virt_orbs[4].append(mo_coeffs.tolist())
						lst_E_corr_virt_orbs[5].append(lst_stop_reason)


					# Catch errors during OVOS
					except AssertionError as e:
						raw_print(f"Error during OVOS with {num_opt_virtual_orbs_current} optimized virtual orbitals: {e}")
						raw_print("Rerunning with the same number of virtual orbitals.")

						# Add error message to list
							# Get results if available
						lst_error_messages.append((num_opt_virtual_orbs_current, str(e), lst_iter_counts[-1] if lst_iter_counts is not None else 0))
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
					raw_print("Previously optimized virtual orbitals were used as starting guess for each OVOS run.")
					raw_print("")

					# Print error messages summary
					if len(lst_error_messages) > 0:
						raw_print("#### Error messages summary ####")
						for num_opt_virtual_orbs_current, error_msg, iter_ in lst_error_messages:
							raw_print("  OVOS w. ", num_opt_virtual_orbs_current, " optimized vorbs failed at iteration ", iter_ ," w. error: ", error_msg)
						raw_print("")

					# Store results
						# Store geometry, RHF total energy, MP2 correlation energy, and OOMP correlation energy
					results.append((geom, rhf.e_tot, MP2_e_corr, ooMP2_e_corr, lst_E_corr[-1], lst_E_corr_virt_orbs))

					# Save data to JSON files
					import json

					str_name = "dissociation_RHF_init"

					str_atom = molecule
					str_basis = basis_set

					raw_print("Saving data to branch/data/"+str_atom+"/"+str_basis+"/...")

					# Save MP2 correlation energy convergence data
					with open("branch/data/"+str_atom+"/"+str_basis+"/"+str_name+".json", "w") as f:
						json.dump(results, f, indent=2)

					raw_print("Data saved to branch/data/"+str_atom+"/"+str_basis+"/"+str_name+".json")
			finally:
				sys.stdout = sys.__stdout__
	finally:
		# Restore terminal settings no matter what
		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

	# Get snapshot
	current, peak = tracemalloc.get_traced_memory()
	raw_print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
	tracemalloc.stop()


	# The parallelized version of the above code...
from multiprocessing import Pool
import functools

def compute_geometry(geom, distances, mol_setup_params, ovos_params, worker_id=None):
	"""
	Worker function: compute OVOS for a single geometry.
	Runs in a separate process.
	"""

	# Reconstruct molecule from geometry string
	mol = pyscf.M(
		atom=geom,
		basis=mol_setup_params['basis'],
		unit="angstrom",
		charge=0,
		spin=0,
		symmetry=False
	)
	
	# RHF
	rhf = mol.RHF().run()
	rhf_e_tot = rhf.e_tot
	
	# MP2
	MP2_e_corr = rhf.MP2().run().e_corr
	
	# OOMP2
	mc = pyscf.mcscf.CASSCF(rhf, mol.nao, mol.nelectron)
	mc.fcisolver = OOMP2()
	mc.internal_rotation = True
	ooMP2_e_tot, ooMP2_e_cas, ooMP2_ci, ooMP2_mo_coeff, ooMP2_mo_energy = mc.kernel()
	ooMP2_e_corr = ooMP2_e_tot - rhf.e_tot
	
	# OVOS @ 50% of full space
	num_electrons = mol.nelec[0] + mol.nelec[1]
	full_space_size = int(rhf.mo_coeff.shape[1])
	max_opt_virtual_orbs = full_space_size * 2 - num_electrons
	num_opt_virtual_orbs_current = int(round(0.5 * max_opt_virtual_orbs / 2) * 2)
		# Setup initial guess for OVOS
	mo_coeffs = np.array([rhf.mo_coeff, rhf.mo_coeff])
	Fmo_rot = None
	Fao_get = rhf.get_fock()
	Fao = np.array([Fao_get, Fao_get])
	
	import time
		# Index of geometry being processed (for debugging)
			# Get index from last number in str geom, which is in the format "Li .0 .0 .0: Li .0 .0 {distance}"
	geom_dist = float(geom.split()[-1])
	geom_idx = np.where(np.isclose(distances, geom_dist))[0][0] + 1
	time.sleep(5*geom_idx)  # Simulate some delay to allow print statements from different processes to interleave less

	# Print progress
	raw_print("")
	raw_print("==========================================================")
	raw_print(f"Processing geometry: {geom} with worker ID: {worker_id}")
	raw_print(f"RHF total energy: {rhf_e_tot}")
	raw_print(f"MP2 correlation energy: {MP2_e_corr}")
	raw_print(f"OOMP2 correlation energy: {ooMP2_e_corr}")
	raw_print(f"Running OVOS with {num_opt_virtual_orbs_current} optimized virtual orbitals...")

	geom_idx_inv = len(distances) - geom_idx + 1
	time.sleep(5*geom_idx_inv)  # Simulate some delay to allow print statements from different processes to interleave less

	# # Suppress output from worker processes to avoid cluttering the terminal
	# from io import StringIO
	# old_stdout = sys.stdout
	# sys.stdout = StringIO()

	try:
		lst_E_corr, lst_iter_counts, mo_coeffs, Fmo_rot, lst_stop_reason = OVOS(
			mol=mol,
			scf=rhf,
			Fao=Fao,
			num_opt_virtual_orbs=num_opt_virtual_orbs_current,
			init_orbs="RHF",
			mo_coeff=mo_coeffs
		).run_ovos(mo_coeffs=mo_coeffs, Fmo_rot=Fmo_rot)
		
		return {
			'geom': geom,
			'rhf_e_tot': rhf_e_tot,
			'MP2_e_corr': MP2_e_corr,
			'ooMP2_e_corr': ooMP2_e_corr,
			'OVOS_e_corr': lst_E_corr[-1],
			'success': True,
			'error': None
		}

	except Exception as e:
		return {
			'geom': geom,
			'rhf_e_tot': None,
			'MP2_e_corr': None,
			'ooMP2_e_corr': None,
			'OVOS_e_corr': None,
			'success': False,
			'error': str(e)
		}

	# finally:
	# 	sys.stdout = old_stdout  # Restore original stdout so we can print results from the main process


# In your main dissociation curve code, replace the sequential loop:
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
	distances = np.linspace(2.5, 6, 10)
	geometries = [f"Li .0 .0 .0; Li .0 .0 {d}" for d in distances]
	
	# Setup parameters for worker function
	mol_setup_params = {
		'basis': "cc-pVDZ"
	}
	
	ovos_params = {}  # Add any other params needed
	
	# Create a partial function for the worker (bind parameters)	
	worker_fn = functools.partial(
		compute_geometry,
		distances=distances,
		mol_setup_params=mol_setup_params,
		ovos_params=ovos_params,
		worker_id=None  # This will be set by the Pool.map to the index of the geometry
	)
	
	# Run in parallel
	print(f"Running {len(geometries)} geometries with {max_parallel_cores} processes...")
			# Use multiprocessing Pool to run worker_fn on each geometry in parallel
	with Pool(processes=max_parallel_cores) as pool:
		results = pool.map(worker_fn, geometries)
	
	# Collect results
	results_list = []
	for result in results:
		if result['success']:
			results_list.append((
				result['geom'],
				result['rhf_e_tot'],
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
		json.dump(results_list, f, indent=2)
	
	print(f"Data saved. Processed {len(results_list)}/{len(geometries)} geometries successfully.")
	
	current, peak = tracemalloc.get_traced_memory()
	print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
	tracemalloc.stop()















# Save miscellaneous data about the molecule and basis set
if False: # Done...
	for basis_set in ["cc-pVDZ"]: # Do: "6-31G", "cc-pVDZ", ...
		for molecule in ["H2O"]: # Do: "CO", "H2O", "HF", "NH3"
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
			with open(f"branch/data/{molecule}/{basis_set}/lst_MP2_different_virt_orbs_random.json", "r") as f:
				data = json.load(f)
					# Get the full-space random MO coeffs from the last OVOS run with random unitary initialization
				rand_mo = np.array(data[4][-1])[0]
				rand_E_corr = data[0][-1][-1]
				# Costum full-space RHF MO for FCI/CASCI reference
					# From .json file corresponding to the atom and basis set, which was generated during the OVOS runs
			with open(f"branch/data/{molecule}/{basis_set}/lst_MP2_different_virt_orbs_RHF_init.json", "r") as f:
				data = json.load(f)
					# Get the full-space RHF MO coeffs from the last OVOS run with RHF initialization
				RHF_mo = np.array(data[4][-1])[0]
				RHF_E_corr = data[0][-1][-1]
					
				# Decide which one to use based on which has the lower MP2 correlation energy (lower is better for correlation energy)
			if rand_E_corr < RHF_E_corr:
				my_custom_mos = rand_mo
			else:
				my_custom_mos = RHF_mo

				# Get RHF MP2 correlation energy for full space reference
			MP2_e_corr = rhf.MP2().run().e_corr
			raw_print()

				# Run CCSD(T)
			# ccsd = pyscf.cc.CCSD(rhf).run()
			# raw_print()

			####OOMP2 (Full space)####
			# Put in the active space all orbitals of the system
			mc = pyscf.mcscf.CASSCF(rhf, mol.nao, mol.nelectron)
			mc.fcisolver = OOMP2()
			# Internal rotation inside the active space needs to be enabled
			mc.internal_rotation = True
			#mc.kernel()
			ooMP2_e_tot, ooMP2_e_cas, ooMP2_ci, ooMP2_mo_coeff, ooMP2_mo_energy = mc.kernel()
			raw_print()

				# Run FCI/CASCI
			# cisolver = pyscf.fci.FCI(mol, my_custom_mos)
			# cisolver_e_tot = cisolver.kernel()[0]
			# raw_print('E(FCI) = %.12f' % cisolver_e_tot, "E_corr = %.12f" % (cisolver_e_tot - rhf.e_tot))

			# casci = rhf.CASCI(full_space_size, num_electrons)
			# casci.kernel(my_custom_mos)
			# raw_print('E(CASCI) = %.12f' % casci.e_tot, "E_corr = %.12f" % (casci.e_tot - rhf.e_tot))
			# raw_print()

			# Save data to JSON file
			import json

			data = {
				"num_electrons": num_electrons,
				"full_space_size": full_space_size,
				"active_space_size": active_space_size,
				"MP2_e_corr": MP2_e_corr,
				"CCSD_e_corr": None, # ccsd.e_corr,
				"CCSD(T)_e_corr": None, # ccsd.e_tot + ccsd.ccsd_t() - rhf.e_tot,
				"FCI_e_corr": None, # cisolver_e_tot - rhf.e_tot,
				"OOMP2_e_corr": ooMP2_e_tot - rhf.e_tot,
				"CASSCF_e_corr": None
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