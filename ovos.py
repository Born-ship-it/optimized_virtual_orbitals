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

# import time


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

	def __init__(self, mol: pyscf.gto.Mole, scf, Fao, num_opt_virtual_orbs: int, mo_coeff, init_orbs: str = "RHF") -> None:
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
		print()
		print("#### Active and inactive spaces ####")
		print("Total number of spin-orbitals: ", self.tot_num_spin_orbs)
		print("Active occupied spin-orbitals: ", self.active_occ_indices)
		print("Active unoccupied spin-orbitals: ", self.active_inocc_indices)
		print("Inactive unoccupied spin-orbitals: ", self.inactive_indices)
		print()

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
			# If Fock is already in spin-orbital basis, we can just take the diagonal as orbital energies
			self.eps = np.diag(Fmo)
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
			self.eri_4fold_ao, [C_vir_a, C_occ_a, C_vir_a, C_occ_a], compact=False
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

			# Build virtual Fock sub-matrix from eps (diagonal, using absolute indices)
			eps_vir = eps[vir_abs]             # shape: (nvir,)
			F_virt = np.diag(eps_vir)          # shape: (nvir, nvir) — local indexing

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
					# eri_as is (nvir_spin, nvir_spin, nocc_spin, nocc_spin) — local indices
					# Active virtual = first nvir local indices
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

							# 1. f_ac δ_bd  — F_virt uses local indices
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
		J_2 = compute_mp2_energy_standard(self, MP1_amplitudes, eri_as)

		# Get MP2 energy using optimized method
		# J_2 = compute_mp2_energy_optimized(self, eps, Fmo_spin, eri_as, MP1_amplitudes)

		# Sanity check: optimized vs standard method
		# assert np.isclose(J_2, J_2_standard, atol=1e-10), f"Optimized MP2 energy {J_2} does not match standard MP2 energy {J_2_standard}!"

		# Print MP2 energy for current active space
			# Note: the denominator in the print statement is the total number of virtual orbitals in the active space,
				# which is the sum of active inocc and inactive indices. 
			# This gives a sense of how much of the virtual space is being included in the MP2 energy calculation.
		print(f"[{len(self.active_inocc_indices)}/{len(self.active_inocc_indices + self.inactive_indices)}]: Computed MP2 correlation energy (spin-orbital): ", J_2)

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
			
			return D_ab

		# Compute D_ab using optimized method
		D_ab = compute_D_ab_block(self, MP1_amplitudes)

		# Check symmetry
		sym_diff = np.max(np.abs(D_ab - D_ab.T))
		if sym_diff > 1e-10:
			print("WARNING: D_ab is not symmetric!")
			# Print the non-symmetric parts
			diff_matrix = np.abs(D_ab - D_ab.T)
			idx = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
			print(f"Max diff at ({idx[0]},{idx[1]}): D={D_ab[idx]:.6e}, D.T={D_ab.T[idx]:.6e}")

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
		print("  Orbital Optimization Diagnostics:")

			# Gradient
		print("  - Gradient:")

		# Norm of the gradient
		print(f"    Gradient norm: {np.linalg.norm(G):.6e}")
		if np.linalg.norm(G) < 1e-8:
			print("        WARNING: Gradient norm is very small!")

		# Check if gradient is reasonable
		if np.linalg.norm(G.flatten()) < 1e-8:
			print("        WARNING: Gradient is essentially zero!")

		# Sanity checks for G
		# 	# Check if gradient is empty
		if np.count_nonzero(G) == 0:
			print("        WARNING: No inactive orbitals -> gradient is empty (expected)")
			print("                 Orbitals cannot be optimized further (full virtual space)")
		
			# Check if gradient has finite values
		assert np.all(np.isfinite(G)), "Gradient G contains non-finite values!"


		def compute_hessian_block(self, t_block, eri_as_block, D_ab, Fmo_spin):
			"""
			Compute Hessian H of J_2 w.r.t. orbital rotations using block arrays.
			Eliminates the inner Python for-loop over occupied pairs.
			
			H_{ae,bf} = 2 * sum_{i>j} t^{ab}_{ij} * <ef||ij>
						- sum_{i>j,c} [t^{ac}_{ij}*<bc||ij> + t^{cb}_{ij}*<ca||ij>] * delta_ef
						+ D_ab * (f_aa - f_bb) * delta_ef
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
			
			# === TERM 3: D_ab * (f_aa - f_bb) * delta_ef ===
			f_diff = f_diag_vir[:, None] - f_diag_vir[None, :]  # (nvir, nvir)
			term3 = D_ab * f_diff
			
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
		print("  - Hessian:")
		eigvals = np.linalg.eigvalsh(H)
		det_H = np.prod(eigvals)
		cond_H = np.linalg.cond(H) if H.size > 0 else np.inf

		# Check eigenvalues
		print(f"    Hessian norm: {np.linalg.norm(H):.6e}, (Neg. Eigval: {np.sum(eigvals < 0)}/{len(eigvals)})")
		if np.any(eigvals < -1e-6):
			print("        WARNING: Hessian has negative eigenvalues!")

			# Determinant of Hessian
		det_H = det_H if H.size > 0 else 0.0
		print(f"    Hessian determinant: {det_H:.6e}")
		if det_H < 1e-4:
			print("        WARNING: Hessian determinant is small!")
		if det_H == 0.0:
			print("        WARNING: Hessian is singular!")

		# Sanity checks for H
		if np.count_nonzero(H) == 0:
			print("        WARNING: Hessian is zero.")
		if not np.all(np.isfinite(H)):
			print("        WARNING: Hessian H contains non-finite values!")
		if not np.allclose(H, H.T, atol=1e-10):
			print("        WARNING: Hessian H is not symmetric!")








		# Step (vi): Use the Newton-Raphson method to minimize the second-order Hylleraas functional

		# solve for rotation parameters
			# Original direct inversion method
				# equation 14: R = - G H^-1 -> R = -G @ np.linalg.inv(H)
			# Alternatively, use Reduced Linear Equation (RLE) method
				# This is a more stable approach that avoids direct inversion of the Hessian, which can be ill-conditioned.
				# The RLE method solves the linear equation H*R = -G using an iterative solver or a block-diagonal approximation, which can be more robust for large systems or when the Hessian has small eigenvalues.
					# The block-diagonal approximation assumes that the Hessian can be approximated as block-diagonal, where each block corresponds to rotations between all active orbitals and one specific inactive orbital.
					# This reduces the problem to solving smaller linear equations for each block, which can be more stable and efficient than inverting the full Hessian.
					# The RLE method can be implemented using an iterative solver like Conjugate Gradient or GMRES, or by directly solving the smaller block-diagonal systems if the Hessian is sufficiently well-approximated by a block-diagonal structure.

		def solve_block_diagonal_RLE(H, G, n_active, n_inactive):
			"""
			Block diagonal RLE method for solving H*R = -G.
			
			Follows the Newton-Raphson approach for orbital rotation parameters R.
			Uses block-diagonal approximation where each block corresponds to
			rotations between all active orbitals and one specific inactive orbital.
			
			Parameters:
			-----------
			H : ndarray
				Hessian matrix (total_size x total_size) where total_size = n_active * n_inactive
			G : ndarray
				Gradient vector (total_size,) or matrix that can be flattened
			n_active : int
				Number of active orbitals
			n_inactive : int
				Number of inactive (virtual) orbitals
				
			Returns:
			--------
			R : ndarray
				Solution vector R = -H⁻¹G (using block-diagonal approximation)
			"""
			# Handle edge cases
			if n_inactive == 0:
				# No inactive orbitals - return empty solution
				return np.array([], dtype=H.dtype)
			
			if n_active == 0:
				# No active orbitals - return empty solution
				return np.array([], dtype=H.dtype)
			
			total_size = n_active * n_inactive
			block_size = n_active
			
			# Validate dimensions
			if total_size == 0:
				return np.array([], dtype=H.dtype)
			
			# Validate and reshape inputs
			if H.shape != (total_size, total_size):
				# Try to reshape, but only if dimensions are compatible
				if H.size == total_size * total_size:
					H = H.reshape(total_size, total_size)
				else:
					raise ValueError(f"Hessian dimension mismatch: H has {H.size} elements, "
								f"expected {total_size * total_size}")
			
			if G.ndim != 1:
				G = G.flatten()
			if G.shape[0] != total_size:
				raise ValueError(f"Gradient dimension mismatch: G has {G.shape[0]} elements, "
							f"expected {total_size}")
			
			# Method 1: Direct block-diagonal solve (most efficient if blocks are truly independent)
			# This assumes H is block-diagonal with minimal coupling between blocks
			return _solve_strict_block_diagonal(H, G, n_active, n_inactive)


		def _solve_strict_block_diagonal(H, G, n_active, n_inactive):
			"""
			Solve assuming strictly block-diagonal Hessian.
			This is the Purvis-Bartlett approximation where only diagonal blocks are considered.
			"""
			if n_inactive == 0 or n_active == 0:
				return np.array([], dtype=H.dtype)
			
			block_size = n_active
			n_blocks = n_inactive
			R = np.zeros(n_active * n_inactive, dtype=H.dtype)
			
			for block_idx in range(n_blocks):
				start = block_idx * block_size
				end = (block_idx + 1) * block_size
				
				# Extract diagonal block H_ii
				H_block = H[start:end, start:end]
				G_block = G[start:end]
				
				# Solve H_ii * R_i = -G_i
				try:
					# Direct inversion
					R_block = -G_block @ np.linalg.inv(H_block)

					# Solve linear system (more stable than direct inversion)
					# R_block = -np.linalg.solve(H_block, G_block)

				except np.linalg.LinAlgError:
					# Fallback to pseudoinverse if singular
					R_block = -G_block @ np.linalg.pinv(H_block)
				
				R[start:end] = R_block
			
			return R

		# Alternative: A simpler wrapper that handles all edge cases
		def solve_block_diagonal_RLE_safe(H, G, n_active, n_inactive):
			"""
			Safe version that handles all edge cases including n_inactive = 0.
			"""
			# Quick return for empty cases
			if n_inactive == 0 or n_active == 0:
				return np.array([], dtype=np.float64)
			
			total_size = n_active * n_inactive
			
			# If H or G are empty/zero-sized, return appropriate result
			if total_size == 0:
				return np.array([], dtype=np.float64)
			
			# Ensure inputs are properly shaped
			if H.shape != (total_size, total_size):
				if H.size == total_size * total_size:
					H = H.reshape(total_size, total_size)
				else:
					raise ValueError(f"Hessian size mismatch. Expected {(total_size, total_size)}, got {H.shape}")
			
			G_flat = np.asarray(G).flatten()
			if len(G_flat) != total_size:
				raise ValueError(f"Gradient size mismatch. Expected {total_size}, got {len(G_flat)}")
			
			# Call the main function
			return solve_block_diagonal_RLE(H, G_flat, n_active, n_inactive)















		# Set to True to use Reduced Linear Equation method
			# Use it when Hessian is ill-conditioned:
		use_RLE = True

			# Direct inversion method
		if not use_RLE:
			G = G.flatten()  # Flatten G to 1D array

			# Solve for R, unoccupied space
				# Handle if H is singular
			if cond_H > 1e12 and np.linalg.matrix_rank(H) < H.shape[0] and det_H == 0.0:
				print("        WARNING: Using pseudo-inverse for orbital rotations.")
				R = - G @ np.linalg.pinv(H)
			else:
				R = - G @ np.linalg.inv(H)

			# Reduced Linear Equation method
		if use_RLE:				
			R = solve_block_diagonal_RLE_safe(H, G, len(self.active_inocc_indices), len(self.inactive_indices))
			
		
		# Initialize R,
			# Matrix: self.full_indices x self.full_indices
		R_matrix = np.zeros((len(self.full_indices), len(self.full_indices), ), dtype=np.float64)

		# Build R_matrix from R[A, E]
		# for idx_A, A in enumerate(self.active_inocc_indices):
		# 	for idx_E, E in enumerate(self.inactive_indices):
		# 		# Extract R_EA and R_AE
		# 		R_AE = R[idx_A * len(self.inactive_indices) + idx_E]
		# 		R_EA = -1.0*R_AE  # Antisymmetry

		# 		# Check antisymmetry
		# 		if abs(R_EA + R_AE) > 1e-10:
		# 			print(f"WARNING: R matrix antisymmetry violated for indices ({E},{A}): R_EA={R_EA:.6e}, R_AE={R_AE:.6e}, sum={R_EA + R_AE:.6e}")

		# 		# Fill in R_matrix
		# 		R_matrix[E, A] = R_AE
		# 		R_matrix[A, E] = R_EA

		# Handle the case when there are no inactive orbitals (full virtual space)
		if len(self.inactive_indices) == 0:
			print("        WARNING: No inactive orbitals -> R matrix is empty (expected)")
		else:
			vir = np.array(self.active_inocc_indices)
			inact = np.array(self.inactive_indices)
			nvir = len(vir)
			ninact = len(inact)

			R_2d = R.reshape(nvir, ninact)
			R_matrix[np.ix_(inact, vir)] = R_2d.T    # R[E,A] = R_AE
			R_matrix[np.ix_(vir, inact)] = -R_2d     # R[A,E] = -R_AE (antisymmetry)


			# Rotation matrix
		print("  - Rotation matrix:")
			# Convergence check based on max element of R_matrix
		max_R_elem = np.max(np.abs(R_matrix))
		print(f"    Rotation norm {np.linalg.norm(R_matrix):.6e}, (Max el.: {max_R_elem:.6e})")
			# Check that R is anti-symmetric
		diff_R = np.linalg.norm(R_matrix + R_matrix.T)
		assert diff_R < 1e-6, f"R_matrix is not anti-symmetric, ||R + R.T|| = {diff_R}"
			# Check that R_matrix has no NaN or Inf values
		assert np.all(np.isfinite(R_matrix)), "R_matrix contains NaN or Inf values!"
			# Check that R_matrix is not all zeros
		count_nonzero_R = np.count_nonzero(R_matrix)
		if count_nonzero_R == 0:
			print("        WARNING: R_matrix is all zeros!")
			# Check that R_matrix is small for convergence
		if max_R_elem < 1e-6:
			print("        WARNING: Rotation parameters are very small -> Orbitals are optimized!")


		# Step (vii): Construct the unitary orbital rotation matrix U = exp(R)

		# Unitary rotation matrix
			# Padé approximation for matrix exponential, more stable than eigendecomposition for large matrices	
		U = scipy.linalg.expm(R_matrix)

		# Numerical checks on U
			# Check U if U^T U = I
		assert np.allclose(U.T @ U, np.eye(len(U)), atol=1e-4), "Unitary rotation matrix U is not unitary!"
			# Check is U has no NaN or Inf values
		assert np.all(np.isfinite(U)), "Unitary rotation matrix U contains NaN or Inf values!"
			# Check that U is not all zeros
		assert not np.allclose(U, 0), "Unitary rotation matrix U is all zeros!"

		# Check that U is orthogonal
			# Note, sometimes due to numerical errors (e-11) U@U.T is not exactly identity, but very close
				# If the difference is too large, raise an error
		diff = np.linalg.norm(U@U.T - np.eye(len(U)))
		assert diff < 1e-6, f"U is not orthogonal, ||U@U.T - I|| = {diff}"

		# Check the active occupied area of U is close to identity
		for i in self.active_occ_indices:
			for j in self.active_occ_indices:
				expected = 1.0 if i == j else 0.0
				if abs(U[i, j] - expected) > 1e-6:
					print(f"WARNING: U deviates from identity in active occupied block at ({i},{j}): U={U[i,j]:.6e}, expected={expected:.6e}")

		# Helper functions for orbital coefficient conversions
		def spatial_to_spin_mo(mo_coeffs):
			"""
			Convert spatial MO coefficients (alpha,beta) -> spin-orbital MO matrix.
			Args:
				mo_coeffs: array-like with shape (2, n_ao, n_spatial) or [C_alpha, C_beta]
						where C_alpha/C_beta are (n_ao, n_spatial).
			Returns:
				mo_coeffs_spin: ndarray (n_ao, 2*n_spatial) with columns [α0,β0,α1,β1,...]
				orb_map: ndarray (n_spin,) mapping spin-index -> spatial-index (p//2)
				orbspin: ndarray (n_spin,) spin label (0=alpha,1=beta)
			"""
			# Normalize input form
			if isinstance(mo_coeffs, (list, tuple)):
				C_alpha, C_beta = mo_coeffs
			else:
				# assume array-like (2, n_ao, n_spatial)
				C_alpha, C_beta = mo_coeffs[0], mo_coeffs[1]

			n_ao, n_spatial = C_alpha.shape
			# assert C_beta.shape == (n_ao, n_spatial), "Alpha/beta shapes mismatch"

			n_spin = 2 * n_spatial
			C_spin = np.zeros((n_ao, n_spin), dtype=C_alpha.dtype)

			# for s in range(n_spatial):
				# C_spin[:, 2*s]   = C_alpha[:, s]  # alpha s
				# C_spin[:, 2*s+1] = C_beta[:, s]   # beta s
			C_spin[:, 0::2] = C_alpha
			C_spin[:, 1::2] = C_beta

			orb_map = np.array([i // 2 for i in range(n_spin)], dtype=int)
			orbspin = np.array([i % 2 for i in range(n_spin)], dtype=int)
			return C_spin, orb_map, orbspin

		def spin_to_spatial_mo(mo_coeffs_spin, orbspin=None):
			"""
			Convert interleaved spin-orbital MO matrix -> spatial (alpha,beta).
			Args:
				mo_coeffs_spin: ndarray (n_ao, n_spin) with interleaved columns [α0,β0,...]
				orbspin: optional ndarray of length n_spin (0/1). If None, assume interleaved.
			Returns:
				mo_coeffs_spatial: ndarray shape (2, n_ao, n_spatial) -> [C_alpha, C_beta]
			"""
			n_ao, n_spin = mo_coeffs_spin.shape
			if orbspin is None:
				orbspin = np.array([i % 2 for i in range(n_spin)], dtype=int)
			# infer n_spatial
			n_spatial = n_spin // 2
			C_alpha = np.zeros((n_ao, n_spatial), dtype=mo_coeffs_spin.dtype)
			C_beta  = np.zeros((n_ao, n_spatial), dtype=mo_coeffs_spin.dtype)

			# if strictly interleaved, faster slicing:
			if np.all(orbspin == np.array([i % 2 for i in range(n_spin)])):
				C_alpha[:, :] = mo_coeffs_spin[:, 0::2]
				C_beta[:, :]  = mo_coeffs_spin[:, 1::2]
			else:
				# general extraction using orbspin and orb_map
				idx_alpha = np.where(orbspin == 0)[0]
				idx_beta  = np.where(orbspin == 1)[0]
				# assert len(idx_alpha) == len(idx_beta) == n_spatial, "Unexpected orbspin layout"
				C_alpha[:, :] = mo_coeffs_spin[:, idx_alpha]
				C_beta[:, :]  = mo_coeffs_spin[:, idx_beta]

			return np.array([C_alpha, C_beta])


		# Step (viii): Rotate the orbitals

			# convert to spin orbital basis
				# Manual spatial→spin conversion: interleave α and β columns
					# PySCF convention: mo_coeffs shape is (n_ao, n_spatial_orbs), columns are orbitals
					# mo_coeffs[0] = alpha, mo_coeffs[1] = beta
					# Result: (n_ao, n_spin_orbs) where spin_orbs = [α0, β0, α1, β1, ...]
		mo_coeffs_spin, orb_map, orbspin = spatial_to_spin_mo(mo_coeffs)

			# apply rotation
				# mo_coeffs_spin shape: (n_ao, num_spin_orbitals)
				# U shape: (num_spin_orbitals, num_spin_orbitals)
				# Result: C_new = C_old @ U (standard orbital rotation)
		mo_coeffs_spin_rot = mo_coeffs_spin @ U

			# convert back to spatial orbital basis
				# Manual spin→spatial conversion: extract α and β columns using orbspin
		mo_coeffs_rot = spin_to_spatial_mo(mo_coeffs_spin_rot, orbspin=orbspin)

			# Check on what orbitals were rotated
		num_rotated = 0
		for spin in [0, 1]:
				# In each spin block
			for s in range(mo_coeffs[spin].shape[1]):
					# Get original and rotated orbitals
				orig_orb = mo_coeffs[spin][:, s]
				rot_orb = mo_coeffs_rot[spin][:, s]
					# Check if they differ significantly
				if not np.allclose(orig_orb, rot_orb, atol=1e-6):
					num_rotated += 1
				# else:
					# print(f"    Orbital {s} (spin {spin}) not rotated.")
		print()
		print(f"    Rotated {num_rotated} spin orbitals out of {mo_coeffs[0].shape[1] + mo_coeffs[1].shape[1]} total.")
		print()

		# In the case of non-canonical orbitals,
			# Does the MO coefficients supposed to stay orthonormal after rotation?
				# Yes, the rotation is unitary, so it should preserve orthonormality
			# Check that the rotated orbitals are still orthonormal

				# check shape
		for spin in [0, 1]:
			expected_shape_spatial = mo_coeffs[spin].shape
			assert mo_coeffs_rot[spin].shape == expected_shape_spatial, f"mo_coeffs_rot shape is {mo_coeffs_rot[spin].shape}, expected {expected_shape_spatial}"

		# Check that the active occupied orbitals are unchanged
			# Convert to spatial orbital index
			for idx in self.active_occ_indices:
				idx = idx // 2  # spatial orbital index
				# Get original and rotated orbitals
				orig_orb = mo_coeffs[spin][:, idx]
				rot_orb = mo_coeffs_rot[spin][:, idx]

				assert np.allclose(orig_orb, rot_orb, atol=1e-4), f"Active occupied spin orbital {idx} (spin {spin}) was changed during rotation!"
			
			# Check that rotated orbitals are orthonormal
				# Sum a coloumn of mo_coeffs to check normalization for a given spin
					# Check: C^T S C = I
				C_i = mo_coeffs_rot[spin]
				norm = C_i.T @ self.S @ C_i
				assert np.allclose(norm, np.eye(norm.shape[0]), atol=1e-6), f"MO coefficients for spin {spin} are not orthonormal!"


		# Rotate the Fock matrix in the MO basis as well
			# Fock matrix in original MO basis: Fmo_spin = C^T Fao C
			# After rotation: Fmo_rot = U^T Fmo_spin U
		Fmo_rot = U.T @ Fmo_spin @ U

			# Check that the rotated Fock matrix is still Hermitian
		# if not np.allclose(Fmo_rot, Fmo_rot.T.conj(), atol=1e-10):
		# 	print("WARNING: Rotated Fock matrix is not Hermitian!")

			# Check that the diagonal elements of the rotated Fock matrix are real
		# if not np.all(np.isreal(np.diag(Fmo_rot))):
		# 	print("WARNING: Diagonal elements of rotated Fock matrix are not real!")

			# Check that the rotated Fock matrix has no NaN or Inf values
		# if not np.all(np.isfinite(Fmo_rot)):
		# 	print("WARNING: Rotated Fock matrix contains NaN or Inf values!")

		# Check if the rotated diagonalized Fock matrix has the same eigenvalues as the original 
			# Should be the same since it's a unitary transformation
		eigvals_original = np.linalg.eigvalsh(Fmo_spin)
		eigvals_rot = np.linalg.eigvalsh(Fmo_rot)
			# Check if they are close within a reasonable numerical tolerance
		if not np.allclose(eigvals_rot, eigvals_original, atol=1e-6):
			print("WARNING: Eigenvalues of rotated Fock matrix differ from original!")
			assert np.allclose(eigvals_rot, eigvals_original, atol=1e-6), "Eigenvalues of rotated Fock matrix differ from original!"

		return mo_coeffs_rot, Fmo_rot


	





	def run_ovos(self,  mo_coeffs, Fmo_rot):
		"""
		Run the OVOS algorithm.
		"""

		converged = False
		max_iter = 1000
		iter_count = 0

		E_corr = None

		while not converged:
			iter_count += 1
			print("#### OVOS Iteration ", iter_count, " ####")
			
			# Check if maximum iterations reached
			if iter_count >= max_iter:
				print("Maximum number of iterations reached. OVOS did not converge.")
				lst_E_corr.append(E_corr)
				lst_iter_counts.append(iter_count)
				# Set converged to False
				converged = False
				break
			

			# Step (iii-iv): Compute MP2 correlation energy and amplitudes
			print(" Step (iii)-(iv): Compute MP2 correlation energy and amplitudes")
			E_corr, MP1_amplitudes, eri_as, Fmo_spin = self.MP2_energy(mo_coeffs = mo_coeffs, Fmo = Fmo_rot)

			# Step (ix): check convergence and stability
			print(" Step (ix): Check convergence and stability")

			# Initialize flags for non-convergence detection
			oscillation_detected = False
			drift_detected = False
			false_convergence_detected = False
			chaos_detected = False
			diverging = False

			# First iteration, initialize lists
			if iter_count == 1:
				lst_E_corr = []
				lst_E_corr.append(E_corr)
				
				lst_iter_counts = []
				lst_iter_counts.append(iter_count)

				lst_stop_reason = []
				
				# Step (v)-(viii): Orbital optimization
				print(" Step (v)-(viii): Orbital optimization")
				mo_coeffs, Fmo_rot = self.orbital_optimization(mo_coeffs=mo_coeffs, 
															MP1_amplitudes=MP1_amplitudes, 
															eri_as=eri_as, 
															Fmo_spin=Fmo_spin)

			# Subsequent iterations: check convergence and stability
			elif iter_count < max_iter and iter_count > 1:
				threshold = 1e-8
				change = np.abs(E_corr - lst_E_corr[-1])
				
				# ==============================================
				# NON-CONVERGENCE DETECTION CHECKS
				# ==============================================
				
				# ----- 1. OSCILLATION DETECTION -----
				# Check if we have enough history to detect oscillation patterns
				# if len(lst_E_corr) >= 500 and change > threshold * 10:
				# 	# Look at recent oscillations (last 4-6 iterations)
				# 	recent_changes = [np.abs(lst_E_corr[i] - lst_E_corr[i-1]) for i in range(-5, 0)]
				# 	recent_signs = [np.sign(lst_E_corr[i] - lst_E_corr[i-1]) for i in range(-5, 0)]
					
				# 	# Detect alternating sign pattern (+ - + - or - + - +)
				# 	if len(recent_signs) >= 4:
				# 		# Check if signs alternate consistently
				# 		alternations = sum(1 for i in range(1, len(recent_signs)) if recent_signs[i] != recent_signs[i-1])
				# 		if alternations >= len(recent_signs) - 1:  # All signs alternate
				# 			# Check if oscillation amplitude isn't decaying
				# 			if np.std(recent_changes) > 0.1 * np.mean(recent_changes):
				# 				oscillation_detected = True
					
				# 	# Detect limit cycle (bouncing between same values)
				# 	if len(lst_E_corr) >= 500 and not oscillation_detected:
				# 		for cycle_len in [2, 3, 4, 6, 8]:  # Check for 2-cycle, 3-cycle, 4-cycle
				# 			if len(lst_E_corr) >= 3 * cycle_len:
				# 				recent_cycle = lst_E_corr[-cycle_len*2:-cycle_len]
				# 				current_cycle = lst_E_corr[-cycle_len:]
				# 				# Check if cycles are nearly identical
				# 				cycle_diff = np.max([np.abs(recent_cycle[i] - current_cycle[i % cycle_len]) 
				# 								for i in range(cycle_len)])
				# 				if cycle_diff < threshold * 10:
				# 					oscillation_detected = True
				# 					print(f"		WARNING: Detected {cycle_len}-cycle limit cycle oscillation!")
				# 					break
				
				# ----- 2. DRIFT DETECTION -----
				# if len(lst_E_corr) >= 8 and not oscillation_detected:
				# 	# Check for monotonic drift over many iterations
				# 	recent_trend = lst_E_corr[-8:]
				# 	is_monotonic = all(recent_trend[i] <= recent_trend[i+1] for i in range(len(recent_trend)-1)) or \
				# 				all(recent_trend[i] >= recent_trend[i+1] for i in range(len(recent_trend)-1))
					
				# 	if is_monotonic:
				# 		# Calculate drift rate
				# 		total_drift = np.abs(recent_trend[-1] - recent_trend[0])
				# 		drift_per_iter = total_drift / 7
						
				# 		# If still drifting significantly after many iterations
				# 		if drift_per_iter > threshold * 0.1 and len(lst_E_corr) > 15:
				# 			drift_detected = True
				
				# ----- 3. STAGNATION DETECTION (false convergence) -----
				# if len(lst_E_corr) >= 10:
				# 	# Check if change was small then grew again
				# 	was_small = any(np.abs(lst_E_corr[i] - lst_E_corr[i-1]) < threshold * 0.1 
				# 				for i in range(-8, -3))
				# 	is_now_large = change > threshold * 2
					
				# 	if was_small and is_now_large:
				# 		false_convergence_detected = True
				
				# ----- 4. CHAOS DETECTION (erratic jumps) -----
				# if len(lst_E_corr) >= 5:
				# 	recent_changes = [np.abs(lst_E_corr[i] - lst_E_corr[i-1]) for i in range(-4, 0)]
				# 	mean_change = np.mean(recent_changes)
				# 	std_change = np.std(recent_changes)
					
				# 	# High variance relative to mean indicates erratic behavior
				# 	if mean_change > threshold * 10 and std_change > 0.5 * mean_change:
				# 		chaos_detected = True
				
				# ----- 5. DIVERGENCE DETECTION (consistent increases) -----
				# if len(lst_E_corr) >= 5 and E_corr > lst_E_corr[-1]:
				# 	last_few = lst_E_corr[-5:]
				# 	if all(last_few[i] <= last_few[i+1] for i in range(len(last_few)-1)):
				# 		diverging = True
				
				# ==============================================
				# REPORT NON-CONVERGENCE WARNINGS
				# ==============================================
				# if oscillation_detected:
				# 	oscillation_amplitude = np.max(recent_changes) - np.min(recent_changes) if len(recent_changes) > 1 else 0
				# 	print(f"		WARNING: Oscillation detected! Amplitude: {oscillation_amplitude:.6e}")
				# 	print(f"		Not converging - in limit cycle.")
				# 	# Optionally set a flag to break after certain number of oscillation iterations
				# 	if not hasattr(self, 'oscillation_counter'):
				# 		self.oscillation_counter = 0
				# 	self.oscillation_counter += 1
				# 	# Break if oscillation persists
				# 	if self.oscillation_counter > 5:
				# 		print(f"		Oscillation persisted for {self.oscillation_counter} iterations. Stopping.")
				# 		converged = False
				# 		lst_E_corr.append(E_corr)
				# 		lst_iter_counts.append(iter_count)
				# 		lst_stop_reason.append("Oscillation")
				# 		break
				
				# if drift_detected:
				# 	print(f"		WARNING: Persistent monotonic drift detected!")
				# 	print(f"		Drift rate: {drift_per_iter:.6e} per iteration")
				# 	print(f"		Not converging - linear drift.")
				# 	# Optionally break after confirming drift
				# 	if not hasattr(self, 'drift_counter'):
				# 		self.drift_counter = 0
				# 	self.drift_counter += 1
				# 	if self.drift_counter > 5:
				# 		print(f"		Drift persisted for {self.drift_counter} iterations. Stopping.")
				# 		converged = False
				# 		lst_E_corr.append(E_corr)
				# 		lst_iter_counts.append(iter_count)
				# 		lst_stop_reason.append("Drift")
				# 		break
				
				# if false_convergence_detected:
				# 	print(f"		WARNING: False convergence detected!")
				# 	print(f"		Change was small, then grew again: {change:.6e}")
				# 	print(f"		Not stable - will not converge.")
				# 	# This is serious - break immediately
				# 	converged = False
				# 	lst_E_corr.append(E_corr)
				# 	lst_iter_counts.append(iter_count)
				# 	lst_stop_reason.append("False Convergence")
				# 	break
				
				# if chaos_detected:
				# 	print(f"		WARNING: Erratic convergence behavior detected!")
				# 	print(f"		Change variance: {std_change:.6e}, Mean: {mean_change:.6e}")
				# 	print(f"		Not stable - may be chaotic.")
				# 	# Optionally break after persistent chaos
				# 	if not hasattr(self, 'chaos_counter'):
				# 		self.chaos_counter = 0
				# 	self.chaos_counter += 1
				# 	if self.chaos_counter > 8:
				# 		print(f"		Chaotic behavior persisted. Stopping.")
				# 		converged = False
				# 		lst_E_corr.append(E_corr)
				# 		lst_iter_counts.append(iter_count)
				# 		lst_stop_reason.append("Chaos")
				# 		break
				
				if diverging:
					print("		WARNING: Consistent energy increases - may be diverging!")
				
				# ==============================================
				# NORMAL CONVERGENCE CHECK
				# ==============================================
				if change < threshold and not (oscillation_detected or drift_detected or chaos_detected):
					# # Additional check: make sure it's not just a temporary plateau
					# if len(lst_E_corr) >= 4:
					# 	# Look back 3 iterations
					# 	recent_changes_small = all(np.abs(lst_E_corr[i] - lst_E_corr[i-1]) < threshold 
					# 							for i in range(-1, 0))
					# 	if recent_changes_small:
					# 		converged = True
					# 		print("OVOS converged in ", iter_count, " iterations.")
					# 		lst_E_corr.append(E_corr)
					# 		lst_iter_counts.append(iter_count)
					# 		lst_stop_reason.append("Convergence")
					# 		break
					# 	else:
					# 		print(f"	Change in correlation energy: {change:.6e} Hartree (threshold: {threshold:.1e})")
					# 		print(f"		Change below threshold but recent history shows instability.")
					# 		print(f"		Continuing to verify convergence...")
					# 		print()
							
					# 		# Step (v)-(viii): Orbital optimization
					# 		print(" Step (v)-(viii): Orbital optimization")
					# 		mo_coeffs, Fmo_rot = self.orbital_optimization(mo_coeffs=mo_coeffs, 
					# 													MP1_amplitudes=MP1_amplitudes, 
					# 													eri_as=eri_as, 
					# 													Fmo_spin=Fmo_spin)
					# else:
					converged = True
					print("OVOS converged in ", iter_count, " iterations.")
					lst_E_corr.append(E_corr)
					lst_iter_counts.append(iter_count)
					lst_stop_reason.append("Convergence")
					break
				else:
					# Only append if we didn't break due to non-convergence
					if not (oscillation_detected and hasattr(self, 'oscillation_counter') and self.oscillation_counter > 10) and \
					not (drift_detected and hasattr(self, 'drift_counter') and self.drift_counter > 5) and \
					not false_convergence_detected and \
					not (chaos_detected and hasattr(self, 'chaos_counter') and self.chaos_counter > 8):
						
						lst_E_corr.append(E_corr)
						lst_iter_counts.append(iter_count)
						lst_stop_reason.append("Not Converged")
						
						print(f"	Change in correlation energy: {change:.6e} Hartree (threshold: {threshold:.1e})")
						
						# If the change is positive, warn the user
						if E_corr > lst_E_corr[-2]:
							print("		WARNING: Correlation energy increased in this iteration!")
						print()
						
						# Step (v)-(viii): Orbital optimization
						print(" Step (v)-(viii): Orbital optimization")
						mo_coeffs, Fmo_rot = self.orbital_optimization(mo_coeffs=mo_coeffs, 
																	MP1_amplitudes=MP1_amplitudes, 
																	eri_as=eri_as, 
																	Fmo_spin=Fmo_spin)
		# Which direction did we go?
		if converged:
			final_E_corr = lst_E_corr[-1]
			initial_E_corr = lst_E_corr[0]
			print()
			print("#### OVOS Summary ####")
			print(f"Initial MP2 correlation energy: {initial_E_corr:.10f} Hartree")
			print(f"Final OVOS correlation energy: {final_E_corr:.10f} Hartree")
			print(f"Total change in correlation energy: {final_E_corr - initial_E_corr:.10f} Hartree")
			if final_E_corr < initial_E_corr:
				print("OVOS successfully lowered the correlation energy.")
			else:
				print("WARNING: OVOS increased the correlation energy!")
			print(f"Total number of iterations: {iter_count}")


		# Print information about the spaces
		print()
		print("#### Active and inactive spaces ####")
		print("Total number of spin-orbitals: ", self.tot_num_spin_orbs)
		print("Active occupied spin-orbitals: ", self.active_occ_indices)
		print("Active unoccupied spin-orbitals: ", self.active_inocc_indices)
		print("Inactive unoccupied spin-orbitals: ", self.inactive_indices)
		print()
		
		# Check if OVOS converged
		if not converged:
			print("OVOS did not converge within the maximum number of iterations.")

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
	# Molecule
	atom_choose_between = [
		"H .0 .0 .0; H .0 .0 0.74144",  # H2 bond length 0.74144 Angstrom
		"Li .0 .0 .0; H .0 .0 1.595",   # LiH bond length 1.595 Angstrom
		"O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;",  # H2O equilibrium geometry
		"C  0.0000  0.0000  0.0000; H  0.0000  0.9350  0.5230; H  0.0000 -0.9350  0.5230;", # CH2 
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
		"cc-pVTZ"
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

	atom, basis = (atom_choose_between[find_atom[select_atom]], basis_choose_between[find_basis[select_basis]])

	# Print start message
	print(" Running OVOS on ", atom, " with basis set ", basis)
	print("")

	# Get number of electrons and full space size in molecular orbitals
	unit = "angstrom" # angstrom or bohr
		# Initialize molecule and UHF
	mol = pyscf.M(atom=atom,
			    basis=basis,
				unit=unit,
				spin=0,  # Closed-shell molecule
				charge=0,
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
	attempts_total = 1000
	if use_random_unitary_init == True:
		try_best_of = True
	else:
		try_best_of = False

		# Loop over different numbers of optimized virtual orbitals
	# List of MP2 correlation energies for different numbers of optimized virtual orbitals
	lst_E_corr_virt_orbs = [[],[],[],[],[]]  # [[E_corr_list], [num_opt_virtual_orbs_list], [iterations_till_convergence_list], [Unr. SCF check list], [mo_coeffs_final]]
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

	while num_opt_virtual_orbs_current < max_opt_virtual_orbs:
		# Increment num_opt_virtual_orbs until OVOS converges successfully
		num_opt_virtual_orbs_current += increment 

		lst_E_corr = None  # Reset lst_E_corr for each run

		print("")
		print("#### OVOS with ", num_opt_virtual_orbs_current, " out of ", max_opt_virtual_orbs," optimized virtual orbitals (Retry count: ", retry_count,") ####")

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

					print("    Using previously optimized orbitals as starting guess.")				

					# Run OVOS
				lst_E_corr, lst_iter_counts, mo_coeffs, Fmo_rot, lst_stop_reason = OVOS(mol=mol, scf=rhf, Fao=Fao, num_opt_virtual_orbs=num_opt_virtual_orbs_current, init_orbs=init_orbs, mo_coeff=mo_coeffs).run_ovos(mo_coeffs=mo_coeffs, Fmo_rot=Fmo_rot)

			if try_best_of == True: # Multiple runs of OVOS with different random initializations
				print("Using random unitary rotated UHF virtual orbitals as starting guess.")

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
				total_iterations_reached_count = 0

				while attempt < attempts_total:
					attempt += 1
					print("")
					print("---- Attempt ", attempt, " of ", attempts_total, " ----")

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
					ovos_obj = OVOS(mol=mol, scf=rhf, Fao=Fao, num_opt_virtual_orbs=num_opt_virtual_orbs_current, init_orbs=init_orbs, mo_coeff=mo_coeffs)
						# Pass, type: ignore
					ovos_obj.eri_4fold_ao = eri_4fold_ao  # Pass the 4-fold AO integrals to avoid recomputation
					ovos_obj.S = S  # Pass the overlap matrix
					ovos_obj.hcore_ao = hcore_ao  # Pass the core Hamiltonian in AO basis

					# Run OVOS with the current random unitary rotated orbitals
					lst_E_corr_attempt, lst_iter_counts_attempt, mo_coeffs_attempt, Fmo_rot_attempts, lst_stop_reason_attempts = ovos_obj.run_ovos(mo_coeffs=mo_coeffs, Fmo_rot=Fmo_rot)

					# Update best result if this attempt is better than the current best
					if best_E_corr is None or lst_E_corr_attempt[-1] < best_E_corr:
						best_E_corr = lst_E_corr_attempt[-1]
						best_lst_E_corr = lst_E_corr_attempt
						best_lst_iter_counts = lst_iter_counts_attempt
						best_mo_coeffs = mo_coeffs_attempt
						best_Fmo_rot = Fmo_rot_attempts
						best_lst_stop_reason = lst_stop_reason_attempts

					# Check if we ended up at the same best result multiple times, which could indicate a local minimum
					if np.isclose(best_E_corr, lst_E_corr_attempt[-1], atol = 1e-6) and best_E_corr is not None:
						best_result_count += 1
						# If we eneded up at the same best result multiple times, we can break early as it's likely a local minimum
						if best_result_count >= 10:
							print(f"Best result of {best_E_corr:.6f} Hartree has been found {best_result_count} times. Breaking early from attempts.")
							break

					# If we ended up at a better result, reset the best result count by a tolerance
					if best_E_corr is not None and lst_E_corr_attempt[-1] < best_E_corr - 1e-6:
						best_result_count = 0					

					# If we keep reaching the maximum number of iterations without convergence
					if lst_iter_counts_attempt is not None and len(lst_iter_counts_attempt) > 0 and lst_iter_counts_attempt[-1] >= 1000:
						total_iterations_reached_count += 1
						# Break early as it's likely not converging
						if total_iterations_reached_count >= 10:
							print(f"Reached maximum number of iterations for 10 attempts. Breaking early from attempts.")
							break

					# Reset total iterations reached count if we get a converged result
					if lst_stop_reason_attempts is not None and len(lst_stop_reason_attempts) > 0 and lst_stop_reason_attempts[-1] == "Convergence":
						total_iterations_reached_count = 0


				# Use the best result from all attempts
				lst_E_corr = best_lst_E_corr
				lst_iter_counts = best_lst_iter_counts
				mo_coeffs = best_mo_coeffs
				Fmo_rot = best_Fmo_rot
				lst_stop_reason = best_lst_stop_reason


			# run_OVOS got stuck in a non-converging loop
			if len(lst_E_corr) >= 2500:
				print("OVOS with ", num_opt_virtual_orbs_current, " optimized virtual orbitals did not converge.")


			# Check alpha/beta are the same for a tolerance - Done after orbital optimization
			diff_alpha_beta = np.max(np.abs(mo_coeffs[0] - mo_coeffs[1]))
			if diff_alpha_beta > 1e-4:
				print("Warning: OVOS with ", num_opt_virtual_orbs_current, " optimized vorbs resulted in different alpha and beta orbitals (max diff: ", diff_alpha_beta, ").")
					# Store message
				lst_message.append(f"OVOS w. {num_opt_virtual_orbs_current} optimized vorbs resulted in different alpha and beta orbitals (max diff: {diff_alpha_beta}). Here largest alpha {np.max(mo_coeffs[0])} and beta {np.max(mo_coeffs[1])} orbital coeffs.")
					# Append True-False flag to lst_E_corr_virt_orbs
				lst_E_corr_virt_orbs[3].append("True")
			else:
				lst_E_corr_virt_orbs[3].append("False")

			# run_OVOS converged to a positive MP2 correlation energy
			if lst_E_corr[-1] > 0:
				print("Warning: OVOS with ", num_opt_virtual_orbs_current, " optimized virtual orbitals converged to a positive MP2 correlation energy.")

			# Store results
			lst_MP2_virt_orbs.append((num_opt_virtual_orbs_current, lst_E_corr[-1], len(lst_E_corr), lst_stop_reason[-1]))
			lst_E_corr_virt_orbs[0].append(lst_E_corr)
			lst_E_corr_virt_orbs[1].append(num_opt_virtual_orbs_current)
			lst_E_corr_virt_orbs[2].append(lst_iter_counts)
			# ...
			lst_E_corr_virt_orbs[4].append(mo_coeffs.tolist())  # Convert numpy array to list for JSON serialization

			# Reset retry count on success
			retry_count = 0


				

		# Catch errors during OVOS
		except AssertionError as e:
			print(f"Error during OVOS with {num_opt_virtual_orbs_current} optimized virtual orbitals: {e}")
			print("Rerunning with the same number of virtual orbitals.")

			# Add error message to list

				# Get results if available
			lst_error_messages.append((num_opt_virtual_orbs_current, str(e), lst_iter_counts[-1] if lst_iter_counts is not None else 0))

			retry_count += 1
			if retry_count >= max_retries:
				print(f"Maximum retries reached for {num_opt_virtual_orbs_current} optimized virtual orbitals. Skipping to next.")
				retry_count = 0
				continue

			num_opt_virtual_orbs_current -= increment  # Decrement to retry the same number
			continue



	# Print summary of the run
	print("Number of electrons: ", num_electrons)
	print("Full space size in molecular orbitals: ", full_space_size)
	print("Maximum number of optimized virtual orbitals tested: ", max_opt_virtual_orbs)
	print("Total OVOS runs completed: ", len(lst_MP2_virt_orbs))
	print("")

	# Print the final MP2 correlation energy after all OVOS and amount of iterations till convergence
	for num_opt_virtual_orbs_current, E_corr, iter_, lst_stop_reason in lst_MP2_virt_orbs:
		print("MP2 correlation energy, for ", num_opt_virtual_orbs_current, f" optimized virtual orbitals: ", '%.5E' % Decimal(E_corr),f" ({(E_corr/MP2.e_corr)*100:.4}%)"+" @ ", iter_, " iterations till convergence (",lst_stop_reason,")")
	print("MP2 correlation energy, for full space: ", '%.5E' % Decimal(MP2.e_corr), "| Difference:", '%.5E' % Decimal(MP2.e_corr - lst_MP2_virt_orbs[-1][1]))
	print("")

	# Print if the check of alpha and beta orbitals were the same
	for msg in lst_message:
		print(msg)
	print("")
	

	# Print what methods were used
	if use_prev_virt_orbs == True:
		print("Previously optimized virtual orbitals were used as starting guess for each OVOS run.")
	if use_random_unitary_init == True:
		print("Random unitary rotations of UHF virtual orbitals were used as starting guess for each OVOS run.")
	print("")

	# Print error messages summary
	if len(lst_error_messages) > 0:
		print("#### Error messages summary ####")
		for num_opt_virtual_orbs_current, error_msg, iter_ in lst_error_messages:
			print("  OVOS w. ", num_opt_virtual_orbs_current, " optimized vorbs failed at iteration ", iter_ ," w. error: ", error_msg)
		print("")







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

	print("Saving data to branch/data/"+str_atom+"/"+str_basis+"/...")

	# Save MP2 correlation energy convergence data
	with open("branch/data/"+str_atom+"/"+str_basis+"/lst_MP2_"+str_name+".json", "w") as f:
		json.dump(lst_E_corr_virt_orbs, f, indent=2)

	print("Data saved to branch/data/"+str_atom+"/"+str_basis+"/lst_MP2_"+str_name+".json")

"""
Get data
"""

for basis_set in ["6-31G"]: # Yet: "6-31G","cc-pVDZ", ... 
	for molecule in ["CO", "H2O", "HF", "NH3"]: # Done: "CO", "H2O", "HF", "NH3"
		print("")
		print("====================================================")
		print("Running OVOS for molecule: ", molecule, " with basis set: ", basis_set)
		print("====================================================")
		print("")

		mol, rhf, num_electrons, full_space_size, MP2 = setup_OVOS(molecule, basis_set)

		# get_OVOS_data(num_opt_virtual_orbs_current=0, retry_count=0, start_guess="RHF", select_atom=molecule, select_basis=basis_set)
		# get_OVOS_data(num_opt_virtual_orbs_current=0, retry_count=0, start_guess="prev", select_atom=molecule, select_basis=basis_set)
		# get_OVOS_data(num_opt_virtual_orbs_current=0, retry_count=0, start_guess="random", select_atom=molecule, select_basis=basis_set)

		# Get I add the terminal output of the OVOS runs to a text file for later reference
		import sys
		original_stdout = sys.stdout  # Save a reference to the original standard output
		with open(f"branch/data/{molecule}/{basis_set}/OVOS_output.txt", "w") as f:
			sys.stdout = f  # Change the standard output to the file we created
			# Run OVOS again to capture the output in the file
			get_OVOS_data(num_opt_virtual_orbs_current=0, retry_count=0, start_guess="RHF", select_atom=molecule, select_basis=basis_set)
			get_OVOS_data(num_opt_virtual_orbs_current=0, retry_count=0, start_guess="prev", select_atom=molecule, select_basis=basis_set)
			# get_OVOS_data(num_opt_virtual_orbs_current=0, retry_count=0, start_guess="random", select_atom=molecule, select_basis=basis_set)
			sys.stdout = original_stdout  # Reset the standard output to its original value
		


# assert False, "Done with OVOS runs. Comment out this line to run more or move on to the next part of the code."

# Save miscellaneous data about the molecule and basis set
if False: # Done...
	for basis_set in ["6-31G", "cc-pVDZ"]:
		for molecule in ["CO", "H2O", "HF", "NH3"]:
			print("#### Miscellaneous data about the molecule and basis set ####")
			print("Molecule: ", molecule)
			print("Basis set: ", basis_set)
			print()

			mol, rhf, num_electrons, full_space_size, MP2 = setup_OVOS(molecule, basis_set)
			active_space_size = full_space_size - num_electrons//2 + 1
			print()
			print("Number of electrons: ", num_electrons)
			print("Full space size in molecular orbitals: ", full_space_size)
			print()

				# Get UHF MP2 correlation energy for full space reference
			MP2_e_corr = rhf.MP2().run().e_corr
			print()

				# Run CCSD(T)
			ccsd = pyscf.cc.CCSD(rhf).run()
			print()

				# Run CASSCF
			# casscf = rhf.CASSCF(full_space_size, num_electrons).run()
			# print()

				# Run FCI/CASCI
			# casci = rhf.CASCI(full_space_size, num_electrons).run()
			# print()

			# Save data to JSON file
			import json

			data = {
				"num_electrons": num_electrons,
				"full_space_size": full_space_size,
				"active_space_size": active_space_size,
				"MP2_e_corr": MP2_e_corr,
				"CCSD_e_corr": ccsd.e_corr,
				"CCSD(T)_e_corr": ccsd.e_tot + ccsd.ccsd_t() - rhf.e_tot,
				"FCI_e_corr": None, # casci.e_tot - rhf.e_tot,
				"CASSCF_e_corr": None # casscf.e_tot - rhf.e_tot
			}

			with open(f"branch/data/{molecule}/{basis_set}/molecule_data.json", "w") as f:
				json.dump(data, f, indent=2)
			print(f"Miscellaneous data saved to branch/data/{molecule}/{basis_set}/molecule_data.json")
			print()
