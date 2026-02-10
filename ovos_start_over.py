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

	def __init__(self, mol: pyscf.gto.Mole, num_opt_virtual_orbs: int, mo_coeff, use, init_orbs: str = "UHF") -> None:
		# Use ...
		self.use_canonical_Fock = use[0]

		# Store parameters
		self.mol = mol
		self.num_opt_virtual_orbs = num_opt_virtual_orbs
		self.init_orbs = init_orbs

		# Perform initial Hartree-Fock calculation to get orbitals
		if init_orbs == "UHF":
			# Set up unrestricted Hartree-Fock calculation 
			self.uhf = pyscf.scf.UHF(mol).run()
			self.e_rhf = self.uhf.e_tot
			self.h_nuc = mol.energy_nuc()

		if init_orbs == "RHF":
			# Set up restricted Hartree-Fock calculation 
			self.rhf = pyscf.scf.RHF(mol).run()
			self.e_rhf = self.rhf.e_tot
			self.h_nuc = mol.energy_nuc()
			# Convert RHF orbitals to UHF format
			self.uhf = self.rhf.to_uhf()

		self.mo_coeffs = mo_coeff 

		# MP2 calculation
		self.MP2 = self.uhf.MP2().run()
		self.MP2_ecorr = self.MP2.e_corr

		# Overlap matrix check
		self.S = mol.intor('int1e_ovlp')

		# Fock matrix in AO basis 
		self.Fao = self.uhf.get_fock()
		
		# Integrals in AO basis
		self.hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
		self.overlap = mol.intor('int1e_ovlp')
		self.eri_4fold_ao = mol.intor('int2e_sph', aosym=1)

		# Number of orbitals
		self.tot_num_spin_orbs = int(2*self.mo_coeffs.shape[1])
			# Check that orbitals are unrestricted
		assert self.mo_coeffs[0].shape[1] == self.mo_coeffs[1].shape[1], "Number of alpha and beta orbitals must be equal for OVOS."
		
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
		# assert self.tot_num_spin_orbs >= self.num_opt_virtual_orbs+self.nelec, "Your space 'num_opt_virtual_orbs' is too large"  

			# Sum a coloumn of mo_coeffs to check normalization for a given spin
				# Chekc: C^T S C = I
		# for spin in [0, 1]:
		# 	C_i = self.mo_coeffs[spin]
			
		# 	norm = C_i.T @ self.S @ C_i
			# assert np.allclose(norm, np.eye(norm.shape[0]), atol=1e-6), f"MO coefficients for spin {spin} are not orthonormal!"

			# Check that the active spaces are correctly built
		# for I in self.active_occ_indices:
		# 	assert I < self.nelec, f"I={I} not less than number of electrons {self.nelec}"
		# for A in self.active_inocc_indices:
		# 	assert A >= self.nelec, f"A={A} not greater than or equal to number of electrons {self.nelec}"
		# 	assert A < self.nelec + self.num_opt_virtual_orbs, f"A={A} not less than number of electrons + num_opt_virtual_orbs {self.nelec + self.num_opt_virtual_orbs}"
		# for E in self.inactive_indices:
		# 	assert E >= self.nelec + self.num_opt_virtual_orbs, f"E={E} not greater than or equal to number of electrons + num_opt_virtual_orbs {self.nelec + self.num_opt_virtual_orbs}"
		# 	assert E < self.tot_num_spin_orbs, f"E={E} not less than total number of spin orbitals {self.tot_num_spin_orbs}"

			# Check that the spaces do not overlap
		# assert len(set(self.active_occ_indices).intersection(set(self.active_inocc_indices))) == 0, "Active occupied and active unoccupied spaces overlap!"
		# assert len(set(self.active_occ_indices).intersection(set(self.inactive_indices))) == 0, "Active occupied and inactive unoccupied spaces overlap!"
		# assert len(set(self.active_inocc_indices).intersection(set(self.inactive_indices))) == 0, "Active unoccupied and inactive unoccupied spaces overlap!"


	def spatial_to_spin_eri(self, eri_aaaa, eri_aabb, eri_bbbb):
		"""
		Convert spatial ERIs to spin-orbital ERIs.
		"""
		n_spatial = eri_aaaa.shape[0]
		n_spin = 2 * n_spatial
		
		eri_spin = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.float64)
		
		for p in range(n_spin):
			pa, sp_p = p // 2, p % 2
			for q in range(n_spin):
				pb, sp_q = q // 2, q % 2
				for r in range(n_spin):
					ra, sp_r = r // 2, r % 2
					for s in range(n_spin):
						rb, sp_s = s // 2, s % 2
						
						# Handle all 16 possible spin combinations
						if (sp_p, sp_q, sp_r, sp_s) == (0, 0, 0, 0):
							eri_spin[p, q, r, s] = eri_aaaa[pa, pb, ra, rb]
						elif (sp_p, sp_q, sp_r, sp_s) == (1, 1, 1, 1):
							eri_spin[p, q, r, s] = eri_bbbb[pa, pb, ra, rb]
						elif (sp_p, sp_q, sp_r, sp_s) == (0, 0, 1, 1):
							# (αα|ββ)
							eri_spin[p, q, r, s] = eri_aabb[pa, pb, ra, rb]
						elif (sp_p, sp_q, sp_r, sp_s) == (1, 1, 0, 0):
							# (ββ|αα) = (αα|ββ) with indices swapped
							eri_spin[p, q, r, s] = eri_aabb[ra, rb, pa, pb]
						elif (sp_p, sp_q, sp_r, sp_s) == (0, 1, 0, 1):
							# (αβ|αβ)
							eri_spin[p, q, r, s] = eri_aabb[pa, ra, pb, rb]
						elif (sp_p, sp_q, sp_r, sp_s) == (1, 0, 1, 0):
							# (βα|βα) = (αβ|αβ) with indices swapped
							eri_spin[p, q, r, s] = eri_aabb[pb, rb, pa, ra]
						else:
							# Other combinations might be zero due to spin conservation
							eri_spin[p, q, r, s] = 0.0
		
		return eri_spin

	def spatial_to_spin_fock(self, Fmo_a, Fmo_b):
		"""
		Convert spatial Fock matrices to spin-orbital Fock matrix.
		Equivalent to the Fortran subroutine spatial_to_spin_fock.
		
		Parameters:
		-----------
		Fmo_a : np.ndarray
			Alpha Fock matrix, shape (n_spatial, n_spatial)
		Fmo_b : np.ndarray
			Beta Fock matrix, shape (n_spatial, n_spatial)
			
		Returns:
		--------
		Fmo_spin : np.ndarray
			Spin-orbital Fock matrix, shape (n_spin, n_spin)
		"""
		n_spatial = Fmo_a.shape[0]
		n_spin = 2 * n_spatial
		
		# Initialize spin Fock matrix
		Fmo_spin = np.zeros((n_spin, n_spin), dtype=np.float64)
		
		# Helper function for Kronecker delta
		def kronecker_delta(i, j):
			return 1 if i == j else 0
		
		# Convert 
		for p in range(n_spin):
			for q in range(n_spin):
				
				# Get spatial indices (0-based)
				spatial_p = p // 2
				spatial_q = q // 2
				
				# Get spin components
				spin_p = p % 2  # 0=alpha, 1=beta
				spin_q = q % 2
				
				# Apply Kronecker delta condition
				spin_factor = kronecker_delta(spin_p, spin_q)
				
				if spin_factor == 0:
					Fmo_spin[p, q] = 0.0
					continue
				
				# Get the appropriate Fock matrix element
				if spin_p == 0:  # alpha spin
					Fmo_spin[p, q] = Fmo_a[spatial_p, spatial_q]
				else:  # beta spin
					Fmo_spin[p, q] = Fmo_b[spatial_p, spatial_q]
		
		return Fmo_spin

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


	# Test for spatial_to_spin_eri optimization method
	def spatial_2_spin_eri_optimized(self, eri_aaaa, eri_aabb, eri_bbbb):
		"""
		Optimized conversion of spatial ERIs to spin-orbital ERIs.
		"""
		n_spatial = eri_aaaa.shape[0]
		n_spin = 2 * n_spatial
		
		eri_spin = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.float64)
		
		# Fill in the blocks directly
		# (αα|αα)
		eri_spin[0::2, 0::2, 0::2, 0::2] = eri_aaaa
		# (ββ|ββ)
		eri_spin[1::2, 1::2, 1::2, 1::2] = eri_bbbb
		# (αα|ββ)
		eri_spin[0::2, 0::2, 1::2, 1::2] = eri_aabb
		# (ββ|αα)
		eri_spin[1::2, 1::2, 0::2, 0::2] = eri_aabb.transpose(2,3,0,1)
		# # (αβ|αβ)
		# eri_spin[0::2, 1::2, 0::2, 1::2] = eri_aabb.transpose(0,2,1,3)
		# # (βα|βα)
		# eri_spin[1::2, 0::2, 1::2, 0::2] = eri_aabb.transpose(1,3,0,2)
		# Other combinations are zero by spin conservation
		
		return eri_spin

	def spatial_to_spin_fock_optimized(self, Fmo_a, Fmo_b):
		n_spatial = Fmo_a.shape[0]
		n_spin = 2 * n_spatial
		
		# Initialize spin Fock matrix
		Fmo_spin = np.zeros((n_spin, n_spin), dtype=np.float64)
		
		# Alpha block (even rows, even columns)
		Fmo_spin[0::2, 0::2] = Fmo_a
		
		# Beta block (odd rows, odd columns)
		Fmo_spin[1::2, 1::2] = Fmo_b
		
		return Fmo_spin
	

	def MP2_energy(self, mo_coeffs, Fmo) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
		"""
		MP2 correlation energy for unrestricted orbitals 

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


			# # Assert if canonical Fock is used,
			# 	# Check that Fock matrices are diagonal, 
			# 	# and that diagonal elements match MO energies from UHF
			# if self.use_canonical_Fock == True:
			# 	# Alpha
			# 	off_diag_a = Fmo_a - np.diag(np.diag(Fmo_a))
			# 	assert np.allclose(off_diag_a, 0.0, atol=1e-10), "Alpha Fock matrix is not diagonal for canonical orbitals!"
			# 	diag_a = np.diag(Fmo_a)
			# 	uhf_mo_energies_a = self.uhf.mo_energy[0]
			# 	assert np.allclose(diag_a, uhf_mo_energies_a, atol=1e-10), "Alpha Fock diagonal elements do not match UHF MO energies!"
			# 	# Beta
			# 	off_diag_b = Fmo_b - np.diag(np.diag(Fmo_b))
			# 	assert np.allclose(off_diag_b, 0.0, atol=1e-10), "Beta Fock matrix is not diagonal for canonical orbitals!"
			# 	diag_b = np.diag(Fmo_b)
			# 	uhf_mo_energies_b = self.uhf.mo_energy[1]
			# 	assert np.allclose(diag_b, uhf_mo_energies_b, atol=1e-10), "Beta Fock diagonal elements do not match UHF MO energies!"


			# print()
			# print("#### Fock and MO sizes ####")
			# print("Fock matrices", np.size(Fmo_a), np.size(Fmo_b))
			# print("MO Coefficients", np.size(mo_coeffs[0]), np.size(mo_coeffs[1]))
			# print("AO Fock matrices", np.size(self.Fao[0]), np.size(self.Fao[1]))
			# print()

			# 	# Convert Fock matrix to spin orbitals
			# t0 = time.time()
			# Fmo_spin = self.spatial_to_spin_fock(Fmo_a, Fmo_b)
			# print("Time for spatial_to_spin_fock: ", time.time() - t0)
			# 	# Test optimized version
			# t1 = time.time()
			Fmo_spin_opt = self.spatial_to_spin_fock_optimized(Fmo_a, Fmo_b)
			# print("Time for spatial_to_spin_fock_optimized: ", time.time() - t1)
			# 	# Check that both methods give the same result
			# Fock_diff = np.max(np.abs(Fmo_spin - Fmo_spin_opt))
			# assert np.allclose(Fmo_spin, Fmo_spin_opt, atol=1e-10), "Optimized spin-orbital Fock conversion does not match original! Max diff: {}".format(Fock_diff)
				
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


		# Check if canonical Fock is used,
		# 	# Check that Fock matrix is diagonal,
		# if self.use_canonical_Fock == True:
		# 	# iterate through each row and check if the right elements is the only non-zero
		# 	for i in range(Fmo_spin.shape[0]):
		# 		# Get how many non-zero elements are in the row

		# 		# 	# Print 5 largerst non-diagonal element for debugging
		# 		# non_diag_elements = np.abs(Fmo_spin[i, :]).copy()
		# 		# non_diag_elements[i] = 0.0  # Exclude diagonal
		# 		# largest_non_diag = np.sort(non_diag_elements)[-5:]
		# 		# print(f"Largest 5 non-diagonal elements in row {i}: {largest_non_diag}")

		# 		# Get relative tolerance for zero comparison
		# 		atol_zero_compare = 1e-6
		# 		non_zero_elements = np.count_nonzero(np.abs(Fmo_spin[i, :]) > atol_zero_compare)

		# 		# There should be only one non-zero element (the diagonal)
		# 		assert non_zero_elements == 1, f"Row {i} of spin-orbital Fock matrix has {non_zero_elements} non-zero elements, expected 1 for canonical orbitals!"

		"""
		CH2 6-31G UHF canonical Fock matrix non-diagonal elements for debugging:
		Largest 5 non-diagonal elements in row 0: [7.7624e-10 1.4163e-08 1.5221e-08 2.3790e-08 2.3861e-08]
		Largest 5 non-diagonal elements in row 1: [5.1300e-09 8.8853e-09 1.1574e-08 1.4719e-08 1.4747e-08]
		Largest 5 non-diagonal elements in row 2: [1.1408e-08 1.4672e-08 2.0790e-08 2.3861e-08 3.6146e-08]
		Largest 5 non-diagonal elements in row 3: [1.1574e-08 1.2642e-08 2.4491e-08 8.2406e-08 9.1904e-08]
		Largest 5 non-diagonal elements in row 4: [9.9920e-16 1.2768e-15 1.3455e-08 4.4543e-08 4.8668e-08]
		Largest 5 non-diagonal elements in row 5: [3.3307e-16 3.8858e-16 2.2867e-08 1.0911e-07 1.1093e-07]
		Largest 5 non-diagonal elements in row 6: [1.4672e-08 5.0825e-08 6.0558e-08 6.0712e-08 1.3032e-07]
		Largest 5 non-diagonal elements in row 7: [5.1300e-09 1.6473e-08 3.4827e-08 6.5534e-08 6.8831e-08]
		Largest 5 non-diagonal elements in row 8: [8.4025e-17 8.8235e-17 9.6345e-17 1.4850e-16 5.8291e-08]
		Largest 5 non-diagonal elements in row 9: [6.6218e-17 1.3196e-16 1.6092e-16 1.9609e-16 6.7883e-08]
		Largest 5 non-diagonal elements in row 10: [6.1482e-09 7.0144e-09 1.5221e-08 2.5759e-08 6.0712e-08]
		Largest 5 non-diagonal elements in row 11: [8.8853e-09 2.3632e-08 3.4538e-08 6.8831e-08 9.1904e-08]
		Largest 5 non-diagonal elements in row 12: [6.7030e-15 9.5202e-15 2.0835e-08 4.4543e-08 4.7427e-08]
		Largest 5 non-diagonal elements in row 13: [3.5527e-15 5.8287e-15 3.1153e-08 4.3840e-08 1.1093e-07]
		Largest 5 non-diagonal elements in row 14: [4.2188e-15 4.9127e-15 1.2999e-08 2.0835e-08 4.8668e-08]
		Largest 5 non-diagonal elements in row 15: [2.4147e-15 3.1641e-15 2.2867e-08 3.1153e-08 4.0886e-08]
		Largest 5 non-diagonal elements in row 16: [1.1408e-08 1.4072e-08 1.4163e-08 2.8352e-08 1.3032e-07]
		Largest 5 non-diagonal elements in row 17: [1.4719e-08 2.3632e-08 2.4491e-08 2.9977e-08 6.5534e-08]
		Largest 5 non-diagonal elements in row 18: [5.0897e-17 5.8702e-17 9.4913e-17 1.1059e-16 5.8291e-08]
		Largest 5 non-diagonal elements in row 19: [9.7364e-17 1.3729e-16 1.5080e-16 1.9896e-16 6.7883e-08]
		Largest 5 non-diagonal elements in row 20: [6.1482e-09 7.0119e-09 2.8352e-08 3.6146e-08 6.0558e-08]
		Largest 5 non-diagonal elements in row 21: [3.1292e-09 3.7928e-09 5.7590e-09 1.2642e-08 1.6473e-08]
		Largest 5 non-diagonal elements in row 22: [1.4072e-08 2.0790e-08 2.3790e-08 2.5759e-08 5.0825e-08]
		Largest 5 non-diagonal elements in row 23: [1.4747e-08 2.9977e-08 3.4538e-08 3.4827e-08 8.2406e-08]
		Largest 5 non-diagonal elements in row 24: [5.7478e-15 6.0080e-15 1.2999e-08 1.3455e-08 4.7427e-08]
		Largest 5 non-diagonal elements in row 25: [2.5564e-15 2.6669e-15 4.0886e-08 4.3840e-08 1.0911e-07]
		"""

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

			#PySCF stores 2e integrals in chemists' notation: (ij|kl) = <ik|jl> in physicists' notation.
				# (alpha alpha | alpha alpha) integrals
		eri_aaaa = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[0], mo_coeffs[0], mo_coeffs[0], mo_coeffs[0]], compact=False)
				# (beta beta | beta beta) integrals
		eri_bbbb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[1], mo_coeffs[1], mo_coeffs[1], mo_coeffs[1]], compact=False)
				# (alpha alpha | beta beta) integrals
		eri_aabb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[0], mo_coeffs[0], mo_coeffs[1], mo_coeffs[1]], compact=False)

		    # reshape AO->MO (chemists' notation)
		norb_alpha, norb_beta = mo_coeffs[0].shape[1], mo_coeffs[1].shape[1]
		eri_aaaa = eri_aaaa.reshape((norb_alpha, norb_alpha, norb_alpha, norb_alpha))
		eri_bbbb = eri_bbbb.reshape((norb_beta,  norb_beta,  norb_beta,  norb_beta))
		eri_aabb = eri_aabb.reshape((norb_alpha, norb_alpha, norb_beta,  norb_beta))


			# Manual assembly of spin-orbital integrals from spatial blocks
				# allocate spin-orbital ERI and Fock
		# t0 = time.time()
		# eri_spin = self.spatial_to_spin_eri(eri_aaaa, eri_aabb, eri_bbbb)
		# print("Time for spatial_to_spin_eri: ", time.time() - t0)

		# 		# Test optimized version
		# t1 = time.time()
		eri_spin_opt = self.spatial_2_spin_eri_optimized(eri_aaaa, eri_aabb, eri_bbbb)
		# print("Time for spatial_2_spin_eri_optimized: ", time.time() - t1)

		# 		# Check that both methods give the same result
		# eri_spin_diff = np.max(np.abs(eri_spin - eri_spin_opt))
		# print("Max difference between eri_spin and eri_spin_opt: ", eri_spin_diff)
		# assert np.allclose(eri_spin, eri_spin_opt, atol=1e-10), "Optimized spin-orbital ERI conversion does not match original! Max diff: {}".format(eri_spin_diff)

			# Use optimized version
		eri_spin = eri_spin_opt


		# Sanity checks 
			# Check that integrals are finite
		# assert np.all(np.isfinite(eri_spin)), "Spin-orbital integrals contain non-finite values!"
			# Check that integrals are not all zero
		# assert np.count_nonzero(eri_spin) > 0, "Spin-orbital integrals are all zero!"
			
			# Convert from chemist's to physicist's notation
				# Chemist's: (pq|rs), Physicist's: <pq|rs> = (pr|qs)
		eri_phys = eri_spin.transpose(0,2,1,3)  # Swap indices 1 and 2

		# Sanity checks
			# Check that integrals are finite
		# assert np.all(np.isfinite(eri_phys)), "Physicist's notation integrals contain non-finite values!"
			# Check that integrals are not all zero
		# assert np.count_nonzero(eri_phys) > 0, "Physicist's notation integrals are all zero!"

			# Antisymmetrized integrals in physicist's notation
				# <pq||rs> = <pq|rs> - <pq|sr>
		eri_as = eri_phys - eri_phys.transpose(0,1,3,2)

		# Sanity checks 
			# Check that integrals are finite
		# assert np.all(np.isfinite(eri_as)), "Antisymmetrized integrals contain non-finite values!"
			# Check that integrals are not all zero
		# assert np.count_nonzero(eri_as) > 0, "Antisymmetrized integrals are all zero!"	















		# ii) Compute MP1 amplitudes (spin-orbital)
		def compute_mp1_amplitudes(self, eps, eri_as):
			n_spin = self.tot_num_spin_orbs
			MP1_amplitudes = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.float64)

			# Full computation with spin considerations
			for a in self.active_inocc_indices:
				eps_a = eps[a]
				for b in self.active_inocc_indices:
					eps_b = eps[b]
					for i in self.active_occ_indices:
						eps_i = eps[i]
						for j in self.active_occ_indices:
							eps_j = eps[j]
				
							
							if self.use_canonical_Fock:
								# Energy denominator: ε_a + ε_b - ε_i - ε_j
								denominator = eps_a + eps_b - eps_i - eps_j
								# Antisymmetrized integral: <ab||ij> = <ab|ij> - <ab|ji>
								integral = eri_as[a, b, i, j]  # <ab||ij>

							# else:
							# 	denominator = Fmo_spin[a, a] + Fmo_spin[b, b] - Fmo_spin[i, i] - Fmo_spin[j, j] - 2 * Fmo_spin[a, b] + 2 * Fmo_spin[i, j]  # Non-canonical denominator with off-diagonal Fock elements

							# 	integral_eri = eri_as[a, b, i, j]
							# 	sum_ = 0.0
								
							# 	for C in self.active_inocc_indices:
							# 		for D in self.active_inocc_indices:
							# 			if C <= D:
							# 				continue
							# 			if C != a and D != b:
							# 				t_cdij = MP1_amplitudes[C, D, i, j]
											
							# 				f_AC = Fmo_spin[a, C] if b == D else 0.0
							# 				f_BD = Fmo_spin[b, D] if a == C else 0.0
							# 				f_AD = Fmo_spin[a, D] if b == C else 0.0
							# 				f_BC = Fmo_spin[b, C] if a == D else 0.0

							# 				parentheses_delta_a = 0.0
							# 				if a == C and b == D:
							# 					parentheses_delta_a = 1.0
												
							# 				parentheses_delta_b = 0.0
							# 				if a == D and b == C:
							# 					parentheses_delta_b = -1.0

							# 				f_II = Fmo_spin[i, i] * (parentheses_delta_a - parentheses_delta_b)
							# 				f_JJ = Fmo_spin[j, j] * (parentheses_delta_a - parentheses_delta_b)
							# 				f_IJ = 2*Fmo_spin[i, j] * (parentheses_delta_a - parentheses_delta_b)

							# 				parentheses_fock = f_AC + f_BD - f_AD - f_BC - f_II - f_JJ - f_IJ

							# 				sum_ -= t_cdij * parentheses_fock
								

							# 	integral = integral_eri - sum_
							
							# MP1 amplitude: t_{ij}^{ab} = -<ab||ij> / (ε_a + ε_b - ε_i - ε_j)
							MP1_amplitudes[a, b, i, j] = -1.0 * integral / denominator
		
			return MP1_amplitudes
		
		# t0 = time.time()
		# MP1_amplitudes = compute_mp1_amplitudes(self, eps, eri_as)
		# print("Time for full MP1 amplitude computation: ", time.time() - t0)

		# Optimized computation of MP1 amplitudes only in active space
		def compute_mp1_amplitudes_optimized(self, eps, eri_as):
			n_spin = self.tot_num_spin_orbs
			
			# Get active indices
			occ_idx = self.active_occ_indices
			vir_idx = self.active_inocc_indices
			
			nocc = len(occ_idx)
			nvir = len(vir_idx)
			
			# Pre-extract energies for active orbitals
			eps_occ = eps[occ_idx]
			eps_vir = eps[vir_idx]
			
			# Create energy denominator tensor using broadcasting
			# Denominator: ε_a + ε_b - ε_i - ε_j
			# We'll create separate tensors for each combination
			eps_vir_a = eps_vir.reshape(-1, 1, 1, 1)        # shape: (nvir, 1, 1, 1)
			eps_vir_b = eps_vir.reshape(1, -1, 1, 1)        # shape: (1, nvir, 1, 1)
			eps_occ_i = eps_occ.reshape(1, 1, -1, 1)        # shape: (1, 1, nocc, 1)
			eps_occ_j = eps_occ.reshape(1, 1, 1, -1)        # shape: (1, 1, 1, nocc)
			
			# Compute denominator tensor
			denominator = eps_vir_a + eps_vir_b - eps_occ_i - eps_occ_j
			
			# Extract the relevant block of antisymmetrized integrals
			# Only need vir-vir-occ-occ block
			integral_block = eri_as[vir_idx][:, vir_idx][:, :, occ_idx][:, :, :, occ_idx]
			
			# Compute MP1 amplitudes for active block only
			MP1_block = -integral_block / denominator
			
			# Create full tensor with zeros
			MP1_amplitudes = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.float64)
			
			# Insert the computed block into the full tensor
			# Using numpy's advanced indexing
			idx_grid = np.ix_(vir_idx, vir_idx, occ_idx, occ_idx)
			MP1_amplitudes[idx_grid] = MP1_block
			
			return MP1_amplitudes

		# t1 = time.time()
		MP1_amplitudes_opt = compute_mp1_amplitudes_optimized(self, eps, eri_as)	
		# print("Time for optimized MP1 amplitude computation: ", time.time() - t1)

		# 	# Check optimized amplitudes match full computation
		# amplitude_diff = np.max(np.abs(MP1_amplitudes_full - MP1_amplitudes))
		# assert np.allclose(MP1_amplitudes_full, MP1_amplitudes, atol=1e-10), "Optimized MP1 amplitudes do not match full computation! Max diff: {}".format(amplitude_diff)

		# Set optimized version
		MP1_amplitudes = MP1_amplitudes_opt

		# Sanity checks
			# Check that amplitudes are finite
		assert np.all(np.isfinite(MP1_amplitudes)), "MP1 amplitudes contain non-finite values!"
			# Check that amplitudes are not all zero
		assert np.count_nonzero(MP1_amplitudes) > 0, "MP1 amplitudes are all zero!"
			# Check amplitude antisymmetry: 
				# t_ij^{ab} = t_ji^{ba}
		assert np.allclose(MP1_amplitudes, MP1_amplitudes.transpose(1,0,3,2), atol=1e-10), "MP1 amplitudes do not satisfy antisymmetry t_ij^ab = t_ji^ba!"





		# iii) Compute MP2 correlation energy (spin-orbital indices)
		def compute_mp2_energy(self, eps, Fmo_spin, eri_as, MP1_amplitudes):
			J_2 = 0.0
			
			# Create lists for easier indexing
			occ_indices = self.active_occ_indices
			virt_indices = self.active_inocc_indices
			
			for idx_i, i in enumerate(occ_indices):
				eps_i = eps[i]
				
				for idx_j, j in enumerate(occ_indices):
					if i <= j:  # i > j restriction
						continue
						
					eps_j = eps[j]
					J_ij = 0.0
					
					# First term: Σ_{a>b} Σ_{c>d} t_ij^{ab} t_ij^{cd} [...]
					term1 = 0.0
					
					# Use symmetry: sum over all a,b,c,d but use independent sums
					# This is more efficient than nested loops
					for idx_a, a in enumerate(virt_indices):					
						for idx_b, b in enumerate(virt_indices):
							if a <= b:  # a > b restriction
								continue

							t_abij = MP1_amplitudes[a, b, i, j]
							
							for idx_c, c in enumerate(virt_indices):
								
								for idx_d, d in enumerate(virt_indices):
									if c <= d:  # c >d restriction
										continue
										
									t_cdij = MP1_amplitudes[c, d, i, j]
									
									# Compute the bracket term
									bracket = 0.0
									
									# f_ac δ_bd
									if b == d:
										# Non-canonical Fock
										if self.use_canonical_Fock == False:
											bracket += Fmo_spin[a, c]
										# Canonical Fock
										else:
											bracket += Fmo_spin[a, c]
											# if a == c:
											# 	bracket += eps[a]
									
									# f_bd δ_ac
									if a == c:
										# Non-canonical Fock
										if self.use_canonical_Fock == False:
											bracket += Fmo_spin[b, d]
										# Canonical Fock
										else:
											bracket += Fmo_spin[b, d]
											# if b == d:
											# 	bracket += eps[b]
									
									# - f_ad δ_bc
									if b == c:
										# Non-canonical Fock
										if self.use_canonical_Fock == False:
											bracket -= Fmo_spin[a, d]
										# Canonical Fock
										else:
											bracket -= Fmo_spin[a, d]
											# if a == d:
											# 	bracket -= eps[a]
									
									# - f_bc δ_ad
									if a == d:
										# Non-canonical Fock
										if self.use_canonical_Fock == False:
											bracket -= Fmo_spin[b, c]
										# Canonical Fock
										else:
											bracket -= Fmo_spin[b, c]
											# if b == c:
											# 	bracket -= eps[b]
									
									# - (ε_i + ε_j)(δ_ac δ_bd - δ_ad δ_bc)
									if a == c and b == d:
										if self.use_canonical_Fock == False:
											bracket -= (Fmo_spin[i, i] + Fmo_spin[j, j])
										else:
											bracket -= (eps_i + eps_j)
									if a == d and b == c:
										if self.use_canonical_Fock == False:
											bracket += (Fmo_spin[i, i] + Fmo_spin[j, j])
										else:
											bracket += (eps_i + eps_j)
									
									term1 += t_abij * t_cdij * bracket
					
					# Second term: 2 Σ_{a>b} t_ij^{ab} <ab|ij>
					term2 = 0.0
					
					for idx_a, a in enumerate(virt_indices):
						
						for idx_b, b in enumerate(virt_indices):
							if b <= a:  # a > b restriction
								continue
								
							t_abij = MP1_amplitudes[a, b, i, j]
							integral = eri_as[a, b, i, j]  
							
							term2 += 2.0 * t_abij * integral

					J_ij = term1 + term2 
					J_2 += J_ij
			return J_2
		
		J_2 = compute_mp2_energy(self, eps, Fmo_spin, eri_as, MP1_amplitudes)

		def compute_mp2_energy_optimized(self, eps, Fmo_spin, eri_as, MP1_amplitudes):
			# Get orbital indices
			occ_indices = self.active_occ_indices
			virt_indices = self.active_inocc_indices
			
			nocc = len(occ_indices)
			nvir = len(virt_indices)
			
			if nocc == 0 or nvir == 0:
				return 0.0
			
			# Convert to numpy arrays for faster indexing
			occ = np.array(occ_indices)
			vir = np.array(virt_indices)
			
			J_2 = 0.0
			
			# Precompute all (a,b) and (c,d) pairs with a>b and c>d
			# Get indices of all a>b pairs
			a_idx, b_idx = np.triu_indices(nvir, k=1)  # upper triangle without diagonal
			n_ab_pairs = len(a_idx)
			
			# Actual orbital indices for virtual pairs
			a_vals = vir[a_idx]  # shape: (n_ab_pairs,)
			b_vals = vir[b_idx]
			
			# For non-canonical case, precompute Fock matrix elements
			if not self.use_canonical_Fock:
				# Extract virtual-virtual block of Fock matrix
				F_virt = Fmo_spin[vir][:, vir]  # shape: (nvir, nvir)
				
				# Map virtual indices to positions in F_virt
				vir_to_idx = {v: i for i, v in enumerate(vir)}
				a_idx_pos = np.array([vir_to_idx[a] for a in a_vals])  # shape: (n_ab_pairs,)
				b_idx_pos = np.array([vir_to_idx[b] for b in b_vals])
				
				# Create arrays for broadcasting
				# For all combinations of (a,b) and (c,d) pairs
				a_pos_all = a_idx_pos[:, None]  # shape: (n_ab_pairs, 1)
				b_pos_all = b_idx_pos[:, None]  # shape: (n_ab_pairs, 1)
				c_pos_all = a_idx_pos[None, :]  # shape: (1, n_ab_pairs)
				d_pos_all = b_idx_pos[None, :]  # shape: (1, n_ab_pairs)
				
				# Create masks for Kronecker deltas
				delta_ac = (a_vals[:, None] == a_vals[None, :])  # shape: (n_ab_pairs, n_ab_pairs)
				delta_bd = (b_vals[:, None] == b_vals[None, :])
				delta_ad = (a_vals[:, None] == b_vals[None, :])
				delta_bc = (b_vals[:, None] == a_vals[None, :])
			
			# Loop over all i>j pairs
			# Outer loop: i from first to last
			for idx_i in range(nocc):
				i = occ[idx_i]
				eps_i = eps[i]
				
				# Inner loop: j from i-1 down to 0 (for i > j)
				for idx_j in range(idx_i):
					j = occ[idx_j]
					eps_j = eps[j]
					
					# Get all t_ij^{ab} for a>b
					t_abij = MP1_amplitudes[a_vals, b_vals, i, j]  # shape: (n_ab_pairs,)
					
					# === TERM 2: 2 Σ_{a>b} t_ij^{ab} <ab|ij> ===
					integrals = eri_as[a_vals, b_vals, i, j]  # shape: (n_ab_pairs,)
					term2 = 2.0 * np.dot(t_abij, integrals)
					
					# === TERM 1: Σ_{a>b} Σ_{c>d} t_ij^{ab} t_ij^{cd} bracket ===
					term1 = 0.0
					
					if self.use_canonical_Fock:
						# Canonical case: Fock matrix is diagonal
						# The bracket simplifies to: (ε_a + ε_b - ε_i - ε_j) when (a,b) = (c,d)
						# And all other terms are zero because δ_ac requires a=c, δ_bd requires b=d, etc.
						
						# Only non-zero contributions are when (a,b) = (c,d)
						# So term1 = Σ_{a>b} t_abij² * (ε_a + ε_b - ε_i - ε_j)
						eps_a = eps[a_vals]
						eps_b = eps[b_vals]
						eps_ab = eps_a + eps_b - eps_i - eps_j
						term1 = np.sum(t_abij**2 * eps_ab)
						
					else:
						# Non-canonical case: need to compute full bracket
						# We'll use vectorization to compute all (a,b),(c,d) pairs at once
						
						# Reshape t_abij for broadcasting
						t_row = t_abij[:, None]  # shape: (n_ab_pairs, 1)
						t_col = t_abij[None, :]  # shape: (1, n_ab_pairs)
						
						# Initialize bracket matrix
						bracket = np.zeros((n_ab_pairs, n_ab_pairs))
						
						# 1. f_ac δ_bd
						# Only when b == d
						mask_bd = delta_bd
						if np.any(mask_bd):
							# Get f_ac = Fmo_spin[a, c] for all a,c where b == d
							f_ac_vals = F_virt[a_pos_all, c_pos_all]
							bracket += f_ac_vals * mask_bd
						
						# 2. f_bd δ_ac
						# Only when a == c
						mask_ac = delta_ac
						if np.any(mask_ac):
							# Get f_bd = Fmo_spin[b, d] for all b,d where a == c
							f_bd_vals = F_virt[b_pos_all, d_pos_all]
							bracket += f_bd_vals * mask_ac
						
						# 3. -f_ad δ_bc
						# Only when b == c
						mask_bc = delta_bc
						if np.any(mask_bc):
							# Get f_ad = Fmo_spin[a, d] for all a,d where b == c
							f_ad_vals = F_virt[a_pos_all, d_pos_all]
							bracket -= f_ad_vals * mask_bc
						
						# 4. -f_bc δ_ad
						# Only when a == d
						mask_ad = delta_ad
						if np.any(mask_ad):
							# Get f_bc = Fmo_spin[b, c] for all b,c where a == d
							f_bc_vals = F_virt[b_pos_all, c_pos_all]
							bracket -= f_bc_vals * mask_ad
						
						# 5. - (ε_i + ε_j)(δ_ac δ_bd - δ_ad δ_bc)
						# Note: We use Fmo_spin diagonal for non-canonical case
						eps_ij_sum = Fmo_spin[i, i] + Fmo_spin[j, j]
						
						# δ_ac δ_bd term
						mask_ac_bd = delta_ac & delta_bd
						bracket[mask_ac_bd] -= eps_ij_sum
						
						# δ_ad δ_bc term
						mask_ad_bc = delta_ad & delta_bc
						bracket[mask_ad_bc] += eps_ij_sum
						
						# Compute term1: sum over all (a,b) and (c,d)
						# t_abij * bracket * t_cdij
						term1 = np.sum(t_row * bracket * t_col)
					
					# Add to total
					J_2 += term1 + term2
			
			return J_2
			
		# Get MP2 energy optimized
		# J_2_opt = compute_mp2_energy_optimized(self, eps, Fmo_spin, eri_as, MP1_amplitudes)

		# Check optimized MP2 energy matches full computation
		# energy_diff = np.abs(J_2 - J_2_opt)
		# assert energy_diff < 1e-10, "Optimized MP2 energy does not match full computation! Diff: {}".format(energy_diff)

		# Set optimized version
		# J_2 = J_2_opt

		print(f"[{len(self.active_inocc_indices)}/{len(self.active_inocc_indices + self.inactive_indices)}]: Computed MP2 correlation energy (spin-orbital): ", J_2)

		# Sanity checks
			# Check that MP2 correlation energy is finite
		assert np.isfinite(J_2), "MP2 correlation energy is not finite!"
			# Check that MP2 correlation energy is not positive
		# assert J_2 <= 0.0, "MP2 correlation energy is not negative!"


		return J_2, MP1_amplitudes, eri_spin, eri_phys, eri_as, Fmo_spin	

	def orbital_optimization(self, mo_coeffs, MP1_amplitudes, eri_spin, eri_phys, eri_as, Fmo_spin) -> np.ndarray:

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

		# Is MP2 energy reached?
			# If full space is used, return original orbitals
		# if len(self.active_inocc_indices) == self.tot_num_spin_orbs - int(self.nelec):
		# 	return mo_coeffs



		
		# Precompute D_ab as 2D array
		def compute_D_ab(self, MP1_amplitudes):
			n_active_virt = len(self.active_inocc_indices)
			D_ab = np.zeros((n_active_virt, n_active_virt), dtype=np.float64)

			for idx_A, A in enumerate(self.active_inocc_indices):
				for idx_B, B in enumerate(self.active_inocc_indices):
					s = 0.0
					for I in self.active_occ_indices:
						for J in self.active_occ_indices:
							if I <= J:
								continue
							for C in self.active_inocc_indices:
								# Use antisymmetric amplitudes
								t_ACIJ = MP1_amplitudes[A, C, I, J]
								t_BCIJ = MP1_amplitudes[B, C, I, J]
								s += t_ACIJ * t_BCIJ

					D_ab[idx_A, idx_B] = s
			return D_ab
		
		# t0 = time.time()
		# D_ab = compute_D_ab(self, MP1_amplitudes)
		# print("Time for D_ab computation: ", time.time() - t0)

		def compute_D_ab_optimized(self, MP1_amplitudes):
			"""
			Version that only considers I<J pairs as in original code.
			"""
			occ_idx = self.active_occ_indices
			vir_idx = self.active_inocc_indices
			
			nocc = len(occ_idx)
			nvir = len(vir_idx)
			
			# Extract block
			idx_grid = np.ix_(vir_idx, vir_idx, occ_idx, occ_idx)
			t_block = MP1_amplitudes[idx_grid]  # shape: (nvir, nvir, nocc, nocc)
			
			# Get indices where I < J
			i_idx, j_idx = np.tril_indices(nocc, k=-1)
			
			# Extract only I<J pairs
			# shape: (nvir, nvir, n_pairs)
			t_unique = t_block[:, :, i_idx, j_idx]
			
			# Compute D_ab
			D_ab = np.einsum('acp,bcp->ab', t_unique, t_unique, optimize=True)
			
			return D_ab
		
		 	# Test optimized version
		# t0 = time.time()
		D_ab_optimized = compute_D_ab_optimized(self, MP1_amplitudes)
		# print("Time for optimized D_ab computation: ", time.time() - t0)
		 	# Check that both methods give the same result
		# D_ab_diff = np.max(np.abs(D_ab - D_ab_optimized))
		# assert np.allclose(D_ab, D_ab_optimized, atol=1e-10), "Optimized D_ab does not match original! Max diff: {}".format(D_ab_diff)
		 	# Use optimized version
		D_ab = D_ab_optimized







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
			
			# Compute gradient G
		def compute_gradient(self, MP1_amplitudes, eri_as, D_ab, Fmo_spin):
			G = np.zeros((len(self.active_inocc_indices), len(self.inactive_indices)), dtype=np.float64)

			for idx_A, A in enumerate(self.active_inocc_indices):
				for idx_E, E in enumerate(self.inactive_indices):
					term1 = 0.0
					
					# Term1: 2 * Σ_{i>j} Σ_{b} t_ij^{ab} <eb|ij>
					for I in self.active_occ_indices:
						for J in self.active_occ_indices:
							if I <= J:
								continue
							
							for idx_B, B in enumerate(self.active_inocc_indices):
								t_ABIJ = MP1_amplitudes[A, B, I, J]
							
								# Choose correct integral notation
									# Use antisymmetrized integral <eb||ij>
								integral = -eri_as[E, B, I, J]
								
								term1 += 2.0 * t_ABIJ * integral
					
					# Term2: 2 * Σ_{b} D_ab f_{eb}
					term2 = 0.0
					for idx_B, B in enumerate(self.active_inocc_indices):
						# In non-canonical fock matrix, f_eb = Fmo_spin[E, B]
						if self.use_canonical_Fock == False:
							# Get overlap S_EB
							# S_EB = 0.0
							# for p in range(self.tot_num_spin_orbs):
							# 	S_EB += mo_coeffs[p, E] * mo_coeffs[p, B]
							# f_EB = self.eps[E] * S_EB
							f_EB = Fmo_spin[E, B]
							term2 += 2.0 * D_ab[idx_A, idx_B] * f_EB
						# In canonical fock matrix, f_eb = eps_e δ_eb
						else:
							# if E == B:
								# Get overlap S_EB
								# S_EB = 0.0
								# for p in range(self.tot_num_spin_orbs):
								# 	S_EB += mo_coeffs[p, E] * mo_coeffs[p, B]
								# f_EB = self.eps[E] * S_EB
							term2 += 2.0 * D_ab[idx_A, idx_B] * Fmo_spin[E, B]

					# # term3:
					# term3 = 0.0
					# if self.use_canonical_Fock == False:
					# 	for I in self.active_occ_indices:
					# 		for J in self.active_occ_indices:
					# 			if I <= J:
					# 				continue
					# 			for idx_B, B in enumerate(self.active_inocc_indices):
					# 				for K in self.active_occ_indices:
					# 					for L in self.active_occ_indices:
					# 						if K <= L:
					# 							continue
					# 						for idx_C, C in enumerate(self.active_inocc_indices):
					# 							t_CBKL = MP1_amplitudes[C, B, K, L]
												
					# 							# \partial t_CBKL/\partial f_IJ = ...
					# 							t_CBKL = 2*t_CBKL/(Fmo_spin[C, C] + Fmo_spin[B, B] - Fmo_spin[K, K] - Fmo_spin[L, L] - 2*Fmo_spin[C,B] - 2*Fmo_spin[K,L])
					# 								# plus some recursive terms that i omit

					# 							term3 += -2.0 * t_CBKL * D_ab[idx_C, idx_B] * Fmo_spin[E,B] * Fmo_spin[I, J]  

					# Combine terms into gradient
					G[idx_A, idx_E] = term1 + term2 # + term3
			return G
		
		# t0 = time.time()
		G = compute_gradient(self, MP1_amplitudes, eri_as, D_ab, Fmo_spin)
		# print("Time for gradient computation: ", time.time() - t0)

		def compute_gradient_optimized(self, MP1_amplitudes, eri_as, D_ab, Fmo_spin):
			occ = np.array(self.active_occ_indices)
			vir = np.array(self.active_inocc_indices)
			inactive = np.array(self.inactive_indices)
			
			nocc = len(occ)
			nvir = len(vir)
			ninactive = len(inactive)
			
			# Early return if no inactive orbitals
			if ninactive == 0:
				return np.zeros((nvir), dtype=np.float64)

			# Get I<J pairs
			i_idx, j_idx = np.triu_indices(nocc, k=1)
			n_pairs = len(i_idx)
			
			# Create index arrays for all dimensions
			# For t_pairs: we need indices for all A, B, and each (I,J) pair
			A_idx = vir[:, None, None]  # shape: (nvir, 1, 1)
			B_idx = vir[None, :, None]  # shape: (1, nvir, 1)
			I_idx = occ[i_idx][None, None, :]  # shape: (1, 1, n_pairs)
			J_idx = occ[j_idx][None, None, :]  # shape: (1, 1, n_pairs)
			
			# Use broadcasting to get all indices
			A_idx_expanded = np.broadcast_to(A_idx, (nvir, nvir, n_pairs))
			B_idx_expanded = np.broadcast_to(B_idx, (nvir, nvir, n_pairs))
			I_idx_expanded = np.broadcast_to(I_idx, (nvir, nvir, n_pairs))
			J_idx_expanded = np.broadcast_to(J_idx, (nvir, nvir, n_pairs))
			
			# Flatten indices for indexing
			A_flat = A_idx_expanded.ravel()
			B_flat = B_idx_expanded.ravel()
			I_flat = I_idx_expanded.ravel()
			J_flat = J_idx_expanded.ravel()
			
			# Extract t_pairs using flat indices
			t_pairs = MP1_amplitudes[A_flat, B_flat, I_flat, J_flat].reshape(nvir, nvir, n_pairs)
			
			# For int_pairs: we need indices for all E, B, and each (I,J) pair
			E_idx = inactive[:, None, None]  # shape: (ninactive, 1, 1)
			B_idx_int = vir[None, :, None]  # shape: (1, nvir, 1)
			I_idx_int = occ[i_idx][None, None, :]  # shape: (1, 1, n_pairs)
			J_idx_int = occ[j_idx][None, None, :]  # shape: (1, 1, n_pairs)
			
			# Use broadcasting
			E_idx_expanded = np.broadcast_to(E_idx, (ninactive, nvir, n_pairs))
			B_idx_int_expanded = np.broadcast_to(B_idx_int, (ninactive, nvir, n_pairs))
			I_idx_int_expanded = np.broadcast_to(I_idx_int, (ninactive, nvir, n_pairs))
			J_idx_int_expanded = np.broadcast_to(J_idx_int, (ninactive, nvir, n_pairs))
			
			# Flatten
			E_flat = E_idx_expanded.ravel()
			B_int_flat = B_idx_int_expanded.ravel()
			I_int_flat = I_idx_int_expanded.ravel()
			J_int_flat = J_idx_int_expanded.ravel()
			
			# Extract int_pairs
			int_pairs = -eri_as[E_flat, B_int_flat, I_int_flat, J_int_flat].reshape(ninactive, nvir, n_pairs)
			
			# Compute term1
			term1 = 2.0 * np.einsum('abp,ebp->ae', t_pairs, int_pairs, optimize=True)
			
			# Term2: 2 * Σ_{b} D_ab f_{eb}
				# Non-canonical Fock matrix version
			if self.use_canonical_Fock == False:
				F_block = Fmo_spin[np.ix_(vir, inactive)]
				term2 = 2.0 * D_ab @ F_block
				# Canonical Fock matrix version
			else:
				eps_block = np.zeros((len(vir), len(inactive)), dtype=np.float64)
				for idx_B, B in enumerate(vir):
					for idx_E, E in enumerate(inactive):
						if B == E:
							eps_block[idx_B, idx_E] = self.eps[E]
				term2 = 2.0 * D_ab @ eps_block

			return term1 + term2
		
		 	# Test optimized version
		# t0 = time.time()
		# G_optimized = compute_gradient_optimized(self, MP1_amplitudes, eri_as, D_ab, Fmo_spin)
		# print("Time for optimized gradient computation: ", time.time() - t0)
		 	# Check that both methods give the same result
		# grad_diff = np.max(np.abs(G.flatten() - G_optimized.flatten()))
		# assert np.allclose(G.flatten(), G_optimized.flatten(), atol=1e-10), "Optimized gradient does not match original! Max diff: {}".format(grad_diff)
		 	# Use optimized version
		# G = G_optimized

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

		# Convergence check
		# max_grad = np.max(np.abs(G))
		# # print(f"  Max gradient element: {max_grad:.6e}")
		# if max_grad < 1e-6:
		# 	print("        WRANING: Gradient below threshold -> Orbitals are optimized!")






			# Compute Hessian H
		def compute_hessian(self, MP1_amplitudes, eri_as, D_ab, Fmo_spin):
			n_active_virt = len(self.active_inocc_indices)
			n_inactive_virt = len(self.inactive_indices)
					# Initialize Hessian
			H = np.zeros((n_active_virt * n_inactive_virt, 
						n_active_virt * n_inactive_virt), dtype=np.float64)

					# Build Hessian entries
			for idx_A, A in enumerate(self.active_inocc_indices):			
				for idx_E, E in enumerate(self.inactive_indices):
					row = idx_A * n_inactive_virt + idx_E
					
					for idx_B, B in enumerate(self.active_inocc_indices):					
						for idx_F, F in enumerate(self.inactive_indices):
							col = idx_B * n_inactive_virt + idx_F
							
							# ===== TERM 1: 2 * Σ_{i>j} t_ij^{ab} <ef|ij> =====
							term1 = 0.0

							for I in self.active_occ_indices:							
								for J in self.active_occ_indices:
									if I <= J:
										continue
									
									t_ABIJ = MP1_amplitudes[A, B, I, J]
									
									# Choose correct integral notation
										# Same-spin: use antisymmetrized integral <eb||ij>
										# Different-spin: simplifies to Coulomb integral <eb|ij> in physicist's notation
									integral = -eri_as[E, F, I, J]
									
									term1 += 2.0 * t_ABIJ * integral
							
							# ===== TERM 2: - Σ_{i>j} Σ_c [t_ij^{ac} <bc|ij> + t_ij^{bc} <ac|ij>] * δ_EF =====
							term2 = 0.0
							if E == F:  # δ_EF
								for I in self.active_occ_indices:								
									for J in self.active_occ_indices:
										if I <= J:
											continue
										
										temp2_sum = 0.0
										for C in self.active_inocc_indices:
											
											t_ACIJ = MP1_amplitudes[A, C, I, J]
											integral_BC = -eri_as[B, C, I, J]
										
											t_CBIJ = MP1_amplitudes[C, B, I, J]
											integral_AC = -eri_as[C, A, I, J]
											temp2_sum += t_ACIJ * integral_BC + t_CBIJ * integral_AC

										term2 -= temp2_sum
							
							# ===== TERMS 3 & 4 =====
							term3 = 0.0
							term4 = 0.0

								# Precomputed D_ab
							Dab = D_ab[idx_A, idx_B]
							
							if E == F:  # Term 3: Dab * (f_aa - f_bb) δ_EF
								# Non-canonical Fock
								if self.use_canonical_Fock == False:
									f_aa = Fmo_spin[A, A]
									f_bb = Fmo_spin[B, B]
								# Canonical Fock
								else:
									# f_aa = self.eps[A]
									# f_bb = self.eps[B]
									f_aa = Fmo_spin[A, A]
									f_bb = Fmo_spin[B, B]
								
								term3 = Dab * (f_aa - f_bb)

							if E != F:  # Term 4: Dab * f_ef (1- δ_EF)
								# Inactive-inactive block of Fock matrix
									# Should not be included in the canonical Fock case,
										# only canonical Fock in active space, a,b,c,d...
								f_ef = Fmo_spin[E, F]
								
								term4 = Dab * f_ef			
							
							# # ===== TERMS 5 =====
							# # These terms are zero for canonical orbitals, so we can skip them in that
							# term5 = 0.0
							# if self.use_canonical_Fock == False:
							# 	for I in self.active_occ_indices:
							# 		for J in self.active_occ_indices:
							# 			if I <= J:
							# 				continue
							# 			t_ABIJ = MP1_amplitudes[A, B, I, J]
							# 			f_IJ = Fmo_spin[I, J]
							# 			integral = -eri_as[E, F, I, J]

							# 			term5 += 2.0 * t_ABIJ * f_IJ * integral
								
							# 	for I in self.active_occ_indices:
							# 		for J in self.active_occ_indices:
							# 			if I <= J:
							# 				continue
							# 			for C in self.active_occ_indices:
							# 				for D in self.active_occ_indices:
							# 					if C <= D:
							# 						continue
							# 					integral = eri_as[C, D, I, J]

							# 					t_CDIJ = MP1_amplitudes[C, D, I, J]
												
							# 					# \partial t_CBKL/\partial f_AB = ...
							# 					t_CDIJ = 2*t_CDIJ/(Fmo_spin[C, C] + Fmo_spin[D, D] - Fmo_spin[I, I] - Fmo_spin[J, J] - 2*Fmo_spin[C,D] - 2*Fmo_spin[I,J])

							# 						# plus some recursive terms that i omit
							# 					delta_EF = 1.0 if E == F else 0.0

							# 					term5 += 2.0 * t_CDIJ * integral * D_ab[idx_A, idx_B] * (1 - delta_EF)

							H[row, col] = term1 + term2 + term3 + term4
			return H
		
		# t0 = time.time()
		# H = compute_hessian(self, MP1_amplitudes, eri_as, D_ab, Fmo_spin)
		# print("Time for Hessian computation: ", time.time() - t0)
	
		def compute_hessian_optimized(self, MP1_amplitudes, eri_as, D_ab, Fmo_spin):
			occ = np.array(self.active_occ_indices)
			vir = np.array(self.active_inocc_indices)
			inactive = np.array(self.inactive_indices)
			
			nocc = len(occ)
			nvir = len(vir)
			ninactive = len(inactive)

			# Early return if no inactive orbitals
			if ninactive == 0:
				return np.zeros((nvir, nvir), dtype=np.float64)
			
			# Get I<J pairs
			i_idx, j_idx = np.triu_indices(nocc, k=1)
			n_pairs = len(i_idx)
			
			# Initialize Hessian in 4D format for easier indexing
			H_4d = np.zeros((nvir, ninactive, nvir, ninactive))
			
			# === PREPARE DATA ===
			# Extract t_pairs: t[A,B,I,J] for I<J
			t_pairs = np.zeros((nvir, nvir, n_pairs))
			for p in range(n_pairs):
				i = occ[i_idx[p]]
				j = occ[j_idx[p]]
				t_pairs[:, :, p] = MP1_amplitudes[np.ix_(vir, vir, [i], [j])].reshape(nvir, nvir)
			
			# === TERM 1: 2 * Σ_{i>j} t_ij^{ab} <ef|ij> ===
			# Extract integrals for term1
			int_ef_pairs = np.zeros((ninactive, ninactive, n_pairs))
			for p in range(n_pairs):
				i = occ[i_idx[p]]
				j = occ[j_idx[p]]
				int_ef_pairs[:, :, p] = -eri_as[np.ix_(inactive, inactive, [i], [j])].reshape(ninactive, ninactive)
			
			# H[A,E,B,F] += 2 * Σ_p t[A,B,p] * int[E,F,p]
			term1_4d = 2.0 * np.einsum('abp,efp->aebf', t_pairs, int_ef_pairs, optimize=True)
			H_4d += term1_4d
			
			# === TERM 2: - Σ_{i>j} Σ_c [t_ij^{ac} <bc|ij> + t_ij^{cb} <ac|ij>] * δ_EF ===
			# Extract integrals for term2: <B,C,I,J> and <A,C,I,J> for I<J
			int_pairs = np.zeros((nvir, nvir, n_pairs))
			for p in range(n_pairs):
				i = occ[i_idx[p]]
				j = occ[j_idx[p]]
				int_pairs[:, :, p] = -eri_as[np.ix_(vir, vir, [i], [j])].reshape(nvir, nvir)
			
			# Compute Σ_c t[A,C,p] * int[B,C,p]
			term2_part1 = np.einsum('acp,bcp->abp', t_pairs, int_pairs, optimize=True)
			# Compute Σ_c t[C,B,p] * int[C,A,p]
			term2_part2 = np.einsum('cbp,cap->abp', t_pairs, int_pairs, optimize=True)
			
			# Sum over p and apply negative sign
			term2_sum = -np.sum(term2_part1 + term2_part2, axis=2)  # shape: (nvir, nvir)
			
			# Apply δ_EF: add to H[A,E,B,E] for all A,B and each E
			# term2_sum is (nvir, nvir), we need to add it to H_4d[:, e, :, e] for each e
			for e in range(ninactive):
				# H_4d[:, e, :, e] has shape (nvir, nvir)
				H_4d[:, e, :, e] += term2_sum
			
			# === TERM 3: Dab * (f_aa - f_bb) δ_EF ===
			# Non-canonical Fock
			if self.use_canonical_Fock == False:
				# Extract f_aa and f_bb for active virtual orbitals
				f_diag_vir = np.array([Fmo_spin[a, a] for a in vir])
				f_diff = f_diag_vir[:, None] - f_diag_vir[None, :]  # shape: (nvir, nvir)
			# Canonical Fock
			else:
				# Extract ε_a and ε_b for active virtual orbitals
				eps_vir = np.array([self.eps[a] for a in vir])
				f_diff = eps_vir[:, None] - eps_vir[None, :]  # shape: (nvir, nvir)
			term3 = D_ab * f_diff  # shape: (nvir, nvir)
			
			# Add to diagonal E,F blocks
			for e in range(ninactive):
				H_4d[:, e, :, e] += term3
			
			# === TERM 4: Dab * f_ef (1- δ_EF) ===
			# Extract f_ef for inactive orbitals
			f_inactive = Fmo_spin[np.ix_(inactive, inactive)]  # shape: (ninactive, ninactive)
			
			# Create term4_4d = D_ab[A,B] * f_inactive[E,F] for E != F
			term4_4d = np.einsum('ab,ef->aebf', D_ab, f_inactive, optimize=True)
			
			# Zero out diagonal E=F elements
			for e in range(ninactive):
				term4_4d[:, e, :, e] = 0.0
			
			H_4d += term4_4d
			
			# Reshape back to 2D
			H = H_4d.reshape(nvir * ninactive, nvir * ninactive)
				
			return H


		# t1 = time.time()
		H_optimized = compute_hessian_optimized(self, MP1_amplitudes, eri_as, D_ab, Fmo_spin)
		# print("Time for optimized Hessian computation: ", time.time() - t1)
		 	# Check that both methods give the same result
		# hess_diff = np.max(np.abs(H - H_optimized))
		# assert np.allclose(H, H_optimized, atol=1e-10), "Optimized Hessian does not match original! Max diff: {}".format(hess_diff)
		 	# Use optimized version
		H = H_optimized

		# Write 2D hessian to file for debugging
		np.savetxt("hessian_debug.txt", H)


			# Hessian
		print("  - Hessian:")

		# Check eigenvalues
		eigvals = np.linalg.eigvalsh(H)  # eigh for symmetric
		print(f"    Hessian norm: {np.linalg.norm(H):.6e}, (Neg. Eigval: {np.sum(eigvals < 0)}/{len(eigvals)})")
		if np.any(eigvals < -1e-6):
			print("        WARNING: Hessian has negative eigenvalues!")

			# Determinant of Hessian
		det_H = np.linalg.det(H) if H.size > 0 else 0.0
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

					# More stable than direct inverse
					#R_block = np.linalg.solve(H_block, -G_block)
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
			# Set it globally
		self.use_RLE_orbopt = use_RLE

			# Direct inversion method
		if not use_RLE:
			G = G.flatten()  # Flatten G to 1D array

			# Solve for R, unoccupied space
				# Handle if H is singular
			if np.linalg.cond(H) > 1e12 and np.linalg.matrix_rank(H) < H.shape[0] and det_H == 0.0:
				print("        WARNING: Using pseudo-inverse for orbital rotations.")
				R = - G @ np.linalg.pinv(H)
				# Handle is H is ill-conditioned, use solve
			# elif np.linalg.cond(H) > 1e12:
			# 	print("        WARNING: Using np.linalg.solve for orbital rotations due to ill-conditioned Hessian.")
			# 	R = np.linalg.solve(H, -G)
			else:
				R = - G @ np.linalg.inv(H)


			# Reduced Linear Equation method
		if use_RLE:				
			R = solve_block_diagonal_RLE_safe(H, G, len(self.active_inocc_indices), len(self.inactive_indices))
			
		
		# Initialize R,
			# Matrix: self.full_indices x self.full_indices
		R_matrix = np.zeros((len(self.full_indices), len(self.full_indices), ), dtype=np.float64)

		# Build R_matrix from R[A, E]
		for idx_A, A in enumerate(self.active_inocc_indices):
			for idx_E, E in enumerate(self.inactive_indices):
				# Extract R_EA and R_AE
				R_AE = R[idx_A * len(self.inactive_indices) + idx_E]
				R_EA = -1.0*R_AE  # Antisymmetry

				# Check antisymmetry
				if abs(R_EA + R_AE) > 1e-10:
					print(f"WARNING: R matrix antisymmetry violated for indices ({E},{A}): R_EA={R_EA:.6e}, R_AE={R_AE:.6e}, sum={R_EA + R_AE:.6e}")

				# Fill in R_matrix
				R_matrix[E, A] = R_AE
				R_matrix[A, E] = R_EA

		# # After solving for R
		# print(f"\nRotation parameters:")
		# print(f"R shape: {R.shape}")
		# print(f"R norm: {np.linalg.norm(R):.6e}")
		# print(f"Max |R|: {np.max(np.abs(R)):.6e}")
		# print(f"Min |R|: {np.min(np.abs(R)):.6e}")

		# # Check R_matrix
		# print(f"\nR_matrix norm: {np.linalg.norm(R_matrix):.6e}")
		# print(f"Max |R_matrix|: {np.max(np.abs(R_matrix)):.6e}")


			# Rotation matrix
		print("  - Rotation matrix:")

				# Convergence check based on max element of R_matrix
		max_R_elem = np.max(np.abs(R_matrix))
		print(f"    Rotation norm {np.linalg.norm(R_matrix):.6e}, (Max el.: {max_R_elem:.6e})")

			# Check shape
		# expected_shape = np.zeros((len(self.full_indices), len(self.full_indices))).shape
		# assert R_matrix.shape == expected_shape, f"R_matrix shape is {R_matrix.shape}, expected {expected_shape}"
			# Check that R is anti-symmetric
		diff_R = np.linalg.norm(R_matrix + R_matrix.T)
		assert diff_R < 1e-6, f"R_matrix is not anti-symmetric, ||R + R.T|| = {diff_R}"
			# Check that R_matrix has no NaN or Inf values
		assert np.all(np.isfinite(R_matrix)), "R_matrix contains NaN or Inf values!"
			# Check that R_matrix is not all zeros
		count_nonzero_R = np.count_nonzero(R_matrix)
		if count_nonzero_R == 0:
			print("        WARNING: R_matrix is all zeros!")




		# Which way is the rotation?		# By convention, we use U = exp(R) where R is anti-symmetric
			# Check that R_matrix is small for convergence
		if max_R_elem < 1e-6:
			print("        WARNING: Rotation parameters are very small -> Orbitals are optimized!")

			# We can also check if the rotation leads to a decrease in energy
				# This requires computing the new orbitals and energy, which is more involved
			# For now, we just print a warning
		if max_R_elem < 1e-8:
			print("        WARNING: Rotation parameters are extremely small -> Orbitals are likely fully optimized!")





		# Step (vii): Construct the unitary orbital rotation matrix U = exp(R)

		# Unitary rotation matrix
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
			# Print the the part of U corresponding to active occupied orbitals
		# print("Active occupied block of U:")
		# print(U[np.ix_(self.active_occ_indices, self.active_occ_indices)])





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

			for s in range(n_spatial):
				C_spin[:, 2*s]   = C_alpha[:, s]  # alpha s
				C_spin[:, 2*s+1] = C_beta[:, s]   # beta s

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
			
			if self.use_canonical_Fock:	
			# Check that rotated orbitals are orthonormal
				# Sum a coloumn of mo_coeffs to check normalization for a given spin
					# Check: C^T S C = I
				C_i = mo_coeffs_rot[spin]
				norm = C_i.T @ self.S @ C_i
				assert np.allclose(norm, np.eye(norm.shape[0]), atol=1e-6), f"MO coefficients for spin {spin} are not orthonormal!"


		# Rotate the Fock matrix in the MO basis as well
			# Only rotate the active virtual block of the Fock matrix, since that's what we use for the next iteration
				# Fmo_spin is the Fock matrix in the spin-orbital basis, we need to rotate it to get the new Fock matrix in the MO basis
					# Fmo_rot = U^T Fmo_spin U
						# This is the standard transformation for a Fock matrix under orbital rotation
			# Isolate the active virtual block U to rotate the relevant part of Fmo
				# In the spin-orbital basis, the active virtual orbitals correspond to certain columns of U, and the occupied orbitals correspond to certain rows/columns of Fmo_spin
				# Leave the occupied-occupied and inactive-inactive blocks of Fmo unchanged, only rotate the active virtual block
		U_active_virtual = U[np.ix_(self.active_inocc_indices, self.active_inocc_indices)]
		Fmo_spin_active_virtual = Fmo_spin[np.ix_(self.active_inocc_indices, self.active_inocc_indices)]
		Fmo_rot_active_virtual = U_active_virtual.T @ Fmo_spin_active_virtual @ U_active_virtual

			# Construct the full rotated Fock matrix in the spin-orbital basis
		Fmo_rot = np.copy(Fmo_spin)
		Fmo_rot[np.ix_(self.active_inocc_indices, self.active_inocc_indices)] = Fmo_rot_active_virtual

			# Diagonalize the full Fock matrix
		eigvals_rot, eigvecs_rot = np.linalg.eigh(Fmo_rot)
		Fmo_rot = eigvecs_rot @ np.diag(eigvals_rot) @ eigvecs_rot.T.conj()

			# Check that the rotated Fock matrix is still Hermitian
		if not np.allclose(Fmo_rot, Fmo_rot.T.conj(), atol=1e-10):
			print("WARNING: Rotated Fock matrix is not Hermitian!")

			# Check that the diagonal elements of the rotated Fock matrix are real
		if not np.all(np.isreal(np.diag(Fmo_rot))):
			print("WARNING: Diagonal elements of rotated Fock matrix are not real!")

			# Check that the rotated Fock matrix has no NaN or Inf values
		if not np.all(np.isfinite(Fmo_rot)):
			print("WARNING: Rotated Fock matrix contains NaN or Inf values!")


		return mo_coeffs_rot, Fmo_rot, self.use_RLE_orbopt


	





	def run_ovos(self,  mo_coeffs, Fmo_rot):
		"""
		Run the OVOS algorithm.
		"""

		converged = False
		max_iter = 500
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
			E_corr, MP1_amplitudes, eri_spin, eri_phys, eri_as, Fmo_spin = self.MP2_energy(mo_coeffs = mo_coeffs, Fmo = Fmo_rot)

			if E_corr > 0.0:
					print("		WARNING: Correlation energy is positive! OVOS may not have converged properly.")

					# Break here to stop on positive correlation energy, 
						# but still record the energy and iteration count
					if iter_count < max_iter and iter_count > 1:
						lst_E_corr.append(E_corr)
						lst_iter_counts.append(iter_count)

						# Account for if first iteration is positive, initialize lists
					if iter_count == 1:
						lst_E_corr = []
						lst_E_corr.append(E_corr)

						lst_iter_counts = []
						lst_iter_counts.append(iter_count)

					break # Remember to comment out if not use_best_of

			# Step (ix): check convergence
			print(" Step (ix): Check convergence")
			# convergence criterion: change in correlation energy < 1e-10 Hartree
			if iter_count < max_iter and iter_count > 1:
				threshold = 1e-8
				if np.abs(E_corr - lst_E_corr[-1]) < threshold:
					converged = True
					print("OVOS converged in ", iter_count, " iterations.")

					lst_E_corr.append(E_corr)
					lst_iter_counts.append(iter_count)

					break

				else:
					lst_E_corr.append(E_corr)
					lst_iter_counts.append(iter_count)

					print(f"	Change in correlation energy: {np.abs(E_corr - lst_E_corr[-2]):.6e} Hartree (threshold: {threshold:.1e})")
						# If the change is positive, warn the user
					if E_corr > lst_E_corr[-2]:
						print("		WARNING: Correlation energy increased in this iteration!")
					print()

					# Step (v)-(viii): Orbital optimization
					print(" Step (v)-(viii): Orbital optimization")
					mo_coeffs, Fmo_rot, use_RLE = self.orbital_optimization(mo_coeffs=mo_coeffs, MP1_amplitudes=MP1_amplitudes, eri_as=eri_as, Fmo_spin=Fmo_spin, eri_spin=eri_spin, eri_phys=eri_phys)


			# # Check if stuck in a limit cycle or oscillatory pattern
			# # Detect persistent oscillation between two or more values
			# if iter_count >= 250:  # Need enough history to detect a pattern
			# 	# Look for repeating patterns in the last few iterations
			# 	recent_energies = lst_E_corr[-min(12, len(lst_E_corr)):]  # Get more points for better detection
				
			# 	if len(recent_energies) >= 16:  # Need enough points for reliable detection
			# 		# Method 1: Check for PERFECT alternation with large amplitude
			# 		# Only trigger if oscillation amplitude is significantly above convergence threshold
			# 		if len(recent_energies) >= 16:
			# 			# Check last 8 points for A-B-A-B pattern with consistent amplitude
			# 			alt_pattern_large = True
			# 			oscillation_amplitude = abs(recent_energies[-1] - recent_energies[-2])
						
			# 			# Check if we have at least 4 cycles of consistent alternation
			# 			for i in range(1, 5):
			# 				idx1 = -2*i
			# 				idx2 = -2*i - 1
			# 				if idx2 >= -len(recent_energies):
			# 					current_amp = abs(recent_energies[idx1] - recent_energies[idx2])
			# 					# Amplitude should be similar and large (> 10x convergence threshold)
			# 					if abs(current_amp - oscillation_amplitude) > 1e-10 or oscillation_amplitude < 1e-7:
			# 						alt_pattern_large = False
			# 						break
						
			# 			# Also verify values alternate back to similar numbers
			# 			if alt_pattern_large:
			# 				# Check if odd-indexed values are similar and even-indexed values are similar
			# 				odd_vals = recent_energies[-1:-9:-2]  # Last, 3rd last, 5th last, 7th last
			# 				even_vals = recent_energies[-2:-9:-2]  # 2nd last, 4th last, 6th last, 8th last
							
			# 				odd_std = np.std(odd_vals) if len(odd_vals) > 1 else 0
			# 				even_std = np.std(even_vals) if len(even_vals) > 1 else 0
							
			# 				# Values in each group should be similar (small std)
			# 				alt_pattern_large = alt_pattern_large and (odd_std < 1e-7 and even_std < 1e-7)
					
			# 		# Method 2: Detect divergence (consistently increasing energy)
			# 		if len(recent_energies) >= 6:
			# 			# Check if last 6 energies show monotonic increase
			# 			increasing_trend = all(recent_energies[i] > recent_energies[i-1] 
			# 								for i in range(-1, -6, -1))
						
			# 			if increasing_trend:
			# 				print("OVOS appears to be diverging (correlation energy consistently increasing).")
			# 				print(f"  Recent trend: {[f'{e:.10f}' for e in recent_energies[-6:]]}")
			# 				print("  Stopping optimization.")
			# 				converged = False
			# 				break
					
			# 		# Method 3: Check for oscillation without convergence over many iterations
			# 		# Calculate if we're making progress toward convergence
			# 		if len(recent_energies) >= 10:
			# 			# Split into first half and second half of recent points
			# 			first_half = recent_energies[:5]
			# 			second_half = recent_energies[5:]
						
			# 			# Calculate range of oscillations in each half
			# 			range_first = max(first_half) - min(first_half)
			# 			range_second = max(second_half) - min(second_half)
						
			# 			# If oscillation range isn't decreasing significantly, we might be stuck
			# 			no_progress = (abs(range_second - range_first) / max(range_first, 1e-10) < 0.1)
						
			# 			# And if we're still far from convergence threshold
			# 			avg_change = np.mean([abs(recent_energies[i] - recent_energies[i-1]) 
			# 								for i in range(1, len(recent_energies))])
						
			# 			# Strong limit cycle detection: oscillation amplitude > 1e-6 AND no progress
			# 			strong_limit_cycle = (oscillation_amplitude > 1e-6 and 
			# 								no_progress and 
			# 								avg_change > 1e-7)
					
			# 		# Trigger limit cycle warning only for CLEAR cases
			# 		if alt_pattern_large and oscillation_amplitude > 1e-6:
			# 			print("OVOS appears to be stuck in a limit cycle/oscillatory pattern.")
			# 			print(f"  Clear A-B-A-B pattern detected with consistent large amplitude.")
			# 			print(f"  Recent energies: {[f'{e:.10f}' for e in recent_energies[-8:]]}")
			# 			print(f"  Oscillation amplitude: {oscillation_amplitude:.2e} Hartree")
			# 			print(f"  (Convergence threshold: {threshold:.1e})")
			# 			print("  Stopping optimization to avoid infinite loop.")
						
			# 			# Set converged to False
			# 			converged = False
			# 			break
			# 		elif strong_limit_cycle:
			# 			print("OVOS appears to be stuck - oscillations not decreasing.")
			# 			print(f"  Average energy change: {avg_change:.2e} (not decreasing toward threshold)")
			# 			print(f"  Oscillation range: {range_second:.2e} (similar to previous: {range_first:.2e})")
			# 			print("  Stopping optimization.")
			# 			converged = False
			# 			break				


			# First iteration, initialize lists
			if iter_count == 1:
				lst_E_corr = []
				lst_E_corr.append(E_corr)

				lst_iter_counts = []
				lst_iter_counts.append(iter_count)

				# Step (v)-(viii): Orbital optimization
				print(" Step (v)-(viii): Orbital optimization")
				mo_coeffs, Fmo_rot, use_RLE = self.orbital_optimization(mo_coeffs=mo_coeffs, MP1_amplitudes=MP1_amplitudes, eri_as=eri_as, Fmo_spin=Fmo_spin, eri_spin=eri_spin, eri_phys=eri_phys)

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

		return lst_E_corr, lst_iter_counts, mo_coeffs, Fmo_rot
	
	


# Molecule
atom_choose_between = [
	"H .0 .0 .0; H .0 .0 0.74144",  # H2 bond length 0.74144 Angstrom
	"Li .0 .0 .0; H .0 .0 1.595",   # LiH bond length 1.595 Angstrom
	"O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;",  # H2O equilibrium geometry
	"C  0.0000  0.0000  0.0000; H  0.0000  0.9350  0.5230; H  0.0000 -0.9350  0.5230;", # CH2 
	"B 0 0 0; H 0 0 1.19; H 0 1.03 -0.40; H 0 -1.03 -0.40",  # BH3 equilibrium geometry
	"N 0 0 0; N 0 0 1.10", # N2 bond length 1.10 Angstrom
]
# Basis set
basis_choose_between = [
	"STO-3G",
	"STO-6G",
	"3-21G",
	"6-31G",
	"DZP",
	"roosdz",
	"anoroosdz",
	"cc-pVDZ",
	"cc-pV5Z",
	"def2-QZVPP",
	"aug-cc-pV5Z",
	"ANO"
]

find_atom = {
	"H2": 0,
	"LiH": 1,
	"H2O": 2,
	"CH2": 3,
	"BH3": 4,
	"N2": 5
}	

find_basis = {
	"STO-3G": 0,
	"STO-6G": 1,
	"3-21G": 2,
	"6-31G": 3,
	"DZP": 4,
	"roosdz": 5,
	"anoroosdz": 6,
	"cc-pVDZ": 7,
	"cc-pV5Z": 8,
	"def2-QZVPP": 9,
	"aug-cc-pV5Z": 10,
	"ANO": 11
}




# Select molecule and basis set
select_atom  = "CH2"  		# Select atom index here
select_basis = "6-31G"  	# Select basis index here
	# I want to run OVOS on CH2 w.
		# Article reference - CH2 !!!
		# (9s7p2d If,5s2p) 	-> [2, ..., 116]	-> E(SCF) = -38.89447 | E(UMP2) = ...      ,  E_corr = -0.182 683
		# ------------------------------------------------------------|-------------------------------------------
		# 6-31G 		 	-> [2, ..., 18 ]	-> E(SCF) = -38.84716 | E(UMP2) = -38.91172,  E_corr = -0.064 553
		# cc-pVDZ 		 	-> [2, ..., 40 ]	-> E(SCF) = -38.87219 | E(UMP2) = -38.98252,  E_corr = -0.110 327
		# DZP 		 		-> [2, ..., 42 ]	-> E(SCF) = -38.87497 | E(UMP2) = -38.99727,  E_corr = -0.122 306
		# roosdz 	 		-> [2, ..., 74 ]	-> E(SCF) = -38.88702 | E(UMP2) = -39.01329,  E_corr = -0.126 263
		# anoroosdz 	 	-> [2, ..., 74 ]	-> E(SCF) = -38.88702 | E(UMP2) = -39.01329,  E_corr = -0.126 263
		# def2-QZVPP 		-> [2, ..., 226]	-> E(SCF) = -38.88865 | E(UMP2) = -39.05821,  E_corr = -0.169 552
		# ANO 		 		-> [2, ..., 334]	-> E(SCF) = -38.88763 | E(UMP2) = -39.07982,  E_corr = -0.192 186
		# cc-pV5Z 	 		-> [2, ..., 394]	-> E(SCF) = -38.88888 | E(UMP2) = -39.07190,  E_corr = -0.183 022 <-- !!!
		# aug-cc-pV5Z 		-> [2, ..., 566]	-> E(SCF) = -38.88897 | E(UMP2) = -39.07315,  E_corr = -0.184 171

atom, basis = (atom_choose_between[find_atom[select_atom]], basis_choose_between[find_basis[select_basis]])

# Print start message
print(" Running OVOS on ", select_atom, " with basis set ", select_basis)
print("")

# Get number of electrons and full space size in molecular orbitals
unit = "angstrom" # angstrom or bohr
	# Initialize molecule and UHF
mol = pyscf.M(atom=atom, basis=basis, unit=unit)
	# Set symmetry
# mol.symmetry = False  # Disable symmetry for OVOS
uhf = pyscf.scf.UHF(mol).run()
	# Number of electrons
num_electrons = mol.nelec[0] + mol.nelec[1]
	# Full space size in molecular orbitals
full_space_size = int(uhf.mo_coeff.shape[1])
	# MP2 correlation energy for the full space
MP2_UHF = uhf.MP2().run()
		# Set reference MP2
MP2 = MP2_UHF

# Can we also opt. with no virt-virt ?? WHY/NOT
# WHERE DO WE FORCE IT TO ONLY TAKE into account
# 	-> It makes sense for UHF ...


# My MO energies are the same throughout - check if they change after orbital optimization
#   -> They do not change for each iteration - OK, do not change as we do unitary transformations only


"""
Run OVOS for different numbers of optimized virtual orbitals
"""
run_different_virt_orbs = True
if run_different_virt_orbs == True:
	# Loop over different numbers of optimized virtual orbitals
	# List of MP2 correlation energies for different numbers of optimized virtual orbitals
	lst_E_corr_virt_orbs = [[],[],[],[]]  # [[E_corr_list], [num_opt_virtual_orbs_list], [iterations_till_convergence_list], [Unr. SCF check list]]
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


	# Flags - Only one at a time is TRUE or both FALSE
		# Flag to indicate if previous virtual orbitals are used
	use_prev_virt_orbs = True
		# Flag if 1000 repeats of different random unitary rotations for each num_opt_virtual_orbs_current are use on UHF orbitals as starting guess
	use_random_unitary_init = False

	# From the article: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			# The initial implementation of our OVOS procedure is restricted 
			# to the closed-shell RHF reference case
		# Flag to indicate if standard UHF initialization is used
	use_UHF_init = False # Also for random unitary and prev. virt. orb. inits
		# Flag to indicate if standard RHF initialization is used
			# Set to inverse of use_UHF_init
	use_RHF_init = not use_UHF_init
	
	# Set flag for Canonical or Non-Canonical Fock matrix
		# Canonical Fock matrix is diagonal in the MO basis
	use_canonical_Fock = True  # True for Canonical Fock matrix, False for Non-Canonical Fock matrix

	# Best of number of tries - True when random initializations are used
	if use_random_unitary_init == True:
		try_best_of = True
	else:
		try_best_of = False
		
		# Number of random initializations to try for each num_opt_virtual_orbs_current
	attempts_total = 1000  # Total attempts for each num_opt_virtual_orbs_current


	while num_opt_virtual_orbs_current < max_opt_virtual_orbs:
		# Increment num_opt_virtual_orbs until OVOS converges successfully
		num_opt_virtual_orbs_current += increment 

		lst_E_corr = None  # Reset lst_E_corr for each run

		print("")
		print("#### OVOS with ", num_opt_virtual_orbs_current, " out of ", max_opt_virtual_orbs," optimized virtual orbitals (Retry count: ", retry_count,") ####")

		try:
			# Re-initialize molecule and UHF for each run
			mol = pyscf.M(atom=atom, basis=basis, unit=unit)

			
			if try_best_of == False: # Single run of OVOS

				if use_UHF_init == True:
					# Get UHF orbitals
					if use_prev_virt_orbs == False or use_random_unitary_init == False and 'mo_coeffs' not in locals() and 'Fmo_rot' not in locals():
						# Only get UHF orbitals if not using previous or random init
							# The if state will prevent re-getting UHF orbitals if using previous or random init
						mo_coeffs = pyscf.scf.UHF(mol).run().mo_coeff
						Fmo_rot = None # Fock matrix in the MO basis will be constructed later based on the initial orbitals
					init_orbs = "UHF"
				if use_RHF_init == True:
					# Get RHF orbitals
					if use_prev_virt_orbs == False or use_random_unitary_init == False and 'mo_coeffs' not in locals() and 'Fmo_rot' not in locals():
						# Only get RHF orbitals if not using previous or random init
							# The if state will prevent re-getting RHF orbitals if using previous or random init
						mo_coeffs = pyscf.scf.RHF(mol).run().to_uhf().mo_coeff
						Fmo_rot = None # Fock matrix in the MO basis will be constructed later based on the initial orbitals
					init_orbs = "RHF"

				print()
				print("Fmo matrix before OVOS (if exists):", isinstance(Fmo_rot, np.ndarray))
				print("Mo_coeffs before OVOS (if exists):", 'mo_coeffs' in locals())
				print()

				if use_prev_virt_orbs == True and 'mo_coeffs' in locals() and 'Fmo_rot' in locals():
					# Use previously optimized orbitals as starting guess
						# Only if mo_coeff exists from previous run
					
					# Note: The first 2(4e) orbitals correspond to the occupied space and should not be changed.
					# We only want to modify the virtual orbitals beyond the occupied space.
					# Extract occupied and virtual orbitals from previous mo_coeff

						# Enforce orthonormality of the modified mo_coeff
					mo_coeffs = mo_coeffs.copy()  # Start from previous optimized orbitals
					Fmo_rot = Fmo_rot  # Start from previous optimized Fock matrix

					print("    Using previously optimized orbitals as starting guess.")				

			
				elif use_random_unitary_init == True:
					# Apply random unitary rotation to virtual orbitals only
						# Number of occupied orbitals
					num_occupied_orbitals = num_electrons // 2
						# Total number of spatial orbitals
					total_spatial_orbitals = mo_coeffs_uhf[0].shape[1]
						# Number of virtual orbitals
					num_virtual_orbitals = total_spatial_orbitals - num_occupied_orbitals

						# Generate random unitary matrix for virtual orbitals
					rand_matrix = np.random.rand(num_virtual_orbitals, num_virtual_orbitals)
					Q, R = np.linalg.qr(rand_matrix)  # QR decomposition to get unitary matrix

						# Rotate virtual orbitals for alpha and beta spins
					mo_coeffs = [np.copy(mo_coeffs_uhf[0]), np.copy(mo_coeffs_uhf[1])]  # Deep copy to avoid modifying original
					Fmo_rot = Fmo_rot if Fmo_rot is not None else None  # Start from previous optimized Fock matrix if it exists, otherwise None
				
					print("Fock matrix before rotation (if exists):", Fmo_rot)

					for spin in [0, 1]:
						# Extract occupied and virtual parts
						C_occ = mo_coeffs_uhf[spin][:, :num_occupied_orbitals]
						C_virt = mo_coeffs_uhf[spin][:, num_occupied_orbitals:]

						# Rotate virtual orbitals
						C_virt_rot = C_virt @ Q

						# Combine back
						mo_coeffs[spin] = np.hstack((C_occ, C_virt_rot))

					# Rotate corresponding block of Fock matrix if it exists
					if Fmo_rot is not None:
						Fmo_rot_block = Fmo_rot[num_occupied_orbitals:, num_occupied_orbitals:]
						Fmo_rot_block_rot = Q.T @ Fmo_rot_block @ Q
						Fmo_rot[num_occupied_orbitals:, num_occupied_orbitals:] = Fmo_rot_block_rot

						# Note: The occupied-occupied and occupied-virtual blocks of the Fock matrix remain unchanged, only the virtual-virtual block is rotated.


					print("Using random unitary rotated UHF virtual orbitals as starting guess.")





				# # Profil the OVOS run
				# import cProfile
				# import pstats
				# import io

				# pr = cProfile.Profile()
				# pr.enable()

					# Run OVOS
				lst_E_corr, lst_iter_counts, mo_coeffs, Fmo_rot = OVOS(mol=mol, num_opt_virtual_orbs=num_opt_virtual_orbs_current, init_orbs=init_orbs, mo_coeff=mo_coeffs, use=[use_canonical_Fock]).run_ovos(mo_coeffs=mo_coeffs, Fmo_rot=Fmo_rot)

				# pr.disable()

				# s = io.StringIO()
				# ps = pstats.Stats(pr, stream=s)
				# ps.strip_dirs().sort_stats('cumtime').print_stats()

				# with open('ovos_profile_stats.txt', 'w+') as f:
				# 	f.write(s.getvalue())

				# 	# Debug stop
				# import sys
				# sys.exit("Debug stop after profiling OVOS.")




			if try_best_of == True: # Multiple runs of OVOS with different random initializations
				# Try multiple random initializations and pick the best result
				attempt = 0
				best_E_corr = None
				best_lst_E_corr = None
				best_lst_iter_counts = None
				best_mo_coeffs = None

				while attempt < attempts_total:
					attempt += 1
					print("")
					print("---- Attempt ", attempt, " of ", attempts_total, " ----")

					# Get new random unitary rotation for virtual orbitals
						# Get UHF orbitals
					mo_coeffs_uhf = pyscf.scf.UHF(mol).run().mo_coeff

					# Apply random unitary rotation to virtual orbitals only
						# Number of occupied orbitals
					num_occupied_orbitals = num_electrons // 2
						# Total number of spatial orbitals
					total_spatial_orbitals = mo_coeffs_uhf[0].shape[1]
						# Number of virtual orbitals
					num_virtual_orbitals = total_spatial_orbitals - num_occupied_orbitals

						# Generate random unitary matrix for virtual orbitals
					rand_matrix = np.random.rand(num_virtual_orbitals, num_virtual_orbitals)
					Q, R = np.linalg.qr(rand_matrix)  # QR decomposition to get unitary matrix

						# Rotate virtual orbitals for alpha and beta spins
					mo_coeffs = [np.copy(mo_coeffs_uhf[0]), np.copy(mo_coeffs_uhf[1])]  # Deep copy to avoid modifying original
					for spin in [0, 1]:
						# Extract occupied and virtual parts
						C_occ = mo_coeffs_uhf[spin][:, :num_occupied_orbitals]
						C_virt = mo_coeffs_uhf[spin][:, num_occupied_orbitals:]

						# Rotate virtual orbitals
						C_virt_rot = C_virt @ Q

						# Combine back
						mo_coeffs[spin] = np.hstack((C_occ, C_virt_rot))
					
					init_orbs = "UHF"
					mo_coeffs = np.array(mo_coeffs)

					# Run OVOS
					lst_E_corr_attempt, lst_iter_counts_attempt, mo_coeffs_attempt, Fmo_rot_attempts = OVOS(mol=mol, num_opt_virtual_orbs=num_opt_virtual_orbs_current, init_orbs=init_orbs, mo_coeff=mo_coeffs, use=[use_canonical_Fock]).run_ovos(mo_coeffs=mo_coeffs, Fmo_rot=Fmo_rot)

					# Check if this is the best result so far
					if best_E_corr is None or lst_E_corr_attempt[-1] < best_E_corr:
						best_E_corr = lst_E_corr_attempt[-1]
						best_lst_E_corr = lst_E_corr_attempt
						best_lst_iter_counts = lst_iter_counts_attempt
						best_mo_coeffs = mo_coeffs_attempt
						best_Fmo_rot = Fmo_rot_attempts

				# Use the best result from all attempts
				lst_E_corr = best_lst_E_corr
				lst_iter_counts = best_lst_iter_counts
				mo_coeffs = best_mo_coeffs
				Fmo_rot = best_Fmo_rot


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
			lst_MP2_virt_orbs.append((num_opt_virtual_orbs_current, lst_E_corr[-1], len(lst_E_corr)))
			lst_E_corr_virt_orbs[0].append(lst_E_corr)
			lst_E_corr_virt_orbs[1].append(num_opt_virtual_orbs_current)
			lst_E_corr_virt_orbs[2].append(lst_iter_counts)


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
	for num_opt_virtual_orbs_current, E_corr, iter_ in lst_MP2_virt_orbs:
		print("MP2 correlation energy, for ", num_opt_virtual_orbs_current, f" optimized virtual orbitals: ", '%.5E' % Decimal(E_corr),f" ({(E_corr/MP2.e_corr)*100:.4}%)"+" @ ", iter_, " iterations till convergence")
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

	if use_UHF_init == True:
		str_name += "_UHF_init" # !!!!
	if use_RHF_init == True:
		str_name += "_RHF_init" # !!!!

	str_atom = select_atom
	str_basis = select_basis

	if use_canonical_Fock == True:
		str_canonical = "use_canonical_Fock"
	else:
		str_canonical = "use_non_canonical_Fock"

	print("Saving data to branch/data/"+str_atom+"/"+str_basis+"/"+str_canonical+"/")

	# Save MP2 correlation energy convergence data
	with open("branch/data/"+str_atom+"/"+str_basis+"/"+str_canonical+"/lst_MP2_"+str_name+".json", "w") as f:
		json.dump(lst_E_corr_virt_orbs, f, indent=2)

	print("Data saved to branch/data/"+str_atom+"/"+str_basis+"/"+str_canonical+"/lst_MP2_"+str_name+".json")
