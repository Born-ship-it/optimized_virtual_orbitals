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

	def __init__(self, mol: pyscf.gto.Mole, num_opt_virtual_orbs: int, init_orbs: str = "UHF") -> None:
		self.mol = mol
		self.num_opt_virtual_orbs = num_opt_virtual_orbs
		self.init_orbs = init_orbs

		# Set up unrestricted Hartree-Fock calculation 
		self.uhf = pyscf.scf.UHF(mol).run()
		self.e_rhf = self.uhf.e_tot
		self.h_nuc = mol.energy_nuc()

		# Set dependent on type of initial orbitals
		if type(init_orbs) == str:
			# MO coefficients (alpha, beta)
			self.mo_coeffs = self.uhf.mo_coeff
		elif type(init_orbs) != str:
			# Get previous orbitals from init_orbs, same format as self.uhf.mo_coeff
			self.mo_coeffs = init_orbs # User-provided orbitals or other method e.g. previous OVOS run

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
		assert self.tot_num_spin_orbs >= self.num_opt_virtual_orbs+self.nelec, "Your space 'num_opt_virtual_orbs' is too large"  

			# Check that the active spaces are correctly built
		for I in self.active_occ_indices:
			assert I < self.nelec, f"I={I} not less than number of electrons {self.nelec}"
		for A in self.active_inocc_indices:
			assert A >= self.nelec, f"A={A} not greater than or equal to number of electrons {self.nelec}"
			assert A < self.nelec + self.num_opt_virtual_orbs, f"A={A} not less than number of electrons + num_opt_virtual_orbs {self.nelec + self.num_opt_virtual_orbs}"
		for E in self.inactive_indices:
			assert E >= self.nelec + self.num_opt_virtual_orbs, f"E={E} not greater than or equal to number of electrons + num_opt_virtual_orbs {self.nelec + self.num_opt_virtual_orbs}"
			assert E < self.tot_num_spin_orbs, f"E={E} not less than total number of spin orbitals {self.tot_num_spin_orbs}"

			# Check that the spaces do not overlap
		assert len(set(self.active_occ_indices).intersection(set(self.active_inocc_indices))) == 0, "Active occupied and active unoccupied spaces overlap!"
		assert len(set(self.active_occ_indices).intersection(set(self.inactive_indices))) == 0, "Active occupied and inactive unoccupied spaces overlap!"
		assert len(set(self.active_inocc_indices).intersection(set(self.inactive_indices))) == 0, "Active unoccupied and inactive unoccupied spaces overlap!"


	def spatial_to_spin_eri(self, eri_aaaa, eri_aabb, eri_bbbb):
		"""
		Convert spatial ERIs to spin-orbital ERIs.

		Parameters:
		-----------
		eri_aaaa : np.ndarray
			alpha-alpha-alpha-alpha two-electron integrals, shape (n_spatial, n_spatial, n_spatial, n_spatial)
		eri_aabb : np.ndarray
			alpha-alpha-beta-beta two-electron integrals, shape (n_spatial, n_spatial, n_spatial, n_spatial)
		eri_bbbb : np.ndarray
			beta-beta-beta-beta two-electron integrals, shape (n_spatial, n_spatial, n_spatial, n_spatial)
			
		Returns:
		--------
		eri_spin : np.ndarray
			Spin-orbital two-electron integrals, shape (n_spin, n_spin, n_spin, n_spin)
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
						
						# Handle possible spin combinations
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
							# Other combinations like (αβ|βα) etc. is zero
							eri_spin[p, q, r, s] = 0.0
		
		return eri_spin

	def spatial_to_spin_fock(self, Fmo_a, Fmo_b):
		"""
		Convert spatial Fock matrices to spin-orbital Fock matrix.
		
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
		
		# Convert using the Fortran logic
		for p in range(n_spin):
			for q in range(n_spin):
				# Fortran: mod(p,2) -> p % 2 in Python
				# Fortran: (p+1)/2 -> p//2 in Python (0-based indexing)
				
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

		# Sort by energy
		spin_data.sort(key=lambda x: x[0])

		# Extract arrays
		orbital_energies = np.array([d[0] for d in spin_data])
		orbspin = np.array([d[1] for d in spin_data])
		spatial_map = np.array([d[2] for d in spin_data])  # spatial index for each energy-ordered position
		spin_orb_map = np.array([d[3] for d in spin_data])  # spin-orbital index in NATURAL ordering

		return orbital_energies, orbspin, spatial_map, spin_orb_map











	def MP2_energy(self, mo_coeffs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
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







			# Convert Fock matrix to spin orbitals
		Fmo_spin = self.spatial_to_spin_fock(Fmo_a, Fmo_b)

		# Sanity checks
			# Check that Fock matrix is finite
		assert np.all(np.isfinite(Fmo_spin)), "Spin-orbital Fock matrix contains non-finite values!"
			# Check that Fock matrix is not all zero
		assert np.count_nonzero(Fmo_spin) > 0, "Spin-orbital Fock matrix is all zero!"
			# Check that Fock matrix is Hermitian
		assert np.allclose(Fmo_spin, Fmo_spin.T.conj(), atol=1e-10), "Spin-orbital Fock matrix is not Hermitian!"








		# Get orbital energies
		eigval_a, eigvec_a = scipy.linalg.eigh(Fmo_a)  # Use eigh for Hermitian matrices
		eigval_b, eigvec_b = scipy.linalg.eigh(Fmo_b)  # Use eigh for Hermitian matrices

		# Already sorted by eigh
		mo_energy_a = np.real(eigval_a)
		mo_energy_b = np.real(eigval_b)

		# Convert to spin orbitals with proper sorting
		self.eps, self.orbspin_sorted, self.spatial_map_sorted, self.spin_orb_map_sorted = self.spatial_to_spin_mo_energy(mo_energy_a, mo_energy_b)

		# Create inverse mapping: from natural spin-orbital index -> energy-ordered index
		self.natural_to_energy = np.zeros(self.tot_num_spin_orbs, dtype=int)
		for energy_idx, natural_spin_idx in enumerate(self.spin_orb_map_sorted):
			self.natural_to_energy[natural_spin_idx] = energy_idx

		# Convert active indices from NATURAL to ENERGY ordering
		self.active_occ_indices_energy = [self.natural_to_energy[i] for i in self.active_occ_indices]
		self.active_inocc_indices_energy = [self.natural_to_energy[i] for i in self.active_inocc_indices]
		self.inactive_indices_energy = [self.natural_to_energy[i] for i in self.inactive_indices]

		# Set for convenience
		eps = self.eps

		# Sanity checks
			# Check that orbital energies are finite
		assert np.all(np.isfinite(eps)), "Spin-orbital MO energies contain non-finite values!"
			# Check that orbital energies are not all zero
		assert np.count_nonzero(eps) > 0, "Spin-orbital MO energies are all zero!"
			# Check that number of orbital energies matches number of spin orbitals
		assert len(eps) == self.tot_num_spin_orbs, "Number of spin-orbital MO energies does not match number of spin orbitals!"




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
		eri_spin = self.spatial_to_spin_eri(eri_aaaa, eri_aabb, eri_bbbb)

		# Sanity checks 
			# Check that integrals are finite
		assert np.all(np.isfinite(eri_spin)), "Spin-orbital integrals contain non-finite values!"
			# Check that integrals are not all zero
		assert np.count_nonzero(eri_spin) > 0, "Spin-orbital integrals are all zero!"
			


				# Convert from chemist's to physicist's notation
					# Chemist's: (pq|rs), Physicist's: <pq|rs> = (pr|qs)
		eri_phys = eri_spin.transpose(0,2,1,3)  # Swap indices 1 and 2

		# Sanity checks
			# Check that integrals are finite
		assert np.all(np.isfinite(eri_phys)), "Physicist's notation integrals contain non-finite values!"
			# Check that integrals are not all zero
		assert np.count_nonzero(eri_phys) > 0, "Physicist's notation integrals are all zero!"



			# Antisymmetrized integrals in physicist's notation
				# <pq||rs> = <pq|rs> - <pq|sr>
		eri_as = eri_phys - eri_phys.transpose(0,1,3,2)

		# Sanity checks 
			# Check that integrals are finite
		assert np.all(np.isfinite(eri_as)), "Antisymmetrized integrals contain non-finite values!"
			# Check that integrals are not all zero
		assert np.count_nonzero(eri_as) > 0, "Antisymmetrized integrals are all zero!"	



		# ii) Compute MP1 amplitudes (spin-orbital)
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
			
						# Energy denominator: ε_a + ε_b - ε_i - ε_j
						denominator = eps_a + eps_b - eps_i - eps_j

						# Antisymmetrized integral for same-spin pairs
							# Simplifies to <ab|ij> for different-spin
						integral = eri_as[a, b, i, j]  # <ab||ij>
						
						# MP1 amplitude: t_{ij}^{ab} = -<ab||ij> / (ε_a + ε_b - ε_i - ε_j)
						MP1_amplitudes[a, b, i, j] = -integral / denominator

						# print(f"MP1 amplitude t_ij^ab for a={a}, b={b}, i={i}, j={j}: {MP1_amplitudes[a, b, i, j]:.6e} for denominator {denominator:.6e} and integral {integral:.6e}")

		# Sanity checks
			# Check that amplitudes are finite
		assert np.all(np.isfinite(MP1_amplitudes)), "MP1 amplitudes contain non-finite values!"
			# Check that amplitudes are not all zero
		assert np.count_nonzero(MP1_amplitudes) > 0, "MP1 amplitudes are all zero!"
			# Check amplitude antisymmetry: 
				# t_ij^{ab} = t_ji^{ba}
		assert np.allclose(MP1_amplitudes, MP1_amplitudes.transpose(1,0,3,2), atol=1e-10), "MP1 amplitudes do not satisfy antisymmetry t_ij^ab = t_ji^ba!"



		# iii) Compute MP2 correlation energy (spin-orbital indices)
	
		J_2 = 0.0
		
		# Create lists for easier indexing
		occ_indices = self.active_occ_indices
		virt_indices = self.active_inocc_indices
		
		for idx_i, i in enumerate(occ_indices):
			eps_i = eps[i]
			
			for idx_j, j in enumerate(occ_indices):
				if j <= i:  # i > j restriction
					continue
					
				eps_j = eps[j]
				J_ij = 0.0
				
				# First term: Σ_{a>b} Σ_{c>d} t_ij^{ab} t_ij^{cd} [...]
				term1 = 0.0
				
				# Use symmetry: sum over all a,b,c,d but use independent sums
				# This is more efficient than nested loops
				for idx_a, a in enumerate(virt_indices):					
					for idx_b, b in enumerate(virt_indices):
						if b <= a:  # a > b restriction
							continue

						t_abij = MP1_amplitudes[a, b, i, j]
						
						for idx_c, c in enumerate(virt_indices):
							
							for idx_d, d in enumerate(virt_indices):
								if d <= c:  # c > d restriction
									continue
									
								t_cdij = MP1_amplitudes[c, d, i, j]
								
								# Compute the bracket term
								bracket = 0.0
								
								# f_ac δ_bd
								if b == d:
									bracket += Fmo_spin[a, c]
								
								# f_bd δ_ac
								if a == c:
									bracket += Fmo_spin[b, d]
								
								# - f_ad δ_bc
								if b == c:
									bracket -= Fmo_spin[a, d]
								
								# - f_bc δ_ad
								if a == d:
									bracket -= Fmo_spin[b, c]
								
								# - (ε_i + ε_j)(δ_ac δ_bd - δ_ad δ_bc)
								if a == c and b == d:
									bracket -= (eps_i + eps_j)
								if a == d and b == c:
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
		
		print(f"[{len(self.active_inocc_indices)}/{len(self.active_inocc_indices + self.inactive_indices)}]: Computed MP2 correlation energy (spin-orbital): ", J_2)
		
		# Sanity checks
			# Check that MP2 correlation energy is finite
		assert np.isfinite(J_2), "MP2 correlation energy is not finite!"
			# Check that MP2 correlation energy is not positive
		if J_2 > 0:
			print("WARNING: MP2 correlation energy is positive, which is unexpected!")


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
		
		# Precompute D_ab as 2D array
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
								# Same-spin: use antisymmetrized integral <eb||ij>
								# Different-spin: use Coulomb integral <eb|ij> in physicist's notation
							integral = eri_as[E, B, I, J]
							
							term1 += 2.0 * t_ABIJ * integral
				
				# Term2: 2 * Σ_{b} D_ab f_{eb}
				term2 = 0.0
				for idx_B, B in enumerate(self.active_inocc_indices):
					term2 += 2.0 * D_ab[idx_A, idx_B] * Fmo_spin[B, E]
				
				# Combine terms into gradient
				G[idx_A, idx_E] = term1 + term2


		# Check if gradient is reasonable
		if np.linalg.norm(G.flatten()) < 1e-8:
			print("WARNING: Gradient is essentially zero!")
			# Check if gradient is empty
		if np.count_nonzero(G) == 0:
			print("No inactive orbitals -> gradient is empty (expected)")
			print("Orbitals cannot be optimized further (full virtual space)")
			# Check if gradient has finite values
		assert np.all(np.isfinite(G)), "Gradient G contains non-finite values!"



			# Compute Hessian H
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
								integral = eri_as[E, F, I, J]
								
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
										integral_BC = eri_as[B, C, I, J]
									
										t_BCIJ = MP1_amplitudes[B, C, I, J]
										integral_AC = eri_as[A, C, I, J]
										temp2_sum += t_ACIJ * integral_BC + t_BCIJ * integral_AC

									term2 -= temp2_sum
						
						# ===== TERMS 3 & 4 =====
						term3 = 0.0
						term4 = 0.0

							# Precomputed D_ab
						Dab = D_ab[idx_A, idx_B]
						
						if E == F:  # Term 3: Dab * (f_aa - f_bb) δ_EF
							f_aa = Fmo_spin[A, A]
							f_bb = Fmo_spin[B, B]
							term3 = Dab * (f_aa - f_bb)

						if E != F:  # Term 4: Dab * f_ef (1- δ_EF)
							term4 = Dab * Fmo_spin[E, F] 
						
						H[row, col] = term1 + term2 + term3 + term4


		# Symmetry check
		if H.size > 0:
			diff_H = np.max(np.abs(H - H.T))	
			if diff_H > 2.5e-3:
				print(f"WARNING: Hessian is not symmetric! Max diff: {np.max(np.abs(H - H.T)):.6e}")
				


		# Print diagnostics
		print("  Orbital Optimization Diagnostics:")
			# Norm of the gradient
		print(f"    Gradient norm: {np.linalg.norm(G):.6e}")
			# Check eigenvalues
		eigvals = np.linalg.eigvalsh(H)  # eigh for symmetric
		print(f"    Hessian norm: {np.linalg.norm(H):.6e}, (Neg. Eigval: {np.sum(eigvals < 0)}/{len(eigvals)})")





		# Step (vi): Use the Newton-Raphson method to minimize the second-order Hylleraas functional

		# solve for rotation parameters
			# Original direct inversion method
				# equation 14: R = - G H^-1 -> R = -G @ np.linalg.inv(H)
			# Alternatively, use Reduced Linear Equation (RLE) method

		def solve_block_diagonal_RLE(H, G, n_active, n_inactive):
			"""
			Block diagonal RLE method as suggested in Adamowicz & Bartlett.
			Only invert diagonal blocks H_{ee}.
			
			Each block corresponds to excitations from occupied orbitals 
			to one inactive virtual orbital (including spin cases).
			"""
			# Try to identify block structure automatically
				# Find size of blocks in Hessian
			block_size = n_active  # Each block corresponds to one inactive orbital
			n_blocks = n_inactive  # Number of inactive orbitals
			total_size = n_active * n_inactive

			# Reshape H and G for block processing
			if H.shape != (total_size, total_size):
				# Reshape H
				H = H.reshape((total_size, total_size))
			if G.shape != (total_size, ):
				# Flatten G
				G = G.flatten()

			# Initialize R vector
			R = np.zeros(total_size, dtype=np.float64)

			# Create block coupling matrix
			coupling_matrix = np.zeros((n_blocks, n_blocks), dtype=bool)
			
			for i in range(n_blocks):
				for j in range(n_blocks):
					if i == j:
						continue
					# Extract block H[i,j]
					i_start = i * block_size
					i_end = (i + 1) * block_size
					j_start = j * block_size
					j_end = (j + 1) * block_size
					
					H_block_ij = H[i_start:i_end, j_start:j_end]
					block_norm = np.linalg.norm(H_block_ij)
					
					if block_norm > 1e-10:  # Significant coupling
						coupling_matrix[i, j] = True

			# Find connected components (groups of coupled blocks)
			visited = [False] * n_blocks
			block_groups = []
			
			for i in range(n_blocks):
				if not visited[i]:
					# Start new group
					group = []
					stack = [i]
					
					while stack:
						current = stack.pop()
						if not visited[current]:
							visited[current] = True
							group.append(current)
							
							# Add coupled blocks
							for j in range(n_blocks):
								if coupling_matrix[current, j] or coupling_matrix[j, current]:
									if not visited[j]:
										stack.append(j)
					
					block_groups.append(group)

			# Solve each group of coupled blocks
			for group_idx, block_indices in enumerate(block_groups):
				if len(block_indices) == 1:
					# Single block - solve independently
					block_idx = block_indices[0]
					start = block_idx * block_size
					end = (block_idx + 1) * block_size
					
					H_block = H[start:end, start:end]
					G_block = G[start:end]
					
					# Solve this block
						#R_block = np.linalg.solve(H_block, -G_block)
					R_block = - G_block @ np.linalg.inv(H_block)
						
					R[start:end] = R_block
					
				else:
					# Multiple coupled blocks - solve together
					# Collect all indices for this group
					indices = []
					for block_idx in block_indices:
						start = block_idx * block_size
						end = (block_idx + 1) * block_size
						indices.extend(range(start, end))
					
					# Extract submatrix for this group
					H_group = H[np.ix_(indices, indices)]
					G_group = G[indices]
					
					# Solve coupled system
						# R_group = np.linalg.solve(H_group, -G_group)
					R_group = - G_group @ np.linalg.inv(H_group)
					
					# Distribute solution
					R[indices] = R_group


			return R



		# Set to True to use Reduced Linear Equation method
			# Use it when Hessian is ill-conditioned:
		use_RLE = False
			# Set it globally
		self.use_RLE_orbopt = use_RLE

			# Direct inversion method
		if not use_RLE:
			G = G.flatten()  # Flatten G to 1D array

			# Solve for R, unoccupied space
			R = - G @ np.linalg.inv(H) # Direct inversion
			# R = np.linalg.solve(H, -G)
			# R = -scipy.linalg.lstsq(H, G)[0]

			# Reduced Linear Equation method
		if use_RLE:				
			R = solve_block_diagonal_RLE(H, G, len(self.active_inocc_indices), len(self.inactive_indices))
			
		
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

			# Check shape
		expected_shape = np.zeros((len(self.full_indices), len(self.full_indices))).shape
		assert R_matrix.shape == expected_shape, f"R_matrix shape is {R_matrix.shape}, expected {expected_shape}"
			# Check that R is anti-symmetric
		diff_R = np.linalg.norm(R_matrix + R_matrix.T)
		assert diff_R < 1e-6, f"R_matrix is not anti-symmetric, ||R + R.T|| = {diff_R}"
			# Check that R_matrix has no NaN or Inf values
		assert np.all(np.isfinite(R_matrix)), "R_matrix contains NaN or Inf values!"
			# Check that R_matrix is not all zeros
		count_nonzero_R = np.count_nonzero(R_matrix)
		if count_nonzero_R == 0:
			print("WARNING: R_matrix is all zeros!")

		# Convergence check based on max element of R_matrix
		max_R_elem = np.max(np.abs(R_matrix))
		print(f"    Rotation norm {np.linalg.norm(R_matrix):.6e}, (Max el.: {max_R_elem:.6e})")




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



		# Helper functions for MO coefficient conversions
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
			assert C_beta.shape == (n_ao, n_spatial), "Alpha/beta shapes mismatch"

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
				assert len(idx_alpha) == len(idx_beta) == n_spatial, "Unexpected orbspin layout"
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

		print(f"    Rotated {num_rotated} spin orbitals out of {mo_coeffs[0].shape[1] + mo_coeffs[1].shape[1]} total.")

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


				# Are the optimal solution when alpha and beta orbitals are the same? !!!!!!!!

		return mo_coeffs_rot, self.use_RLE_orbopt


	





	def run_ovos(self,  mo_coeffs):
		"""
		Run the OVOS algorithm.
		"""

		converged = False
		max_iter = 500
		iter_count = 0

		while not converged:
			iter_count += 1
			print("#### OVOS Iteration ", iter_count, " ####")
			
			# Step (iii-iv): Compute MP2 correlation energy and amplitudes
			E_corr, MP1_amplitudes, eri_spin, eri_phys, eri_as, Fmo_spin = self.MP2_energy(mo_coeffs = mo_coeffs)

			# Step (ix): check convergence
				# convergence criterion: change in correlation energy < 1e-10 Hartree
			if iter_count > 2:
				threshold = 1e-10
				if np.abs(E_corr - lst_E_corr[-1]) < threshold:
					converged = True
					print("OVOS converged in ", iter_count, " iterations.")

					lst_E_corr.append(E_corr)
					lst_iter_counts.append(iter_count)

					break
				else:
					lst_E_corr.append(E_corr)
					lst_iter_counts.append(iter_count)

			else:
				lst_E_corr = []
				lst_E_corr.append(E_corr)

				lst_iter_counts = []
				lst_iter_counts.append(iter_count)

			if iter_count == max_iter:
				print("Maximum number of iterations reached. OVOS did not converge.")
				lst_E_corr.append(E_corr)
				lst_iter_counts.append(iter_count)
				break
			
			# Step (v-viii): Orbital optimization
			mo_coeffs, use_RLE_orbopt = self.orbital_optimization(mo_coeffs, MP1_amplitudes=MP1_amplitudes, eri_spin=eri_spin, eri_phys=eri_phys, eri_as=eri_as, Fmo_spin=Fmo_spin)



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

		return lst_E_corr, lst_iter_counts, mo_coeffs 
	
	
"""
TESTS !!!!!!!!!!!!!!!!!!!!!!! 
Inspiration from SlowQuant Github...
"""