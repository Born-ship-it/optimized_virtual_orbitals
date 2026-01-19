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
from pyscf.cc.addons import spatial2spin, spin2spatial
from pyscf.scf.addons import convert_to_ghf

from decimal import Decimal


# Options:
np.set_printoptions(precision=4, suppress=True, linewidth=200)

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

		if self.init_orbs == "UHF":
			# MO coefficients (alpha, beta)
			self.mo_coeffs = self.uhf.mo_coeff

		# MP2 calculation
		self.MP2 = self.uhf.MP2().run()
		self.MP2_ecorr = self.MP2.e_corr
		print()
		print("MP2 correlation energy: ", self.MP2_ecorr)

		# Overlap matrix check
		self.S = mol.intor('int1e_ovlp')

		# Sum a coloumn of mo_coeffs to check normalization for a given spin
			# Chekc: C^T S C = I
		for spin in [0, 1]:
			C_i = self.mo_coeffs[spin]
			
			norm = C_i.T @ self.S @ C_i
			assert np.allclose(norm, np.eye(norm.shape[0]), atol=1e-6), f"MO coefficients for spin {spin} are not orthonormal!"
		
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

		# build spin orbital coefficients
			# [0,1,0,1,...] for alpha and beta spin orbitals
		self.orbspin = np.array([0,1]*self.tot_num_spin_orbs)

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

		# Precompute valid I>J, A>B, C>D combinations to avoid redundant calculations
			# Not implemented yet!
		self.active_occ_indices_valid = [(I, J) for I in self.active_occ_indices for J in self.active_occ_indices if I > J]
		self.active_inocc_indices_valid = [(A, B) for A in self.active_inocc_indices for B in self.active_inocc_indices if A > B]
		self.inactive_indices_valid = [(E, F) for E in self.inactive_indices for F in self.inactive_indices if E > F]
			# This lets us transform nested loops like:
			# for A in self.active_inocc_indices:
			# 	for B in self.active_inocc_indices:
			# 		if A > B:
			# into:
			# for (A, B) in self.active_inocc_indices_valid:
			#
			# For different num_opt_virtual_orbs, see below:
			# num_opt_virtual_orbs = 2 → 1 pair (A,B) with A>B
			# num_opt_virtual_orbs = 3 → 3 pairs (A,B) with A>B
			# num_opt_virtual_orbs = 4 → 6 pairs (A,B) with A>B
			# num_opt_virtual_orbs = 5 → 10 pairs (A,B) with A>B
			# num_opt_virtual_orbs = 6 → 15 pairs (A,B) with A>B
			# num_opt_virtual_orbs = 7 → 21 pairs (A,B) with A>B

		# Check if valid indices are correctly built
		for (I, J) in self.active_occ_indices_valid:
			assert I in self.active_occ_indices, f"I={I} not in active_occ_indices"
			assert J in self.active_occ_indices, f"J={J} not in active_occ_indices"
			assert I > J, f"I={I} not greater than J={J}"


		# Print information about the spaces
		print()
		print("#### Active and inactive spaces ####")
		print("Total number of spin-orbitals: ", self.tot_num_spin_orbs)
		print("Active occupied spin-orbitals: ", self.active_occ_indices)
		print("Active unoccupied spin-orbitals: ", self.active_inocc_indices)
		print("Inactive unoccupied spin-orbitals: ", self.inactive_indices)
		print()



		# Check that the number of optimized virtual orbitals is not too large
		assert self.tot_num_spin_orbs >= self.num_opt_virtual_orbs+self.nelec, "Your space 'num_opt_virtual_orbs' is too large"  


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

		norb_alpha = mo_coeffs[0].shape[1]
		norb_beta = mo_coeffs[1].shape[1]

		# Fock matrix in MO basis 
		Fmo_a = mo_coeffs[0].T @ self.Fao[0] @ mo_coeffs[0]
		Fmo_b = mo_coeffs[1].T @ self.Fao[1] @ mo_coeffs[1]
		Fmo = (Fmo_a, Fmo_b)

		# Orbital energies (spin-orbital representation)
		eigval_a, eigvec_a = scipy.linalg.eig(Fmo_a)
		eigval_b, eigvec_b = scipy.linalg.eig(Fmo_b)
		sorting_a = np.argsort(eigval_a)
		sorting_b = np.argsort(eigval_b)
		mo_energy_a = np.real(eigval_a[sorting_a])
		mo_energy_b = np.real(eigval_b[sorting_b])
		orbital_energies = []
		for i in range(eigval_a.shape[0]):
			orbital_energies.append(float(mo_energy_a[i]))
			orbital_energies.append(float(mo_energy_b[i]))
		
		# Numerical checks
			# Check that orbital energies are sorted for each spin
		assert np.all(np.diff(mo_energy_a) >= 0), "Alpha orbital energies are not sorted!"
		assert np.all(np.diff(mo_energy_b) >= 0), "Beta orbital energies are not sorted!"
			# Check that orbital energies are finite
		assert np.all(np.isfinite(orbital_energies)), "Orbital energies contain non-finite values!"
			# Check lengths
		expected_length = norb_alpha + norb_beta
		assert len(orbital_energies) == expected_length, f"Orbital energies length is {len(orbital_energies)}, expected {expected_length}"
			




		#PySCF stores 2e integrals in chemists' notation: (ij|kl) = <ik|jl> in physicists' notation.

		# (alpha alpha | alpha alpha) integrals
		eri_aaaa = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[0], mo_coeffs[0], mo_coeffs[0], mo_coeffs[0]], compact=False)
		#eri_aaaa = eri_aaaa.reshape(norb_alpha, norb_alpha, norb_alpha, norb_alpha)

		# (beta beta | beta beta) integrals
		eri_bbbb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[1], mo_coeffs[1], mo_coeffs[1], mo_coeffs[1]], compact=False)
		#eri_bbbb = eri_bbbb.reshape(norb_beta, norb_beta, norb_beta, norb_beta)

		# (alpha alpha | beta beta) integrals
		# These are the (ij|kl) where i,j are alpha, k,l are beta
		eri_aabb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[0], mo_coeffs[0], mo_coeffs[1], mo_coeffs[1]], compact=False)
		#eri_aabb = eri_aabb.reshape(norb_alpha, norb_alpha, norb_beta, norb_beta)

		norb_total = norb_alpha + norb_beta
		#eri_spin = np.zeros((norb_total, norb_total, norb_total, norb_total))

		#See https://pyscf.org/_modules/pyscf/cc/addons.html#spatial2spin
		eri_spin = spatial2spin([eri_aaaa, eri_aabb, eri_bbbb], orbspin=None)
		Fmo_spin = spatial2spin([Fmo[0], Fmo[1]], orbspin=None)

		# 	# Check symmetry of eri_spin
		# for p in range(norb_total):
		# 	for q in range(norb_total):
		# 		for r in range(norb_total):
		# 			for s in range(norb_total):
		# 				# Symmetry 1: Always valid
		# 				assert np.isclose(eri_spin[p,q,r,s], eri_spin[q,p,s,r], atol=1e-6), \
		# 					f"eri_spin symmetry failed: (pq|rs) != (qp|sr) for p={p}, q={q}, r={r}, s={s}!"
						
		# 				# Symmetry 2: Only for same-spin quartets
		# 				if self.orbspin[p] == self.orbspin[q] == self.orbspin[r] == self.orbspin[s]:
		# 					assert np.isclose(eri_spin[p,q,r,s], eri_spin[r,s,p,q], atol=1e-6), \
		# 						f"eri_spin symmetry failed: (pq|rs) != (rs|pq) for same-spin p={p}, q={q}, r={r}, s={s}!"



		# Initialize MP1 amplitudes array
		MP1_amplitudes = np.zeros((norb_total, norb_total, norb_total, norb_total))

		# Define t1 function for MP1 amplitudes
		def t1(I,J,A,B) -> float:
			#MP1 amplitudes:
			t1 = -1.0*( (eri_spin[A,I,B,J]- eri_spin[A,J,B,I])
			/ (orbital_energies[A] + orbital_energies[B] 
			- orbital_energies[I] - orbital_energies[J]) )
			return t1 

		# Build MP1 amplitudes
		for I in self.active_occ_indices:
			for J in self.active_occ_indices:
				for A in self.active_inocc_indices:
					for B in self.active_inocc_indices:
						MP1_amplitudes[A,I,B,J] = t1(I,J,A,B)

			# Check antisymmetry of MP1_amplitudes
		for I in self.active_occ_indices:
			for J in self.active_occ_indices:
				for A in self.active_inocc_indices:
					for B in self.active_inocc_indices:

						t_abij = MP1_amplitudes[A,I,B,J]

						# Antisymmetry check: t_abij = - t_abji
						t_abji = MP1_amplitudes[A,J,B,I]
						# Antismmetry check: t_abij = - t_baij
						t_baij = MP1_amplitudes[B,I,A,J]

						# Assert antisymmetry
						assert np.isclose(t_abij, -t_abji, atol=1e-6), f"MP1_amplitudes antisymmetry failed: t_abij != - t_abji for I={I}, J={J}, A={A}, B={B}!"
						assert np.isclose(t_abij, -t_baij, atol=1e-6), f"MP1_amplitudes antisymmetry failed: t_abij != - t_baij for I={I}, J={J}, A={A}, B={B}!"

						# # Print check results
						# if not np.isclose(t_abij, -t_abji, atol=1e-10):
						# 	print("t_abij = - t_abji:", '%.2E' % Decimal(t_abij + t_abji))
						# if not np.isclose(t_abij, -t_baij, atol=1e-10):
						# 	print("t_abij = - t_baij:", '%.2E' % Decimal(t_abij + t_baij))
						


		# # Sanity check for t1 function
		# 	# For initial UHF orbitals, compare t1 amplitudes with PySCF MP2 amplitudes
		# if self.init_orbs == "UHF" and self.num_opt_virtual_orbs == 2:			
		# 		# Compare with PySCF MP2 amplitudes for a sample set of indices
		# 			# Initial count of failed checks
		# 	failed_checks = 0

		# 			# Total number of spin orbitals combinations to check
		# 	total_checks = len(self.active_occ_indices_valid)*len(self.active_inocc_indices_valid)
		# 	current_check = 0
			
		# 			# Get all indices I,J,A,B to test
		# 	for (I, J) in self.active_occ_indices_valid:
		# 		for (A, B) in self.active_inocc_indices_valid:

		# 			current_check += 1
					
		# 			sample_t1 = t1(I, J, A, B)

		# 				# Check that t1 amplitude is finite
		# 			assert np.isfinite(sample_t1), f"t1 amplitude for I={I}, J={J}, A={A}, B={B} is not finite!"
			
		# 				# Energy denominator should not be zero
		# 			denom = orbital_energies[A] + orbital_energies[B] - orbital_energies[I] - orbital_energies[J]
		# 			assert denom != 0, f"Energy denominator for I={I}, J={J}, A={A}, B={B} is zero!"

		# 				# Antisymmetry check: t1(I,J,A,B) = - t1(J,I,A,B)
		# 			t1_ij, t1_ji  = t1(I, J, A, B), t1(J, I, A, B)
		# 			assert np.isclose(t1_ij, -t1_ji, atol=1e-10), f"t1 antisymmetry failed for I={I}, J={J}, A={A}, B={B}!"

		# 				# MP2.t2 is tuple: (t2_aaaa, t2_aabb, t2_bbbb)
					
		# 					# Determine which component based on spin combination
		# 			spatial_A = A // 2
		# 			spatial_B = B // 2
		# 			spatial_I = I // 2
		# 			spatial_J = J // 2

		# 					# Correct for PySCF virtual orbital ordering
		# 						# Get n_occ_alpha and n_occ_beta
		# 			n_occ_alpha = self.mol.nelec[0]
		# 			n_occ_beta = self.mol.nelec[1]

		# 						# Check spin combination - PySCF only stores 3 types: (αα|αα), (αα|ββ), (ββ|ββ)
		# 			spin_I = self.orbspin[I]
		# 			spin_J = self.orbspin[J]
		# 			spin_A = self.orbspin[A]
		# 			spin_B = self.orbspin[B]
			
		# 						# Get MP2 t2 amplitude from PySCF
		# 			MP2_t2_amp = None
		# 			spin_combination = None

		# 			if spin_I == 0 and spin_J == 0 and spin_A == 0 and spin_B == 0:
		# 					# (αα|αα) - stored in t2[0]
		# 				spin_combination = "(αα|αα)"
		# 				spatial_A -= n_occ_alpha
		# 				spatial_B -= n_occ_alpha
		# 				MP2_t2_amp = self.MP2.t2[0][spatial_I, spatial_J, spatial_A, spatial_B]

		# 			elif spin_I == 1 and spin_J == 1 and spin_A == 1 and spin_B == 1:
		# 					# (ββ|ββ) - stored in t2[2]
		# 				spin_combination = "(ββ|ββ)"
		# 				spatial_A -= n_occ_beta
		# 				spatial_B -= n_occ_beta
		# 				MP2_t2_amp = self.MP2.t2[2][spatial_I, spatial_J, spatial_A, spatial_B]

		# 			elif spin_I == 0 and spin_J == 0 and spin_A == 1 and spin_B == 1:
		# 					# (αα|ββ) - stored in t2[1]
		# 				spin_combination = "(αα|ββ)"
		# 				spatial_A -= n_occ_beta
		# 				spatial_B -= n_occ_beta
		# 				MP2_t2_amp = self.MP2.t2[1][spatial_I, spatial_J, spatial_A, spatial_B]
						
		# 			elif spin_I == 1 and spin_J == 1 and spin_A == 0 and spin_B == 0:
		# 					# (ββ|αα) - also stored in t2[1]
		# 				spin_combination = "(ββ|αα)"
		# 				spatial_A -= n_occ_alpha
		# 				spatial_B -= n_occ_alpha
		# 				MP2_t2_amp = self.MP2.t2[1][spatial_I, spatial_J, spatial_A, spatial_B]
		# 			else:
		# 					# Mixed-spin combinations like (αα|βα) are not stored in PySCF MP2.t2
		# 				spin_str = f"({'α' if spin_I==0 else 'β'}{'α' if spin_J==0 else 'β'}|{'α' if spin_A==0 else 'β'}{'α' if spin_B==0 else 'β'})"
		# 				spin_combination = f"mixed-{spin_str}"
					
		# 				# Print check results
					
		# 			if MP2_t2_amp is not None and not np.isclose(sample_t1, MP2_t2_amp, atol=1e-8):
		# 				failed_checks += 1
		# 				# print()
		# 				# print(f"  Spin combination: {spin_combination}")
		# 				# print(f"  Indices: I={I} ({'α' if spin_I==0 else 'β'}), J={J} ({'α' if spin_J==0 else 'β'}), A={A} ({'α' if spin_A==0 else 'β'}), B={B} ({'α' if spin_B==0 else 'β'})")
		# 				# print(f"  Checking t2 amplitude for spatial: I={spatial_I}, J={spatial_J}, A={spatial_A}, B={spatial_B}...")
		# 				# print(f"  Diff = {sample_t1 - MP2_t2_amp}")
		# 				# assert np.isclose(sample_t1, MP2_t2_amp, atol=1e-8), f"t1 amplitude {sample_t1} does not match PySCF MP2 amplitude {MP2_t2_amp} for I={I}, J={J}, A={A}, B={B}!"
		# 	# Summary of t1 amplitude checks
		# 	assert failed_checks == 0, f"{failed_checks} out of {total_checks} t1 amplitude checks failed against PySCF MP2 amplitudes!"






		# Compute MP2 correlation energy
			# Equation 10: J_2 = ...
				# Term 1: \sum_{I>J} \sum_{A>B} \sum_{C>D} t_ABIJ t_CDIJ * ( F_AC \delta_BD + F_BD \delta_BC - F_AD \delta_BC - F_BC \delta_AD - (e_I + e_J) (\delta_AC \delta_BD - \delta_AD \delta_BC) )
				# Term 2: \sum_{I>J} \sum_{A>B} 2 t_ABIJ ( <AI|BJ> - <AJ|BI> )

		J_2 = 0
		for (I, J) in self.active_occ_indices_valid:
			first_term = 0
			for (A, B) in self.active_inocc_indices_valid:
				for (C, D) in self.active_inocc_indices_valid:

					t_abij = t1(I=I,J=J,A=A,B=B)
					t_cdij = t1(I=I,J=J,A=C,B=D)

					if B==D: # delta_BD
						first_term += t_abij*t_cdij*Fmo_spin[A,C]
					if A==C: # delta_AC
						first_term += t_abij*t_cdij*Fmo_spin[B,D]
					if B==C: # delta_BC
						first_term += -1.0*t_abij*t_cdij*Fmo_spin[A,D]
					if A==D: # delta_AD
						first_term += -1.0*t_abij*t_cdij*Fmo_spin[B,C]
					if A==C and B==D: # delta_AC delta_BD
						first_term += -1.0*t_abij*t_cdij*(
							orbital_energies[I] + orbital_energies[J])
					if A==D and B==C: # delta_AD delta_BC
						first_term += t_abij*t_cdij*(
							orbital_energies[I] + orbital_energies[J])

			second_term = 0
			for (A, B) in self.active_inocc_indices_valid:
				second_term += 2*t1(I=I,J=J,A=A,B=B)*(eri_spin[A,I,B,J]- eri_spin[A,J,B,I])
			
			J_2 += first_term+second_term

		# MP2 = self.uhf.MP2().run()
		# assert np.abs(J_2 - MP2.e_corr) < 1e-6, "|J_2 - E_corr_MP2| < 1e-6 !!!"  

		# Check that MP1_amplitudes is not all zeros
		assert not np.allclose(MP1_amplitudes, 0), "MP1_amplitudes is all zeros!"
		# Check if MP1_amplitudes has NaN or Inf values
		assert np.all(np.isfinite(MP1_amplitudes)), "MP1_amplitudes contains NaN or Inf values"

		# Check that eri_spin is not all zeros
		assert not np.allclose(eri_spin, 0), "eri_spin is all zeros!"
		# Check if eri_spin has NaN or Inf values
		assert np.all(np.isfinite(eri_spin)), "eri_spin contains NaN or Inf values!"

		# Check that Fmo_spin is not all zeros
		assert not np.allclose(Fmo_spin, 0), "Fmo_spin is all zeros!"
		# Check if Fmo_spin has NaN or Inf values
		assert np.all(np.isfinite(Fmo_spin)), "Fmo_spin contains NaN or Inf values!"

		return J_2, MP1_amplitudes, eri_spin, Fmo_spin





	def orbital_optimization(self, mo_coeffs, MP1_amplitudes, eri_spin, Fmo_spin) -> np.ndarray:

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

		# # Helper function to access MP1 amplitudes with proper antisymmetry handling
		# def get_t1_amplitude(A: int, I: int, B: int, J: int) -> float:
		# 	if A > B:
		# 		return MP1_amplitudes[A,I,B,J]
		# 	elif A < B:
		# 		return -MP1_amplitudes[B,I,A,J]
		# 	else:  # A == B (Pauli exclusion)
		# 		return 0.0

		# 	# Verify that get_t1_amplitude respects antisymmetry
		# for (I, J) in self.active_occ_indices_valid:
		# 	for (A, B) in self.active_inocc_indices_valid:
		# 		t_abij = get_t1_amplitude(A, I, B, J)
		# 		t_baij = get_t1_amplitude(B, I, A, J)
		# 		assert np.isclose(t_abij, -t_baij, atol=1e-10), f"get_t1_amplitude antisymmetry failed for I={I}, J={J}, A={A}, B={B}!"





		# Step (v): Compute the gradient and Hessian of the second-order Hylleraas functional

		# Precompute D_AB matrix
			# Equation 13: D_AB = 2 \sum_{I>J} \sum_C t_ABIJ t_CBJI
				# Eqs. 12a and 12b use D_AB

			# For whole unoccupied space (inefficient)
		n_active = len(self.active_inocc_indices)
		D_AB_cache = np.zeros((n_active * n_active))
		for idx_A, A in enumerate(self.active_inocc_indices):
			for idx_B, B in enumerate(self.active_inocc_indices):
				D_AB = 0
				for I in self.active_occ_indices:
					for J in self.active_occ_indices:
						if I > J:
							for C in self.active_inocc_indices:
								D_AB += 2.0*MP1_amplitudes[A,I,C,J]*MP1_amplitudes[B,I,C,J]

				flat_index_D = idx_A * n_active + idx_B
				D_AB_cache[flat_index_D] = D_AB
			# Check that D_AB_cache is symmetric
		D_AB_2D = D_AB_cache.reshape(n_active, n_active)
		assert np.allclose(D_AB_2D, D_AB_2D.T), "D_AB_cache is not symmetric!"








		# Define gradient functions
		def gradient(E: int, A: int, idx_A: int) -> float:
			#Equation 12a: G_EA = ...
				# Term 1: 2 \sum_{I>J} \sum_B t_ABIJ ( <EI|BJ> - <EJ|BI> )
				# Term 2: + 2 \sum_B D_AB F_EB

			first_term = 0
			for (I, J) in self.active_occ_indices_valid:
				for B in self.active_inocc_indices:
					first_term += 2.0*MP1_amplitudes[A,I,B,J]*(eri_spin[E,I,B,J] - eri_spin[E,J,B,I])

			second_term = 0
			for idx_B, B in enumerate(self.active_inocc_indices):
				flat_index_D = idx_A * n_active + idx_B
				D_AB = D_AB_cache[flat_index_D]
				second_term += 2.0*D_AB*Fmo_spin[E,B]

			return first_term + second_term

		# Define Hessian function
		def hessian(E: int, A: int, F: int, B: int, idx_A: int, idx_B: int) -> float:
			#Equation 12b: H_EA,FB = ...
				# Term 1: 2 \sum_{I>J} t_ABIJ ( <EI|FJ> - <EJ|FI> )
				# Term 2: - 1.0 * \sum_{I>J} \sum_C [ t_ACIJ ( <BI|CJ> - <BJ|CI> ) + t_CBIJ ( <CI|AJ> - <CJ|AI> ) ] \delta_EF
				# Term 3: + D_AB ( F_AA - F_BB ) \delta_EF 
				# Term 4: + D_AB F_EF
				# Term 5: - 1.0 * D_AB F_EF \delta_EF

			# Get D_AB from cache
			flat_index_D = idx_A * len(self.active_inocc_indices) + idx_B
			D_AB = D_AB_cache[flat_index_D]

			# Calculate each term separately
				# Term 1, symmetric
			first_term = 0
			for I in self.active_occ_indices:
				for J in self.active_occ_indices:
					if I > J:
						first_term += 2.0*MP1_amplitudes[A,I,B,J]*(eri_spin[E,I,F,J] - eri_spin[E,J,F,I])

				# Term 2, symmetric
			second_term = 0			
			for I in self.active_occ_indices:
				for J in self.active_occ_indices:
					if I > J:
						for C in self.active_inocc_indices:
							if E==F:
								second_term +=-1.0*(
									MP1_amplitudes[A,I,C,J]*(eri_spin[B,I,C,J] - eri_spin[B,J,C,I]) 
									+ MP1_amplitudes[C,I,B,J]*(eri_spin[C,I,A,J] - eri_spin[C,J,A,I])
								)
				
				# Term 3 and 5, symmetric
			third_term = 0
			fifth_term = 0
			if E==F:
				third_term += D_AB*(Fmo_spin[A,A] - Fmo_spin[B,B])
				fifth_term += -1.0*D_AB*Fmo_spin[E,F] 

				# Term 4, symmetric
			forth_term = 0
			forth_term = D_AB*Fmo_spin[E,F]

			return first_term + second_term + third_term + forth_term + fifth_term

			# Check how many times \delta_EF is used
		delta_EF_count = 0
		for idx_E, E in enumerate(self.inactive_indices):
			for idx_F, F in enumerate(self.inactive_indices):
				if E==F:
					delta_EF_count += 1
		expected_delta = len(self.inactive_indices)  # Should equal number of E indices
		assert expected_delta == expected_delta, f"δ_EF triggered {expected_delta} times, expected {expected_delta}"	


		# # Test hessian function symmetry directly
		# print("\n=== Testing Hessian function symmetry ===")
		# test_cases = [
		# 	(self.inactive_indices[0], self.active_inocc_indices[0], 
		# 	self.inactive_indices[1], self.active_inocc_indices[1]),
		# ]

		# for E, A, F, B in test_cases:
		# 	idx_A = self.active_inocc_indices.index(A)
		# 	idx_B = self.active_inocc_indices.index(B)
		# 	idx_E = self.inactive_indices.index(E)
		# 	idx_F = self.inactive_indices.index(F)
			
		# 	H_EAFB = hessian(E, A, F, B, idx_A, idx_B)
		# 	H_FBEA = hessian(F, B, E, A, idx_B, idx_A)
			
		# 	print(f"H[E={E},A={A},F={F},B={B}] = {H_EAFB:.8f}")
		# 	print(f"H[F={F},B={B},E={E},A={A}] = {H_FBEA:.8f}")
		# 	print(f"Difference: {abs(H_EAFB - H_FBEA):.2e}")
			
		# 	if not np.isclose(H_EAFB, H_FBEA, atol=1e-8):
		# 		print("❌ ASYMMETRIC - problem is in hessian function!")
				
		# 		# Test each term separately
		# 		print("\nDebugging individual terms:")
		# 		# You can manually compute each term for both orderings here
		# 	else:
		# 		print("✓ Symmetric - problem is in matrix building")


		# Build gradient and Hessian matrices
			# Build gradient matrix G[A, E] and Hessian tensor H[A, E, B, F]
		n_active = len(self.active_inocc_indices)
		n_inactive = len(self.inactive_indices)
		
		G = np.zeros((n_active * n_inactive))
		H = np.zeros((n_active * n_inactive, n_active * n_inactive))

		for idx_A, A in enumerate(self.active_inocc_indices):
			for idx_E, E in enumerate(self.inactive_indices):
			
				# Gradient G[E,A]
				flat_index_G = idx_A * n_inactive + idx_E
				G[flat_index_G] = gradient(E, A, idx_A)

				for idx_B, B in enumerate(self.active_inocc_indices):
					for idx_F, F in enumerate(self.inactive_indices):
					
						# Hessian H[E,A,F,B]
						flat_index_H = (idx_A * n_inactive + idx_E, idx_B * n_inactive + idx_F)
						H[flat_index_H] = hessian(E, A, F, B, idx_A, idx_B)
	

		# Ensure float64 for numerical stability
		# H = H.astype(np.float128)
		# G = G.astype(np.float128)

		# Reshape G and H for Newton-Raphson
		# G = G.reshape(n_active * n_inactive)
		# H = H.reshape(n_active * n_inactive, n_active * n_inactive)

		# Debugging symmetry of H
		print(f"Hessian H: \n {H}")

		# After building H matrix, before symmetry check:
		# print("\n=== Checking matrix building ===")
		
			# # Test all combinations of (E,A) and (F,B) of H matrix against hessian function
		# for idx_i_test in range(n_active * n_inactive):
		# 	for idx_j_test in range(n_active * n_inactive):
		# 		i_test = idx_i_test
		# 		j_test = idx_j_test

		# 		E_i = self.inactive_indices[i_test % n_inactive]
		# 		A_i = self.active_inocc_indices[i_test // n_inactive]
		# 		F_j = self.inactive_indices[j_test % n_inactive]
		# 		B_j = self.active_inocc_indices[j_test // n_inactive]

		# 		H_ij_matrix = H[i_test, j_test]
		# 		H_ij_function = hessian(
		# 			E=E_i,
		# 			A=A_i,
		# 			F=F_j,
		# 			B=B_j,
		# 			idx_A=i_test // n_inactive,
		# 			idx_B=j_test // n_inactive,
		# 		)

		# 		if not np.isclose(H_ij_matrix, H_ij_function, atol=1e-8):
		# 			print(f"❌ Mismatch at H[{i_test},{j_test}] for (E={E_i},A={A_i},F={F_j},B={B_j}):")
		# 			print(f"   From matrix: {H_ij_matrix:.8f}")
		# 			print(f"   From function: {H_ij_function:.8f}")
		# 			print(f"   Difference: {abs(H_ij_matrix - H_ij_function):.2e}")

		# 	# Check symmetry of H matrix 
		# 		# For all combinations of (E,A) and (F,B), check H[E,A,F,B] == H[F,B,E,A]
		# symmetry_failures = 0
		# for idx_i in range(n_active * n_inactive):
		# 	for idx_j in range(n_active * n_inactive):
		# 		H_ij = H[idx_i, idx_j]
		# 		H_ji = H[idx_j, idx_i]

		# 		if not np.isclose(H_ij, H_ji, atol=1e-8):
		# 			symmetry_failures += 1

		# print(f"\nTotal symmetry failures: {symmetry_failures}")
		
		# # Diagnose worst case: i=0, j=27
		# print("\n=== Diagnosing worst asymmetry case ===")
		# print("Case: i=0, j=27 → (E=6,A=4,F=17,B=5)")
		
		# idx_A_0 = 0  # A=4 is first active virtual
		# idx_B_0 = 1  # B=5 is second active virtual
		
		# H_0_27_direct = hessian(6, 4, 17, 5, idx_A_0, idx_B_0)
		# H_27_0_direct = hessian(17, 5, 6, 4, idx_B_0, idx_A_0)
		
		# print(f"hessian(E=6, A=4, F=17, B=5) = {H_0_27_direct:.8e}")
		# print(f"hessian(E=17, A=5, F=6, B=4) = {H_27_0_direct:.8e}")
		# print(f"Function asymmetry: {abs(H_0_27_direct - H_27_0_direct):.2e}")
		# print(f"Is hessian function symmetric? {np.isclose(H_0_27_direct, H_27_0_direct, atol=1e-8)}")
		
		# # Check if E≠F condition matters
		# print(f"\nE=6, F=17 → E≠F (δ_EF = 0)")
		# print("Terms with δ_EF should be zero")

				# Diagnose worst case: i=0, j=27
		print("\n=== Diagnosing worst asymmetry case ===")
		print("Case: i=0, j=27 → (E=6,A=4,F=17,B=5)")

		idx_A_0 = 0  # A=4 is first active virtual
		idx_B_0 = 1  # B=5 is second active virtual

		# Compute each term separately for both orderings
		def hessian_terms(E, A, F, B, idx_A, idx_B):
			"""Return all 5 terms of hessian separately"""
			flat_index_D = idx_A * n_active + idx_B
			D_AB = D_AB_cache[flat_index_D]
			
			# Term 1
			term1 = 0
			for (I, J) in self.active_occ_indices_valid:
				term1 += 2.0*MP1_amplitudes[A,I,B,J]*(eri_spin[E,I,F,J] - eri_spin[E,J,F,I])
			
			# Term 2
			term2 = 0
			if E == F:
				for (I, J) in self.active_occ_indices_valid:
					for C in self.active_inocc_indices:
						term2 += -1.0*(MP1_amplitudes[A,I,C,J]*(eri_spin[B,I,C,J] - eri_spin[B,J,C,I]) + MP1_amplitudes[C,I,B,J]*(eri_spin[C,I,A,J] - eri_spin[C,J,A,I]))
			
			# Terms 3, 4, 5
			term3 = D_AB*(Fmo_spin[A,A] - Fmo_spin[B,B]) if E == F else 0
			term4 = D_AB*Fmo_spin[E,F]
			term5 = -D_AB*Fmo_spin[E,F] if E == F else 0
			
			return term1, term2, term3, term4, term5

		t1_a, t2_a, t3_a, t4_a, t5_a = hessian_terms(6, 4, 17, 5, idx_A_0, idx_B_0)
		t1_b, t2_b, t3_b, t4_b, t5_b = hessian_terms(17, 5, 6, 4, idx_B_0, idx_A_0)

		print(f"\nhessian(E=6, A=4, F=17, B=5):")
		print(f"  Term 1: {t1_a:.8e}")
		print(f"  Term 2: {t2_a:.8e}")
		print(f"  Term 3: {t3_a:.8e}")
		print(f"  Term 4: {t4_a:.8e}")
		print(f"  Term 5: {t5_a:.8e}")
		print(f"  Total:  {sum([t1_a, t2_a, t3_a, t4_a, t5_a]):.8e}")

		print(f"\nhessian(E=17, A=5, F=6, B=4):")
		print(f"  Term 1: {t1_b:.8e}")
		print(f"  Term 2: {t2_b:.8e}")
		print(f"  Term 3: {t3_b:.8e}")
		print(f"  Term 4: {t4_b:.8e}")
		print(f"  Term 5: {t5_b:.8e}")
		print(f"  Total:  {sum([t1_b, t2_b, t3_b, t4_b, t5_b]):.8e}")

		print(f"\nTerm differences:")
		print(f"  Term 1 diff: {abs(t1_a - t1_b):.8e}")
		print(f"  Term 2 diff: {abs(t2_a - t2_b):.8e}")
		print(f"  Term 3 diff: {abs(t3_a - t3_b):.8e}")
		print(f"  Term 4 diff: {abs(t4_a - t4_b):.8e}")
		print(f"  Term 5 diff: {abs(t5_a - t5_b):.8e}")



		# We loop over all combinations of (E,A) and (F,B) to fill G and H
			# Error: H is not symmetric! - We are seeing zero for all upper off-diagonal elements
				# H matrix is of shape (n_active*n_inactive, n_active*n_inactive)
					# Upper off-diagonal elements are H[i,j] where i < j
				# Check if H[i,j] == H[j,i] for all i,j
					# If not, find which elements are not equal
		# failed_symmetry_checks = 0
		
		# for i in range(len(self.active_inocc_indices)*len(self.inactive_indices)):
		# 	for j in range(len(self.active_inocc_indices)*len(self.inactive_indices)):
		# 		if i < j:
		# 			if not np.isclose(hessian(
		# 				E=self.inactive_indices[i % len(self.inactive_indices)],
		# 				A=self.active_inocc_indices[i // len(self.inactive_indices)],
		# 				F=self.inactive_indices[j % len(self.inactive_indices)],
		# 				B=self.active_inocc_indices[j // len(self.inactive_indices)],
		# 				idx_A=i // len(self.inactive_indices),
		# 				idx_B=j // len(self.inactive_indices),
		# 			),
		# 			hessian(
		# 				E=self.inactive_indices[j % len(self.inactive_indices)],
		# 				A=self.active_inocc_indices[j // len(self.inactive_indices)],
		# 				F=self.inactive_indices[i % len(self.inactive_indices)],
		# 				B=self.active_inocc_indices[i // len(self.inactive_indices)],
		# 				idx_A=j // len(self.inactive_indices),
		# 				idx_B=i // len(self.inactive_indices),
		# 			)):
		# 				# print(f"Hessian asymmetry found at indices i={i}, j={j}")
		# 				# print(f"  H[{i},{j}] != H[{j},{i}]"+f"  H[{i},{j}] = {hessian(E=self.inactive_indices[i % len(self.inactive_indices)], A=self.active_inocc_indices[i // len(self.inactive_indices)], F=self.inactive_indices[j % len(self.inactive_indices)], B=self.active_inocc_indices[j // len(self.inactive_indices)], idx_A=i // len(self.inactive_indices), idx_B=j // len(self.inactive_indices))}"+f"  H[{j},{i}] = {hessian(E=self.inactive_indices[j % len(self.inactive_indices)], A=self.active_inocc_indices[j // len(self.inactive_indices)], F=self.inactive_indices[i % len(self.inactive_indices)], B=self.active_inocc_indices[i // len(self.inactive_indices)], idx_A=j // len(self.inactive_indices), idx_B=i // len(self.inactive_indices))}")
		# 				failed_symmetry_checks += 1
		# 	# Final assertion
		# assert failed_symmetry_checks == 0, f"{failed_symmetry_checks} Hessian asymmetry checks failed! \n Symmetry: {np.allclose(H, H.T, atol=1e8)};"	


		# For G all zeros check, this means we have converged
		if np.allclose(G, 0):
			print("Gradient G is all zeros. Orbital optimization has converged.")
			return mo_coeffs  # Return original MO coefficients if converged

		# Numerical checks
			# Check that G is not all zeros
		assert not np.allclose(G, 0), "Gradient G is all zeros!"
			# Check normal of G
		norm_G = np.linalg.norm(G)
		assert norm_G > 1e-8, f"Norm of gradient G is too small: {norm_G}"
			# Check that H is symmetric
		assert np.allclose(H, H.T, atol= 1e-8), f"Hessian H is not symmetric! [max(H-H.T) = {np.max(H-H.T):.2e}]"
			# Check that H is not all zeros
		assert not np.allclose(H, 0), "Hessian H is all zeros!"
			# Check if H is invertible by checking its condition number
		cond_H = np.linalg.cond(H)
		#assert cond_H < 1e12, f"Hessian H is ill-conditioned (cond = {cond_H}). Cannot invert."
			# Check if H has NaN or Inf values
		assert np.all(np.isfinite(H)), "Hessian H contains NaN or Inf values"
			# Check if G has NaN or Inf values
		assert np.all(np.isfinite(G)), "Gradient G contains NaN or Inf values"

		# Shapes of G and H for checking
		# np.set_printoptions(precision=4, suppress=True, linewidth=200)
		# print("G shape: ", G.shape)
		# print("G matrix: \n", G)
		# print("H shape: ", H.shape)
		# print("H tensor: \n", H)


		# Step (vi): Use the Newton-Raphson method to minimize the second-order Hylleraas functional

		# solve for rotation parameters
			# Original direct inversion method
				# equation 14: R = - G H^-1 -> R = -G @ np.linalg.inv(H)


		# Set to True to use Reduced Linear Equation method
		use_RLE = False  


		# Direct inversion method
		if not use_RLE:
			# Solve for R, unoccupied space
			R = - G @ np.linalg.inv(H)
			#R = - np.linalg.solve(H, G)
			# print("R shape (direct inversion): ", R.shape)
			# print("R matrix (direct inversion): \n", R)
			
			# Initialize R,
				# Matrix: self.full_indices x self.full_indices
			R_matrix = np.zeros((len(self.full_indices), len(self.full_indices)))

			# Build R_matrix from R[A, E]
			for idx_A, A in enumerate(self.active_inocc_indices):
				for idx_E, E in enumerate(self.inactive_indices):
					
					# Check indixes
					# print(f"Setting R_matrix for A={A}, E={E} from R[{idx_A}, {i_E}] = {R[idx_A, i_E]}")

					# R[A, E] gives the rotation parameter
						# Flat indexing: R[idx_A * idx_E]
					flat_index_R = idx_A * len(self.inactive_indices) + idx_E
					R_matrix[E, A] = R[flat_index_R]  
					R_matrix[A, E] = -R[flat_index_R]  # Anti-symmetry


			

		# Try the option of implementing the RLE for R,
			# as described in the paper, instead of direct inversion of H.
		if use_RLE:	
			# ...
			raise NotImplementedError("Reduced Linear Equation (RLE) method not implemented yet.")
		

		# Numerical checks on R_matrix
			# Typeset R to float64 for numerical stability
		# R_matrix = R_matrix.astype(np.float128)
			# Check shape
		expected_shape = np.zeros((len(self.full_indices), len(self.full_indices))).shape
		assert R_matrix.shape == expected_shape, f"R_matrix shape is {R_matrix.shape}, expected {expected_shape}"
			# Print shape of R_matrix for checking
		# print("R_matrix shape: ", R_matrix.shape)
			# Pretty print R for checking
		# print("Orbital rotation parameter matrix R: \n", R_matrix)
			# Check that R is anti-symmetric
		diff_R = np.linalg.norm(R_matrix + R_matrix.T)
		assert diff_R < 1e-6, f"R_matrix is not anti-symmetric, ||R + R.T|| = {diff_R}"
			# Print norm of R for checking
		norm_R = np.linalg.norm(R_matrix)
		# print("Norm of R_matrix: ", norm_R)




		# Step (vii): Construct the unitary orbital rotation matrix U = exp(R)

		# Unitary rotation matrix
		U = scipy.linalg.expm(R_matrix)

		# Typeset U to float64 for numerical stability
		# U = U.astype(np.float128)

		# Numerical checks on U
			# Check shape
		expected_shape_U = np.zeros((len(self.full_indices), len(self.full_indices))).shape
		assert U.shape == expected_shape_U, f"U shape is {U.shape}, expected {expected_shape_U}"
			# Print shape of U for checking
		# print("U shape: ", U.shape)	
			# Pretty print U for checking
		# print("Unitary rotation matrix U: \n", U)


		# Check that U is orthogonal
			# Note, sometimes due to numerical errors (e-11) U@U.T is not exactly identity, but very close
				# If the difference is too large, raise an error
		diff = np.linalg.norm(U@U.T - np.eye(len(U)))
		assert diff < 1e-6, f"U is not orthogonal, ||U@U.T - I|| = {diff}"




		# Step (viii): Rotate the orbitals



		# rotate orbitals, 
			# convert to spin orbital basis
		mo_coeffs_spin = spatial2spin([mo_coeffs[0], mo_coeffs[1]],orbspin=self.orbspin)
			
			# apply rotation
				# mo_coffs_spin shape: (n_AO, num_spin_orbitals)
				# U shape: (num_spin_orbitals, num_spin_orbitals)
		mo_coeffs_spin_rot = mo_coeffs_spin @ U
			
				# Print shape for checking
		# print("mo_coeffs_spin shape: ", mo_coeffs_spin.shape)
		
			# convert back to spatial orbital basis
		mo_coeffs_rot = spin2spatial(mo_coeffs_spin_rot, orbspin=self.orbspin)

				# check shape
		for spin in [0, 1]:
			# print()
			# print(f"Spin {spin} mo_coeffs_rot shape: ", mo_coeffs_rot[spin].shape)

			expected_shape_spatial = mo_coeffs[spin].shape
			assert mo_coeffs_rot[spin].shape == expected_shape_spatial, f"mo_coeffs_rot shape is {mo_coeffs_rot[spin].shape}, expected {expected_shape_spatial}"

		# Pretty print rotated MO coefficients for checking
			diff_matrix = mo_coeffs_rot[spin] - mo_coeffs[spin]

			# print()
			# print(f"   Difference in MO coefficients for spin {spin} after rotation: \n", diff_matrix)

		# Check that rotated orbitals are orthonormal
			# Sum a coloumn of mo_coeffs to check normalization for a given spin
				# Check: C^T S C = I
		
			C_i = mo_coeffs_rot[spin]
			
			norm = C_i.T @ self.S @ C_i
			
			# print()
			# print(f"   Normalization check for spin {spin} after rotation: \n", norm)
			# print()

			assert np.allclose(norm, np.eye(norm.shape[0]), atol=1e-6), f"MO coefficients for spin {spin} are not orthonormal!"


		# Print shape and matrix of matrices and vectors for checking
		print("")
		print("Density matrix, shape: ", D_AB_cache.shape, "\n", D_AB_cache)
		print("")
		print("Gradient G, shape: ", G.shape, "\n", G)
		print("")
		print("Hessian H, shape: ", H.shape, "\n", H)
		print("")
		print("Orbital rotation parameter matrix R, shape: ", R_matrix.shape, "\n", R_matrix)
		print("")
		print("Unitary rotation matrix U, shape: ", U.shape, "\n", U)
		print("")
		print("Rotated MO coefficients, shape: ", mo_coeffs_rot[spin].shape, "\n", mo_coeffs_rot[spin])
		print("")



		return mo_coeffs_rot


	





	def run_ovos(self,  mo_coeffs):
		"""
		Run the OVOS algorithm.
		"""

		converged = False
		max_iter = 1000
		iter = 0

		while not converged and iter < max_iter:
			iter += 1
			print("#### OVOS Iteration ", iter, " ####")
			
			E_corr, MP1_amplitudes, eri_spin, Fmo_spin = self.MP2_energy(mo_coeffs = mo_coeffs)
			print("MP2 correlation energy: ", E_corr)
			print()

			# Step (ix): check convergence
			# convergence criterion: change in correlation energy < 1e-6 Hartree
			if iter > 1:
				threshold = 1e-8 if len(self.active_inocc_indices) < 4 else 1e-6
				if np.abs(E_corr - lst_E_corr[-1]) < threshold:
					converged = True
					print("OVOS converged in ", iter, " iterations.")
				else:
					lst_E_corr.append(E_corr)
			else:
				lst_E_corr = []
				lst_E_corr.append(E_corr)

			# If MP2 goes positive, stop the optimization
			#if E_corr > 0:
				#print("Warning: MP2 correlation energy is positive. Stopping OVOS optimization.")
				#break

			mo_coeffs = self.orbital_optimization(mo_coeffs, MP1_amplitudes=MP1_amplitudes, eri_spin=eri_spin, Fmo_spin=Fmo_spin)

		# Print information about the spaces
		print()
		print("#### Active and inactive spaces ####")
		print("Total number of spin-orbitals: ", self.tot_num_spin_orbs)
		print("Active occupied spin-orbitals: ", self.active_occ_indices)
		print("Active unoccupied spin-orbitals: ", self.active_inocc_indices)
		print("Inactive unoccupied spin-orbitals: ", self.inactive_indices)
		print()

		
		# Return list of correlation energies
		# print("List of MP2 correlation energies during OVOS optimization, w. length ", len(lst_E_corr) ,": ", lst_E_corr)

		print("PySCF MP2")
		MP2 = self.uhf.MP2().run()
		print("MP2 correlation energy: ", MP2.e_corr)

		# Check if OVOS converged
		if not converged:
			assert False, "OVOS did not converge within the maximum number of iterations."

		return lst_E_corr
	
	
	
		
# CTRL + ' (thingy under ENTER) to comment/uncomment multiple lines


# Make a presentation for how the matrices develop during OVOS... !!!
	# ...

# Make R bigger to account for full space!



"""
Objective:
- Get LiH STO-3G molecule working with OVOS at different numbers of optimized virtual orbitals
	* Point might be to instead solve the Newton-Raphson equation for the orbital rotation parameters R
		with a different method than direct inversion of the Hessian, e.g. conjugate gradient
		The other proposed method in the paper is:
			"A block generalization of the reduced linear equation 
			technique of Purvis and Bartlett is used. 
			It is recognized that the Hessian matrix is dominated 
			by square blocks located diagonally. A particular block is 
			related to the rotation of all possible active orbitals with one 
			nonactive orbital (Hea,eb)' Only inverses of all diagonally 
			located blocks are kept in the core."
		It is called the "reduced linear equation (RLE) method" in the following paper:
			The reduced linear equation method in coupled cluster theory,
				by George D. Purvis III and Rodney J. Bartlett,
				at Battelle Columbus Laboratories. Columbus, Ohio 43201 
				(Received 24 September 1980; accepted 14 April 1981) 

				
		Get correct integral notation for eri_spin
			eri_spin[A,I,B,J] should represent <AI|BJ> in physicist's notation
			PySCF chemist's notation: (ij|kl) = physicist's <ik|jl>
			So eri_spin[A,I,B,J] stored as (AI|BJ)_chem = <AB|IJ>_phys
	
	PySCF's spatial2spin produces ERIs with [occupied, occupied, virtual, virtual] index ordering,
		not [virtual, occupied, virtual, occupied]

	Therefore, we need to permute the indices to get the correct ordering
	eri_spin = eri_spin.transpose(0,2,1,3)  # Now eri_spin[A,I,B,J] = <AI|BJ> in physicist's notation
				
	
	Make a function w. scipy opimize to minimize the Hylleraas functional
		Should optimize Rotation matrix R directly, R_ea
			Each spin orbital a in active_inocc_indices should have a rotation parameter R_ea for each inactive orbital e in inactive_indices
				phi -> phi' = exp(R) phi_b + exp(R) phi_a
	def ...() -> ...:

"""








# Molecule
atom_choose_between = [
	"H .0 .0 .0; H .0 .0 0.74144",  # H2 bond length 0.74144 Angstrom
	"Li .0 .0 .0; H .0 .0 1.595",   # LiH bond length 1.595 Angstrom
	"O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;",  # H2O equilibrium geometry
	"C  0.0000  0.0000  0.0000; H  0.0000  0.9350  0.5230; H  0.0000 -0.9350  0.5230;" # CH2 
]
# Basis set
basis_choose_between = [
	"STO-3G",
	"6-31G",
]

# Unit
unit="angstrom"


# Select molecule and basis set
select_atom = 1  # Select molecule index here
select_basis = 1 # Select molecule index here

atom, basis = (atom_choose_between[select_atom], basis_choose_between[select_basis])

# Get number of electrons and full space size in molecular orbitals
mol = pyscf.M(atom=atom, basis=basis, unit=unit)
	# Number of electrons
num_electrons = mol.nelec[0] + mol.nelec[1]
	# Full space size in molecular orbitals
full_space_size = int(pyscf.scf.UHF(mol).run().mo_coeff.shape[1])




"""
Simple run for Phillip!
"""
one_run = True
if one_run == True:
	num_opt_virtual_orbs = 2  # Set number of optimized virtual orbitals here

	print("")
	print("#### OVOS with ", num_opt_virtual_orbs, " optimized virtual orbitals ####")
	print("")

	uhf = pyscf.scf.UHF(mol).run()
	mo_coeff = uhf.mo_coeff 


	try:
		# Get list of MP2 correlation energies during OVOS optimization
		lst_E_corr = OVOS(mol=mol, num_opt_virtual_orbs=num_opt_virtual_orbs).run_ovos(mo_coeff)

	except AssertionError as e:
		print("")
		print(f"Error during OVOS with {num_opt_virtual_orbs} optimized virtual orbitals: {e}")


	# Print the final MP2 correlation energy after all OVOS and amount of iterations till convergence
	print("")
	print("MP2 correlation energy, for ", num_opt_virtual_orbs, " optimized virtual orbitals:", '%.2E' % Decimal(lst_E_corr[-1]), " @ ", len(lst_E_corr), " iterations till convergence")
	print("")

	# Print
	print("Number of electrons: ", num_electrons)
	print("Full space size in molecular orbitals: ", full_space_size)
	print("")



"""
Run OVOS for different numbers of optimized virtual orbitals
"""
run_different_virt_orbs = False
if run_different_virt_orbs == True:
	# Loop over different numbers of optimized virtual orbitals
	# List of MP2 correlation energies for different numbers of optimized virtual orbitals
	lst_E_corr_virt_orbs = [[],[]]  # [[E_corr_list], [num_opt_virtual_orbs_list]]
	lst_MP2_virt_orbs = []  # [(num_opt_virtual_orbs, E_corr, iterations_till_convergence), ...]

	# Error messages for failed runs
	lst_error_messages = []

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
			# Re-initialize molecule and UHF for each run
			mol = pyscf.M(atom=atom, basis=basis, unit=unit)
				
			uhf = pyscf.scf.UHF(mol).run()
			mo_coeff = uhf.mo_coeff 

			lst_E_corr = OVOS(mol=mol, num_opt_virtual_orbs=num_opt_virtual_orbs_current).run_ovos(mo_coeff)

			# run_OVOS got stuck in a non-converging loop
			if len(lst_E_corr) >= 1000:
				print("OVOS with ", num_opt_virtual_orbs_current, " optimized virtual orbitals did not converge. Rerunning with the same number of virtual orbitals.")
				num_opt_virtual_orbs_current -= increment  # Decrement to retry the same number
				continue

			# run_OVOS converged to a positive MP2 correlation energy
			if lst_E_corr[-1] > 0:
				print("Warning: OVOS with ", num_opt_virtual_orbs_current, " optimized virtual orbitals converged to a positive MP2 correlation energy. Rerunning with the same number of virtual orbitals.")
				num_opt_virtual_orbs_current -= increment  # Decrement to retry the same number
				continue

			# Store results
			lst_MP2_virt_orbs.append((num_opt_virtual_orbs_current, lst_E_corr[-1], len(lst_E_corr)))
			lst_E_corr_virt_orbs[0].append(lst_E_corr)
			lst_E_corr_virt_orbs[1].append(num_opt_virtual_orbs_current)

			# Reset retry count on success
			retry_count = 0

		except AssertionError as e:
			print(f"Error during OVOS with {num_opt_virtual_orbs_current} optimized virtual orbitals: {e}")
			print("Rerunning with the same number of virtual orbitals.")

			# Add error message to list

				# Get results if available
			if lst_E_corr is None:
				iter_ = 1
			else:
				iter_ = len(lst_E_corr)

			lst_error_messages.append((num_opt_virtual_orbs_current, str(e), iter_))

			retry_count += 1
			if retry_count >= max_retries:
				print(f"Maximum retries reached for {num_opt_virtual_orbs_current} optimized virtual orbitals. Skipping to next.")
				retry_count = 0
				continue

			num_opt_virtual_orbs_current -= increment  # Decrement to retry the same number
			continue


	# Print the final MP2 correlation energy after all OVOS and amount of iterations till convergence
	print("")
	for num_opt_virtual_orbs_current, E_corr, iter_ in lst_MP2_virt_orbs:
		print("MP2 correlation energy, for ", num_opt_virtual_orbs_current, f" optimized virtual orbitals: ", '%.2E' % Decimal(E_corr), "  @ ", iter_, " iterations till convergence")
	print("")

	# Print summary of the run
	print("Number of electrons: ", num_electrons)
	print("Full space size in molecular orbitals: ", full_space_size)
	print("Maximum number of optimized virtual orbitals tested: ", max_opt_virtual_orbs)
	print("Total OVOS runs completed: ", len(lst_MP2_virt_orbs))
	print("")

	# Print error messages summary
	if len(lst_error_messages) > 0:
		print("#### Error messages summary ####")
		for num_opt_virtual_orbs_current, error_msg, iter_ in lst_error_messages:
			print("  OVOS w. ", num_opt_virtual_orbs_current, " optimized vorbs failed at iteration ", iter_ ," w. error: ", error_msg)
		print("")


	# Save data to JSON files
	import json

	str_name = "different_virt_orbs"

	if select_atom == 0:
		str_atom = "H2"
	elif select_atom == 1:
		str_atom = "LiH"
	elif select_atom == 2:
		str_atom = "H2O"
	elif select_atom == 3:
		str_atom = "CH2"

	if select_basis == 0:
		str_basis = "STO-3G"
	elif select_basis == 1:
		str_basis = "6-31G"

	# Save MP2 correlation energy convergence data
	with open("branch/data/"+str_atom+"/"+str_basis+"/lst_MP2_"+str_name+".json", "w") as f:
		json.dump(lst_E_corr_virt_orbs, f, indent=2)

	print("Data saved to branch/data/"+str_atom+"/"+str_basis+"/...")






# """
# Time profiling 
# """
# time_profile = False
# if time_profile == True and run_single_ovos == True:
# 	import cProfile
# 	import pstats

# 	opt = "opt_C" # Lable for optimization settings, see Optimization_options.md

# 	cProfile.run('OVOS(mol=mol, num_opt_virtual_orbs=6).run_ovos(mo_coeff)', 'branch/profil/profiling_results_'+opt+'.prof')

# 	# Print the profiling results
# 	with open('branch/profil/profiling_results_'+opt+'.txt', 'w') as f:
# 		stats = pstats.Stats('branch/profil/profiling_results_'+opt+'.prof', stream=f)
# 		stats.sort_stats('cumulative')  # Sort by cumulative time
# 		stats.print_stats()














# """
# Run OVOS algorithm for N cycles and store MP2 correlation energy convergence data.
# Save data to JSON files
# """
# run_cycles = False
# if run_cycles == True:
# 	import json
	
# 	# Set maximum number of optimized virtual orbitals to test
# 		# Denoted in molecular orbitals (not spin orbitals)
# 	max_opt_virtual_orbs = full_space_size*2 - num_electrons
# 		# Set statrting number of optimized virtual orbitals and increment
# 	num_opt_virtual_orbs_current = 0  # Start with number of occupied orbitals
# 	increment = 1 # Increment by 1 optimized virtual orbital

# 	# Retry bounds for vorb
# 	max_retries_vorb = 3
# 	retry_count_vorb = 0

# 	for num_opt_virtual_orbs in range(1, max_opt_virtual_orbs+1, increment):
		
# 		print("")
# 		print("#### OVOS with ", num_opt_virtual_orbs, " out of ", max_opt_virtual_orbs," optimized virtual orbitals ####")
# 		print("")

# 		# List of MP2 correlation energy convergence data for each cycle
# 		lst_E_corr_cycle = []
# 		iter_conv_cycle = []

# 		# Number of OVOS cycles
# 		cycle_max = 25 # N = 100
# 		cycle_max_run = cycle_max # To keep track of actual number of runs including restarts
# 		max_cycle_max = 100 # To avoid infinite loops
# 		cycle = 0

# 		# Retry bounds
# 		max_retries = 5
# 		retry_count = 0

# 		while cycle < cycle_max:
# 			print("")
# 			print("#### OVOS Cycle ", cycle+1, " out of", cycle_max_run," ####")

# 			try:
# 				# Re-initialize molecule and UHF for each cycle
# 				mol = pyscf.M(atom=atom, basis=basis, unit=unit)
# 				uhf = pyscf.scf.UHF(mol).run()
# 				mo_coeff = uhf.mo_coeff

# 				run_OVOS = OVOS(mol=mol, num_opt_virtual_orbs=num_opt_virtual_orbs)

# 				lst_E_corr = run_OVOS.run_ovos(mo_coeff)

# 				# run_OVOS got stuck in a non-converging loop
# 				if len(lst_E_corr) >= 600:
# 					print("OVOS cycle did not converge. Restarting cycle.")
# 					cycle_max_run += 1
# 					continue

# 				# If the last cycle's correlation energy converges to a positive value, skip storing the data
# 				if lst_E_corr[-1] > 0:
# 					print("Warning: OVOS converged to a positive MP2 correlation energy. Skipping data storage for this cycle.")
# 					print("")

# 					# Do a new cycle still keeping the max number of cycles the same
# 					cycle_max_run += 1
# 				else:
# 					lst_E_corr_cycle.append(lst_E_corr)

# 					# Reset retry count on success
# 					retry_count = 0
# 					retry_count_vorb = 0

# 					# Increment cycle count only on successful completion
# 					cycle += 1
				
# 			except AssertionError as e:
# 				print(f"Error during OVOS cycle: {e}")
# 				print("Restarting cycle.")
# 				cycle_max_run += 1

# 				retry_count += 1
# 				if retry_count >= max_retries:
# 					print(f"Maximum retries reached for this cycle. Skipping to next.")
# 					retry_count = 0
# 					continue

# 				retry_count_vorb += 1
# 				if retry_count_vorb >= max_retries_vorb:
# 					print(f"Maximum retries reached for number of optimized virtual orbitals. Moving to next number of virtual orbitals.")
# 					retry_count_vorb = 0
# 					break

# 				if cycle_max_run >= max_cycle_max:
# 					print("Reached maximum allowed cycles. Exiting.")
# 					break

# 				continue

# 		str_name = "cycle_"+str(cycle_max)
# 		str_name += "_vorb_"+str(num_opt_virtual_orbs)
		
# 		if select_atom == 0:
# 			str_atom = "H2"
# 		elif select_atom == 1:
# 			str_atom = "LiH"
# 		elif select_atom == 2:
# 			str_atom = "H2O"

# 		if select_basis == 0:
# 			str_basis = "STO-3G"
# 		elif select_basis == 1:
# 			str_basis = "6-31G"

# 		# If lst_E_corr_cycle is empty, skip saving
# 		if len(lst_E_corr_cycle) == 0:
# 			print("No successful OVOS cycles completed for ", num_opt_virtual_orbs, " optimized virtual orbitals. Skipping data saving.")
# 			continue

# 		# Save MP2 correlation energy convergence data
# 		with open("branch/data/"+str_atom+"/"+str_basis+"/lst_E_corr_"+str_name+".json", "w") as f:
# 			json.dump(lst_E_corr_cycle, f, indent=2)