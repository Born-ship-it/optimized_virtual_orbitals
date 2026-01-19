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

			# Sum a coloumn of mo_coeffs to check normalization for a given spin
				# Chekc: C^T S C = I
		for spin in [0, 1]:
			C_i = self.mo_coeffs[spin]
			
			norm = C_i.T @ self.S @ C_i
			assert np.allclose(norm, np.eye(norm.shape[0]), atol=1e-6), f"MO coefficients for spin {spin} are not orthonormal!"

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



		# i) Fock matrix in spin-orbital basis & Two-electron integrals in spin-orbital basis

			#PySCF stores 2e integrals in chemists' notation: (ij|kl) = <ik|jl> in physicists' notation.
				# (alpha alpha | alpha alpha) integrals
		eri_aaaa = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[0], mo_coeffs[0], mo_coeffs[0], mo_coeffs[0]], compact=False)
				# (beta beta | beta beta) integrals
		eri_bbbb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[1], mo_coeffs[1], mo_coeffs[1], mo_coeffs[1]], compact=False)
				# (alpha alpha | beta beta) integrals
		eri_aabb = pyscf.ao2mo.kernel(self.eri_4fold_ao, [mo_coeffs[0], mo_coeffs[0], mo_coeffs[1], mo_coeffs[1]], compact=False)

			# reshape AO->MO (chemists' notation)
		norb_alpha = mo_coeffs[0].shape[1]
		norb_beta  = mo_coeffs[1].shape[1]
		eri_aaaa = eri_aaaa.reshape((norb_alpha, norb_alpha, norb_alpha, norb_alpha))
		eri_bbbb = eri_bbbb.reshape((norb_beta,  norb_beta,  norb_beta,  norb_beta))
		eri_aabb = eri_aabb.reshape((norb_alpha, norb_alpha, norb_beta,  norb_beta))


			# Manual assembly of spin-orbital integrals from spatial blocks
		n_spatial = mo_coeffs[0].shape[1]
		n_spin = 2 * n_spatial

				# mapping: spin-orb index -> spatial MO index (0,0,1,1,...)
		orb_map = np.array([i // 2 for i in range(n_spin)], dtype=int)
				# spin labels (0=alpha, 1=beta) for tests / indexing
		self.orbspin = np.array([i % 2 for i in range(n_spin)], dtype=int)

			# ensure spatial blocks are contiguous float64
		eri_aaaa = np.ascontiguousarray(eri_aaaa, dtype=np.float64)
		eri_aabb = np.ascontiguousarray(eri_aabb, dtype=np.float64)
		eri_bbbb = np.ascontiguousarray(eri_bbbb, dtype=np.float64)
		Fmo_a = np.ascontiguousarray(Fmo[0], dtype=np.float64)
		Fmo_b = np.ascontiguousarray(Fmo[1], dtype=np.float64)

			# allocate spin-orbital ERI and Fock
		eri_spin = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.float64)
		Fmo_spin = np.zeros((n_spin, n_spin), dtype=np.float64)

			# assemble ERI: route by spin pattern (p,q,r,s) -> select correct spatial block
		for p in range(n_spin):
			pa, sp_p = orb_map[p], self.orbspin[p]
			for q in range(n_spin):
				pb, sp_q = orb_map[q], self.orbspin[q]
				for r in range(n_spin):
					ra, sp_r = orb_map[r], self.orbspin[r]
					for s in range(n_spin):
						rb, sp_s = orb_map[s], self.orbspin[s]
						if (sp_p, sp_q, sp_r, sp_s) == (0,0,0,0):
							eri_spin[p,q,r,s] = eri_aaaa[pa,pb,ra,rb]
						elif (sp_p, sp_q, sp_r, sp_s) == (1,1,1,1):
							eri_spin[p,q,r,s] = eri_bbbb[pa,pb,ra,rb]
						elif (sp_p, sp_q, sp_r, sp_s) == (0,0,1,1):
							eri_spin[p,q,r,s] = eri_aabb[pa,pb,ra,rb]
						elif (sp_p, sp_q, sp_r, sp_s) == (1,1,0,0):
							# eri_aabb is (alpha,alpha,beta,beta) -> swap order
							eri_spin[p,q,r,s] = eri_aabb[ra,rb,pa,pb]

			# assemble spin-orbital Fock
		for p in range(n_spin):
			pa, sp_p = orb_map[p], self.orbspin[p]
			for q in range(n_spin):
				pb, sp_q = orb_map[q], self.orbspin[q]
				if sp_p == sp_q == 0:
					Fmo_spin[p,q] = Fmo_a[pa,pb]
				elif sp_p == sp_q == 1:
					Fmo_spin[p,q] = Fmo_b[pa,pb]
				else:
					Fmo_spin[p,q] = 0.0

			# antisymmetrized integrals for same-spin usage: <pq||rs> = (pq|rs) - (pq|sr)
		eri_as = eri_spin - eri_spin.transpose(0,1,3,2)

			# quick sanity checks
				# Check that manual assembly yielded non-zero integrals
		assert np.count_nonzero(eri_spin) > 0, "Manual ERI assembly yielded all zeros!"
				# Check that infinite or NaN values are not present
		assert np.all(np.isfinite(eri_spin)), "Manual ERI contains non-finite values!"
				# Check symmetry properties of the integrals
		assert np.allclose(eri_spin, eri_spin.transpose(1,0,2,3), atol=1e-10), "Manual ERI fails p<->q symmetry!"
		assert np.allclose(eri_spin, eri_spin.transpose(0,1,3,2), atol=1e-10), "Manual ERI fails r<->s symmetry!"
		assert np.allclose(eri_spin, eri_spin.transpose(2,3,0,1), atol=1e-10), "Manual ERI fails (pq)<->(rs) symmetry!"	
				# Check antisymmetrized integrals for same-spin
		assert np.allclose(eri_as[:,:,:, :], -eri_as[:,:,:, :].transpose(0,1,3,2), atol=1e-10), "Antisymmetrized integrals fail r<->s antisymmetry!"

				# Check that Fock matrix is Hermitian
		assert np.allclose(Fmo_spin, Fmo_spin.T, atol=1e-10), "Fock matrix is not Hermitian!"
				# Check non-zero Fock matrix
		assert np.count_nonzero(Fmo_spin) > 0, "Fock matrix is all zeros!"
				# Check that infinite or NaN values are not present
		assert np.all(np.isfinite(Fmo_spin)), "Fock matrix contains non-finite values!"




		# ii) Compute MP1 amplitudes (spin-orbital)
		MP1_amplitudes = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.float64)

		occ_indices = list(range(self.nelec))
		virt_indices = list(range(self.nelec, n_spin))
		eps = np.array(orbital_energies, dtype=np.float64)

		for a in virt_indices:
			for b in virt_indices:
				for i in occ_indices:
					for j in occ_indices:
						same_spin_IJ = (self.orbspin[i] == self.orbspin[j])
						same_spin_AB = (self.orbspin[a] == self.orbspin[b])
						same_spin_AI = (self.orbspin[a] == self.orbspin[i])
						same_spin_BJ = (self.orbspin[b] == self.orbspin[j])

						# Only construct amplitudes for spin-consistent index patterns
						if (same_spin_IJ and same_spin_AB and same_spin_AI and same_spin_BJ) \
						or (not same_spin_IJ and not same_spin_AB and same_spin_AI and same_spin_BJ):

							denom = eps[a] + eps[b] - eps[i] - eps[j]
							if abs(denom) < 1e-12:
								t_val = 0.0
							else:
								# Same-spin pairs use antisymmetrized integrals, opposite-spin use Coulomb only
								if same_spin_IJ and same_spin_AB:
									integral = eri_as[a, b, i, j]        # eri_as = eri_spin - eri_spin.transpose(0,1,3,2)
								else:
									integral = eri_spin[a, b, i, j]
								t_val = - integral / denom

							MP1_amplitudes[a, b, i, j] = t_val
						else:
							MP1_amplitudes[a, b, i, j] = 0.0

		# Sanity checks
			# Check that MP1 amplitudes are not all zeros
		assert np.count_nonzero(MP1_amplitudes) > 0, "MP1 amplitudes are all zeros!"
			# Check that infinite or NaN values are not present
		assert np.all(np.isfinite(MP1_amplitudes)), "MP1 amplitudes contain non-finite values!"
			# Check symmetry properties of MP1 amplitudes
		assert np.allclose(MP1_amplitudes, -MP1_amplitudes.transpose(1,0,2,3), atol=1e-10), "MP1 amplitudes fail A<->B antisymmetry!"
		




		# iii) Compute MP2 correlation energy (spin-orbital indices)
		# J_2 = sum_{i>j} J_ij^(2)
		eps = np.array(orbital_energies, dtype=np.float64)
		F = Fmo_spin
		occ = list(range(self.nelec))
		virt = list(range(self.nelec, n_spin))

		J_2 = 0.0

		for i_idx, i in enumerate(occ):
			for j_idx, j in enumerate(occ):
				if i <= j:
					continue

				J_ij = 0.0

				# Term A: double sum over a>b and c>d: t_abij * t_cdij * [ ... F/eps/delta bracket ... ]
				termA = 0.0
				for a in virt:
					for b in virt:
						if a <= b:
							continue
						t_ab = MP1_amplitudes[a, b, i, j]
						if abs(t_ab) > 0.0:
							continue

						for c in virt:
							for d in virt:
								if c <= d:
									continue
								t_cd = MP1_amplitudes[c, d, i, j]
								if abs(t_cd) > 0.0:
									continue

								# Kronecker deltas
								delta_bd = 1.0 if b == d else 0.0
								delta_bc = 1.0 if b == c else 0.0
								delta_ac = 1.0 if a == c else 0.0
								delta_ad = 1.0 if a == d else 0.0

								# Fock-related term: (f_ac δ_bd - f_ad δ_bc) + (f_bd δ_ac - f_bc δ_ad)
								term_F_delta = (F[a, c] * delta_bd - F[a, d] * delta_bc) + (F[b, d] * delta_ac - F[b, c] * delta_ad)

								# Energy-related term: - (eps_i + eps_j) * (δ_ac δ_bd - δ_ad δ_bc)
								term_eps_delta = - (eps[i] + eps[j]) * (delta_ac * delta_bd - delta_ad * delta_bc)

								bracket = term_F_delta + term_eps_delta

								termA += t_ab * t_cd * bracket

				# Term B: 2 * sum_{a>b} t_abij * <ab|ij>
				termB = 0.0
				for a in virt:
					for b in virt:
						if a <= b:
							continue
						t_ab = MP1_amplitudes[a, b, i, j]
						if abs(t_ab) > 0.0:
							continue

						# <ab|ij> in our assembled integrals is eri_spin[a,b,i,j] (chemists' ordering)
						termB += 2.0 * t_ab * eri_spin[a, b, i, j]

				J_ij = termA + termB
				J_2 += J_ij

		print("Computed MP2 correlation energy (spin-orbital): ", J_2)
		
		# Sanity checks
			# Check that MP2 correlation energy is finite
		assert np.isfinite(J_2), "MP2 correlation energy is not finite!"
			# Check that MP2 correlation energy is not positive
		assert J_2 <= 0.0, "MP2 correlation energy is not negative!"



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
		# Setup values
		num_active_inocc = len(self.active_inocc_indices)
		num_active_occ = len(self.active_occ_indices)
		num_inactive = len(self.inactive_indices)



		# Step (v): Compute the gradient and Hessian of the second-order Hylleraas functional

			# Compute gradient G
				# 12a equation, in spatial orbitals: 
					# G[AE, ] = Term 1 + Term 2
						# Term 1: + 2.0 \sum_{I>J} \sum_{B} t_ABIJ <EB|IJ>
						# Term 2: + 2.0 \sum_{B} D_AB * f_EB
			# where D_AB = ∑_{I>J,C} t_ACIJ * t_BCIJ (second-order density)

		G = np.zeros((num_active_inocc * num_inactive, ))

		for idx_A, A in enumerate(self.active_inocc_indices):
			for idx_E, E in enumerate(self.inactive_indices):
				flat_index_G = idx_A * num_inactive + idx_E

				# Term 1: 2∑_{I>J,B} t_ABIJ <EB|IJ>
				term1 = 0.0
				for I in self.active_occ_indices:
					for J in self.active_occ_indices:
						if I > J:  # Only compute for I>J
							same_spin_IJ = (self.orbspin[I] == self.orbspin[J])
							
							for B in self.active_inocc_indices:
								same_spin_AB = (self.orbspin[A] == self.orbspin[B])
								same_spin_EB = (self.orbspin[E] == self.orbspin[B])

								same_spin_AI = (self.orbspin[A] == self.orbspin[I])
								same_spin_BJ = (self.orbspin[B] == self.orbspin[J])

								same_spin_EI = (self.orbspin[E] == self.orbspin[I])
								same_spin_EJ = (self.orbspin[E] == self.orbspin[J])

								# Check spin matching for t_ABIJ
								if (same_spin_IJ and same_spin_AB and same_spin_AI and same_spin_BJ) or (not same_spin_IJ and not same_spin_AB and same_spin_AI and same_spin_BJ):
									t_ABIJ = MP1_amplitudes[A, B, I, J]

									# Check spin matching for <EB|IJ>
									if (same_spin_IJ and same_spin_EB and same_spin_EI and same_spin_EJ) or (not same_spin_IJ and not same_spin_EB and same_spin_EI and same_spin_EJ):
										if same_spin_IJ and same_spin_EB:
											# Same-spin: antisymmetrized
											integral = eri_spin[E, B, I, J] - eri_spin[E, B, J, I]
										else:
											# Opposite-spin: Coulomb only
											integral = eri_spin[E, B, I, J]
										
										term1 += 2.0 * t_ABIJ * integral

				# Term 2: 2∑_B D_AB F_EB
				term2 = 0.0
				for B in self.active_inocc_indices:
					# Compute D_AB = ∑_{I>J,C} t_AIC * t_BJC
					D_AB = 0.0
					for I in self.active_occ_indices:
						for J in self.active_occ_indices:
							if I > J:
								same_spin_IJ = (self.orbspin[I] == self.orbspin[J])
								
								for C in self.active_inocc_indices:
									same_spin_AC = (self.orbspin[A] == self.orbspin[C])
									same_spin_BC = (self.orbspin[B] == self.orbspin[C])

									same_spin_AI = (self.orbspin[A] == self.orbspin[I])
									same_spin_CJ = (self.orbspin[C] == self.orbspin[J])
									
									# Get t_AIC J (keeping J fixed)
									if (same_spin_IJ and same_spin_AC and same_spin_AI and same_spin_CJ) or (not same_spin_IJ and not same_spin_AC and same_spin_AI and same_spin_CJ):
										t_AICJ = MP1_amplitudes[A, C, I, J] if C > A or (C == A and same_spin_AC) else -MP1_amplitudes[C, A, I, J] if same_spin_IJ and same_spin_AC else MP1_amplitudes[C, A, I, J]
									else:
										t_AICJ = 0.0
									
									same_spin_BI = (self.orbspin[B] == self.orbspin[I])
									same_spin_CJ = (self.orbspin[C] == self.orbspin[J])

									# Get t_BJC I (keeping I fixed, swapping J and C roles)
									if (same_spin_IJ and same_spin_BC and same_spin_BI and same_spin_CJ) or (not same_spin_IJ and not same_spin_BC and same_spin_BI and same_spin_CJ):
										t_BJCI = MP1_amplitudes[B, C, I, J] if C > B or (C == B and same_spin_BC) else -MP1_amplitudes[C, B, I, J] if same_spin_IJ and same_spin_BC else MP1_amplitudes[C, B, I, J]
									else:
										t_BJCI = 0.0
									
									D_AB += t_AICJ * t_BJCI
					
					term2 += 2.0 * D_AB * Fmo_spin[E, B]

				G[flat_index_G] = term1 + term2

			# Check G is not all zeros
		count_nonzero_G = np.count_nonzero(G)
		assert count_nonzero_G > 0, "Gradient G is all zeros!"
			# Check G for NaN or Inf values
		assert np.all(np.isfinite(G)), "Gradient G contains non-finite values!"

			# Continue by looking into G
		# print("\n #### GRADIENT G STATISTICS ####")
		# print("Statistics of gradient G values:")
		# print("  Shape: ", G.shape)
		# print("  Min: ", np.min(G))
		# print("  Max: ", np.max(G))
		# print("  Mean: ", np.mean(G))
		# print("  Std: ", np.std(G))
		# print("  Number of non-zero elements: ", count_nonzero_G)
		# print("="*70)
		
			# Compute Hessian H
				#Equation 12b, in spatial orbitals: H[EA, FB] = Term 1 + Term 2 + Term 3 + Term 4 + Term 5 + Term 6 + Term 7
					# Term 1: + 2.0 * \sum_{I>J} t_ABIJ <EF|IJ>
					# Term 2: - 1.0 * \sum_{I>J} \sum_C t_ACIJ <BC|IJ> \delta_EF
					# Term 3: - 1.0 * \sum_{I>J} \sum_C t_CBIJ <CA|IJ> \delta_EF
					# Term 4: + D_AB F_AA \delta_EF 
					# Term 5: - D_AB F_BB \delta_EF 
					# Term 6: - D_AB F_EF \delta_EF
					# Term 7: + D_AB F_EF
				# Where D_AB = ∑_{I>J,C} t_ACIJ * t_BCIJ (second-order density)

		H = np.zeros((num_active_inocc * num_inactive, num_active_inocc * num_inactive))
		for idx_A, A in enumerate(self.active_inocc_indices):
			for idx_E, E in enumerate(self.inactive_indices):
				flat_index_H_row = idx_A * num_inactive + idx_E

				for idx_B, B in enumerate(self.active_inocc_indices):
					for idx_F, F in enumerate(self.inactive_indices):
						flat_index_H_col = idx_B * num_inactive + idx_F
                    
						# Term 1: 2∑_{I>J} t_ABIJ <EF|IJ>
						term1 = 0.0
						for I in self.active_occ_indices:
							for J in self.active_occ_indices:
								if I > J:
									same_spin_IJ = (self.orbspin[I] == self.orbspin[J])
									same_spin_AB = (self.orbspin[A] == self.orbspin[B])
									same_spin_EF = (self.orbspin[E] == self.orbspin[F])

									same_spin_AI = (self.orbspin[A] == self.orbspin[I])
									same_spin_BJ = (self.orbspin[B] == self.orbspin[J])

									# Get t_ABIJ with proper sign handling
									if (same_spin_IJ and same_spin_AB and same_spin_AI and same_spin_BJ) or (not same_spin_IJ and not same_spin_AB and same_spin_AI and same_spin_BJ):
										if A > B:
											t_ABIJ = MP1_amplitudes[A, B, I, J] #4,5,3,2
										elif B > A:
											# Need to extract with swapped indices
											if same_spin_AB:  # same-spin: antisymmetric
												t_ABIJ = MP1_amplitudes[B, A, J, I] #5,4,3,2
											else:  # opposite-spin: symmetric
												t_ABIJ = MP1_amplitudes[B, A, J, I]
										else:  # A == B
											t_ABIJ = 0.0
									else:
										t_ABIJ = 0.0
										
									same_spin_EI = (self.orbspin[E] == self.orbspin[I])
									same_spin_FJ = (self.orbspin[F] == self.orbspin[J])

									# Check spin matching for <EF|IJ>
									if t_ABIJ != 0.0 and ((same_spin_IJ and same_spin_EF and same_spin_EI and same_spin_FJ) or (not same_spin_IJ and not same_spin_EF and same_spin_EI and same_spin_FJ)):
										if same_spin_IJ and same_spin_EF:
											# Same-spin: antisymmetrized
											integral = eri_spin[E, F, I, J] - eri_spin[E, F, J, I]
										else:
											# Opposite-spin: Coulomb only
											integral = eri_spin[E, F, I, J]
										
										term1 += 2.0 * t_ABIJ * integral

									# Check specific case, seen being asymmetric when it should be symmetric
									if flat_index_H_col == 7 and flat_index_H_row == 0:
										print()
										print(f"  For Hessian, H[{flat_index_H_row},{flat_index_H_col}], ({idx_A * num_inactive} + {idx_E}, {idx_B * num_inactive} + {idx_F})")
										print(f"  Debug Hessian Term 1: A={A}, B={B}, E={E}, F={F}, I={I}, J={J}, t_ABIJ={t_ABIJ:.6f}, integral={integral:.6f}, contrib={2.0 * t_ABIJ * integral:.6f}")
										print()

									if flat_index_H_col == 0 and flat_index_H_row == 7:
										print()
										print(f"  For Hessian, H[{flat_index_H_row},{flat_index_H_col}], ({idx_A * num_inactive} + {idx_E}, {idx_B * num_inactive} + {idx_F})")
										print(f"  Debug Hessian Term 1 (swapped): A={A}, B={B}, E={E}, F={F}, I={I}, J={J}, t_BAIJ={t_ABIJ:.6f}, integral={integral:.6f}, contrib={2.0 * t_ABIJ * integral:.6f}")
										print()

						# Terms 2-7 involve D_AB
						D_AB = 0.0
						for I in self.active_occ_indices:
							for J in self.active_occ_indices:
								if I > J:
									same_spin_IJ = (self.orbspin[I] == self.orbspin[J])
									
									for C in self.active_inocc_indices:
										same_spin_AC = (self.orbspin[A] == self.orbspin[C])
										same_spin_BC = (self.orbspin[B] == self.orbspin[C])

										same_spin_AI = (self.orbspin[A] == self.orbspin[I])
										same_spin_CJ = (self.orbspin[C] == self.orbspin[J])
										
										# Get t_ACIJ
										if (same_spin_IJ and same_spin_AC and same_spin_AI and same_spin_CJ) or (not same_spin_IJ and not same_spin_AC and same_spin_AI and same_spin_CJ):
											t_ACIJ = MP1_amplitudes[A, C, I, J] if C > A or (C == A and same_spin_AC) else -MP1_amplitudes[C, A, I, J] if same_spin_IJ and same_spin_AC else MP1_amplitudes[C, A, I, J]
										else:
											t_ACIJ = 0.0

										same_spin_BI = (self.orbspin[B] == self.orbspin[I])
										same_spin_CJ = (self.orbspin[C] == self.orbspin[J])

										# Get t_BCIJ
										if (same_spin_IJ and same_spin_BC and same_spin_BI and same_spin_CJ) or (not same_spin_IJ and not same_spin_BC and same_spin_BI and same_spin_CJ):
											t_BCIJ = MP1_amplitudes[B, C, I, J] if C > B or (C == B and same_spin_BC) else -MP1_amplitudes[C, B, I, J] if same_spin_IJ and same_spin_BC else MP1_amplitudes[C, B, I, J]
										else:
											t_BCIJ = 0.0
										D_AB += t_ACIJ * t_BCIJ
						# Term 2: - ∑_{IJC} t_ACIJ <BC|IJ> δ_EF
						term2 = 0.0
						if E == F:
							for I in self.active_occ_indices:
								for J in self.active_occ_indices:
									if I > J:
										same_spin_IJ = (self.orbspin[I] == self.orbspin[J])
										
										for C in self.active_inocc_indices:
											same_spin_AC = (self.orbspin[A] == self.orbspin[C])
											same_spin_BC = (self.orbspin[B] == self.orbspin[C])

											same_spin_AI = (self.orbspin[A] == self.orbspin[I])
											same_spin_CJ = (self.orbspin[C] == self.orbspin[J])

											# Get t_ACIJ
											if (same_spin_IJ and same_spin_AC and same_spin_AI and same_spin_CJ) or (not same_spin_IJ and not same_spin_AC and same_spin_AI and same_spin_CJ):
												if A > C:
													t_ACIJ = MP1_amplitudes[A, C, I, J]
												elif C > A:
													t_ACIJ = -MP1_amplitudes[C, A, I, J] if same_spin_AC else MP1_amplitudes[C, A, I, J]
												else:  # A == C
													t_ACIJ = 0.0  # antisymmetry for same-spin
											else:
												t_ACIJ = 0.0

											same_spin_BI = (self.orbspin[B] == self.orbspin[I])
											same_spin_CJ = (self.orbspin[C] == self.orbspin[J])

											# <BC|IJ>
											if (same_spin_IJ and same_spin_BC and same_spin_BI and same_spin_CJ) or (not same_spin_IJ and not same_spin_BC and same_spin_BI and same_spin_CJ):
												if same_spin_IJ and same_spin_BC:
													integral = eri_spin[B, C, I, J] - eri_spin[B, C, J, I]
												else:
													integral = eri_spin[B, C, I, J]
												
												term2 -= t_ACIJ * integral

												
						# Term 3: - ∑_{IJC} t_CBIJ <CA|IJ> δ_EF
						term3 = 0.0
						if E == F:
							for I in self.active_occ_indices:
								for J in self.active_occ_indices:
									if I > J:
										same_spin_IJ = (self.orbspin[I] == self.orbspin[J])
										
										for C in self.active_inocc_indices:
											same_spin_CB = (self.orbspin[C] == self.orbspin[B])
											same_spin_CA = (self.orbspin[C] == self.orbspin[A])

											same_spin_CI = (self.orbspin[C] == self.orbspin[I])
											same_spin_BJ = (self.orbspin[B] == self.orbspin[J])

											# Get t_CBIJ
											if (same_spin_IJ and same_spin_CB and same_spin_CI and same_spin_BJ) or (not same_spin_IJ and not same_spin_CB and same_spin_CI and same_spin_BJ):
												if C > B:
													t_CBIJ = MP1_amplitudes[C, B, I, J]
												elif B > C:
													t_CBIJ = -MP1_amplitudes[B, C, I, J] if same_spin_CB else MP1_amplitudes[B, C, I, J]
												else:  # C == B
													t_CBIJ = 0.0
											else:
												t_CBIJ = 0.0

											same_spin_CI = (self.orbspin[C] == self.orbspin[I])
											same_spin_AJ = (self.orbspin[A] == self.orbspin[J])

											# <CA|IJ>
											if (same_spin_IJ and same_spin_CA and same_spin_CI and same_spin_AJ) or (not same_spin_IJ and not same_spin_CA and same_spin_CI and same_spin_AJ):
												if same_spin_IJ and same_spin_CA:
													integral = eri_spin[C, A, I, J] - eri_spin[C, A, J, I]
												else:
													integral = eri_spin[C, A, I, J]
												
												term3 -= t_CBIJ * integral

												
						# Term 4: + D_AB F_AA δ_EF
						term4 = D_AB * Fmo_spin[A, A] if E == F else 0.0

						# Term 5: - D_AB F_BB δ_EF
						term5 = -D_AB * Fmo_spin[B, B] if E == F else 0.0

						# Term 6: - D_AB F_EF δ_EF
						term6 = -D_AB * Fmo_spin[E, F] if E == F else 0.0

						# Term 7: + D_AB F_EF
						term7 = D_AB * Fmo_spin[E, F]

						H[flat_index_H_row, flat_index_H_col] = term1 + term2 + term3 + term4 + term5 + term6 + term7

						# Detailed printout of Hessian contributions
						if abs(H[flat_index_H_row, flat_index_H_col]) > 1e-6:
							print(f"\nComputing Hessian H[{flat_index_H_row},{flat_index_H_col}] for A={A}, E={E}, B={B}, F={F}:")
							print(f"  Contributions:")
							print(f"    Term 1: {term1:.6f}")
							print(f"    Term 2: {term2:.6f}")
							print(f"    Term 3: {term3:.6f}")
							print(f"    Term 4: {term4:.6f}")
							print(f"    Term 5: {term5:.6f}")
							print(f"    Term 6: {term6:.6f}")
							print(f"    Term 7: {term7:.6f}")
							print(f"  Total: {H[flat_index_H_row, flat_index_H_col]:.6f}")
							print("")
			
			# Check H is not all zeros
		count_nonzero_H = np.count_nonzero(H)
		assert count_nonzero_H > 0, "Hessian H is all zeros!"
			# Check H for NaN or Inf values
		assert np.all(np.isfinite(H)), "Hessian H contains non-finite values!"

			# Continue by looking into H
		# print("\n #### HESSIAN H STATISTICS ####")
		# print("Statistics of Hessian H values:")
		# print("  Shape: ", H.shape)
		# print("  Min: ", np.min(H))
		# print("  Max: ", np.max(H))
		# print("  Mean: ", np.mean(H))
		# print("  Std: ", np.std(H))
		# print("  Number of non-zero elements: ", count_nonzero_H)
		# print("="*70)

			# Check that H is symmetric by looking at upper and lower triangles
		print("\n ### CHECKING HESSIAN SYMMETRY ###")
		for i in range(H.shape[0]):
			for j in range(i+1, H.shape[1]):
				diff = np.abs(H[i,j] - H[j,i])
				if diff > 1e-6:
					print(f"H[{i},{j}] = {H[i,j]:.6e}, H[{j},{i}] = {H[j,i]:.6e}, |diff| = {diff:.6e}")

					# Calculate specific indices A,E,B,F from if diff is large
					idx_A = i // num_inactive
					idx_E = i % num_inactive
					idx_B = j // num_inactive
					idx_F = j % num_inactive

					A = self.active_inocc_indices[idx_A]
					E = self.inactive_indices[idx_E]
					B = self.active_inocc_indices[idx_B]
					F = self.inactive_indices[idx_F]

					print(f"  Corresponding to A={A}, E={E}, B={B}, F={F} -> Orbital spins: A({self.orbspin[A]}), E({self.orbspin[E]}), B({self.orbspin[B]}), F({self.orbspin[F]})")
				
		
					print("  Diagnostic for pair", i,j, "-> A,E,B,F:", A,E,B,F)
					for I in self.active_occ_indices:
						for J in self.active_occ_indices:
							if I <= J: continue
							print("   t(A,B,I,J) ", MP1_amplitudes[A,B,I,J], " t(B,A,I,J) ", MP1_amplitudes[B,A,I,J],
								" eri[E,F,I,J] ", eri_spin[E,F,I,J], " eri[F,E,I,J] ", eri_spin[F,E,I,J])
							
							
			# Check H is symmetric
		diff_H = np.linalg.norm(H - H.T)
		assert diff_H < 1e-6, f"Hessian H is not symmetric, ||H - H.T|| = {diff_H}"


		# Step (vi): Use the Newton-Raphson method to minimize the second-order Hylleraas functional

		# solve for rotation parameters
			# Original direct inversion method
				# equation 14: R = - G H^-1 -> R = -G @ np.linalg.inv(H)

		# Set to True to use Reduced Linear Equation method
		use_RLE = False  

		# Direct inversion method
		if not use_RLE:
			# Float precision for numerical stability
			G = G.astype(np.float64)
			H = H.astype(np.float64)

			# Solve for R, unoccupied space
			R = - G @ np.linalg.inv(H)

			# Initialize R,
				# Matrix: self.full_indices x self.full_indices
			R_matrix = np.zeros((len(self.full_indices), len(self.full_indices)))

			# Build R_matrix from R[A, E]
			for idx_A, A in enumerate(self.active_inocc_indices):
				for idx_E, E in enumerate(self.inactive_indices):
					
					flat_index_R = idx_A * len(self.inactive_indices) + idx_E

					R_matrix[E, A] = R[flat_index_R]  
					R_matrix[A, E] = -R[flat_index_R]  # Anti-symmetry
			

		# Try the option of implementing the RLE for R,
			# as described in the paper, instead of direct inversion of H.
		if use_RLE:	
			# ...
			raise NotImplementedError("Reduced Linear Equation (RLE) method not implemented yet.")
		

			# Check shape
		expected_shape = np.zeros((len(self.full_indices), len(self.full_indices))).shape
		assert R_matrix.shape == expected_shape, f"R_matrix shape is {R_matrix.shape}, expected {expected_shape}"
			# Check that R is anti-symmetric
		diff_R = np.linalg.norm(R_matrix + R_matrix.T)
		assert diff_R < 1e-6, f"R_matrix is not anti-symmetric, ||R + R.T|| = {diff_R}"
			# Check that R_matrix has no NaN or Inf values
		assert np.all(np.isfinite(R_matrix)), "R_matrix contains NaN or Inf values!"
			# Check that R_matrix is not all zeros
		assert not np.allclose(R_matrix, 0), "R_matrix is all zeros!"




		# Step (vii): Construct the unitary orbital rotation matrix U = exp(R)

		# Float precision for numerical stability
		R_matrix = R_matrix.astype(np.float64)

		# Unitary rotation matrix
		U = scipy.linalg.expm(R_matrix)

		# Numerical checks on U
			# Check U if U^T U = I
		assert np.allclose(U.T @ U, np.eye(len(U)), atol=1e-6), "Unitary rotation matrix U is not unitary!"
		
			# Check is U has no NaN or Inf values
		assert np.all(np.isfinite(U)), "Unitary rotation matrix U contains NaN or Inf values!"
			# Check that U is not all zeros
		assert not np.allclose(U, 0), "Unitary rotation matrix U is all zeros!"


		# Check that U is orthogonal
			# Note, sometimes due to numerical errors (e-11) U@U.T is not exactly identity, but very close
				# If the difference is too large, raise an error
		diff = np.linalg.norm(U@U.T - np.eye(len(U)))
		assert diff < 1e-6, f"U is not orthogonal, ||U@U.T - I|| = {diff}"

		# Step (viii): Rotate the orbitals

				# rotate orbitals, 
			# convert to spin orbital basis
				# Manual spatial→spin conversion: interleave α and β columns
					# PySCF convention: mo_coeffs shape is (n_ao, n_spatial_orbs), columns are orbitals
					# mo_coeffs[0] = alpha, mo_coeffs[1] = beta
					# Result: (n_ao, n_spin_orbs) where spin_orbs = [α0, β0, α1, β1, ...]
		n_ao, n_spatial_orbs = mo_coeffs.shape[1], mo_coeffs.shape[2]
		mo_coeffs_spin = np.zeros((n_ao, 2 * n_spatial_orbs))
		for i in range(n_spatial_orbs):
			mo_coeffs_spin[:, 2*i] = mo_coeffs[0][:, i]      # alpha orbital i (column)
			mo_coeffs_spin[:, 2*i+1] = mo_coeffs[1][:, i]    # beta orbital i (column)

			
			# apply rotation
				# mo_coeffs_spin shape: (n_ao, num_spin_orbitals)
				# U shape: (num_spin_orbitals, num_spin_orbitals)
				# Result: C_new = C_old @ U (standard orbital rotation)
		mo_coeffs_spin_rot = mo_coeffs_spin @ U

			# convert back to spatial orbital basis
				# Manual spin→spatial conversion: extract α and β columns using orbspin
		alpha_indices = np.where(self.orbspin == 0)[0]  # indices where spin is alpha
		beta_indices = np.where(self.orbspin == 1)[0]   # indices where spin is beta
		mo_coeffs_alpha_rot = mo_coeffs_spin_rot[:, alpha_indices]  # extract alpha columns
		mo_coeffs_beta_rot = mo_coeffs_spin_rot[:, beta_indices]    # extract beta columns
		mo_coeffs_rot = np.array([mo_coeffs_alpha_rot, mo_coeffs_beta_rot])


				# check shape
		for spin in [0, 1]:
			expected_shape_spatial = mo_coeffs[spin].shape
			assert mo_coeffs_rot[spin].shape == expected_shape_spatial, f"mo_coeffs_rot shape is {mo_coeffs_rot[spin].shape}, expected {expected_shape_spatial}"

		# Check that rotated orbitals are orthonormal
			# Sum a coloumn of mo_coeffs to check normalization for a given spin
				# Check: C^T S C = I
			C_i = mo_coeffs_rot[spin]
			norm = C_i.T @ self.S @ C_i
			assert np.allclose(norm, np.eye(norm.shape[0]), atol=1e-6), f"MO coefficients for spin {spin} are not orthonormal!"


			# Print shape and matrix of matrices and vectors for checking
		# print("")
		# print("Density matrix, shape: ", D_AB_cache.shape, "\n", D_AB_cache)
		# print("")
		# print("Gradient G, shape: ", G.shape, "\n", G)
		# print("")
		# print("Hessian H, shape: ", H.shape, "\n", H)
		# print("")
		# print("Orbital rotation parameter matrix R, shape: ", R_matrix.shape, "\n", R_matrix)
		# print("")
		# print("Unitary rotation matrix U, shape: ", U.shape, "\n", U)
		# print("")
			# Print mo_coeffs before and after rotation for checking
		# print("")
		# print("MO coefficients before rotation, shape: ", mo_coeffs[0].shape, "\n", mo_coeffs)
		# print("")
		# print("MO coefficients after rotation, shape: ", mo_coeffs_rot[0].shape, "\n", mo_coeffs_rot)
		# print("")
		# diff_mo_coeffs = np.linalg.norm(mo_coeffs_rot[0] - mo_coeffs[1])
		# print("Diff in Rotated MO coefficients, shape: ", mo_coeffs_rot[spin].shape, "\n", diff_mo_coeffs)
		# print("")

				# Are the optimal solution when alpha and beta orbitals are the same? !!!!!!!!

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
			# print("MP2 correlation energy: ", E_corr)
			# print()

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

find_atom = {
	"H2": 0,
	"LiH": 1,
	"H2O": 2,
	"CH2": 3
}	

find_basis = {
	"STO-3G": 0,
	"6-31G": 1
}




# Select molecule and basis set
select_atom = "LiH"  # Select molecule index here
select_basis = "STO-3G" # Select molecule index here

atom, basis = (atom_choose_between[find_atom[select_atom]], basis_choose_between[find_basis[select_basis]])

# Get number of electrons and full space size in molecular orbitals
mol = pyscf.M(atom=atom, basis=basis, unit=unit)
	# Number of electrons
num_electrons = mol.nelec[0] + mol.nelec[1]
	# Full space size in molecular orbitals
full_space_size = int(pyscf.scf.UHF(mol).run().mo_coeff.shape[1])



"""
Run OVOS for different numbers of optimized virtual orbitals
"""
run_different_virt_orbs = True
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

	while num_opt_virtual_orbs_current < 2: # max_opt_virtual_orbs:  
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

	str_atom = select_atom
	str_basis = select_basis

	# Save MP2 correlation energy convergence data
	with open("branch/data/"+str_atom+"/"+str_basis+"/lst_MP2_"+str_name+".json", "w") as f:
		json.dump(lst_E_corr_virt_orbs, f, indent=2)

	print("Data saved to branch/data/"+str_atom+"/"+str_basis+"/...")
