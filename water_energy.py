import pyscf
from pyscf import scf, cc, mcscf

import numpy as np

import os
# Force single-threaded
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

Opt_coord = "O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692"  # H2O equilibrium geometry

mol = pyscf.M(  # Redefine molecule given new coordinates
    atom= Opt_coord,
    basis="6-31G",  # Define basis
    spin= 0,  # Define the spin (2S)
    charge=0,  # Define the charge
    unit="angstrom",  # Define the coordinates in units of Bohr
    symmetry=False,  # Set symmetry
    #cart=False,  # Set cartesian orbitals
    #max_memory=50000,
    #verbose = 10,
)

mf = mol.RHF().run()
#uhf = scf.UHF(mol)
#mf.MP2().run()
E_RHF = mf.e_tot
#E_UHF = uhf.e_tot
E_MP2 = mf.MP2().run().e_tot
mycc = cc.CCSD(mf).run()
et = mycc.ccsd_t()

ncas, nelecas = (13,10) #FCI
#casscf = mf.CASSCF(ncas, nelecas).run()
casci = mf.CASCI(ncas, nelecas).run()
#ucasscf = mcscf.UCASSCF(uhf, ncas, nelecas)
#casscf.kernel()
#E_CASSCF = casscf.e_tot
E_casci = casci.e_tot

print()
print("#of orbitals", len(mf.mo_coeff))
print('RHF total energy', E_RHF)
print('MP2 total energy', E_MP2)
print('CCSD total energy', mycc.e_tot)
print('CCSD(T) total energy', mycc.e_tot + et)
print('FCI total energy', E_casci)
print()
print()
print("E_corr_MP2", E_MP2-E_RHF)
print("E_corr_CCSD",mycc.e_tot-E_RHF)
print("E_corr_CCSD(T)",mycc.e_tot + et-E_RHF)
print("E_corr_FCI",E_casci -E_RHF)