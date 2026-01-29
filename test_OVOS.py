"""
OVOS test script

The OVOS algorithm minimizes the second-order correlation energy (MP2)
using orbital rotations.

Please select a valid molecule and basis set from the lists provided below.

Requires: pyscf
"""

import os
# Force single-threaded
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# For decimal formatting
from decimal import Decimal

import pyscf

def test_OVOS_CH2_6_31G_6_virt_orbs():
    """
    Test OVOS on CH2 with 6-31G basis set and 6 optimized virtual orbitals
    """
        # Select molecule and basis set
    select_atom = "CH2"  		# Select atom index here
    select_basis = "6-31G"  	# Select basis index here
        # Number of optimized virtual orbitals
    num_opt_virtual_orbs_current = 4  

    # Run OVOS
    run_OVOS(select_atom, select_basis, num_opt_virtual_orbs_current)


def test_OVOS_H2O_6_31G_5_virt_orbs():
    """
    Test OVOS on H2O with 6-31G basis set and 5 optimized virtual orbitals
    """
        # Select molecule and basis set
    select_atom = "H2O"  		# Select atom
    select_basis = "6-31G"  	# Select basis
        # Number of optimized virtual orbitals
    num_opt_virtual_orbs_current = 5  

    # Run OVOS
    run_OVOS(select_atom, select_basis, num_opt_virtual_orbs_current)


def test_OVOS_BH3_6_31G_6_virt_orbs():
    """
    Test OVOS on BH3 with 6-31G basis set and 6 optimized virtual orbitals
    """
        # Select molecule and basis set
    select_atom = "BH3"  		# Selected atom 
    select_basis = "6-31G"  	# Selected basis
        # Number of optimized virtual orbitals
    num_opt_virtual_orbs_current = 6 

    # Run OVOS
    run_OVOS(select_atom, select_basis, num_opt_virtual_orbs_current)


def test_OVOS_N2_6_31G_7_virt_orbs():
    """
    Test OVOS on N2 with 6-31G basis set and 7 optimized virtual orbitals
    """
        # Select molecule and basis set
    select_atom = "N2"  		# Selected atom 
    select_basis = "6-31G"  	# Selected basis
        # Number of optimized virtual orbitals
    num_opt_virtual_orbs_current = 7 

    # Run OVOS
    run_OVOS(select_atom, select_basis, num_opt_virtual_orbs_current)




# Helper function to run OVOS
def run_OVOS(select_atom, select_basis, num_opt_virtual_orbs_current):
    
    """
    Setup OVOS
    """
        # Molecule
    atom_choose_between = [
        "O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;",  # H2O equilibrium geometry
        "C  0.0000  0.0000  0.0000; H  0.0000  0.9350  0.5230; H  0.0000 -0.9350  0.5230;", # CH2 
        "B 0 0 0; H 0 0 1.19; H 0 1.03 -0.40; H 0 -1.03 -0.40",  # BH3 equilibrium geometry
        "N 0 0 0; N 0 0 1.10", # N2 bond length 1.10 Angstrom
    ]
        # Basis set
    basis_choose_between = ["STO-3G",	"STO-6G",	"3-21G",	"6-31G",	"DZP",	"roosdz",	"anoroosdz",	"cc-pVDZ",	"cc-pV5Z",	"def2-QZVPP",	"aug-cc-pV5Z",	"ANO"]
        
        # Dictionaries to find index by name
    get_atom = {"H2": 0,	"LiH": 1,	"H2O": 2,	"CH2": 3,	"BH3": 4,	"N2": 5}	
        # Get atom and basis
    atom, basis = (atom_choose_between[get_atom[select_atom]],    basis_choose_between[basis_choose_between.index(select_basis)],)

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
    MP2 = uhf.MP2().run()

    # Print summary of the run
    print("Number of electrons: ", num_electrons)
    print("Full space size in molecular orbitals: ", full_space_size)
    print("MP2 correlation energy for the full space: ", MP2.e_corr)
    print("Number of optimized virtual orbitals: ", num_opt_virtual_orbs_current)
    print("")

    """
    Run OVOS
    """
        # Import OVOS module
    from ovos.py import OVOS

        # Initial orbitals: UHF molecular orbitals
    mo_coeffs = uhf.mo_coeff
    init_orbs = "UHF"
    print("    Using previously optimized orbitals as starting guess.")				
    print("")

        # Run OVOS
    try:
        init_OVOS = OVOS(mol, num_opt_virtual_orbs_current, init_orbs)
        lst_E_corr, lst_iter_counts, mo_coeffs = init_OVOS.run_ovos(mo_coeffs)

    except AssertionError as e:
        # Print error message
        print(f"Error during OVOS with {num_opt_virtual_orbs_current} optimized virtual orbitals: {e}")
        lst_error_messages = ((num_opt_virtual_orbs_current, str(e), lst_iter_counts[-1] if lst_iter_counts is not None else 0))

    """
    Print results
    """
        # Print the final MP2 correlation energy after OVOS and amount of iterations till convergence
    if 'lst_E_corr' in locals():
            # Get final energy
        E_corr = lst_E_corr[-1]
            # Get number of iterations
        iter_ = lst_iter_counts[-1]
            # Print results
        print("#### OVOS results ####")
        print("MP2 correlation energy, for ", num_opt_virtual_orbs_current, f" optimized virtual orbitals: ", '%.5E' % Decimal(E_corr), "  @ ", iter_, " iterations till convergence")
        print("MP2 correlation energy, for full space: ", '%.5E' % Decimal(MP2.e_corr), "| Difference:", '%.5E' % Decimal(MP2.e_corr - E_corr))
        print("")

        # Print error messages summary
    if 'lst_error_messages' in locals():
            # Unpack error messages
        num_opt_virtual_orbs_current, error_msg, iter_ = lst_error_messages
            # Print summary
        print("#### Error messages ####")
        print("  OVOS w. ", num_opt_virtual_orbs_current, " optimized vorbs failed at iteration ", iter_ ," w. error: ", error_msg)
        print("")
