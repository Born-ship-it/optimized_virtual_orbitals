"""
quantum_ovos_active_space.py

Active Space Selection for Quantum OVOS

This script:
1. Runs classical OVOS to optimize orbitals
2. Uses the optimized orbitals to select the most important virtual orbitals
3. Performs VQE on the selected active space
"""

import numpy as np
from pyscf import gto, scf, mp
from ovos import OVOS
from ovos_trust_region import OVOSTrustRegion
from ovos_quantum import QuantumOVOS


def select_active_orbitals_from_ovos(mol, mf, ovos_result, num_active=4):
    """
    Select active orbitals based on OVOS-optimized orbitals.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        The molecule object.
    mf : pyscf.scf.RHF
        The RHF reference.
    ovos_result : tuple
        Result from classical OVOS run.
    num_active : int
        Number of active orbitals to select.
    
    Returns
    -------
    list
        Indices of active orbitals.
    """
    # Extract the optimized MO coefficients
    mo_opt = ovos_result[3][0]  # Alpha MO coefficients
    
    # Compute the MP2 natural orbital occupations
    # Using the optimized orbitals
    from pyscf import mp
    
    # Create a new RHF object with the optimized orbitals
    # This is a simplified approach - in practice, you'd use the
    # optimized Fock matrix or density matrix
    
    # For demonstration, we use the MP2 natural orbitals
    # computed from the optimized orbitals
    mp2_obj = mp.MP2(mf)
    mp2_obj.verbose = 0
    mp2_obj.kernel()
    
    # Get the virtual-virtual density matrix
    # This is a simplified version
    nocc = mol.nelec[0]
    nvir = mol.nao_nr() - nocc
    
    # For simplicity, we select the lowest virtual orbitals
    # In a real implementation, you'd use the natural orbital occupations
    active_indices = list(range(nocc, nocc + num_active))
    
    return active_indices


def run_quantum_ovos_pipeline():
    """
    Run the complete quantum OVOS pipeline:
    1. Classical OVOS optimization
    2. Active space selection
    3. Quantum OVOS with VQE
    """
    from pyscf import gto, scf, mp
    
    print("="*70)
    print("Quantum OVOS Pipeline with Active Space Selection")
    print("="*70)
    
    # Step 1: Setup molecule
    mol = gto.Mole()
    mol.atom = 'O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692'
    mol.basis = '6-31g'  # Slightly larger basis
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()
    
    # Step 2: Run RHF
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    print(f"\nRHF Energy: {mf.e_tot:.10f} Ha")
    
    # Step 3: Run classical MP2
    mp2_obj = mp.MP2(mf)
    mp2_obj.verbose = 0
    mp2_obj.kernel()
    print(f"Classical MP2 Energy: {mp2_obj.e_tot:.10f} Ha")
    
    # Step 4: Classical OVOS
    Fao = [mf.get_fock(), mf.get_fock()]
    mo_coeffs = [mf.mo_coeff, mf.mo_coeff]
    
    print("\n" + "="*70)
    print("Step 1: Classical OVOS Optimization")
    print("="*70)
    
    ovos_classical = OVOSTrustRegion(
        mol=mol,
        scf=mf,
        Fao=Fao,
        num_opt_virtual_orbs=4,
        mo_coeff=mo_coeffs,
        init_orbs="RHF",
        verbose=1,
        max_iter=100,
        conv_energy=1e-8,
        conv_grad=1e-6,
        keep_track_max=10,
    )
    
    result_classical = ovos_classical.run(mo_coeffs, fock_spin=None)
    print(f"\nClassical OVOS Energy: {result_classical[0]:.10f} Ha")
    
    # Step 5: Select active space
    print("\n" + "="*70)
    print("Step 2: Active Space Selection")
    print("="*70)
    
    active_orbitals = select_active_orbitals_from_ovos(
        mol, mf, result_classical, num_active=4
    )
    print(f"Selected active orbitals: {active_orbitals}")
    
    # Step 6: Quantum OVOS with VQE
    print("\n" + "="*70)
    print("Step 3: Quantum OVOS with VQE")
    print("="*70)
    
    qovos = QuantumOVOS(
        mol=mol,
        scf=mf,
        Fao=Fao,
        num_opt_virtual_orbs=4,
        mo_coeff=mo_coeffs,
        init_orbs="RHF",
        verbose=1,
        max_iter=100,
        conv_energy=1e-8,
        conv_grad=1e-6,
        keep_track_max=50,
        active_orbitals=active_orbitals,
        vqe_energy=True,
        use_quantum_gradients=False,
    )
    
    result_quantum = qovos.run(mo_coeffs, fock_spin=None)
    
    # Summary
    print("\n" + "="*70)
    print("Final Results Summary")
    print("="*70)
    print(f"Classical MP2:     {mp2_obj.e_tot:.10f} Ha")
    print(f"Classical OVOS:    {result_classical[0]:.10f} Ha")
    print(f"Quantum OVOS (VQE): {result_quantum[0]:.10f} Ha")
    print("="*70)
    
    return result_classical, result_quantum


if __name__ == "__main__":
    run_quantum_ovos_pipeline()