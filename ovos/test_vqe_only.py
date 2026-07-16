"""
test_vqe_only.py

Standalone VQE test on active space for water.
Verifies that the VQE energy is computed correctly within the active space.
"""

import numpy as np
import time
import warnings
from pyscf import gto, scf, mp, ao2mo

# Import the quantum evaluator from your code
try:
    from ovos_quantum_full import QuantumEnergyEvaluator, QuantumHamiltonianBuilder
except ImportError:
    from ovos_quantum_full import QuantumEnergyEvaluator, QuantumHamiltonianBuilder


def setup_molecule(basis='cc-pVDZ'):
    """Set up water molecule."""
    mol = gto.Mole()
    mol.atom = 'O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692'
    mol.basis = basis
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()
    
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mol, mf


def get_active_orbitals_for_vqe(mol, num_opt_virtual_orbs):
    """
    Get the active orbital indices for VQE.
    These are: all occupied orbitals + active virtual orbitals.
    """
    n_occ = mol.nelec[0] + mol.nelec[1]
    active_indices = list(range(n_occ + num_opt_virtual_orbs))
    return active_indices


def compute_active_space_mp2(mol, mf, active_spin_indices):
    """
    Compute MP2 correlation energy in the active space only.
    active_spin_indices are spin-orbital indices (0, 1, 2, ...)
    """
    # Convert spin indices to spatial indices (i // 2)
    active_spatial_indices = sorted(set([i // 2 for i in active_spin_indices]))
    
    # Get the MO coefficients for the active space
    mo_coeff_full = mf.mo_coeff
    mo_coeff_active = mo_coeff_full[:, active_spatial_indices]
    
    # Get the active space Fock matrix
    fock_ao = mf.get_fock()
    fock_mo = mo_coeff_full.T @ fock_ao @ mo_coeff_full
    fock_active = fock_mo[np.ix_(active_spatial_indices, active_spatial_indices)]
    
    # Get the active space 2-electron integrals in MO basis
    eri_ao = mol.intor('int2e')
    eri_mo = ao2mo.incore.general(eri_ao, (mo_coeff_active, mo_coeff_active, 
                                            mo_coeff_active, mo_coeff_active))
    n_active_spatial = len(active_spatial_indices)
    eri_mo = eri_mo.reshape(n_active_spatial, n_active_spatial, 
                            n_active_spatial, n_active_spatial)
    
    # Number of occupied orbitals in the active space
    n_occ = mol.nelec[0]  # Number of occupied spatial orbitals
    n_active = n_active_spatial
    
    eps_active = np.diag(fock_active)
    
    energy = 0.0
    # Sum over occupied pairs (i,j) and virtual pairs (a,b)
    for i in range(n_occ):
        for j in range(n_occ):
            for a in range(n_occ, n_active):
                for b in range(n_occ, n_active):
                    # <ij||ab> = <ij|ab> - <ij|ba>
                    v_ijab = eri_mo[i, j, a, b] - eri_mo[i, j, b, a]
                    denom = eps_active[i] + eps_active[j] - eps_active[a] - eps_active[b]
                    if abs(denom) > 1e-12:
                        energy += v_ijab * v_ijab / denom
    
    # MP2 correlation energy is 1/4 of the sum
    return energy * 0.25


def run_vqe_test(verbose=1):
    """Run a standalone VQE test on the active space."""
    print("="*70)
    print("Standalone VQE Test on Active Space")
    print("="*70)
    
    # Setup molecule - use 6-31G for faster testing
    print("\n🔹 Setting up water molecule (6-31G)...")
    mol, mf = setup_molecule('6-31G')
    print(f"   RHF Energy (full): {mf.e_tot:.10f} Ha")
    print(f"   Number of spin-orbitals (full): {2 * mol.nao_nr()}")
    
    # Define active space
    num_opt_virtual_orbs = 4  # Active virtual spin-orbitals
    active_orbitals = get_active_orbitals_for_vqe(mol, num_opt_virtual_orbs)
    n_active_spin = len(active_orbitals)
    n_occ = mol.nelec[0] + mol.nelec[1]
    n_active_virtual = n_active_spin - n_occ
    
    print(f"\n📐 Active Space:")
    print(f"   Active occupied:     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")
    print(f"   Active virtual:      [10, 11, 12, 13]")
    print(f"   Active indices:      {active_orbitals}")
    print(f"   Number of qubits:    {n_active_spin}")
    print(f"   Active electrons:    {n_occ}")
    print(f"   Active virtuals:     {n_active_virtual}")
    
    nuclear_repulsion = mol.energy_nuc()
    rhf_active = mf.e_tot
    
    print(f"\n🔹 Active Space RHF Energy: {rhf_active:.10f} Ha")
    
    # Compute active space MP2 for comparison
    print("\n🔹 Computing MP2 in active space...")
    mp2_active = compute_active_space_mp2(mol, mf, active_orbitals)
    total_mp2_active = rhf_active + mp2_active
    print(f"   MP2 active correlation: {mp2_active:.10f} Ha")
    print(f"   MP2 active total:       {total_mp2_active:.10f} Ha")
    
    # Initialize the quantum energy evaluator
    print("\n🔹 Initializing Quantum Energy Evaluator...")
    start = time.time()
    
    evaluator = QuantumEnergyEvaluator(
        mol, mf, 
        active_orbitals=active_orbitals,
        verbose=verbose,
        use_simple_ansatz=False,  # Use UCCSD directly
        max_qubits=20,
        cache_dir='hamiltonian_cache_test'
    )
    init_time = time.time() - start
    print(f"   Initialization time: {init_time:.2f} s")
    
    if not evaluator.can_run_vqe:
        print(f"\n❌ Cannot run VQE: {evaluator.n_qubits} qubits > {evaluator.max_qubits}")
        return
    
    print(f"\n🔹 Hamiltonian built with {len(evaluator.hamiltonian)} Pauli terms")
    
    # Check initial parameters
    print(f"\n🔹 Initial parameters norm: {np.linalg.norm(evaluator.initial_params):.6f}")
    print(f"   Initial parameters min/max: {np.min(evaluator.initial_params):.6f} / {np.max(evaluator.initial_params):.6f}")
    
    # Run full VQE optimization within active space
    print("\n🔹 Running VQE optimization (active space)...")
    start = time.time()
    
    vqe_opt_energy, opt_params = evaluator._run_vqe_statevector(evaluator.hamiltonian)
    opt_time = time.time() - start
    
    vqe_opt_total = vqe_opt_energy + nuclear_repulsion
    vqe_opt_corr = vqe_opt_total - rhf_active
    
    print(f"\n📊 VQE Results:")
    print(f"   Optimized electronic energy: {vqe_opt_energy:.10f} Ha")
    print(f"   Total energy:                {vqe_opt_total:.10f} Ha")
    print(f"   Correlation energy:          {vqe_opt_corr:.10f} Ha")
    print(f"   Optimization time:           {opt_time:.2f} s")
    print(f"   Number of evaluations:       {len(evaluator._energy_history)}")
    
    # Check if parameters changed
    final_params_norm = np.linalg.norm(opt_params) if opt_params is not None else 0
    initial_params_norm = np.linalg.norm(evaluator.initial_params)
    print(f"\n   Initial params norm: {initial_params_norm:.6f}")
    print(f"   Final params norm:   {final_params_norm:.6f}")
    print(f"   Norm change:         {final_params_norm - initial_params_norm:.6f}")
    
    # Check if energy changed
    if len(evaluator._energy_history) > 1:
        first_energy = evaluator._energy_history[0]
        last_energy = evaluator._energy_history[-1]
        energy_change = last_energy - first_energy
        print(f"\n   Energy change: {energy_change:.10f} Ha")
        if abs(energy_change) < 1e-8:
            print("   ⚠️  Energy did not change during optimization.")
            print("      This indicates the VQE is stuck or the Hamiltonian is wrong.")
        else:
            print(f"   ✅ Energy changed by {abs(energy_change):.10f} Ha")
    
    # Summary
    print("\n" + "="*70)
    print("📊 Summary Comparison")
    print("="*70)
    print(f"{'Method':<25} {'Correlation (Ha)':<18} {'Total (Ha)':<18}")
    print("-"*70)
    print(f"{'Active RHF':<25} {0.0:>12.8f}   {rhf_active:>12.8f}")
    print(f"{'Active MP2':<25} {mp2_active:>12.8f}   {total_mp2_active:>12.8f}")
    print(f"{'VQE (optimized)':<25} {vqe_opt_corr:>12.8f}   {vqe_opt_total:>12.8f}")
    print("="*70)
    
    return {
        'active_rhf': rhf_active,
        'mp2_active_corr': mp2_active,
        'mp2_active_total': total_mp2_active,
        'vqe_opt_energy': vqe_opt_energy,
        'vqe_opt_total': vqe_opt_total,
        'vqe_opt_corr': vqe_opt_corr,
        'energy_history': evaluator._energy_history,
        'initial_params_norm': initial_params_norm,
        'final_params_norm': final_params_norm,
    }


def run_vqe_with_simple_ansatz():
    """Run VQE with the simple EfficientSU2 ansatz for comparison."""
    print("\n" + "="*70)
    print("VQE Test with Simple Ansatz")
    print("="*70)
    
    mol, mf = setup_molecule('6-31G')
    active_orbitals = get_active_orbitals_for_vqe(mol, 4)
    nuclear_repulsion = mol.energy_nuc()
    rhf_active = mf.e_tot
    
    print(f"\nActive space: {active_orbitals}")
    print(f"Active RHF: {rhf_active:.10f} Ha")
    
    evaluator = QuantumEnergyEvaluator(
        mol, mf,
        active_orbitals=active_orbitals,
        verbose=1,
        use_simple_ansatz=True,
        simple_ansatz_reps=2,
        max_qubits=20,
        cache_dir='hamiltonian_cache_test_simple'
    )
    
    print(f"\nAnsatz parameters: {evaluator.ansatz.num_parameters}")
    print(f"Hamiltonian terms: {len(evaluator.hamiltonian)}")
    
    start = time.time()
    energy, params = evaluator._run_vqe_statevector(evaluator.hamiltonian)
    elapsed = time.time() - start
    
    total_energy = energy + nuclear_repulsion
    corr_energy = total_energy - rhf_active
    
    print(f"\nResults:")
    print(f"  Optimized electronic energy: {energy:.10f} Ha")
    print(f"  Total energy:                {total_energy:.10f} Ha")
    print(f"  Correlation energy:          {corr_energy:.10f} Ha")
    print(f"  Time:                        {elapsed:.2f} s")
    print(f"  Evaluations:                 {len(evaluator._energy_history)}")
    
    return {
        'energy': energy,
        'total': total_energy,
        'corr': corr_energy,
        'time': elapsed,
        'evaluations': len(evaluator._energy_history),
    }


if __name__ == "__main__":
    results = run_vqe_test(verbose=1)
    
    print("\n" + "="*70)
    print("Testing with Simple Ansatz...")
    simple_results = run_vqe_with_simple_ansatz()