"""
test_quantum_ovos.py

Test script to compare classical OVOS vs Quantum OVOS
Compatible with Qiskit 2.4.2, qiskit-nature 0.7.2

Both methods report correlation energy for fair comparison.
"""

import time
import warnings
import numpy as np
from pyscf import gto, scf, mp
from ovos import OVOS

# Import Quantum OVOS
try:
    from ovos_quantum_full import QuantumOVOSFull
except ImportError as e:
    warnings.warn(f"Quantum OVOS import failed: {e}")
    QuantumOVOSFull = None


def check_qiskit_installation():
    """Check if Qiskit is properly installed."""
    try:
        import qiskit
        import qiskit_nature
        print(f"Qiskit version: {qiskit.__version__}")
        print(f"Qiskit Nature version: {qiskit_nature.__version__}")
        
        # Check for VQE availability
        try:
            from qiskit_algorithms import VQE
            print("✓ qiskit_algorithms available")
        except ImportError:
            print("✗ qiskit_algorithms not available")
        
        # Check for Aer
        try:
            from qiskit_aer import Aer
            print("✓ qiskit-aer available")
        except ImportError:
            print("✗ qiskit-aer not available")
        
        return True
    except ImportError as e:
        print(f"Qiskit not properly installed: {e}")
        return False


def get_active_orbitals_for_vqe(mol, num_opt_virtual_orbs):
    """
    Get the active orbital indices for VQE.
    These are: all occupied orbitals + active virtual orbitals.
    Inactive virtuals are excluded from the VQE calculation.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        The molecule object.
    num_opt_virtual_orbs : int
        Number of active virtual spin-orbitals.
    
    Returns
    -------
    list
        Indices of active orbitals for VQE.
    """
    n_occ = mol.nelec[0] + mol.nelec[1]  # Total occupied electrons
    n_spatial = mol.nao_nr()
    n_spin = 2 * n_spatial
    
    # Active occupied: 0 to n_occ-1
    # Active virtual: n_occ to n_occ + num_opt_virtual_orbs - 1
    active_indices = list(range(n_occ + num_opt_virtual_orbs))
    
    return active_indices


def to_float(value):
    """Convert complex or numpy number to float."""
    if isinstance(value, (complex, np.complexfloating)):
        return np.real(value)
    return float(value)


def run_classical_ovos(mol, mf, mo_coeffs, num_opt_virtual_orbs, **kwargs):
    """
    Run classical OVOS and return correlation energy only.
    """
    Fao = [mf.get_fock(), mf.get_fock()]
    params = {
        'verbose': 1,
        'max_iter': 100,
        'conv_energy': 1e-8,
        'conv_grad': 1e-6,
        'keep_track_max': 10,
        'num_opt_virtual_orbs': num_opt_virtual_orbs,
    }
    params.update(kwargs)
    
    ovos_obj = OVOS(mol=mol, scf=mf, Fao=Fao, mo_coeff=mo_coeffs, **params)
    start = time.time()
    result = ovos_obj.run(mo_coeffs, fock_spin=None)
    elapsed = time.time() - start
    
    # result[0] is the correlation energy (already)
    return {
        'correlation_energy': to_float(result[0]),
        'total_energy': to_float(result[0]) + to_float(mf.e_tot),
        'time': elapsed,
        'iterations': len(result[1]),
        'stop_reason': result[-1],
        'energy_history': [to_float(e) for e in result[1]],
    }


def run_quantum_ovos(mol, mf, mo_coeffs, num_opt_virtual_orbs, active_orbitals, **kwargs):
    """
    Run Quantum OVOS and return both correlation and total energy.
    """
    Fao = [mf.get_fock(), mf.get_fock()]
    params = {
        'verbose': 1,  # Reduce to 1 for cleaner output
        'max_iter': 100,
        'conv_energy': 1e-8,
        'conv_grad': 1e-6,
        'keep_track_max': 0,
        'num_opt_virtual_orbs': num_opt_virtual_orbs,
        'vqe_energy': True,
        'active_orbitals': active_orbitals,
        'max_vqe_qubits': 30,
        'switch_to_uccsd_iter': 3,
        'use_quantum_gradients': True,
        'grad_eps': 1e-5,
    }
    params.update(kwargs)
    
    qovos = QuantumOVOSFull(mol=mol, scf=mf, Fao=Fao, mo_coeff=mo_coeffs, **params)
    start = time.time()
    result = qovos.run(mo_coeffs, fock_spin=None)
    elapsed = time.time() - start
    
    # Convert complex numbers to floats
    total_energy = to_float(result[0])
    correlation_energy = to_float(result[0]) - to_float(mf.e_tot)
    
    return {
        'total_energy': total_energy,
        'correlation_energy': correlation_energy,
        'time': elapsed,
        'iterations': len(result[1]),
        'stop_reason': result[-1],
        'energy_history': [to_float(e) for e in result[1]],
        'vqe_energy_history': [to_float(e) for e in qovos.quantum_energy_hist] if hasattr(qovos, 'quantum_energy_hist') else [],
    }


def compare_methods():
    """
    Compare classical OVOS and Quantum OVOS on water.
    Both report correlation energy for fair comparison.
    """
    print("="*80)
    print("Classical vs Quantum OVOS Comparison (Correlation Energy)")
    print("="*80)
    
    # Check Qiskit installation
    qiskit_available = check_qiskit_installation()
    if not qiskit_available:
        print("\n⚠️  WARNING: Qiskit not available. Only classical OVOS will run.")
    
    # Setup molecule - using 6-31G for a more interesting test
    mol = gto.Mole()
    mol.atom = 'O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692'
    mol.basis = '6-31G'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()
    
    # Run RHF
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    print(f"\n🔹 RHF Energy:        {mf.e_tot:.10f} Ha")
    print(f"🔹 Number of spin-orbitals: {2 * mol.nao_nr()}")
    
    # Compute classical MP2 reference
    mp2_obj = mp.MP2(mf)
    mp2_obj.verbose = 0
    mp2_obj.kernel()
    print(f"\n📊 Reference Values:")
    print(f"   Classical MP2 Total:       {mp2_obj.e_tot:.10f} Ha")
    print(f"   Classical MP2 Correlation: {mp2_obj.e_corr:.10f} Ha")
    
    # Initial data
    Fao = [mf.get_fock(), mf.get_fock()]
    mo_coeffs = [mf.mo_coeff, mf.mo_coeff]
    
    # Define active space parameters
    num_opt_virtual_orbs = 2  # Active virtual spin-orbitals
    active_orbitals_vqe = get_active_orbitals_for_vqe(mol, num_opt_virtual_orbs)
    
    print(f"\n📐 Active Space for VQE:")
    print(f"   Active occupied:     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")
    print(f"   Active virtual:      [10, 11, 12, 13]")
    print(f"   Inactive virtual:    [14, 15, ...] (excluded from VQE)")
    print(f"   VQE active indices:  {active_orbitals_vqe}")
    print(f"   Number of qubits:    {len(active_orbitals_vqe)}")
    
    results = {}
    
    # ============================================================
    # Method 1: Classical OVOS
    # ============================================================
    print("\n" + "="*70)
    print("💻 Classical OVOS (Correlation Energy)")
    print("="*70)
    
    result_classical = run_classical_ovos(
        mol, mf, mo_coeffs,
        num_opt_virtual_orbs=num_opt_virtual_orbs,
    )
    results['Classical OVOS'] = result_classical
    print(f"\n✅ Correlation Energy: {result_classical['correlation_energy']:.10f} Ha")
    print(f"   Total Energy:       {result_classical['total_energy']:.10f} Ha")
    print(f"   Iterations:         {result_classical['iterations']}")
    print(f"   Time:               {result_classical['time']:.2f} s")
    print(f"   Stop Reason:        {result_classical['stop_reason']}")
    
    # ============================================================
    # Method 2: Quantum OVOS (if available)
    # ============================================================
    if QuantumOVOSFull is not None and qiskit_available:
        print("\n" + "="*70)
        print("⚛️   Quantum OVOS (VQE Correlation Energy)")
        print("   VQE uses only active orbitals, OVOS optimizes all virtuals")
        print("="*70)
        
        try:
            result_quantum = run_quantum_ovos(
                mol, mf, mo_coeffs,
                num_opt_virtual_orbs=num_opt_virtual_orbs,
                active_orbitals=active_orbitals_vqe,
                max_iter=5,  # Reduced for testing
            )
            results['Quantum OVOS'] = result_quantum
            print(f"\n✅ Correlation Energy: {result_quantum['correlation_energy']:.10f} Ha")
            print(f"   Total Energy:       {result_quantum['total_energy']:.10f} Ha")
            print(f"   Iterations:         {result_quantum['iterations']}")
            print(f"   Time:               {result_quantum['time']:.2f} s")
            print(f"   Stop Reason:        {result_quantum['stop_reason']}")
            
            # Show VQE energy history if available
            if result_quantum.get('vqe_energy_history'):
                print(f"   VQE Evaluations:    {len(result_quantum['vqe_energy_history'])}")
                if result_quantum['vqe_energy_history']:
                    print(f"   VQE Initial Total:  {result_quantum['vqe_energy_history'][0]:.10f} Ha")
                    print(f"   VQE Final Total:    {result_quantum['vqe_energy_history'][-1]:.10f} Ha")
        except Exception as e:
            print(f"\n❌ Quantum OVOS failed: {e}")
            import traceback
            traceback.print_exc()
            results['Quantum OVOS'] = None
    else:
        print("\n⚠️  Quantum OVOS not available (qiskit-nature not installed or import failed)")
        results['Quantum OVOS'] = None
    
    # ============================================================
    # Summary Table
    # ============================================================
    print("\n" + "="*85)
    print("📊 Summary: Correlation Energy Comparison")
    print("="*85)
    print(f"{'Method':<25} {'Correlation (Ha)':<18} {'Total (Ha)':<18} {'Iter':<6} {'Time (s)':<10}")
    print("-"*85)
    
    for name, data in results.items():
        if data is not None:
            prefix = "⚛️ " if "Quantum" in name else "💻 "
            print(f"{prefix}{name:<23} {data['correlation_energy']:>12.8f}   {data['total_energy']:>12.8f}   {data['iterations']:>4}     {data['time']:>8.2f}")
        else:
            print(f"{name:<25} {'N/A':>12}   {'N/A':>12}   {'-':>4}     {'-':>8}")
    print("="*85)
    
    # ============================================================
    # Explanation
    # ============================================================
    print("\n💡 Key Differences:")
    print("  • Classical OVOS: Uses MP2 correlation energy for orbital optimization")
    print("  • Quantum OVOS:   Uses VQE total energy for orbital optimization")
    print("  • Both methods optimize the same active virtual space")
    print("  • VQE is performed on active orbitals, excluding inactive virtuals")
    print("  • Inactive virtuals are still included in the OVOS optimization")
    
    # Calculate difference if both methods succeeded
    if results.get('Classical OVOS') and results.get('Quantum OVOS'):
        diff = results['Classical OVOS']['correlation_energy'] - results['Quantum OVOS']['correlation_energy']
        print(f"\n📈 Difference (Classical - Quantum): {diff:.10f} Ha")
        if abs(diff) < 1e-6:
            print("   → Both methods find the same minimum within numerical tolerance.")
        else:
            print("   → The quantum energy landscape differs from the classical MP2 landscape.")


if __name__ == "__main__":
    # Run the main comparison
    compare_methods()