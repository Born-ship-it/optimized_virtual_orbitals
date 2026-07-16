"""
Check Qiskit imports for debugging.
"""

print("Checking Qiskit imports...")

# Check qiskit version
try:
    import qiskit
    print(f"✓ Qiskit version: {qiskit.__version__}")
except ImportError:
    print("✗ Qiskit not installed")

# Check primitives - try different import paths
print("\nChecking qiskit.primitives:")
try:
    from qiskit.primitives import Estimator
    print("  ✓ Estimator imported from qiskit.primitives")
except ImportError as e:
    print(f"  ✗ Estimator import failed: {e}")
    try:
        from qiskit.primitives import EstimatorV2
        print("  ✓ EstimatorV2 imported from qiskit.primitives")
    except ImportError:
        print("  ✗ No Estimator found in qiskit.primitives")

# Check qiskit_algorithms
print("\nChecking qiskit_algorithms:")
try:
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import L_BFGS_B
    print("  ✓ VQE imported from qiskit_algorithms")
    print(f"  ✓ L_BFGS_B imported from qiskit_algorithms")
except ImportError as e:
    print(f"  ✗ qiskit_algorithms import failed: {e}")

# Check qiskit-nature
print("\nChecking qiskit-nature:")
try:
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.circuit.library import UCCSD
    print("  ✓ qiskit-nature imports successful")
except ImportError as e:
    print(f"  ✗ qiskit-nature import failed: {e}")

# Check qiskit-aer
print("\nChecking qiskit-aer:")
try:
    from qiskit_aer import AerSimulator
    print("  ✓ qiskit-aer imported successfully")
except ImportError as e:
    print(f"  ✗ qiskit-aer import failed: {e}")

print("\n" + "="*50)
print("Summary:")
print("="*50)
# Count successes
success_count = 0
# Simplistic summary - just report that we checked
print("Check complete. Review the output above for any ✗ indicators.")