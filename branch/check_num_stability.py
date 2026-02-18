"""
Check if my computer's numerical computations are stable.
"""

import numpy as np
np.__config__.show()

import scipy
import scipy.linalg
import logging
import time
import platform
import sys
from datetime import datetime

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_version_info():
    """Get version info for installed packages."""
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

class CheckNumStability:
    """
    Check numerical stability of computations on the current machine.
    """

    def run(self) -> dict:
        start_time = time.time()
        logger.info("Starting numerical stability check.")
        
        failed_checks = ""
        stable = True  # Initialize at the start

        # Generate two random matrices and compute condition numbers
        np.random.seed(0)
        A = np.random.rand(100, 100)
        cond_number = np.linalg.cond(A)

        B = np.random.rand(100, 100)
        cond_number_B = np.linalg.cond(B)

        # Perform a matrix inversion using numpy
        try:
            A_inv = B@np.linalg.inv(A)
            # stable remains True if we get here
        except np.linalg.LinAlgError:
            failed_checks += "      Numpy inversion failed.\n"
            stable = False

        # Check if np.linalg.inv and np.linalg.solve give similar results
        try:
            x1 = B@np.linalg.inv(A)
            x2 = np.linalg.solve(A.T, B.T).T
            if not np.allclose(x1, x2, atol=1e-5):
                failed_checks += "      Numpy inv and solve methods do not agree.\n"
                stable = False
        except np.linalg.LinAlgError:
            failed_checks += "      Numpy solve failed.\n"
            stable = False

        # Check a matrix inversion using scipy
        try:
            x3 = B@scipy.linalg.inv(A)
            # stable remains unchanged if we get here
        except scipy.linalg.LinAlgError:
            failed_checks += "      Scipy inversion method failed.\n"
            stable = False

        # Check if scipy.linalg.inv and scipy.linalg.solve give similar results
        try:
            x4 = B@scipy.linalg.inv(A)
            x5 = scipy.linalg.solve(A.T, B.T).T
            if not np.allclose(x4, x5, atol=1e-5):
                failed_checks += "      Scipy and Numpy solve methods do not agree.\n"
                stable = False
        except scipy.linalg.LinAlgError:
            failed_checks += "      Scipy or Numpy solve failed.\n"
            stable = False

        end_time = time.time()
        duration = end_time - start_time

        # Gather system information
        try:
            import psutil
            memory_info = psutil.virtual_memory()._asdict()
            cpu_count = psutil.cpu_count(logical=True)
        except ImportError:
            memory_info = "psutil not installed"
            cpu_count = "unknown"
        
        system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "memory_info": memory_info,
            "cpu_count": cpu_count,
        }

        # Prepare result message
        if stable and cond_number < 1e10:
            status = "PASS"
            message = f"✓ Numerical computations are stable. Condition number: {cond_number:.2e}"
        else:
            status = "FAIL - Stable: "+str(stable)
            message = f"✗ Numerical computations are unstable. Condition number: {cond_number:.2e}"
            # These checks failed
            message += "\n   Failed checks:\n" + failed_checks
            message += "Consider checking your hardware or software environment."

        
        logger.info("Numerical stability check completed.")
        result = {
            "name": "Numerical Stability Check",
            "status": status,
            "message": message,
            "details": {
                "condition_number": cond_number,
                "condition_number_B": cond_number_B,
                "stable": stable,
                "duration_seconds": duration,
                "system_info": system_info,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "version_info": get_version_info(),
            }
        }
        return result

if __name__ == "__main__":
    check = CheckNumStability()
    result = check.run()
    
    # Print results
    print("\n" + "="*60)
    print(result["name"])
    print("="*60)
    print(result["message"])
    print("-"*60)
    print(f"Status: {result['status']}")
    print(f"Duration: {result['details']['duration_seconds']:.4f} seconds")
    print(f"\nCondition numbers:")
    print(f"  Matrix A: {result['details']['condition_number']:.2e}")
    print(f"  Matrix B: {result['details']['condition_number_B']:.2e}")
    print(f"\nSystem info:")
    print(f"  Platform: {result['details']['system_info']['platform']}")
    print(f"  NumPy: {result['details']['system_info']['numpy_version']}")
    print(f"  SciPy: {result['details']['system_info']['scipy_version']}")
    print(f"  CPU count: {result['details']['system_info']['cpu_count']}")
    print("="*60)
