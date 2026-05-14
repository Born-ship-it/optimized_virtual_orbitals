"""
Worker process for VQE optimization - separate module to avoid Numba pickling issues.
Each worker process reconstructs WF objects independently (fresh Numba compilation).
"""

import os
import sys
import numpy as np
import psutil

# Set threading BEFORE any SlowQuant/Numba import
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS


def vqe_optimize_worker(
    wf_type,              # "OVOS", "UHF", or "UMP2"
    num_electrons,        # Total electrons
    active_space,         # Tuple: ((n_alpha, n_beta), n_active_virt)
    mo_coeff_alpha,       # numpy array (n_basis, n_spatial_orbs) - SERIALIZABLE
    mo_coeff_beta,        # numpy array (n_basis, n_spatial_orbs) - SERIALIZABLE
    h_core_array,         # numpy array (n_basis, n_basis) - SERIALIZABLE
    g_eri_array,          # numpy array (n_basis**4,) or reshaped - SERIALIZABLE
    oo,                   # Boolean: orbital optimization on/off
    atol,                 # Float: tolerance
    core_id=None          # Optional: CPU core to pin to
):
    """
    Worker function that runs VQE optimization in a separate process.
    
    All inputs are pickle-able (numpy arrays, scalars, tuples).
    Wave function reconstruction happens INSIDE this worker process,
    avoiding the need to pickle Numba-compiled functions.
    
    Returns:
        tuple: (wf_type, stats_dict)
    """
    
    # Pin this worker to a specific CPU core if requested
    if core_id is not None:
        try:
            p = psutil.Process()
            p.cpu_affinity([core_id])
            print(f"{wf_type} pinned to core {core_id}", flush=True)
        except Exception as e:
            print(f"Warning: Could not pin {wf_type} to core {core_id}: {e}", flush=True)
    
    try:
        # Reconstruct wave function INSIDE worker process
        # (Numba JIT compilation happens here, not in main process)
        mo_coeff_list = [mo_coeff_alpha, mo_coeff_beta]
        
        WF = UnrestrictedWaveFunctionUPS(
            num_electrons,
            active_space,
            mo_coeff_list,
            h_core_array,
            g_eri_array,
            "utups",
            {"n_layers": 1},
            include_active_kappa=False,
        )
        
        # Import run_ucc_and_get_stats from main module
        # (safe because this is a fresh Python process with fresh imports)
        from ovos_vqe_uups import run_ucc_and_get_stats
        
        # Run optimization
        stats = run_ucc_and_get_stats(WF, "BFGS", oo, atol)
        
        print(f"{wf_type} optimization completed successfully", flush=True)
        return wf_type, stats
        
    except Exception as e:
        print(f"ERROR in {wf_type} worker process: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise