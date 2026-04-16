"""
Try OVOS with VQE and see if it can find the ground state energy of a molecule.
"""

# Module metadata - for reload verification
import time
_LOAD_TIME = time.time()
_MODULE_VERSION = "backup_version"

# Set Numba to use TBB threading (thread-safe) BEFORE importing SlowQuant
import os
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Import sys
import sys
from pathlib import Path

# === GET CORRECT IMPORT PATH ===
# Find the project root (parent of both 'ovos' and 'backup' folders)
script_dir = Path(__file__).parent  # backup/
project_root = script_dir.parent     # optimized_virtual_orbitals/

# Add project root to path so we can import 'ovos' package
sys.path.insert(0, str(project_root))

# === CLEAR CACHE ===
modules_to_delete = [mod for mod in list(sys.modules.keys()) if 'ovos' in mod.lower()]
for mod_name in modules_to_delete:
    del sys.modules[mod_name]

import gc
gc.collect()

# Import OVOS
from ovos.ovos import OVOS
import ovos.ovos as ovos_module

print(f"✅ OVOS imported from: {ovos_module.__file__}")
print(f"✅ Load timestamp: {ovos_module._LOAD_TIME}")
print(f"✅ Module version: {ovos_module._MODULE_VERSION}\n")

# Import PySCF
from pyscf import gto, scf

# Import JSON and custom Numpy encoder for saving data
import json
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy scalars, arrays, and None."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Other imports
import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_OVOS_convergence_from_data(molecule, basis, methods=None):
    """
    Plot OVOS convergence results from saved JSON data files.
    
    Reads JSON files saved by ovos_object() and plots convergence data
    using the same structure as ovos_plot.py's plot_OVOS_convergence().
    
    Parameters
    ----------
    molecule : str
        Molecule name (e.g., "Li2", "CO", "H2O")
    basis : str
        Basis set (e.g., "6-31G", "cc-pVDZ")
    methods : list, optional
        List of methods to plot (e.g., ["RHF", "prev", "random"])
        If None, plots all available methods
    """
    
    # Get molecule name for file paths
    molecule_name = molecule.split()[0]
    if molecule_name == "Li" and "Li" in molecule.split(";")[1]:
        molecule_name = "Li2"
    elif molecule_name == "C" and "O" in molecule.split(";")[1]:
        molecule_name = "CO"
    elif molecule_name == "O" and "H" in molecule.split(";")[1]:
        molecule_name = "H2O"
    elif molecule_name == "H" and "F" in molecule.split(";")[1]:
        molecule_name = "HF"
    elif molecule_name == "N" and "H" in molecule.split(";")[1]:
        molecule_name = "NH3"
    
    # Data directory
    data_dir = project_root / "backup" / "data" / molecule_name / basis / "OVOS"
    
    if methods is None:
        methods = ["RHF", "prev", "random"]
    
    # Try to load data for each method
    conv_data = {}
    for method in methods:
        json_file = data_dir / f"lst_MP2_OVOS_virt_orbs_{method}.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                conv_data[method] = json.load(f)
            print(f"✅ Loaded {method} data from {json_file}")
        else:
            print(f"⚠️  {method} data not found at {json_file}")
    
    if not conv_data:
        print("❌ No data files found!")
        return
    
    # Extract molecule parameters
    num_electrons = None
    full_space_size = None
    E_corr_MP2 = None
    
    for method, data in conv_data.items():
        if data[1]:  # num_opt_virtual_orbs list
            # Create temporary molecule to get parameters
            mol_temp = gto.Mole()
            mol_temp.atom = molecule
            mol_temp.basis = basis
            mol_temp.unit = 'Angstrom'
            mol_temp.spin = 0
            mol_temp.charge = 0
            mol_temp.symmetry = False
            mol_temp.verbose = 0
            mol_temp.build()
            
            mf_temp = scf.RHF(mol_temp)
            mf_temp.verbose = 0
            mf_temp.kernel()
            
            num_electrons = mol_temp.nelectron
            full_space_size = mf_temp.mo_coeff.shape[1]
            
            MP2_temp = mf_temp.MP2().run()
            E_corr_MP2 = MP2_temp.e_corr
            break
    
    if num_electrons is None:
        print("❌ Could not determine molecule parameters!")
        return

    ### === REFERENCE DATA === ###
    ref_data_file = data_dir / "molecule_data.json"
    E_corr_OOMP2 = None
    E_corr_FCI = None

    if ref_data_file.exists():
        with open(ref_data_file, "r") as f:
            ref_data = json.load(f)
        
        E_corr_OOMP2 = ref_data.get("OOMP2_e_corr", None)
        E_corr_FCI = ref_data.get("FCI_e_corr", None)
        
        print(f"✅ Loaded reference data from {ref_data_file}")
    else:
        print(f"⚠️  Reference data file not found at {ref_data_file}")
    
    print(f"\n#### Reference MP2 Correlation Energy for Full Space: {E_corr_MP2:.6f} Hartree ####")
    print(f"Number of electrons: {num_electrons}")
    print(f"Full space size (MOs): {full_space_size}")
    print(f"Number of occupied MOs: {num_electrons//2}")
    print(f"Number of virtual MOs: {full_space_size - num_electrons//2}\n")
    
    # Prepare data structures for plotting
    conv_virtual_orbs_data = {}
    
    for method, data in conv_data.items():
        conv_virtual_orbs_data[method] = {
            'num_virtual_orbitals': data[1],           # [1]
            'MP2_final_energies': [e[-1] for e in data[0]],  # Last energy from each [0]
            'MP2_energies_per_iteration': data[0],     # [0]
            'iterations_to_converge': data[2],         # [2]
            'alpha_beta_check': data[3],               # [3]
        }
    
    # Initialize figure
    fig_size_length = 12
    fig_vo, ax_vo = plt.subplots(figsize=(fig_size_length, 7))
    
    # Title
    suptitle_txt = f'OVOS Convergence: {molecule_name}/{basis}, Full Space ({num_electrons}e,{full_space_size}o)'
    fig_vo.suptitle(suptitle_txt, fontsize=16)
    
    # Color map for different virtual orbitals
    active_space_size = full_space_size - num_electrons // 2 + 1
    colors = plt.cm.hsv(np.linspace(0, 1, active_space_size))
    
    # Define markers and x-offsets for each method
    marker_style = {
        'RHF': ('D', -0.5),
        'prev': ('s', 0.0),
        'random': ('^', 0.5)
    }
    
    # Plot converged MP2 correlation energy vs number of virtual orbitals
    for method, data_dict in conv_virtual_orbs_data.items():
        if method not in marker_style:
            continue
        
        marker, x_offset = marker_style[method]
        
        for i, num_virt_orbs in enumerate(data_dict['num_virtual_orbitals']):
            MP2_vorb = data_dict['MP2_final_energies'][i]
            if MP2_vorb < 0.001:  # Only plot negative energies
                ax_vo.scatter(num_virt_orbs + x_offset, MP2_vorb, marker=marker, 
                            alpha=1.0, color=colors[i] if i < len(colors) else colors[-1])
    
    # Plot iteration counts as lines connecting initial to final energy
    for method, data_dict in conv_virtual_orbs_data.items():
        if method not in marker_style:
            continue
        
        marker, x_offset = marker_style[method]
        
        for i, num_virt_orbs in enumerate(data_dict['num_virtual_orbitals']):
            MP2_vorb = data_dict['MP2_final_energies'][i]
            if MP2_vorb < 0.001:
                iter_max = data_dict['iterations_to_converge'][i][-1] if isinstance(data_dict['iterations_to_converge'][i], list) else data_dict['iterations_to_converge'][i]
                MP2_iter_vorb = data_dict['MP2_energies_per_iteration'][i][0]
                
                ax_vo.scatter(num_virt_orbs + x_offset, MP2_iter_vorb, marker=marker, 
                            alpha=0.25, color=colors[i] if i < len(colors) else colors[-1])
                ax_vo.plot([num_virt_orbs + x_offset, num_virt_orbs + x_offset], 
                          [MP2_iter_vorb, MP2_vorb], 
                          color=colors[i] if i < len(colors) else colors[-1], alpha=0.3)
                ax_vo.text(num_virt_orbs + x_offset + 0.075, (MP2_iter_vorb + MP2_vorb)/2, 
                          str(iter_max), fontsize=8, alpha=0.75)
    
    # Legend for markers
    for method in methods:
        if method in marker_style and method in conv_virtual_orbs_data:
            marker, x_offset = marker_style[method]
            ax_vo.scatter([], [], marker=marker, color='black', label=f'Start Guess: {method}')
    
    # Make a line connecting best of the 3 methods for each virtual orbital count
        # This will show the best convergence path across different initializations
    coordinates_for_lines = []
    for i in range(len(conv_virtual_orbs_data['RHF']['num_virtual_orbitals'])):
        best_energy = None
        best_x = None
        
        for method in methods:
            if method in conv_virtual_orbs_data:
                data_dict = conv_virtual_orbs_data[method]
                num_virt_orbs = data_dict['num_virtual_orbitals'][i]
                MP2_vorb = data_dict['MP2_final_energies'][i]
                
                if MP2_vorb < 0.001 and (best_energy is None or MP2_vorb < best_energy):
                    best_energy = MP2_vorb
                    marker, x_offset = marker_style[method]
                    best_x = num_virt_orbs + x_offset

        if best_energy is not None and best_x is not None:
            coordinates_for_lines.append((best_x, best_energy))
    if coordinates_for_lines:
        x_coords, y_coords = zip(*coordinates_for_lines)
        ax_vo.plot(x_coords, y_coords, color='black', linestyle='-', alpha=0.25, label='Best Convergence Path')

    # Reference lines
        # MP2 correlation energy for full space
    ax_vo.axhline(E_corr_MP2, color='red', linestyle='--', 
                 label='Full Space MP2', linewidth=2, alpha=0.75)
        # OO-MP2 correlation energy for full space (if available)
    if E_corr_OOMP2 is not None:
        ax_vo.axhline(E_corr_OOMP2, color='blue', linestyle='--', 
                 label='Full Space OO-MP2', linewidth=2, alpha=0.75)
        # FCI correlation energy for full space (if available)
    if E_corr_FCI is not None:
        ax_vo.axhline(E_corr_FCI, color='green', linestyle='--', 
                 label='Full Space FCI', linewidth=2, alpha=0.75)


    # Set axis limits
    lower_bound = []
    for method, data_dict in conv_virtual_orbs_data.items():
        lower_bound.extend(data_dict['MP2_final_energies'])
    lower_bound.append(E_corr_MP2)
    lower_bound.append(E_corr_OOMP2 if E_corr_OOMP2 is not None else E_corr_MP2)
    lower_bound.append(E_corr_FCI if E_corr_FCI is not None else E_corr_MP2)
    
    min_lower_bound = min(lower_bound)
    ax_vo.set_ylim(min_lower_bound + min_lower_bound*0.01, 0.0)
    
    # Set x-axis
    ax_vo.set_xlim(1, active_space_size*2 - 1)
    x_ticks = list(range(2, active_space_size*2, 2))
    x_labels = [str(x//2) for x in x_ticks]
    ax_vo.set_xticks(x_ticks)
    ax_vo.set_xticklabels(x_labels)
    
    ax_vo.set_xlabel('Number of Active Unoccupied Orbitals')
    ax_vo.set_ylabel('Correlation Energy [Hartree]')
    
    # Grid with colored intervals
    for i in range(1, active_space_size + 1):
        x_start = i*2 - 1
        x_end = i*2 + 1
        if i % 2 == 0:
            ax_vo.axvspan(x_start, x_end, color='lightgray', alpha=0.2)
    
    ax_vo.grid(which='major', axis='y', linestyle='--', alpha=0.7)
    ax_vo.minorticks_on()
    ax_vo.grid(which='minor', axis='y', linestyle=':', alpha=0.5)
    
    ax_vo.legend(loc='upper right')
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / "backup" / "images" / molecule_name / basis
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"ovos_convergence_{molecule_name}_{basis}.png"
    plt.savefig(output_file, dpi=150)
    print(f"✅ Plot saved to: {output_file}\n")
    plt.close()





# Setup data collection for OVOS with generic molecule and basis set
def mol_data(molecule, basis, method="RHF", num_random_attempts=10):
    """
    Retrieve molecule data and optionally generate random orbital initializations.
    
    Parameters
    ----------
    molecule : str
        Molecular geometry string (e.g., "Li .0 .0 .0; Li .0 .0 2.673")
    basis : str
        Basis set name (e.g., "6-31G")
    method : str
        Initialization method:
        - "RHF": Use RHF orbitals
        - "random": Generate random unitary rotations of virtual orbitals
    num_random_attempts : int
        Only used if method="random"; number of random rotations to generate
    
    Returns
    -------
    If method == "RHF":
        (mol, mf, Fao, mo_coeffs, num_electrons, num_orbitals, E_ref, E_corr_MP2)
    
    If method == "random":
        (mol, mf, Fao, mo_coeffs_list, num_electrons, num_orbitals, E_ref, E_corr_MP2)
        where mo_coeffs_list is a list of num_random_attempts random initializations
    """
    
    # Create molecule and perform SCF calculation
    mol = gto.Mole()
    mol.atom = molecule
    mol.basis = basis
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # RHF reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    E_ref = mf.e_tot

    # Fock matrix in AO basis
    Fao = [mf.get_fock(), mf.get_fock()]
    
    # Number of electrons and orbitals
    num_electrons = mol.nelectron
    num_orbitals = mf.mo_coeff.shape[1]
    num_occupied = num_electrons // 2
    num_virtual = num_orbitals - num_occupied

    # Reference MP2 correlation energy for full space
    MP2 = mf.MP2().run()
    E_corr_MP2 = MP2.e_corr

    # ===== STANDARD METHOD: Return RHF orbitals =====
    if method == "RHF":
        mo_coeffs = [mf.mo_coeff, mf.mo_coeff]
        return mol, mf, Fao, mo_coeffs, num_electrons, num_orbitals, E_ref, E_corr_MP2

    # ===== RANDOM METHOD: Generate random unitary rotations =====
    elif method == "random":
        mo_coeffs_rhf = [mf.mo_coeff, mf.mo_coeff]
        mo_coeffs_list = []
        
        for attempt in range(num_random_attempts):
            # Generate random unitary matrix for virtual orbitals
            rand_matrix = np.random.rand(num_virtual, num_virtual)
            Q, R = np.linalg.qr(rand_matrix)  # QR decomposition to get unitary matrix
            
            # Apply random unitary rotation to virtual orbitals
            mo_coeffs_trial = [np.copy(mo_coeffs_rhf[0]), np.copy(mo_coeffs_rhf[1])]
            
            for spin in [0, 1]:
                C_occ = mo_coeffs_rhf[spin][:, :num_occupied]
                C_virt = mo_coeffs_rhf[spin][:, num_occupied:]
                C_virt_rot = C_virt @ Q
                mo_coeffs_trial[spin] = np.hstack((C_occ, C_virt_rot))
            
            mo_coeffs_list.append(np.array(mo_coeffs_trial))
        
        return mol, mf, Fao, mo_coeffs_list, num_electrons, num_orbitals, E_ref, E_corr_MP2
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'RHF' or 'random'.")
def run_ovos_for_virtual_orbs(molecule, basis, method, num_opt_virtual_orbs):
    """
    Run OVOS for a single virtual orbital count.
    This function is designed to be called in parallel for different num_opt_virtual_orbs values.
    
    Parameters
    ----------
    molecule : str
        Molecular geometry
    basis : str
        Basis set
    method : str
        Initialization method ("RHF", "prev", or "random")
    num_opt_virtual_orbs : int
        Number of optimized virtual orbitals for this task
    
    Returns
    -------
    dict
        Dictionary containing results for this virtual orbital count
    """
    # Get molecule data once per worker
    mol, mf, Fao, mo_coeffs, num_electrons, num_orbitals, E_ref, E_corr_MP2 = mol_data(
        molecule, basis, method="RHF"
    )
    
    # Create OVOS object and run
    ovos = OVOS(
        mol=mol, scf=mf, Fao=Fao,
        num_opt_virtual_orbs=2*num_opt_virtual_orbs,
        mo_coeff=mo_coeffs,
        init_orbs="RHF",
        verbose=1, max_iter=1000,
        conv_energy=1e-8, conv_grad=1e-6,
        keep_track_max=50
    )

    E_corr, E_corr_hist, E_corr_iter, E_corr_mo, _, stop_reason = ovos.run(
        mo_coeffs, fock_spin=None
    )

    # Check alpha/beta difference
    diff_alpha_beta = np.max(np.abs(E_corr_mo[0] - E_corr_mo[1]))
    alpha_beta_check = "True" if diff_alpha_beta > 1e-4 else "False"

    # Return results as dictionary
    return {
        'num_opt_virtual_orbs': 2*num_opt_virtual_orbs,
        'E_corr_hist': E_corr_hist,
        'E_corr_iter': E_corr_iter,
        'E_corr_mo': E_corr_mo,
        'alpha_beta_check': alpha_beta_check,
        'stop_reason': stop_reason,
        'final_energy': E_corr,
        'num_orbitals': num_orbitals,
        'num_electrons': num_electrons,
        'E_corr_MP2': E_corr_MP2
    }

def run_ovos_random_attempt(molecule, basis, num_opt_virtual_orbs, mo_coeffs_trial, 
                            eri_4fold_ao, S, hcore_ao, attempt_idx, num_iter_max):
    """
    Run a single OVOS random attempt in parallel.
    
    Parameters
    ----------
    molecule : str
        Molecular geometry
    basis : str
        Basis set
    num_opt_virtual_orbs : int
        Number of optimized virtual orbitals (in spatial orbitals)
    mo_coeffs_trial : np.ndarray
        Trial MO coefficients for this attempt
    eri_4fold_ao : np.ndarray
        Precomputed AO integrals
    S : np.ndarray
        Overlap matrix
    hcore_ao : np.ndarray
        Core Hamiltonian
    attempt_idx : int
        Attempt number
    
    Returns
    -------
    dict
        Results dictionary for this attempt
    """
    # Reimport for worker process
    from ovos.ovos import OVOS
    from pyscf import gto, scf
    
    # Create molecule
    mol = gto.Mole()
    mol.atom = molecule
    mol.basis = basis
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # RHF reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    
    # Fock matrix in AO basis
    Fao = [mf.get_fock(), mf.get_fock()]
    
    # Create OVOS object and run
    ovos = OVOS(
        mol=mol, scf=mf, Fao=Fao,
        num_opt_virtual_orbs=2*num_opt_virtual_orbs,
        mo_coeff=mo_coeffs_trial,
        init_orbs="RHF",
        verbose=0, max_iter=1000,
        conv_energy=1e-8, conv_grad=1e-6,
        keep_track_max=50
    )
    
    # Pass precomputed integrals
    ovos.eri_4fold_ao = eri_4fold_ao
    ovos.S = S
    ovos.hcore_ao = hcore_ao

    # Run OVOS
    E_corr, E_corr_hist, E_corr_iter, E_corr_mo, _, stop_reason = ovos.run(
        mo_coeffs_trial, fock_spin=None
    )

    return {
        'attempt_idx': attempt_idx,
        'final_energy': E_corr,
        'E_corr_hist': E_corr_hist,
        'E_corr_iter': E_corr_iter,
        'E_corr_mo': E_corr_mo,
        'stop_reason': stop_reason
    }


def ovos_object(molecule, basis, method="RHF"):
    """
    Run OVOS for a given molecule and basis set.
    
    Parameters
    ----------
    molecule : str
        Molecular geometry
    basis : str
        Basis set
    method : str
        "RHF", "prev", or "random"
    """
    
    print(f"=== OVOS VQE for {molecule} with {basis} basis (method: {method}) ===\n")
    
    molecule_name = molecule.split()[0]
    if molecule_name == "Li" and "Li" in molecule.split(";")[1]:
        molecule_name = "Li2"
    if molecule_name == "C" and "O" in molecule.split(";")[1]:
        molecule_name = "CO"
    if molecule_name == "O" and "H" in molecule.split(";")[1]:
        molecule_name = "H2O"
    if molecule_name == "H" and "F" in molecule.split(";")[1]:
        molecule_name = "HF"
    if molecule_name == "N" and "H" in molecule.split(";")[1]:
        molecule_name = "NH3"

    # Check if the files exist before running OVOS
        # Path
    output_dir = project_root / "backup" / "data" / molecule_name / basis / "OVOS"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"lst_MP2_OVOS_virt_orbs_{method}.json"
    # if output_file.exists() and method != "random":
    #     print(f"⚠️  Output file already exists at {output_file}, skipping OVOS run to avoid overwriting.\n")
    #     return

    # Get molecule data to determine number of virtual orbitals
    mol, mf, Fao, mo_coeffs, num_electrons, num_orbitals, E_ref, E_corr_MP2 = mol_data(
        molecule, basis, method="RHF"
    )
    
    max_num_opt_virt = num_orbitals - num_electrons // 2
    
    # Initialize data collection structure
    lst_E_corr_virt_orbs = [[], [], [], [], [], []]

    # ===== PARALLEL EXECUTION for RHF method =====
    if method == "RHF":
        print(f"Running OVOS in parallel mode for virtual orbital iterations...\n")
        
        num_cores = os.cpu_count()
        num_workers = min(num_cores, max_num_opt_virt)
        
        print(f"System: {num_cores} cores available")
        print(f"Starting {num_workers} parallel workers for {max_num_opt_virt} virtual orbital counts\n")
        
        from concurrent.futures import ProcessPoolExecutor
        
        virt_orb_counts = list(range(1, max_num_opt_virt + 1))
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(run_ovos_for_virtual_orbs, molecule, basis, method, num_virt): num_virt
                for num_virt in virt_orb_counts
            }
            
            results_dict = {}
            for future in futures:
                num_virt = futures[future]
                try:
                    result = future.result()
                    results_dict[num_virt] = result
                except Exception as e:
                    print(f"Error processing {num_virt} virtual orbitals: {e}")
        
        for num_virt in sorted(results_dict.keys()):
            result = results_dict[num_virt]
            lst_E_corr_virt_orbs[0].append(result['E_corr_hist'])
            lst_E_corr_virt_orbs[1].append(result['num_opt_virtual_orbs'])
            lst_E_corr_virt_orbs[2].append(result['E_corr_iter'])
            lst_E_corr_virt_orbs[3].append(result['alpha_beta_check'])
            lst_E_corr_virt_orbs[4].append(result['E_corr_mo'])
            lst_E_corr_virt_orbs[5].append(result['stop_reason'])
            
            print(f"[{num_virt:>2}/{max_num_opt_virt}] "
                  f"Corr. energy: {result['final_energy']:.4f} Ha, "
                  f"ratio to MP2: {result['final_energy']/result['E_corr_MP2']:.4f}, "
                  f"iter: {len(result['E_corr_hist'])}")

    # ===== SEQUENTIAL/PARALLEL EXECUTION for OTHER METHODS =====
    else:
        # Get molecule data with correct initialization
        if method == "random":
            num_random_attempts = 1000
            mol, mf, Fao, mo_coeffs_list, num_electrons, num_orbitals, E_ref, E_corr_MP2 = mol_data(
                molecule, basis, method="random", num_random_attempts=num_random_attempts
            )
        else:
            mol, mf, Fao, mo_coeffs, num_electrons, num_orbitals, E_ref, E_corr_MP2 = mol_data(
                molecule, basis, method="RHF"
            )

        # Precompute integrals for random method
        if method == "random":
            eri_4fold_ao = mol.intor('int2e_sph', aosym=1)
            S = mol.intor('int1e_ovlp')
            hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

        # Iterate over all virtual orbitals
        num_opt_virtual_orbs = 0
        while num_opt_virtual_orbs < (num_orbitals - num_electrons // 2):
            num_opt_virtual_orbs += 1

            # ===== RANDOM METHOD WITH PARALLEL ATTEMPTS =====
            if method == "random":
                print(f"[{num_opt_virtual_orbs:>2}/{num_orbitals-num_electrons//2}] "
                      f"Running {num_random_attempts} attempts in parallel...")
                
                # Load RHF results for this virtual orbital count to use as stopping criterion
                output_dir = project_root / "backup" / "data" / molecule_name / basis / "OVOS"
                rhf_data_file = output_dir / f"lst_MP2_OVOS_virt_orbs_RHF.json"
                rhf_threshold = None
                rhf_iter = None
                num_iter_max = 1000
                
                if rhf_data_file.exists():
                    with open(rhf_data_file, "r") as f:
                        rhf_data = json.load(f)
                    # Find corresponding virtual orbital count in RHF data
                    try:
                        rhf_idx = rhf_data[1].index(2*num_opt_virtual_orbs)
                        rhf_threshold = rhf_data[0][rhf_idx][-1]    # Final RHF energy for this orbital count
                        rhf_iter = rhf_data[2][rhf_idx][-1]         # Iterations to converge for this orbital count
                        num_iter_max = rhf_iter + 50                # Set max iterations for random attempts based on RHF convergence
                        print(f"  RHF reference energy for {2*num_opt_virtual_orbs} spin-orbs: {rhf_threshold:.6f} Ha @ iter {rhf_iter}")
                    except (ValueError, IndexError):
                        print(f"  ⚠️  RHF data not found for {2*num_opt_virtual_orbs} spin-orbs, no early stopping")

                from concurrent.futures import ProcessPoolExecutor
                import threading

                num_cores = os.cpu_count()
                num_workers = min(10, num_random_attempts)
                
                # Thread-safe flag for early stopping
                stop_attempts = threading.Event()
    
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(
                            run_ovos_random_attempt,
                            molecule, basis, num_opt_virtual_orbs,
                            mo_coeffs_trial, eri_4fold_ao, S, hcore_ao,
                            attempt_idx, num_iter_max
                        ): attempt_idx
                        for attempt_idx, mo_coeffs_trial in enumerate(mo_coeffs_list, start=1)
                    }
                    
                    # Collect results as they complete
                    attempt_results = {}
                    best_attempt_idx = None
                    best_energy = None
                    completed_count = 0

                    for future in futures:
                        if stop_attempts.is_set():
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break
                        
                        attempt_idx = futures[future]
                        try:
                            result = future.result()
                            attempt_results[attempt_idx] = result
                            completed_count += 1
                            final_energy = result['final_energy']
                            iter_count = result['E_corr_iter'][-1]
                            
                            # Update best result
                            if best_energy is None or final_energy <= best_energy:
                                if best_energy is not None and final_energy == best_energy:
                                    if iter_count < best_iter:
                                        best_energy = final_energy
                                        best_attempt_idx = attempt_idx
                                        best_iter = iter_count
                                if best_energy is None or final_energy < best_energy:
                                    best_energy = final_energy
                                    best_attempt_idx = attempt_idx
                                    best_iter = iter_count

                                keep_count = 0  # Reset keep count if we find a new best result
                            elif completed_count >= 100:
                                keep_count += 1  # Increment keep count if we do not find a better result
                            
                            final_energy_diff = final_energy - rhf_threshold if rhf_threshold is not None else None
                            energy_diff_str = "True" if final_energy_diff is not None and final_energy_diff < 0 else "False"

                            print(f"  Attempt {attempt_idx:>4}/{num_random_attempts} completed: "
                                f"< RHF = {energy_diff_str:<6} @ {iter_count:>5} (best: {best_energy:.6f} Ha @ {best_iter}, {keep_count}/100)")
                            
                            # Early stopping criterion: beat RHF threshold by margin
                            if rhf_threshold is not None:
                                energy_gap = best_energy - rhf_threshold
                                margin = 0.0001  # Stop if we're within 0.0001 Ha of RHF
                                min_attempts = 100  # Require at least 10 attempts before stopping
                                iter_gap = best_iter - rhf_iter

                                if (energy_gap < margin and iter_gap <= 0 and keep_count >= min_attempts) or keep_count >= min_attempts:  
                                    print(f"\n  ✅ Early stopping: Found result {abs(energy_gap):.6f} Ha better than RHF!")
                                    print(f"     Completed {completed_count}/{num_random_attempts} attempts\n")
                                    stop_attempts.set()
                            
                        except Exception as e:
                            print(f"  ⚠️  Attempt {attempt_idx} failed: {e}")
                
                # Use best attempt result
                if best_attempt_idx is not None:
                    best_result = attempt_results[best_attempt_idx]

                    E_corr_hist = best_result['E_corr_hist']
                    E_corr_iter = best_result['E_corr_iter']
                    E_corr_mo = best_result['E_corr_mo']
                    stop_reason = best_result['stop_reason']
                    
                    print(f"  ✅ Best: Corr. energy = {best_energy:.6f} Ha, "
                        f"ratio to MP2 = {best_energy/E_corr_MP2:.4f}, "
                        f"iter = {best_iter}, attempt = {best_attempt_idx}/{num_random_attempts}\n")
                else:
                    print(f"  ❌ No successful attempts for {2*num_opt_virtual_orbs} spin-orbs, skipping...\n")
                    continue

            # ===== PREV METHOD (sequential with warm-start) =====
            elif method == "prev":
                ovos = OVOS(
                    mol=mol, scf=mf, Fao=Fao,
                    num_opt_virtual_orbs=2*num_opt_virtual_orbs,
                    mo_coeff=mo_coeffs,
                    init_orbs="RHF",
                    verbose=0, max_iter=1000,
                    conv_energy=1e-8, conv_grad=1e-6,
                    keep_track_max=50
                )

                E_corr, E_corr_hist, E_corr_iter, E_corr_mo, _, stop_reason = ovos.run(
                    mo_coeffs, fock_spin=None
                )

                print(f"[{num_opt_virtual_orbs:>2}/{num_orbitals-num_electrons//2}] "
                      f"Corr. energy: {E_corr:.4f} Ha, "
                      f"ratio to MP2: {E_corr/E_corr_MP2:.4f}, "
                      f"iter: {len(E_corr_hist)}")

                # Update mo_coeffs for next iteration
                mo_coeffs = E_corr_mo

            # Store results
            diff_alpha_beta = np.max(np.abs(E_corr_mo[0] - E_corr_mo[1]))
            alpha_beta_check = "True" if diff_alpha_beta > 1e-4 else "False"

            lst_E_corr_virt_orbs[0].append(E_corr_hist)
            lst_E_corr_virt_orbs[1].append(2*num_opt_virtual_orbs)
            lst_E_corr_virt_orbs[2].append(E_corr_iter)
            lst_E_corr_virt_orbs[3].append(alpha_beta_check)
            lst_E_corr_virt_orbs[4].append(E_corr_mo)
            lst_E_corr_virt_orbs[5].append(stop_reason)

    # Save to JSON    
    with open(output_file, "w") as f:
        json.dump(lst_E_corr_virt_orbs, f, indent=2, cls=NumpyEncoder)

    print(f"✅ Data saved to: {output_file}\n")


def save_molecule_reference_data(molecule, basis):
    """
    Save reference molecular data (MP2, OOMP2, FCI energies) to JSON.
    
    Matches the structure from ovos_working.py lines 2704-2823.
    Returns a dictionary with molecular parameters and reference energies.
    
    Parameters
    ----------
    molecule : str
        Molecule name (e.g., "Li2", "CO", "H2O")
    basis : str
        Basis set (e.g., "6-31G", "cc-pVDZ")
    
    Returns
    -------
    dict
        Dictionary containing:
        - num_electrons: total electron count
        - full_space_size: number of spatial orbitals
        - active_space_size: virtual orbitals + 1
        - MP2_e_corr: MP2 correlation energy (RHF)
        - OOMP2_e_corr: OO-MP2 correlation energy
        - FCI_e_corr: FCI correlation energy (if computed)
    """
    
    # Reconstruct molecule string for file paths
        # where molecule is the geometry string used in the original calculations (e.g., "Li .0 .0 .0; Li .0 .0 2.673")
    if molecule == "Li .0 .0 .0; Li .0 .0 2.673":
        molecule_str = "Li2"
    elif molecule == "C .0 .0 .0; O .0 .0 1.128":
        molecule_str = "CO"
    elif molecule == "O .0 .0 .0; H .0 .0 0.958; H .0 .0 -0.958":
        molecule_str = "H2O"
    elif molecule == "H .0 .0 .0; F .0 .0 0.917":
        molecule_str = "HF"
    elif molecule == "N .0 .0 .0; H .0 .0 .0; H .0 .0 0.94; H .0 .0 -0.94":
        molecule_str = "NH3"
    
    geom = molecule

    molecule = molecule_str  # Use simplified name for file paths
    
    # Create output directory path
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "backup" / "data" / molecule / basis / "OVOS"
    output_file = output_dir / "molecule_data.json"

    # ===== CHECK IF FILE ALREADY EXISTS =====
    if output_file.exists():
        print(f"✅ File already exists: {output_file}")
        # Load and return existing data
        with open(output_file, "r") as f:
            data = json.load(f)
        return data
    
    print(f"\n#### Reference Molecular Data: {molecule}/{basis} ####")
    
    # Create molecule
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = basis
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()
    
    # RHF calculation
    rhf = scf.RHF(mol)
    rhf.verbose = 0
    rhf.kernel()
    
    # Molecular parameters
    num_electrons = mol.nelectron
    full_space_size = rhf.mo_coeff.shape[1]
    active_space_size = full_space_size - num_electrons // 2 + 1
    
    print(f"Number of electrons: {num_electrons}")
    print(f"Full space size (MOs): {full_space_size}")
    print(f"Number of occupied MOs: {num_electrons // 2}")
    print(f"Number of virtual MOs: {full_space_size - num_electrons // 2}")
    
    # === MP2 Correlation Energy ===
    MP2 = rhf.MP2().run()
    MP2_e_corr = MP2.e_corr
    print(f"MP2 correlation energy: {MP2_e_corr:.10f} Hartree")
    
    # === OO-MP2 Correlation Energy ===
    # Restricted OOMP2 via CASSCF with all orbitals in active space
    print("Computing OO-MP2 (Orbital-Optimized MP2)...")
    
    class OOMP2(object):
        """Restricted OOMP2 solver for CASSCF."""
        def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
            if isinstance(nelec, (int, np.integer)):
                na = nelec // 2
                nb = nelec // 2
            else:
                na, nb = int(nelec[0]), int(nelec[1])
            
            n_elec = na + nb
            
            fakemol = gto.M(verbose=0)
            fakemol.nelectron = n_elec
            
            fake_hf = scf.RHF(fakemol)
            fake_hf._eri = h2
            fake_hf.get_hcore = lambda *args: h1
            fake_hf.get_ovlp = lambda *args: np.eye(norb)
            
            fake_hf.mo_coeff = np.eye(norb)
            fake_hf.mo_occ = np.zeros(norb)
            fake_hf.mo_occ[:n_elec // 2] = 2
            
            self.mp2 = pyscf.mp.MP2(fake_hf)
            self.mp2.verbose = 0
            
            e_corr, t2 = self.mp2.kernel()
            e_tot = self.mp2.e_tot + ecore
            return e_tot, t2
        
        def make_rdm12(self, t2, norb, nelec):
            dm1 = self.mp2.make_rdm1(t2)
            dm2 = self.mp2.make_rdm2(t2)
            if isinstance(dm2, (tuple, list)):
                dm2 = sum(dm2)
            return dm1, dm2
    
    # Run CASSCF with full space as active space
    import pyscf
    mc = pyscf.mcscf.CASSCF(rhf, full_space_size, num_electrons)
    mc.fcisolver = OOMP2()
    mc.internal_rotation = True
    mc.verbose = 0
    mc.conv_tol = 1e-6
    
    try:
        ooMP2_e_tot = mc.kernel()[0]
        OOMP2_e_corr = ooMP2_e_tot - rhf.e_tot
        print(f"OO-MP2 correlation energy: {OOMP2_e_corr:.10f} Hartree")
    except Exception as e:
        print(f"⚠️  OO-MP2 computation failed: {e}")
        OOMP2_e_corr = None
    
    # === FCI Correlation Energy (only for small systems) ===
    FCI_e_corr = None
    if basis == "6-31G" and molecule in ["HF", "H2O", "Li2"]:
        print("Computing FCI (Full Configuration Interaction)...")
        try:
            cisolver = pyscf.fci.FCI(mol, rhf.mo_coeff)
            fci_e_tot = cisolver.kernel()[0]
            FCI_e_corr = fci_e_tot - rhf.e_tot
            print(f"FCI correlation energy: {FCI_e_corr:.10f} Hartree")
        except Exception as e:
            print(f"⚠️  FCI computation failed: {e}")
            FCI_e_corr = None
    else:
        print("FCI skipped (only computed for 6-31G with HF, H2O)")
    
    # === Save to JSON ===
    data = {
        "num_electrons": int(num_electrons),
        "full_space_size": int(full_space_size),
        "active_space_size": int(active_space_size),
        "MP2_e_corr": float(MP2_e_corr),
        "OOMP2_e_corr": float(OOMP2_e_corr) if OOMP2_e_corr is not None else None,
        "FCI_e_corr": float(FCI_e_corr) if FCI_e_corr is not None else None
    }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    print(f"✅ Reference data saved to: {output_file}\n")
    
    return data



if __name__ == "__main__":
    molecules = [
        "Li .0 .0 .0; Li .0 .0 2.673",  # Li2 molecule
        "H .0 .0 .0; F .0 .0 0.917",  # HF molecule
        "C .0 .0 .0; O .0 .0 1.128",    # CO molecule
        "O .0 .0  0.1173; H .0 0.7572 -0.4692; H .0 -0.7572 -0.4692;",  # H2O equilibrium geometry
        "N .0 .0 .0; H .0 .0 1.012; H .0 0.926 -0.239; H .0 -0.926 -0.239"  # NH3 molecule
    ]
    basis_sets = [
        "6-31G",     # Done: Li2, HF 
        "cc-pVDZ"
    ]
    methods = [
        "RHF",       # Always start from RHF orbitals
        "prev",      # Use previous iteration's MO coefficients as starting point
        "random",    # Start from random orbitals
    ]

    # ovos_object(molecules[0], basis_sets[0], methods[0])  # Run for Li2 with 6-31G using "prev" method

    # Debug run
        # A single run for number of optimized virtual orbitals = 5 for Li2 with 6-31G using "RHF" method
        # This will test the OVOS workflow and data saving without running the full set of calculations
    # run_ovos_for_virtual_orbs(molecules[2], basis_sets[0], method="RHF", num_opt_virtual_orbs=5)

    for molecule in molecules[3:4]:     
        for basis in basis_sets[0:1]:   # Only run for 6-31G basis to save time
            for method in methods:
                ovos_object(molecule, basis, method)

            try:
                save_molecule_reference_data(molecule, basis)
            except Exception as e:
                print(f"❌ Failed for {molecule}/{basis}: {e}\n")

            # After running ovos_object(), plot the results
            plot_OVOS_convergence_from_data(molecule, basis, methods=["RHF", "prev", "random"])


