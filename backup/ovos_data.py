"""
Try OVOS with VQE and see if it can find the ground state energy of a molecule.


Notes for future:
- check if resulting method deviates in molecular orbitals coefficients between each other and how far from reference MP2 and OO-MP2
    - Find a way to visualize MO coefficients, and their differences, in a way that is not just a huge table of numbers (e.g., heatmap, or difference plot)
    - Use this to comment on MO coefficients used as initial state for VQE -> is there a noticeable difference from "RHF" to not use it as a starting point for VQE? 
    

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
    molecule_name = get_molecule_name(molecule)

    print(f"\nPlotting OVOS convergence from data for {molecule_name}/{basis} for methods: {methods}\n")
    
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


# Plot convergence behaviour for each number of optimized virtual orbitals using saved data
    # Make a plot for each optimized virtual orbital count, showing the convergence history for each method (RHF, prev, random) on the same plot for direct comparison
def plot_OVOS_convergence_histories(molecule, basis, methods=None):
    """
    Plot OVOS convergence histories for each number of optimized virtual orbitals.
    """

    # Get data
    molecule_name = get_molecule_name(molecule)

    print(f"\nPlotting OVOS convergence histories for {molecule_name}/{basis} for methods: {methods}\n")

    # Make plot
        # For each number of optimized virtual orbitals,
            # create a new plot that is saved to its new file
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "backup" / "data" / molecule_name / basis / "OVOS"

        # Define markers for each method
    marker_style = {
        'RHF': ('D'),
        'prev': ('s'),
        'random': ('^')
    }

         # gather each method in the same plot for each number of opt. virt. orb.
    
    MP2_energies_per_iteration = {}

    if methods is None:
        methods = ["RHF", "prev", "random"]
    for method in methods:
        json_file = data_dir / f"lst_MP2_OVOS_virt_orbs_{method}.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                conv_data = json.load(f)
            print(f"✅ Loaded {method} data from {json_file}")
        else:
            print(f"⚠️  {method} data not found at {json_file}")
            continue
        
        for num_virt_orbs, energies in zip(conv_data[1], conv_data[0]):
            # Skip the full space point (num_virt_orbs = full_space_size - num_electrons//2) since it doesn't have a convergence history
            if num_virt_orbs == conv_data[1][-1]:
                continue

            # # Need to remove the last 50 points as these were just the "keep track" history and not the actual convergence history
            # energies = energies[:-50] if len(energies) > 50 else energies

            if num_virt_orbs not in MP2_energies_per_iteration:
                MP2_energies_per_iteration[num_virt_orbs] = {}
            MP2_energies_per_iteration[num_virt_orbs][method] = energies
        
    # Color map for different virtual orbitals
        # Get data for molecule by file backup/data/{molecule_name}/{basis}/OVOS/molecule_data.json
    molecule_data_file = data_dir / "molecule_data.json"
    if molecule_data_file.exists():
        with open(molecule_data_file, "r") as f:
            molecule_data = json.load(f)
        num_electrons = molecule_data.get("num_electrons", None)
        full_space_size = molecule_data.get("full_space_size", None)
    
    active_space_size = full_space_size - num_electrons // 2 + 1
    colors = plt.cm.hsv(np.linspace(0, 1, active_space_size))
        
        # Plot each number of optimized virtual orbitals in a separate plot
    for num_virt_orbs, method_energies in MP2_energies_per_iteration.items():
        plt.figure(figsize=(8, 5))

        # Make a list of widths for each line based on the number of iterations to converge, so that lines that took more iterations are thicker
        iterations_to_converge = [len(method_energies[method]) for method in method_energies]
        width_lst = [2, 3.5, 5]
            # sort methods by number of iterations to converge so that the method that converged the fastest is plotted on top
        methods_sorted = sorted(method_energies.keys(), key=lambda m: len(method_energies[m]))

        # For lowest energy
        min_energies = 0

                # Add markers and color dependent on the method and number of virtual orbitals
        for method in methods_sorted:
            energies = method_energies[method]
                # Update lowest energy of methods
            min_energies = min(min_energies, min(energies))

            # Marker
            marker = marker_style.get(method, 'o')

            # Color based on the number of virtual orbitals, with a gradient from blue (few virtual orbitals) to red (many virtual orbitals)
            color = colors[num_virt_orbs//2 - 1] if num_virt_orbs//2 - 1 < len(colors) else colors[-1]
            
            # First and last points with markers, and the rest with a see-through line to show the convergence path
            plt.scatter(1, energies[0], marker=marker, color=color, alpha=0.5)
            plt.scatter(len(energies), energies[-1], marker=marker, color=color)

            # See through line connecting the points to show the convergence path, with the same color as the points but more transparent
            plt.plot(range(1, len(energies) + 1), energies, color=color, alpha=0.25, linewidth=width_lst[methods_sorted.index(method)])

        # Not sorted methods for legend
        for method in method_energies.keys():
            # Marker
            marker = marker_style.get(method, 'o')

            # Legend for markers
            plt.scatter([], [], marker=marker, color='black', label=f'Start Guess: {method}')

        # Reference line 
            # Plottet if final MP2 corr energy is within 90% of the full space MP2 corr energy
        final_MP2_energy = min_energies
        E_corr_MP2 = molecule_data.get("MP2_e_corr", None)
            # for full space MP2 correlation energy
        plt.axhline(E_corr_MP2, color='red', linestyle='--', label='Full Space MP2', linewidth=2, alpha=0.75)
            # for full space OO-MP2 correlation energy
        E_corr_OOMP2 = molecule_data.get("OOMP2_e_corr", None)
        plt.axhline(E_corr_OOMP2, color='blue', linestyle='--', label='Full Space OO-MP2', linewidth=2, alpha=0.75)

        # Title and labels
        plt.title(f'OVOS Convergence History: {molecule_name}/{basis}, {num_virt_orbs} Optimized Virtual Orbitals')
        plt.xlabel('Iteration')
        plt.ylabel('Correlation Energy [Hartree]')

        # Set x-axis to start from 0
        plt.xlim(0, None)

        # Grid and Legend
        plt.grid()
        plt.legend()
        
        # Save figure
        output_dir = project_root / "backup" / "images" / molecule_name / basis / "convergence_histories"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"ovos_convergence_history_{molecule_name}_{basis}_{num_virt_orbs}_virt_orbs.png"
        plt.savefig(output_file, dpi=150)
        print(f"✅ Plot saved to: {output_file}")
        plt.close()
        

# Plot MO coefficient heat-maps for the initial and final orbitals of OVOS, and their differences, for a given molecule, basis set, and method (RHF, prev, random)
def plot_OVOS_MO_coefficients(molecule, basis, method="RHF"):
    """
    Plot MO coefficient heatmaps for initial and final orbitals of OVOS, and their differences.
    
    Parameters
    ----------
    molecule : str
        Molecular geometry
    basis : str
        Basis set
    method : str
        Initialization method ("RHF", "prev", or "random")
    """
    # Molecule name for file paths
    molecule_name = get_molecule_name(molecule)    

    print(f"\nPlotting MO coefficients for {molecule_name}/{basis} with method: {method}\n")

    # Get data from JSON file saved by ovos_object()
    data_dir = project_root / "backup" / "data" / molecule_name / basis / "OVOS"
    json_file = data_dir / f"lst_MP2_OVOS_virt_orbs_{method}.json"

    method = np.array(methods) if isinstance(methods, list) else np.array([methods])
    
    if not json_file.exists():
        print(f"⚠️  Data file not found at {json_file}")
        return
    
    with open(json_file, "r") as f:
        conv_data = json.load(f)
    print(f"✅ Loaded data from {json_file}")

    # Plot MO coefficient heatmaps for the first and last point of the convergence history, and their difference
        # For the first and last point, we will plot the MO coefficients for both alpha and beta spin, and their difference
    for i, num_virt_orbs in enumerate(conv_data[1]):
        if num_virt_orbs == conv_data[1][-1]:
            continue  # Skip full space point

        mo_coeffs_initial = conv_data[4][i][0]  # Initial MO coefficients (alpha and beta)
        mo_coeffs_final = conv_data[4][i][-1]    # Final MO coefficients (alpha and beta)

        # Make sure mo_coeffs_initial and mo_coeffs_final are numpy arrays
            # Rebuild numpy arrays from lists if needed
        if isinstance(mo_coeffs_initial, list):
            mo_coeffs_initial = np.array(mo_coeffs_initial)
        if isinstance(mo_coeffs_final, list):
            mo_coeffs_final = np.array(mo_coeffs_final)

        # Plot heatmaps for alpha and beta spin, and their difference
            # Here i want each plot to have two rows with alpha spin on top and beta spin on the bottom, and three columns with initial, final, and difference
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'MO Coefficients: {molecule_name}/{basis}, {num_virt_orbs} Optimized Virtual Orbitals, Method: {method}', fontsize=16)
        
            # For which all should share one colorbar, placed on the right hand side of the figure in the middle
        vmin = min(np.min(mo_coeffs_initial), np.min(mo_coeffs_final))
        vmax = max(np.max(mo_coeffs_initial), np.max(mo_coeffs_final))

        print()
        print(f"MO Coefficients for {method} method with {num_virt_orbs} optimized virtual orbitals:")
        print(mo_coeffs_initial[0])

            # Alpha 
                # spin initial
        im0 = axes[0, 0].imshow(mo_coeffs_initial[0], vmin=vmin, vmax=vmax, cmap='viridis')
        axes[0, 0].set_title('Alpha Spin - Initial')
                # spin final
        im1 = axes[0, 1].imshow(mo_coeffs_final[0], vmin=vmin, vmax=vmax, cmap='viridis')
        axes[0, 1].set_title('Alpha Spin - Final')
                # spin difference
        im2 = axes[0, 2].imshow(mo_coeffs_final[0] - mo_coeffs_initial[0], cmap='bwr', vmin=-vmax, vmax=vmax)
        axes[0, 2].set_title('Alpha Spin - Difference')
            # Beta
                # spin initial
        im3 = axes[1, 0].imshow(mo_coeffs_initial[1], vmin=vmin, vmax=vmax, cmap='viridis')
        axes[1, 0].set_title('Beta Spin - Initial')
                # spin final
        im4 = axes[1, 1].imshow(mo_coeffs_final[1], vmin=vmin, vmax=vmax, cmap='viridis')
        axes[1, 1].set_title('Beta Spin - Final')
                # spin difference
        im5 = axes[1, 2].imshow(mo_coeffs_final[1] - mo_coeffs_initial[1], cmap='bwr', vmin=-vmax, vmax=vmax)
        axes[1, 2].set_title('Beta Spin - Difference')
        
            # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im1, cax=cbar_ax)
                
            # Save figure
        output_dir = project_root / "backup" / "images" / molecule_name / basis / "MO_coefficients"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"MO_coefficients_{molecule_name}_{basis}_{method}_{num_virt_orbs}_virt_orbs.png"
        plt.savefig(output_file, dpi=150)
        print(f"✅ Plot saved to: {output_file}")
        plt.close()



# Plot the MO coefficient of all methods vs MP2 for the last point
def plot_OVOS_MO_coefficient_diff_MP2(molecule, basis):
    """
    Plot the difference in MO coefficients between the final OVOS orbitals and the MP2 orbitals for all methods (RHF, prev, random) for the last point of the convergence history.
    """

    # Molecule name for file paths
    molecule_name = get_molecule_name(molecule)
    methods = ["RHF", "prev", "random"]

    print(f"\nPlotting MO coefficient differences vs MP2 for {molecule_name}/{basis} for methods: {methods}\n")

    # Check if the plot already exists for all methods, and if so, skip the calculation and plotting
    output_dir = project_root / "backup" / "images" / molecule_name / basis / "MO_coefficients_vs_MP2"
    output_dir.mkdir(parents=True, exist_ok=True)
    files_exist = [False] * len(methods)
    for method in methods:
        output_file = output_dir / f"MO_coefficients_vs_MP2_{molecule_name}_{basis}_{method}.png"
        if output_file.exists():
            print(f"✅ Plot already exists for {method} method at {output_file}, skipping calculation and plotting.")
            files_exist[methods.index(method)] = True
    if all(files_exist):
        print("✅ All plots already exist, skipping calculation and plotting.")
        # return

    # Calculate MP2 MO coefficients for the full space
        # We need to create a molecule and perform an RHF calculation to get the MO coefficients, since MP2 orbitals are the same as RHF orbitals
    mol = gto.Mole()
    mol.atom = molecule
    mol.basis = basis
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.build()
        # Perform RHF calculation to get MO coefficients (MP2 orbitals are the same as RHF orbitals)
    mf = scf.RHF(mol)
    mf.kernel()
        # Do MP2
    MP2 = mf.MP2().run()
    mo_coeffs_MP2 = MP2.mo_coeff  # MO coefficients for MP2 (same as RHF)

    # Find 75% of the virtual orbitals to determine which point in the convergence history to plot
    num_electrons = mol.nelectron
    full_space_size = mf.mo_coeff.shape[1]
    num_virtual_orbitals = full_space_size - num_electrons // 2
    num_opt_virt_orb = int(num_virtual_orbitals * 0.75)

    # Get data for each method
    methods = ["RHF", "prev", "random"]
    mo_coeffs_final = {}
    for method in methods:
        data_dir = project_root / "backup" / "data" / molecule_name / basis / "OVOS"
        json_file = data_dir / f"lst_MP2_OVOS_virt_orbs_{method}.json"
        
        if not json_file.exists():
            print(f"⚠️  Data file not found at {json_file}")
            continue
        
        with open(json_file, "r") as f:
            conv_data = json.load(f)
        print(f"✅ Loaded {method} data from {json_file}")

        mo_coeffs_final[method] = conv_data[4][num_opt_virt_orb]  # Final MO coefficients (alpha and beta) for the last point


    # Plot the MO coefficients for method beside MP2 for the last point, and their difference
        # Plot heatmaps for alpha and beta spin, and their difference
    for method in methods:
        if method not in mo_coeffs_final:
            continue
        
        mo_coeffs_ovos = np.array(mo_coeffs_final[method])
        
        # Create GridSpec for better control over layout
        # 2 rows x 4 columns: [heatmap, heatmap, buffer, heatmap]
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 4, right=0.88, hspace=0.1, wspace=0.1, 
                              width_ratios=[1, 1, 0.2, 1])
        
        # Create axes for the 8 subplots (2 rows, skipping buffer columns)
        axes = []
        for i in range(2):
            # First two columns (OVOS and MP2)
            axes.append(fig.add_subplot(gs[i, 0]))
            axes.append(fig.add_subplot(gs[i, 1]))
            # Skip buffer column (i, 2)
            # Last two columns (Difference plots)
            axes.append(fig.add_subplot(gs[i, 3]))

        # Title
        fig.suptitle(f'MO Coefficients vs MP2: {molecule_name}/{basis}, Method: {method}, {num_opt_virt_orb} Optimized Orbitals', 
                     fontsize=24, y=0.98)

        # Determine color scales
        vmin = min(np.min(mo_coeffs_ovos[0]), np.min(mo_coeffs_MP2))
        vmax = max(np.max(mo_coeffs_ovos[0]), np.max(mo_coeffs_MP2))
        
        diff_vmax = max(
            np.max(np.abs(mo_coeffs_ovos[0] - mo_coeffs_MP2[:, :mo_coeffs_ovos[0].shape[1]])),
            np.max(np.abs(mo_coeffs_ovos[1] - mo_coeffs_MP2[:, :mo_coeffs_ovos[1].shape[1]]))
        )
        diff_vmin = -diff_vmax

        # Alpha spin plots
        im0 = axes[0].imshow(mo_coeffs_ovos[0], vmin=vmin, vmax=vmax, cmap='viridis')
        axes[0].set_ylabel('Basis Functions')
        
        im1 = axes[1].imshow(mo_coeffs_MP2[:, :mo_coeffs_ovos[0].shape[1]], vmin=vmin, vmax=vmax, cmap='viridis')
        
        im2 = axes[2].imshow(mo_coeffs_ovos[0] - mo_coeffs_MP2[:, :mo_coeffs_ovos[0].shape[1]], 
                             cmap='RdBu_r', vmin=diff_vmin, vmax=diff_vmax)

        # Beta spin plots
        im3 = axes[3].imshow(mo_coeffs_ovos[1], vmin=vmin, vmax=vmax, cmap='viridis')
        axes[3].set_ylabel('Basis Functions')
        
        im4 = axes[4].imshow(mo_coeffs_MP2[:, :mo_coeffs_ovos[1].shape[1]], vmin=vmin, vmax=vmax, cmap='viridis')
        
        im5 = axes[5].imshow(mo_coeffs_ovos[1] - mo_coeffs_MP2[:, :mo_coeffs_ovos[1].shape[1]], 
                             cmap='RdBu_r', vmin=diff_vmin, vmax=diff_vmax)

        # Plot a black stripped line in each heatmap to indicate the separation between occupied and virtual orbitals, which is at num_electrons//2
        num_occupied_orbitals = num_electrons // 2
        for ax in axes:
            ax.axvline(num_occupied_orbitals - 0.5, color='black', linestyle='--', linewidth=1)

        # Add x-labels for bottom row
        for i in [3, 4, 5]:
            axes[i].set_xlabel('Molecular Orbitals')
        # Set x-ticks to show every 2nd MO, even, for better readability
        for ax in [axes[0], axes[1], axes[3], axes[4]]:
            ax.set_xticks(np.arange(0, mo_coeffs_ovos[0].shape[1], 2))
                # Check if //2 is less than the number of ticks not skipping every 2nd
            mo_labels = mo_coeffs_ovos[0].shape[1]//2 if mo_coeffs_ovos[0].shape[1]//2 == len(np.arange(0, mo_coeffs_ovos[0].shape[1], 2)) else mo_coeffs_ovos[0].shape[1]//2 + 1
            ax.set_xticklabels(np.arange(0, mo_labels, 1))
            ax.set_yticks(np.arange(0, mo_coeffs_ovos[0].shape[1], 2))
            ax.set_yticklabels(np.arange(0, mo_labels, 1))

        # Colorbar 1: For viridis (MO coefficients)
        cbar_ax1 = fig.add_axes([0.59, 0.16, 0.015, 0.7])
        cbar1 = fig.colorbar(im1, cax=cbar_ax1)
        cbar1.set_label('MO Coefficient Value', rotation=270, labelpad=20, fontweight='bold')

        # Colorbar 2: For RdBu (differences)
        cbar_ax2 = fig.add_axes([0.89, 0.16, 0.015, 0.7])
        cbar2 = fig.colorbar(im2, cax=cbar_ax2)
        cbar2.set_label('Difference (OVOS - MP2)', rotation=270, labelpad=18, fontweight='bold')

        # Add labels for the coloumns on the top of coloumns for OVOS, MP2 and difference and on the left hand side for alpha and beta spin
            # Column labels (top)
        fig.text(0.23, 0.92, 'OVOS Orbitals', ha='center', va='top', fontsize=12, fontweight='bold')
        fig.text(0.47, 0.92, 'MP2 Orbitals', ha='center', va='top', fontsize=12, fontweight='bold')
        fig.text(0.78, 0.92, 'Difference (OVOS - MP2)', ha='center', va='top', fontsize=12, fontweight='bold')
        
            # Row labels (left side)
        fig.text(0.085, 0.695, 'Alpha Spin', ha='left', va='center', rotation=90, fontsize=12, fontweight='bold')
        fig.text(0.085, 0.295, 'Beta Spin', ha='left', va='center', rotation=90, fontsize=12, fontweight='bold')
        
        # Save figure
        output_file = output_dir / f"MO_coefficients_vs_MP2_{molecule_name}_{basis}_{method}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_file}")
        plt.close()



# Plot a 3D surface plot as a heatmap over the surface - illustrating the convergence landscape of OVOS as a function of a optimized virtual orbitals, for a given molecule, basis set, and method (RHF, prev, random)
    # Axis x: convergence criterion energy 
    # Axis y: convergence criterion gradient norm
    # Axis z: value of final correlation energy achieved by OVOS for that point in the convergence landscape
    # Heatmap color: number of iterations to converge for that point in the convergence landscape (darker colors for more iterations, lighter colors for fewer iterations)
def plot_OVOS_convergence_landscape(molecule, basis, method="RHF", conv_criteria=None):
    """
    Plot a 3D surface plot illustrating the convergence landscape of OVOS as a function of convergence criteria (energy and gradient norm).
    
    Parameters
    ----------
    molecule : str
        Molecular geometry
    basis : str
        Basis set
    method : str
        Initialization method ("RHF", "prev", or "random")
    """
    # Molecule name for file paths
    molecule_name = get_molecule_name(molecule)

    print(f"\nPlotting OVOS convergence landscape for {molecule_name}/{basis}, Method: {method}...\n")

    # Get all the data from JSON file saved in each their own convergence_criteria file
        # Save hte conv_data for each criteria
    conv_data = {}
    loaded_criteria = []
    missing_criteria = []
        # Gather all the data for each convergence criteria combinations of conv_energy and conv_grad in conv_criteria
    for criteria in conv_criteria:
        conv_energy = criteria["conv_energy"]
        conv_grad = criteria["conv_grad"]
            # Convert to scientific notation with 2 decimal places for file path
        conv_energy = f"{conv_energy:.0e}"
        conv_grad = f"{conv_grad:.0e}"

        # Cutoff conv criteria values from before a certain threshold for better visualization of the landscape, since the points with very loose conv criteria are not interesting and they make the landscape less clear
            # If below 1e-4 for energy and 1e-4 for gradient norm, skip the point
        if float(conv_energy) > 1e-3 or float(conv_grad) > 1e-3:
            continue

            # If conv_ starts with "5e-", skip
        if conv_energy.startswith("5e-") or conv_grad.startswith("5e-"):
            continue

        json_path = project_root / "backup" / "data" / molecule_name / basis / "OVOS" / f"convergence_criteria" / method 
        json_file = json_path / f"OVOS_convergence_{conv_energy}_{conv_grad}_keep_none.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                conv_data[(conv_energy, conv_grad)] = json.load(f)
            loaded_criteria.append((conv_energy, conv_grad))
        else:
            missing_criteria.append((conv_energy, conv_grad))
            
    
    # Check which criteria were loaded for debugging
    print(f"Loaded data for convergence criteria combinations (energy, grad):")
        # If all were loaded, print a success message    
    if len(loaded_criteria) == len(conv_criteria):
        print("✅ Successfully loaded data for all convergence criteria combinations.")
    if len(missing_criteria) > 0:
        print(f"⚠️  Missing data for convergence criteria combinations (energy, grad): {missing_criteria}\n")
    

    # Prepare data for 3D surface plot
        # We will create a grid of energy and gradient norm values, and for each point in the grid, we will find the corresponding final correlation energy and number of iterations to converge from the conv_data
    energy_values = []
    grad_norm_values = []
    final_correlation_energies = []
    iterations_to_converge = []

    # Delete num_opt_virt_orb variable if it exists, since we will set it from the data and it should be the same for all points in the convergence landscape
    if 'num_opt_virt_orb' in locals():
        del num_opt_virt_orb

    for (conv_energy, conv_grad), data in conv_data.items():
        conv_energy = float(conv_energy)
        conv_grad = float(conv_grad)

        final_correlation_energy = float(data["E_corr_hist"][-1])
        iterations = len(data["E_corr_hist"])

        # Set num_opt_virt_orb from file
            # Do it once as it should be the same for all points in the convergence landscape, since we are only changing the convergence criteria and not the number of optimized virtual orbitals
        if 'num_opt_virt_orb' not in locals():
            num_opt_virt_orb = data["num_opt_virt_orbs"]

        energy_values.append(conv_energy)
        grad_norm_values.append(conv_grad)
        final_correlation_energies.append(final_correlation_energy)
        iterations_to_converge.append(iterations)

    # Check max min values for debugging
    print(f"\nEnergy values range: {min(energy_values):.2e} to {max(energy_values):.2e}")
    print(f"Gradient norm values range: {min(grad_norm_values):.2e} to {max(grad_norm_values):.2e}")
    print(f"Final correlation energies range: {min(final_correlation_energies):.2e} to {max(final_correlation_energies):.2e}")
    print(f"Iterations to converge range: {min(iterations_to_converge)} to {max(iterations_to_converge)}\n")

    # Convert lists to numpy arrays for easier manipulation
    energy_values = np.array(energy_values)
    grad_norm_values = np.array(grad_norm_values)
    final_correlation_energies = np.array(final_correlation_energies)
    iterations_to_converge = np.array(iterations_to_converge)

    # Transform to log10 scale
    energy_values = np.log10(energy_values)
    grad_norm_values = np.log10(grad_norm_values)

        # Create a grid for energy and gradient norm
    energy_grid, grad_norm_grid = np.meshgrid(np.unique(energy_values), np.unique(grad_norm_values))
            # Check the shapes of the grids and the values for debugging
    print(f"Energy grid shape: {energy_grid.shape}, unique energy values: {np.unique(energy_values)}")
    print(f"Gradient norm grid shape: {grad_norm_grid.shape}, unique gradient norm values: {np.unique(grad_norm_values)}")
                # Compare to the amount of combinations of convergence criteria we have in conv_data for debugging
    print(f"Number of unique energy values: {len(np.unique(energy_values))}")
    print(f"Number of unique gradient norm values: {len(np.unique(grad_norm_values))}")
    print(f"Number of combinations of convergence criteria in conv_data: {len(conv_data)}\n")

        # Create a grid for final correlation energies and iterations to converge    
    final_correlation_energy_grid = np.zeros_like(energy_grid)
    iterations_to_converge_grid = np.zeros_like(energy_grid)

    for i in range(energy_grid.shape[0]):
        for j in range(energy_grid.shape[1]):
            energy = energy_grid[i, j]
            grad_norm = grad_norm_grid[i, j]
            mask = (energy_values == energy) & (grad_norm_values == grad_norm)
            if np.any(mask):
                final_correlation_energy_grid[i, j] = final_correlation_energies[mask][0]
                iterations_to_converge_grid[i, j] = iterations_to_converge[mask][0]
            else:
                final_correlation_energy_grid[i, j] = np.nan
                iterations_to_converge_grid[i, j] = np.nan
            
            # Check the values for debugging
    print(f"Final correlation energy grid shape: {final_correlation_energy_grid.shape}")
    print(f"Iterations to converge grid shape: {iterations_to_converge_grid.shape}\n")
                # Check the range of values of criteria in the grids for debugging
    print(f"Energy grid values range: {np.nanmin(energy_grid):.2e} to {np.nanmax(energy_grid):.2e}")
    print(f"Gradient norm grid values range: {np.nanmin(grad_norm_grid):.2e} to {np.nanmax(grad_norm_grid):.2e}")
    print(f"Final correlation energy grid values range: {np.nanmin(final_correlation_energy_grid):.2e} to {np.nanmax(final_correlation_energy_grid):.2e}")
    print(f"Iterations to converge grid values range: {np.nanmin(iterations_to_converge_grid)} to {np.nanmax(iterations_to_converge_grid)}\n")

    # Create 3D surface plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

        # Normalize iterations_to_converge_grid to [0, 1] for colormap
    vmin_iter = np.nanmin(iterations_to_converge_grid)
    vmax_iter = np.nanmax(iterations_to_converge_grid)
    norm_iter = plt.Normalize(vmin=vmin_iter, vmax=vmax_iter)
    colors_surface = plt.cm.viridis(norm_iter(iterations_to_converge_grid))
    
    surf = ax.plot_surface(energy_grid, grad_norm_grid, final_correlation_energy_grid, facecolors=colors_surface, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.9, zorder=1)

    # Plot a point for the choosen convergence criteria combinations
            # conv_energy=1e-8, conv_grad=1e-6
    chosen_energy = -8
    chosen_grad = -6
        # find the corresponding final correlation energy for the chosen convergence criteria from the grid for debugging
    mask_chosen = (energy_grid == chosen_energy) & (grad_norm_grid == chosen_grad)
    if np.any(mask_chosen):
        chosen_final_correlation_energy = final_correlation_energy_grid[mask_chosen][0]
        print(f"Final correlation energy for chosen convergence criteria: {chosen_final_correlation_energy:.2e} (x,y: {chosen_energy:.2e}, {chosen_grad:.2e})")
        ax.plot([chosen_energy, chosen_energy], [chosen_grad, chosen_grad], [chosen_final_correlation_energy, chosen_final_correlation_energy], marker='o', markersize=10, color='red', label='Chosen Convergence Criteria', zorder=10)
    else:
        print("⚠️  Chosen convergence criteria not found in the grid, skipping plotting the point.")

        # Create a proper colorbar with the normalization
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm_iter)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5, label='Iterations to Converge')
    cbar.ax.yaxis.labelpad = 100

        # Check that the axis labels are correct for debugging
            # Get the values of the axes from surf for debugging
    # print(f"Energy axis values: {surf._vec[0]}")
    # print(f"Gradient norm axis values: {surf._vec[1]}")
    # print(f"Final correlation energy axis values: {surf._vec[2]}\n") 

        # Set labels and title
    ax.set_xlabel('Convergence Criterion: Energy')
    ax.set_ylabel('Convergence Criterion: Gradient Norm')
    ax.set_zlabel('Final Correlation Energy [Hartree]')
    ax.set_title(f'OVOS Convergence Landscape: {molecule_name}/{basis}, Method: {method}, Optimized Virtual Orbitals: {num_opt_virt_orb}')
    
        # Set zlim to the range of final correlation energies for better visualization
    ax.set_zlim(np.nanmin(final_correlation_energy_grid), #- 0.001 * abs(np.nanmin(final_correlation_energy_grid)),
                 np.nanmax(final_correlation_energy_grid)) # + 0.001 * abs(np.nanmax(final_correlation_energy_grid)))  # Add 10% padding to the top of the z-axis for better visualization

    # Replace the current ticks on the x and y axis with the original convergence criteria values (not in log scale) for better interpretability
        # Get the original convergence criteria values from conv_data for debugging
    original_energy_values = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    original_grad_norm_values = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
        
        # Set xlim and ylim to the original values for better interpretability
    ax.set_xlim(np.log10(min(original_energy_values)), np.log10(max(original_energy_values)))
    ax.set_ylim(np.log10(min(original_grad_norm_values)), np.log10(max(original_grad_norm_values)))
        
        # Set the ticks to the original values
    ax.set_xticks(np.log10(original_energy_values))
    ax.set_xticklabels([f"{val:.0e}" for val in original_energy_values])
    ax.set_yticks(np.log10(original_grad_norm_values))
    ax.set_yticklabels([f"{val:.0e}" for val in original_grad_norm_values])

        # But make the first ticks label empty for better visualization, since the first ticks are often very close to each other and overlap
    ax.set_xticklabels([''] + [f"{val:.0e}" for val in original_energy_values[1:]])
    ax.set_yticklabels([''] + [f"{val:.0e}" for val in original_grad_norm_values[1:]])
            # Remove the first tick on the x and y axis for better visualization, since the first ticks are often very close to each other and overlap
    ax.set_xticks(np.log10(original_energy_values[1:]))
    ax.set_yticks(np.log10(original_grad_norm_values[1:]))

        # invert the x and y axis to have the smallest convergence criteria in the front left corner of the plot for better visualization
    ax.invert_xaxis()
    ax.invert_yaxis()

        # Move the z-axis ticks out from the z-axis line for better visibility
    ax.tick_params(axis='z', pad=10, direction='inout')
            # Pad the axis labels for better visibility
    ax.xaxis.labelpad = 12
    ax.yaxis.labelpad = 12
    ax.zaxis.labelpad = 20
            # Rotate the x and y axis tick labels for better visibility
    ax.set_xticklabels(ax.get_xticklabels(), rotation=135, ha='left')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right')
                # Negative pad label ticks for better visibility
    ax.tick_params(axis='x', pad=-5)
    ax.tick_params(axis='y', pad=-5)

    plt.tight_layout()

    # Set view angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Save figure
    output_dir = project_root / "backup" / "images" / molecule_name / basis / "convergence_landscape"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"OVOS_convergence_landscape_{molecule_name}_{basis}_{method}.png"
    plt.savefig(output_file, dpi=150)
    print(f"✅ Plot saved to: {output_file}")
    plt.close()



# ================================================================================================
# Helper functions for running OVOS and collecting data for different molecules, basis sets, and initialization methods (RHF, random)
# ================================================================================================

# Helper function to extract molecule name from the geometry string, for use in file paths and plot titles
def get_molecule_name(molecule):
    # Get data from JSON file saved by ovos_object()
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
    
    return molecule_name

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


def run_ovos_for_virtual_orbs(molecule, basis, method, num_opt_virtual_orbs,
                              conv_energy=1e-8, conv_grad=1e-6, keep_track_max=50):
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
        verbose=0, max_iter=1000,
        conv_energy=conv_energy, conv_grad=conv_grad,
        keep_track_max=keep_track_max
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
        verbose=0, max_iter=num_iter_max,
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
    
    # Get molecule name for file paths and print setup message
    molecule_name = get_molecule_name(molecule)

    print(f"=== OVOS VQE for {molecule_name} with {basis} basis (method: {method}) ===")
    print(f"📁 Setting up data collection for {molecule_name} with {basis} basis and method {method}...\n")

    # Check if the files exist before running OVOS
        # Path
    output_dir = project_root / "backup" / "data" / molecule_name / basis / "OVOS"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"lst_MP2_OVOS_virt_orbs_{method}.json"
    if output_file.exists():  # For random method we want to overwrite to get new random initializations
        print(f"⚠️  Output file already exists at {output_file}, skipping OVOS run to avoid overwriting.\n")
        return

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
            num_random_attempts = 1000 if basis == "6-31G" else 500  # More attempts for smaller basis sets to get good statistics
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
                num_iter_max = 1000 if basis == "6-31G" else 500  # Default max iterations for random attempts if no RHF data is available
                
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
                            completed_count_max = 500 if basis == "6-31G" else 250

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
                                
                            elif completed_count >= completed_count_max:
                                # If we are at full space, we want to do all attempts to get a good distribution of results, so we won't stop early
                                if num_opt_virtual_orbs < (num_orbitals - num_electrons // 2):
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
                if num_opt_virtual_orbs == 1:
                    # Start with RHF orbitals for the first virtual orbital count
                    mo_coeffs_warmstart = [mf.mo_coeff, mf.mo_coeff]
                else:
                    mo_coeffs_warmstart = mo_coeffs  # Use last converged orbitals as warm start for next run

                # Run OVOS with warmstart initialization
                ovos = OVOS(
                    mol=mol, scf=mf, Fao=Fao,
                    num_opt_virtual_orbs=2*num_opt_virtual_orbs,
                    mo_coeff=mo_coeffs_warmstart,
                    init_orbs="RHF",
                    verbose=0, max_iter=1000,
                    conv_energy=1e-8, conv_grad=1e-6,
                    keep_track_max=50
                )

                E_corr, E_corr_hist, E_corr_iter, E_corr_mo, _, stop_reason = ovos.run(
                    mo_coeffs_warmstart, fock_spin=None
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

# Make of data for different convergence criteria for one method, molecule, and basis set
def run_ovos_convergence_criteria_single(molecule, basis, method, conv, mol, mf, Fao, mo_coeffs, num_electrons, num_orbitals, E_ref, E_corr_MP2):
    """
    Run OVOS for a single set of convergence criteria and collect results.
    
    This function is designed to be called in parallel for different convergence criteria.
    
    Parameters
    ----------
    molecule : str
        Molecular geometry
    basis : str
        Basis set
    method : str
        Initialization method ("RHF", "prev", or "random")
    conv : dict
        Convergence criteria dictionary, e.g.:
        {'conv_energy': 1e-8, 'conv_grad': 1e-6, 'keep_track_max': "none"}
    mol, mf, Fao, mo_coeffs, num_electrons, num_orbitals, E_ref, E_corr_MP2 : precomputed molecule data
    
    Returns
    -------
    dict
        Dictionary containing results for this set of convergence criteria
    """
    
    conv_energy = conv['conv_energy']
    conv_grad = conv['conv_grad']
    keep_track_max = conv['keep_track_max']

    # Do 75% of the maximum number of virtual orbitals for this molecule/basis set to speed up convergence testing
    num_opt_virtual_orbs = int(0.75 * (num_orbitals - num_electrons // 2))

    # Create OVOS object and run with specified convergence criteria
    ovos = OVOS(
        mol=mol, scf=mf, Fao=Fao,
        num_opt_virtual_orbs=num_opt_virtual_orbs,
        mo_coeff=mo_coeffs,
        init_orbs="RHF",
        verbose=0, max_iter=1000,
        conv_energy=conv_energy, conv_grad=conv_grad,
        keep_track_max=keep_track_max
    )

    E_corr, E_corr_hist, E_corr_iter, E_corr_mo, _, stop_reason = ovos.run(
        mo_coeffs, fock_spin=None
    )

    return {
        'conv_energy': conv_energy,
        'conv_grad': conv_grad,
        'keep_track_max': keep_track_max,
        'final_energy': E_corr,
        'E_corr_hist': E_corr_hist,
        'E_corr_iter': E_corr_iter,
        'E_corr_mo': E_corr_mo,
        'stop_reason': stop_reason,
        'num_opt_virt_orbs': num_opt_virtual_orbs,
        'E_corr_MP2': E_corr_MP2
    }


def ovos_convergence_criteria(molecule, basis, method, conv_criteria):
    """
    Run OVOS for different convergence criteria and collect data.
    
    Parameters
    ----------
    molecule : str
        Molecular geometry
    basis : str
        Basis set
    method : str
        Initialization method ("RHF", "prev", or "random")
    conv_criteria : list of dict
        List of convergence criteria dictionaries, e.g.:
        [
            {'conv_energy': 1e-6,  'conv_grad': 1e-4, 'keep_track_max': "none"},
            {'conv_energy': 1e-8,  'conv_grad': 1e-6, 'keep_track_max': "none"},
            {'conv_energy': 1e-10, 'conv_grad': 1e-8, 'keep_track_max': "none"},
        ]
    """
    # Check if the length of files exist before running OVOS
    molecule_name = get_molecule_name(molecule)

    print()
    print(f"=== OVOS Convergence Criteria Testing for {molecule_name} with {basis} basis (method: {method}) ===\n")


    output_dir = project_root / "backup" / "data" / molecule_name / basis / "OVOS" / "convergence_criteria" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(output_dir.glob("OVOS_convergence_*.json"))
    if len(existing_files) == len(conv_criteria):
        print(f"✅ Found existing files for all {len(conv_criteria)} convergence criteria in {output_dir}. Skipping OVOS runs to avoid overwriting.\n")
        return
    else:
        print(f"📁 Found {len(existing_files)} existing files in {output_dir}, but {len(conv_criteria)} convergence criteria to test. Proceeding with OVOS runs for missing criteria...\n")
    
    # I want to parallelize over the different convergence criteria
        # But do one initial run to get the molecule data and precompute integrals, then pass those to the parallel workers to avoid redundant calculations
    mol, mf, Fao, mo_coeffs, num_electrons, num_orbitals, E_ref, E_corr_MP2 = mol_data(
        molecule, basis, method="RHF"
    )

    # Setup data collection structure
    results = []

    # Delete the conv_criteria that already have files to avoid overwriting
    conv_criteria_to_run = []
    count_missing = 0
    for conv in conv_criteria:
        conv_energy_str = f"{conv['conv_energy']:.0e}"
        conv_grad_str = f"{conv['conv_grad']:.0e}"
        file_name = f"OVOS_convergence_{conv_energy_str}_{conv_grad_str}_keep_none.json"
        output_file = output_dir / file_name
        if output_file.exists() == False:
            conv_criteria_to_run.append(conv)
            count_missing += 1

    print(f"📁 {count_missing}/{len(conv_criteria)} convergence criteria do not have existing files and will be run.\n")
    if count_missing == 0:
        print("All convergence criteria already have results files. No OVOS runs needed.\n")
        return


    # Run OVOS for each set of convergence criteria in parallel
    workers = min(len(conv_criteria_to_run), 1)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                 run_ovos_convergence_criteria_single,
                molecule, basis, method, conv, mol, mf, Fao, mo_coeffs, num_electrons, num_orbitals, E_ref, E_corr_MP2
            ): conv for conv in conv_criteria_to_run
        }

        # Save each result as it completes and check if already exists to avoid overwriting        
        for future in futures:
            conv = futures[future]
            
            # Keep track of what number out of total runs we are at for better progress monitoring
            current_run = len(results) + 1
            total_runs = len(conv_criteria_to_run)

            try:
                # Check if file already exists for this convergence criteria
                conv_energy_str = f"{conv['conv_energy']:.0e}"
                conv_grad_str = f"{conv['conv_grad']:.0e}"

                # Save result to JSON
                print(f"[{current_run}/{total_runs}] Running OVOS for conv_energy={conv_energy_str}, conv_grad={conv_grad_str}")
                result = future.result()
                results.append(result)
                print(f"  ✅ Final energy: {result['final_energy']:.6f} Ha, "
                    f"ratio to MP2: {result['final_energy']/result['E_corr_MP2']:.4f}, "
                    f"iter: {len(result['E_corr_hist'])}, stop reason: {result['stop_reason']}")

                file_name = f"OVOS_convergence_{conv_energy_str}_{conv_grad_str}_keep_none.json"
                output_file = output_dir / file_name

                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2, cls=NumpyEncoder)
                print(f"  ✅ Saved to {output_file}")

            except Exception as e:
                print(f"  ⚠️  Error processing conv_energy={conv['conv_energy']}, conv_grad={conv['conv_grad']}, keep_track_max=none: {e}")



              





# ================================================================================================
# Helper function to save reference molecular data (MP2, OOMP2, FCI energies) to JSON for a given molecule and basis set
# ================================================================================================
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
    elif molecule == "O .0 .0  0.1173; H .0 0.7572 -0.4692; H .0 -0.7572 -0.4692":
        molecule_str = "H2O"
    elif molecule == "H .0 .0 .0; F .0 .0 0.917":
        molecule_str = "HF"
    elif molecule == "N .0 .0 .0; H .0 .0 1.012; H .0 0.926 -0.239; H .0 -0.926 -0.239":
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
    if basis == "6-31G" and molecule in ["HF", "H2O"]:
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
        "Li .0 .0 .0; Li .0 .0 2.673",  # Li2 molecule 0:1
        "H .0 .0 .0; F .0 .0 0.917",  # HF molecule    1:2
        "C .0 .0 .0; O .0 .0 1.128",    # CO molecule  2:3
        "O .0 .0  0.1173; H .0 0.7572 -0.4692; H .0 -0.7572 -0.4692",  # H2O equilibrium geometry
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
    # run_ovos_for_virtual_orbs(molecules[2], basis_sets[0], method="RHF", num_opt_virtual_orbs=6)

    for basis in basis_sets[1:2]:   # Done: 6-31G
        for molecule in molecules:  # Done: HF, H2O | Todo: Li2, CO, NH3
            for method in methods[0:2]:
                ovos_object(molecule, basis, method)

            try:
                save_molecule_reference_data(molecule, basis)
            except Exception as e:
                print(f"❌ Failed for {molecule}/{basis}: {e}\n")

            # After running ovos_object(), plot the results
            plot_OVOS_convergence_from_data(molecule, basis, methods=["RHF", "prev", "random"])
                # Plot convergence histories for all methods on the same plot for comparison
            plot_OVOS_convergence_histories(molecule, basis, methods=["RHF", "prev", "random"])

            # Plot MO coefficient convergence for all methods
            plot_OVOS_MO_coefficient_diff_MP2(molecule, basis)


    # Test convergence criteria and early stopping for "RHF" method
            # See if it is possbile converge closer to MP2, for different convergence criteria and early stopping parameters
        # For a certain molecule, basis, and virtual orbital count
            # Molecule: HF
            # Basis: 6-31G
            # Virtual orbital count: 3 (6 spin-orbitals, 50% of virtual space)
        # Analyze the convergence history and check how their energies evolve compared to the MP2 reference, and whether the early stopping criterion based on beating RHF is effective in practice.
            # Criteria
            # - Convergence in energy    = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
            # - Convergence in gradient  = [1e-2, 1e-4, 1e-6, 1e-8,  1e-10]
            # - Early stopping parameter = "none" (No early stopping)

    if False:  # Set to True to run convergence criteria testing
        conv_criteria = []
        # Do alot of points between 1e-4 and 1e-10 to see the trend
        for conv_energy in [1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6, 1e-7, 5e-7, 1e-8, 5e-8, 1e-9, 5e-9, 1e-10, 5e-10, 1e-11, 5e-11, 1e-12]:
            for conv_grad in [1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6, 1e-7, 5e-7, 1e-8, 5e-8, 1e-9, 5e-9, 1e-10, 5e-10, 1e-11, 5e-11, 1e-12]:
                conv_criteria.append({
                    'conv_energy': conv_energy,
                    'conv_grad': conv_grad,
                    'keep_track_max': "none"
                })
            # Total of 17x17 = 289 different convergence criteria combinations
            
            # For each run, save the convergence history, final energy, and number of iterations to a JSON file with the criteria in the filename for later analysis.
        for molecule in molecules[1:2]:
                    # Data
                ovos_convergence_criteria(molecule, basis_sets[0], method="RHF", conv_criteria=conv_criteria)
                    # Plot
                plot_OVOS_convergence_landscape(molecule, basis_sets[0], method="RHF", conv_criteria=conv_criteria)