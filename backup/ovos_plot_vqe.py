"""
I intend to plot VQE results...

"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

def get_num_opt_virtual_orbitals(molecule, basis, dist):
    # Get the number of optimal "virtual" orbitals for this molecule and basis, which is the same for all dists and seeds
    # We can get it from the filename of the VQE results files, which is like:
    #       UPS_OVOS_HF_6-31G_"dist"_opt_num_4_False_"seed".json
    # We can extract the number from the filename by splitting the filename and getting the part after "opt_num_" and before "_False"
    folder = f"backup/data/{molecule}/{basis}/VQE/OVOS/{dist}/"
    files = os.listdir(folder)
    num_opt_virtual_orbitals = []
        # Be able to handle multiple files with different numbers of optimal virtual orbitals, and return a list of the unique numbers
    for file in files:
        if file.endswith(".json"):
            num_opt_virtual_orbitals.append(int(file.split("opt_num_")[1].split("_False")[0]))
        # Get the unique numbers of optimal virtual orbitals
    num_opt_virtual_orbitals = list(set(num_opt_virtual_orbitals))
    print(f"Number of optimal virtual orbitals for {molecule} {basis}: {num_opt_virtual_orbitals}")
    return num_opt_virtual_orbitals


def gather_vqe_results(molecule, basis, method, dist_list, seeds_lst, num_opt_virtual_orbitals):
    # First gather the data from the VQE results
    # For example, 
    #       Molecule: HF 
    #       Basis: 6-31G
    #       Folder: backup/data/HF/6-31G/VQE/"dist"/...
    # then for each "dist" folder, reference to a number...
    #     we have files like:
    #           UPS_OVOS_HF_6-31G_"dist"_opt_num_4_False_"seed".json
    #     here only "dist" will be the same for the folder 
    #     and "seed" will be different for each run
    data = {}
    for dist in dist_list:
        energies = []
        for seed in range(seeds_lst):
            filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method}_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbitals}_False_{seed}.json"
            with open(filename, 'r') as f:
                result = json.load(f)
                energies.append(result['final_energy'])
        data[dist] = energies

    return data
    
def gather_seeds_lst(molecule, basis, method, dist, num_opt_virtual_orbitals):
    # gather the seeds there is looked over by the names of the files in the folder
    # for example, in the folder backup/data/HF/6-31G/VQE/"dist"/, we have files like:
    #       UPS_OVOS_HF_6-31G_"dist"_opt_num_4_False_"seed".json
    # we can extract the seeds from the filenames
    # we can use os.listdir to list the files in the folder and then extract the seeds from the filenames
    folder = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/"
    files = os.listdir(folder)
    seeds_lst = []
    for file in files:
        # Do it from behind the last underscore and before the .json
        if file.endswith(".json"):
            seed = file.split("_")[-1].split(".")[0]
            seeds_lst.append(seed)
    return seeds_lst

def gather_dist_lst(molecule, basis, method, num_opt_virtual_orbitals):
    # gather the dist there is looked over by the names of the folders in the VQE folder
    # for example, in the folder backup/data/HF/6-31G/VQE/, we have folders like:
    #       "dist1", "dist2", ...
    # we can use os.listdir to list the folders in the VQE folder and then extract the dist from the folder names
    folder = f"backup/data/{molecule}/{basis}/VQE/{method}/"
    dist_list = []
        # Get the dist list for the correct number of optimal virtual orbitals
    for dist in os.listdir(folder):
        dist_folder = f"{folder}/{dist}/"
        if os.path.isdir(dist_folder):
            files = os.listdir(dist_folder)
            for file in files:
                if file.endswith(".json") and f"opt_num_{num_opt_virtual_orbitals}_" in file:
                    dist_list.append(dist)
                    break  # No need to check more files in this folder once we find a match
    return dist_list

def make_vqe_results_file(molecule, basis, dist_list, seeds_lst, num_opt_virtual_orbitals):
    # Run over each file in dist folder and get the lowest energy of those final_energy
    # Do so for OVOS, UHF, and UMP2 folder with data, and gather the data in a dictionary and save it as a json file for later plotting
    #      So for OVOS, UHF, and UMP2 as "keys"
    #        the files: "backup/data/HF/6-31G/VQE/"keys"/"dist"/UPS_OVOS_HF_6-31G_"dist"_opt_num_4_False_"seed".json
    #      we can extract the final_energy from each file and get the lowest energy for each "dist" and save it in a dictionary like:
    #      {
    #           "OVOS": {
    #               "dist1": lowest_energy,
    #               "dist2": lowest_energy,
    #               ...
    #           },
    #           "UHF": {
    #               "dist1": lowest_energy,
    #               "dist2": lowest_energy,
    #               ...
    #           },
    #           "UMP2": {
    #               "dist1": lowest_energy,
    #               "dist2": lowest_energy,
    #               ...
    #           }
    #      }
    for num_opt_virtual_orbital in [num_opt_virtual_orbitals]:
        data = {}
        for method in ["OVOS", "UHF", "UMP2"]:
            method_data = {}
            for dist in dist_list:
                energies = []
                energies_initial = []
                for seed in seeds_lst:
                    if method == "UMP2":
                        method_name = "UMP2_NO"
                    else:
                        method_name = method
                    filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method_name}_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbital}_False_{seed}.json"
                    with open(filename, 'r') as f:
                        result = json.load(f)
                        energies.append(result['final_energy'])
                        energies_initial.append(result['iter_energies'][0])
                energy_min = min(energies)
                method_data[dist] = [energies_initial[energies.index(energy_min)], energy_min]
            data[method] = method_data
        
        file_name = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_results_{num_opt_virtual_orbital}.json"
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)


def make_vqe_dist_results_file(molecule, basis, dist, seeds_lst, num_opt_virtual_orbitals):
    # Run over only one dist folder and get the lowest energy of those final_energy for each method
    # Do so for OVOS, UHF, and UMP2 folder with data, and gather the data in a dictionary and save it as a json file for later plotting
    #      So for OVOS, UHF, and UMP2 as "keys"
    #        the files: "backup/data/HF/6-31G/VQE/"keys"/"dist"/UPS_OVOS_HF_6-31G_"dist"_opt_num_4_False_"seed".json
    #      we can extract the final_energy from each file and get the lowest energy for each "dist" and save it in a dictionary like:
    #      {
    #           "OVOS": lowest_energy,
    #           "UHF": lowest_energy,
    #           "UMP2": lowest_energy
    #      }
    data = {}
    for method in ["OVOS", "UHF", "UMP2"]:
        energies = []
        energies_initial = []
        for seed in seeds_lst:
            if method == "UMP2":
                method_name = "UMP2_NO"
            else:
                method_name = method
            filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method_name}_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbitals}_False_{seed}.json"
            with open(filename, 'r') as f:
                result = json.load(f)
                energies.append(result['final_energy'])
                energies_initial.append(result['iter_energies'][0])

        # Get index of the lowest energy
            # Save the lowest energy and initial energy for this method and dist in the data dictionary
        energy_min = min(energies)
        data[method] = [energies_initial[energies.index(energy_min)], energy_min]

    file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbitals}.json"
    if not os.path.exists(f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/"):
        os.makedirs(f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/")
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

    # print(f"VQE results write to {file_name} for dist {dist}: {data}")

def plot_vqe_results(molecule, basis, dist_list):
    # Plot the VQE results for each dist and method
    # We can use the data from the VQE results file we generated in the previous function
    # The file is like: "backup/data/HF/6-31G/VQE/VQE_HF_6-31G_results.json"
    file_name = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_results.json"
    with open(file_name, 'r') as f:
        data = json.load(f)

    methods = ["OVOS", "UHF", "UMP2"]
    color = ['blue', 'purple', 'green']
    marker = ['D', 'X', 'P']

    # Make figure
    plt.figure(figsize=(10, 6))

    for method in methods:
        energies = [data[method][dist] for dist in dist_list]
        plt.plot(dist_list, energies, label=method, color=color[methods.index(method)], marker=marker[methods.index(method)], linestyle='-')

    plt.xlabel("Dist")
    plt.ylabel("Energy")
    plt.title(f"VQE Results for {molecule} {basis}")
    plt.suptitle("Potential Energy Surface")
    plt.legend()
    
    # Save the plot
    plt.savefig(f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_results.png")

def plot_vqe_dist_results(molecule, basis, dist_list):
    # Plot the VQE results for the dist we have already generated
    # Use make_vqe_dist_results_file to generate the results file for the dist we have already generated
    # The file is like: "backup/data/HF/6-31G/VQE/"dist"/VQE_HF_6-31G_"dist"_results.json
    # dist_list = gather_dist_lst(molecule, basis, "OVOS")
        # Except the last one, which might not be done...
    # dist_list = dist_list[:-1]

    # Make figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ["OVOS", "UHF", "UMP2"]
    method_labels = {"OVOS": "OVOS (75%)", "UHF": "UHF", "UMP2": "UMP2 Nat. Orbs"}
    colors = {'OVOS': 'blue', 'UHF': 'purple', 'UMP2': 'green'}
    marker = {'OVOS':'D', 'UHF': 'X', 'UMP2': 'P'}
    
    # Convert dist_list strings to floats for proper numeric plotting
    dist_list_float = [float(d) for d in dist_list]
    
    # Collect data organized by method
    data_by_method = {method: {'distances': [], 'energies': [], 'initial energies': [], 'UHF reference': [], 'RHF reference': [], 'nuclear repulsion': []} for method in methods}
    
    for dist in dist_list:
        file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results.json"
        try:
            with open(file_name, 'r') as f:
                data = json.load(f)
            
            for method in methods:
                if method in data:
                    initial_energy, final_energy = data[method][0], data[method][1]
                    data_by_method[method]['distances'].append(float(dist))
                    data_by_method[method]['energies'].append(final_energy)
                    data_by_method[method]['initial energies'].append(initial_energy)
                else:
                    print(f"Warning: Method {method} not found in data for dist {dist}")
        except FileNotFoundError:
            print(f"Warning: File not found {file_name}")
            continue

        file_name_uhf_ref = f"backup/data/{molecule}/6-31G/VQE/UHF/{dist}/UHF_{molecule}_6-31G_{dist}_reference_energy.txt"
        try:
            with open(file_name_uhf_ref, 'r') as f:
                uhf_reference_energy = float(f.read().strip())
                data_by_method['UHF']['UHF reference'].append(uhf_reference_energy)
        except FileNotFoundError:
            print(f"Warning: UHF reference energy file not found {file_name_uhf_ref}")
            data_by_method['UHF']['UHF reference'].append(None)  # Append None if reference energy is missing
    
        file_name_rhf_ref = f"backup/data/{molecule}/6-31G/VQE/UHF/{dist}/RHF_{molecule}_6-31G_{dist}_reference_energy.txt"
        try:
            with open(file_name_rhf_ref, 'r') as f:
                rhf_reference_energy = float(f.read().strip())
                data_by_method['UHF']['RHF reference'].append(rhf_reference_energy)
        except FileNotFoundError:
            print(f"Warning: RHF reference energy file not found {file_name_rhf_ref}")
            data_by_method['UHF']['RHF reference'].append(None)  # Append None if reference energy is missing

        file_name_nuclear_repulsion = f"backup/data/{molecule}/6-31G/VQE/UHF/{dist}/nuclear_repulsion_{molecule}_6-31G_{dist}_energy.txt"
        try:
            with open(file_name_nuclear_repulsion, 'r') as f:
                nuclear_repulsion_energy = float(f.read().strip())
                data_by_method['UHF']['nuclear repulsion'].append(nuclear_repulsion_energy)
        except FileNotFoundError:
            print(f"Warning: Nuclear repulsion energy file not found {file_name_nuclear_repulsion}")
            data_by_method['UHF']['nuclear repulsion'].append(None)  # Append None if nuclear repulsion energy is missing
    
    # Plot each method as a continuous line (no markers)
    data_by_method_for_plotting = {method: {'distances': [], "final_energies": [], "rhf_ref_energies": []} for method in methods}
    for method in methods:
        distances = data_by_method[method]['distances']
        energies = data_by_method[method]['energies']
        init_energies = data_by_method[method]['initial energies']
        uhf_ref_energies = data_by_method['UHF']['UHF reference']
        rhf_ref_energies = data_by_method['UHF']['RHF reference']
        nuclear_repulsion_energy = data_by_method['UHF']['nuclear repulsion']

        # Sort by distance for proper line connection
        sorted_data = sorted(zip(distances, energies, init_energies, uhf_ref_energies, rhf_ref_energies, nuclear_repulsion_energy))
        distances_sorted = [d[0] for d in sorted_data]
        energies_sorted = [e[1] for e in sorted_data]
        init_energies_sorted = [e[2] for e in sorted_data]
        uhf_ref_energies = [f[3] for f in sorted_data]
        rhf_ref_energies = [g[4] for g in sorted_data]
        nuclear_repulsion_energy = [n[5] for n in sorted_data]

        # Add a invisble point for zero distance if not already present
        if 0.0 not in distances_sorted and any(d < 0 for d in distances_sorted):
            print(f"Adding zero distance point for method {method} since negative distances are present but zero is missing.")
            # Insert after the negative distances and before the positive distances
            insert_index = next((i for i, d in enumerate(distances_sorted) if d > 0), len(distances_sorted))
            distances_sorted.insert(insert_index, 0.0)
            energies_sorted.insert(insert_index, energies_sorted[insert_index])  # Use the energy of the
            init_energies_sorted.insert(insert_index, init_energies_sorted[insert_index])  # Use the initial energy of the same point
            uhf_ref_energies.insert(insert_index, uhf_ref_energies[insert_index])  # Use the UHF reference energy of the same point
            rhf_ref_energies.insert(insert_index, rhf_ref_energies[insert_index])  # Use the RHF reference energy of the same point
            nuclear_repulsion_energy.insert(insert_index, nuclear_repulsion_energy[insert_index])  # Use the nuclear repulsion energy of the same point
        
        # Add nuclear repulsion energy to the energies_sorted
        energies_sorted = [e + n if e is not None and n is not None else e for e, n in zip(energies_sorted, nuclear_repulsion_energy)]
            # Add to energies_method for later 
        data_by_method_for_plotting[method]['distances'] = distances_sorted
        data_by_method_for_plotting[method]['final_energies'] = energies_sorted
        data_by_method_for_plotting[method]['rhf_ref_energies'] = rhf_ref_energies
        data_by_method_for_plotting[method]['UHF reference'] = uhf_ref_energies

        print(f"Length of distances_sorted for method {method}: {len(distances_sorted)}")
        print(f"Length of energies_sorted for method {method}: {len(energies_sorted)}")

        # Plot final energies as a continuous line
        ax.plot(distances_sorted, energies_sorted, 
               label=method_labels[method],
               color=colors[method],
               linestyle='-',
               linewidth = 2)
        # # Plot initial energies as a dashed line
        # ax.plot(distances_sorted, init_energies_sorted,
        #        label=f"{method_labels[method]} Initial",
        #        color=colors[method],
        #        linestyle='--',
        #        linewidth=1.5)
        

        if method == 'UHF':
            # # Also plot the UHF reference energies as a dashed line
            # if any(uhf_ref_energies):
            #     ax.plot(distances_sorted, uhf_ref_energies, 
            #            label="UHF Reference", 
            #            color=colors[method], 
            #            linestyle='--', 
            #            linewidth=1.5)
            # Also plot the RHF reference energies as a dotted line
            if any(rhf_ref_energies):
                ax.plot(distances_sorted, rhf_ref_energies, 
                       label="RHF Reference", 
                       color="red", 
                       linestyle='--', 
                       linewidth=1.5)

    # # Set custom ticks at 0.5 Å intervals
    custom_ticks = np.arange(-0.5, 3.5, 0.5)
    # custom_ticks = np.arange(0.7, 1.4, 0.1)
    matching_ticks = [tick for tick in custom_ticks 
                     if any(np.isclose(tick, d, atol=0.01) for d in distances_sorted)]
    ax.set_xticks(matching_ticks)
    
    # print(rhf_ref_energies)

    # Set axis limits to span full range
    min_custom_ticks = min(matching_ticks) if matching_ticks else min(distances_sorted)
    max_custom_ticks = max(matching_ticks) if matching_ticks else max(distances_sorted)
    ax.set_xlim(min(custom_ticks), max(custom_ticks))
    ax.set_ylim(min(energies_sorted) - 5, max(energies_sorted))  # Adjust y-axis limits based on energy range, with some padding
        # For matching the custom ticks, we can set the xlim to be a bit wider than the range of the custom ticks
    # ax.set_xlim(0.7, 1.3)
    # ax.set_ylim(-100.0, -99.8) # RHF
    # ax.set_ylim(-102.5, -102.0) # UHF
        
    
    # Labels and formatting
    ax.set_xlabel("Interatomic Distance (Angstrom)", fontsize=12)
    ax.set_ylabel("Energy (Hartree)", fontsize=12)
    ax.set_title(f"Potential Energy Surface for {molecule} ({basis})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    
    # Save the plot
    output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_dist_results.png"
    plt.savefig(output_path, dpi=300)
    print(f"VQE dist results plot saved to {output_path}")
    
    # Add a zoomed inset for the region around the equilibrium bond length (e.g., 0.7 to 1.3 Angstrom)
    inset_ax = ax.inset_axes([0.5, 0.5, 0.47, 0.47])  # [x0, y0, width, height]
    for method in methods:
        inset_ax.plot(data_by_method_for_plotting[method]['distances'], 
                data_by_method_for_plotting[method]['final_energies'],
                label=method_labels[method],
                color=colors[method],
                linestyle='-',
                linewidth = 2)

    inset_ax.plot(data_by_method_for_plotting['UHF']['distances'], 
                data_by_method_for_plotting['UHF']['rhf_ref_energies'],
                label="RHF Reference",
                color="red",
                linestyle='--',
                linewidth=1.5)

    inset_ax.set_xlim(0.7, 2)
    inset_ax.set_ylim(-100.0, -99.75)  # Adjust y-axis limits to zoom in on the region around the equilibrium bond length
    inset_ax.set_yticks(np.arange(-100.0, -99.8, 0.05))
    inset_ax.set_xticks(np.arange(0.7, 2.1, 0.2))
    inset_ax.grid(True, alpha=0.3)

        # Add gray in main plot between the two insets to indicate the zoomed area
    ax.axvspan(0.6, 2, color='gray', alpha=0.1, zorder=0)
    inset_ax.axvspan(0.6, 2, color='gray', alpha=0.1, zorder=0)
    inset_ax.axvspan(0.6, 2, color='gray', alpha=0.1, zorder=0)

    plt.tight_layout()
    
    # Save the plot
    output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_dist_results_w_zoom.png"
    plt.savefig(output_path, dpi=300)
    print(f"VQE dist results plot saved to {output_path}")

def plot_vqe_curve_results(molecule, basis, dist_list_, num_opt_virtual_orbitals):    
    # Make sure num_opt_virtual_orbitals is at least a list of one element, which is the number of optimal virtual orbitals for the OVOS method, and we can use it to get the dist_list for the correct number of optimal virtual orbitals
    if not isinstance(num_opt_virtual_orbitals, list):
        num_opt_virtual_orbitals = [num_opt_virtual_orbitals]
    dist_list = dist_list_[-1]
    if len(dist_list) > 1:
        dist_list_25 = dist_list_[0]
        print(dist_list_25)

    methods = ["OVOS", "UHF", "UMP2"]
    method_labels = {"OVOS": "OVOS (75%)", "UHF": "UHF", "UMP2": "UMP2 Nat. Orbs"}
    colors = {'OVOS': 'blue', 'UHF': 'purple', 'UMP2': 'green'}
    marker = {'OVOS':'D', 'UHF': 'X', 'UMP2': 'P'}
    
    # Convert dist_list strings to floats for proper numeric plotting
    dist_list_float = [float(d) for d in dist_list]
    
    # Collect data organized by method
    data_by_method = {method: {'distances': [], 'energies': [], 'initial energies': [], 'UHF reference': [], 'RHF reference': [], 'nuclear repulsion': []} for method in methods}
    
    for dist in dist_list:
        num_opt_virtual_orbital = num_opt_virtual_orbitals[-1]
        file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbital}.json"
        try:
            with open(file_name, 'r') as f:
                data = json.load(f)
            
            for method in methods:
                if method in data:
                    initial_energy, final_energy = data[method][0], data[method][1]
                    data_by_method[method]['distances'].append(float(dist))
                    data_by_method[method]['energies'].append(final_energy)
                    data_by_method[method]['initial energies'].append(initial_energy)
                    # print(f"Data for method {method} at dist {dist}: initial energy = {initial_energy}, final energy = {final_energy}")
                else:
                    print(f"Warning: Method {method} not found in data for dist {dist}")
        except FileNotFoundError:
            print(f"Warning: File not found {file_name}")
            continue

        file_name_uhf_ref = f"backup/data/{molecule}/6-31G/VQE/UHF/{dist}/UHF_{molecule}_6-31G_{dist}_reference_energy.txt"
        try:
            with open(file_name_uhf_ref, 'r') as f:
                uhf_reference_energy = float(f.read().strip())
                data_by_method['UHF']['UHF reference'].append(uhf_reference_energy)
        except FileNotFoundError:
            print(f"Warning: UHF reference energy file not found {file_name_uhf_ref}")
            data_by_method['UHF']['UHF reference'].append(None)  # Append None if reference energy is missing
    
        file_name_rhf_ref = f"backup/data/{molecule}/6-31G/VQE/UHF/{dist}/RHF_{molecule}_6-31G_{dist}_reference_energy.txt"
        try:
            with open(file_name_rhf_ref, 'r') as f:
                rhf_reference_energy = float(f.read().strip())
                data_by_method['UHF']['RHF reference'].append(rhf_reference_energy)
        except FileNotFoundError:
            print(f"Warning: RHF reference energy file not found {file_name_rhf_ref}")
            data_by_method['UHF']['RHF reference'].append(None)  # Append None if reference energy is missing

        file_name_nuclear_repulsion = f"backup/data/{molecule}/6-31G/VQE/UHF/{dist}/nuclear_repulsion_{molecule}_6-31G_{dist}_energy.txt"
        try:
            with open(file_name_nuclear_repulsion, 'r') as f:
                nuclear_repulsion_energy = float(f.read().strip())
                data_by_method['UHF']['nuclear repulsion'].append(nuclear_repulsion_energy)
        except FileNotFoundError:
            print(f"Warning: Nuclear repulsion energy file not found {file_name_nuclear_repulsion}")
            data_by_method['UHF']['nuclear repulsion'].append(None)  # Append None if nuclear repulsion energy is missing
    
    # Redo the data collection for plotting to ensure it's sorted by distance and includes the nuclear repulsion energy in the final energies
    data_by_method_for_plotting = {method: {'distances': [], "final_energies": [], "rhf_ref_energies": []} for method in methods}
    for method in methods:
        distances = data_by_method[method]['distances']
        energies = data_by_method[method]['energies']
        init_energies = data_by_method[method]['initial energies']
        uhf_ref_energies = data_by_method['UHF']['UHF reference']
        rhf_ref_energies = data_by_method['UHF']['RHF reference']
        nuclear_repulsion_energy = data_by_method['UHF']['nuclear repulsion']

        # Sort by distance for proper line connection
        sorted_data = sorted(zip(distances, energies, init_energies, uhf_ref_energies, rhf_ref_energies, nuclear_repulsion_energy))
        distances_sorted =          [d[0] for d in sorted_data]
        energies_sorted =           [e[1] for e in sorted_data]
        init_energies_sorted =      [e[2] for e in sorted_data]
        uhf_ref_energies =          [f[3] for f in sorted_data]
        rhf_ref_energies =          [g[4] for g in sorted_data]
        nuclear_repulsion_energy =  [n[5] for n in sorted_data]

        # Add a invisble point for zero distance if not already present
        if 0.0 not in distances_sorted and any(d < 0 for d in distances_sorted):
            print(f"Adding zero distance point for method {method} since negative distances are present but zero is missing.")
            # Insert after the negative distances and before the positive distances
            insert_index = next((i for i, d in enumerate(distances_sorted) if d > 0), len(distances_sorted))
            distances_sorted.insert(insert_index, 0.0)
            energies_sorted.insert(insert_index, energies_sorted[insert_index])  # Use the energy of the
            init_energies_sorted.insert(insert_index, init_energies_sorted[insert_index])  # Use the initial energy of the same point
            uhf_ref_energies.insert(insert_index, uhf_ref_energies[insert_index])  # Use the UHF reference energy of the same point
            rhf_ref_energies.insert(insert_index, rhf_ref_energies[insert_index])  # Use the RHF reference energy of the same point
            nuclear_repulsion_energy.insert(insert_index, nuclear_repulsion_energy[insert_index])  # Use the nuclear repulsion energy of the same point
        
        # Add nuclear repulsion energy to the energies_sorted
        energies_sorted = [e + n if e is not None and n is not None else e for e, n in zip(energies_sorted, nuclear_repulsion_energy)]
            # Add to energies_method for later 
        data_by_method_for_plotting[method]['distances'] = distances_sorted
        data_by_method_for_plotting[method]['final_energies'] = energies_sorted
        data_by_method_for_plotting[method]['rhf_ref_energies'] = rhf_ref_energies
        data_by_method_for_plotting[method]['UHF reference'] = uhf_ref_energies

        print(f"Length of distances_sorted for method {method}: {len(distances_sorted)}")
        print(f"Length of energies_sorted for method {method}: {len(energies_sorted)}")


    # Get the data from the other num_opt_virtual_orbitals if there are multiple and plot them as well, but with different different color as it is also OVOS
        # Done for dist_list_25 and num_opt_virtual_orbitals[0], which is 25% of the virtual orbitals, and we can compare it with the OVOS with 75% of the virtual orbitals
    if len(num_opt_virtual_orbitals) > 1:
        num_opt_virtual_orbital = num_opt_virtual_orbitals[0]
        data_by_method_25 = {method: {'distances': [], "final_energies": []} for method in methods}
        print(f"Dist list for 25% virt. orbs: {dist_list_25} for num_opt_virtual_orbital: {num_opt_virtual_orbital}")
        for dist in dist_list_25:
            file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbital}.json"
            try:
                with open(file_name, 'r') as f:
                    data = json.load(f)
                
                for method in methods:
                    if method in data:
                        initial_energy, final_energy = data[method][0], data[method][1]
                        data_by_method_25[method]['distances'].append(float(dist))
                        data_by_method_25[method]['final_energies'].append(final_energy)
                        print(f"Data for method {method} at dist {dist} for 25% virt. orbs: initial energy = {initial_energy}, final energy = {final_energy}")
                    else:
                        print(f"Warning: Method {method} not found in data for dist {dist}")
            except FileNotFoundError:
                print(f"Warning: File not found {file_name}")
                continue

        # Need to add nuclear repulsion energy to the final energies for the 25% virt. orbs data as well
        for method in methods:
            distances = data_by_method_25[method]['distances']
            energies = data_by_method_25[method]['final_energies']
            nuclear_repulsion_energy = data_by_method['UHF']['nuclear repulsion']  # Use the same nuclear repulsion energy as the other data since it's the same for the same dist

            # Add nuclear repulsion energy to the energies_sorted
                # Find only the correct nuclear repulsion energy for each distance and add it to the corresponding energy
            nuclear_repulsion_energy_for_dist = []
            for d in distances:
                if d in data_by_method['UHF']['distances']:
                    index = data_by_method['UHF']['distances'].index(d)
                    nuclear_repulsion_energy_for_dist.append(data_by_method['UHF']['nuclear repulsion'][index])
                else:
                    print(f"Warning: Distance {d} not found in UHF distances for nuclear repulsion energy. Appending None.")
                    nuclear_repulsion_energy_for_dist.append(None)
            energies = [e + n if e is not None and n is not None else e for e, n in zip(energies, nuclear_repulsion_energy_for_dist)]
                # Update the final energies with the ones that include nuclear repulsion energy
            data_by_method_25[method]['final_energies'] = energies


    # A plot that is just the zoomed in region around the equilibrium bond length (e.g., 0.7 to 1.3 Angstrom)
    plt.figure(figsize=(10, 6))
        # Line plot
    for method in methods:
        plt.plot(data_by_method_for_plotting[method]['distances'], 
                data_by_method_for_plotting[method]['final_energies'],
                color=colors[method],
                linestyle='-',
                linewidth = 2)

    if len(num_opt_virtual_orbitals) > 1:
        for method in ["OVOS"]:
            plt.plot(data_by_method_25[method]['distances'], 
                    data_by_method_25[method]['final_energies'],
                    color="orange",
                    linestyle='--',
                    linewidth = 2,
                    label=f"{method_labels[method]} (25% virt. orbs)")

        # RHF Reference line
    plt.plot(data_by_method_for_plotting['UHF']['distances'], 
                data_by_method_for_plotting['UHF']['rhf_ref_energies'],
                label="RHF Reference",
                color="red",
                linestyle='--',
                linewidth=1.5)

        # Point plot
    for method in methods:
        plt.scatter(data_by_method_for_plotting[method]['distances'], 
                    data_by_method_for_plotting[method]['final_energies'],
                    color=colors[method],
                    marker=marker[method],
                    label=f"{method_labels[method]} Points")

    if len(num_opt_virtual_orbitals) > 1:
        for method in ["OVOS"]:
            plt.scatter(data_by_method_25[method]['distances'], 
                        data_by_method_25[method]['final_energies'],
                        color="orange",
                        marker=marker[method],
                        label=f"{method_labels[method]} (25% virt. orbs) Points")

    plt.xlim(0.7, 2)
    plt.ylim(-100.0, -99.75)  # Adjust y-axis limits to zoom in on the region around the equilibrium bond length
    plt.xticks(np.arange(0.7, 2.1, 0.2))
    plt.yticks(np.arange(-100.0, -99.8, 0.05))
    plt.xlabel("Interatomic Distance (Angstrom)", fontsize=12)
    plt.ylabel("Energy (Hartree)", fontsize=12)
    plt.title(f"Zoomed Potential Energy Surface for {molecule} ({basis})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()



    # Save the plot
    output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_dist_results_zoom.png"
    plt.savefig(output_path, dpi=300)
    print(f"Zoomed VQE dist results plot saved to {output_path}")

    

if True:
    # Run HF 6-31G VQE results file generation
    molecule = "HF"
    basis = "6-31G"
    method = "OVOS" # Placeholder for getting dist and seed list

        # Get dist list from the folder
    # dist_list = gather_dist_lst(molecule, basis, method)
    dist_list = np.arange(0.7, 2.025, 0.025).round(5).tolist()
        # Get the number of optimal "virtual" orbitals for this molecule and basis, which is the same for all dists and seeds
    num_opt_virtual_orbitals = get_num_opt_virtual_orbitals(molecule, basis, dist_list[0])
            # Set dist list with negatives floats first and then positive floats, and sorted by absolute value
    # dist_list = sorted(dist_list, key=lambda x: abs(4.0-float(x)))[::-1]
    dist_list_save = []
    for num_opt_virtual_orbital in num_opt_virtual_orbitals:
        dist_list = gather_dist_lst(molecule, basis, method, num_opt_virtual_orbital)
            # Save dist_list
        dist_list_save.append(dist_list)
        print(f"Dist list for {molecule} {basis} method {method}: {dist_list}")
            # For each dist, get seeds list and make VQE results file for that dist
        if len(dist_list) < 30:
            seeds_lst = [9] # Only seed 9
        else:
            seeds_lst = gather_seeds_lst(molecule, basis, method, dist_list[0], num_opt_virtual_orbital) # Get seeds list from the first dist, assuming it's the same for all dists
        
        print(f"Seeds list for {molecule} {basis} method {method} dist {dist_list[0]}: {seeds_lst}")

        for dist in dist_list:
                # ... and make the VQE results file for that dist
            make_vqe_dist_results_file(molecule, basis, dist, seeds_lst, num_opt_virtual_orbital)

        # Get the dist list again for full file generation
        make_vqe_results_file(molecule, basis, dist_list, seeds_lst, num_opt_virtual_orbital)

    plot_vqe_curve_results(molecule, basis, dist_list_save, num_opt_virtual_orbitals)

if False:
    # Exame the spread of final energies for a dist in a method to see if there is a lot of variance in the final energies for different seeds, which might indicate convergence issues
    molecule = "HF"
    basis = "6-31G"
    methods = ["OVOS", "UHF", "UMP2"]

    # Get the number of optimal "virtual" orbitals for this molecule and basis, which is the same for all dists and seeds
    num_opt_virtual_orbitals = get_num_opt_virtual_orbitals(molecule, basis, dist=None)


    dist_list = gather_dist_lst(molecule, basis, methods[0], num_opt_virtual_orbitals)
    min_energy_seed_list = []
    for num_opt_virtual_orbital in num_opt_virtual_orbitals:
        for dist in dist_list:
            # dist = "1.3"

            seeds_lst = gather_seeds_lst(molecule, basis, methods[0], dist, num_opt_virtual_orbital)
            energies_method = {"OVOS": [], "UHF": [], "UMP2": []}
            for method in methods:
                for seed in seeds_lst:
                    str_method = method if method != "UMP2" else "UMP2_NO"
                    filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{str_method}_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbital}_False_{seed}.json"
                    with open(filename, 'r') as f:
                        result = json.load(f)
                        energies_method[method].append(result['final_energy'])

            print(f"Final energies for {molecule} {basis} dist {dist}:")
            for method in methods:
                # print(f"{method}: {energies_method[method]}")
                # print(f"{method} energy range: {min(energies_method[method])} to {max(energies_method[method])}, spread: {max(energies_method[method]) - min(energies_method[method])}")
                # print(f"{method} energy mean: {np.mean(energies_method[method])}, std: {np.std(energies_method[method])}")
                    # In which seed did we find the lowest energy for this method and dist?
                min_energy = min(energies_method[method])
                min_energy_seed = seeds_lst[energies_method[method].index(min_energy)]
                    # Add the seed with the lowest energy for this method and dist to the min_energy_seed_list
                min_energy_seed_list.append((method, dist, min_energy_seed, min_energy))
                    # Print the lowest energy and the seed for this method and dist
                print(f"{method} lowest energy: {min_energy} found in seed {min_energy_seed}")
            print()

    print("Summary of lowest energy seeds for each method and dist:")
        # Let us do some analysis on the min_energy_seed_list to see if there are any patterns in which seeds give the lowest energy for each method and dist
    for method in methods:
        print(f"Method: {method}")
        method_seeds = [entry for entry in min_energy_seed_list if entry[0] == method]
        for dist, seed, energy in sorted([(entry[1], entry[2], entry[3]) for entry in method_seeds], key=lambda x: float(x[0])):
            print(f"Dist: {dist}, Seed with lowest energy: {seed}, Lowest energy: {energy}")
        print()

# Examine the final energy of UHF vs. OVOS at 25% vs. 75%
if False:
    molecule = "HF"
    basis = "6-31G"
    dist_list = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    num_opt_virtual_orbitals = get_num_opt_virtual_orbitals(molecule, basis, dist=dist_list[0])
    seeds_lst = [9, 8] # Only seeed 9
    
    energies_UHF_25 = []
    energies_UHF_75 = []
    energies_OVOS_25 = []
    energies_OVOS_75 = []

    for dist in dist_list:
        for pro in ["25", "75"]:
            if pro == "25":
                i = 0
            else:
                i = -1

            filename_uhf = f"backup/data/{molecule}/{basis}/VQE/UHF/{dist}/UPS_UHF_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbitals[i]}_False_{seeds_lst[i]}.json"
            with open(filename_uhf, 'r') as f:
                result_uhf = json.load(f)
                if pro == "25":
                    energies_UHF_25.append(result_uhf['final_energy'])
                else:
                    energies_UHF_75.append(result_uhf['final_energy'])

            filename_ovos = f"backup/data/{molecule}/{basis}/VQE/OVOS/{dist}/UPS_OVOS_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbitals[i]}_False_{seeds_lst[i]}.json"
            with open(filename_ovos, 'r') as f:
                result_ovos = json.load(f)
                if pro == "25":
                    energies_OVOS_25.append(result_ovos['final_energy'])
                else:
                    energies_OVOS_75.append(result_ovos['final_energy'])

    # Energies for UHF 25%, UHF 75%, OVOS 25%, and OVOS 75% for each dist
    print("Dist | UHF 25% | UHF 75% | OVOS 25% | OVOS 75%")
    for i, dist in enumerate(dist_list):
        print(f"{dist} | {energies_UHF_25[i]} | {energies_UHF_75[i]} | {energies_OVOS_25[i]} | {energies_OVOS_75[i]}")
    
# Examine the Energy of OVOS at 25% vs. 75% for each dist to see if there is a consistent pattern in which one is lower than the other, which might indicate that one is more stable than the other
if False:
    molecule = "HF"
    basis = "6-31G"
    dist_list = [1.375] # Only dist 1.375
    seeds_lst = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

    energies_OVOS_75 = []

    for dist in dist_list:
        for seed in seeds_lst:
            filename_ovos = f"backup/data/{molecule}/{basis}/VQE/OVOS/{dist}/UPS_OVOS_{molecule}_{basis}_{dist}_opt_num_4_False_{seed}.json"
            with open(filename_ovos, 'r') as f:
                result_ovos = json.load(f)
                energies_OVOS_75.append(result_ovos['E_corr_OVOS'])
    print(f"OVOS 75% energies for dist {dist}: {energies_OVOS_75}")
    print(f"OVOS 75% energy range for dist {dist}: {min(energies_OVOS_75)} to {max(energies_OVOS_75)}, spread: {max(energies_OVOS_75) - min(energies_OVOS_75)}")
    # Result: They give the same OVOS 75% energy












def verify_h2_pes_sanity(dist_list, molecule="HH"):
    """Check if H2 PES has physically reasonable behavior."""
    from pyscf import gto, scf
    
    print("Distance | VQE (raw) | VQE + E_nuc | HF Total | VQE vs HF | Status")
    print("-" * 80)
    
    for dist in sorted(dist_list):
        dist_float = float(dist)
        
        # Get reference HF energy
        mol = gto.Mole()
        mol.atom = f"H 0 0 0; H 0 0 {dist_float:.5f}"
        mol.basis = '6-31G'
        mol.unit = 'Angstrom'
        mol.verbose = 0
        mol.build()
        
        mf = scf.RHF(mol)
        mf.kernel()
        
        hf_total = mf.e_tot
        e_nuc = mol.energy_nuc()
        hf_elec = hf_total - e_nuc
        
        # Load your VQE result
        file_name = f"backup/data/HH/6-31G/VQE/VQE_{molecule}_6-31G_results.json"
        try:
            with open(file_name, 'r') as f:
                data = json.load(f)
                if str(dist) in data.get("OVOS", {}):
                    vqe_raw = data["OVOS"][str(dist)][0]
                else:
                    vqe_raw = None
        except:
            vqe_raw = None
        
        if vqe_raw:
            # Try adding nuclear repulsion
            vqe_corrected = vqe_raw + e_nuc
            diff = hf_total - vqe_corrected
            
            # Check if it makes sense now
            if diff > 0 and diff < 0.1:  # VQE should be within ~0.1 Ha of HF
                status = "✓ REASONABLE"
            elif diff > 0:
                status = "⚠ VQE > HF (bad convergence?)"
            else:
                status = "✗ VQE < HF (impossible)"
            
            print(f"{dist_float:.2f} Å   | {vqe_raw:9.6f} | {vqe_corrected:11.6f} | {hf_total:8.6f} | {diff:9.6f} | {status}")
        else:
            print(f"{dist_float:.2f} Å   | {'MISSING':9s} |     -      | {hf_total:8.6f} |    -     | NO DATA")

# verify_h2_pes_sanity(gather_dist_lst("HH", "6-31G", "OVOS"), "HH")


