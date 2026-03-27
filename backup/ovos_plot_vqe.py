"""
I intend to plot VQE results...

"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os


def gather_vqe_results(molecule, basis, method, dist_list, seeds_lst):
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
            filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method}_{molecule}_{basis}_{dist}_opt_num_4_False_{seed}.json"
            with open(filename, 'r') as f:
                result = json.load(f)
                energies.append(result['final_energy'])
        data[dist] = energies

    return data
    
def gather_seeds_lst(molecule, basis, method, dist):
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

def gather_dist_lst(molecule, basis, method):
    # gather the dist there is looked over by the names of the folders in the VQE folder
    # for example, in the folder backup/data/HF/6-31G/VQE/, we have folders like:
    #       "dist1", "dist2", ...
    # we can use os.listdir to list the folders in the VQE folder and then extract the dist from the folder names
    folder = f"backup/data/{molecule}/{basis}/VQE/dist/"
    dist_list = os.listdir(folder)
    return dist_list

def make_vqe_results_file(molecule, basis, dist_list, seeds_lst):
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
                filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method_name}_{molecule}_{basis}_{dist}_opt_num_4_False_{seed}.json"
                with open(filename, 'r') as f:
                    result = json.load(f)
                    energies.append(result['final_energy'])
                    energies_initial.append(result['iter_energies'][0])
            energy_min = min(energies)
            method_data[dist] = [energies_initial[energies.index(energy_min)], energy_min]
        data[method] = method_data
    
    file_name = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_results.json"
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


def make_vqe_dist_results_file(molecule, basis, dist, seeds_lst):
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
            filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method_name}_{molecule}_{basis}_{dist}_opt_num_4_False_{seed}.json"
            with open(filename, 'r') as f:
                result = json.load(f)
                energies.append(result['final_energy'])
                energies_initial.append(result['iter_energies'][0])

        # Get index of the lowest energy
            # Save the lowest energy and initial energy for this method and dist in the data dictionary
        energy_min = min(energies)
        data[method] = [energies_initial[energies.index(energy_min)], energy_min]

    file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results.json"
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

def plot_vqe_curve_results(molecule, basis, dist_list):    
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

    # A plot that is just the zoomed in region around the equilibrium bond length (e.g., 0.7 to 1.3 Angstrom)
    plt.figure(figsize=(10, 6))
        # Line plot
    for method in methods:
        plt.plot(data_by_method_for_plotting[method]['distances'], 
                data_by_method_for_plotting[method]['final_energies'],
                color=colors[method],
                linestyle='-',
                linewidth = 2)

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
            # Set dist list with negatives floats first and then positive floats, and sorted by absolute value
    # dist_list = sorted(dist_list, key=lambda x: abs(4.0-float(x)))[::-1]
    print(f"Dist list for {molecule} {basis} method {method}: {dist_list}")

    for dist in dist_list:
            # For each dist, get seeds list and make VQE results file for that dist
        seeds_lst = gather_seeds_lst(molecule, basis, method, dist) # Get seeds list from the first dist, assuming it's the same for all dists
            # ... and make the VQE results file for that dist
        print(f"Seeds list for {molecule} {basis} dist {dist}: {seeds_lst}")
        make_vqe_dist_results_file(molecule, basis, dist, seeds_lst)

    # Get the dist list again for full file generation
    make_vqe_results_file(molecule, basis, dist_list, seeds_lst)

    plot_vqe_curve_results(molecule, basis, dist_list)
















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


