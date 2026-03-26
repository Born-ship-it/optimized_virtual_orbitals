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
    folder = f"backup/data/{molecule}/{basis}/VQE/{method}/"
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
    
    file_name = f"backup/data/{molecule}/{basis}/VQE/VQE_HF_6-31G_results.json"
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

    file_name = f"backup/data/{molecule}/{basis}/VQE/{dist}/VQE_HF_6-31G_{dist}_results.json"
    if not os.path.exists(f"backup/data/{molecule}/{basis}/VQE/{dist}/"):
        os.makedirs(f"backup/data/{molecule}/{basis}/VQE/{dist}/")
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

    # print(f"VQE results write to {file_name} for dist {dist}: {data}")

def plot_vqe_results(molecule, basis, dist_list):
    # Plot the VQE results for each dist and method
    # We can use the data from the VQE results file we generated in the previous function
    # The file is like: "backup/data/HF/6-31G/VQE/VQE_HF_6-31G_results.json"
    file_name = f"backup/data/{molecule}/{basis}/VQE/VQE_HF_6-31G_results.json"
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
    plt.savefig(f"backup/data/{molecule}/{basis}/VQE/VQE_HF_6-31G_results.png")

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
    
    # Convert dist_list strings to floats for proper numeric plotting
    dist_list_float = [float(d) for d in dist_list]
    
    # Collect data organized by method
    data_by_method = {method: {'distances': [], 'energies': []} for method in methods}
    
    for dist in dist_list:
        file_name = f"backup/data/{molecule}/{basis}/VQE/{dist}/VQE_HF_6-31G_{dist}_results.json"
        try:
            with open(file_name, 'r') as f:
                data = json.load(f)
            
            for method in methods:
                if method in data:
                    initial_energy, final_energy = data[method][0], data[method][1]
                    data_by_method[method]['distances'].append(float(dist))
                    data_by_method[method]['energies'].append(final_energy)
        except FileNotFoundError:
            print(f"Warning: File not found {file_name}")
            continue
    
    # Plot each method as a continuous line (no markers)
    for method in methods:
        distances = data_by_method[method]['distances']
        energies = data_by_method[method]['energies']
        
        # Sort by distance for proper line connection
        sorted_data = sorted(zip(distances, energies))
        distances_sorted = [d[0] for d in sorted_data]
        energies_sorted = [e[1] for e in sorted_data]

        # Add a invisble point for zero distance if not already present
        if 0.0 not in distances_sorted:
            # Insert after the negative distances and before the positive distances
            insert_index = next((i for i, d in enumerate(distances_sorted) if d > 0), len(distances_sorted))
            distances_sorted.insert(insert_index, 0.0)
            energies_sorted.insert(insert_index, energies_sorted[insert_index])  # Use the energy of the
        
        ax.plot(distances_sorted, energies_sorted, 
               label=method_labels[method],
               color=colors[method],
               linestyle='-',
               linewidth = 2)

    # Set custom ticks at 0.5 Å intervals
    custom_ticks = np.arange(-0.5, 3.5, 0.5)
    matching_ticks = [tick for tick in custom_ticks 
                     if any(np.isclose(tick, d, atol=0.01) for d in distances_sorted)]
    ax.set_xticks(matching_ticks)
    
    # Set axis limits to span full range
    ax.set_xlim(min(distances_sorted), max(distances_sorted))
    ax.set_ylim(min(energies_sorted) - 0.5, max(energies_sorted) + 0.5)
    
    # Labels and formatting
    ax.set_xlabel("Interatomic Distance (Angstrom)", fontsize=12)
    ax.set_ylabel("Energy (Hartree)", fontsize=12)
    ax.set_title(f"Potential Energy Surface for {molecule} ({basis})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    
    # Save the plot
    output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_HF_6-31G_dist_results.png"
    plt.savefig(output_path, dpi=300)
    print(f"VQE dist results plot saved to {output_path}")
    

    # Insert lower right corner a mini plot showing the zoomed-in view around the equilibrium distance (around 2.0 Å)
    inset_ax = fig.add_axes([0.5, 0.2, 0.4, 0.4])  # [left, bottom, width, height]
    for method in methods:
        distances = data_by_method[method]['distances']
        energies = data_by_method[method]['energies']
        
        # Sort by distance for proper line connection
        sorted_data = sorted(zip(distances, energies))
        distances_sorted = [d[0] for d in sorted_data]
        energies_sorted = [e[1] for e in sorted_data]

        # Add a invisble point for zero distance if not already present
        if 0.0 not in distances_sorted:
            # Insert after the negative distances and before the positive distances
            insert_index = next((i for i, d in enumerate(distances_sorted) if d > 0), len(distances_sorted))
            distances_sorted.insert(insert_index, 0.0)
            energies_sorted.insert(insert_index, energies_sorted[insert_index])  # Use the energy of the
        
        inset_ax.plot(distances_sorted, energies_sorted, 
               label=method_labels[method],
               color=colors[method],
               linestyle='-',
               linewidth = 2)

    # Hhighlight the zoomed in area in the main plot with a rectangle
    ax.add_patch(plt.Rectangle((1.9, -102.5), 0.2, 0.5, fill=False, edgecolor='red', linewidth=1.5, linestyle='--'))
        # also in the mini plot
    inset_ax.add_patch(plt.Rectangle((1.9, -102.5), 0.2, 0.5, fill=False, edgecolor='red', linewidth=1.5, linestyle='--'))

    inset_ax.set_xlim(1.9-0.005, 2.1+0.005)
    inset_ax.set_ylim(-102.5-0.025, -102.0+0.025)
    inset_ax.set_xlabel("Distance (Å)", fontsize=10)
    inset_ax.set_ylabel("Energy (Hartree)", fontsize=10)
    inset_ax.set_title("Zoomed View", fontsize=10)
    inset_ax.grid(True, alpha=0.3)
    inset_ax.tick_params(labelsize=8)

    plt.tight_layout()
    
    # Save the plot
    output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_HF_6-31G_dist_results_w_zoom.png"
    plt.savefig(output_path, dpi=300)
    print(f"VQE dist results plot saved to {output_path}")
    

# Run HF 6-31G VQE results file generation
molecule = "HF"
basis = "6-31G"
method = "OVOS"
    # Get dist list from the folder
dist_list = gather_dist_lst(molecule, basis, method)
        # Set dist list with negatives floats first and then positive floats, and sorted by absolute value
dist_list = sorted(dist_list, key=lambda x: abs(4.0-float(x)))[::-1]
print(f"Dist list for {molecule} {basis} method {method}: {dist_list}")

for dist in dist_list:
        # For each dist, get seeds list and make VQE results file for that dist
    seeds_lst = gather_seeds_lst(molecule, basis, method, dist) # Get seeds list from the first dist, assuming it's the same for all dists
        # ... and make the VQE results file for that dist
    print(f"Seeds list for {molecule} {basis} dist {dist}: {seeds_lst}")
    make_vqe_dist_results_file(molecule, basis, dist, seeds_lst)

# Get the dist list again for full file generation
make_vqe_results_file(molecule, basis, dist_list, seeds_lst)

plot_vqe_dist_results(molecule, basis, dist_list)

