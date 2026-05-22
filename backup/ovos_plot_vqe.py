"""
I intend to plot VQE results...

"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os


def get_num_opt_virtual_orbitals(molecule, basis, dist, oo):
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
            if oo == False:  
                if f"_False" in file and not file.endswith("_False_True.json") and not file.endswith("_True_True.json"):  # Make sure to only get the files that are for the False case and not the True case
                    num_opt_virtual_orbitals.append(int(file.split("opt_num_")[1].split(f"_False")[0]))
            if oo == True:
                if f"_True" in file and not file.endswith("_False_True.json") and not file.endswith("_True_True.json"):  # Make sure to only get the files that are for the True case and not the False case
                    num_opt_virtual_orbitals.append(int(file.split("opt_num_")[1].split(f"_True")[0]))
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
    
def gather_seeds_lst(molecule, basis, method, dist, num_opt_virtual_orbitals, oo):
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
            if f"_{oo}_" in file:  # Make sure to only get the files that are for the correct oo case and not the other case
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

def make_vqe_results_file(molecule, basis, dist_list, seeds_lst, num_opt_virtual_orbitals, oo):
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
    if type(seeds_lst) is bool and seeds_lst == True:
        for oo in [oo]: #[True, False]:
            for num_opt_virtual_orbital in [num_opt_virtual_orbitals]:
                data = {}
                for method in ["OVOS", "UHF", "UMP2"]:
                    method_data = {}
                    for dist in dist_list:
                        energies = []
                        energies_initial = []
                        if method == "UMP2":
                            method_name = "UMP2_NO"
                        else:
                            method_name = method
                        filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method_name}_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbital}_{oo}_True.json"
                        with open(filename, 'r') as f:
                            result = json.load(f)
                            energies.append(result['final_energy'])
                            energies_initial.append(result['iter_energies'][0])
                        energy_min = min(energies)
                        seed_min = "True"
                        method_data[dist] = [energies_initial[energies.index(energy_min)], energy_min, seed_min]  # Save the initial energy, lowest energy, and seed for this method and dist in the method_data dictionary
                    data[method] = method_data
                
                file_name = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_results_{num_opt_virtual_orbital}_{oo}_True.json"
                with open(file_name, 'w') as f:
                    json.dump(data, f, indent=4)

    elif type(seeds_lst) is not bool:
        for oo in [oo]: #[True, False]:
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
                            filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method_name}_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbital}_{oo}_{seed}.json"
                            with open(filename, 'r') as f:
                                result = json.load(f)
                                energies.append(result['final_energy'])
                                energies_initial.append(result['iter_energies'][0])
                        energy_min = min(energies)
                        seed_min = seeds_lst[energies.index(energy_min)]
                        method_data[dist] = [energies_initial[energies.index(energy_min)], energy_min, seed_min]  # Save the initial energy, lowest energy, and seed for this method and dist in the method_data dictionary
                    data[method] = method_data
                
                file_name = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_results_{num_opt_virtual_orbital}_{oo}_False.json"
                with open(file_name, 'w') as f:
                    json.dump(data, f, indent=4)


def make_vqe_dist_results_file(molecule, basis, dist, seeds_lst, num_opt_virtual_orbitals, oo):
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
    # print(f"\n Gathering VQE results for {dist} with {num_opt_virtual_orbitals} for {seeds_lst}...")
    if type(seeds_lst) is bool and seeds_lst == True:
        for oo in [oo]: #[True, False]:
            # print(f"   Prev. oo = {oo}")
            data = {}
            for method in ["OVOS", "UHF", "UMP2"]:
                energies = []
                energies_initial = []
                energies_iterations = []
                mo_type_by_seed = []
                if method == "UMP2":
                    method_name = "UMP2_NO"
                else:
                    method_name = method
                filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method_name}_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbitals}_{oo}_True.json"
                try:
                    with open(filename, 'r') as f:
                        result = json.load(f)
                        energies.append(result['final_energy'])                 # Final energy
                        energies_initial.append(result['iter_energies'][0])     # Initial energy
                        energies_iterations.append(result['iterations'])         # Number of iterations
                        mo_type_by_seed.append(check_vqe_mo_restricted_or_unrestricted(filename))
                        # MO_type = ...
                except FileNotFoundError:
                    print(f"Warning: VQE result file not found {filename} for method {method}, dist {dist}, seed {seed}")

                # Get index of the lowest energy
                    # Save the lowest energy and initial energy for this method and dist in the data dictionary
                energy_min = min(energies)
                seed_min = "True"
                iteratoins_min = energies_iterations[energies.index(energy_min)]
                data[method] = [energies_initial[energies.index(energy_min)], energy_min, iteratoins_min, mo_type_by_seed[energies.index(energy_min)], seed_min]  
                # Save the initial energy, lowest energy, and MO type for this method and dist in the data dictionary

            file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbitals}_{oo}_True.json"
            if not os.path.exists(f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/"):
                os.makedirs(f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/")
            with open(file_name, 'w') as f:
                json.dump(data, f, indent=4)
        
    else:
        for oo in [oo]: #[True, False]:
            data = {}
            for method in ["OVOS", "UHF", "UMP2"]:
                energies = []
                energies_initial = []
                energies_iterations = []
                mo_type_by_seed = []
                for seed in seeds_lst:
                    if method == "UMP2":
                        method_name = "UMP2_NO"
                    else:
                        method_name = method
                    filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method_name}_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbitals}_{oo}_{seed}.json"
                    try:
                        with open(filename, 'r') as f:
                            result = json.load(f)
                            energies.append(result['final_energy'])                 # Final energy
                            energies_initial.append(result['iter_energies'][0])     # Initial energy
                            energies_iterations.append(result['iterations'])         # Number of iterations
                            mo_type_by_seed.append(check_vqe_mo_restricted_or_unrestricted(filename))
                            # MO_type = ...
                    except FileNotFoundError:
                        print(f"Warning: VQE result file not found {filename} for method {method}, dist {dist}, seed {seed}")

                # Get index of the lowest energy
                    # Save the lowest energy and initial energy for this method and dist in the data dictionary
                energy_min = min(energies)
                seed_min = seeds_lst[energies.index(energy_min)]
                iteratoins_min = energies_iterations[energies.index(energy_min)]
                data[method] = [energies_initial[energies.index(energy_min)], energy_min, iteratoins_min, mo_type_by_seed[energies.index(energy_min)], seed_min]  # Save the initial energy, lowest energy, and MO type for this method and dist in the data dictionary

            file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbitals}_{oo}_False.json"
            if not os.path.exists(f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/"):
                os.makedirs(f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/")
            with open(file_name, 'w') as f:
                json.dump(data, f, indent=4)

    # print(f"VQE results write to {file_name} for dist {dist}: {data}")

def check_vqe_mo_restricted_or_unrestricted(filename):
    # Check if the resulting MOs in the filename is of type, by getting the 
        # Alpha and beta MOs, are the same or different? 
        # "mo": [
        #           [
        #              [ ...
        #              ],
        #              ...
        #              [ ...
        #              ]
        #           ], # Alpha MOs
        #           [
        #              [ ...
        #              ],
        #              ...
        #              [ ...
        #              ]
        #           ] # Beta MOs
        #       ]

    # Open file and load json data
    with open(filename, 'r') as f:
        data = json.load(f) 

    # Get the MOs from the data
    mo_alpha = data['mo'][0]  # Alpha MOs
    mo_beta = data['mo'][1]   # Beta MOs

    # Check if the alpha and beta MOs are the same or different
    if mo_alpha == mo_beta:
        # print(f"The MOs in the file {filename} are restricted (same for alpha and beta).")
        return "restricted"
    else:
        # print(f"The MOs in the file {filename} are unrestricted (different for alpha and beta).")
        return "unrestricted"
                

def plot_vqe_curve_results(molecule, basis, dist_list_, num_opt_virtual_orbitals, plot_init, plot_prev, oo):    
    # Make sure num_opt_virtual_orbitals is at least a list of one element, which is the number of optimal virtual orbitals for the OVOS method, and we can use it to get the dist_list for the correct number of optimal virtual orbitals
    if not isinstance(num_opt_virtual_orbitals, list):
        num_opt_virtual_orbitals = [num_opt_virtual_orbitals]
    dist_list = dist_list_[-1]
    if len(dist_list) > 1:
        dist_list_25 = dist_list_[0]
        # print(dist_list_25)

    methods = ["OVOS", "UHF", "UMP2"]
    method_labels = {"OVOS": "OVOS", "UHF": "UHF", "UMP2": "UMP2"}
    colors = {'OVOS': 'blue', 'UHF': 'purple', 'UMP2': 'green'}
    marker = {'OVOS':'D', 'UHF': 'X', 'UMP2': 'P'}
    
    # Convert dist_list strings to floats for proper numeric plotting
    # print(dist_list, dist_list_)
    # if type(dist_list) is list:
    if plot_prev == False:
        dist_list_ = dist_list_[0]
    dist_list_float = [float(d) for d in dist_list_]
    dist_list = [dist_list_float]
    print(dist_list)
    # else:
    #     dist_list = dist_list_
    #     dist_list_float = dist_list
    
    # Collect data organized by method
    data_by_method = {method: {'distances': [], 'energies': [], 'initial energies': [], 'iterations': [], 'UHF reference': [], 'RHF reference': [], 'nuclear repulsion': []} for method in methods}
    
    # Collect if the MOs are restricted or unrestricted for each dist and method, and print it out
    mo_type_by_method_and_dist = {method: {} for method in methods}

    dist_list = dist_list[0]
    # print(f"\nDist list for plotting: {dist_list} for num_opt_virtual_orbitals: {num_opt_virtual_orbitals}")

    for dist in dist_list:
        num_opt_virtual_orbital = num_opt_virtual_orbitals[-1]
        
        if plot_prev == True:
            file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbital}_{oo}_True.json"
        else:
            file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbital}_{oo}_False.json"
        
        try:
            with open(file_name, 'r') as f:
                data = json.load(f)
            
            for method in methods:
                if method in data:
                    initial_energy, final_energy = data[method][0], data[method][1]
                    data_by_method[method]['distances'].append(float(dist))
                    data_by_method[method]['energies'].append(final_energy)
                    data_by_method[method]['initial energies'].append(initial_energy)
                    data_by_method[method]['iterations'].append(data[method][2])  # Save the number of iterations for this method and dist
                    mo_type_by_method_and_dist[method][dist] = data[method][3]  # Save the MO type for this method and dist
                    # print(f"Data for method {method} at dist {dist}: initial energy = {initial_energy}, final energy = {final_energy}")
                else:
                    print(f"Warning: Method {method} not found in data for dist {dist}")
        except FileNotFoundError:
            print(f"(1) Warning: File not found {file_name}")
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
        iterations = data_by_method[method]['iterations']
        uhf_ref_energies = data_by_method['UHF']['UHF reference']
        rhf_ref_energies = data_by_method['UHF']['RHF reference']
        nuclear_repulsion_energy = data_by_method['UHF']['nuclear repulsion']

        # Sort by distance for proper line connection
        sorted_data = sorted(zip(distances, energies, init_energies, uhf_ref_energies, rhf_ref_energies, nuclear_repulsion_energy, iterations))
        distances_sorted =          [d[0] for d in sorted_data]
        energies_sorted =           [e[1] for e in sorted_data]
        init_energies_sorted =      [e[2] for e in sorted_data]
        uhf_ref_energies =          [f[3] for f in sorted_data]
        rhf_ref_energies =          [g[4] for g in sorted_data]
        nuclear_repulsion_energy =  [n[5] for n in sorted_data]
        iterations_sorted =         [i[6] for i in sorted_data]

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
            iterations_sorted.insert(insert_index, iterations_sorted[insert_index])  # Use the number of iterations of the same point
        
        # Add nuclear repulsion energy to the energies_sorted
        energies_sorted = [e + n if e is not None and n is not None else e for e, n in zip(energies_sorted, nuclear_repulsion_energy)]
        energies_initial_sorted = [e + n if e is not None and n is not None else e for e, n in zip(init_energies_sorted, nuclear_repulsion_energy)]
            # Add to energies_method for later 
        data_by_method_for_plotting[method]['distances'] = distances_sorted
        data_by_method_for_plotting[method]['initial energies'] = energies_initial_sorted
        data_by_method_for_plotting[method]['final_energies'] = energies_sorted
        data_by_method_for_plotting[method]['iterations'] = iterations_sorted
        data_by_method_for_plotting[method]['rhf_ref_energies'] = rhf_ref_energies
        data_by_method_for_plotting[method]['UHF reference'] = uhf_ref_energies

        # print(f"Length of distances_sorted for method {method}: {len(distances_sorted)}")
        # print(f"Length of energies_sorted for method {method}: {len(energies_sorted)}")


    # # Get the data from the other num_opt_virtual_orbitals if there are multiple and plot them as well, but with different different color as it is also OVOS
    #     # Done for dist_list_25 and num_opt_virtual_orbitals[0], which is 25% of the virtual orbitals, and we can compare it with the OVOS with 75% of the virtual orbitals
    # if len(num_opt_virtual_orbitals) > 1:
    #     num_opt_virtual_orbital = num_opt_virtual_orbitals[0]
    #     data_by_method_25 = {method: {'distances': [], "final_energies": []} for method in methods}
    #     print(f"Dist list for 25% virt. orbs: {dist_list_25} for num_opt_virtual_orbital: {num_opt_virtual_orbital}")
    #     for dist in dist_list_25:
    #         file_name = f"backup/data/{molecule}/6-31G/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbital}.json"
    #         try:
    #             with open(file_name, 'r') as f:
    #                 data = json.load(f)
                
    #             for method in methods:
    #                 if method in data:
    #                     initial_energy, final_energy = data[method][0], data[method][1]
    #                     data_by_method_25[method]['distances'].append(float(dist))
    #                     data_by_method_25[method]['final_energies'].append(final_energy)
    #                     print(f"Data for method {method} at dist {dist} for 25% virt. orbs: initial energy = {initial_energy}, final energy = {final_energy}")
    #                 else:
    #                     print(f"Warning: Method {method} not found in data for dist {dist}")
    #         except FileNotFoundError:
    #             print(f"Warning: File not found {file_name}")
    #             continue

    #     # Need to add nuclear repulsion energy to the final energies for the 25% virt. orbs data as well
    #     for method in methods:
    #         distances = data_by_method_25[method]['distances']
    #         energies = data_by_method_25[method]['final_energies']
    #         energies_initial = data_by_method[method]['initial energies']  # Use the initial energies from the other data since it's the same for the same dist
    #         nuclear_repulsion_energy = data_by_method['UHF']['nuclear repulsion']  # Use the same nuclear repulsion energy as the other data since it's the same for the same dist

    #         # Add nuclear repulsion energy to the energies_sorted
    #             # Find only the correct nuclear repulsion energy for each distance and add it to the corresponding energy
    #         nuclear_repulsion_energy_for_dist = []
    #         for d in distances:
    #             if d in data_by_method['UHF']['distances']:
    #                 index = data_by_method['UHF']['distances'].index(d)
    #                 nuclear_repulsion_energy_for_dist.append(data_by_method['UHF']['nuclear repulsion'][index])
    #             else:
    #                 print(f"Warning: Distance {d} not found in UHF distances for nuclear repulsion energy. Appending None.")
    #                 nuclear_repulsion_energy_for_dist.append(None)
    #         energies = [e + n if e is not None and n is not None else e for e, n in zip(energies, nuclear_repulsion_energy_for_dist)]
    #         energies_initial = [e + n if e is not None and n is not None else e for e, n in zip(energies_initial, nuclear_repulsion_energy_for_dist)]
    #             # Update the final energies with the ones that include nuclear repulsion energy
    #         data_by_method_25[method]['final_energies'] = energies


    # A plot that is just the zoomed in region around the equilibrium bond length (e.g., 0.7 to 1.3 Angstrom)
    plt.figure(figsize=(12, 7))
        # Line plot
    for method in methods:
        plt.plot(data_by_method_for_plotting[method]['distances'], 
                data_by_method_for_plotting[method]['final_energies'],
                color=colors[method],
                linestyle='-',
                linewidth = 2)

        # RHF Reference line
    plt.plot(data_by_method_for_plotting['UHF']['distances'], 
                data_by_method_for_plotting['UHF']['UHF reference'],
                label="RHF Reference",
                color="red",
                linestyle='--',
                linewidth=1.5)
    # if molecule == "Li2":
    #     # Plot UHF reference line
    #     plt.plot(data_by_method_for_plotting['UHF']['distances'], 
    #                 data_by_method_for_plotting['UHF']['UHF reference'],
    #                 label="UHF Reference",
    #                 color="purple",
    #                 linestyle='--',
    #                 linewidth=1.5)

        # Point plot
    for method in methods:
        if plot_init:
            if molecule == "Li2":
                points_to_plot = [0, 5, 10, 15, 20, 25, 30, 35]  # Indices of the points to plot
            else: # 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9
                points_to_plot = [0, 8, 16, 24, 32, 40, 48]  # Indices of the points to plot for HF since there are less points

            plt.scatter([data_by_method_for_plotting[method]['distances'][i] for i in points_to_plot], 
                        [data_by_method_for_plotting[method]['final_energies'][i] for i in points_to_plot],
                        color=colors[method],
                        marker=marker[method],
                        label=f"tUPS {method_labels[method]}")
        else:
            plt.scatter(data_by_method_for_plotting[method]['distances'], 
                        data_by_method_for_plotting[method]['final_energies'],
                        color=colors[method],
                        marker=marker[method],
                        label=f"tUPS {method_labels[method]}")

    # if len(num_opt_virtual_orbitals) > 1:
    #     for method in ["OVOS"]:
    #         plt.scatter(data_by_method_25[method]['distances'], 
    #                     data_by_method_25[method]['final_energies'],
    #                     color="orange",
    #                     marker=marker[method],
    #                     label=f"{method_labels[method]} (25% virt. orbs) Points")
            
        # Initial energy points
    if plot_init:
        loc_text = ["right", "center", "left"]  # Location of the text for each method, corresponding to the order of methods
        for method in methods:
            # Only do the following points [0, 5, 10, ...]
                # Take the x,y data: data_by_method_for_plotting[method]['distances'], 
                                # data_by_method_for_plotting[method]['initial energies'],
            if molecule == "Li2":
                points_to_plot = [0, 5, 10, 15, 20, 25, 30, 35]  # Indices of the points to plot
            else: # 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9
                points_to_plot = [0, 8, 16, 24, 32, 40, 48]  # Indices of the points to plot for HF since there are less points

            data_to_plot_dist = [data_by_method_for_plotting[method]['distances'][i] for i in points_to_plot]
            data_to_plot_initial = [data_by_method_for_plotting[method]['initial energies'][i] for i in points_to_plot]
            data_to_plot_iterations = [data_by_method_for_plotting[method]['iterations'][i] for i in points_to_plot]
            
            # And seperate each method's point with a little distance s.t. OVOS in the middle on the point and UHF and UMP2 on the left and right of the point respectively, to make it easier to see the points and the text for each method
            if molecule == "Li2":
                dist_add_for_method = [-0.1, 0.0, 0.1] 
            else:
                dist_add_for_method = [-0.05, 0.0, 0.05]

            if method == "UHF":
                data_to_plot_dist = [d + dist_add_for_method[0] for d in data_to_plot_dist]
            elif method == "OVOS":
                data_to_plot_dist = [d + dist_add_for_method[1] for d in data_to_plot_dist]
            elif method == "UMP2":
                data_to_plot_dist = [d + dist_add_for_method[2] for d in data_to_plot_dist]


            plt.scatter(data_to_plot_dist,
                        data_to_plot_initial,
                        color=colors[method],
                        marker=marker[method],
                        alpha=0.5)

                # Text at initial energy points that shows the number of iterations it took to converge to the final energy for each method and dist, and make the text in the same color as the points for each method
            for i, dist in enumerate(data_to_plot_dist):

                if plot_prev == False:
                    y_add = 0.01
                else:
                    y_add = 0.001

                plt.text(dist, data_to_plot_initial[i]+y_add, f"{data_to_plot_iterations[i]} ", color=colors[method], fontsize=8, ha=loc_text[methods.index(method)], va='bottom')

        # Color intervals over the plot grey and white for every 2.25-2.75, 2.75-3.25, 3.25-3.75, 3.75-4.25, 4.25-4.75, 4.75-5.25, 5.25-5.75, 5.75-6.25 for Li2, and for every 0.9-1.1, 1.1-1.3 for HF
        if molecule == "Li2":
            intervals = [2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25]
        else:
            intervals = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
        for i in range(len(intervals)-1):
            if i % 2 == 0:
                plt.axvspan(intervals[i], intervals[i+1], color='grey', alpha=0.1)
            else:
                plt.axvspan(intervals[i], intervals[i+1], color='white', alpha=0.1)


    if molecule == "Li2":
        if plot_init:
            plt.xlim(2.3, 6.2)     
        else:
            plt.xlim(2.5, 6.0)
        if not plot_init:
            plt.ylim(-14.885, -14.80) 
    else: # HF, H2O
        if plot_init:
            plt.xlim(0.6, 2.0)
        else:
            plt.xlim(0.7, 2.0)
        plt.xticks(np.arange(0.7, 2.1, 0.2))
        # plt.ylim(-76,-75.6)
        # plt.ylim(-100.0, -99.75)  # Adjust y-axis limits to zoom in on the region around the equilibrium bond length
        # plt.yticks(np.arange(-100.0, -99.8, 0.05))
    plt.xlabel("Interatomic Distance (Angstrom)", fontsize=14)
    plt.ylabel("Energy (Hartree)", fontsize=14)

    if oo == False and plot_prev == True:    
        plt.title(f"Potential Energy Surface for {molecule}/{basis} w. Previous Thetas", fontsize=16)
    elif oo == True and plot_prev == True:
        plt.title(f"Potential Energy Surface for {molecule}/{basis} w. Previous Thetas and Optimized Orbitals", fontsize=16)
    elif oo == True and plot_prev == False:
        plt.title(f"Potential Energy Surface for {molecule}/{basis} w. Best of Random Thetas and Optimized Orbitals", fontsize=16)
    else:
        plt.title(f"Potential Energy Surface for {molecule}/{basis} w. Best of Random Thetas", fontsize=16)

    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=12)
    plt.tight_layout()

    # Print the MO type for each method and dist
    for method in methods:
        print(f"MO type for method {method}:")
        # gather list for printing ranges of dist with the same MO type
        mo_type_ranges = {}
        for dist, mo_type in mo_type_by_method_and_dist[method].items():
            if mo_type not in mo_type_ranges:
                mo_type_ranges[mo_type] = []
            mo_type_ranges[mo_type].append(float(dist))
        for mo_type, dist_values in mo_type_ranges.items():
            dist_values_sorted = sorted(dist_values)
            print(f"  MO type {mo_type} for distances: {dist_values_sorted[0]} to {dist_values_sorted[-1]} Angstrom")

    # Save the plot
    if plot_init:
        if plot_prev:
            output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_dist_results_{oo}_prev.png"
        else:
            output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_dist_results_{oo}.png"
    else:
        if plot_prev:
            output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_dist_results_zoom_{oo}_prev.png"
        else:
            output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_dist_results_zoom_{oo}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Zoomed VQE dist results plot saved to {output_path}")

    # # Make a new plot:
    #     # Plot the potential energy surface for this molecule and basis using the VQE results files for all dists and methods, and compare it with the reference energies (e.g., RHF reference energy) to see how well the VQE results match the reference energies across different dists, and if there are any trends in the VQE results compared to the reference energies as we change the dist
    #         # Include the initial energies and make a line that connect the initial down to the final and write the number of iterations it took for each method and dist, and also include the UHF reference energy as a horizontal line for comparison, and we can also include the RHF reference energy as another horizontal line for comparison, and we can also include the nuclear repulsion energy as another horizontal line for comparison, and we can make a zoomed in plot that focuses on the region around the equilibrium bond length to better see the differences between the methods in that region
    # num_opt_virtual_orbitals = num_opt_virtual_orbitals[0]
    # # Get the inital, final and iteration for the molecule...
    # lst_final_energies = {"OVOS": [], "UHF": [], "UMP2": []}
    # lst_initial_energies = {"OVOS": [], "UHF": [], "UMP2": []}
    # lst_iterations = {"OVOS": [], "UHF": [], "UMP2": []}
    #     # Open the files
    # for method in methods:
    #     for dist in dist_list:
    #         file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbital}.json"
    #         try:
    #             with open(file_name, 'r') as f:
    #                 data = json.load(f)
    #                 if method in data:
    #                     initial_energy, final_energy = data[method][0], data[method][1]
    #                     lst_final_energies[method].append(final_energy)
    #                     lst_initial_energies[method].append(initial_energy)
    #                     # Get the number of iterations from the corresponding VQE result file for this method and dist
    #                     seed = data[method][3]  # Get the seed for this method and dist

    #                     if method == "UMP2":
    #                         method_name = "UMP2_NO"
    #                     else:
    #                         method_name = method

    #                     filename_vqe_result = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method_name}_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbitals}_False_{seed}.json"
    #                     try:
    #                         with open(filename_vqe_result, 'r') as f_vqe:
    #                             result_vqe = json.load(f_vqe)
    #                             iterations = result_vqe['iterations']
    #                             lst_iterations[method].append(iterations)
    #                     except FileNotFoundError:
    #                         print(f"Warning: VQE result file not found {filename_vqe_result} for method {method}, dist {dist}, seed {seed}")
    #                         lst_iterations[method].append(None)  # Append None if VQE result file is missing
    #                 else:
    #                     print(f"Warning: Method {method} not found in data for dist {dist}")
    #         except FileNotFoundError:
    #             print(f"Warning: File not found {file_name}")
    #             continue

    # lst_distances = data_by_method_for_plotting['OVOS']['distances']  # Use the distances from the OVOS method for plotting since they should be the same for all methods

    # # Make a plot
    #     # The plot should resemble the one above, 
    #         # but with a lower alpha point for the initial point, and a text by this point in the color of the point that says the number of iterations it took to converge to the final energy
        
    # plt.figure(figsize=(10, 6))

    # for method in methods:
    #     # Use lst_final_energies[method] for the final energies and lst_initial_energies[method] for the initial energies, and lst_iterations[method] for the number of iterations
    #     plt.plot(lst_distances, lst_final_energies[method], color=colors[method], linestyle='-', linewidth=2, label=f"{method_labels[method]} Final")
    #     plt.scatter(lst_distances, lst_initial_energies[method], color=colors[method], marker=marker[method], label=f"{method_labels[method]} Initial", alpha=0.5)
    #     for i, dist in enumerate(lst_distances):
    #         plt.text(dist, lst_initial_energies[method][i], f"{lst_iterations[method][i]} iters", fontsize=8, color=colors[method], ha='center', va='bottom')

    # plt.xlabel("Interatomic Distance (Angstrom)", fontsize=12)
    # plt.ylabel("Energy (Hartree)", fontsize=12)
    # plt.title(f"VQE Convergence for {molecule} ({basis})", fontsize=14)
    # plt.grid(True, alpha=0.3)
    # plt.legend(loc="upper left", fontsize=10)
    # plt.tight_layout()

    # # SAve the plot
    # output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_dist_results_convergence.png"
    # plt.savefig(output_path, dpi=300)
    # print(f"VQE convergence plot saved to {output_path}")




def print_e_corr_ovos_vs_ump2(molecule, basis, dist, num_opt_virtual_orbitals, seeds_lst):
    # Get the final energy of OVOS and UMP2 for the given molecule, basis, dist, and num_opt_virtual_orbitals,
    # and print the correlation energy (E_corr = E_final - E_RHF_reference) for both methods for comparison
    # See it as a sanity check by comparing the correlation energy of OVOS and UMP2, and see if they are in the same ballpark, which can indicate if OVOS is capturing a similar amount of correlation energy as UMP2

    # Get seed from molecule/basis/dist/VQE_molecule_basis_dist_resutls_num_opt_virtual_orbitals.json file, which is the seed that gives the lowest final energy for OVOS for this molecule, basis, dist, and num_opt_virtual_orbitals
    # open the file
    file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbitals}.json"
        # e.g
        # {
        #     "OVOS": [
        #         -104.447912318759,
        #         -104.751277,
        #         "unrestricted",
        #         "20"
        #     ],
        #     "UHF": [
        #         -104.7247836893968,
        #         -104.740794,
        #         "unrestricted",
        #         "10"
        #     ],
        #     "UMP2": [
        #         -104.73767243374019,
        #         -104.767248,
        #         "restricted",
        #         "10"
        #     ]
        # }
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
            seed_ovos = data["OVOS"][4]  # Get the seed for OVOS that gives the lowest final energy
            seed_ump2 = data["UMP2"][4]  # Get the seed for UMP2 that gives the lowest final energy
            
    except FileNotFoundError:
        print(f"Warning: VQE dist results file not found {file_name}")
        seed_ovos = None
        seed_ump2 = None

    # Get the final E_corr energies for OVOS and UMP2 using the seeds
    corresponding_E_corr_OVOS = None
    corresponding_E_corr_UMP2 = None
    if seed_ovos is not None:
        filename_OVOS = f"backup/data/{molecule}/{basis}/VQE/OVOS/{dist}/UPS_OVOS_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbitals}_False_{seed_ovos}.json"
        filename_UMP2 = f"backup/data/{molecule}/{basis}/VQE/UMP2/{dist}/UPS_UMP2_NO_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbitals}_False_{seed_ump2}.json"
        try:
            with open(filename_OVOS, 'r') as f:
                result_OVOS = json.load(f)
                corresponding_E_corr_OVOS = result_OVOS['E_corr_OVOS']
        except FileNotFoundError:
            print(f"Warning: OVOS VQE result file not found {filename_OVOS}")
            corresponding_E_corr_OVOS = None
        try:
            with open(filename_UMP2, 'r') as f:
                result_UMP2 = json.load(f)
                E_UHF = result_UMP2['uhf_energy']
                E_UMP2 = result_UMP2['ump2_energy']
                E_UMP2_NO = result_UMP2['ump2_no_energy']
                corresponding_E_corr_UMP2 = E_UMP2 - E_UHF  # Correlation energy for UMP2 is the difference between UMP2 energy and UHF reference energy
        except FileNotFoundError:
            print(f"Warning: UMP2 VQE result file not found {filename_UMP2}")
            corresponding_E_corr_UMP2 = None

    ratio = None
    if corresponding_E_corr_OVOS is not None and corresponding_E_corr_UMP2 is not None and corresponding_E_corr_UMP2 != 0:
        ratio = corresponding_E_corr_OVOS / corresponding_E_corr_UMP2

    # Get the spread of the correlation energies for OVOS and UMP2 across different seeds for this molecule, basis, dist, and num_opt_virtual_orbitals, and print it out to see if there is a lot of variance in the correlation energies for different seeds, which might indicate convergence issues
    corresponding_E_corr_OVOS_lst = []
    corresponding_E_corr_UMP2_lst = []
    for seed in seeds_lst:  
        filename_OVOS = f"backup/data/{molecule}/{basis}/VQE/OVOS/{dist}/UPS_OVOS_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbitals}_False_{seed}.json"
        filename_UMP2 = f"backup/data/{molecule}/{basis}/VQE/UMP2/{dist}/UPS_UMP2_NO_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbitals}_False_{seed}.json"
        try:
            with open(filename_OVOS, 'r') as f:
                result_OVOS = json.load(f)
                corresponding_E_corr_OVOS_lst.append(result_OVOS['E_corr_OVOS'])
        except FileNotFoundError:
            print(f"Warning: OVOS VQE result file not found {filename_OVOS} for seed {seed}")
        try:
            with open(filename_UMP2, 'r') as f:
                result_UMP2 = json.load(f)
                E_UHF = result_UMP2['uhf_energy']
                E_UMP2 = result_UMP2['ump2_energy']
                E_UMP2_NO = result_UMP2['ump2_no_energy']
                corresponding_E_corr_UMP2_lst.append(E_UMP2 - E_UHF)  # Correlation energy for UMP2 is the difference between UMP2 energy and UHF reference energy
        except FileNotFoundError:
            print(f"Warning: UMP2 VQE result file not found {filename_UMP2} for seed {seed}")

    spread = None
    if corresponding_E_corr_OVOS_lst and corresponding_E_corr_UMP2_lst:
        spread_OVOS = max(corresponding_E_corr_OVOS_lst) - min(corresponding_E_corr_OVOS_lst)
        spread_UMP2 = max(corresponding_E_corr_UMP2_lst) - min(corresponding_E_corr_UMP2_lst)
        spread = (spread_OVOS, spread_UMP2)

    print(f"[{float(dist):.3f} Å] E_corr, OVOS: {corresponding_E_corr_OVOS:6.4f} Hartree ({seed_ovos:>3}), UMP2: {corresponding_E_corr_UMP2:6.4f} Hartree ({seed_ump2:>3}), Ratio: {ratio:.2f}" if ratio is not None else f"[{float(dist):.3f} Angstrom] Correlation energy for OVOS: {corresponding_E_corr_OVOS}, UMP2: {corresponding_E_corr_UMP2}, Ratio: undefined (UMP2 correlation energy is zero or missing)")    
    print(f"          Spread of E_corr across seeds for OVOS: {spread_OVOS:.2e} Hartree, UMP2: {spread_UMP2:.2e} Hartree" if spread is not None else "Spread of correlation energies across seeds could not be calculated due to missing data.")

def gather_and_print_vqe_final_energy_spread(molecule, basis, method, dist, num_opt_virtual_orbital, seeds_lst):
    # Gather the final energies for all seeds for this molecule, basis, method, dist, and num_opt_virtual_orbital, and then print the range and standard deviation of the final energies to see if there is a lot of variance in the final energies for different seeds, which might indicate convergence issues

    if method == "UMP2":
        method_name = "UMP2_NO"
    else:
        method_name = method

    final_energies = []
    for seed in seeds_lst:
        filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method_name}_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbital}_False_{seed}.json"
        try:
            with open(filename, 'r') as f:
                result = json.load(f)
                final_energies.append(result['final_energy'])
        except FileNotFoundError:
            print(f"Warning: VQE result file not found {filename} for method {method}, dist {dist}, seed {seed}")

    if final_energies:
        energy_range = max(final_energies) - min(final_energies)
        energy_std_dev = np.std(final_energies)
        print(f"[{float(dist):.3f} Å] VQE energy spread for {method}: Range = {energy_range:.6f} Hartree, Std Dev = {energy_std_dev:.6f} Hartree")
    else:
        print(f"No final energies found for {method} at dist {dist} to calculate spread.")


if False:
    # Plot the OO True but prev False ie 5 Random...
    for molecule in ["Li2", "HF", "H2O"]:
        # molecule = "Li2"
        basis = "6-31G"
        method = "OVOS" # Placeholder for getting dist and seed list
        for oo in [False, True]:  

            print(f"  \n Processing molecule {molecule} with basis {basis} and method {method} with optimized orbitals = {oo}...")

            # I know i have yet to run oo == True
                # Skip
            if oo == True and molecule in ["Li2", "H2O"]:
                print(f"Skipping molecule {molecule} with basis {basis} and method {method} with optimized orbitals = {oo} since I have not run it yet...")
                continue


                # Get dist list from the folder
            # dist_list = gather_dist_lst(molecule, basis, method)
            dist_list = [1.0] # For getting number of optimal virtual orbitals for this molecule and basis, which is the same for all dists and seeds, we can just use one dist, and we can use the same dist list for all num_opt_virtual_orbitals as well since they should be the same
                # Get the number of optimal "virtual" orbitals for this molecule and basis, which is the same for all dists and seeds
            num_opt_virtual_orbitals = get_num_opt_virtual_orbitals(molecule, basis, dist_list[0], False)
                    # Set dist list with negatives floats first and then positive floats, and sorted by absolute value

            # dist_list = sorted(dist_list, key=lambda x: abs(4.0-float(x)))[::-1]
            dist_list_save = []
            for num_opt_virtual_orbital in num_opt_virtual_orbitals:
                dist_list = gather_dist_lst(molecule, basis, method, num_opt_virtual_orbital)
                print(f"Dist list for {molecule} {basis} method {method} num_opt_virtual_orbital {num_opt_virtual_orbital}: {dist_list}")

                    # If the molecule is Li2, we only want to the range above 2.5 Angstrom, so we can filter the dist_list to only include dist that are above 2.5 Angstrom, and we can use this filtered dist_list for the rest of the code
                if molecule == "Li2":
                    dist_list = [dist for dist in dist_list if float(dist) >= 2.5]
                else:
                    dist_list = [dist for dist in dist_list if float(dist) >= 0.7]

                    # Save dist_list
                dist_list_save.append(dist_list)
                    # For each dist, get seeds list and make VQE results file for that dist
                if len(dist_list) < 3:
                    seeds_lst = [9] # Only seed 9 or 8
                else:
                    seeds_lst = gather_seeds_lst(molecule, basis, method, dist_list[0], num_opt_virtual_orbital, oo) # Get seeds list from the first dist, assuming it's the same for all dists
                    # Remove the last seed from the seeds_lst
                    seeds_lst = seeds_lst[:-1]
                
                print(f"Seeds list for {molecule} {basis} method {method} dist {dist_list[0]}: {seeds_lst}")

                for dist in dist_list:
                        # ... and make the VQE results file for that dist
                    make_vqe_dist_results_file(molecule, basis, dist, seeds_lst, num_opt_virtual_orbital, oo)

                # Get the dist list again for full file generation
                make_vqe_results_file(molecule, basis, dist_list, seeds_lst, num_opt_virtual_orbital, oo)

                # Check the correlation energy of OVOS vs. UMP2 for this molecule, basis, dist, and num_opt_virtual_orbitals as a sanity check
                # for dist in dist_list:
                #     print_e_corr_ovos_vs_ump2(molecule, basis, dist, num_opt_virtual_orbital, seeds_lst)

                # Check the spread of VQE final energies for all seeds for this molecule, basis, method, dist, and num_opt_virtual_orbitals to see if there are convergence issues
                # We can do this by gathering the final energies for all seeds for this molecule, basis, method, dist, and num_opt_virtual_orbitals, and then print the range and standard deviation of the final energies to see if there is a lot of variance in the final energies for different seeds, which might indicate convergence issues
                # for dist in dist_list:
                #     for method in ["OVOS", "UHF", "UMP2"]:
                #         gather_and_print_vqe_final_energy_spread(molecule, basis, method, dist, num_opt_virtual_orbital, seeds_lst)

            # for oo in [oo]: # [True, False]:
            # plot_vqe_curve_results(molecule, basis, dist_list_, num_opt_virtual_orbitals, plot_init, plot_prev, oo):    
            plot_vqe_curve_results(molecule, basis, dist_list_save, num_opt_virtual_orbitals, True, False, oo)
            plot_vqe_curve_results(molecule, basis, dist_list_save, num_opt_virtual_orbitals, False, False, oo)
        
if False:
    # Need to plot the VQE curve for one seed = "True", and both oo = True and False...
        # So we can see the difference in using prev. final thetas and keep trying to find best from random...
    
    for molecule in ["Li2", "HF", "H2O"]:
        basis = "6-31G"
        method = "OVOS" # Placeholder for getting dist and seed list
        
        for oo in [False, True]:

            print(f"  \n Processing molecule {molecule} with basis {basis} and method {method} with optimized orbitals = {oo}...")

            # Get dist list from the folder
            dist_list = [1.0] # For getting number of optimal virtual orbitals for this molecule and basis, which is the same for all dists and seeds, we can just use one dist, and we can use the same dist list for all num_opt_virtual_orbitals as well since they should be the same
            # Get the number of optimal "virtual" orbitals for this molecule and basis, which is the same for all dists and seeds
            num_opt_virtual_orbital = get_num_opt_virtual_orbitals(molecule, basis, dist_list[0], False)[0]
            
            # for num_opt_virtual_orbital in num_opt_virtual_orbitals:
            dist_list = gather_dist_lst(molecule, basis, method, num_opt_virtual_orbital)
            print(f"Dist list for {molecule} {basis} method {method} num_opt_virtual_orbital {num_opt_virtual_orbital}: {dist_list}")
            if molecule == "Li2":
                dist_list = [dist for dist in dist_list if float(dist) >= 2.5]
            else:
                dist_list = [dist for dist in dist_list if float(dist) >= 0.7]
                
            # Here i need to designate the seed to "True" as i do not use a specific seed but the prev.
            seed_lst = True

            for dist in dist_list:
                make_vqe_dist_results_file(molecule, basis, dist, seed_lst, num_opt_virtual_orbital, oo)

            make_vqe_results_file(molecule, basis, dist_list, seed_lst, num_opt_virtual_orbital, oo)

            print(f"\nFinished gathering VQE results for {molecule} {basis} for all dists and num_opt_virtual_orbitals, now plotting the curves...")    
            print(f"Number of optimal virtual orbitals: {num_opt_virtual_orbital}")
            print(f"Dist list for plotting: {dist_list}")

            # plot_vqe_curve_results(molecule, basis, dist_list_, num_opt_virtual_orbitals, plot_init, plot_prev, oo):    
            plot_vqe_curve_results(molecule, basis, dist_list, [num_opt_virtual_orbital], True, True, oo)
            plot_vqe_curve_results(molecule, basis, dist_list, [num_opt_virtual_orbital], False, True, oo)






def plot_vqe_curve_results_best_points(molecule, basis, dist_list, num_opt_virtual_orbitals, plot_init, oo):
        # I need to plot like plot_vqe_curve_results but the best point for each combo of oo and prev
        # and also the number of iterations to convergence for each method and dist at the initial energy point

        # Make sure num_opt_virtual_orbitals is at least a list of one element, which is the number of optimal virtual orbitals for the OVOS method, and we can use it to get the dist_list for the correct number of optimal virtual orbitals
    if not isinstance(num_opt_virtual_orbitals, list):
        num_opt_virtual_orbitals = [num_opt_virtual_orbitals]
    

    methods = ["OVOS", "UHF", "UMP2"]
    method_labels = {"OVOS": "OVOS", "UHF": "UHF", "UMP2": "UMP2"}
    colors = {'OVOS': 'blue', 'UHF': 'purple', 'UMP2': 'green'}
    marker = {'OVOS':'D', 'UHF': 'X', 'UMP2': 'P'}

    oo_prev_combos = {f"{oo}_False": [], f"{oo}_True": []}

    for plot_prev in [False, True]:

        # Skip True_False for all except HF
        if plot_prev == False and oo == True and molecule in ["Li2", "H2O"]:
            print(f"\n Skipping combination of plot_prev = {plot_prev} and oo = {oo} for molecule {molecule} since I have not run it yet...")
            continue
        else:
            print(f"\n Processing combination of plot_prev = {plot_prev} and oo = {oo} for molecule {molecule}...")

        # Convert dist_list strings to floats for proper numeric plotting
        # print(dist_list, dist_list_)
        # if type(dist_list) is list:
        # if plot_prev == False:
        #     dist_list_ = dist_list_[0]
        dist_list_float = [float(d) for d in dist_list]
        dist_list = [dist_list_float]
        # print(dist_list)
        
        # Collect data organized by method
        data_by_method = {method: {'distances': [], 'energies': [], 'initial energies': [], 'iterations': [], 'UHF reference': [], 'RHF reference': [], 'nuclear repulsion': []} for method in methods}
        
        # Collect if the MOs are restricted or unrestricted for each dist and method, and print it out
        mo_type_by_method_and_dist = {method: {} for method in methods}

        dist_list = dist_list[0]
        # print(f"\nDist list for plotting: {dist_list} for num_opt_virtual_orbitals: {num_opt_virtual_orbitals}")

        for dist in dist_list:
            num_opt_virtual_orbital = num_opt_virtual_orbitals[-1]
            
            if plot_prev == True:
                file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbital}_{oo}_True.json"
            else:
                file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbital}_{oo}_False.json"
            
            try:
                with open(file_name, 'r') as f:
                    data = json.load(f)
                
                for method in methods:
                    if method in data:
                        initial_energy, final_energy = data[method][0], data[method][1]
                        data_by_method[method]['distances'].append(float(dist))
                        data_by_method[method]['energies'].append(final_energy)
                        data_by_method[method]['initial energies'].append(initial_energy)
                        data_by_method[method]['iterations'].append(data[method][2])  # Save the number of iterations for this method and dist
                        mo_type_by_method_and_dist[method][dist] = data[method][3]  # Save the MO type for this method and dist
                        # print(f"Data for method {method} at dist {dist}: initial energy = {initial_energy}, final energy = {final_energy}")
                    else:
                        print(f"Warning: Method {method} not found in data for dist {dist}")
            except FileNotFoundError:
                print(f"(1) Warning: File not found {file_name}")
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
            iterations = data_by_method[method]['iterations']
            uhf_ref_energies = data_by_method['UHF']['UHF reference']
            rhf_ref_energies = data_by_method['UHF']['RHF reference']
            nuclear_repulsion_energy = data_by_method['UHF']['nuclear repulsion']

            # Sort by distance for proper line connection
            sorted_data = sorted(zip(distances, energies, init_energies, uhf_ref_energies, rhf_ref_energies, nuclear_repulsion_energy, iterations))
            distances_sorted =          [d[0] for d in sorted_data]
            energies_sorted =           [e[1] for e in sorted_data]
            init_energies_sorted =      [e[2] for e in sorted_data]
            uhf_ref_energies =          [f[3] for f in sorted_data]
            rhf_ref_energies =          [g[4] for g in sorted_data]
            nuclear_repulsion_energy =  [n[5] for n in sorted_data]
            iterations_sorted =         [i[6] for i in sorted_data]

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
                iterations_sorted.insert(insert_index, iterations_sorted[insert_index])  # Use the number of iterations of the same point
            
            # Add nuclear repulsion energy to the energies_sorted
            energies_sorted = [e + n if e is not None and n is not None else e for e, n in zip(energies_sorted, nuclear_repulsion_energy)]
            energies_initial_sorted = [e + n if e is not None and n is not None else e for e, n in zip(init_energies_sorted, nuclear_repulsion_energy)]
                # Add to energies_method for later 
            data_by_method_for_plotting[method]['distances'] = distances_sorted
            data_by_method_for_plotting[method]['initial energies'] = energies_initial_sorted
            data_by_method_for_plotting[method]['final_energies'] = energies_sorted
            data_by_method_for_plotting[method]['iterations'] = iterations_sorted
            data_by_method_for_plotting[method]['rhf_ref_energies'] = rhf_ref_energies
            data_by_method_for_plotting[method]['UHF reference'] = uhf_ref_energies

        # Save to correct oo_prev_combos
        combo_key = f"{oo}_{plot_prev}"
        oo_prev_combos[combo_key] = data_by_method_for_plotting

    # Get the best point at each interatomic distance of the oo/prev combos
    # For each distance, compare the final energies of the 4 combos and take the lowest one as the best point for that distance, and save it to a new dictionary best_points_by_dist that has the same structure as data_by_method_for_plotting but only includes the best point for each distance
    best_points_by_dist = {method: {'distances': [], "final_energies": [], "initial energies": [], "iterations": [], 'UHF reference': [], 'rhf_ref_energies': []} for method in methods}
    for method in methods:
        for dist in data_by_method_for_plotting[method]['distances']:
            best_energy = float('inf')
            best_initial_energy = None
            best_iterations = None
            best_uhf_ref_energy = None
            best_rhf_ref_energy = None
            for combo_key, combo_data in oo_prev_combos.items():
                if combo_key == "True_False" and molecule in ["Li2", "H2O"]:
                    continue
                if dist in combo_data[method]['distances']:
                    index = combo_data[method]['distances'].index(dist)
                    energy = combo_data[method]['final_energies'][index]
                    if energy is not None and energy < best_energy:
                        best_energy = energy
                        best_initial_energy = combo_data[method]['initial energies'][index]
                        best_iterations = combo_data[method]['iterations'][index]
                        best_uhf_ref_energy = combo_data[method]['UHF reference'][index]
                        best_rhf_ref_energy = combo_data[method]['rhf_ref_energies'][index]
            if best_energy != float('inf'):
                best_points_by_dist[method]['distances'].append(dist)
                best_points_by_dist[method]['final_energies'].append(best_energy)
                best_points_by_dist[method]['initial energies'].append(best_initial_energy)
                best_points_by_dist[method]['iterations'].append(best_iterations)
                best_points_by_dist[method]['UHF reference'].append(best_uhf_ref_energy)
                best_points_by_dist[method]['rhf_ref_energies'].append(best_rhf_ref_energy) 

    # Set the data for plotting to be the best points by dist
    data_by_method_for_plotting = best_points_by_dist

    # A plot that is just the zoomed in region around the equilibrium bond length (e.g., 0.7 to 1.3 Angstrom)
    plt.figure(figsize=(12, 7))
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
                label="UHF Reference",
                color="red",
                linestyle='-.',
                linewidth=1.5)

    #     # UHF Reference line
    # plt.plot(data_by_method_for_plotting['UHF']['distances'], 
    #             data_by_method_for_plotting['UHF']['UHF reference'],
    #             label="UMP2 Reference",
    #             color="red",
    #             linestyle='--',
    #             linewidth=1.5)

        # Point plot
    for method in methods:
        if plot_init:
            if molecule == "Li2":
                points_to_plot = [0, 5, 10, 15, 20, 25, 30, 35]  # Indices of the points to plot
            else: # 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9
                points_to_plot = [0, 8, 16, 24, 32, 40, 48]  # Indices of the points to plot for HF since there are less points

            plt.scatter([data_by_method_for_plotting[method]['distances'][i] for i in points_to_plot], 
                        [data_by_method_for_plotting[method]['final_energies'][i] for i in points_to_plot],
                        color=colors[method],
                        marker=marker[method],
                        label=f"tUPS {method_labels[method]}")
        else:
            plt.scatter(data_by_method_for_plotting[method]['distances'], 
                        data_by_method_for_plotting[method]['final_energies'],
                        color=colors[method],
                        marker=marker[method],
                        label=f"tUPS {method_labels[method]}")

        # Initial energy points
    if plot_init:
        loc_text = ["right", "center", "left"]  # Location of the text for each method, corresponding to the order of methods
        for method in methods:
            # Only do the following points [0, 5, 10, ...]
                # Take the x,y data: data_by_method_for_plotting[method]['distances'], 
                                # data_by_method_for_plotting[method]['initial energies'],
            if molecule == "Li2":
                points_to_plot = [0, 5, 10, 15, 20, 25, 30, 35]  # Indices of the points to plot
            else: # 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9
                points_to_plot = [0, 8, 16, 24, 32, 40, 48]  # Indices of the points to plot for HF since there are less points

            data_to_plot_dist = [data_by_method_for_plotting[method]['distances'][i] for i in points_to_plot]
            data_to_plot_initial = [data_by_method_for_plotting[method]['initial energies'][i] for i in points_to_plot]
            data_to_plot_iterations = [data_by_method_for_plotting[method]['iterations'][i] for i in points_to_plot]
            
            # And seperate each method's point with a little distance s.t. OVOS in the middle on the point and UHF and UMP2 on the left and right of the point respectively, to make it easier to see the points and the text for each method
            if molecule == "Li2":
                dist_add_for_method = [-0.1, 0.0, 0.1] 
            else:
                dist_add_for_method = [-0.05, 0.0, 0.05]

            if method == "UHF":
                data_to_plot_dist = [d + dist_add_for_method[0] for d in data_to_plot_dist]
            elif method == "OVOS":
                data_to_plot_dist = [d + dist_add_for_method[1] for d in data_to_plot_dist]
            elif method == "UMP2":
                data_to_plot_dist = [d + dist_add_for_method[2] for d in data_to_plot_dist]


            plt.scatter(data_to_plot_dist,
                        data_to_plot_initial,
                        color=colors[method],
                        marker=marker[method],
                        alpha=0.5)

                # Text at initial energy points that shows the number of iterations it took to converge to the final energy for each method and dist, and make the text in the same color as the points for each method
            for i, dist in enumerate(data_to_plot_dist):

                if plot_prev == False:
                    y_add = 0.01
                else:
                    y_add = 0.001

                plt.text(dist, data_to_plot_initial[i]+y_add, f"{data_to_plot_iterations[i]} ", color=colors[method], fontsize=8, ha=loc_text[methods.index(method)], va='bottom')

        # Color intervals over the plot grey and white for every 2.25-2.75, 2.75-3.25, 3.25-3.75, 3.75-4.25, 4.25-4.75, 4.75-5.25, 5.25-5.75, 5.75-6.25 for Li2, and for every 0.9-1.1, 1.1-1.3 for HF
        if molecule == "Li2":
            intervals = [2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25]
        else:
            intervals = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
        for i in range(len(intervals)-1):
            if i % 2 == 0:
                plt.axvspan(intervals[i], intervals[i+1], color='grey', alpha=0.1)
            else:
                plt.axvspan(intervals[i], intervals[i+1], color='white', alpha=0.1)


    if molecule == "Li2":
        if plot_init:
            plt.xlim(2.3, 6.2)     
        else:
            plt.xlim(2.5, 6.0)
        if not plot_init:
            plt.ylim(-14.885, -14.80) 
    else: # HF, H2O
        if plot_init:
            plt.xlim(0.6, 2.0)
        else:
            plt.xlim(0.7, 2.0)
        plt.xticks(np.arange(0.7, 2.1, 0.2))
        # plt.ylim(-76,-75.6)
        # plt.ylim(-100.0, -99.75)  # Adjust y-axis limits to zoom in on the region around the equilibrium bond length
        # plt.yticks(np.arange(-100.0, -99.8, 0.05))
    plt.xlabel("Interatomic Distance (Angstrom)", fontsize=14)
    plt.ylabel("Energy (Hartree)", fontsize=14)

    if oo == True:
        plt.title(f"Potential Energy Surface for {molecule}/{basis} w. Obital Optimization", fontsize=16)
    else:
        plt.title(f"Potential Energy Surface for {molecule}/{basis}", fontsize=16)

    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=12)
    plt.tight_layout()

    # Print the MO type for each method and dist
    for method in methods:
        print(f"MO type for method {method}:")
        # gather list for printing ranges of dist with the same MO type
        mo_type_ranges = {}
        for dist, mo_type in mo_type_by_method_and_dist[method].items():
            if mo_type not in mo_type_ranges:
                mo_type_ranges[mo_type] = []
            mo_type_ranges[mo_type].append(float(dist))
        for mo_type, dist_values in mo_type_ranges.items():
            dist_values_sorted = sorted(dist_values)
            print(f"  MO type {mo_type} for distances: {dist_values_sorted[0]} to {dist_values_sorted[-1]} Angstrom")

    # Save the plot
    if plot_init:
        output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_best_of_oo_{oo}.png"
    else:
        output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_best_of_zoom_oo_{oo}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Zoomed VQE dist results plot saved to {output_path}")





if True:
    # I need to plot like plot_vqe_curve_results but the best point for each combo of oo and prev
        # and also the number of iterations to convergence for each method and dist at the initial energy point


    # Need to write a new function that adapts from plot_vqe_curve_results...
    for molecule in ["Li2", "HF", "H2O"]:
        basis = "6-31G"
        method = "OVOS" # Placeholder for getting dist and seed list
        
        for oo in [True, False]:
            print(f"  \n Processing molecule {molecule} with basis {basis} and method {method} with oo {oo}...")

            # Get dist list from the folder
            dist_list = [1.0] # For getting number of optimal virtual orbitals for this molecule and basis, which is the same for all dists and seeds, we can just use one dist, and we can use the same dist list for all num_opt_virtual_orbitals as well since they should be the same
            # Get the number of optimal "virtual" orbitals for this molecule and basis, which is the same for all dists and seeds
            num_opt_virtual_orbital = get_num_opt_virtual_orbitals(molecule, basis, dist_list[0], False)[0]
            
            # for num_opt_virtual_orbital in num_opt_virtual_orbitals:
            dist_list = gather_dist_lst(molecule, basis, method, num_opt_virtual_orbital)
            print(f"Dist list for {molecule} {basis} method {method} num_opt_virtual_orbital {num_opt_virtual_orbital}: {dist_list}")
            if molecule == "Li2":
                dist_list = [dist for dist in dist_list if float(dist) >= 2.5]
            else:
                dist_list = [dist for dist in dist_list if float(dist) >= 0.7]
                

            # print(f"\nFinished gathering VQE results for {molecule} {basis} for all dists and num_opt_virtual_orbitals, now plotting the curves...")    
            # print(f"Number of optimal virtual orbitals: {num_opt_virtual_orbital}")
            # print(f"Dist list for plotting: {dist_list}")

            # plot_vqe_curve_results(molecule, basis, dist_list_, num_opt_virtual_orbitals, plot_init, plot_prev, oo):    
            plot_vqe_curve_results_best_points(molecule, basis, dist_list, [num_opt_virtual_orbital], True, oo)
            plot_vqe_curve_results_best_points(molecule, basis, dist_list, [num_opt_virtual_orbital], False, oo)






















plt.close('all')  # Close all open figures
assert len(plt.get_fignums()) == 0, "Some figures still open!"


























































def make_vqe_iterations_to_convergence_file(molecule, basis, dist_list, seed_lst, num_opt_virtual_orbital, oo):
    # Make a file that gathers the number of iterations to convergence for each method and dist for this molecule, basis, num_opt_virtual_orbital, and oo, and save it as a json file
    iterations_to_convergence = {}
    for dist in dist_list:
        iterations_to_convergence[dist] = {}
        for method in ["OVOS", "UHF", "UMP2"]:
            if method == "UMP2":
                method_name = "UMP2_NO"
            else:
                method_name = method

            if seed_lst != True:
                # We need the best seed ... and not the list for filename
                # need to look at .json 
                    # backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_results_{num_opt_virtual_orbital}_{oo}_False.json
                    # to get the best seed for this molecule, basis, num_opt_virtual_orbital, and oo

                best_seed = None
                best_final_energy = float('inf')
                filename_results = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_results_{num_opt_virtual_orbital}_{oo}_False.json"
                try:
                    with open(filename_results, 'r') as f:
                        results_data = json.load(f)
                        # Structure of results_data should be like:
                            # {
                            #     "OVOS": {
                            #         "0.7": [
                            #             -105.84748652645231,
                            #             -106.696834,
                            #             "1024"
                            #         ],
                            #         ...
                            #     },
                            #     "UHF": ...
                            #     "UMP2": ...
                            # }
                        if method in results_data and dist in results_data[method]:
                            seed_for_dist = results_data[method][dist][2]  # Get the seed for this method and dist
                            seed_lst = seed_for_dist
                        else:
                            print(f"Warning: Method {method} or dist {dist} not found in results data for {molecule} {basis} num_opt_virtual_orbital {num_opt_virtual_orbital} oo {oo}")
                except FileNotFoundError:
                    print(f"Warning: VQE results file not found {filename_results} for {molecule} {basis} num_opt_virtual_orbital {num_opt_virtual_orbital} oo {oo}")
                    seed_lst = None            

            filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_{method_name}_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbital}_{oo}_{seed_lst}.json"
            try:
                with open(filename, 'r') as f:
                    result = json.load(f)
                    iterations_to_convergence[dist][method] = result['iterations']
            except FileNotFoundError:
                print(f"Warning: VQE result file not found {filename} for method {method}, dist {dist}, seed {seed_lst}")
                iterations_to_convergence[dist][method] = None  # Set to None if VQE result file is missing

    if seed_lst != True:
        seed_lst = False # Convert seed_lst to False for the filename if it's not True, since we use "True" in the filename to indicate using previous thetas instead of specific seeds

    output_filename = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_iter_to_conv_oo_{oo}_prev_{seed_lst}.json"
    with open(output_filename, 'w') as f_out:
        json.dump(iterations_to_convergence, f_out, indent=4)
    print(f"Iterations to convergence data saved to {output_filename}")

def plot_iterations_to_convergence_statistics(molecule, basis, iterations_data_all):
    # Plot the statistics for the number of iterations to convergence for each method across different dists for each combination of oo and prev to see if there are any trends in the number of iterations to convergence based on using previous thetas or not, and based on using optimal orbitals or not
    
    # Make a plot
        # x-axis: combination of oo and prev (e.g., "OO True, Prev True", "OO True, Prev False", "OO False, Prev True", "OO False, Prev False")
            # Each tick should hold a point for each method (OVOS, UHF, UMP2) that represents the number of iterations to convergence for that method for this molecule and basis 
                # the point should show the average number of iterations to convergence across different dists for this method and combination of oo and prev, and the error bar should show the standard deviation of the number of iterations to convergence across different dists for this method and combination of oo and prev
        # y-axis: number of iterations to convergence
    
    combinations = []
    
    avg_iterations = {"OVOS": [], "UHF": [], "UMP2": []}
    std_iterations = {"OVOS": [], "UHF": [], "UMP2": []}
    median_iterations = {"OVOS": [], "UHF": [], "UMP2": []}
    methods_iterations = {"OVOS": [], "UHF": [], "UMP2": []}

    methods = ["OVOS", "UHF", "UMP2"]
    method_labels = {"OVOS": "OVOS", "UHF": "UHF", "UMP2": "UMP2"}
    colors = {'OVOS': 'blue', 'UHF': 'purple', 'UMP2': 'green'}
    marker = {'OVOS':'D', 'UHF': 'X', 'UMP2': 'P'}

    for (oo, prev), iterations_data in iterations_data_all.items():
        combination_label = f"{oo}, {prev}"
        combinations.append(combination_label)
        if iterations_data is not None:
            for method in methods:
                method_iterations = [iterations_data[dist][method] for dist in iterations_data if iterations_data[dist][method] is not None]
                if method_iterations:
                    methods_iterations[method].append(method_iterations)

                    avg_iterations[method].append(np.mean(method_iterations))
                    std_iterations[method].append(np.std(method_iterations))
                    median_iterations[method].append(np.median(method_iterations))

                    # print(f"Combination: {combination_label}, Method: {method}, Iterations to Convergence: {method_iterations}, Average: {avg_iterations[method][-1]:.2f}, Std Dev: {std_iterations[method][-1]:.2f}")
                else:
                    methods_iterations[method].append(None)

                    avg_iterations[method].append(None)
                    std_iterations[method].append(None)
                    median_iterations[method].append(None)
        else:
            for method in methods:
                methods_iterations[method].append(None)
                
                avg_iterations[method].append(None)
                std_iterations[method].append(None)
                median_iterations[method].append(None)
    
    x = np.arange(len(combinations))
    width = 0.2
    
    plt.figure(figsize=(12, 7))
    
    for i, method in enumerate(methods):
        # As avg_iterations[method] e.g = [None, None, 68.08333333333333, 92.13888888888889]
            # And i want to skip the None values for the method when plotting
            # I need to plot each separately
        positions = x + (i - 1) * width  # This offsets each method left/right
        boxplot_data = [methods_iterations[method][j] for j in range(len(combinations)) if avg_iterations[method][j] is not None]
        positions = [positions[j] for j in range(len(combinations)) if avg_iterations[method][j] is not None]
         
        print(f"Length of boxplot data for method {method}: {len(boxplot_data)}, Positions: {len(positions)}")

        plt.boxplot(boxplot_data,
                    positions=positions,
                    widths=width * 0.8,
                    patch_artist=True,
                    medianprops=dict(color='darkred', linewidth=2.5),
                    boxprops=dict(facecolor=colors[method], alpha=0.7),
                    whiskerprops=dict(color=colors[method], linewidth=1.5),
                    capprops=dict(color=colors[method], linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor=colors[method], markersize=5, alpha=0.5),
                    label=f"tUPS {method_labels[method]}"
            )
                        

        # for j, avg_iter in enumerate(avg_iterations[method]):
        #     # Skip the none values for the method when plotting
        #     if avg_iter is not None:
        #         # plt.bar(x[j] + i*width, avg_iter, width=width, yerr=std_iterations[method][j], capsize=5, label=f"tUPS {method_labels[method]}" if j==0 else "", color=colors[method], alpha=0.7)
                
            # plt.bar(x + i*width, avg_iterations[method], width=width, yerr=std_iterations[method], capsize=5, label="tUPS "+method_labels[method], color=colors[method], alpha=0.7)
        
    plt.xticks(x, combinations)
    # Set only lower y-axis limit to 0, since we cannot have negative iterations to convergence, but we can have a wide range of values for the upper y-axis limit depending on the molecule and basis, so we can set it to auto
    plt.ylim(bottom=0)

    plt.xlabel("Combination of Optimal Orbitals and Previous Thetas (oo, prev)", fontsize=14)
    plt.ylabel("Iterations to Convergence", fontsize=14)
    
    plt.title(f"VQE Iterations to Convergence for {molecule} ({basis})", fontsize=16)
    
    plt.xticks(x + width, combinations)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(loc="upper left", fontsize=12)
    plt.tight_layout()
    
    output_path = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_iterations_to_convergence_statistics.png"
    plt.savefig(output_path, dpi=300)
    print(f"VQE iterations to convergence statistics plot saved to {output_path}")

if False:
    # Gather data to plot iterations to convergence for 
        # Each: oo True/false and prev True/False, 
        # any trends in the number of iterations to convergence
    
    # Make a file for the number of iterations to convergence for each molecule
        # Need to make the files or do they already exist? 
    


    # check if file
        #     output_filename = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_iter_to_conv_oo_{oo}_prev_{seed_lst}.json"
            # exists for each combination of oo and prev, if not make the file by gathering the number of iterations to convergence for each method and dist for this molecule, basis, num_opt_virtual_orbital, and oo, and save it as a json file
    if False:
        for molecule in ["Li2", "HF", "H2O"]:
            basis = "6-31G"
            method = "OVOS" # Placeholder for getting dist and seed list

            for oo in [True, False]:
                for prev in [True, False]:

                    # Number of optimal virtual orbitals 
                    dist_list = [1.0] 
                    num_opt_virtual_orbital = get_num_opt_virtual_orbitals(molecule, basis, dist_list[0], False)

                    # I have yet to run True oo and False Prev
                    if oo == True and prev == False and molecule in ["Li2", "H2O"]:
                        print(f"Warning: No optimal virtual orbitals found for {molecule} {basis} with oo {oo}. Skipping this combination.")
                        continue
                    num_opt_virtual_orbital = num_opt_virtual_orbital[0]

                    # Geometries
                    dist_list = gather_dist_lst(molecule, basis, method, num_opt_virtual_orbital)
                    if molecule == "Li2":
                        dist_list = [dist for dist in dist_list if float(dist) >= 2.5]
                    else:
                        dist_list = [dist for dist in dist_list if float(dist) >= 0.7]
                    
                    # Seeds
                    if prev == True:
                        seed_lst = True
                    else:
                        seed_lst = gather_seeds_lst(molecule, basis, method, dist_list[0], num_opt_virtual_orbital, oo) # Get seeds list from the first dist, assuming it's the same for all dists
                        seed_lst = seed_lst[:-1]  # Remove the last seed from the seeds_lst
                    

                    output_filename = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_iter_to_conv_oo_{oo}_prev_{seed_lst}.json"
                    if not os.path.exists(output_filename):
                        make_vqe_iterations_to_convergence_file(molecule, basis, dist_list, seed_lst, num_opt_virtual_orbital, oo)
                    else:
                        print(f"File {output_filename} already exists, skipping data gathering for this combination of oo and prev.")

    # With a .json file for each
        # oo True/false and prev True/False 
        #w. structure like:
            # {
            #     "0.7": {
            #         "OVOS": 157,
            #         "UHF": 100,
            #         "UMP2": 49
            #     },
            #     ...
            # }
    
    # want to plot statistics for each molecule and basis
    for molecule in ["Li2", "HF", "H2O"]:
        basis = "6-31G"

        # Gather the data for all combinations of oo and prev for this molecule and basis
        iterations_data_all = {}
        for oo in [True, False]:
            for prev in [True, False]:
                filename = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_iter_to_conv_oo_{oo}_prev_{prev}.json"
                try:
                    with open(filename, 'r') as f:
                        iterations_data = json.load(f)
                        iterations_data_all[(oo, prev)] = iterations_data
                except FileNotFoundError:
                    print(f"Warning: Iterations to convergence file not found {filename} for {molecule} {basis} oo {oo} prev {prev}")
                    iterations_data_all[(oo, prev)] = None  # Set to None if file is missing
        # Now we have the iterations data for all combinations of oo and prev for this molecule and basis in iterations_data_all, we can plot the statistics for each method across different dists for each combination of oo and prev to see if there are any trends in the number of iterations to convergence based on using previous thetas or not, and based on using optimal orbitals or not
        plot_iterations_to_convergence_statistics(molecule, basis, iterations_data_all)


plt.close('all')  # Close all open figures
assert len(plt.get_fignums()) == 0, "Some figures still open!"




























# Reference-state overlap
# For small molecules (Li2, HF, H2O) and a small basis set (6-31G), compute the overlap of the OVOS-optimised reference state with the FCI ground state.
# At a specific geometry... and varying the number of optimised virtual orbitals N'_virt to see how the overlap changes as we include more optimised virtual orbitals in the reference state.
#       - compute the FCI ground state using PySCF fci.FCI().
#       - Compute the overlap F = |⟨Φ_ref|Ψ_FCI⟩|² for:
#           * |Φ_HF⟩ (standard HF reference)
#           * |Φ_OVOS⟩ (OVOS-optimised reference, varying N'_virt)
#       - Plot F_OVOS vs N'_virt/N_virt^max and compare to F_HF.
#       - Table: molecule | basis | N'_virt | F_HF | F_OVOS | ΔF
#
# Now the data structure for OVOS files is like:
    # backup/data/{molecule}/{basis}/OVOS/lst_MP2_different_virt_orbs_{init}.json
        # For init in ["prev", "random", "RHF"]

if False:
    import json
    import numpy as np
    from pyscf import gto, scf, fci, ao2mo
    from pyscf.fci.cistring import num_strings, str2addr
    from pyscf.fci.addons import transform_ci
    from pyscf.fci import direct_uhf

    # ------------------------------------------------------------
    # 1. Load OVOS data
    # ------------------------------------------------------------
    molecule = "HF"
    basis = "6-31G"
    init = "RHF"

    filename = f"backup/data/{molecule}/{basis}/OVOS/lst_MP2_OVOS_virt_orbs_{init}.json"
    with open(filename, 'r') as f:
        data = json.load(f)
        N_virt_opt_lst = data[1]
        energy_lst = data[0]
        mo_coefficients_lst = data[4]          # list of (2, nao, norb_full) arrays

    energy_final_lst = [energy[-1] for energy in energy_lst]
    diff_energy_final_lst = [energy[-1] - energy[0] for energy in energy_lst]

    # ------------------------------------------------------------
    # 2. Build molecule and run UHF
    # ------------------------------------------------------------
    mol_geo = "H .0 .0 .0; F .0 .0 0.917"
    mol = gto.Mole()
    mol.atom = mol_geo
    mol.basis = basis
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    n_alpha, n_beta = mol.nelec          # (5,5)
    nocc = n_alpha

    mf_uhf = scf.UHF(mol)
    mf_uhf.kernel()
    mo_alpha = mf_uhf.mo_coeff[0]        # (nao, nmo)
    mo_beta  = mf_uhf.mo_coeff[1]
    nmo = mo_alpha.shape[1]
    S = mol.intor('int1e_ovlp')

    # ------------------------------------------------------------
    # 3. Full CI in the UHF MO basis using direct_uhf
    # ------------------------------------------------------------
    # Using direct_uhf.kernel ensures the correct integral format is passed
    # Alternatively, the high-level FCI object can be used:
    cisolver = fci.FCI(mf_uhf)
    e_fci, ci_fci = cisolver.kernel()
    print(f"FCI energy: {e_fci:.8f} Hartree")
    print(f"FCI correlation energy: {e_fci - mf_uhf.e_tot:.6f} Hartree\n")

    # ------------------------------------------------------------
    # 4. HF reference CI vector in the full UHF basis (unit vector)
    # ------------------------------------------------------------
    n_str_full = num_strings(nmo, n_alpha)
    ref_hf_full = np.zeros((n_str_full, n_str_full))
    occ_mask = (1 << n_alpha) - 1
    addr_alpha = str2addr(nmo, n_alpha, occ_mask)
    addr_beta  = str2addr(nmo, n_beta, occ_mask)
    ref_hf_full[addr_alpha, addr_beta] = 1.0
    amp_hf_fci = np.dot(ci_fci.conj().ravel(), ref_hf_full.ravel())
    overlap_hf_fci = abs(amp_hf_fci)**2

    # ------------------------------------------------------------
    # 5. Process each OVOS orbital set
    # ------------------------------------------------------------
    overlap_ovos_fci = []
    overlap_ovos_hf  = []

    for N_virt, mo_full in zip(N_virt_opt_lst, mo_coefficients_lst):
        mo_full = np.asarray(mo_full)          # shape (2, nao, nmo_full)
        if mo_full.ndim == 3:
            mo_alpha_full = mo_full[0]
            mo_beta_full  = mo_full[1]
        else:
            mo_alpha_full = mo_full
            mo_beta_full  = mo_full

        # Truncate to occupied + first N_virt virtuals
        mo_alpha_trunc = mo_alpha_full[:, :nocc + N_virt]
        mo_beta_trunc  = mo_beta_full[:, :nocc + N_virt]
        norb_ovos = mo_alpha_trunc.shape[1]

        # Transformation matrices from UHF basis to truncated OVOS basis
        u_alpha = mo_alpha.T @ S @ mo_alpha_trunc   # (nmo, norb_ovos)
        u_beta  = mo_beta.T  @ S @ mo_beta_trunc

        # Transform FCI vector to OVOS basis
        ci_ovos = transform_ci(ci_fci, (n_alpha, n_beta), (u_alpha, u_beta))

        # Overlap with FCI (the OVOS reference is the first configuration)
        overlap_amplitude = ci_ovos[0, 0]
        overlap_ovos_fci.append(abs(overlap_amplitude)**2)

        # Overlap with HF determinant (transform HF reference to OVOS basis)
        ref_hf_ovos = transform_ci(ref_hf_full, (n_alpha, n_beta), (u_alpha, u_beta))
        overlap_hf = abs(np.dot(ref_hf_ovos.conj().ravel(), ci_ovos.ravel()))**2
        overlap_ovos_hf.append(overlap_hf)

    # ------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------
    print(f"\nResults for {molecule} / {basis} (init = {init})")
    print(f"HF–FCI overlap (reference): {overlap_hf_fci:.8f}\n")
    print(f"{'N_virt_opt':>12} | {'F_OVOS_FCI':>12} | {'F_OVOS_HF':>12} | {'Energy (Hartree)':>16} | {'ΔE from initial':>16}")
    print("-" * 80)
    for n, ov_fci, ov_hf, e, de in zip(N_virt_opt_lst, overlap_ovos_fci, overlap_ovos_hf, energy_final_lst, diff_energy_final_lst):
        print(f"{n//2:>12} | {ov_fci:>12.8f} | {ov_hf:>12.8f} | {e:>16.8f} | {de:>16.6f}")

    # ------------------------------------------------------------
    # 7. Detailed orbital contributions in the largest OVOS space
    # ------------------------------------------------------------
    max_idx = np.argmax(N_virt_opt_lst)   # index of the largest OVOS set
    N_max = N_virt_opt_lst[max_idx]
    mo_full = mo_coefficients_lst[max_idx]

    # Build the truncated orbitals for the largest set
    mo_full = np.asarray(mo_full)
    if mo_full.ndim == 3:
        mo_alpha_full = mo_full[0]
        mo_beta_full  = mo_full[1]
    else:
        mo_alpha_full = mo_full
        mo_beta_full  = mo_full

    mo_alpha_trunc = mo_alpha_full[:, :nocc + N_max]
    mo_beta_trunc  = mo_beta_full[:, :nocc + N_max]
    norb_trunc = mo_alpha_trunc.shape[1]

    # Transform the FCI vector to this largest truncated space
    u_alpha = mo_alpha.T @ S @ mo_alpha_trunc
    u_beta  = mo_beta.T  @ S @ mo_beta_trunc
    ci_ovos_max = transform_ci(ci_fci, (n_alpha, n_beta), (u_alpha, u_beta))

    # Normalize the transformed CI vector (fixes small numerical deviations)
    norm = np.linalg.norm(ci_ovos_max.ravel())
    ci_ovos_max /= norm

    # Create the FCI solver again (or reuse the existing `cisolver`)
    # to have access to the `make_rdm1` method.
    # We need to pass the new number of orbitals (`norb_trunc`) to the solver.
    cisolver_new = fci.FCI(mf_uhf, mo=(mo_alpha_trunc, mo_beta_trunc))

    # Build the 1‑RDM using the spin‑resolved method
    # Note: `make_rdm1s` returns (dm1_alpha, dm1_beta) directly
    rdm1_alpha, rdm1_beta = cisolver_new.make_rdm1s(ci_ovos_max, norb_trunc, (n_alpha, n_beta))

    # Natural orbital occupations (sum of alpha and beta)
    occ_alpha = np.diag(rdm1_alpha)
    occ_beta  = np.diag(rdm1_beta)
    occ_total = occ_alpha + occ_beta

    # Separate occupied (first n_alpha) and virtual (next N_max) parts
    occ_occupied = occ_total[:n_alpha]
    occ_virtual  = occ_total[n_alpha:n_alpha+N_max]

    print("\n--- Natural orbital occupations in the largest OVOS space ---")
    print(f"Orbital space: {norb_trunc} orbitals (occupied + {N_max//2} virtuals)")
    print("\nOccupied orbitals (HF reference = 2.0):")
    for i, occ in enumerate(occ_occupied):
        dev = occ - 2.0
        print(f"  occ-{i+1:2d} : total occ = {occ:.6f}  (Δ = {dev:+.6f})")

    print("\nVirtual orbitals (HF reference = 0.0):")
    for i, occ in enumerate(occ_virtual):
        print(f"  virt-{i+1:2d} : total occ = {occ:.6f}  (correlation contribution = {occ:.6f})")

    # Optionally, sort virtual orbitals by occupation (largest first)
    sorted_idx = np.argsort(occ_virtual)[::-1]
    print("\nVirtual orbitals ranked by occupation (most important first):")
    for rank, idx in enumerate(sorted_idx, 1):
        occ_val = occ_virtual[idx]
        print(f"  rank {rank:2d} : orbital {idx+1:2d}  total occ = {occ_val:.6f}")



if False:
    import json
    import numpy as np
    from pyscf import gto, scf, fci
    from pyscf.fci.cistring import make_strings

    # Initialize data structure to store results for all molecules
        # "basis": {
        #     "Molecule": {
        #         "HF_FCI_overlap": value,
        #         "N_virt_opt": {
        #             "F_OVOS_FCI": value,
        #             }
        #        }
        #    }

    data_basis_molecules = {"6-31G": {
        "Li2": {
            "HF_FCI_overlap": None,
            "N_virt_opt": [],
            "F_OVOS_FCI": [],
            "MP2_corr_energy": [],
        }, "HF": {
            "HF_FCI_overlap": None,
            "N_virt_opt": [],
            "F_OVOS_FCI": [],
            "MP2_corr_energy": [],
        }, "H2O": {
            "HF_FCI_overlap": None,
            "N_virt_opt": [],
            "F_OVOS_FCI": [],
            "MP2_corr_energy": [],
        }, "NH3": {
            "HF_FCI_overlap": None,
            "N_virt_opt": [],
            "F_OVOS_FCI": [],
            "MP2_corr_energy": [],
        }, "CO": {
            "HF_FCI_overlap": None,
            "N_virt_opt": [],
            "F_OVOS_FCI": [],
            "MP2_corr_energy": [],
        }}}

    for molecule in ["Li2", "HF", "H2O", "NH3"]:
        print(f"\nProcessing molecule {molecule} for reference state overlap analysis...")

        # ------------------------------------------------------------
        # 1. Load OVOS data (as in your snippet)
        # ------------------------------------------------------------
        molecule = molecule
        basis = "6-31G"
        init = "RHF"

        filename = f"backup/data/{molecule}/{basis}/OVOS/lst_MP2_OVOS_virt_orbs_{init}.json"
        with open(filename, 'r') as f:
            data = json.load(f)
            N_virt_opt_lst = data[1]
            energy_lst = data[0]
            mo_coefficients_lst = data[4]      # list of (2, nao, norb_full) arrays

        energy_final_lst = [energy[-1] for energy in energy_lst]
        diff_energy_final_lst = [energy[-1] - energy[0] for energy in energy_lst]

        # --------------------------------------------
        # Get mol_geo from molecule
        # --------------------------------------------
        if molecule == "Li2":
            mol_geo = "Li 0 0 0; Li 0 0 2.673"
        elif molecule == "HF":
            mol_geo = "H .0 .0 .0; F .0 .0 0.917"
        elif molecule == "H2O":
            mol_geo = 'O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692' 
        elif molecule == "NH3":
            mol_geo = 'N 0 0 0; H 0 0 1.012; H 0 0.935 -0.262; H 0 -0.935 -0.262'
        elif molecule == "CO":
            mol_geo = 'C 0 0 0; O 0 0 1.128'
        else:
            raise ValueError(f"Unknown molecule: {molecule}")

        # ------------------------------------------------------------
        # 2. Build molecule and run UHF (the basis for FCI)
        # ------------------------------------------------------------
        mol_geo = mol_geo
        mol = gto.Mole()
        mol.atom = mol_geo
        mol.basis = basis
        mol.unit = 'Angstrom'
        mol.spin = 0
        mol.charge = 0
        mol.symmetry = False
        mol.verbose = 0
        mol.build()

        mf_uhf = scf.UHF(mol)
        mf_uhf.kernel()
        mo_alpha_uhf = mf_uhf.mo_coeff[0]    # (nao, nmo)
        mo_beta_uhf  = mf_uhf.mo_coeff[1]
        nmo = mo_alpha_uhf.shape[1]
        S = mol.intor('int1e_ovlp')            # AO overlap matrix

        # ------------------------------------------------------------
        # 3. Full CI in the UHF MO basis
        # ------------------------------------------------------------
        try:
            cisolver = fci.FCI(mf_uhf)
            cisolver.verbose = 4
            e_fci, ci_fci = cisolver.kernel()
            print(f"FCI energy: {e_fci:.8f} Hartree")
            print(f"FCI correlation energy: {e_fci - mf_uhf.e_tot:.6f} Hartree\n")
        except Exception as e:
            print(f"Error running FCI for {molecule} with basis {basis}: {e}")
            continue

        # ------------------------------------------------------------
        # 4a. Overlap between the OVOS determinant and the FCI vector
        # ------------------------------------------------------------
        lst_overlaps = []
        lst_fidelities = []
        
        # Occupied orbitals in the OVOS determinant (first n_alpha / n_beta columns)
        n_alpha, n_beta = mol.nelec                # (5,5) for HF
        occ_ovos_a = list(range(n_alpha))
        occ_ovos_b = list(range(n_beta))

        # Generate all UHF occupation strings (bit‑strings) for alpha and beta
        strs_a = make_strings(range(nmo), n_alpha)   # list of ints
        strs_b = make_strings(range(nmo), n_beta)

        def occ_indices(bitstr, nmo):
            """Return list of orbital indices where bit is 1."""
            return [i for i in range(nmo) if (bitstr >> i) & 1]

        for mo_coeff in mo_coefficients_lst:
            # Pick the optimized OVOS MO coefficients (last entry)
            mo_ovos = mo_coeff          # shape (2, nao, nmo)
            C_ovos_a = np.asarray(mo_ovos[0])                      # (nao, nmo)
            C_ovos_b = np.asarray(mo_ovos[1])

            # Transformation from UHF basis to OVOS basis (unitary)
            U_a = C_ovos_a.T @ S @ mo_alpha_uhf        # (nmo, nmo)
            U_b = C_ovos_b.T @ S @ mo_beta_uhf

            # Precompute <UHF_det|OVOS_det> = det( U[occ_uhf, occ_ovos] ) for each string
            det_a = {}
            for bit_a in strs_a:
                occ_a = occ_indices(bit_a, nmo)
                submat = U_a[np.ix_(occ_a, occ_ovos_a)]
                det_a[bit_a] = np.linalg.det(submat)

            det_b = {}
            for bit_b in strs_b:
                occ_b = occ_indices(bit_b, nmo)
                submat = U_b[np.ix_(occ_b, occ_ovos_b)]
                det_b[bit_b] = np.linalg.det(submat)

            # Dot product with the FCI vector (real, so no complex conjugation)
            overlap = 0.0
            for i, bit_a in enumerate(strs_a):
                for j, bit_b in enumerate(strs_b):
                    amp = det_a[bit_a] * det_b[bit_b]   # <UHF_det|OVOS_det>
                    overlap += amp * ci_fci[i, j]

            fidelity = overlap**2

            lst_overlaps.append(overlap)
            lst_fidelities.append(fidelity)

        # ------------------------------------------------------------
        # 4b. Overlap between the UHF determinant and the FCI vector
        # ------------------------------------------------------------
        uhf_a_str = 0
        for i in range(n_alpha):
            uhf_a_str |= (1 << i)
        uhf_b_str = 0
        for i in range(n_beta):
            uhf_b_str |= (1 << i)

        idx_a = np.where(strs_a == uhf_a_str)[0][0]
        idx_b = np.where(strs_b == uhf_b_str)[0][0]
        
        overlap_hf = ci_fci[idx_a, idx_b]
        fidelity_hf = overlap_hf**2

        # ------------------------------------------------------------
        # 5. print results
        # ------------------------------------------------------------
        print(f"\nResults for {molecule} / {basis} (init = {init})")
        print(f"\n   UHF |FCI> fidelity: {fidelity_hf:.8f}\n")
        print(f"   {'N_virt_opt':>12} | {'F_OVOS_FCI':>12} | {'Energy (Hartree)':>16}")
        print("-" * 60)
        for n, ov_fci, e in zip(N_virt_opt_lst, lst_fidelities, energy_final_lst):
            print(f"   {n//2:>12} | {ov_fci:>12.8f} | {e:>16.8f}")

        # ------------------------------------------------------------
        # 6. Store results in the data structure
        # ------------------------------------------------------------
        data_basis_molecules[basis][molecule]["HF_FCI_overlap"] = fidelity_hf
        data_basis_molecules[basis][molecule]["N_virt_opt"] = N_virt_opt_lst
        data_basis_molecules[basis][molecule]["F_OVOS_FCI"] = lst_fidelities
        data_basis_molecules[basis][molecule]["MP2_corr_energy"] = energy_final_lst

    # Save the data structure to a JSON file for later analysis
    output_filename = f"backup/data/reference_state_overlaps.json"
    with open(output_filename, 'w') as f:
        json.dump(data_basis_molecules, f, indent=4)
    print(f"\nReference state overlap data saved to {output_filename}")

if False:
    # Get the file and data therefrom
    output_filename = f"backup/data/reference_state_overlaps.json"
    with open(output_filename, 'r') as f:
        data_basis_molecules = json.load(f)
    print(f"\nReference state overlap data loaded from {output_filename}")

    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    import matplotlib.pyplot as plt
    import numpy as np

    basis = "6-31G"
    data_mol = data_basis_molecules[basis]

    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_mol)))

    for (mol_name, mol_data), color in zip(data_mol.items(), colors):
        if not mol_data["N_virt_opt"]:   # skip if no data (e.g., CO)
            continue
        N = mol_data["N_virt_opt"]
        fid = mol_data["F_OVOS_FCI"]
        uhf_fid = mol_data["HF_FCI_overlap"]
        mp2_energy = mol_data["MP2_corr_energy"]
        
        ax1.plot(N, fid, 'o-', color=color, label=f"{mol_name}")
        ax1.axhline(y=uhf_fid, linestyle='--', color=color, alpha=0.6,
                    label="")
        
    # Set label for legend entry for UHF reference fidelity
    ax1.axhline(y=0, linestyle='--', color="black", alpha=0.6,
                    label="UHF")

    # Set the y-axis limit to [0, 1] since fidelity cannot exceed 1
    ax1.set_ylim(0.9, 1.0)

    ax1.set_xlabel("Number of optimised virtual orbitals", fontsize=14)
    ax1.set_ylabel("Fidelity with FCI ground state", fontsize=14)
    ax1.set_title(f"Overlap of OVOS determinant with FCI ({basis} basis)", fontsize=16)
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"backup/data/ovos_fidelity_mp2_{basis}.png", dpi=300, bbox_inches='tight')

    # Save the plot to png file
    output_plot_filename = f"backup/data/ovos_fidelity_mp2_{basis}.png"
    plt.savefig(output_plot_filename, dpi=300, bbox_inches='tight')
    print(f"OVOS fidelity and MP2 energy plot saved to {output_plot_filename}")
    plt.close()  # Close the figure to free memory















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


