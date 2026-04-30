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
    if type(seeds_lst) is bool and seeds_lst == True:
        for oo in [False]: #[True, False]:
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

    else:
        for oo in [False]: #[True, False]:
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
                
                file_name = f"backup/data/{molecule}/{basis}/VQE/VQE_{molecule}_6-31G_results_{num_opt_virtual_orbital}_{oo}_True.json"
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
    # print(f"\n Gathering VQE results for {dist} with {num_opt_virtual_orbitals} for {seeds_lst}...")
    if type(seeds_lst) is bool and seeds_lst == True:
        for oo in [False]: #[True, False]:
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
        for oo in [False]: #[True, False]:
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

            file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbitals}_{oo}.json"
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
            file_name = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/VQE_{molecule}_6-31G_{dist}_results_{num_opt_virtual_orbital}_{oo}.json"
        
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
    plt.xlabel("Interatomic Distance (Angstrom)", fontsize=12)
    plt.ylabel("Energy (Hartree)", fontsize=12)
    plt.title(f"Potential Energy Surface for {molecule} ({basis})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=10)
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
    for molecule in ["Li2", "HF", "H2O"]:
        # molecule = "Li2"
        basis = "6-31G"
        method = "OVOS" # Placeholder for getting dist and seed list

            # Get dist list from the folder
        # dist_list = gather_dist_lst(molecule, basis, method)
        dist_list = [1.0] # For getting number of optimal virtual orbitals for this molecule and basis, which is the same for all dists and seeds, we can just use one dist, and we can use the same dist list for all num_opt_virtual_orbitals as well since they should be the same
            # Get the number of optimal "virtual" orbitals for this molecule and basis, which is the same for all dists and seeds
        num_opt_virtual_orbitals = get_num_opt_virtual_orbitals(molecule, basis, dist_list[0])
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
            if len(dist_list) < 5:
                seeds_lst = [9] # Only seed 9 or 8
            else:
                seeds_lst = gather_seeds_lst(molecule, basis, method, dist_list[0], num_opt_virtual_orbital) # Get seeds list from the first dist, assuming it's the same for all dists
                # Remove the last seed from the seeds_lst
                seeds_lst = seeds_lst[:-1]
            
            print(f"Seeds list for {molecule} {basis} method {method} dist {dist_list[0]}: {seeds_lst}")

            for dist in dist_list:
                    # ... and make the VQE results file for that dist
                make_vqe_dist_results_file(molecule, basis, dist, seeds_lst, num_opt_virtual_orbital)

            # Get the dist list again for full file generation
            make_vqe_results_file(molecule, basis, dist_list, seeds_lst, num_opt_virtual_orbital)

            # Check the correlation energy of OVOS vs. UMP2 for this molecule, basis, dist, and num_opt_virtual_orbitals as a sanity check
            # for dist in dist_list:
            #     print_e_corr_ovos_vs_ump2(molecule, basis, dist, num_opt_virtual_orbital, seeds_lst)

            # Check the spread of VQE final energies for all seeds for this molecule, basis, method, dist, and num_opt_virtual_orbitals to see if there are convergence issues
            # We can do this by gathering the final energies for all seeds for this molecule, basis, method, dist, and num_opt_virtual_orbitals, and then print the range and standard deviation of the final energies to see if there is a lot of variance in the final energies for different seeds, which might indicate convergence issues
            # for dist in dist_list:
            #     for method in ["OVOS", "UHF", "UMP2"]:
            #         gather_and_print_vqe_final_energy_spread(molecule, basis, method, dist, num_opt_virtual_orbital, seeds_lst)

        for oo in [False]: # [True, False]:
            plot_vqe_curve_results(molecule, basis, dist_list_save, num_opt_virtual_orbitals, True, False, oo)
            plot_vqe_curve_results(molecule, basis, dist_list_save, num_opt_virtual_orbitals, False, False, oo)
        

if True:
    # Need to plot the VQE curve for one seed = "True", and both oo = True and False...
        # So we can see the difference in using prev. final thetas and keep trying to find best from random...
    
    for molecule in ["Li2", "HF", "H2O"]:
        basis = "6-31G"
        method = "OVOS" # Placeholder for getting dist and seed list

        # Get dist list from the folder
        dist_list = [1.0] # For getting number of optimal virtual orbitals for this molecule and basis, which is the same for all dists and seeds, we can just use one dist, and we can use the same dist list for all num_opt_virtual_orbitals as well since they should be the same
        # Get the number of optimal "virtual" orbitals for this molecule and basis, which is the same for all dists and seeds
        num_opt_virtual_orbital = get_num_opt_virtual_orbitals(molecule, basis, dist_list[0])[0]
        
        # for num_opt_virtual_orbital in num_opt_virtual_orbitals:
        dist_list = gather_dist_lst(molecule, basis, method, num_opt_virtual_orbital)
        if molecule == "Li2":
            dist_list = [dist for dist in dist_list if float(dist) >= 2.5]
        else:
            dist_list = [dist for dist in dist_list if float(dist) >= 0.7]
            
        # Here i need to designate the seed to "True" as i do not use a specific seed but the prev.
        seed_lst = True

        for dist in dist_list:
            make_vqe_dist_results_file(molecule, basis, dist, seed_lst, num_opt_virtual_orbital)

        make_vqe_results_file(molecule, basis, dist_list, seed_lst, num_opt_virtual_orbital)

        print(f"\nFinished gathering VQE results for {molecule} {basis} for all dists and num_opt_virtual_orbitals, now plotting the curves...")    
        print(f"Number of optimal virtual orbitals: {num_opt_virtual_orbital}")
        print(f"Dist list for plotting: {dist_list}")

        for oo in [False]: # [True, False]:
            plot_vqe_curve_results(molecule, basis, dist_list, [num_opt_virtual_orbital], True, True, oo)
            plot_vqe_curve_results(molecule, basis, dist_list, [num_opt_virtual_orbital], False, True, oo)


























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


