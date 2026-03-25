"""
I intend to plot VQE results...

"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os


def gather_vqe_results(molecule, basis, dist_list, seeds_lst):
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
            filename = f"backup/data/{molecule}/{basis}/VQE/{dist}/UPS_OVOS_{molecule}_{basis}_{dist}_opt_num_4_False_{seed}.json"
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

def gather_dist_lst(molecule, method, basis):
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
            for seed in seeds_lst:
                filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_OVOS_{molecule}_{basis}_{dist}_opt_num_4_False_{seed}.json"
                with open(filename, 'r') as f:
                    result = json.load(f)
                    energies.append(result['final_energy'])
            method_data[dist] = min(energies)
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
        for seed in seeds_lst:
            filename = f"backup/data/{molecule}/{basis}/VQE/{method}/{dist}/UPS_OVOS_{molecule}_{basis}_{dist}_opt_num_4_False_{seed}.json"
            with open(filename, 'r') as f:
                result = json.load(f)
                energies.append(result['final_energy'])
        data[method] = min(energies)

    file_name = f"backup/data/{molecule}/{basis}/VQE/VQE_HF_6-31G_{dist}_results.json"
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

# Run HF 6-31G VQE results file generation
    # For only one dist
        # Get seeds list from the folder
molecule = "HF"
basis = "6-31G"
method = "OVOS"
dist = 0.5
seeds_lst = gather_seeds_lst(molecule, basis, methods, dist)
print(f"Seeds list for {molecule} {basis} dist {dist}: {seeds_lst}")
make_vqe_dist_results_file(molecule, basis, dist, seeds_lst)