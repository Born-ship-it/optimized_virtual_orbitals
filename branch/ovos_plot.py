"""
Plot convergence of OVOS algorithm.

CTRL+SHIFT+P, select 'WSL: Reopen Folder in WSL'
 > uv run branch/ovos_plot.py
"""

import os

import matplotlib as mpl		
import matplotlib.pyplot as plt
import numpy as np
import json


def plot_OVOS_convergence(atom, basis):

        # Set optimization RLE
    rle_optimization = False  

        # Set previous OVOS results
    use_previous_ovos = False  
        # Set random unitary rotation from UHF orbitals
    use_random_rotation = False
        # Set UHF orbitals as start guess
    use_uhf_start = False
        # Set RHF orbitals as start guess
    use_rhf_start = False
        # Set UCASSCF
    use_UCASSCF = False

        # Set if canonical Fock
    use_canonical_fock = True


        # Set combine previous OVOS results with RLE
    use_combine = True  # Set to True to combine previous OVOS results with RLE results
    if use_combine == True:
        use_previous_ovos = True
        # use_random_rotation = True
        # use_uhf_start = True
        use_rhf_start = True


    # Molecule
    atom_choose_between = [
        "H .0 .0 .0; H .0 .0 0.74144",  # H2 bond length 0.74144 Angstrom
        "Li .0 .0 .0; H .0 .0 1.595",   # LiH bond length 1.595 Angstrom
        "O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;",  # H2O equilibrium geometry
        "C  0.0000  0.0000  0.0000; H  0.0000  0.9350  0.5230; H  0.0000 -0.9350  0.5230;", # CH2 
        "B 0 0 0; H 0 0 1.19; H 0 1.03 -0.40; H 0 -1.03 -0.40",  # BH3 equilibrium geometry
        "N 0 0 0; N 0 0 1.10", # N2 bond length 1.10 Angstrom
        "C 0 0 0; O 0 0 1.128", # CO bond length 1.128 Angstrom
        "H 0 0 0; F 0 0 0.917", # HF bond length 0.917 Angstrom
        "N 0 0 0; H 0 0 1.012; H 0 0.935 -0.262; H 0 -0.935 -0.262" # NH3 equilibrium geometry
    ]

    # Get molecule parameters in atom_choose_between from atom name 
    atom_dict = {
        "H2": 0,
        "LiH": 1,
        "H2O": 2,
        "CH2": 3,
        "BH3": 4,
        "N2": 5,
        "CO": 6,
        "HF": 7,
        "NH3": 8
    }	


    # Get misc parameters for molecule
        # from file:
            # 	# Save data to JSON file
            # import json

            # data = {
            # 	"num_electrons": num_electrons,
            # 	"full_space_size": full_space_size,
            # 	"active_space_size": active_space_size,
            # 	"MP2_e_corr": MP2_e_corr,
            # 	"CCSD_e_corr": ccsd.e_corr,
            # 	"CCSD(T)_e_corr": ccsd.e_tot + ccsd.ccsd_t() - rhf.e_tot,
            # 	"FCI_e_corr": casci.e_tot - rhf.e_tot,
            # 	"CASSCF_e_corr": casscf.e_tot - rhf.e_tot
            # }

            # with open(f"branch/data/{molecule}/{basis_set}/molecule_data.json", "w") as f:
            # 	json.dump(data, f, indent=2)
            # print(f"Miscellaneous data saved to branch/data/{molecule}/{basis_set}/molecule_data.json")
            # print()

        # Get number of electrons, full space size, active space size, MP2 correlation energy, CCSD(T) correlation energy
    with open(f"branch/data/{atom}/{basis}/molecule_data.json", "r") as f:
        molecule_data = json.load(f)
        num_electrons = molecule_data["num_electrons"]
        full_space_size = molecule_data["full_space_size"]
        active_space_size = molecule_data["active_space_size"]
        MP2_e_corr = molecule_data["MP2_e_corr"]
        CCSD_e_corr = molecule_data["CCSD_e_corr"]
        CCSD_T_e_corr = molecule_data["CCSD(T)_e_corr"]
        FCI_e_corr = molecule_data["FCI_e_corr"]
        CASSCF_e_corr = molecule_data["CASSCF_e_corr"]



    print()
    print("#### Reference MP2 Correlation Energy for Full Space: ", MP2_e_corr, " Hartree ####")
    print("Number of electrons: ", num_electrons
    , "| Full space size (MOs): ", full_space_size
    , "| Number of occupied MOs: ", num_electrons//2 , "| Number of virtual MOs: ", full_space_size - num_electrons//2)
    print()


    """
    Plot OVOS convergence results for different number of optimized virtual orbitals
    """

    virtual_orbs = True  # Set to True to plot different number of virtual orbitals data
    if virtual_orbs == True:
        # Store data for different number of virtual orbitals in a structured way
        conv_virtual_orbs_UHF = {
            'num_virtual_orbitals_UHF': [],
            'MP2_final_energies_UHF': [],
            'MP2_energies_per_iteration_UHF': [],
            'iterations_to_converge_UHF': [],
            'unrestricted_scf_check_UHF': []
        }

        conv_virtual_orbs_RHF = {
            'num_virtual_orbitals_RHF': [],
            'MP2_final_energies_RHF': [],
            'MP2_energies_per_iteration_RHF': [],
            'iterations_to_converge_RHF': [],
            'unrestricted_scf_check_RHF': []
        }

        conv_virtual_orbs_prev = {
            'num_virtual_orbitals_prev': [],
            'MP2_final_energies_prev': [],
            'MP2_energies_per_iteration_prev': [],
            'iterations_to_converge_prev': [],
            'unrestricted_scf_check_prev': []
        }
        
        conv_virtual_orbs_random = {
            'num_virtual_orbitals_random': [],
            'MP2_final_energies_random': [],
            'MP2_energies_per_iteration_random': [],
            'iterations_to_converge_random': [],
            'unrestricted_scf_check_random': []
        }

        conv_virtual_orbs_UCASSCF = {
            'num_virtual_orbitals_UCASSCF': [],
            'final_energy_UCASSCF': [],
        }

        # Load data for different number of virtual orbitals
            # If cannonical Fock
        if use_canonical_fock == True:
            canonical_Fock = "use_canonical_fock"
        else:
            canonical_Fock = "use_non_canonical_fock"

        # Load previous OVOS results
        if use_previous_ovos == True:
            with open("branch/data/"+atom+"/"+basis+"/lst_MP2_different_virt_orbs_prev.json", "r") as f:
                lst_E_corr_prev = json.load(f)  
        # Load random rotation results
        if use_random_rotation == True:
            with open("branch/data/"+atom+"/"+basis+"/lst_MP2_different_virt_orbs_random.json", "r") as f: 
                lst_E_corr_random = json.load(f)
        # Load RHF start guess results
        if use_rhf_start == True:
            with open("branch/data/"+atom+"/"+basis+"/lst_MP2_different_virt_orbs_RHF_init.json", "r") as f:
                lst_E_corr_RHF = json.load(f)
        # Load UHF start guess results
        if use_uhf_start == True:
            with open("branch/data/"+atom+"/"+basis+"/lst_MP2_different_virt_orbs_UHF_init.json", "r") as f:
                lst_E_corr_UHF = json.load(f)
        # Load UCASSCF results
        if use_UCASSCF == True:
            with open("branch/data/"+atom+"/"+basis+"/lst_UCASSCF_results_RHF.json", "r") as f:
                lst_E_corr_UCASSCF = json.load(f)


        # For each number of virtual orbitals, extract data
        if use_uhf_start == True:
            for n_vorb, E_corr in enumerate(lst_E_corr_UHF[0]):

                # Get converged MP2 correlation energy
                final_energies = lst_E_corr_UHF[0][n_vorb][-1]
                    # Store final energies
                conv_virtual_orbs_UHF['MP2_final_energies_UHF'].append(final_energies)

                # Get all MP2 energies per iteration
                mp2_energies_iter = lst_E_corr_UHF[0][n_vorb]
                    # Store mp2 energies per iteration
                conv_virtual_orbs_UHF['MP2_energies_per_iteration_UHF'].append(mp2_energies_iter)

                # Get number of virtual orbitals
                num_virt_orbs = lst_E_corr_UHF[1][n_vorb]
                    # Store number of virtual orbitals
                conv_virtual_orbs_UHF['num_virtual_orbitals_UHF'].append(num_virt_orbs)

                # Get iterations to convergence
                iter_to_converge = lst_E_corr_UHF[2][n_vorb][-1]
                    # Store iterations to converge
                conv_virtual_orbs_UHF['iterations_to_converge_UHF'].append(iter_to_converge)

                # Get unrestricted scf check
                unrestricted_check = lst_E_corr_UHF[3][n_vorb]
                    # Store unrestricted scf check
                conv_virtual_orbs_UHF['unrestricted_scf_check_UHF'].append(unrestricted_check)

        if use_rhf_start == True:
            for n_vorb, E_corr in enumerate(lst_E_corr_RHF[0]):

                # Get converged MP2 correlation energy
                final_energies = lst_E_corr_RHF[0][n_vorb][-1]
                    # Store final energies
                conv_virtual_orbs_RHF['MP2_final_energies_RHF'].append(final_energies)

                # Get all MP2 energies per iteration
                mp2_energies_iter = lst_E_corr_RHF[0][n_vorb]
                    # Store mp2 energies per iteration
                conv_virtual_orbs_RHF['MP2_energies_per_iteration_RHF'].append(mp2_energies_iter)

                # Get number of virtual orbitals
                num_virt_orbs = lst_E_corr_RHF[1][n_vorb]
                    # Store number of virtual orbitals
                conv_virtual_orbs_RHF['num_virtual_orbitals_RHF'].append(num_virt_orbs)

                # Get iterations to convergence
                iter_to_converge = lst_E_corr_RHF[2][n_vorb][-1]
                    # Store iterations to converge
                conv_virtual_orbs_RHF['iterations_to_converge_RHF'].append(iter_to_converge)

                # Get unrestricted scf check
                unrestricted_check = lst_E_corr_RHF[3][n_vorb]
                    # Store unrestricted scf check
                conv_virtual_orbs_RHF['unrestricted_scf_check_RHF'].append(unrestricted_check)

        if use_previous_ovos == True:
            # For each number of virtual orbitals, extract data
            for n_vorb, E_corr in enumerate(lst_E_corr_prev[0]):

                # Get converged MP2 correlation energy
                final_energies = lst_E_corr_prev[0][n_vorb][-1]
                    # Store final energies
                conv_virtual_orbs_prev['MP2_final_energies_prev'].append(final_energies)

                # Get all MP2 energies per iteration
                mp2_energies_iter = lst_E_corr_prev[0][n_vorb]
                    # Store mp2 energies per iteration
                conv_virtual_orbs_prev['MP2_energies_per_iteration_prev'].append(mp2_energies_iter)

                # Get number of virtual orbitals
                num_virt_orbs = lst_E_corr_prev[1][n_vorb]
                    # Store number of virtual orbitals
                conv_virtual_orbs_prev['num_virtual_orbitals_prev'].append(num_virt_orbs)

                # Get iterations to convergence
                iter_to_converge = lst_E_corr_prev[2][n_vorb][-1]
                    # Store iterations to converge
                conv_virtual_orbs_prev['iterations_to_converge_prev'].append(iter_to_converge)

                # Get unrestricted scf check
                unrestricted_check = lst_E_corr_prev[3][n_vorb]
                    # Store unrestricted scf check
                conv_virtual_orbs_prev['unrestricted_scf_check_prev'].append(unrestricted_check)

        if use_random_rotation == True:
            # For each number of virtual orbitals, extract data
            for n_vorb, E_corr in enumerate(lst_E_corr_random[0]):

                # Get converged MP2 correlation energy
                final_energies = lst_E_corr_random[0][n_vorb][-1]
                    # Store final energies
                conv_virtual_orbs_random['MP2_final_energies_random'].append(final_energies)

                # Get all MP2 energies per iteration
                mp2_energies_iter = lst_E_corr_random[0][n_vorb]
                    # Store mp2 energies per iteration
                conv_virtual_orbs_random['MP2_energies_per_iteration_random'].append(mp2_energies_iter)

                # Get number of virtual orbitals
                num_virt_orbs = lst_E_corr_random[1][n_vorb]
                    # Store number of virtual orbitals
                conv_virtual_orbs_random['num_virtual_orbitals_random'].append(num_virt_orbs)

                # Get iterations to convergence
                iter_to_converge = lst_E_corr_random[2][n_vorb][-1]
                    # Store iterations to converge
                conv_virtual_orbs_random['iterations_to_converge_random'].append(iter_to_converge)

                # Get unrestricted scf check
                unrestricted_check = lst_E_corr_random[3][n_vorb]
                    # Store unrestricted scf check
                conv_virtual_orbs_random['unrestricted_scf_check_random'].append(unrestricted_check)

        if use_UCASSCF == True:
            for n_vorb, E_corr in enumerate(lst_E_corr_UCASSCF[0]):

                # Get converged UCASSCF energy
                final_energies = lst_E_corr_UCASSCF[0][n_vorb][-1]
                    # Store final energies
                conv_virtual_orbs_UCASSCF['final_energy_UCASSCF'].append(final_energies)

                # Get number of virtual orbitals
                num_virt_orbs = lst_E_corr_UCASSCF[1][n_vorb][-1]*2
                    # Store number of virtual orbitals
                conv_virtual_orbs_UCASSCF['num_virtual_orbitals_UCASSCF'].append(num_virt_orbs)

        # Print max number of virtual orbitals converged for each method
        if use_uhf_start == True:
            print("Max number of virtual orbitals converged for UHF start guess: ", max(conv_virtual_orbs_UHF['num_virtual_orbitals_UHF']))
        if use_rhf_start == True:
            print("Max number of virtual orbitals converged for RHF start guess: ", max(conv_virtual_orbs_RHF['num_virtual_orbitals_RHF']))
        if use_previous_ovos == True:
            print("Max number of virtual orbitals converged for previous OVOS start guess: ", max(conv_virtual_orbs_prev['num_virtual_orbitals_prev']))
        if use_random_rotation == True:
            print("Max number of virtual orbitals converged for random rotation start guess: ", max(conv_virtual_orbs_random['num_virtual_orbitals_random']))
        if use_UCASSCF == True:
            print("Max number of virtual orbitals converged for UCASSCF: ", max(conv_virtual_orbs_UCASSCF['num_virtual_orbitals_UCASSCF']))
        print()

        # Initialize figure
        if basis == "STO-3G":
            scalefactor = 0.75
        elif basis == "6-31G":
            scalefactor = 1.0
        elif basis == "cc-pVDZ":
            scalefactor = 2.0
        elif basis == "cc-pVTZ":
            scalefactor = 5.0
        fig_size_lenght = 12 * scalefactor
        fig_vo, ax_vo = plt.subplots(figsize=(fig_size_lenght, 7)) # 12/num_total_virorbs, 7

        # Title of plot: mention molecule/basis_set/active_space/full_space_size
        suptitle_txt = f'OVOS Convergence: {atom}/{basis}, Full Space ({num_electrons}e,{full_space_size}o)'
        fig_vo.suptitle(suptitle_txt, fontsize=16)

        # Color map for different virtual orbitals
            # Strong vibrant colors
        colors = plt.cm.hsv(np.linspace(0, 1, active_space_size))

        # Small x-axis movement for better visualization
        x_move = 0.0
        if use_combine == True:
            x_move = 0.5 

        # Plot converged MP2 correlation energy as a function of number of virtual orbitals
            # Do not plot if positive energy
        if use_uhf_start == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_UHF['num_virtual_orbitals_UHF']):
                MP2_vorb = conv_virtual_orbs_UHF['MP2_final_energies_UHF'][i]
                if MP2_vorb < 0.001:
                    ax_vo.scatter(num_virt_orbs - x_move, MP2_vorb, marker='o', alpha=1.0, color=colors[i])
        
        if use_rhf_start == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_RHF['num_virtual_orbitals_RHF']):
                MP2_vorb = conv_virtual_orbs_RHF['MP2_final_energies_RHF'][i]
                if MP2_vorb < 0.001:
                    ax_vo.scatter(num_virt_orbs - x_move, MP2_vorb, marker='D', alpha=1.0, color=colors[i])

        if use_previous_ovos == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_prev['num_virtual_orbitals_prev']):
                MP2_vorb = conv_virtual_orbs_prev['MP2_final_energies_prev'][i]
                if MP2_vorb < 0.001:
                    ax_vo.scatter(num_virt_orbs , MP2_vorb, marker='s', alpha=1.0, color=colors[i])
                    
        if use_random_rotation == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_random['num_virtual_orbitals_random']):
                MP2_vorb = conv_virtual_orbs_random['MP2_final_energies_random'][i]
                if MP2_vorb < 0.001:
                    ax_vo.scatter(num_virt_orbs + x_move, MP2_vorb, marker='^', alpha=1.0, color=colors[i])

        if use_UCASSCF == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_UCASSCF['num_virtual_orbitals_UCASSCF']):
                UCASSCF_energy = conv_virtual_orbs_UCASSCF['final_energy_UCASSCF'][i]
                if UCASSCF_energy < 0.001:
                    ax_vo.scatter(num_virt_orbs + x_move, UCASSCF_energy, marker='X', alpha=1.0, color=colors[i])

        # Plot intial MP2 correlation energy per iteration as a function of number of virtual orbitals
        if use_uhf_start == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_UHF['num_virtual_orbitals_UHF']):  
                MP2_vorb = conv_virtual_orbs_UHF['MP2_final_energies_UHF'][i]

                # Get Max iterations to converge for each run
                iter_max = conv_virtual_orbs_UHF['iterations_to_converge_UHF'][i]
                    # Get MP2 energy at first iteration
                MP2_iter_vorb = conv_virtual_orbs_UHF['MP2_energies_per_iteration_UHF'][i][0]
                    # Plot with marker the first iteration point
                ax_vo.scatter(num_virt_orbs - x_move, MP2_iter_vorb, marker='o', alpha=0.25, color=colors[i])
                # Draw a line from first iteration to converged energy
                    # And write number of iterations to converge next to the line
                ax_vo.plot([num_virt_orbs - x_move, num_virt_orbs - x_move], [MP2_iter_vorb, MP2_vorb], color=colors[i], alpha=0.3)
                    # Annotate number of iterations to converge
                ax_vo.text(num_virt_orbs - x_move + 0.075, (MP2_iter_vorb + MP2_vorb)/2, str(iter_max), fontsize=8, alpha=0.75)

        if use_rhf_start == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_RHF['num_virtual_orbitals_RHF']): 
                MP2_vorb = conv_virtual_orbs_RHF['MP2_final_energies_RHF'][i]

                # Get Max iterations to converge for each run
                iter_max = conv_virtual_orbs_RHF['iterations_to_converge_RHF'][i]
                    # Get MP2 energy at first iteration
                MP2_iter_vorb = conv_virtual_orbs_RHF['MP2_energies_per_iteration_RHF'][i][0]
                    # Plot with marker the first iteration point
                ax_vo.scatter(num_virt_orbs - x_move, MP2_iter_vorb, marker='D', alpha=0.25, color=colors[i])
                # Draw a line from first iteration to converged energy
                    # And write number of iterations to converge next to the line
                ax_vo.plot([num_virt_orbs - x_move, num_virt_orbs - x_move], [MP2_iter_vorb, MP2_vorb], color=colors[i], alpha=0.3)
                    # Annotate number of iterations to converge
                ax_vo.text(num_virt_orbs - x_move + 0.075, (MP2_iter_vorb + MP2_vorb)/2, str(iter_max), fontsize=8, alpha=0.75)

        if use_previous_ovos == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_prev['num_virtual_orbitals_prev']): 
                MP2_vorb = conv_virtual_orbs_prev['MP2_final_energies_prev'][i]

                # Get Max iterations to converge for each run
                iter_max = conv_virtual_orbs_prev['iterations_to_converge_prev'][i]
                    # Get MP2 energy at first iteration
                MP2_iter_vorb = conv_virtual_orbs_prev['MP2_energies_per_iteration_prev'][i][0]
                    # Plot with marker the first iteration point
                ax_vo.scatter(num_virt_orbs, MP2_iter_vorb, marker='s', alpha=0.25, color=colors[i])
                # Draw a line from first iteration to converged energy
                    # And write number of iterations to converge next to the line
                ax_vo.plot([num_virt_orbs, num_virt_orbs], [MP2_iter_vorb, MP2_vorb], color=colors[i], alpha=0.3)
                    # Annotate number of iterations to converge
                ax_vo.text(num_virt_orbs + 0.075, (MP2_iter_vorb + MP2_vorb)/2, str(iter_max), fontsize=8, alpha=0.75)

        if use_random_rotation == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_random['num_virtual_orbitals_random']):
                MP2_vorb = conv_virtual_orbs_random['MP2_final_energies_random'][i]

                # Get Max iterations to converge for each run
                iter_max = conv_virtual_orbs_random['iterations_to_converge_random'][i]
                    # Get MP2 energy at first iteration
                MP2_iter_vorb = conv_virtual_orbs_random['MP2_energies_per_iteration_random'][i][0]
                    # Plot with marker the first iteration point
                ax_vo.scatter(num_virt_orbs + x_move, MP2_iter_vorb, marker='^', alpha=0.25, color=colors[i])
                # Draw a line from first iteration to converged energy
                    # And write number of iterations to converge next to the line
                ax_vo.plot([num_virt_orbs + x_move, num_virt_orbs + x_move], [MP2_iter_vorb, MP2_vorb], color=colors[i], alpha=0.3)
                    # Annotate number of iterations to converge
                ax_vo.text(num_virt_orbs + x_move + 0.075, (MP2_iter_vorb + MP2_vorb)/2, str(iter_max), fontsize=8, alpha=0.75)

        if use_UCASSCF == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_UCASSCF['num_virtual_orbitals_UCASSCF']):
                UCASSCF_energy = conv_virtual_orbs_UCASSCF['final_energy_UCASSCF'][i]

                # Get MP2 energy at first iteration
                UCASSCF_iter_energy = conv_virtual_orbs_UCASSCF['final_energy_UCASSCF'][i]  # Assuming UCASSCF energy is the same as MP2 energy at first iteration for visualization
                    # Plot with marker the first iteration point
                ax_vo.scatter(num_virt_orbs + x_move, UCASSCF_iter_energy, marker='X', alpha=0.25, color=colors[i])
                # Draw a line from first iteration to converged energy
                    # And write number of iterations to converge next to the line
                ax_vo.plot([num_virt_orbs + x_move, num_virt_orbs + x_move], [UCASSCF_iter_energy, UCASSCF_energy], color=colors[i], alpha=0.3)
                    # Annotate number of iterations to converge
                # ax_vo.text(num_virt_orbs + x_move + 0.075, (UCASSCF_iter_energy + UCASSCF_energy)/2, 'UCASSCF', fontsize=8, alpha=0.75)
        
        # Legend for markers
        if use_rhf_start == True:
            ax_vo.scatter([], [], marker='D', color='black', label='Start Guess: RHF')
        if use_uhf_start == True:
            ax_vo.scatter([], [], marker='o', color='black', label='Start Guess: UHF')
        if use_previous_ovos == True:
            ax_vo.scatter([], [], marker='s', color='black', label='Start Guess: Prev.')
        if use_random_rotation == True:
            ax_vo.scatter([], [], marker='^', color='black', label='Start Guess: Best Random')
        if use_UCASSCF == True:
            ax_vo.scatter([], [], marker='X', color='black', label='UCASSCF Energy')

        # A horizontal line at the mp2 energy value at full space
            # Get full space MP2 energy
        full_space_MP2 = MP2_e_corr
        ax_vo.axhline(full_space_MP2, color='red', linestyle='--', label='Full Space MP2 Energy', linewidth=2, alpha=0.75)
        
            # Get full space CCSD energy
        # full_space_CCSD = CCSD_e_corr
        # ax_vo.axhline(full_space_CCSD, color='green', linestyle='--', label='Full Space CCSD Energy', linewidth=2, alpha=0.75)
        
            # Get full space CCSD(T) energy
        full_space_CCSD_T = CCSD_T_e_corr
        ax_vo.axhline(full_space_CCSD_T, color='blue', linestyle='--', label='Full Space CCSD(T) Energy', linewidth=2, alpha=0.75)

        #     # Get full space FCI energy
        # full_space_FCI = FCI_e_corr
        # ax_vo.axhline(full_space_FCI, color='purple', linestyle='--', label='Full Space FCI Energy', linewidth=2, alpha=0.75)

            # Get full space CASSCF energy
        # full_space_CASSCF = CASSCF_e_corr
        # ax_vo.axhline(full_space_CASSCF, color='orange', linestyle='--', label='Full Space CASSCF Energy', linewidth=2, alpha=0.75)

        # Set y-axis limits for better visualization
        lower_bound = []
        if use_uhf_start == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_UHF['num_virtual_orbitals_UHF']):
                MP2_vorb = conv_virtual_orbs_UHF['MP2_final_energies_UHF'][i]
                lower_bound.append(MP2_vorb)

        if use_rhf_start == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_RHF['num_virtual_orbitals_RHF']):
                MP2_vorb = conv_virtual_orbs_RHF['MP2_final_energies_RHF'][i]
                lower_bound.append(MP2_vorb)

        if use_previous_ovos == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_prev['num_virtual_orbitals_prev']):
                MP2_vorb = conv_virtual_orbs_prev['MP2_final_energies_prev'][i]
                lower_bound.append(MP2_vorb)
        
        if use_random_rotation == True:
            for i, num_virt_orbs in enumerate(conv_virtual_orbs_random['num_virtual_orbitals_random']):
                MP2_vorb = conv_virtual_orbs_random['MP2_final_energies_random'][i]
                lower_bound.append(MP2_vorb)

        # Include MP2 energy at full space and CCSD(T) energy at full space in lower bound calculation
        lower_bound.append(full_space_MP2)
        lower_bound.append(full_space_CCSD_T)

            # Find lower bound
        ax_vo.set_ylim(min(lower_bound) - 0.0005, 0)

        # Draw a line between the best of options for each number of virtual orbitals
            # the numbers of virtual orbitals, e.g. [2, 4, 6, 8, 10, 12, 14, 16, 18]
            # For each number of virtual orbitals, find the best final energy among the different start guesses
            # Remember it needs to jump by 2 in spin-orbitals
            # Remember the x-axis is in spin-orbitals, so need to multiply number of virtual orbitals by 2
            # Remember the offset x_move for each for better visualization
                # For UHF start guess: -x_move
                # For RHF start guess: -x_move/2
                # For previous OVOS: +x_move/2
                # For random rotation: +x_move
        best_points_x = []
        best_points_y = []
        for n_vorb in range(1, active_space_size*2):
            candidates = []
            x_positions = []
            if use_uhf_start == True:
                if n_vorb in conv_virtual_orbs_UHF['num_virtual_orbitals_UHF']:
                    idx = conv_virtual_orbs_UHF['num_virtual_orbitals_UHF'].index(n_vorb)
                    candidates.append(conv_virtual_orbs_UHF['MP2_final_energies_UHF'][idx])
                    x_positions.append(n_vorb - x_move)
            if use_rhf_start == True:
                if n_vorb in conv_virtual_orbs_RHF['num_virtual_orbitals_RHF']:
                    idx = conv_virtual_orbs_RHF['num_virtual_orbitals_RHF'].index(n_vorb)
                    candidates.append(conv_virtual_orbs_RHF['MP2_final_energies_RHF'][idx])
                    x_positions.append(n_vorb - x_move)
            if use_previous_ovos == True:
                if n_vorb in conv_virtual_orbs_prev['num_virtual_orbitals_prev']:
                    idx = conv_virtual_orbs_prev['num_virtual_orbitals_prev'].index(n_vorb)
                    candidates.append(conv_virtual_orbs_prev['MP2_final_energies_prev'][idx])
                    x_positions.append(n_vorb)
            if use_random_rotation == True:
                if n_vorb in conv_virtual_orbs_random['num_virtual_orbitals_random']:
                    idx = conv_virtual_orbs_random['num_virtual_orbitals_random'].index(n_vorb)
                    candidates.append(conv_virtual_orbs_random['MP2_final_energies_random'][idx])
                    x_positions.append(n_vorb + x_move)
            
            if len(candidates) > 0:
                best_energy = min(candidates)
                best_index = candidates.index(best_energy)
                best_x = x_positions[best_index]
                best_points_x.append(best_x)
                best_points_y.append(best_energy)
        
        ax_vo.plot(best_points_x, best_points_y, color='black', linestyle='-', linewidth=1.5, label='OVOS', alpha=0.25)      


        # Set x-axis limits and ticks with increased spacing
        ax_vo.set_xlim(1, active_space_size*2 - 1)
        x_ticks = list(range(2, active_space_size*2, 2))
            # For each number in x_ticks, take the half of it and put it as label, e.g. 2 -> 1, 4 -> 2, 6 -> 3, ...
                # Remove the last one
        x_labels = [str(x//2) for x in x_ticks]
        ax_vo.set_xticks(x_ticks)
        ax_vo.set_xticklabels(x_labels)

        ax_vo.set_xlabel('Number of Virtual Orbitals')


        # Add an opposite y-axis showing an procentage of correlation energy recovered
        def energy_to_percentage(energy):
            full_energy = full_space_MP2
            percentage = (energy / full_energy) * 100
            return percentage
        
        def percentage_to_energy(percentage):
            full_energy = full_space_MP2
            energy = (percentage / 100) * full_energy
            return energy
        
        secax = ax_vo.secondary_yaxis('right', functions=(energy_to_percentage, percentage_to_energy))
        secax.set_ylabel('Percentage of Correlation Energy Recovered (%)')
        secax.set_yticks([25, 50, 75, 80, 85, 90, 95, 100, 105, 110])
        secax.set_ylim(energy_to_percentage(ax_vo.get_ylim()[0]), energy_to_percentage(ax_vo.get_ylim()[1]))
        
        
        # Set colored intervals inside grid to show the interval of number of virtual orbitals
            # For each interval between two number of virtual orbitals on x-axis shift by 0.5
            # Make a horizontal colored band between each interval
        for i in range(1, active_space_size + 1):
            x_start = i*2 - 1
            x_end = i*2 + 1
            if i % 2 == 0:
                ax_vo.axvspan(x_start, x_end, color='lightgray', alpha=0.2)
            else:
                ax_vo.axvspan(x_start, x_end, color='white', alpha=0.0)

        # Labels and titles
        ax_vo.set_ylabel('Correlation Energy [Hartree]')
        # ax_vo.set_title('OVOS Convergence vs Number of Virtual Orbitals')
        ax_vo.legend(loc='upper right')
        
        """
        Apependix !!!!!!!! to show table aswell
        """

        # # Add table that highlights the best option for each number of virtual orbitals,
        #     # And if they were restricted or unrestricted
        # # Create table data
        # table_data = []
        # for n_vorb in range(1, active_space_size*2):
        #     best_energy = None
        #     best_method = None
        #     if use_uhf_start == True:
        #         if n_vorb in conv_virtual_orbs_UHF['num_virtual_orbitals_UHF']:
        #             idx = conv_virtual_orbs_UHF['num_virtual_orbitals_UHF'].index(n_vorb)
        #             energy = conv_virtual_orbs_UHF['MP2_final_energies_UHF'][idx]
        #             if best_energy is None or energy < best_energy:
        #                 best_energy = energy
        #                 best_method = 'UHF'
        #     if use_rhf_start == True:
        #         if n_vorb in conv_virtual_orbs_RHF['num_virtual_orbitals_RHF']:
        #             idx = conv_virtual_orbs_RHF['num_virtual_orbitals_RHF'].index(n_vorb)
        #             energy = conv_virtual_orbs_RHF['MP2_final_energies_RHF'][idx]
        #             if best_energy is None or energy < best_energy:
        #                 best_energy = energy
        #                 best_method = 'RHF'
        #     if use_previous_ovos == True:
        #         if n_vorb in conv_virtual_orbs_prev['num_virtual_orbitals_prev']:
        #             idx = conv_virtual_orbs_prev['num_virtual_orbitals_prev'].index(n_vorb)
        #             energy = conv_virtual_orbs_prev['MP2_final_energies_prev'][idx]
        #             if best_energy is None or energy < best_energy:
        #                 best_energy = energy
        #                 best_method = 'Prev.'
        #     if use_random_rotation == True:
        #         if n_vorb in conv_virtual_orbs_random['num_virtual_orbitals_random']:
        #             idx = conv_virtual_orbs_random['num_virtual_orbitals_random'].index(n_vorb)
        #             energy = conv_virtual_orbs_random['MP2_final_energies_random'][idx]
        #             if best_energy is None or energy < best_energy:
        #                 best_energy = energy
        #                 best_method = 'Random'

        #     # Determine SCF type for best method
        #     scf_type = None
        #     if best_method == 'UHF' and use_uhf_start:
        #         idx = conv_virtual_orbs_UHF['num_virtual_orbitals_UHF'].index(n_vorb)
        #         scf_type = 'UHF/' if conv_virtual_orbs_UHF['unrestricted_scf_check_UHF'][idx] else 'RHF/'
        #     elif best_method == 'RHF' and use_rhf_start:
        #         idx = conv_virtual_orbs_RHF['num_virtual_orbitals_RHF'].index(n_vorb)
        #         scf_type = 'UHF/' if conv_virtual_orbs_RHF['unrestricted_scf_check_RHF'][idx] else 'RHF/'
        #     elif best_method == 'Prev.' and use_previous_ovos:
        #         idx = conv_virtual_orbs_prev['num_virtual_orbitals_prev'].index(n_vorb)
        #         scf_type = 'UHF/' if conv_virtual_orbs_prev['unrestricted_scf_check_prev'][idx] else 'RHF/'
        #     elif best_method == 'Random' and use_random_rotation:
        #         idx = conv_virtual_orbs_random['num_virtual_orbitals_random'].index(n_vorb)
        #         scf_type = 'UHF/' if conv_virtual_orbs_random['unrestricted_scf_check_random'][idx] else 'RHF/'
            
        #     if best_energy is not None:
        #         table_data.append([n_vorb, f"{best_energy:.6f}", best_method, scf_type])
            
        
        # # Transpose table data to flip rows and columns
        # if table_data:
        #     # Extract headers as first column values
        #     num_vorbs = [row[0] for row in table_data]
        #     energies = [row[1] for row in table_data]
        #     methods = [row[2] for row in table_data]
        #     scf_types = [row[3] for row in table_data]

        #     # Do a combined methods + scf types for better visualization
        #     methods_scf = [f"{scf}{meth}" for scf, meth in zip(scf_types, methods)]
            
        #     # Create transposed table data with row labels as first column
        #     transposed_data = [
        #         ['Num. Virt. Orbs'] + [str(v//2) for v in num_vorbs],
        #         ['Best Guess/ \n Orbitals'] + methods_scf
        #     ]
            
        #     # Create table without column labels, using first column as row labels
        #     table = ax_vo.table(cellText=transposed_data,
        #                         cellLoc='center',
        #                         loc='bottom', bbox=[0.0, -0.25, 1.0, 0.15])
        #     table.auto_set_font_size(False)
        #     table.set_fontsize(8)
        #     table.scale(1, 1.2)
        

        # Grid
            # Major grid lines for right y-axis: secax
        ax_vo.grid(which='major', axis='y', linestyle='--', alpha=0.7)
            # Minor grid lines for left y-axis: ax_vo
        ax_vo.minorticks_on()
        ax_vo.grid(which='minor', axis='y', linestyle=':', alpha=0.5)


        # Finalize plot
        plt.tight_layout()

        # Save plot
        if rle_optimization == True and use_combine == False:
            plt.savefig("branch/images/vorb/"+atom+"/ovos_conv_vs_vorb_"+atom+"_"+basis+"_RLE.png", dpi=150)
            print("Plot saved to branch/images/vorb/.png with RLE")
        elif use_previous_ovos == True and use_combine == False:
            plt.savefig("branch/images/vorb/"+atom+"/ovos_conv_vs_vorb_"+atom+"_"+basis+"_prev.png", dpi=150)
            print("Plot saved to branch/images/vorb/.png with previous OVOS results")
        elif use_previous_ovos == True and use_combine == True:
            plt.savefig("branch/images/vorb/"+atom+"/ovos_conv_vs_vorb_"+atom+"_"+basis+"_comb.png", dpi=150)
            print("Plot saved to branch/images/vorb/.png with combined standard and previous OVOS results")
        else:
            plt.savefig("branch/images/vorb/"+atom+"/ovos_conv_vs_vorb_"+atom+"_"+basis+".png", dpi=150)
            print("Plot saved to branch/images/vorb/.png standard")

for atom in ["CO", "H2O", "HF", "NH3"]:
    for basis in ["6-31G", "cc-pVDZ"]:
        plot_OVOS_convergence(atom, basis)