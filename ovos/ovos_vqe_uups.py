"""
Try OVOS with VQE and see if it can find the ground state energy of a molecule.
"""
    # Math and general imports
import numpy as np
import time
    # SlowQuant imports
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
    # Pyscf imports
from pyscf import gto, scf, mp
    # OVOS imports
from ovos import OVOS
    # Json
import json
    # Other imports
from multiprocessing import Pool
import os


# # Qiskit
#     # Qiskit imports
# from qiskit_aer import AerSimulator
# from qiskit_ibm_runtime import SamplerV2 as SamplerV2IBM
# from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
#     # Qiskit Setup
# aer = AerSimulator()
# sampler = SamplerV2IBM(mode=aer)
# mapper = ParityMapper(num_particles=(1, 1))
#     # Initialize Quantum Interface
# QI = QuantumInterface(sampler, "fUCCSD", mapper, shots=10)

# To get iterations and energy from SlowQuant
import sys
import io
import re

class Dee:
    """Write to multiple streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

class Tee(io.StringIO):
    """
    A writeable stream that writes to both an original stream (e.g. the terminal)
    and to an internal StringIO buffer.
    """
    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream

    def write(self, s):
        # Write to the buffer (for later retrieval)
        super().write(s)
        # Write to the original stream (the terminal) and flush immediately
        self.original_stream.write(s)
        self.original_stream.flush()

    def flush(self):
        super().flush()
        self.original_stream.flush()
def run_ucc_and_get_stats(wf, str_, orbital_optimization, atol=1e-6):
    # Save the original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Create tee streams that write to both the terminal and a buffer
    tee_stdout = Tee(original_stdout)
    tee_stderr = Tee(original_stderr)

    # Replace the system streams with our tees
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    try:
        # Run the optimization – output will appear on the terminal AND be captured
        
        # Run the optimization – output will appear on the terminal AND be captured
        wf.run_wf_optimization_1step(str_, orbital_optimization=orbital_optimization, tol=atol, maxiter=5000)
    finally:
        # Restore the original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    # Combine the captured output from both streams
    output = tee_stdout.getvalue() + tee_stderr.getvalue()

    # --- Now parse the output exactly as before ---
    lines = output.splitlines()
    
    stats = {
        'iterations': None,
        'function_evaluations': None,
        'gradient_evaluations': None,
        'final_energy': None,
        'success': None,
    }

    for line in lines:
        line_stripped = line.strip()

        if 'Optimization terminated successfully' in line_stripped:
            stats['success'] = True
        elif 'Optimization failed.' in line_stripped or 'Unsuccessful' in line_stripped:
            stats['success'] = False

        match = re.search(r'Current function value:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line_stripped)
        if match:
            stats['final_energy'] = float(match.group(1))

        match = re.search(r'Iterations:\s*(\d+)', line_stripped)
        if match:
            stats['iterations'] = int(match.group(1))

        match = re.search(r'Function evaluations:\s*(\d+)', line_stripped)
        if match:
            stats['function_evaluations'] = int(match.group(1))

        match = re.search(r'Gradient evaluations:\s*(\d+)', line_stripped)
        if match:
            stats['gradient_evaluations'] = int(match.group(1))

        # Start parsing the progress table for energy vs. iteration
            # Find the iteration number and energy from the progress table
                # As it looks like:
            # --------Iteration # | Iteration time [s] | Electronic energy [Hartree]
            # --------     1      |        13.63       |    -104.9398385314308797    |     N/A    
            # --------     2      |         4.96       |    -105.0266208692542591    |     N/A    
            # --------     3      |         4.94       |    -105.1111607905682206    |     N/A    
            # --------     4      |         4.97       |    -105.1684778451402309    |     N/A    
            # --------     5      |         3.67       |    -105.1768966082391188    |     N/A    
            # --------     6      |         4.99       |    -105.1770801982351173    |     N/A    
            # --------     7      |         5.13       |    -105.1770821629680626    |     N/A    
            # --------     8      |         5.13       |    -105.1770821674887202    |     N/A    
            # --------     9      |         5.16       |    -105.1770821725482108    |     N/A    
            # --------     10     |         5.09       |    -105.1770821815187134    |     N/A    
            # I want a list of energies till the final iteration, and the final iteration number, to be able to plot energy vs. iteration.
        if '|' in line_stripped and 'Iteration #' not in line_stripped :
            parts = line_stripped.split('|')
            if len(parts) >= 3:
                iteration_part = parts[0].strip().lstrip('-').strip()
                energy_part = parts[2].strip()
                try:
                    energy_val = float(energy_part)
                    if 'iter_energies' not in stats:
                        stats['iter_energies'] = []
                    stats['iter_energies'].append(energy_val)
                except ValueError:
                    pass


    # Fallback to progress table if summary missing
    if stats['iterations'] is None:
        iteration_numbers = []
        for line in lines:
            if '|' in line and 'Iteration #' not in line:
                parts = line.split('|')
                if len(parts) >= 1:
                    first_col = parts[0].strip().lstrip('-').strip()
                    try:
                        iteration_numbers.append(int(first_col))
                    except ValueError:
                        pass
        if iteration_numbers:
            stats['iterations'] = max(iteration_numbers)

    return stats



def VQE_OVOS(atom, basis, dist, num_opt_virtual_orbs, oo, seed):
    molecule = str(atom.split()[0])  # Get the first element symbol for naming
    if molecule == "H":
        molecule = "HF"
    elif molecule == "C":
        molecule = "CO"
    elif molecule == "N":
        molecule = "NH3"
    elif molecule == "O":
        molecule = "H2O"

    name_out = f"backup/data/{molecule}/{basis}/VQE/{dist}/OVOS_{molecule}_{dist}_{basis}_VQE_opt_num_{num_opt_virtual_orbs}_{oo}_output.txt"
    if not os.path.exists(os.path.dirname(name_out)):
        os.makedirs(os.path.dirname(name_out))
    with open(name_out, "w") as f:
        sys.stdout = Dee(sys.__stdout__, f)
        try:
            # Water molecule, minimal basis
            mol = gto.Mole()
            mol.atom = atom
            mol.basis = basis
            mol.unit = 'Angstrom'
            mol.spin = 0
            mol.charge = 0
            mol.symmetry = False
            mol.verbose = 0
            mol.build()
                # Get one- and two-electron integrals
            h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
            g_eri = mol.intor("int2e")

                # Number of electrons and orbitals
            num_electrons = mol.nelectron
            num_orbitals = mol.nao_nr() 
            print(f"Number of electrons: {num_electrons}, Number of orbitals: {num_orbitals}")

            # Create OVOS object and run
                # RHF reference for OVOS
            mf = scf.RHF(mol)
            mf.verbose = 0
            mf.kernel()

                # Initial data (RHF orbitals)
            Fao = [mf.get_fock(), mf.get_fock()]
            mo_coeffs = [mf.mo_coeff, mf.mo_coeff]
                    # Check if mo_coeffs are unrestricted or restricted
            if np.isclose(mo_coeffs[0], mo_coeffs[1], atol=1e-12).all():
                print("Initial MO coefficients are restricted (RHF-like).")
            else:
                print("Initial MO coefficients are unrestricted (UHF-like).")

                # Set up OVOS
            num_opt_virtual_orbs = int(num_opt_virtual_orbs * (num_orbitals - num_electrons//2))  # Convert fraction to actual number of orbitals
            print(f"Optimizing {num_opt_virtual_orbs} active virtual orbitals (out of {num_orbitals - num_electrons//2} total virtual orbitals).")
            ovos = OVOS(
                mol=mol,
                scf=mf,
                Fao=Fao,
                num_opt_virtual_orbs=num_opt_virtual_orbs*2,      # active virtual spin‑orbitals
                mo_coeff=mo_coeffs,
                init_orbs="RHF",
                verbose=1,
                max_iter=100,
                conv_energy=1e-8,
                conv_grad=1e-4,
                keep_track_max=50
            )
                # Run OVOS
            E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)
            E_corr = E_corr[-1]  # Final correlation energy
            E_tot = E_corr + mf.e_tot
            print(f"\nOptimization finished. Final MP2 energy = {E_tot} Hartree. (ΔE_corr = {E_corr} Hartree)")
            # Check if mo_coeffs are unrestricted or restricted
            if np.isclose(E_corr_mo[0], E_corr_mo[1], atol=1e-12).all():
                print("OVOS optimized orbitals are restricted (RHF-like).")
            else:
                print("OVOS optimized orbitals are unrestricted (UHF-like).")
            print()

            # SlowQuant
            print()
            print("Running VQE optimization with SlowQuant using OVOS-optimized orbitals as reference...")
                # Initialize for UPS wave function with OVOS-optimized orbitals
            WF_ovos = UnrestrictedWaveFunctionUPS(
                mol.nelectron,
                ((mol.nelectron//2,mol.nelectron//2), num_electrons//2+num_opt_virtual_orbs),                   # CAS(2,2) for H2O in 6-31G
                E_corr_mo,  # Use OVOS-optimized orbitals
                h_core,
                g_eri,
                "utups",
                {"n_layers":1},
                include_active_kappa=False,
            )
                # Initialize thetas randomly for reproducibility
            np.random.seed(seed)
            thetas = (2*np.pi*np.random.random(len(WF_ovos.thetas)) - np.pi).tolist()       
            atol = 1e-6
                # Use same random seed and initial thetas for fair comparison
            WF_ovos.thetas = thetas
            
                # Optimize WF
            stats_ovos_opt = run_ucc_and_get_stats(WF_ovos, "BFGS", oo, atol)
                    # Get optimization iterations/evaluations
            iter_ovos_opt = stats_ovos_opt['iterations']
            eval_ovos_opt = [stats_ovos_opt['function_evaluations'], stats_ovos_opt['gradient_evaluations']]
                    # Get optimized energy
            E_ovos_opt = stats_ovos_opt['final_energy']
            E_ovos_hist = stats_ovos_opt['iter_energies']
            print(f"OVOS optimized energy = {E_ovos_opt} Hartree @ iterations {iter_ovos_opt}.")

                # Save w. OVOS-optimized orbitals for later comparison
            file_out_ovos = f"backup/data/{molecule}/{basis}/VQE/OVOS/{dist}/UPS_OVOS_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbs}_{oo}_{seed}.json"
            if not os.path.exists(os.path.dirname(file_out_ovos)):
                os.makedirs(os.path.dirname(file_out_ovos))
            with open(file_out_ovos, "w") as f:
                json.dump({
                    "dist": dist,
                    "thetas": thetas,
                    "oo": oo,
                    "mo": E_corr_mo,
                    "iterations": iter_ovos_opt,
                    "final_energy": E_ovos_opt,
                    "iter_energies": E_ovos_hist,
                    "function_evaluations": eval_ovos_opt[0],
                    "gradient_evaluations": eval_ovos_opt[1],
                }, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


            # Compare with UHF reference
            print()
            print("Calculating UHF reference for comparison...")
                # UHF reference for comparison
            mf_uhf = scf.UHF(mol)
            mf_uhf.verbose = 0
            mf_uhf.frozen = list(range(num_electrons//2 + num_opt_virtual_orbs, num_orbitals))  # Freeze the same virtual orbitals as in OVOS optimization for a fair comparison
            mf_uhf.kernel()

            #     # Initualize for UPS wave function with UHF orbitals
            WF_uhf = UnrestrictedWaveFunctionUPS(
                mol.nelectron, 
                ((mol.nelectron//2, mol.nelectron//2), num_electrons//2+num_opt_virtual_orbs),                   # CAS(2,2) for H2O in 6-31G
                mf_uhf.mo_coeff,  
                h_core,
                g_eri,
                "utups",
                {"n_layers":1},
                include_active_kappa=False,
            )
            WF_uhf.thetas = thetas

                # Optimize WF
            stats_uhf_opt = run_ucc_and_get_stats(WF_uhf, "BFGS", oo, atol)
                    # Get optimization iterations/evaluations
            iter_uhf_opt = stats_uhf_opt['iterations']
            eval_uhf_opt = [stats_uhf_opt['function_evaluations'], stats_uhf_opt['gradient_evaluations']]
                    # Get optimized energy
            E_uhf_opt = stats_uhf_opt['final_energy']
            E_uhf_hist = stats_uhf_opt['iter_energies']
            print(f"UHF energy = {E_uhf_opt} Hartree @ iterations {iter_uhf_opt}.")

                # Save w. UHF orbitals for later comparison
            file_out_uhf = f"backup/data/{molecule}/{basis}/VQE/UHF/{dist}/UPS_UHF_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbs}_{oo}_{seed}.json"
            if not os.path.exists(os.path.dirname(file_out_uhf)):
                os.makedirs(os.path.dirname(file_out_uhf))
            with open(file_out_uhf, "w") as f:
                json.dump({
                    "dist": dist,
                    "thetas": thetas,
                    "oo": oo,
                    "mo": mf_uhf.mo_coeff,
                    "iterations": iter_uhf_opt,
                    "final_energy": E_uhf_opt,
                    "iter_energies": E_uhf_hist,
                    "function_evaluations": eval_uhf_opt[0],
                    "gradient_evaluations": eval_uhf_opt[1],
                }, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


            # Compare with UMP2 natural orbital reference
            print()
            print("Calculating UMP2 natural orbitals for comparison...")
                # UMP2 natural orbitals for comparison
            mf_ump2 = scf.UHF(mol).run()
            fs_orbs = mf_ump2.mo_coeff[0].shape[1]
            active_orbs = num_electrons//2 + num_opt_virtual_orbs
            print(f"Total number of orbitals: {fs_orbs}, Active orbitals: {active_orbs} (occupied + active virtual), Freezing the rest for UMP2 natural orbital calculation.")
            frozen_orbs = fs_orbs - active_orbs
            frozen_orbs_idx = list(range(active_orbs, fs_orbs))  # Freeze the highest-energy virtual orbitals
            print(f"Freezing {frozen_orbs} orbitals (indices {frozen_orbs_idx}) for UMP2 natural orbital calculation.")

            ump2_obj = mp.UMP2(mf_ump2, frozen=frozen_orbs_idx).run()
            mp2_no_coeff = ump2_obj.make_fno()[1]
            
            # check if mp2_no_coeff are unrestricted or restricted
            if np.isclose(mp2_no_coeff[0], mp2_no_coeff[1], atol=1e-12).all():
                print("UMP2 natural orbitals are restricted (RHF-like).")
            else:
                print("UMP2 natural orbitals are unrestricted (UHF-like).")
            
            # Energy of UMP2 natural orbital reference
            mf_ump2_no = scf.UHF(mol)
            mf_ump2_no.kernel(mo_coeff=mp2_no_coeff)
            ump2_no_obj = mp.UMP2(mf_ump2_no, frozen=frozen_orbs_idx).run()

            # Fair comparison
            uhf_energy = mf_ump2.e_tot          # No freezing
            mp2_energy = ump2_obj.e_tot         # With frozen
            mp2_no_energy = ump2_no_obj.e_tot   # With frozen
            corr_diff = ump2_no_obj.e_corr - ump2_obj.e_corr
                # Summary of reference energies
            print(f"UHF energy: {uhf_energy:.6f} Hartree")
            print(f"UMP2 energy (original): {mp2_energy:.6f} Hartree, correlation: {ump2_obj.e_corr:.6f} Hartree")
            print(f"UMP2 energy (natural orbitals): {mp2_no_energy:.6f} Hartree, correlation: {ump2_no_obj.e_corr:.6f} Hartree")
            print(f"Correlation energy difference: {corr_diff:.6f} Hartree")
            print()

            #     # Initialize for UPS wave function with UMP2 natural orbitals
            WF_ump2_no = UnrestrictedWaveFunctionUPS(
                mol.nelectron, 
                ((mol.nelectron//2, mol.nelectron//2), num_electrons//2+num_opt_virtual_orbs),                   # CAS(2,2) for H2O in 6-31G
                mp2_no_coeff, 
                h_core,
                g_eri,
                "utups",
                {"n_layers":1},
                include_active_kappa=False,
            )
            WF_ump2_no.thetas = thetas

                # Optimize WF
            stats_ump2_no_opt = run_ucc_and_get_stats(WF_ump2_no, "BFGS", oo, atol)
                    # Get optimization iterations/evaluations
            iter_ump2_no_opt = stats_ump2_no_opt['iterations']
            eval_ump2_no_opt = [stats_ump2_no_opt['function_evaluations'], stats_ump2_no_opt['gradient_evaluations']]
                    # Get optimized energy
            E_ump2_no_opt = stats_ump2_no_opt['final_energy']
            E_ump2_no_hist = stats_ump2_no_opt['iter_energies']
            print(f"UHF natural orbital energy = {E_ump2_no_opt} Hartree @ iterations {iter_ump2_no_opt}.")

                # Save w. UMP2 natural orbitals for later comparison
            file_out_ump2_no = f"backup/data/{molecule}/{basis}/VQE/UMP2/{dist}/UPS_UMP2_NO_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbs}_{oo}_{seed}.json"
            if not os.path.exists(os.path.dirname(file_out_ump2_no)):
                os.makedirs(os.path.dirname(file_out_ump2_no))
            with open(file_out_ump2_no, "w") as f:
                json.dump({
                    "dist": dist,
                    "thetas": thetas,
                    "oo": oo,
                    "mo": mp2_no_coeff,
                    "uhf_energy": uhf_energy,
                    "ump2_energy": mp2_energy,
                    "ump2_no_energy": mp2_no_energy,
                    "iterations": iter_ump2_no_opt,
                    "final_energy": E_ump2_no_opt,
                    "iter_energies": E_ump2_no_hist,
                    "function_evaluations": eval_ump2_no_opt[0],
                    "gradient_evaluations": eval_ump2_no_opt[1],
                }, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

            # Summary of results
            print("\nSummary of results:")
            print(f"Molecule: {atom} with basis set {basis} for a total of {num_electrons} electrons and {num_orbitals} orbitals.")
            print(f"OVOS correlation energy: {E_tot:.6f} Hartree (Active unocc. orbitals: {num_opt_virtual_orbs})")
            print(f"OVOS energy: {E_ovos_opt:.6f} Hartree  @ iterations {iter_ovos_opt} (Eval. func. {eval_ovos_opt[0]}, grad. {eval_ovos_opt[0]}")
            print(f"UHF energy: {E_uhf_opt:.6f} Hartree  @ iterations {iter_uhf_opt} (Eval. func. {eval_uhf_opt[0]}, grad. {eval_uhf_opt[0]}")
            print(f"UMP2 natural orbital energy: {E_ump2_no_opt:.6f} Hartree  @ iterations {iter_ump2_no_opt} (Eval. func. {eval_ump2_no_opt[0]}, grad. {eval_ump2_no_opt[0]}")

            data_out = {
                "molecule": atom,
                "basis": basis,
                "num_electrons": num_electrons,
                "num_orbitals": num_orbitals,
                "num_opt_virtual_orbs": num_opt_virtual_orbs,
                "thetas": thetas,
                "oo": oo,
                "E_corr_OVOS": E_tot,
                "E_ovos_opt": E_ovos_opt,
                "iter_ovos_opt": iter_ovos_opt,
                "eval_ovos_opt": eval_ovos_opt,
                "E_uhf_opt": E_uhf_opt,
                "iter_uhf_opt": iter_uhf_opt,
                "eval_uhf_opt": eval_uhf_opt,
                "E_ump2_no_opt": E_ump2_no_opt,
                "iter_ump2_no_opt": iter_ump2_no_opt,
                "eval_ump2_no_opt": eval_ump2_no_opt,
            }

            return data_out
        finally:
            sys.stdout = sys.__stdout__


# Define the molecule
    # HF, CO, NH3, H2O
atom_1 = "H 0 0 0; F 0 0 0.917" # HF bond length 0.917 Angstrom
atom_2 = "N 0 0 0; H 0 0 1.012; H 0 0.935 -0.262; H 0 -0.935 -0.262" # NH3 equilibrium geometry	
atom_3 = "O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;"  # H2O equilibrium geometry
atom_4 = "Li 0 0 0; H 0 0 1.595" # LiH bond length 1.595 Angstrom
atom_5 = "C 0 0 0; O 0 0 1.128" # CO bond length 1.128 Angstrom

    # Lst
atoms_lst = [atom_1, atom_2, atom_3, atom_4, atom_5]
basis_lst = ["6-31G", "cc-pVDZ"] # "6-31G", "cc-pVDZ"
num_opt_virtual_orbs_lst = [0.75]
oo_lst = [True, False]


def run_single_seed(args):
    """Wrapper function for multiprocessing - takes tuple of arguments"""
    atom, basis, dist, num_opt_virtual_orbs, oo, seed = args
    return VQE_OVOS(atom, basis, dist, num_opt_virtual_orbs, oo, seed)


# HF
# for dist in [-0.05, -0.025, -0.0125, 0.0125, 0.025, 0.05]: # Total num points: 36
#     # Set dist to 2 decimal places for consistent naming and to avoid floating point issues in file names
#     dist = round(dist, 5)
#     # Make the VQE folders for dist if they don't exist
#     if not os.path.exists(f"backup/data/HF/6-31G/VQE/{dist}"):
#         os.makedirs(f"backup/data/HF/6-31G/VQE/{dist}")
#     for method in ["OVOS", "UHF", "UMP2"]:
#         if not os.path.exists(f"backup/data/HF/6-31G/VQE/{method}/{dist}"):
#             os.makedirs(f"backup/data/HF/6-31G/VQE/{method}/{dist}")

for dist in [-0.025, -0.0125, 0.0125, 0.025, 0.05]: # Total num points: 36
    if dist == 0.0: # Skip the 0.0 point to avoid issues with naming and because it's not physically meaningful for a diatomic molecule
        continue
    # Set dist to 2 decimal places for consistent naming and to avoid floating point issues in file names
    dist = round(dist, 5)
    for atom in [f"H 0 0 0; F 0 0 {dist:.5f}"]: 
        for basis in [basis_lst[0]]:
            for num_opt_virtual_orbs in [num_opt_virtual_orbs_lst[0]]: # 0.25,0.5,0.75
                for oo in [oo_lst[1]]: # True, False
                    print(f"\nRunning VQE with OVOS optimization for {atom} in basis {basis} with {num_opt_virtual_orbs*100:.0f}% active virtual orbitals and orbital opt. = {oo}...")
                            # To ensure reproducibility, set random seed for SlowQuant optimizations
                    # for seed in [42, 123, 14, 10, 20, 21, 101, 404, 8, 13]:  # Run each configuration with different random seeds to assess variability
                    #     VQE_OVOS(atom, basis, dist, num_opt_virtual_orbs, oo, seed)

                    # Prepare arguments for each seed
                    seeds = [42, 123, 14, 10, 20, 21, 101, 404, 8, 13]
                    args_list = [(atom, basis, dist, num_opt_virtual_orbs, oo, seed) for seed in seeds]

                    # Run in parallel with one process per core
                    num_cores = len(seeds) if len(seeds) < 11 else int(len(seeds)/2)  # Use all available cores
                    with Pool(processes=num_cores) as pool:
                        pool.map(run_single_seed, args_list)
                    # Everything seed wil run before moving a dist forward...
# CO, NH3, H2O ...



# PLOT -> FS OVOS vs. MP2 vs. UHF initializations for each molecule/basis/active virtual orbital configuration.
# MP2 kun i active space, ikke hele virtual space, for å se hvordan det påvirker energies and convergence.... 
# Min af 10 theta configurationer ... 


# Send til Phillip -> Teori + Classical results ... 





# Table for presenting results:
#       I use 
# Step 1: Run with include_active_kappa = False
# | Results Summary for 6-31G with VQE Optimization (include_active_kappa = False) |
# | Molecule | Basis Set | # Electrons | # Orbitals | OVOS Num. of Act. Unocc. | OVOS Corr. Energy  | OVOS Opt. Energy @ Iterations | UHF Optimized Energy @ Iterations | UHF Opt. Energy Diff  |  Same or diff. Theta? |
# |          |           |             |            |      25%, 50%, 75%       |     [Hartree]      |           [Hartree]           |             [Hartree]             |       [Hartree]       |                       |
# |----------|-----------|-------------|------------|--------------------------|--------------------|-------------------------------|-----------------------------------|-----------------------|-----------------------|
# | H2O      | 6-31G     | 10          | 7          | 2 (4)  [RHF]             | -76.026963 (33.4%) | -84.896027  @ 40              | -84.904256  @ 38                  | -0.008229             | Same                  |
# | H2O      | 6-31G     | 10          | 9          | 4 (8)  [UHF]             | -76.101365 (91.1%) | -84.858729  @ 43              | -84.891800  @ 39                  | -0.033071             | Same                  |
# | H2O      | 6-31G     | 10          | 11         | 6 (12) [UHF]             | -76.111242 (98.8%) | -84.840472  @ 27              | -84.861685  @ 25                  | -0.021213             | Same                  |
# | H2O      | cc-pVDZ   | 10          | 8          | 4 (8)  [RHF]             | -76.142620 (56.8%) | -84.892488  @ 80              | -84.958209  @ 84                  | -0.065721             | Same                  |
# |..........|...........|.............|............|..........................|....................|...............................|...................................|.......................|.......................|
# | HF       | 6-31G     | 10          | 6          | 1 (2)  [RHF]             | -100.00188 (14.4%) | -104.825261 @ 27              | -104.829580 @ 25                  | -0.004319             | Same                  |
# | HF       | 6-31G     | 10          | 8          | 3 (6)  [UHF]             | -100.07082 (67.9%) | -104.802017 @ 46              | -104.805675 @ 34                  | -0.003658             | Same                  |
# | HF       | 6-31G     | 10          | 11         | 4 (8)  [UHF]             | -100.10745 (96.3%) | -104.775620 @ 43              | -104.789234 @ 18                  | -0.013614             | Same                  |
# | HF       | cc-pVDZ   | 10          | 8          | 3 (6)  [RHF]             | -100.10733 (33.6%) | -104.840054 @ 116             | -104.845022 @ 74                  | -0.004968             | Same                  |
# |..........|...........|.............|............|..........................|....................|...............................|...................................|.......................|.......................|
# | CO       | 6-31G     | 14          | 9          | 2 (4)  [UHF]             | -112.742906 (35.7%)| -134.989406 @ 125             | -134.988008 @ 142                 | +0.001398             | Same                  | 
# | CO       | 6-31G     | 14          | 12         | 5 (10) [UHF]             | -112.842877 (82.8%)| -134.981360 @ 66              | -134.967924 @ 72                  | +0.013436             | Same                  |                 
# | CO       | 6-31G     | 14          | 15         | 8 (16) [UHF]             | -112.873902 (97.4%)| ...                           | ...                               | ...                   | Same                  |
# |..........|...........|.............|............|..........................|....................|...............................|...................................|.......................|.......................|
# | NH3      | 6-31G     | 10          | 7          | 2 (4)  [RHF]             | -56.173502 (26.4%) | -68.156893  @ 47              | -68.159458  @ 37                  | -0.002565             | Same                  |
# | NH3      | 6-31G     | 10          | 10         | 5 (10) [UHF]             | -56.244012 (88.0%) | -68.127669  @ 41              | -68.143925  @ 38                  | -0.016256             | Same                  |
# | NH3      | 6-31G     | 10          | 12         | 7 (14) [UHF]             | -56.255257 (97.9%) | -68.075272  @ 39              | -68.136328  @ 37                  | -0.061056             | Same                  |
# | NH3      | cc-pVDZ   | 10          | 11         | 6 (12) [RHF]             | -56.276992 (59.4%) | -68.166951  @ 66              | -68.187273  @ 56                  | -0.020322             | Same                  |
# |----------|-----------|-------------|------------|--------------------------|--------------------|-------------------------------|-----------------------------------|-----------------------|-----------------------|
# | LiH ...
#
# Run each configuration with a few times with different random seeds to get an idea of the variability in the results (e.g. due to optimization getting stuck in local minima).
# Get mean and standard deviation of the energies and iterations for each configuration.
# 
# Plot: Energy vs. iteration for each configuration to visualize convergence behavior. (OVOS vs. UHF initializations, different numbers of active virtual orbitals, etc.)
#    Compare initial energies (OVOS vs. UHF) to see how much OVOS improves the starting point for VQE optimization. 
#
#
# Step 2: Run with include_active_kappa = True
# | Results Summary for H2O in 6-31G with VQE Optimization (include_active_kappa = True) |
# | Molecule | Basis Set | # Electrons | # Orbitals | OVOS Num. of Act. Unocc. | OVOS Corr. Energy (Hartree) | OVOS Opt. Energy (Hartree) @ Iterations | UHF Optimized Energy (Hartree) @ Iterations | UHF Opt. Energy Diff (Hartree) |
# |----------|-----------|-------------|------------|--------------------------|-----------------------------|-----------------------------------------|---------------------------------------------|--------------------------------|
# | H2O      | 6-31G     | 10          | 7          | 2 (4)  
# | H2O      | 6-31G     | 10          | 7          | 4 (8)  
# | H2O      | 6-31G     | 10          | 7          | 6 (12)  
# |..........|...........|.............|............|..........................|.............................|.........................................|.............................................|................................|
# | HF       | 6-31G     | 10          | 6          | 1 (2)
# | HF       | 6-31G     | 10          | 6          | 3 (6)
# | HF       | 6-31G     | 10          | 6          | 4 (8)
# | HF       | cc-PVDZ   | ...
# |..........|...........|.............|............|..........................|.............................|................................
# | ...
#
#
#


# # SlowQuant
#     # Initialize for UPS wave function with OVOS-optimized orbitals
# WF_ovos = UnrestrictedWaveFunctionUPS(
#     mol.nelectron,  # e.g. 10 electrons
#     ((6,4), num_electrons//2+num_opt_virtual_orbs),                   
#     E_corr_mo,          # Use OVOS-optimized orbitals
#     h_core,
#     g_eri,
#     "utups",
#     {"n_layers":1},
#     include_active_kappa=False,
# )
#     # Initialize thetas randomly for reproducibility
# np.random.seed(42)
# thetas = (2*np.pi*np.random.random(len(WF_ovos.thetas)) - np.pi).tolist()   
# WF_ovos.thetas = thetas
# WF_ovos.run_wf_optimization_1step("BFGS", orbital_optimization=True, tol=1e-6, maxiter=5000)

#     # Initualize for UPS wave function with UHF orbitals
# WF_uhf = UnrestrictedWaveFunctionUPS(
#     mol.nelectron,  # e.g. 10 electrons
#     ((6,4), num_electrons//2+num_opt_virtual_orbs),                  
#     mf_uhf.mo_coeff,  # Use UHF orbitals
#     h_core,
#     g_eri,
#     "utups",
#     {"n_layers":1},
#     include_active_kappa=False,
# )
#     # Use same random seed and initial thetas for fair comparison
# WF_uhf.thetas = thetas
# WF_uhf.run_wf_optimization_1step("BFGS", orbital_optimization=True, tol=1e-6, maxiter=5000)