"""
Try OVOS with VQE and see if it can find the ground state energy of a molecule.
"""
# Set Numba to use TBB threading (thread-safe) BEFORE importing SlowQuant
import os
os.environ['NUMBA_THREADING_LAYER'] = 'omp'  # Use OpenMP (thread-safe)
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

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
import logging

# Global logger instance
_current_logger = None

def set_logger(logger):
    """Set the global logger for print wrapper"""
    global _current_logger
    _current_logger = logger

def log_print(*args, **kwargs):
    """Wrapper that routes print() to logger.info() or terminal"""
    if _current_logger is not None:
        msg = " ".join(str(a) for a in args)
        _current_logger.info(msg)
    else:
        # Fallback to regular print if no logger set
        print(*args, **kwargs)

# Replace built-in print in VQE_OVOS scope
def setup_logging_in_function(seed, dist, molecule, basis, num_opt_virtual_orbs, oo):
    """Setup logger and return log_print wrapper for use inside VQE_OVOS"""
    logger = logging.getLogger(f"VQE_{seed}_{dist}")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    
    # Create output file path
    name_out = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/OVOS_{molecule}_{dist}_{basis}_VQE_opt_num_{num_opt_virtual_orbs}_{oo}_{seed}_output.txt"
    os.makedirs(os.path.dirname(name_out), exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(name_out, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with prefix
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(f'[Seed {seed}@{dist:.4f}] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, name_out


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



def VQE_OVOS(atom, molecule, basis, dist, num_opt_virtual_orbs, oo, seed):
    # name_out = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/OVOS_{molecule}_{dist}_{basis}_VQE_opt_num_{num_opt_virtual_orbs}_{oo}_{seed}_output.txt"
    # if not os.path.exists(os.path.dirname(name_out)):
    #     os.makedirs(os.path.dirname(name_out))

    # with open(name_out, "w") as f:
    #     # sys.stdout = Dee(sys.__stdout__, f)
    #     original_stdout = sys.stdout

        # Setup logging
    logger, name_out = setup_logging_in_function(seed, dist, molecule, basis, num_opt_virtual_orbs, oo)
    set_logger(logger)


    if True:  # Just capture to list and write at the end to avoid issues with multiprocessing
        try:    
            log_print(f"\nRunning VQE with OVOS optimization for {molecule} w. bond length {dist} in basis {basis} with {num_opt_virtual_orbs*100:.0f}% active virtual orbitals and orbital opt. = {oo}...")

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
            log_print(f"Built molecule {molecule} with basis {basis} and bond length {dist} Angstrom.")
                # Get one- and two-electron integrals
            h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
            # print("Calculated one-electron integrals (kinetic + nuclear).")
            g_eri = mol.intor("int2e")
            # print("Calculated two-electron integrals.")

                # Number of electrons and orbitals
            num_electrons = mol.nelectron
            num_orbitals = mol.nao_nr() 
            log_print(f"Number of electrons: {num_electrons}, Number of orbitals: {num_orbitals}")

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
                log_print("Initial MO coefficients are restricted (RHF-like).")
            else:
                log_print("Initial MO coefficients are unrestricted (UHF-like).")

                # Set up OVOS
            num_opt_virtual_orbs = int(num_opt_virtual_orbs * (num_orbitals - num_electrons//2))  # Convert fraction to actual number of orbitals
            log_print(f"Optimizing {num_opt_virtual_orbs} active virtual orbitals (out of {num_orbitals - num_electrons//2} total virtual orbitals).")
            ovos = OVOS(
                mol=mol,
                scf=mf,
                Fao=Fao,
                num_opt_virtual_orbs=num_opt_virtual_orbs*2,      # active virtual spin‑orbitals
                mo_coeff=mo_coeffs,
                init_orbs="RHF",
                verbose=1,
                max_iter=500,
                conv_energy=1e-8,
                conv_grad=1e-4,
                keep_track_max=50
            )
                # Run OVOS
            E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)
            E_corr = E_corr[-1]  # Final correlation energy
            E_tot = E_corr + mf.e_tot
            log_print(f"\nOptimization finished. Final MP2 energy = {E_tot} Hartree. (ΔE_corr = {E_corr} Hartree)")
            # Check if mo_coeffs are unrestricted or restricted
            if np.isclose(E_corr_mo[0], E_corr_mo[1], atol=1e-12).all():
                log_print("OVOS optimized orbitals are restricted (RHF-like).")
            else:
                log_print("OVOS optimized orbitals are unrestricted (UHF-like).")
            log_print()

            # # Kill for debug here
            # assert False, "Debug stop after OVOS optimization to check results before proceeding to VQE optimizations with different references. Remove this line to run the full script."

            # SlowQuant
            log_print()
            log_print("Running VQE optimization with SlowQuant using OVOS-optimized orbitals as reference...")
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
            log_print(f"OVOS optimized energy = {E_ovos_opt} Hartree @ iterations {iter_ovos_opt}.")

                # Save w. OVOS-optimized orbitals for later comparison
            file_out_ovos = f"backup/data/{molecule}/{basis}/VQE/OVOS/{dist}/UPS_OVOS_{molecule}_{basis}_{dist}_opt_num_{num_opt_virtual_orbs}_{oo}_{seed}.json"
            if not os.path.exists(os.path.dirname(file_out_ovos)):
                os.makedirs(os.path.dirname(file_out_ovos))
            with open(file_out_ovos, "w") as f:
                json.dump({
                    "dist": dist,
                    "thetas": thetas,
                    "oo": oo,
                    "E_corr_OVOS": E_corr,
                    "mo": E_corr_mo,
                    "iterations": iter_ovos_opt,
                    "final_energy": E_ovos_opt,
                    "iter_energies": E_ovos_hist,
                    "function_evaluations": eval_ovos_opt[0],
                    "gradient_evaluations": eval_ovos_opt[1],
                }, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


            # Compare with UHF reference
            log_print()
            log_print("Calculating UHF reference for comparison...")
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
            log_print(f"UHF energy = {E_uhf_opt} Hartree @ iterations {iter_uhf_opt}.")

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
            log_print()
            log_print("Calculating UMP2 natural orbitals for comparison...")
                # UMP2 natural orbitals for comparison
            mf_ump2 = scf.UHF(mol).run()
            fs_orbs = mf_ump2.mo_coeff[0].shape[1]
            active_orbs = num_electrons//2 + num_opt_virtual_orbs
            log_print(f"Total number of orbitals: {fs_orbs}, Active orbitals: {active_orbs} (occupied + active virtual), Freezing the rest for UMP2 natural orbital calculation.")
            frozen_orbs = fs_orbs - active_orbs
            frozen_orbs_idx = list(range(active_orbs, fs_orbs))  # Freeze the highest-energy virtual orbitals
            log_print(f"Freezing {frozen_orbs} orbitals (indices {frozen_orbs_idx}) for UMP2 natural orbital calculation.")

            ump2_obj = mp.UMP2(mf_ump2, frozen=frozen_orbs_idx).run()
            mp2_no_coeff = ump2_obj.make_fno()[1]
            
            # check if mp2_no_coeff are unrestricted or restricted
            if np.isclose(mp2_no_coeff[0], mp2_no_coeff[1], atol=1e-12).all():
                log_print("UMP2 natural orbitals are restricted (RHF-like).")
            else:
                log_print("UMP2 natural orbitals are unrestricted (UHF-like).")
            
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
            log_print(f"UHF energy: {uhf_energy:.6f} Hartree")
            log_print(f"UMP2 energy (original): {mp2_energy:.6f} Hartree, correlation: {ump2_obj.e_corr:.6f} Hartree")
            log_print(f"UMP2 energy (natural orbitals): {mp2_no_energy:.6f} Hartree, correlation: {ump2_no_obj.e_corr:.6f} Hartree")
            log_print(f"Correlation energy difference: {corr_diff:.6f} Hartree")
            log_print()

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
            log_print(f"UHF natural orbital energy = {E_ump2_no_opt} Hartree @ iterations {iter_ump2_no_opt}.")

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
            log_print("\nSummary of results:")
            log_print(f"Molecule: {atom} with basis set {basis} for a total of {num_electrons} electrons and {num_orbitals} orbitals.")
            log_print(f"OVOS correlation energy: {E_tot:.6f} Hartree (Active unocc. orbitals: {num_opt_virtual_orbs})")
            log_print(f"OVOS energy: {E_ovos_opt:.6f} Hartree  @ iterations {iter_ovos_opt} (Eval. func. {eval_ovos_opt[0]}, grad. {eval_ovos_opt[0]}")
            log_print(f"UHF energy: {E_uhf_opt:.6f} Hartree  @ iterations {iter_uhf_opt} (Eval. func. {eval_uhf_opt[0]}, grad. {eval_uhf_opt[0]}")
            log_print(f"UMP2 natural orbital energy: {E_ump2_no_opt:.6f} Hartree  @ iterations {iter_ump2_no_opt} (Eval. func. {eval_ump2_no_opt[0]}, grad. {eval_ump2_no_opt[0]}")

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
        
            # At end:
            log_print(f"✓ Completed. Output saved to {name_out}")
        
            return data_out
        
        # finally:
        except Exception as e:
            logger.error(f"✗ ERROR in seed {seed}: {e}", exc_info=True)
            raise
        finally:
            set_logger(None)  # Clear logger reference


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
    atom, molecule, basis, dist, num_opt_virtual_orbs, oo, seed = args
    return VQE_OVOS(atom, molecule, basis, dist, num_opt_virtual_orbs, oo, seed)


# Verify:
# def verify_pes_energy(atom_string, basis, dist, final_energy):
#     """Verify VQE energy makes physical sense"""
#     from pyscf import gto, scf
    
#     mol = gto.Mole()
#     mol.atom = atom_string
#     mol.verbose = 0
#     mol.basis = basis
#     mol.unit = 'Angstrom'
#     mol.build()
    
#     # Get reference energies  
#     mf_uhf = scf.UHF(mol)
#     mf_uhf.kernel()
    
#     print(f"Distance: {dist}")
#     print(f"  UHF energy:               {mf_uhf.e_tot:.6f} Hartree")
#     print(f"  Your VQE energy:          {final_energy:.6f} Hartree")
#     print(f"  VQE vs UHF difference:    {final_energy - mf_uhf.e_tot:.6f} Hartree")
#     print()
    
#     # VQE should be lower (better) than HF but not lower than true ground state
#     assert final_energy <= mf_uhf.e_tot + 1e-4, "VQE energy is HIGHER than HF - something is wrong!"

# verify_pes_energy("H 0 0 0; F 0 0 -0.5", basis_lst[0], -0.5, -108.812036)  # HF equilibrium geometry energy from OVOS MP2


# HF
def run_hf_vqe():
        # Rounds of dist:
            # ['0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0']
            # ['0.725', '0.75', '0.775', '0.825', '0.85', '0.875', '0.925', '0.95', '0.975', '1.025', '1.05', '1.075', '1.125', '1.15', '1.175', '1.225', '1.25', '1.275', '1.325', '1.35', '1.375', '1.425', '1.45', '1.475', '1.525', '1.55', '1.575', '1.625', '1.65', '1.675', '1.725', '1.75', '1.775', '1.825', '1.85', '1.875', '1.925', '1.95', '1.975']
            # Together:
                    # [0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2, 1.225, 1.25, 1.275, 1.3, 1.325, 1.35, 1.375, 1.4, 1.425, 1.45, 1.475, 1.5, 1.525, 1.55, 1.575, 1.6, 1.625, 1.65, 1.675, 1.7, 1.725, 1.75, 1.775, 1.8, 1.825, 1.85, 1.875, 1.9, 1.925, 1.95, 1.975, 2.0]
                # With length 53, from 0.7 to 2.0 in steps of 0.025
        # Intterupted at 1.375, missing 1.4 and onwards, so I will start from 1.375 to 2.0 in steps of 0.025 to fill in the rest of the data for HF
    HF_list_full = np.arange(1.4, 2.025, 0.025).round(5).tolist()
        
    if False:
        for dist in HF_list_full:
            dist = round(dist, 5)
            # Make the VQE folders for dist if they don't exist
            if not os.path.exists(f"backup/data/HF/6-31G/VQE/dist/{dist}"):
                os.makedirs(f"backup/data/HF/6-31G/VQE/dist/{dist}")
            for method in ["OVOS", "UHF", "UMP2"]:
                if not os.path.exists(f"backup/data/HF/6-31G/VQE/{method}/{dist}"):
                    os.makedirs(f"backup/data/HF/6-31G/VQE/{method}/{dist}")

        # Rounds of seeds:
            # [42, 123, 14, 10, 20, 21, 101, 404, 8, 13]
            # [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199]
    HF_list_full_seeds = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199]
    if False:
        for dist in HF_list_full:
            dist = round(dist, 5)
            for atom in [f"H 0 0 0; F 0 0 {dist:.5f}"]: 
                for basis in [basis_lst[0]]:
                    for num_opt_virtual_orbs in [0.25]: #[num_opt_virtual_orbs_lst[0]]: # 0.25,0.5,0.75
                        for oo in [oo_lst[1]]: # True, False
                            print(f"\nRunning VQE with OVOS optimization for {atom} in basis {basis} with {num_opt_virtual_orbs*100:.0f}% active virtual orbitals and orbital opt. = {oo}...")
                            # To ensure reproducibility, set random seed for SlowQuant optimizations
                                # Prepare arguments for each seed
                            seeds = HF_list_full_seeds
                            args_list = [(atom, basis, dist, num_opt_virtual_orbs, oo, seed) for seed in seeds]

                            # Run in parallel with one process per core
                            num_cores = len(seeds) if len(seeds) < 11 else  10  # Use all available cores
                            with Pool(processes=num_cores) as pool:
                                pool.map(run_single_seed, args_list)
                            # Every seed wil run before moving a dist forward...

    # Rewrite to one seed but make Pool over an amount of dist variations instead of seeds, to get the rest of the HF data for all dist variations for one seed (e.g. 9) to verify in plots, and then I can run the rest of the seeds in parallel over dist variations once I verify the data looks correct for one seed.
    if False:
        # Set seed
        seed = 9
        # Make args list for all dist variations for one seed
        args_list = []
        for dist in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]: # HF_list_full: # [0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2, 1.225, 1.25, 1.275, 1.3, 1.325, 1.35, 1.375, 1.4, 1.425, 1.45, 1.475, 1.5, 1.525, 1.55, 1.575, 1.6, 1.625, 1.65, 1.675, 1.7, 1.725, 1.75, 1.775, 1.8, 1.825, 1.85, 1.875, 1.9, 1.925, 1.95, 1.975]:
            dist = round(dist, 5)
            for atom in [f"H 0 0 0; F 0 0 {dist:.5f}"]: 
                for basis in [basis_lst[0]]:
                    for num_opt_virtual_orbs in [0.25]: #[num_opt_virtual_orbs_lst[0]]: # 0.25,0.5,0.75
                        for oo in [oo_lst[1]]: # True, False
                            args_list.append((atom, basis, dist, num_opt_virtual_orbs, oo, seed))

        # Run in parallel with one process per core
        num_cores = len(args_list) if len(args_list) < 11 else 10  # Use all available cores
        with Pool(processes=num_cores) as pool:
            pool.map(run_single_seed, args_list)

    # Rewrite to 10 seeds for one dist variation (e.g. 1.375) to verify the variability across seeds for one dist variation, and then I can run the rest of the dist variations in parallel over seeds once I verify the data looks correct for one dist variation.
    if False:
        # Set dist
        dist = 1.375
        dist = round(dist, 5)
        # Make args list for all seeds for one dist variation
        args_list = []
        for atom in [f"H 0 0 0; F 0 0 {dist:.5f}"]: 
            for basis in [basis_lst[0]]:
                for num_opt_virtual_orbs in [0.75]: #[num_opt_virtual_orbs_lst[0]]: # 0.25,0.5,0.75
                    for oo in [oo_lst[1]]: # True, False
                        for seed in [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:  # Run each configuration with different random seeds to assess variability
                            args_list.append((atom, basis, dist, num_opt_virtual_orbs, oo, seed))

        # Run in parallel with one process per core
        num_cores = len(args_list) if len(args_list) < 11 else 10  # Use all available cores
        with Pool(processes=num_cores) as pool:
            pool.map(run_single_seed, args_list)
                            

    # Run and get the UHF data for all atom dist variations to verify in plots
    if False:
        for dist in HF_list_full:
            dist = round(dist, 5)
            for atom in [f"H 0 0 0; F 0 0 {dist}"]: 
                for basis in [basis_lst[0]]:
                    for hf in ["UHF", "RHF"]:
                        print(f"\nRunning UHF for {atom} in basis {basis} at dist {dist}...")
                        # set up molecule
                        mol = gto.Mole()
                        mol.atom = atom     # f"H 0 0 0; F 0 0 {dist}"
                        mol.basis = basis
                        mol.unit = 'Angstrom'
                        mol.spin = 0
                        mol.charge = 0
                        mol.symmetry = False
                        mol.verbose = 0
                        mol.build()
                        
                        # Get reference energies  
                        if hf == "UHF":
                            hf_energy = mol.UHF().run().e_tot
                        else:
                            hf_energy = mol.RHF().run().e_tot

                        # Save HF reference energy for later comparison
                        name_hf = f"backup/data/HF/6-31G/VQE/UHF/{dist}/{hf}_HF_{basis}_{dist}_reference_energy.txt"
                        with open(name_hf, "w") as f:
                            f.write(f"{hf_energy:.6f}\n")
                        print(f"HF reference energy for {atom} at dist {dist} saved to {name_hf}.")

    # Run and get nuclear repulsion energies for all atom dist variations to verify plots
    if False:
        for dist in HF_list_full:
            dist = round(dist, 5)
            for atom in [f"H 0 0 0; F 0 0 {dist}"]: 
                for basis in [basis_lst[0]]:
                    print(f"\nCalculating nuclear repulsion energy for {atom} in basis {basis} at dist {dist}...")
                    # set up molecule
                    mol = gto.Mole()
                    mol.atom = atom     # f"H 0 0 0; F 0 0 {dist}"
                    mol.basis = basis
                    mol.unit = 'Angstrom'
                    mol.spin = 0
                    mol.charge = 0
                    mol.symmetry = False
                    mol.verbose = 0
                    mol.build()
                    
                    # Get nuclear repulsion energy  
                    nuc_rep_energy = mol.energy_nuc()

                    # Save nuclear repulsion energy for later comparison
                    name_nuc_rep = f"backup/data/HF/6-31G/VQE/UHF/{dist}/nuclear_repulsion_HF_{basis}_{dist}_energy.txt"
                    with open(name_nuc_rep, "w") as f:
                        f.write(f"{nuc_rep_energy:.6f}\n")
                    print(f"Nuclear repulsion energy for {atom} at dist {dist} saved to {name_nuc_rep}.")

# H2O
def run_h2o_vqe(): # - at: 1.8501... start from here next time
    # Let us explore the dist list we need for H2O
        # Keeping the angle the same and varying the O-H bond length around the equilibrium geometry:
            # 'O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692'

    # Meaning we will have to calculate the bond length to make sure the angle is the same and only the bond length is varied
        # Find angle from equilibrium geometry
    str_equi = 'O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692'
        # Get coordinates
    coords = []
    for line in str_equi.split(";"):
        parts = line.strip().split()
        element = parts[0]
        x, y, z = map(float, parts[1:])
        coords.append((element, x, y, z))
        # print(f"Element: {element}, Coordinates: ({x}, {y}, {z})")
        # Calculate bond length O-H
    O_coords = [x for x in coords if x[0] == "O"][0][1:]  # Get O coordinates
    H_coords = [x for x in coords if x[0] == "H"]  # Get H coordinates
    bond_lengths = [] # Equilibrium O-H bond lengths
    for H in H_coords:
        bond_length = np.sqrt((O_coords[0] - H[1])**2 + (O_coords[1] - H[2])**2 + (O_coords[2] - H[3])**2)
        bond_lengths.append(bond_length)
    # print(f"Equilibrium O-H bond lengths: {bond_lengths}")
        # Find angle H-O-H
    H1_coords = H_coords[0][1:]
    H2_coords = H_coords[1][1:]
    vector_OH1 = np.array(H1_coords) - np.array(O_coords)
    vector_OH2 = np.array(H2_coords) - np.array(O_coords)
    cos_angle = np.dot(vector_OH1, vector_OH2) / (np.linalg.norm(vector_OH1) * np.linalg.norm(vector_OH2))
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    print(f"Equilibrium H-O-H angle: {angle_deg:.2f} degrees")
        # Now we can vary the O-H bond length while keeping the angle fixed by scaling the H coordinates accordingly
            # Try 10 different bond length variations around the equilibrium geometry, e.g. from 0.75 to 1.25 times the equilibrium bond length
    bond_length_set = [1.55, 1.575, 1.6, 1.625, 1.65, 1.675, 1.7, 1.725, 1.75, 1.775, 1.8, 1.825, 1.85, 1.875, 1.9, 1.925, 1.95, 1.975, 2.0]
            # Figure out the procentages we need to get the bond_length_set values
    bond_length_variations = [bl / bond_lengths[0] for bl in bond_length_set]
                # Round off to 5 decimals
    bond_length_variations = [round(blv, 4) for blv in bond_length_variations]

    dist_list_h2o = []
            # Assert that the angle is the same for all variations
    for variation in bond_length_variations:
        new_H_coords = []
        for H in H_coords:
            vector_OH = np.array(H[1:]) - np.array(O_coords)
            new_vector_OH = vector_OH * variation  # Scale the vector to change the bond length
            new_H_coord = np.array(O_coords) + new_vector_OH  # Get new H coordinates
            new_H_coords.append(new_H_coord)
        # Check angle is the same
        vector_OH1_new = new_H_coords[0] - np.array(O_coords)
        vector_OH2_new = new_H_coords[1] - np.array(O_coords)
        cos_angle_new = np.dot(vector_OH1_new, vector_OH2_new) / (np.linalg.norm(vector_OH1_new) * np.linalg.norm(vector_OH2_new))
        angle_rad_new = np.arccos(cos_angle_new)
        angle_deg_new = np.degrees(angle_rad_new)
        assert np.isclose(angle_deg, angle_deg_new, atol=1e-5), f"Angle changed for variation {variation}: {angle_deg} vs {angle_deg_new}"
        # Create new atom string for this variation
        atom_str = f"O {O_coords[0]:.4f} {O_coords[1]:.4f} {O_coords[2]:.4f}; "
        atom_str += f"H {new_H_coords[0][0]:.4f} {new_H_coords[0][1]:.4f} {new_H_coords[0][2]:.4f}; "
        atom_str += f"H {new_H_coords[1][0]:.4f} {new_H_coords[1][1]:.4f} {new_H_coords[1][2]:.4f}"
        dist_list_h2o.append(atom_str)
        # print(f"Variation {variation*100:.0f}%: {atom_str} with angle {angle_deg_new:.2f} degrees")
    
    # Get me a list of the translated bond length O-H for each variation to use as dist in the file names for H2O
    dist_list_h2o_bond_lengths = []
    for atom_str in dist_list_h2o:
        coords = []
        for line in atom_str.split(";"):
            parts = line.strip().split()
            element = parts[0]
            x, y, z = map(float, parts[1:])
            coords.append((element, x, y, z))
        O_coords = [x for x in coords if x[0] == "O"][0][1:]  # Get O coordinates
        H_coords = [x for x in coords if x[0] == "H"]  # Get H coordinates
        bond_length = np.sqrt((O_coords[0] - H_coords[0][1])**2 + (O_coords[1] - H_coords[0][2])**2 + (O_coords[2] - H_coords[0][3])**2)
        dist_list_h2o_bond_lengths.append(bond_length)

    print("\nGenerated H2O geometries with varying O-H bond lengths while keeping the H-O-H angle fixed:")
    for i, atom_str in enumerate(dist_list_h2o):
        print(f"Variation {i+1:2d}: {atom_str} | O-H bond length = {dist_list_h2o_bond_lengths[i]:.4f} Å")


    # Make the files for H2O if they don't exist
    atom = "H2O"
    molecule = "H2O"
    basis_lst = ["6-31G", "cc-pVDZ"]
    basis = basis_lst[0]

    for i, atom_str in enumerate(dist_list_h2o):
        dist = dist_list_h2o_bond_lengths[i]
        dist = round(dist, 4)
        # Make the VQE folders for dist if they don't exist
        if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/dist/{dist}"):
            os.makedirs(f"backup/data/{atom}/{basis}/VQE/dist/{dist}")
        for method in ["OVOS", "UHF", "UMP2"]:
            if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}"):
                os.makedirs(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}")
    
    # Make the nuclear repulsion energy files for H2O if they don't exist
    for i, atom_str in enumerate(dist_list_h2o):
        dist = dist_list_h2o_bond_lengths[i]
        dist = round(dist, 4)
        # set up molecule
        mol = gto.Mole()
        mol.atom = atom_str
        mol.basis = basis_lst[0]
        mol.unit = 'Angstrom'
        mol.spin = 0
        mol.charge = 0
        mol.symmetry = False
        mol.verbose = 0
        mol.build()
        
        # Get nuclear repulsion energy  
        nuc_rep_energy = mol.energy_nuc()

        # Save nuclear repulsion energy for later comparison
        name_nuc_rep = f"backup/data/{atom}/{basis}/VQE/UHF/{dist}/nuclear_repulsion_{molecule}_{basis_lst[0]}_{dist}_energy.txt"
        if not os.path.exists(name_nuc_rep):
            with open(name_nuc_rep, "w") as f:
                f.write(f"{nuc_rep_energy:.6f}\n")
            print(f"Nuclear repulsion energy for {atom_str} at dist {dist} saved to {name_nuc_rep}.")
            skip_nuc_rep_calculation = False
        else:
            skip_nuc_rep_calculation = True
    if skip_nuc_rep_calculation == True:       
        print(f"Nuclear repulsion energy files already exists for {molecule} skipping calculation.")

    # Make the UHF/RHF reference energy files for H2O if they don't exist
    for i, atom_str in enumerate(dist_list_h2o):
        dist = dist_list_h2o_bond_lengths[i]
        dist = round(dist, 4)
        for hf in ["UHF", "RHF"]:
            # set up molecule
            mol = gto.Mole()
            mol.atom = atom_str
            mol.basis = basis_lst[0]
            mol.unit = 'Angstrom'
            mol.spin = 0
            mol.charge = 0
            mol.symmetry = False
            mol.verbose = 0
            mol.build()
            
            # Get reference energies  
            if hf == "UHF":
                hf_energy = mol.UHF().run().e_tot
            else:
                hf_energy = mol.RHF().run().e_tot

            # Save HF reference energy for later comparison
            name_hf = f"backup/data/{atom}/{basis}/VQE/UHF/{dist}/{hf}_{molecule}_{basis_lst[0]}_{dist}_reference_energy.txt"
            if not os.path.exists(name_hf):
                with open(name_hf, "w") as f:
                    f.write(f"{hf_energy:.6f}\n")
                print(f"HF reference energy for {atom_str} at dist {dist} saved to {name_hf}.")
                skip_hf_calculation = False
            else:
                skip_hf_calculation = True
    if skip_hf_calculation == True:
        print(f"HF reference energy file already exists for {molecule}, skipping calculation.")

    # Run the VQE optimizations for H2O for all dist variations for one seed to verify the data looks correct for one seed before running the rest of the seeds in parallel over dist variations
    oo_lst = [True, False]
    # seed = 9
    seed_list = [42, 123, 14, 10, 20, 21, 101, 404, 8, 13]

    args_list = []
    for i, atom_str in enumerate(dist_list_h2o):
        dist = dist_list_h2o_bond_lengths[i]
        dist = round(dist, 4)
        for basis in [basis_lst[0]]:
            for num_opt_virtual_orbs in [0.75]: #[num_opt_virtual_orbs_lst[0]]: # 0.25,0.5,0.75
                for oo in [oo_lst[1]]: # True, False
                    for seed in seed_list:
                        print(f"{i+1:2d} / {len(dist_list_h2o)}: {seed:3d}, Prep. VQE runs for O-H bond length {dist:.4f} Å, with {num_opt_virtual_orbs*100:.0f}% active virtual orbitals and orbital opt. = {oo}...")
                        # Args for each run: atom string, basis, dist, num_opt_virtual_orbs, oo, seed
                            # atom, molecule, basis, dist, num_opt_virtual_orbs, oo, seed = args
                        args_list.append((atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed))

                        # For debug try without parallelization first to verify the data looks correct for one seed before running the rest of the seeds in parallel over dist variations
                        # VQE_OVOS(atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed)
    return args_list

def run_single(args):
    """Wrapper function for multiprocessing.dummy.Pool"""
    atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed = args
    return VQE_OVOS(atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed)

# Run
if __name__ == "__main__":
    # H2O
    args_list = run_h2o_vqe()

    num_cores = os.cpu_count()
    # Use 80% of cores to leave headroom for OS/PySCF background tasks
    num_workers = 10 #max(1, int(num_cores * 0.8))
    
    print(f"\nSystem: {num_cores} cores available")
    print(f"Starting {num_workers} parallel VQE workers...")
    print(f"Each worker runs on its own CPU core (process-based, no GIL)\n")
    
    # True multiprocessing - each worker = separate Python process
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_single, args_list))
    
    print(f"✓ Completed {len(results)} VQE runs")
    print(f"Performance: ~{len(args_list)/num_workers:.1f} runs per core on average")

    # # Run in serial
    # for args in args_list:
    #     run_single(args)






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