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
from pyscf import gto, scf, mp, mcscf
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
        else:
            # No iterations, optimization terminated immediately
                # Set iter_energies to just the final energy if not already set from the progress table
            if 'iter_energies' not in stats and stats['final_energy'] is not None:
                stats['iter_energies'] = [stats['final_energy']]


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



def VQE_OVOS(atom, molecule, basis, dist, num_opt_virtual_orbs, oo, seed, thetas, thetas_bool=False):
    # name_out = f"backup/data/{molecule}/{basis}/VQE/dist/{dist}/OVOS_{molecule}_{dist}_{basis}_VQE_opt_num_{num_opt_virtual_orbs}_{oo}_{seed}_output.txt"
    # if not os.path.exists(os.path.dirname(name_out)):
    #     os.makedirs(os.path.dirname(name_out))

    # with open(name_out, "w") as f:
    #     # sys.stdout = Dee(sys.__stdout__, f)
    #     original_stdout = sys.stdout

        # Setup logging
    logger, name_out = setup_logging_in_function(seed, dist, molecule, basis, num_opt_virtual_orbs, oo)
    set_logger(logger)

    # Change seed to True if thetas_boll is True to use thetas as input instead of random initialization
    if thetas_bool == True:
        log_print(f"Using provided thetas for seed {seed} instead of random initialization.")
        seed = "True"  # Just to indicate in the logs that we are using provided thetas instead of random initialization


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
                max_iter=1000,
                conv_energy=1e-15,
                conv_grad=1e-4,
                keep_track_max=50
            )
                # Run OVOS
            E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)
            E_corr = E_corr  # Final correlation energy
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
            atol = 1e-6
                # Use same random seed and initial thetas for fair comparison
                    # Initialize thetas randomly for reproducibility
            if thetas_bool == False:
                np.random.seed(seed)
                thetas = (2*np.pi*np.random.random(len(WF_ovos.thetas)) - np.pi).tolist()       

            if thetas_bool == False:
                WF_ovos.thetas = thetas
            else:
                WF_ovos.thetas = thetas[0]
            
            
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
            if thetas_bool == False:
                WF_uhf.thetas = thetas
            else:
                WF_uhf.thetas = thetas[1]

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
            mf_ump2 = mol.UHF(verbose=0).run()
            ump2_obj = mp.UMP2(mf_ump2).run()
            noons, mp2_no_coeff = mcscf.addons.make_natural_orbitals(ump2_obj)

                # Energies for reference
            uhf_energy = mf_ump2.e_tot
            ump2_energy = ump2_obj.e_tot

            # Convert to full spin-orbital basis by duplicating alpha and beta orbitals
                # If rhf-like, just duplicate the same orbitals for alpha and beta
            if isinstance(mp2_no_coeff, np.ndarray) and mp2_no_coeff.ndim == 2:
                # Restricted case: duplicate the same orbitals for alpha and beta
                mp2_no_coeff = [mp2_no_coeff, mp2_no_coeff.copy()]
            elif isinstance(mp2_no_coeff, (list, tuple)) and len(mp2_no_coeff) == 2:
                # Already unrestricted, so we can use the alpha and beta orbitals as they are
                pass
            else:
                raise ValueError("Unexpected case for UMP2 natural orbitals: neither RHF-like nor UHF-like. Please check the orbitals.")

            # check if mp2_no_coeff are unrestricted or restricted
            if np.isclose(mp2_no_coeff[0], mp2_no_coeff[1], atol=1e-12).all():
                log_print("UMP2 natural orbitals are restricted (RHF-like).")
            else:
                log_print("UMP2 natural orbitals are unrestricted (UHF-like).")

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
            if thetas_bool == False:
                WF_ump2_no.thetas = thetas
            else:
                WF_ump2_no.thetas = thetas[2]

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
                    "ump2_energy": ump2_energy,
                    "ump2_no_energy": E_ump2_no_hist[0],
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
                "thetas_final": [WF_ovos.thetas, WF_uhf.thetas, WF_ump2_no.thetas], 
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
    # HF, CO, NH3, H2O, Li2
# atom_1 = "H 0 0 0; F 0 0 0.917" # HF bond length 0.917 Angstrom
# atom_2 = "N 0 0 0; H 0 0 1.012; H 0 0.935 -0.262; H 0 -0.935 -0.262" # NH3 equilibrium geometry	
# atom_3 = "O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;"  # H2O equilibrium geometry
# atom_4 = "Li 0 0 0; Li 0 0 2.673" # Li2 bond length 2.673 Angstrom
# atom_5 = "C 0 0 0; O 0 0 1.128" # CO bond length 1.128 Angstrom

    # Lst
basis_lst = ["6-31G", "cc-pVDZ"] # "6-31G", "cc-pVDZ"
num_opt_virtual_orbs_lst = [0.75]
oo_lst = [True, False]

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


# HF - DONE, 30
def run_hf_vqe():
    # Let us explore the dist list we need for HF
        # Varying the bond length around the equilibrium geometry:
            # 'H 0 0 0; F 0 0 0.917'
    
    dist_list = np.arange(0.675, 2.025, 0.025)  # From 0.7 to 2.0 Angstrom in steps of 0.05 for faster testing
    dist_list = [round(d, 3) for d in dist_list]  # Round to 4 decimals for cleaner output
    
    # Make the files for HF if they don't exist
    atom = "HF"
    molecule = "HF"
    basis_lst = ["6-31G", "cc-pVDZ"]
    basis = basis_lst[0]

    for dist in dist_list:
        # Make the VQE folders for dist if they don't exist
        if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/dist/{dist}"):
            os.makedirs(f"backup/data/{atom}/{basis}/VQE/dist/{dist}")
        for method in ["OVOS", "UHF", "UMP2"]:
            if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}"):
                os.makedirs(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}")

    # Make the nuclear repulsion energy files for HF if they don't exist
    for dist in dist_list:
        # set up molecule
        mol = gto.Mole()
        mol.atom = f"H 0 0 0; F 0 0 {dist}"
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
            print(f"Nuclear repulsion energy for {atom} at dist {dist} saved to {name_nuc_rep}.")
            skip_nuc_rep_calculation = False
        else:
            skip_nuc_rep_calculation = True
    if skip_nuc_rep_calculation == True:       
        print(f"Nuclear repulsion energy files already exists for {molecule} skipping calculation.")

    # Make the UHF/RHF reference energy files for HF if they don't exist
    for dist in dist_list:
        for hf in ["UHF", "RHF"]:
            # set up molecule
            mol = gto.Mole()
            mol.atom = f"H 0 0 0; F 0 0 {dist}"
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
                print(f"HF reference energy for {atom} at dist {dist} saved to {name_hf}.")
                skip_hf_calculation = False
            else:
                skip_hf_calculation = True
    if skip_hf_calculation == True:
        print(f"HF reference energy file already exists for {molecule}, skipping calculation.")

    # Run the VQE optimizations for HF for all dist variations for one seed to verify the data looks correct for one seed before running the rest of the seeds in parallel over dist variations
    oo_lst = [True, False]
    # seed_list = [8]
    seed_list = [42] 
    # Have already:
        # 8, 9, 10, 13, 14, 20, 21, 42, 101, 109, 119, 123, 129, 139, 404
            # A total of 16...
        # 14 more seeds to run for a total of 30 seeds per dist variation, which should give us a good picture of the distribution of results for each dist variation.

    args_list = []
    for dist in dist_list:
        atom_str = f"H 0 0 0; F 0 0 {dist}"
        for basis in [basis_lst[0]]:
            for num_opt_virtual_orbs in [0.75]: #[num_opt_virtual_orbs_lst[0]]: # 0.25,0.5,0.75
                for oo in [oo_lst[1]]: # True, False
                    for seed in seed_list:
                        print(f"{dist:.3f} Å: {seed:3d}, Prep. VQE runs for H-F bond length {dist:.3f} Å, with {num_opt_virtual_orbs*100:.0f}% active virtual orbitals and orbital opt. = {oo}...")
                        # Args for each run: atom string, basis, dist, num_opt_virtual_orbs, oo, seed
                            # atom, molecule, basis, dist, num_opt_virtual_orbs, oo, seed = args
                        args_list.append((atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed))

                        # For debug try without parallelization first to verify the data looks correct for one seed before running the rest of the seeds in parallel over dist variations
                        # VQE_OVOS(atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed)
    return args_list


# H2O - 
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
    bond_length_set = np.arange(0.675, 2.025, 0.025)
            # Figure out the procentages we need to get the bond_length_set values
    bond_length_variations = [bl / bond_lengths[0] for bl in bond_length_set]
                # Round off to 3 decimals
    bond_length_variations = [round(blv, 3) for blv in bond_length_variations]

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
        dist = round(dist, 3)
        # Make the VQE folders for dist if they don't exist
        if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/dist/{dist}"):
            os.makedirs(f"backup/data/{atom}/{basis}/VQE/dist/{dist}")
        for method in ["OVOS", "UHF", "UMP2"]:
            if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}"):
                os.makedirs(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}")
    
    # Make the nuclear repulsion energy files for H2O if they don't exist
    for i, atom_str in enumerate(dist_list_h2o):
        dist = dist_list_h2o_bond_lengths[i]
        dist = round(dist, 3)
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
        dist = round(dist, 3)
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
    seed_list = [42] #, 91, 64, 111, 128, 256, 303, 512, 1024, 2048, 4096, 8192, 16384]
    # Done: 8, 10, 13, 14, 20, 21, 42, 101, 123, 404
        # Total: 16 seeds

    args_list = []
    for i, atom_str in enumerate(dist_list_h2o):
        dist = dist_list_h2o_bond_lengths[i]
        dist = round(dist, 3)
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


# CO
def run_co_vqe():
    # Let us explore the dist list we need for CO: 'C 0 0 0; O 0 0 1.128'
        # A diatomic molecule,
            # so the dist will be the bond length between C and O,
            # which we can vary around the equilibrium bond length of 1.128 Angstrom.
    # dist_list = np.arange(0.7, 2.025, 0.025).round(3).tolist()  # From 0.7 to 2.0 in steps of 0.025
        # Trial dist list
    dist_list = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    
    print("\nGenerated CO geometries with varying C-O bond lengths:")
    for dist in dist_list:
        atom_str = f"C 0 0 0; O 0 0 {dist:.3f}"
        print(f"Bond length {dist:.3f} Å: {atom_str}")

    # Make the files for CO if they don't exist
    atom = "CO"
    molecule = "CO"
    basis_lst = ["6-31G", "cc-pVDZ"]
    basis = basis_lst[0]

    for dist in dist_list:
        # Make the VQE folders for dist if they don't exist
        if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/dist/{dist}"):
            os.makedirs(f"backup/data/{atom}/{basis}/VQE/dist/{dist}")
        for method in ["OVOS", "UHF", "UMP2"]:
            if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}"):
                os.makedirs(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}")

    # Make the nuclear repulsion energy files for CO if they don't exist
    for dist in dist_list:
        atom_str = f"C 0 0 0; O 0 0 {dist:.3f}"
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
        name_nuc_rep = f"backup/data/{atom}/{basis}/VQE/UHF/{dist}/nuclear_repulsion_{molecule}_{basis}_{dist}_energy.txt"
        if not os.path.exists(name_nuc_rep):
            with open(name_nuc_rep, "w") as f:
                f.write(f"{nuc_rep_energy:.6f}\n")
            print(f"Nuclear repulsion energy for {atom_str} at dist {dist} saved to {name_nuc_rep}.")
            skip_nuc_rep_calculation = False
        else:
            skip_nuc_rep_calculation = True
    if skip_nuc_rep_calculation == True:
        print(f"Nuclear repulsion energy files already exists for {molecule} skipping calculation.")

    # Make the UHF/RHF reference energy files for CO if they don't exist
    for dist in dist_list:
        atom_str = f"C 0 0 0; O 0 0 {dist:.3f}"
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
            name_hf = f"backup/data/{atom}/{basis}/VQE/UHF/{dist}/{hf}_{molecule}_{basis}_{dist}_reference_energy.txt"
            if not os.path.exists(name_hf):
                with open(name_hf, "w") as f:
                    f.write(f"{hf_energy:.6f}\n")
                print(f"HF reference energy for {atom_str} at dist {dist} saved to {name_hf}.")
                skip_hf_calculation = False
            else:
                skip_hf_calculation = True
    if skip_hf_calculation == True:
        print(f"HF reference energy file already exists for {molecule}, skipping calculation.")

    # Run the VQE optimizations for CO for all dist variations for one seed to verify the data looks correct for one seed before running the rest of the seeds in parallel over dist variations
    oo_lst = [True, False]
    seed_list = [9] # Trial seed, verify...
    # seed_list = [42, 123, 14, 10, 20, 21, 101, 404, 8, 13]

    args_list = []
    for dist in dist_list:
        atom_str = f"C 0 0 0; O 0 0 {dist:.3f}"
        for basis in [basis_lst[0]]:
            for num_opt_virtual_orbs in [0.75]: #[num_opt_virtual_orbs_lst[0]]: # 0.25,0.5,0.75
                for oo in [oo_lst[1]]: # True, False
                    for seed in seed_list:
                        print(f"{dist:.3f} Å: {seed:3d}, Prep. VQE runs for C-O bond length {dist:.3f} Å, with {num_opt_virtual_orbs*100:.0f}% active virtual orbitals and orbital opt. = {oo}...")
                        # Args for each run: atom string, basis, dist, num_opt_virtual_orbs, oo, seed
                            # atom, molecule, basis, dist, num_opt_virtual_orbs, oo, seed = args
                        args_list.append((atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed))

                        # For debug try without parallelization first to verify the data looks correct for one seed before running the rest of the seeds in parallel over dist variations
                        # VQE_OVOS(atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed)
    return args_list


# NH3
def run_nh3_vqe():
    # Let us explore the dist list we need for NH3: 'N 0 0 0; H 0 0 1.012; H 0 0.935 -0.262; H 0 -0.935 -0.262'
        # We can vary the N-H bond length around the equilibrium geometry while keeping the H-N-H angles fixed.
        # We can calculate the N-H bond length and H-N-H angle from the equilibrium geometry
        # Then we can vary the N-H bond length while keeping the H-N-H angles fixed by scaling the H coordinates accordingly, similar to what we did for H2O.
    
    # Get equilibrium geometry coordinates
    str_equi = 'N 0 0 0; H 0 0 1.012; H 0 0.935 -0.262; H 0 -0.935 -0.262'
    coords = []
    for line in str_equi.split(";"):
        parts = line.strip().split()
        element = parts[0]
        x, y, z = map(float, parts[1:])
        coords.append((element, x, y, z))
    N_coords = [x for x in coords if x[0] == "N"][0][1:]  # Get N coordinates
    H_coords = [x for x in coords if x[0] == "H"]  # Get H coordinates
    bond_lengths = [] # Equilibrium N-H bond lengths
    for H in H_coords:
        bond_length = np.sqrt((N_coords[0] - H[1])**2 + (N_coords[1] - H[2])**2 + (N_coords[2] - H[3])**2)
        bond_lengths.append(bond_length)
    H1_coords = H_coords[0][1:]
    H2_coords = H_coords[1][1:]
    H3_coords = H_coords[2][1:]
    vector_NH1 = np.array(H1_coords) - np.array(N_coords)
    vector_NH2 = np.array(H2_coords) - np.array(N_coords)
    vector_NH3 = np.array(H3_coords) - np.array(N_coords)
    cos_angle_1 = np.dot(vector_NH1, vector_NH2) / (np.linalg.norm(vector_NH1) * np.linalg.norm(vector_NH2))
    cos_angle_2 = np.dot(vector_NH1, vector_NH3) / (np.linalg.norm(vector_NH1) * np.linalg.norm(vector_NH3))
    cos_angle_3 = np.dot(vector_NH2, vector_NH3) / (np.linalg.norm(vector_NH2) * np.linalg.norm(vector_NH3))
    angle_1_rad = np.arccos(cos_angle_1)
    angle_2_rad = np.arccos(cos_angle_2)
    angle_3_rad = np.arccos(cos_angle_3)
    angle_1_deg = np.degrees(angle_1_rad)
    angle_2_deg = np.degrees(angle_2_rad)
    angle_3_deg = np.degrees(angle_3_rad)
    print(f"Equilibrium N-H bond lengths: {bond_lengths}")
    print(f"Equilibrium H-N-H angles: {angle_1_deg:.2f}, {angle_2_deg:.2f}, {angle_3_deg:.2f} degrees")

    # Now we can vary the N-H bond length while keeping the angles fixed by scaling to the following bond length variations around the equilibrium geometry, e.g. from 0.75 to 1.25 times the equilibrium bond length
    bond_length_set = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        # Figure out the procentages we need to get the bond_length_set values
    bond_length_variations = [bl / bond_lengths[0] for bl in bond_length_set]
        # Round off to 3 decimals
    bond_length_variations = [round(blv, 3) for blv in bond_length_variations]

    dist_list_nh3 = []
        # Assert that the angles are the same for all variations
    for variation in bond_length_variations:
        new_H_coords = []
        for H in H_coords:
            vector_NH = np.array(H[1:]) - np.array(N_coords)
            new_vector_NH = vector_NH * variation  # Scale the vector to change the bond length
            new_H_coord = np.array(N_coords) + new_vector_NH  # Get new H coordinates
            new_H_coords.append(new_H_coord)
        # Check angles are the same
        vector_NH1_new = new_H_coords[0] - np.array(N_coords)
        vector_NH2_new = new_H_coords[1] - np.array(N_coords)
        vector_NH3_new = new_H_coords[2] - np.array(N_coords)
        cos_angle_1_new = np.dot(vector_NH1_new, vector_NH2_new) / (np.linalg.norm(vector_NH1_new) * np.linalg.norm(vector_NH2_new))
        cos_angle_2_new = np.dot(vector_NH1_new, vector_NH3_new) / (np.linalg.norm(vector_NH1_new) * np.linalg.norm(vector_NH3_new))
        cos_angle_3_new = np.dot(vector_NH2_new, vector_NH3_new) / (np.linalg.norm(vector_NH2_new) * np.linalg.norm(vector_NH3_new))
        angle_1_rad_new = np.arccos(cos_angle_1_new)
        angle_2_rad_new = np.arccos(cos_angle_2_new)
        angle_3_rad_new = np.arccos(cos_angle_3_new)
        angle_1_deg_new = np.degrees(angle_1_rad_new)
        angle_2_deg_new = np.degrees(angle_2_rad_new)
        angle_3_deg_new = np.degrees(angle_3_rad_new)
        assert np.isclose(angle_1_deg, angle_1_deg_new, atol=1e-5), f"Angle 1 changed for variation {variation}: {angle_1_deg} vs {angle_1_deg_new}"
        assert np.isclose(angle_2_deg, angle_2_deg_new, atol=1e-5), f"Angle 2 changed for variation {variation}: {angle_2_deg} vs {angle_2_deg_new}"
        assert np.isclose(angle_3_deg, angle_3_deg_new, atol=1e-5), f"Angle 3 changed for variation {variation}: {angle_3_deg} vs {angle_3_deg_new}"
        # Create new atom string for this variation
        atom_str = f"N {N_coords[0]:.3f} {N_coords[1]:.3f} {N_coords[2]:.3f}; "
        atom_str += f"H {new_H_coords[0][0]:.3f} {new_H_coords[0][1]:.3f} {new_H_coords[0][2]:.3f}; "
        atom_str += f"H {new_H_coords[1][0]:.3f} {new_H_coords[1][1]:.3f} {new_H_coords[1][2]:.3f}; "
        atom_str += f"H {new_H_coords[2][0]:.3f} {new_H_coords[2][1]:.3f} {new_H_coords[2][2]:.3f}"
        dist_list_nh3.append(atom_str)
        print(f"Variation {variation*100:.0f}%: {atom_str} with angles {angle_1_deg_new:.2f}, {angle_2_deg_new:.2f}, {angle_3_deg_new:.2f} degrees")

    # Make the files for NH3 if they don't exist
    atom = "NH3"
    molecule = "NH3"
    basis_lst = ["6-31G", "cc-pVDZ"]
    basis = basis_lst[0]

    for i, atom_str in enumerate(dist_list_nh3):
        dist = bond_length_set[i]
        # Make the VQE folders for dist if they don't exist
        if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/dist/{dist}"):
            os.makedirs(f"backup/data/{atom}/{basis}/VQE/dist/{dist}")
        for method in ["OVOS", "UHF", "UMP2"]:
            if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}"):
                os.makedirs(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}")

    # Make the nuclear repulsion energy files for NH3 if they don't exist
    for i, atom_str in enumerate(dist_list_nh3):
        dist = bond_length_set[i]
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
        name_nuc_rep = f"backup/data/{atom}/{basis}/VQE/UHF/{dist}/nuclear_repulsion_{molecule}_{basis}_{dist}_energy.txt"
        if not os.path.exists(name_nuc_rep):
            with open(name_nuc_rep, "w") as f:
                f.write(f"{nuc_rep_energy:.6f}\n")
            print(f"Nuclear repulsion energy for {atom_str} at dist {dist} saved to {name_nuc_rep}.")
            skip_nuc_rep_calculation = False
        else:
            skip_nuc_rep_calculation = True
    if skip_nuc_rep_calculation == True:
        print(f"Nuclear repulsion energy files already exists for {molecule} skipping calculation.")

    # Make the UHF/RHF reference energy files for NH3 if they don't exist
    for i, atom_str in enumerate(dist_list_nh3):
        dist = bond_length_set[i]
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
            name_hf = f"backup/data/{atom}/{basis}/VQE/UHF/{dist}/{hf}_{molecule}_{basis}_{dist}_reference_energy.txt"
            if not os.path.exists(name_hf):
                with open(name_hf, "w") as f:
                    f.write(f"{hf_energy:.6f}\n")
                print(f"HF reference energy for {atom_str} at dist {dist} saved to {name_hf}.")
                skip_hf_calculation = False
            else:
                skip_hf_calculation = True
    if skip_hf_calculation == True:
        print(f"HF reference energy file already exists for {molecule}, skipping calculation.")

    # Run the VQE optimizations for NH3 for all dist variations for one seed to verify the data looks correct for one seed before running the rest of the seeds in parallel over dist variations
    oo_lst = [True, False]
    seed_list = [9] # Trial seed, verify...
    # seed_list = [42, 123, 14, 10, 20, 21, 101, 404, 8, 13]

    args_list = []
    for i, atom_str in enumerate(dist_list_nh3):
        dist = bond_length_set[i]
        for basis in [basis_lst[0]]:
            for num_opt_virtual_orbs in [0.75]: #[num_opt_virtual_orbs_lst[0]]: # 0.25,0.5,0.75
                for oo in [oo_lst[1]]: # True, False
                    for seed in seed_list:
                        print(f"{i+1:2d} / {len(dist_list_nh3)}: {seed:3d}, Prep. VQE runs for N-H bond length {dist:.3f} Å, with {num_opt_virtual_orbs*100:.0f}% active virtual orbitals and orbital opt. = {oo}...")
                        # Args for each run: atom string, basis, dist, num_opt_virtual_orbs, oo, seed
                            # atom, molecule, basis, dist, num_opt_virtual_orbs, oo, seed = args
                        args_list.append((atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed))

                        # For debug try without parallelization first to verify the data looks correct for one seed before running the rest of the seeds in parallel over dist variations
                        # VQE_OVOS(atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed)
    return args_list


# Li2
def run_li2_vqe():
    # Let us explore the dist list we need for Li2: "Li .0 .0 .0; H .0 .0 1.595"
        # A diatomic molecule,
            # so the dist will be the bond length between the two Li atoms,
            # which we can vary around the equilibrium bond length of 1.6 Angstrom.
    
    # Trial dist list
    dist_list = np.arange(2.4, 6.1, 0.1).round(1).tolist()  # Trial

    print("\nGenerated Li2 geometries with varying Li-Li bond lengths:")
    for dist in dist_list:
        atom_str = f"Li 0 0 0; Li 0 0 {dist:.3f}"
        print(f"Bond length {dist:.3f} Å: {atom_str}")
    
    # Make the files for Li2 if they don't exist
    atom = "Li2"
    molecule = "Li2"
    basis_lst = ["6-31G", "cc-pVDZ"]
    basis = basis_lst[0]

    # for dist in dist_list:
    #     # Make the VQE folders for dist if they don't exist
    #     if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/dist/{dist}"):
    #         os.makedirs(f"backup/data/{atom}/{basis}/VQE/dist/{dist}")
    #     for method in ["OVOS", "UHF", "UMP2"]:
    #         if not os.path.exists(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}"):
    #             os.makedirs(f"backup/data/{atom}/{basis}/VQE/{method}/{dist}")

    # Make the nuclear repulsion energy files for Li2 if they don't exist
    # for dist in dist_list:
    #     atom_str = f"Li 0 0 0; Li 0 0 {dist:.3f}"
    #     # set up molecule
    #     mol = gto.Mole()
    #     mol.atom = atom_str
    #     mol.basis = basis_lst[0]
    #     mol.unit = 'Angstrom'
    #     mol.spin = 0
    #     mol.charge = 0
    #     mol.symmetry = False
    #     mol.verbose = 0
    #     mol.build()
        
    #     # Get nuclear repulsion energy  
    #     nuc_rep_energy = mol.energy_nuc()

    #     # Save nuclear repulsion energy for later comparison
    #     name_nuc_rep = f"backup/data/{atom}/{basis}/VQE/UHF/{dist}/nuclear_repulsion_{molecule}_{basis}_{dist}_energy.txt"
    #     if True: # not os.path.exists(name_nuc_rep):
    #         with open(name_nuc_rep, "w") as f:
    #             f.write(f"{nuc_rep_energy:.6f}\n")
    #         print(f"Nuclear repulsion energy for {atom_str} at dist {dist} saved to {name_nuc_rep}.")
    #         skip_nuc_rep_calculation = False
    #     else:
    #         skip_nuc_rep_calculation = True
    # if skip_nuc_rep_calculation == True:
    #     print(f"Nuclear repulsion energy files already exists for {molecule} skipping calculation.")

    # Make the UHF/RHF reference energy files for Li2 if they don't exist
    # for dist in dist_list:
    #     atom_str = f"Li 0 0 0; Li 0 0 {dist:.3f}"
    #     for hf in ["UHF", "RHF"]:
    #         # set up molecule
    #         mol = gto.Mole()
    #         mol.atom = atom_str
    #         mol.basis = basis_lst[0]
    #         mol.unit = 'Angstrom'
    #         mol.spin = 0
    #         mol.charge = 0
    #         mol.symmetry = False
    #         mol.verbose = 0
    #         mol.build()
            
    #         # Get reference energies  
    #         if hf == "UHF":
    #             hf_energy = mol.UHF().run().e_tot
    #         else:
    #             hf_energy = mol.RHF().run().e_tot

    #         # Save HF reference energy for later comparison
    #         name_hf = f"backup/data/{atom}/{basis}/VQE/UHF/{dist}/{hf}_{molecule}_{basis}_{dist}_reference_energy.txt"
    #         if True: # not os.path.exists(name_hf):
    #             with open(name_hf, "w") as f:
    #                 f.write(f"{hf_energy:.6f}\n")
    #             print(f"HF reference energy for {atom_str} at dist {dist} saved to {name_hf}.")
    #             skip_hf_calculation = False
    #         else:
    #             skip_hf_calculation = True
    # if skip_hf_calculation == True:
    #     print(f"HF reference energy file already exists for {molecule}, skipping calculation.")

    # Run the VQE optimizations for Li2 for all dist variations for one seed to verify the data looks correct for one seed before running the rest of the seeds in parallel over dist variations
    oo_lst = [True, False]
    seed_list = [42]
        # Done: 8, 9, 10, 13, 14, 20, 21, 42, 101, 123, 404, 32, 72, 111, 128, 64, 91, 256, 152, 303
            # Total seeds done:

    args_list = []
    for dist in dist_list:
        atom_str = f"Li 0 0 0; Li 0 0 {dist:.3f}"
        for basis in [basis_lst[0]]:
            for num_opt_virtual_orbs in [0.75]: #[num_opt_virtual_orbs_lst[0]]: # 0.25,0.5,0.75
                
                # CHANGE IT HERE TO RUN BOTH WITH AND WITHOUT ORBITAL OPTIMIZATION
                for oo in [oo_lst[0]]: # True, False

                    for seed in seed_list:
                        print(f"{dist:.3f} Å: {seed:3d}, Prep. VQE runs for Li-Li bond length {dist:.3f} Å, with {num_opt_virtual_orbs*100:.0f}% active virtual orbitals and orbital opt. = {oo}...")
                        # Args for each run: atom string, basis, dist, num_opt_virtual_orbs, oo, seed
                            # atom, molecule, basis, dist, num_opt_virtual_orbs, oo, seed = args
                        args_list.append((atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed))

                        # For debug try without parallelization first to verify the data looks correct for one seed before running the rest of the seeds in parallel over dist variations
                        # VQE_OVOS(atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed)
    return args_list
    


def run_single(args):
    """Wrapper function for multiprocessing.dummy.Pool"""
    atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed, thetas, thetas_bool = args

    return VQE_OVOS(atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed, thetas, thetas_bool)

# Run
if __name__ == "__main__":
    # Molecule: HF, H2O, CO, NH3, Li2
    # args_list = run_hf_vqe()  # HF,  Done 
    # args_list = run_h2o_vqe() # H2O, Done
    args_list = run_li2_vqe()   # 16...

    if False:
        # Set thetas to empty list... and thetas_bool to False
        args_list = [(atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed, [], False) for (atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed) in args_list]

        num_cores = os.cpu_count()
        num_workers = 10
        
        print(f"\nSystem: {num_cores} cores available")
        print(f"Starting {num_workers} parallel VQE workers...")
        print(f"Each worker runs on its own CPU core (process-based, no GIL)\n")
        
        # True multiprocessing - each worker = separate Python process
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(run_single, args_list))
        
        print(f"✓ Completed {len(results)} VQE runs")
        print(f"Performance: ~{len(args_list)/num_workers:.1f} runs per core on average")
    
    if True:
        # Run in serial to allow for prev. thetas to be used as initial guess for next run
        args_list = [(atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed, [], True) for (atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed) in args_list]
                # From the data file entry for "thetas" of the random runs
        atom_str = args_list[-1][0]
        molecule_name = args_list[-1][1]
        basis_name = args_list[-1][2]
        last_dist_in_dist_list = args_list[-1][3]
        num_opt_virtual_orbs = args_list[-1][4]     # %
        oo = args_list[-1][5]

        # Get num_orbitals and num_electrons 
        mol = gto.Mole()
        mol.atom = atom_str
        mol.basis = basis_name
        mol.unit = 'Angstrom'
        mol.spin = 0
        mol.charge = 0
        mol.symmetry = False
        mol.verbose = 0
        mol.build()

        num_orbitals = mol.nao_nr()  # Number of atomic orbitals
        num_electrons = mol.nelectron  # Number of electrons

        num_opt_virtual_orbs = int(num_opt_virtual_orbs * (num_orbitals - num_electrons//2))

        # Start random thetas for the first run, then use thetas from each run as initial guess for the next run
            # Get the length of the thetas list we need for the first run to generate random thetas of the correct length
        len_thetas = None
        previous_thetas = None

        file_name = f"backup/data/{molecule_name}/{basis_name}/VQE/OVOS/{last_dist_in_dist_list}/UPS_OVOS_{molecule_name}_{basis_name}_{last_dist_in_dist_list}_opt_num_{num_opt_virtual_orbs}_False_8.json"
        print(f"Looking for existing thetas in \n     {file_name} \n to determine length for random theta generation...")

        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                data = json.load(f)
                thetas_from_file = data.get("thetas", [])
                if thetas_from_file:
                    len_thetas = len(thetas_from_file)
                    print(f"Found existing thetas in {file_name}, using length {len_thetas} for random theta generation.")
                else:
                    print(f"No thetas found in {file_name}, will determine length from first run.")
        else:            
            assert False, "No existing file found to determine theta length from first run. Please run one VQE optimization with random thetas first to generate the file with thetas for the correct length, then run this script again to use those thetas as initial guess for the rest of the runs."

        # Clean start
        if False:
            np.random.seed(42)  # For reproducibility of random thetas
            thetas = (2*np.pi*np.random.random(len_thetas) - np.pi).tolist()       
            thetas = [thetas, thetas, thetas]  

            prev_dist = None

        # Continue start
            
            # Need to continue for Li2!!!!!!!!11 w. OO True...

        if True: # Found for oo = Ture each dist takes a while, need to be able to start from thetas from a previous dist to avoid having to run all dists sequentially from the start...
            # Dist to get prev. from
            oo_str = "False"
            prev_dist = 2.4
            # Get thetas from the file for the prev_dist
            file_name_prev = f"backup/data/{molecule_name}/{basis_name}/VQE/OVOS/{prev_dist}/UPS_OVOS_{molecule_name}_{basis_name}_{prev_dist}_opt_num_{num_opt_virtual_orbs}_{oo_str}_True.json"
            print(f"Looking for existing thetas in \n     {file_name_prev} \n to use as initial guess for the first run...")
            if os.path.exists(file_name_prev):
                with open(file_name_prev, "r") as f:
                    data = json.load(f)
                    thetas = data.get("thetas", [])
                    if thetas:
                        print(f"Found existing thetas in {file_name_prev}, using them as initial guess for the first run.")
                    else:
                        print(f"No thetas found in {file_name_prev}")
            else:
                assert False, f"No existing file found at {file_name_prev} to get thetas from. Please run a VQE optimization for dist {prev_dist} with orbital optimization first to generate the file with thetas, then run this script again to use those thetas as initial guess for the rest of the runs."

        # Run the VQE optimizations in serial, using thetas from each run as initial guess for the next run
        for args in args_list:
            # Skip the runs until we reach the dist we have thetas for, then start using thetas from each run as initial guess for the next run
            dist = args[3]
            if prev_dist is not None:
                if dist <= prev_dist: 
                    print(f"Skipping dist {dist} since done...")
                    continue
            atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed, _, _ = args
            print(f"Running VQE for {molecule} at dist {dist} with seed {seed}...")
            data_out = VQE_OVOS(atom_str, molecule, basis, dist, num_opt_virtual_orbs, oo, seed, thetas, True)
            thetas = data_out.get("thetas_final", thetas)  # Update thetas for the next run, if thetas_final is not in data_out, keep using the same thetas
            previous_thetas = thetas  # Use thetas from this run as initial guess for next run




# TO DO:
# - Run VQE PREV. THETAS w. OO False
    # H2O

# - Run VQE RANDOM for 5 seeds w. OO True
    # H2O, HF, Li2

# - Run VQE PREV. THETAS w. OO True
    # H2O, HF | continue












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