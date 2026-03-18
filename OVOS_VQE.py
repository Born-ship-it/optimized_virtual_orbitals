"""
Try OVOS with VQE and see if it can find the ground state energy of a molecule.
"""
    # Math and general imports
import numpy as np
import time
    # SlowQuant imports
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
    # Pyscf imports
import pyscf
    # OVOS imports
from ovos_clean import OVOS


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


# Define the molecule
atom = "O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692;",  # H2O equilibrium geometry
basis = "6-31G"

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

    # Number of electrons and orbitals
num_electrons = mol.nelectron
num_orbitals = mol.nao_nr() 
print(f"Number of electrons: {num_electrons}, Number of orbitals: {num_orbitals}")

# RHF reference
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()

# Create OVOS object and run
    # Start Timer OVOS
start_ovos = time.time()

    # Initial data (RHF orbitals)
Fao = [mf.get_fock(), mf.get_fock()]
mo_coeffs = [mf.mo_coeff, mf.mo_coeff]

    # Set up OVOS
ovos = OVOS(
    mol=mol,
    scf=mf,
    Fao=Fao,
    num_opt_virtual_orbs=2,      # active virtual spin‑orbitals
    mo_coeff=mo_coeffs,
    init_orbs="RHF",
    verbose=1,
    max_iter=1000,
    conv_energy=1e-8,
    conv_grad=1e-6,
    keep_track_max=50
)
    # Run OVOS
E_corr, E_corr_hist, E_corr_iter, E_corr_mo, E_corr_fock, stop_reason = ovos.run(mo_coeffs, fock_spin=None)
E_tot = E_corr + rhf.e_tot
print(f"\nOptimization finished. Final MP2 energy = {E_tot} Hartree. (ΔE_corr = {E_corr} Hartree)")

# SlowQuant
"""
Initialize for UCC wave function.

Args:
    cas: CAS(num_active_elec, num_active_orbs),
            orbitals are counted in spatial basis.
    mo_coeffs: Initial orbital coefficients.
    integral_generator: Integral generator object.
    excitations: Unitary coupled cluster excitation operators.
    include_active_kappa: Include active-active orbital rotations.
"""
    # Initialize for UCC wave function with OVOS-optimized orbitals
WF_ovos = WaveFunctionUCC(
    (num_electrons, num_orbitals),
    E_corr_mo,
    mol,
    "SD",
)
    # Get OVOS energy
E_ovos = WF_ovos.get_energy(E_corr_mo)
print(f"OVOS energy = {E_ovos} Hartree.")
    # End Timer OVOS
end_ovos = time.time()
print(f"OVOS optimization took {end_ovos - start_ovos:.2f} seconds.")

    # Optimize WF
WF_ovos.run_wf_optimization_1step("BFGS", True)
        # Get optimized energy
E_ovos_opt = WF_ovos.get_energy(E_corr_mo)
print(f"OVOS optimized energy = {E_ovos_opt} Hartree.")

    # End Timer OVOS w. WF optimization
end_ovos_w_opt = time.time()
print(f"OVOS optimization took {end_ovos_w_opt - start_ovos:.2f} seconds.")


# Compare with RHF reference
    # Start Timer RHF
start_rhf = time.time()

    # Initialize for UCC wave function with RHF orbitals
WF_rhf = WaveFunctionUCC(
    (num_electrons, num_orbitals),
    mf.mo_coeff,
    mol,
    "SD",
)
        # Get RHF energy
E_rhf = WF_rhf.get_energy(mf.mo_coeff)
print(f"RHF energy = {E_rhf} Hartree.")
    # End Timer RHF
end_rhf = time.time()
print(f"RHF initialization took {end_rhf - start_rhf:.2f} seconds.")

    # Optimize WF
WF_ovos.run_wf_optimization_1step("BFGS", True)
        # Get optimized energy
E_rhf_opt = WF_rhf.get_energy(mf.mo_coeff)
print(f"RHF optimized energy = {E_rhf_opt} Hartree.")

    # End Timer RHF w. WF optimization
end_rhf_w_opt = time.time()
print(f"RHF initialization took {end_rhf_w_opt - start_rhf:.2f} seconds.")



# Summary of results
print("\nSummary of results:")
print(f"OVOS correlation energy: {E_tot:.6f} Hartree")
print(f"OVOS energy: {E_ovos:.6f} Hartree (Time: {end_ovos - start_ovos:.2f} seconds)")
print(f"OVOS optimized energy: {E_ovos_opt:.6f} Hartree (Time: {end_ovos_w_opt - start_ovos:.2f} seconds)")
print(f"RHF energy: {E_rhf:.6f} Hartree (Time: {end_rhf - start_rhf:.2f} seconds)")
print(f"RHF optimized energy: {E_rhf_opt:.6f} Hartree (Time: {end_rhf_w_opt - end_rhf:.2f} seconds)")