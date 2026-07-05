#!/usr/bin/env python3
"""
Compare OVOS methods with optional virtual orbital sorting.
If sorting_methods is empty, only canonical order is run.
"""

import time
import sys
import numpy as np
from pyscf import gto, scf, mp

from ovos import OVOS
from ovos_trust_region import OVOSTrustRegion
from ovos_gauss_seidel import OVOSGaussSeidel
from ovos_branch_bound import OVOSBranchBound

# ===========================================================================
# SORTING FUNCTIONS
# ===========================================================================
def get_virtual_weights_amplitude(mol, mf):
    mp2_obj = mp.MP2(mf)
    mp2_obj.verbose = 0
    mp2_obj.kernel()
    t2 = mp2_obj.t2
    nvir = t2.shape[2]
    weights = np.zeros(nvir)
    for a in range(nvir):
        weights[a] = np.sum(t2[:, :, a, :]**2)
    return weights

def get_virtual_weights_natural(mol, mf):
    mp2_obj = mp.MP2(mf)
    mp2_obj.verbose = 0
    mp2_obj.kernel()
    t2 = mp2_obj.t2
    nocc, nvir = t2.shape[0], t2.shape[2]
    D = np.zeros((nvir, nvir))
    for i in range(nocc):
        for j in range(i):
            for c in range(nvir):
                D += np.outer(t2[i, j, :, c], t2[i, j, :, c])
    eigvals, _ = np.linalg.eigh(D)
    return eigvals[::-1]

def get_sorted_mo_coeffs_and_perm(mol, mf, method='mp2_natural'):
    nocc = mol.nelec[0]
    nvir = mf.mo_coeff.shape[1] - nocc
    if method == 'mp2_weight':
        weights = get_virtual_weights_amplitude(mol, mf)
        vir_idx_sorted = np.argsort(weights)[::-1]
    elif method == 'mp2_natural':
        weights = get_virtual_weights_natural(mol, mf)
        vir_idx_sorted = np.argsort(weights)[::-1]
    elif method == 'mp2_natural_reverse':
        weights = get_virtual_weights_natural(mol, mf)
        vir_idx_sorted = np.argsort(weights)
    else:
        raise ValueError(f"Unknown sorting method: {method}")
    perm = list(range(nocc)) + [nocc + i for i in vir_idx_sorted]
    mo_coeff = mf.mo_coeff
    mo_sorted = mo_coeff[:, perm]
    return [mo_sorted, mo_sorted], perm

def get_active_indices(mol, perm, num_opt_virtual_orbs):
    nocc = mol.nelec[0]
    active_positions = range(nocc, nocc + num_opt_virtual_orbs)
    return sorted([perm[pos] for pos in active_positions])

def get_canonical_active_indices(mol, num_opt_virtual_orbs):
    nocc = mol.nelec[0]
    return list(range(nocc, nocc + num_opt_virtual_orbs))

# ===========================================================================
# TEST RUNNER
# ===========================================================================
def setup_molecule(basis='6-31G'):
    mol = gto.Mole()
    mol.atom = 'O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692'
    mol.basis = basis
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mol, mf

def print_orbital_info(mol, num_opt_virtual_orbs, e_corr_full):
    nelec = mol.nelec[0] + mol.nelec[1]
    n_spatial = mol.nao_nr()
    n_spin = 2 * n_spatial
    n_occ = nelec
    n_virt_act = num_opt_virtual_orbs
    n_virt_inact = n_spin - n_occ - n_virt_act
    grad_dim = n_virt_act * n_virt_inact
    print("="*60)
    print(f"Orbital space details for {mol.basis} basis")
    print("="*60)
    print(f"  Number of spin-orbitals (total):     {n_spin}")
    print(f"  Occupied spin-orbitals:              {n_occ}")
    print(f"  Active virtual spin-orbitals:        {n_virt_act}")
    print(f"  Inactive virtual spin-orbitals:      {n_virt_inact}")
    print(f"  Optimisation parameters (gradient):  {grad_dim}")
    print("-"*60)
    print(f"  Full MP2 correlation energy (all virtuals):  {e_corr_full:.10f} Ha")
    print("="*60)

def run_method(cls, mol, mf, mo_coeffs, num_opt_virtual_orbs, **kwargs):
    Fao = [mf.get_fock(), mf.get_fock()]
    params = {
        'verbose': 0,
        'max_iter': 100,
        'conv_energy': 1e-8,
        'conv_grad': 1e-6,
        'keep_track_max': 10,
        'num_opt_virtual_orbs': num_opt_virtual_orbs,
    }
    params.update(kwargs)
    ovos_obj = cls(mol=mol, scf=mf, Fao=Fao, mo_coeff=mo_coeffs, **params)
    start = time.time()
    result = ovos_obj.run(mo_coeffs, fock_spin=None)
    elapsed = time.time() - start
    return {
        'energy': result[0],
        'iterations': len(result[1]),
        'time': elapsed,
        'stop_reason': result[-1],
    }

def main():
    # === USER PARAMETERS ===
    basis = 'cc-pVDZ'
    num_opt_virtual_orbs = 4
    # Set to empty list [] to only run canonical order (no sorting)
    sorting_methods = []  # e.g., ['mp2_natural', 'mp2_natural_reverse']
    # Choose OVOS variants to test
    method_defs = [
        ("Original Newton", OVOS, {}),
        ("Trust‑Region Newton", OVOSTrustRegion, {'trust_radius': 5.0}),
        ("Gauss‑Seidel", OVOSGaussSeidel, {
            'inner_iter': 3, 'omega': 1.2, 'use_trust_scaling': True, 'residual_tol': 1e-4
        }),
        # ("Branch&Bound", OVOSBranchBound, {
        #     'global_tol': 1e-2, 'max_nodes': 100, 'max_bb_vars': 4, 'verbose': 1
        # }),
    ]
    # =============================

    print(f"Setting up water molecule ({basis})...")
    mol, mf = setup_molecule(basis)

    mp2_full = mp.MP2(mf)
    mp2_full.verbose = 0
    mp2_full.kernel()
    e_corr_full = mp2_full.e_corr
    e_mp2_full = mf.e_tot + e_corr_full
    print(f"RHF energy:           {mf.e_tot:.10f} Ha")
    print(f"Full MP2 total:       {e_mp2_full:.10f} Ha")
    print(f"Full MP2 correlation: {e_corr_full:.10f} Ha\n")
    print_orbital_info(mol, num_opt_virtual_orbs, e_corr_full)

    # Prepare MO coefficients
    mo_canonical = [mf.mo_coeff, mf.mo_coeff]
    mo_sorted = None
    used_method = None

    # If sorting methods provided, try each until one changes active set
    if sorting_methods:
        canon_active = get_canonical_active_indices(mol, num_opt_virtual_orbs)
        print(f"\nCanonical active virtual indices: {canon_active}\n")
        for method in sorting_methods:
            mo_sorted, perm = get_sorted_mo_coeffs_and_perm(mol, mf, method=method)
            sorted_active = get_active_indices(mol, perm, num_opt_virtual_orbs)
            print(f"Sorting method '{method}' active virtual indices: {sorted_active}")
            if sorted_active != canon_active:
                print(f"  -> Active set differs! Using '{method}' for all OVOS runs.\n")
                used_method = method
                break
            else:
                print("  -> Active set identical to canonical. Trying next method...")
        if used_method is None:
            print("\n" + "="*80)
            print("WARNING: All sorting methods yielded the same active virtual set as canonical.")
            print("         No improvement in MP2 energy is expected. Exiting to save time.")
            print("="*80)
            sys.exit(0)
        print(f"\n--- Using sorting method: {used_method} ---")
    else:
        print("\n--- No sorting methods provided; running canonical order only. ---")

    # Run OVOS on canonical (and sorted if applicable)
    results = {}
    for name, cls, kwargs in method_defs:
        print(f"\n--- {name} ---")
        print("  Canonical order...")
        res_canon = run_method(cls, mol, mf, mo_canonical, num_opt_virtual_orbs, **kwargs)
        if mo_sorted is not None:
            print("  Sorted order...")
            res_sorted = run_method(cls, mol, mf, mo_sorted, num_opt_virtual_orbs, **kwargs)
        else:
            res_sorted = None
        results[name] = (res_canon, res_sorted)

    # Print comparison table
    print("\n" + "="*90)
    print(f"Comparison of OVOS methods for {basis} basis, {num_opt_virtual_orbs} active virtuals")
    if used_method:
        print(f"Sorting method: {used_method}")
    else:
        print("Sorting method: None (canonical only)")
    print("="*90)
    print(f"{'Method':<25} {'Order':<10} {'E_corr (Ha)':<15} {'Iter':<8} {'Time (s)':<8} {'Stop reason'}")
    print("-"*90)
    for name, (canon, sorted_) in results.items():
        print(f"{name:<25} {'Canonical':<10} {canon['energy']:>12.8f}   {canon['iterations']:>5}     {canon['time']:>6.2f}     {canon['stop_reason']}")
        if sorted_ is not None:
            improvement = sorted_['energy'] - canon['energy']
            print(f"{name:<25} {'Sorted':<10} {sorted_['energy']:>12.8f}   {sorted_['iterations']:>5}     {sorted_['time']:>6.2f}     {sorted_['stop_reason']}  (improvement: {improvement: .6f})")
            print(f"{'':<25} {'Diff to full':<10} {e_corr_full - sorted_['energy']:>12.8f}   {'':<8} {'':<8} {'':<15}")
        print("-"*90)
    print("="*90)

if __name__ == "__main__":
    main()