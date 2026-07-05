#!/usr/bin/env python3
"""
Run OVOS with virtual orbitals sorted by their MP2 correlation contribution.

The procedure:
1. Compute a full MP2 calculation (all virtuals) to obtain t2 amplitudes.
2. For each virtual orbital, compute its weight = sum_{i>j, b} (t_ab^ij)^2.
3. Sort virtual orbitals by weight (descending).
4. Reorder the MO coefficients so that the top `num_opt_virtual_orbs` are placed
   immediately after the occupied orbitals (active virtuals), and the rest become inactive.
5. Run OVOS with this new ordering.
6. Compare the energy with the original (canonical) ordering.

The script prints both results.
"""

import numpy as np
from pyscf import gto, scf, mp
from ovos import OVOS  # your original OVOS class
from ovos_trust_region import OVOSTrustRegion  # optional, but we'll use the best one

def get_virtual_weights(mol, mf):
    """
    Compute MP2 t2 amplitudes and calculate a weight for each virtual orbital.
    Weight = sum_{i>j, b} (t_ab^ij)^2  (sum over occupied pairs and other virtuals)
    Returns a weight array of length (nvir), where nvir = number of virtual spatial orbitals.
    """
    # Run MP2 with all orbitals
    mp2_obj = mp.MP2(mf)
    mp2_obj.verbose = 0
    mp2_obj.kernel()
    t2 = mp2_obj.t2  # shape (nocc, nocc, nvir, nvir) (chemist ordering: i,j,a,b)
    nocc, nvir = t2.shape[0], t2.shape[2]
    # Sum over i>j and over b
    weights = np.zeros(nvir)
    for a in range(nvir):
        # Sum over all occupied i>j and all b
        # We'll use einsum for efficiency: t2[i,j,a,b] squared summed over i,j,b
        # But only i>j? MP2 t2 already antisymmetric, so sum all i,j will double count but still gives a weight.
        # We'll sum over all i,j and b, but divide by 2? Actually, we want a measure; the exact factor doesn't matter.
        # We'll do: weights[a] = sum_{i,j,b} (t2[i,j,a,b])^2
        # This includes both i>j and j>i, but it's fine for ranking.
        weights[a] = np.sum(t2[:, :, a, :]**2)
    return weights

def reorder_mo_coeffs(mo_coeff, occ_idx, vir_idx_sorted):
    """
    Reorder the MO coefficient matrix so that the virtual orbitals are in the order given.
    Occupied orbitals remain unchanged.
    mo_coeff: (nao, nmo) array
    occ_idx: slice or indices for occupied (e.g., 0:nocc)
    vir_idx_sorted: list of indices in the original virtual ordering, sorted by weight descending.
    Returns new_mo_coeff.
    """
    nocc = occ_idx.stop if isinstance(occ_idx, slice) else len(occ_idx)
    # Get occupied columns
    occ_cols = mo_coeff[:, :nocc]
    # Get virtual columns in the new order
    vir_cols = mo_coeff[:, nocc + np.array(vir_idx_sorted)]
    return np.hstack([occ_cols, vir_cols])

def run_ovos_sorted(mol, mf, num_opt_virtual_orbs, method_class=OVOSTrustRegion, verbose=0):
    """
    Run OVOS with sorted virtual orbitals.

    Parameters:
    - mol, mf: PySCF molecule and RHF object.
    - num_opt_virtual_orbs: number of active virtual spin-orbitals (must be even).
    - method_class: OVOS variant to use (e.g., OVOSTrustRegion).
    - verbose: verbosity level.

    Returns:
    - E_corr_sorted: correlation energy from sorted order.
    - E_corr_original: correlation energy from canonical order (for comparison).
    - (optional) more details.
    """
    # --- Original (canonical) order ---
    Fao = [mf.get_fock(), mf.get_fock()]
    mo_coeffs_orig = [mf.mo_coeff, mf.mo_coeff]  # alpha and beta (same for RHF)
    ovos_orig = method_class(
        mol=mol, scf=mf, Fao=Fao,
        num_opt_virtual_orbs=num_opt_virtual_orbs,
        mo_coeff=mo_coeffs_orig,
        verbose=verbose
    )
    result_orig = ovos_orig.run(mo_coeffs_orig, fock_spin=None)
    E_corr_orig = result_orig[0]

    # --- Compute virtual weights (spatial) ---
    weights = get_virtual_weights(mol, mf)
    # Sort virtual indices by weight descending
    vir_idx_sorted = np.argsort(weights)[::-1]  # descending

    # --- Reorder MO coefficients ---
    nocc = mol.nelec[0]  # for RHF, alpha and beta have same number of occupied
    mo_coeff_new = []
    for mo_coeff in mo_coeffs_orig:
        mo_new = reorder_mo_coeffs(mo_coeff, slice(0, nocc), vir_idx_sorted)
        mo_coeff_new.append(mo_new)

    # --- Run OVOS with sorted order ---
    ovos_sorted = method_class(
        mol=mol, scf=mf, Fao=Fao,
        num_opt_virtual_orbs=num_opt_virtual_orbs,
        mo_coeff=mo_coeff_new,
        verbose=verbose
    )
    result_sorted = ovos_sorted.run(mo_coeff_new, fock_spin=None)
    E_corr_sorted = result_sorted[0]

    return E_corr_orig, E_corr_sorted

def main():
    # Example: water, cc-pVDZ, 4 active virtuals
    mol = gto.Mole()
    mol.atom = 'O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692'
    mol.basis = 'cc-pVDZ'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    num_opt_virtual_orbs = 4  # even number of spin-orbitals
    print(f"Water, {mol.basis}, active virtual spin-orbitals = {num_opt_virtual_orbs}")
    print("Running OVOS with original (canonical) ordering...")
    E_orig, E_sorted = run_ovos_sorted(mol, mf, num_opt_virtual_orbs,
                                       method_class=OVOSTrustRegion, verbose=0)

    print(f"\nCanonical order correlation energy: {E_orig:.10f} Ha")
    print(f"Sorted order correlation energy:     {E_sorted:.10f} Ha")
    print(f"Improvement:                         {E_sorted - E_orig: .10f} Ha")

if __name__ == "__main__":
    main()