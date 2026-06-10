import numpy as np
from pyscf import gto, scf, mp, ao2mo

# ---------------------------------------------------------------------
# H2O molecule with cc-pVDZ
# ---------------------------------------------------------------------
mol = gto.M(
    atom="H .0 .0 .0; F .0 .0 0.917",
    basis='cc-pvdz',
    unit='Angstrom',
    spin=0,
    charge=0,
    symmetry=False,
    verbose=0
)

# RHF calculation
mf = scf.RHF(mol)
mf.kernel()

# ---------------------------------------------------------------------
# MP2 correlation energy from PySCF (UMP2 works with RHF reference)
# ---------------------------------------------------------------------
ump2 = mp.UMP2(mf)
e_corr = ump2.run().e_corr
print(f"\nTotal MP2 correlation energy (from PySCF UMP2): {e_corr:.10f} Ha")

# ---------------------------------------------------------------------
# Compute per‑orbital MP2 contributions using full 4‑index MO integrals
# ---------------------------------------------------------------------
mo_energy = mf.mo_energy
nocc = mol.nelectron // 2
norb = mf.mo_coeff.shape[1]

# Get full MO integrals (chemist notation)
eri_2d = ao2mo.kernel(mol, mf.mo_coeff)
eri_4d = ao2mo.restore(1, eri_2d, norb)

mp2_contrib = np.zeros(norb)

# Sum over all occupied i, j and virtual a, b (including i=j and a=b, but those vanish)
for i in range(nocc):
    for j in range(nocc):
        for a in range(nocc, norb):
            for b in range(nocc, norb):
                iajb = eri_4d[i, a, j, b]
                ibja = eri_4d[i, b, j, a]
                denom = mo_energy[i] + mo_energy[j] - mo_energy[a] - mo_energy[b]
                pair = (2.0*iajb*iajb - iajb*ibja) / denom
                contrib = pair / 4.0          # per‑orbital share
                mp2_contrib[i] += contrib
                mp2_contrib[j] += contrib
                mp2_contrib[a] += contrib
                mp2_contrib[b] += contrib

print(f"Sum of per‑orbital MP2 contributions: {mp2_contrib.sum():.10f} Ha (should match total MP2 correlation energy)")
print()

# ---------------------------------------------------------------------
# Truncate to 10 decimal places
# ---------------------------------------------------------------------
mp2_trunc = np.floor(mp2_contrib * 1e10) / 1e10   # truncate, not round

print("=== RHF spatial orbital energies (original order) ===")
print(" idx |     Energy (Ha)     |      dE (Ha)       |  MP2 contrib (10 digits) | MP2 change")
print("-" * 90)

for i, e in enumerate(mo_energy):
    if i == 0:
        dE = 0.0
        change_sign = " "
    else:
        dE = mo_energy[i] - mo_energy[i-1]
        # Compare truncated values
        if mp2_trunc[i] < mp2_trunc[i-1]:
            change_sign = "+"
        elif mp2_trunc[i] > mp2_trunc[i-1]:
            change_sign = "-"
        else:
            change_sign = "="
    print(f" {i:2d}  |  {e:18.10f}  |  {dE:16.10f}  |  {mp2_trunc[i]:16.10f}       |     {change_sign}")
    if i == nocc - 1:
        print("-" * 90)