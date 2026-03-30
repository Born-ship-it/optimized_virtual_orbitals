# Pitoňák et al. (2006) - OVOS for Molecular Properties

**Paper Key:** `Pitonak2009`  
**Title:** Optimized Virtual Orbitals for Correlated Calculations: Towards Large-Scale CCSD(T) Calculations of Molecular Dipole Moments and Polarizabilities  
**Journal:** *J. Mol. Struct. THEOCHEM* **768**, 79–89 (2006)  
**DOI:** 10.1016/j.theochem.2006.05.018  
**Authors:** Michal Pitoňák, Filip Holka, Pavel Neogrady, Miroslav Urban

---

## Chapters / Main Sections

- [x] **Introduction:** OVOS for molecular properties; ~50% reduction → 16× speedup
- [x] **OVOS Theory:** Unitary transformations, virtual space partitioning (V1/V2/V3), overlap functional
- [x] **Optimization:** Iterative procedure, Adamowicz-Bartlett E2 correction
- [x] **Test Systems:** F⁻, CO, formaldehyde, thiophene, push-pull butadiene
- [x] **Basis Set Study:** Systematic cc-pVXZ, aug-cc-pVXZ, d-aug-cc-pVXZ analysis
- [x] **Property Results:** Dipole moments (±0.002 a.u.), polarizabilities (±0.02-0.05 a.u.)
- [x] **CBS Extrapolation:** Full vs. OVOS differences → 0.07 a.u. at limit
- [x] **Conclusions:** OVOS enables accurate properties with >100× speedup

---

## Key Findings

### Main Contributions

1. **OVOS Extended to Molecular Properties:** Not just energies but dipole moments & polarizabilities  
   - Properties more sensitive than energies; 50% truncation still adequate
   - Maintains chemical accuracy (2×10⁻³ a.u. for dipoles)

2. **Systematic Basis Set Performance:** OVOS improves with larger basis sets!
   - Counterintuitive: Larger basis → less virtual redundancy → more selective OVOS
   - Enables reliable CBS extrapolations (systematic limit prediction)

3. **Adamowicz-Bartlett Correction Essential:** Zero overhead, recovers >99% correlation
   - Use full MP2 + reduced CCSD: CC²^OVOS = E2(full) + [CC(OVOS) - E2(OVOS)]
   - Difference between "okay" (97%) and "publishable" (99%+) accuracy

4. **Practical Implementation:** Symmetry-balanced selection critical; iterative not always convergent

5. **System Scope:** Successfully applied to diffuse anions, aromatics, push-pull conjugated systems

### Numerical Results

| System | Basis | Method | % Orbitals | Dipole Error | Polarizability Error |
|--------|-------|--------|-----------|----------------|----------------------|
| CO | aug-cc-pVTZ | CCSD(T) | 50% | 0.001 a.u. | ±0.05 a.u. |
| F⁻ | d-aug-cc-pV5Z | CCSD(T) | ~50% | Minimal | 0.07 a.u. (2% final) |
| Thiophene | aug-cc-pVTZ | CCSD(T) | 50% | 0.003 a.u. | ±0.08 a.u. |

### Relevance to Thesis

- **Properties Not Just Energies:** Critical for understanding OVOS scope
- **CBS Strategy:** Demonstrates systematic basis-set behavior & limit prediction
- **VQE Bridge:** Polarizability analog for quantum response property calculations

