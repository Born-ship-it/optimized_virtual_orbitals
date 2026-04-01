# Lee & Head-Gordon (2018) - Regularized OO-MP2

**Paper Key:** `Lee2018`  
**Title:** Regularized Orbital-Optimized Second-Order Møller−Plesset Perturbation Theory: A Reliable Fifth-Order-Scaling Electron Correlation Model with Orbital Energy Dependent Regularizers  
**Journal:** *J. Chem. Theory Comput.* **14**, 5203–5219 (2018)  
**DOI:** 10.1021/acs.jctc.8b00731  
**Authors:** Joonho Lee, Martin Head-Gordon  
**Affiliation:** UC Berkeley / Lawrence Berkeley National Laboratory

---

## Summary

This paper addresses fundamental limitations of **orbital-optimized MP2 (OOMP2)** through orbital energy-dependent regularization. OOMP2 optimizes orbitals to improve MP2 correlation energy recovery, analogous to OVOS, but encounters convergence failures during bond-breaking. The authors develop κ-OOMP2 and σ-OOMP2 methods with regularized denominators to restore proper behavior.

---

## Chapters / Main Sections

- [x] **Introduction & Motivation** (pp. 5203-5204)
  - OOMP2: Economical approximation to Brueckner orbitals
  - Problems: Small energy denominators → divergence (especially on bond-breaking)
  - Missing Coulson-Fischer points (restricted→unrestricted transition)
  - Goal: Regularization scheme preserving OOMP2 accuracy while fixing pathologies

- [x] **OOMP2 Background** (pp. 5204-5205)
  - Hylleraas functional J_H (MP2 correlation energy upper bound)
  - Orbital optimization rotates occupied-virtual block
  - Unitary transformation preserves CCSD/MP2 invariance within O/V blocks
  - Pseudocanonical orbitals; singles effects negligible

- [x] **Regularization Strategies** (pp. 5204-5208)
  - **Type 1: κ-Regularizers** - additive shift in energy denominators: Δ' = Δ + κ
  - **Type 2: σ-Regularizers** - multiplicative scaling: Δ' = σ·Δ
  - **Motivation:** DSRG-inspired; denominator-dependent damping not uniform constants
  - Orbital energy dependent forms avoid damping large denominators unnecessarily

- [x] **Orbital Gradient Derivation** (pp. 5205-5208)
  - Lagrangian with orbital rotation parameters Θ
  - Variation δ_3 yields orbital rotation conditions
  - Regularization modifications to gradient straightforward
  - Stability Hessian analysis shows restoration of R→U transition at regularizer strength

- [x] **Training & Testing** (p. 5208)
  - Training set: W4-11 benchmark (44 molecules)
  - Optimal κ = 1.45 E_h^-1 identified
  - Scaled variants κ-S-OOMP2, σ-S-OOMP2 trained on TAE140 subset
  - Test sets: RSE43 (reaction energies), TA13 (thermochemistry)

- [x] **Bond-Breaking Results** (pp. 5208-5209)
  - **H₂ dissociation:** Unregularized OOMP2 diverges; κ-OOMP2 smooth curve
  - **C₂H₆ → C₂H₄ + H₂:** Regularization enables proper dissociation
  - **Coulson-Fischer point restored:** κ-OOMP2 shows continuous R→U transition (matches HF behavior)
  - Regularized methods reach proper U-solution at large bond extension

- [x] **Biradicaloid Chemistry** (pp. 5209-5210)
  - Singlet biradicaloid test cases: singlet fission relevant systems
  - Yamaguchi spin-projection for biradical character
  - **κ-OOMP2 advantage:** Captures strong biradicaloid character where OOMP2 fails to converge
  - Biradical index S² behavior physically sound

- [x] **Thermochemistry Performance** (p. 5210)
  - RSE43 test: Reaction energies; κ-OOMP2 matches CCSD(T) quality
  - TA13 test: Thermochemistry; modifications negligible (regularization sets strong enough)
  - MAD (mean absolute deviation) comparable to calibrated MP2

---

## Key Findings

### Main Contributions

1. **Regularization Strategy for OOMP2:**
   - Addresses fundamental convergence failure during orbital optimization
   - Energy denominators in MP2 can vanish for certain orbital rotations
   - Orbital energy-dependent regularization (κ, σ) avoids ad-hoc parameter tuning
   - Enables OOMP2 on bond-breaking/biradicaloid systems (previously impossible)

2. **Restoration of Restricted→Unrestricted Stability:**
   - Unregularized OOMP2 discontinuously breaks symmetry (or fails entirely)
   - κ-OOMP2 with κ=1.45 E_h^-1 restores Coulson-Fischer points
   - Continuous R→U transition proper dissociation behavior
   - Matches HF theoretical expectations

3. **OOMP2 as Economical Brueckner Approximation:**
   - Brueckner orbitals expensive (require iterative singles); OOMP2 cheaper
   - Orbital optimization removes artificial spin-symmetry-breaking (UHF artifacts)
   - Regularization makes OOMP2 robust for weakly-correlated & bond-breaking regimes
   - Fifth-order scaling O(N⁵) vs. CCSD(T) O(N⁷)

4. **Biradicaloid Accessibility:**
   - First demonstration of OOMP2-type orbital optimization on singlet biradicaloids
   - Yields correct biradical character (S²) through Yamaguchi projection
   - Regularization essential; unregularized version diverges
   - Opens new application domain for efficient correlation methods

5. **Practical Performance:**
   - κ-OOMP2 matches CCSD(T) accuracy on thermochemistry & reaction energies
   - Scaled variants (κ-S-OOMP2) fine-tuned for specific property classes
   - Computationally simpler than CCSD(T); RI-MP2 infrastructure leveraged

### Relevance to OVOS Thesis

- **Orbital Optimization Philosophy:** Complements OVOS conceptually (both optimize virtual space)
- **Alternative Approach:** OO-MP2 full-space systematic optimization; OVOS truncates redundant orbitals
- **Regularization Technique:** Regularization strategy may apply to OVOS convergence issues
- **Biradicaloid Context:** OVOS + VQE for electronic structure problems should consider these regimes
- **CCSD(T) Alternative:** Different path to high-accuracy correlated calculations (for comparison/discussion)

### Important Equations/Concepts

| Concept | Equation/Reference | Significance |
|---------|-------------------|--------------|
| Hylleraas Functional | $J_H[Ψ] = ⟨Ψ\|\hat{H}-E_0\|\Ψ⟩ + 2⟨Ψ\|\hat{V}-E_0\|Ψ_0⟩$ (Eq. 1) | MP2 energy upper bound; objective for optimization |
| MP2 Lagrangian | $\mathcal{L}[Θ] = [J_H] + E_1^{HF} + \text{orbital constraints}$ (Eq. 6) | Hylleraas + HF + multipliers for unitary rotations |
| Amplitude Equation | $t^{ab}_{ij} = -\frac{⟨ij\|\|ab⟩}{Δ^{ab}_{ij}}$ where $Δ^{ab}_{ij} = ε_i + ε_j - ε_a - ε_b$ (Eq. 9-10) | Orbital optimization can make Δ → 0 (divergence!) |
| κ-Regularization | $Δ'^{ab}_{ij} = Δ^{ab}_{ij} + κ$ | Additive energy shift; κ=1.45 E_h^-1 optimal |
| σ-Regularization | $Δ'^{ab}_{ij} = σ · Δ^{ab}_{ij}$ | Multiplicative scaling (alternative form) |
| Orbital Rotation | $U_{OV} = \exp[\Delta - (\Delta)^†]$ where Δ = Θ matrix (Eq. 18-19) | Parameterizes occupied-virtual mixing |
| Unitary Invariance | CCSD/MP2 invariant under OO,VV rotations; only OV rotations needed | Reduces optimization dimension |

### Numerical Results Summary

| System | Method | Accuracy vs CCSD(T) | Test Set | Notes |
|--------|--------|-------------------|----------|-------|
| H₂ (dissociation) | κ-OOMP2 | Smooth curve | Bond-breaking | Unregularized diverges |
| C₂H₆ → C₂H₄ + H₂ | κ-OOMP2 | Proper dissociation | Reaction path | Coulson-Fischer restored |
| Thermochemistry | κ-OOMP2 | ±CCSD(T) | TA13 set | Scaled variant negligible improvement |
| Reaction energies | κ-OOMP2 | ±CCSD(T) | RSE43 set | Mean absolute deviation comparable |
| Singlet biradicaloids | κ-OOMP2 | Correct S² | Biradical systems | Unregularized fails convergence |

---

## Notes & Highlights

### Key Insights

1. **Why Small Denominators Matter:**
   - Standard MP2: Denominators always positive (ε_i,ε_j > ε_a,ε_b for occupied)
   - OOMP2 orbital rotation: Can make ε_a approach ε_i → denominator → 0 → divergence
   - Regularization fix: Ensure Δ' bounded away from 0 always
   - Trade-off: Some correlation energy damped, but method converges & continuous transition restored

2. **Coulson-Fischer Points Physical Meaning:**
   - In HF: Lowest eigenvalue of R→U stability Hessian crosses zero at symmetry-breaking point
   - CP theorem: If lowest eigenvalue <0, broken-symmetry solution lower
   - OOMP2 without regularization: Can completely miss this transition (discontinuous or missing)
   - κ-OOMP2: Ensures continuous tracking; proper dissociation physics recovered

3. **Why Orbital Energy-Dependent Regularization?**
   - Constant offset κ: Crude; dampen all denominators equally
   - κ/Δ form: Better; preserve large denominators, selectively dampen small ones
   - But DSRG motivation: Each excitation deserves denominator-dependent treatment
   - Practical: κ = 1.45 E_h^-1 found empirically to balance stabilization + accuracy

4. **Brueckner Orbitals Connection:**
   - Brueckner: Iteratively optimize singles to remove T₁ (expensive)
   - OOMP2: Approximate Brueckner by optimizing orbitals at MP2 level (cheaper)
   - Regularized OOMP2: Makes approximation more robust
   - Still cheaper than full Brueckner but captures key orbital relaxation

5. **Biradicaloid Significance for OVOS:**
   - OVOS develops on benchmark systems (thermochemistry, geometry)
   - This paper shows OOMP2 useful for biradicaloids (weak correlations + strong static effects)
   - Questions: **Can OVOS be applied to biradicaloid systems?** Likely yes; regularization idea transferable
   - VQE + OVOS targeting excited states might benefit from similar insights

### Connection to Other Papers

| Related Paper | Connection | Notes |
|---------------|-----------|-------|
| **Adamowicz & Bartlett 1987** | Orbital optimization philosophy | Both minimize correlation functional |
| **Neogrady 2005** | Overlap functional optimization | Alternative orbital optimization approach |
| **Lee paper focus** | OO-MP2 vs OVOS | Different scaling, different target (full space opt vs truncation) |

### Terminology & Cautions

- **OOMP2** = Orbital-Optimized MP2 (full virtual space optimization)
- **Regularization** = Controlled modification of energy denominators to prevent divergence
- **κ, σ** = Regularization parameters (additive, multiplicative)
- **Brueckner orbitals** = Optimal orbitals from removing T₁ amplitudes
- **Coulson-Fischer point** = Symmetry-breaking transition in restricted→unrestricted
- **Biradicaloid** = Weakly bonded system with intermediate spin character (NOT fully broken bond)

### Limitations & Open Questions

- **Empirical training:** κ = 1.45 E_h^-1 found on W4-11; generality uncertain for other systems
- **Large molecules:** Testing limited to modest sizes; scaling unknown
- **Property calculations:** Only energies tested; dipole moments, response untested
- **Multireference:** Still single-reference based; MR-generalization unclear
- **Hardware:** RI-MP2 infrastructure required; not all codes support orbital optimization

### Questions for Investigation

1. **Relationship to OVOS:** Could OVOS + regularization improve convergence in difficult geometries?
2. **Property extensions:** Can κ-OOMP2 reliably predict dipole moments / polarizabilities (like Pitonak work)?
3. **Excited states:** Does regularization extend to OO-MP2 excited-state methods?
4. **System size:** How does κ behave for very large molecules?
5. **VQE connection:** Could regularization idea stabilize VQE ansatzes?
- [ ] VQE Implications
- [ ] Conclusions

---

## Key Findings

### OO-MP2 Formulation
*Document the orbital-optimised MP2 approach*

### Brueckner Connection
*Explain why OO-MP2 orbitals approximate Brueckner orbitals (T1 → 0)*

### VQE Relevance
*How OO-MP2 orbitals relate to better ground state overlap*

### Numerical Results
*Benchmark systems and accuracy metrics*

---

## Relevance to Thesis

- **Theory:** Connection between OVOS and Brueckner orbitals
- **VQE:** Using optimised orbitals for quantum computing
- **Performance:** How orbital optimisation improves correlation descriptions
- **Critical Citation:** Foundation for Brueckner orbital claim

---

## Notes & Annotations

*Key differences from OVOS, potential synergies, implications for VQE orbital choice*
