# Adamowicz & Bartlett (1987) - Optimized Virtual Orbital Space

**Paper Key:** `Adamowicz1987`  
**Title:** Optimized Virtual Orbital Space for High-Level Correlated Calculations  
**Journal:** *J. Chem. Phys.* **86**, 6314–6324 (1987)  
**DOI:** 10.1063/1.452468

---

## Chapters / Main Sections

- [x] Introduction & Motivation (pp. 1-2)
  - Problem: Large virtual orbital spaces limit correlated calculations
  - Solution approach: Optimize active virtual orbital space
- [x] Theoretical Foundation (pp. 2-3)
  - Hartree-Fock reference and correlation perturbation
  - Comparison with other virtual orbital reduction methods
  - Advantages of energy-based optimization
- [x] Second-Order Hylleraas Functional (pp. 3-4)
  - Functional form: $E_2^{H}[\phi_I] = (\phi_I|H_0 - E_0|\phi_I) + 2(\phi_d|V - E_0|\phi_0) = J_2$ (Eq. 2)
  - Intermediate normalization: $(\phi_1|\phi_0) = 0$ (Eq. 3)
- [x] Optimization Methods - Newton-Raphson (pp. 4-5)
  - Orbital rotation parameterization ($\phi_a' = \phi_a + \sum_e R_{ea}\phi_e$)
  - Gradient expressions (Eq. 12a): $G_{ea} = 2\sum_{i>j} t_{ijb}^{(1)}(ij|eb) + 2ID_{ab}^{(2)1,b}$
  - Hessian expressions (Eq. 12b): Block-diagonal structure exploited
- [x] Computational Implementation (pp. 4-5)
  - Two-step procedure: Configuration coefficients + orbital rotations computed independently
  - Block form of Hessian matrix (Eq. 8) reduces linear equation complexity
  - Cost: ~$n_c N_{virt}$ transformation + Newton-Raphson iterations
- [x] Numerical Results - CH₂ (pp. 5-6)
  - Full basis: 70 virtual orbitals, correlation energy E₂ = -0.182683 a.u.
  - OVOS(46): 65.7% of orbitals, retains 99.0% of E₂ at MBPT(2), 98.8% at CCSDT-1
  - Combined with exact E₂: achieves ~100% correlation energy even with 22-8 orbitals
- [x] Applications (pp. 6-8)
  - CH₂ (¹A₁): Singlet excited state with multireference character
  - B₂H₆ → 2BH₃: Dissociation curve with weak D₀ = 35.5 kcal/mol
  - H₂O₂ → 2OH: Peroxide dissociation
- [x] Conclusions & Outlook (p. 11)

---

## Key Findings

### Main Contributions

1. **OVOS Method Development:**
   - Proposes Optimized Virtual Orbital Space (OVOS): systematic reduction of virtual orbital dimension using second-order Hylleraas functional minimization
   - Key innovation: direct energy optimization (not ad-hoc truncation)
   - Diagonalizes Fock operator in virtual space → seamless integration with existing MBPT/CC codes

2. **Computational Efficiency Gains:**
   - Asymptotic cost scaling reduced from $\sim n_c N_{virt}^4$ (standard CC/MBPT) to $\sim n_c(N_{virt}')^4$ where $N_{virt}' \ll N_{virt}$
   - Example factor-4 speedup: B₂H₆ at CCSD level with OVOS(30): 4.4–9.3× faster than full space

3. **Correlation Energy Recovery:**
   - **CH₂:** Reducing 70→46 virtual orbitals retains **99–99.2%** of E₂, MBPT(4), CCSDT-1 energies
   - **50% reduction** (70→35): Still recovers **96–97%** at all correlation levels
   - **Combined with exact E₂:** Even 22-8 orbitals (11-31% of full space) achieve **~100%** correlation at MBPT(3/4) and CCSDT-1

4. **Robustness Along Potential Energy Curves:**
   - Dissociation energy D₀ (B₂H₆) converges smoothly with OVOS dimension across all geometries
   - Accurate treatment of weak bonds (35.5 kcal/mol) with 50-60% orbital reduction
   - Validates OVOS for structure predictions and reaction dynamics

5. **Comparison with Other Methods:**
   - Outperforms frozen natural orbitals (FNO) and pseudonatural orbitals in accuracy
   - Unlike V_{N-1} methods, directly energy-optimized rather than derived from density matrix
   - Advantage over IC-SCF or IVO: no ad-hoc parameters, systematic optimization principle

### Relevance to OVOS Thesis

- **Foundational:** This is THE defining paper of OVOS — all subsequent work references Adamowicz & Bartlett 1987
- **Theoretical basis:** Understand the Hylleraas functional minimization and Newton-Raphson orbital optimization for your thesis background
- **OVOS + VQE motivation:** Shows dramatic correlation energy recovery with minimal orbitals → directly applicable to reducing qubit requirements in quantum simulators
- **Citation essential:** Every reference to "OVOS method" or "original formulation" must cite this work

### Important Equations/Concepts

| Concept | Equation/Reference | Significance |
|---------|-------------------|--------------|
| Hylleraas Functional | $E_2^H[\phi_I] = (\phi_I\|H_0 - E_0\|\phi_I) + 2(\phi_d\|V - E_0\|\phi_0)$ | Upper bound to second-order energy; optimization objective |
| Intermediate Normalization | $(\phi_1\|\phi_0) = 0$ | Ensures orthogonality of perturbed wave function |
| Orbital Rotation | $\phi_a' = \phi_a + \sum_e R_{ea}\phi_e$ (exponential form) | Unitary transformation of active vs. non-active space |
| Gradient | $G_{ea} = 2\sum_{i>j} t_{ijb}^{(1)}(ij\|eb) + 2ID_{ab}^{(2)1,b}$ | Derivative of functional w.r.t. rotation parameters |
| Hessian | $H_{ea,jb} = \partial G_{ea}/\partial R_{jb}$ | Second derivative; block-diagonal structure exploited |
| Pair Separability | See Eq. (10) | Functional decomposes over electron pairs → computational simplification |

### Numerical Results Summary

| System | N_orbitals | MBPT(2) Retent. | MBPT(4) Retent. | CCSDT-1 Retent. | +Exact E₂ |
|--------|-----------|-----------------|-----------------|-----------------|-----------|
| CH₂ | 70→46 (66%) | 99.0% | 99.2% | 98.8% | ~100% |
| CH₂ | 70→35 (50%) | 96.4% | 97.2% | 96.7% | ~100% |
| CH₂ | 70→22 (31%) | 88.7% | 90.6% | 90.2% | 98.9% |
| B₂H₆ dissl. | 54→30 (56%) | — | — | 38.8 vs 39.2 kcal/mol | —  |
| B₂H₆ comp. + T | 54→30 (56%) | — | — | CCSDT(T) accurate | — |

- All energies in a.u. or kcal/mol as indicated
- "Retent." = percentage of full-space correlation energy recovered
- "+Exact E₂" = OVOS higher orders combined with exact second-order energy

---

## Notes & Annotations

### Key Insights & Highlights

1. **Why Second-Order Hylleraas Functional?**
   - Provides rigorous upper bound to E₂
   - Pair-separable form: $\sum_{i>j} J_{ij}^{(2)}$ → computationally tractable
   - Alternative orderings (V_{N-1}, pseudonatural) are ad-hoc; Hylleraas is energy-based

2. **Orbital Diagonalization Advantage**
   - OVOS orbitals diagonalize the Fock operator (canonical)
   - **Computational benefit:** No code modifications needed; works drop-in with standard MBPT(MP2/MP3/MP4) and CCSD(T) implementations
   - This is why GAUSSIAN 82 users could easily adopt OVOS

3. **Why Does 50% Reduction Still Give ~100% Correlation?**
   - Most correlation energy is *concentrated* in a small set of virtual orbitals
   - "Active" orbitals capture the essential virtual space physics
   - Suggests correlation manifests in surprisingly compact orbital subsets
   - **VQE connection:** Minimal qubit overhead if we select optimal virtual orbitals!

4. **Exact E₂ as "Rescue Device"**
   - OVOS captures higher orders (MBPT(3), MBPT(4), CCSD(T)) accurately on reduced space
   - But lower OVOS to extreme (12-22 orbitals) → E₂ lags behind
   - **Solution:** Use exact E₂ from full space + higher-order OVOS corrections → recovers ~100% energy
   - **Strategy:** Computationally cheap full-space E₂ calculation is ~$O(N_{virt}^3)$; higher orders expensive, so run on reduced space

5. **Geometry Dependence & Robustness**
   - B₂H₆ dissociation shows OVOS active space adapts smoothly along PES
   - No discontinuities or sudden orbital reordering
   - Suggests OVOS is suitable for dynamics, IRC calculations

6. **Open Questions from Paper (future directions)**
   - How does OVOS behave with **multireference systems** (CH₂ shows multireference character but still RHF reference)?
   - Extension to **UHF/ROHF** references for radicals
   - How does OVOS compare with **natural orbital iterations** in CI?
   - Can OVOS handle **excited states** beyond ground state?

### Connection to Other Papers in Reading List

| Related Paper | Connection | Notes |
|---------------|-----------|-------|
| **Lee 2018 (OO-MP2)** | Orbital-optimized MP2 is conceptually similar but optimizes over *full* virtual space | OVOS is restricted optimization; OO-MP2 is unrestricted. Lee argues OO-MP2 → Brueckner orbitals; OVOS likely related. |
| **Pitoňák 2009 (CCSD(T))** | Extends OVOS to high-level CC with property calculations | Direct application of Adamowicz method to molecular properties |
| **Rolik 2011 (Cost Reduction)** | Compares active-space truncation vs. OVOS-like approaches | Puts OVOS in broader context of virtual space reduction strategies |
| **Neogrády 2011 (CC+OVOS review)** | Comprehensive review crediting Adamowicz 1987 | How OVOS integrated into practical CC codes |

### Notation & Convention Notes

- $i, j, k, ...$ = occupied (hole) orbitals
- $a, b, c, ...$ = virtual (particle) orbitals in active space
- $e, f, ...$ = virtual orbitals in non-active space
- $E_0$ = HF energy; $E_1$ = first-order correction; $E_2$ = second-order correlation energy
- $N_{virt}$ = full virtual orbital space dimension; $N_{virt}'$ = active (OVOS) dimension

### Cautions & Limitations

- **Restricted to closed-shell RHF reference** in 1987 paper (extensions mentioned but not implemented)
- **Initial basis set effects:** Larger basis → more rigorous OVOS (tested with modest bases 62-74 contracted Gaussian functions)
- **Multireference limitations:** While CH₂ test shows some success, formal RHF assumption may break down for strongly multireference systems
- **No explicit orbital localization:** OVOS orbitals are canonical Fock eigenvectors, not chemically localized

### Questions for Investigation

1. How do OVOS orbitals compare to **Brueckner orbitals**? (Lee 2018 might clarify)
2. Can OVOS be combined with **frozen natural orbitals (FNO)** for further reduction?
3. What is the **relationship between OVOS and active-space selection** in CASSCF/DMRG?
4. How does OVOS perform on **transition states and reaction barriers**?
5. **VQE-specific:** Do OVOS-prepared initial states reduce VQE iterations compared to HF references?

### Terminology to Adopt in Thesis

- Use **"Optimized Virtual Orbital Space (OVOS)"** consistently (not "OVOS method" alone initially—clarify the meaning)
- When citing Adamowicz & Bartlett, note the 1987 **J. Chem. Phys.** publication to distinguish from extended work
- Distinguish **active vs. non-active orbitals** in virtual space (standard in this field)
