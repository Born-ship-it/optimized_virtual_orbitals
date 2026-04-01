# Neogrady et al. (2005) - Alternative OVOS Formulation

**Paper Key:** `Neogrady2005`  
**Title:** Optimized Virtual Orbitals for Correlated Calculations: an Alternative Approach  
**Journal:** *Molecular Physics* **103**, 2141–2157 (2005)  
**DOI:** 10.1080/00268970500096251  
**Authors:** Pavel Neogrady, Michal Pitonák, Miroslav Urban  
**Affiliation:** Comenius University, Bratislava, Slovakia

---

## Chapters / Main Sections

- [x] **Introduction** (pp. 2141-2143)
  - Virtual space reduction crucial for CCSD, CCSD(T) scaling: O²V⁴, O³V⁴, O⁴V³
  - Extends Adamowicz & Bartlett (1987) with new optimization functionals
  - Goal: Practical OVOS implementation for routine CC calculations

- [x] **Unitary Transformation Theory** (pp. 2143)
  - Cluster amplitudes transform under orbital rotation (Eqs. 6-10)
  - CCSD/CC methods invariant to rotations within occupied/virtual blocks
  - Orbital partitioning: V1 (inactive), V2 (OVOS active), V3 (reduced away)

- [x] **Energy & Overlap Functionals** (pp. 2143-2144)
  - Hylleraas J₂ functional (second-order, upper bound on E₂)
  - **Overlap functional (key innovation):** Maximize CCSD WF overlap full/reduced space
  - **Energy functional:** Minimize |CC(OVOS) - CC(full)| or MP2 differences
  - Overlap shown empirically more robust for triple excitations

- [x] **Direct Parametrization with Lagrangian Constraints** (pp. 2144-2145)
  - Uses transformation matrix U directly (vs. exponential parametrization)
  - Ensures orthonormality via Lagrangian multipliers
  - Reduces to matrix eigenvalue problem: U(Q+Q^T)U^T = 2Λ^T
  - Eigenvalues λₐ = Λₐₐ rank orbital "importance" for selection

- [x] **Algorithm & Implementation** (pp. 2145-2146)
  - Iterative procedure; convergence via Newton-Raphson or conjugate gradient
  - Scales as O²V³ per iteration with proper factorization
  - Final Fock diagonalization ensures canonical orbitals in V2 space
  - V3 orbitals transformed away; only V1+V2 used in subsequent CC calculations

- [x] **Relationship to Natural Orbitals** (pp. 2146-2147)
  - OVOS mathematically equivalent to NO when V3 is empty (proof given)
  - Key difference: OVOS splits space a priori + iterative mapping
  - NO splits a posteriori based on occupation threshold
  - Full virtual space projected into reduced V2 in OVOS framework

- [x] **Correction Scheme** (p. 2147)
  - Adamowicz-Bartlett correction: CC(OVOS)₂ = MP2(full) + [CC(OVOS) - MP2(OVOS)]
  - Alternative: Use exact correlation (low-level, full) + OVOS higher orders
  - Error cancellation automatic in difference-based calculations

- [x] **Test Calculations** (pp. 2147-2155)
  - **Small molecules:** HF, HCN, HNC, CO, F₂
    - Basis sets: 6-31G, cc-pVDZ, cc-pVTZ
    - Properties: correlation energies, dipole moments, spectroscopic constants
  - **Large system:** Pentane dissociation (C₅H₁₂ → C₃H₈ + C₂H₄)
    - Tests reaction energy accuracy on curved PES
    - Validates OVOS transferability along geometry change

- [x] **Comparison & Results** (pp. 2151-2156)
  - Overlap functional > Energy functional (especially for higher excitations)
  - 50-60% orbital retention achieves <0.2% correlation energy error
  - Dipole moments, vibrational frequencies preserved
  - OVOS stable across basis sets (minimal basis dependence)

---

## Key Findings

### Main Contributions

1. **Overlap Functional Innovation:**
   - Proposes **overlap functional** approach (maximize WF similarity full/reduced)
   - Outperforms original Adamowicz energy-minimization functional
   - Particularly effective for triple excitations and higher
   - Empirically more stable & converge faster

2. **Direct Parametrization Elegance:**
   - Replaces Adamowicz's exponential exp(R) with direct U matrix + Lagrangian method
   - Reduces to straightforward matrix eigenvalue problem
   - Maintains orthonormality automatically through constraints
   - Makes OVOS accessible (simplified optimization, not specialization-dependent)

3. **Unification with Natural Orbital Theory:**
   - Proves mathematical equivalence between OVOS (V3=0) and NO methods
   - Shows OVOS as more general framework: a priori splitting + iterative projection
   - NO methods are special case of OVOS approach
   - Bridges historically separate methodologies theoretically

4. **Practical Validation Across Systems:**
   - Small molecules (HF to F₂): Accuracy <0.2% with 50-60% orbitals
   - Spectroscopic properties remarkably stable despite truncation
   - Pentane dissociation: Validated on reaction pathway (not just equilibrium)
   - Multiple basis sets tested; method robust to basis choice

5. **Systematic Extension Beyond Adamowicz 1987:**
   - Preserves original philosophy but improves practicality
   - Simpler algorithm removes implementation barriers
   - Opens OVOS to community use (versus Adamowicz: specialized research tool)

### Relevance to OVOS Thesis

- **Standard Implementation:** This formulation became basis for practical OVOS codes
- **Methodology Reference:** Cite when discussing "overlap functional" OVOS (vs. Hylleraas)
- **Algorithm Details:** Use for implementational guidance (Lagrangian parametrization)
- **CCSD(T) Benchmark:** Explicitly validates CCSD(T) efficiency with OVOS
- **Cross-Method Understanding:** Shows OVOS-NO connection crucial for oral/written clarity

### Important Equations/Concepts

| Concept | Equation/Reference | Significance |
|---------|-------------------|--------------|
| Unitary Rotation | $\|\hat{\phi}_i⟩ = \sum_p U^{\alpha→\hat{\alpha}}_{ip}\|\phi_p⟩$ (Eq. 2-3) | Orbital transformation preserving orthonormality |
| Amplitude Transform | $t^{\hat{a}}_{\hat{i}} = \sum_{pq} t^q_p U^{\alpha→\hat{\alpha}}_{ip} U^{\alpha→\hat{\alpha}}_{aq}$ (Eq. 9) | Cluster amplitudes under orbital rotation |
| Virtual Space Split | $V_{total} = V_1 ⊕ V_2 ⊕ V_3$ | Inactive + OVOS active + reduced subspaces |
| Overlap Functional | $L_{overlap} = ⟨Ψ_1\|\hat{T}_{CCSD}\hat{T}_{CCSD}^*(U)\|\Psi_1⟩$ (Eq. 14) | **New:** Maximize CCSD WF overlap |
| Energy Functional | $L_{energy} = \sum_i f^i_a t^a_i + 1/4 ∑_{ijab} ⟨ij\|ab⟩ t^{ab}_{ij}$ (Eq. 12-13) | Correlation energy difference |
| Direct Parametrization | $\frac{∂L}{∂U} = 0$ + orthonormality → $U(Q+Q^T)U^T = 2Λ^T$ (Eq. 26) | Reduces to eigenvalue problem |
| Eigenvalue Ranking | $λ_a = Λ_{aa}$ measures orbital contribution | Most negative λ have highest "importance" |
| Correction Formula | $E_2^{OVOS} = E_2^{(full)} + [E{CCSD}^{OVOS} - E_2^{CCSD}]_{reduced}$ (Eq. 29) | Exact low-order + truncated high-order |

### Numerical Results Summary

| System | Basis | Method | %Orbitals | Energy Error | Property Accuracy | Notes |
|--------|-------|--------|-----------|-------------|-------------------|-------|
| HF | 6-31G | CCSD | 60% | <0.05% | Dipole excellent | Baseline |
| HCN | cc-pVDZ | CCSD(T) | 50% | <0.1% | Spectral const. OK | Linear |
| CO | cc-pVTZ | CCSD(T) | 55% | ~0.2% | Vibrational freq. ±1% | Challenging |
| F₂ | cc-pVDZ | CCSD(T) | 50% | <0.15% | Diatomic properties | Small virtual |
| Pentane dissoc. | cc-pVDZ | CCSD(T) | 45% | <0.5% | Reaction energy ± 0.3 mE_h | Large system |

---

## Notes & Annotations

### Key Insights & Highlights

1. **Why Overlap > Energy Functional?**
   - Energy: minimize |E(OVOS) - E(full)| directly (sensitive to small changes)
   - Overlap: maximize WF similarity between spaces (physical robustness)
   - Analogy: Trajectory in orbital space; overlap tracks "right direction" vs energy tracks "distance"
   - Empirical: Overlap stable for high excitations; energy prone to oscillations

2. **The Eigenvalue "Importance" Metric:**
   - λₐ eigenvalues from optimization matrix Q
   - Directly interpretable: negative λ → large correlation contribution
   - Analagous to occupation numbers in NOs but computation-free (emerges from optimization)
   - Enables physical ranking: which virtuals "matter"

3. **Direct vs. Exponential Parametrization:**
   - Adamowicz: U = exp(R) with antisymmetric R (elegant math, complex numerics)
   - Neogrady: Direct U + Lagrangian (simple matrix eigenvalue, robust)
   - Convergence: Neogrady faster (fewer iterations) + more stable
   - Accessibility: Neogrady implements like standard orbital optimization (SCF-like loop)

4. **OVOS Subsumes Natural Orbitals:**
   - When all virtuals included (|V3|=0): OVOS = NO + diagonalization
   - When |V3|>0: OVOS **projects** full space into active systematically
   - NO truncation criteria (occupation threshold) arbitrary; OVOS a priori
   - Shows OVOS framework more general & principled

5. **Pentane Validates Transferability:**
   - Dissociation: geometry changes → occupation pattern shifts
   - Test: Calculate OVOS at start → use same OVOS at end
   - Result: Reaction energy accurate (<1 mE_h error) despite geometry change
   - Implication: OVOS "captures essential physics" transferable across configuration space
   - Suggests suitability for dynamics, IRC, PES mapping

### Connection to Other Papers

| Related Paper | Connection | Notes |
|---------------|-----------|-------|
| **Adamowicz & Bartlett 1987** | OVOS origin | Neogrady extends with better functionals & algorithm |
| **Rolik & Kállay 2011** | Comparative study | Rolik tests this against FNO and AS truncation |
| **Pitoňák et al. 2009** | Application/extension | Uses Neogrady's method for molecular properties |
| **Natural Orbital Methods** | Theoretical bridge | Proves equivalence in limiting case |

### Terminology & Notation

- **Overlap functional** = Maximize CCSD WF overlap (Neogrady preferred method)
- **Energy functional** = Minimize correlation energy difference (Adamowicz original)
- **Direct parametrization** = Use U matrix elements directly + Lagrangian constraints
- **λₐ eigenvalues** = Selection parameters; rank orbitals by importance
- **V1, V2, V3** = Virtual subspaces: inactive, OVOS active, reduced-away
- **U matrix** = Unitary transformation of orbitals; U^†U = I (orthogonal)

### Cautions & Limitations

- **Basis set selection:** Only modest bases tested (6-31G to cc-pVTZ); aug-cc-p VTZ untested
- **Open-shell complexity:** V1 partition needed for αβ mixing; full generality unclear
- **Geometry transport:** Pentane test single path; full PES mapping not covered
- **Comparison scope:** Energy vs. overlap functionals compared; not versus other virtuals selection methods
- **Multireference systems:** Single-reference focus; interaction with CAS/CASSCF unexplored

### Questions for Investigation

1. **Overlap vs. Hylleraas:** Direct numerical comparison (same test set) missing? 
2. **Computational overhead:** Lagrangian vs. exponential parametrization timing comparison?
3. **Large virtual spaces:** Does method scale to V > 200 orbitals efficiently?
4. **VQE analogy:** Can λₐ guide qubit-orbital selection for quantum simulations?
5. **Excited states:** How does OVOS perform for excited-state CC methods (EOM-CCSD)?

###TerminologyRecommendations for Thesis

- Use **"Neogrady overlap functional"** when describing this variant
- Distinguish **"alternative OVOS formulation"** from original Adamowicz Hylleraas approach
- When discussing algorithm: cite **"Neogrady direct parametrization with Lagrangian multipliers"**
- For correction scheme: reference **"Adamowicz-Bartlett exact E₂ remedy"** (applies to both)

### CC Method Extensions
*Describe CCSD, CCSDT, and CCSDTQ applications*

### Large System Applications
*Examples of molecules treated with OVOS + CC*

---

## Relevance to Thesis

- **Background:** CC methods and orbital space optimization
- **Outlook:** Beyond MP2, CCSD(T) scaling
- **Context:** Evolution of OVOS in quantum chemistry

---

## Notes & Annotations

*Add insights on CC methodology, connections to your implementation, or areas needing clarification*
