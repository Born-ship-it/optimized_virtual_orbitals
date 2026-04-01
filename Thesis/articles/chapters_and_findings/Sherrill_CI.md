# Sherrill & Schaefer - Configuration Interaction Method

**Paper Key:** CI_Sherrill_Schaefer_1999  
**Full Citation:** Sherrill, C. D. & Schaefer, H. F. (1999). The Configuration Interaction Method: Advances in Highly Correlated Approaches. *Advances in Quantum Chemistry*, **34**, 143–269. Academic Press.  
**Lead Authors:** C. David Sherrill (UC Berkeley), Henry F. Schaefer III (University of Georgia)  
**Affiliation:** Center for Computational Quantum Chemistry, University of Georgia

---

## Summary

Comprehensive **review of configuration interaction (CI) methods** for solving the electronic Schrödinger equation. Emphasizes **highly correlated CI methods** including full CI, CISD, multireference CI, restricted active space (RAS) CI, and second-order CI (SOCI). Covers determinant-based algorithms, computational strategies, and practical implementations. Essential reference for understanding active space selection and truncation strategies in quantum chemistry.

---

## Chapters / Main Sections

- [x] **Introduction & Historical Context** (pp. 146–149)
  - CI as simplest conceptually correct method for electron correlation
  - Historical development since Hartree-Fock era
  - Limitations of single-reference CC methods (CCSD, CCSD(T)) for multireference systems
  - Recognition of need for multireference methods

- [x] **CI Theory Fundamentals** (pp. 149–156)
  - Solving electronic Schrödinger equation: HIΨ⟩ = E|Ψ⟩
  - Linear expansion in N-electron basis functions (Slater determinants or CSFs)
  - Variational theorem: computed energy upper bound to exact ground state
  - Correlation energy definition: E_corr = E₀ - E_HF
  - Distinction between dynamical and nondynamical correlation

- [x] **Slater Determinants & Matrix Elements** (pp. 155–157)
  - Slater's rules for computing ⟨Φᵢ|Ĥ|Φⱼ⟩
  - Second quantization formalism using creation/annihilation operators
  - Coupling coefficients: one-electron and two-electron terms
  - One vs two spin-orbital differences in determinants

- [x] **Single-Reference CI** (pp. 157–165)
  - **CISD (CI singles and doubles):** Most common truncation
  - Accuracy: DZP basis → bond lengths 0.4% error, frequencies 4% error
  - **SOCI (second-order CI):** Multireference variant; active space approach
  - Size extensivity problem: CISD not extensive; CCSD addresses this
  - Comparison: CISD scales as N⁶ (same as CCSD)

- [x] **Multi-Reference CI** (pp. 149, 165–166)
  - Multireference CISD with chosen reference configurations
  - Reference selection challenge: depends on molecule & geometry
  - **SOCI success:** Nearly parallels full CI surfaces; provides uniformity
  - Size extensivity corrections needed for larger systems

- [x] **Active Space Selection & RAS** (pp. 166–167)
  - **Restricted Active Space (RAS) CI:** Divide orbitals into 3 subspaces
    - Primary core: doubly occupied (no excitations)
    - Primary active: variable occupation
    - Secondary core: unoccupied (limited excitations allowed)
  - RAS capable of evaluating SOCI and CISD[TQ] wavefunctions
  - A priori selection strategy for CI space

- [x] **Determinant-Based Algorithms** (pp. 167–171)
  - Alpha and beta string notation (Handy, 1980)
  - Vectorized algorithms for full CI (Knowles-Handy)
  - Olsen's improvement: reduced operation count, maintained vectorization
  - Graphical vs nongraphical methods for string addressing
  - String replacement lists for efficient computation

- [x] **Truncation Schemes** (pp. 165–167)
  - **CISD[TQ]:** SOCI with higher-than-quadruple excitations excluded
  - Performance near SOCI for single-reference dominated systems
  - Flexibility in a priori CI space selection
  - Trade-off: computational cost vs accuracy and transferability

- [x] **Computational Aspects** (pp. 166–171)
  - Integral transformation: atomic orbital → molecular orbital basis
  - Iterative diagonalization methods (Davidson, Olsen)
  - Direct CI: coefficients evaluated on-the-fly (storage constraint)
  - Full CI dimension: factorial growth with electrons and basis functions

- [x] **Applications** (pp. 172–173)
  - Full CI benchmarks for calibrating other methods (MBPT, CC)
  - Excited state description (multiple eigenvalues)
  - Bond-breaking reactions (multireference regime)
  - Transition metal complexes (inherently multireference)

- [x] **Comparison with Other Methods** (pp. 146–149)
  - **CI vs CC:** CI simpler conceptually; CC more size-extensive
  - **CISD vs CCSD:** CCSD recovers some triples/quadruples via products; more accurate
  - **SOCI vs full CI:** SOCI requires much less effort; nearly parallel surfaces
  - **CI advantages:** Exact application to all geometries; multi-reference capability

- [x] **Analytic Gradients & MCSCF Integration** (pp. 149)
  - Beyond scope but essential for geometry optimization
  - Connection to complete active space (CAS) SCF methods
  - Orbital optimization in multiconfigurational context

---

## Key Findings

### 1. CI Framework: Core Principles

**Matrix formulation of Schrödinger equation:**
$$HC = ESC \quad \text{(orthonormal basis: } HC = EC\text{)}$$

where $H_{IJ} = \langle \Phi_I | \hat{H} | \Phi_J \rangle$ and $C$ is the vector of CI coefficients.

**Key insight:** CI is "exact theory" in principle (includes all basis functions); in practice, truncated to finite CI spaces.

**Expansion basis choices:**
- **Slater determinants:** Simpler matrix elements (Slater's rules), but typically 2–4× longer vectors
- **CSFs:** Fewer functions needed (already eigenfunctions of Ŝ²), but more complex matrix element evaluations

### 2. Correlation Treatment: Dynamical vs Nondynamical

**Correlation energy definition:**
$$E_{\text{corr}} = E_0 - E_{HF}$$
(negative; measures energy recovered by allowing electrons to avoid instantaneous repulsions)

**Two components:**
1. **Dynamical correlation:** At equilibrium; ~95% recovered by CISD
2. **Nondynamical (static) correlation:** Inadequacy of single reference; grows at stretched geometries; crucial for transition metals, bond breaking

**Example (H₂O, cc-pVDZ basis):**
| Geometry | E_corr (Htr) |
|----------|--------------|
| R_e      | -0.218       |
| 1.5·R_e  | -0.270       |
| 3.0·R_e  | -0.568       |

→ Correlation energy **increases** when bonds stretched (nondynamical dominates)

### 3. Variational Theorem & Energy Convergence

**Statement:** CI ground state energy ≤ exact ground state energy

**Proof insight:** Error is **quadratic** in wavefunction error:
$$E - E_0 = (\langle\Psi - \Delta\Psi|\hat{H}|\Psi - \Delta\Psi\rangle) / (\langle\Psi - \Delta\Psi|\Psi - \Delta\Psi\rangle)$$

→ Energy converges faster than other properties (e.g., dipole moments)

**MacDonald-Hylleraas-Undheim relations:** As more determinants added, eigenvalues improve monotonically.

### 4. Single-Reference CISD Performance

**Accuracy reported:**
- **DZP basis:** Bond lengths ~0.4% error, frequencies ~4% error
- **TZBP basis (with CCSD):** Bond lengths ~0.2% error, frequencies ~2% error
- **Limitation:** CISD **not size extensive** → degrades with system size

**Comparison with CCSD:**
- Both scale as N⁶ computationally
- **CCSD superior:** Accounts for some triples/quadruples via products of singles/doubles
- **CISD advantage:** Simpler; exact for excited states (no "reference bias")

### 5. Multireference CI & Active Space Strategy

**SOCI (Second-Order CI):**
- Multireference CISD: all possible electron distributions within chosen active space
- Dramatically more expensive than single-reference CISD
- **Advantage:** Nearly parallel potential energy surfaces (unlike traditional MR-CI); suitable as "model chemistry"
- **Computational cost:** Too high for routine use on large molecules (as of 1999)

**CISD[TQ] approach:**
- SOCI with exclusion of >quadruple excitations
- Nearly equivalent to SOCI for single-reference-dominated systems
- More computationally feasible

**Active space concept:**
- Core orbitals (doubly occupied, frozen in CI)
- Active orbitals (variable occupation)
- Virtual orbitals (unoccupied or restricted excitations)
- Direct impact on CI dimension reduction

### 6. Restricted Active Space (RAS) CI

**Three orbital subspaces:**
1. **RAS I (primary core):** Doubly occupied; no excitations
2. **RAS II (primary active):** Full excitations allowed
3. **RAS III (secondary virtual):** Singly or doubly occupied from active orbitals

**Capability:** Efficiently computes SOCI and CISD[TQ] wavefunctions.

**A priori selection:** Active space defined beforehand (not adapted iteratively).

### 7. Truncation by Excitation Level

**Factorial CI space growth:**
- Full CI dimension grows factorially: $\binom{N_{\text{basis}}}{N_{\text{elec}}}$
- For N=10 electrons, 100 basis functions → ~10¹³ determinants (intractable)
- Even incomplete basis: exponential growth

**Truncation strategies:**

| Scheme | Reference | Excitations | Size | Comment |
|--------|-----------|-------------|------|---------|
| CISD   | HF        | S, D        | Medium | 95% correlation (equilibrium) |
| CISD[TQ] | Multiple active | S,D,T,Q only | Large | Bridges CISD-SOCI gap |
| SOCI   | All active | Unlimited | Huge | Multireference benchmark |
| Full CI | All determinants | Unlimited | Factorial | Exact; benchmarking only |

---

## Accuracy & Computational Scaling

### Size Extensivity Issue

**Problem:** CISD energy per electron **decreases** with system size → unphysical

**Formula example:** Two non-interacting molecules:
- CISD prediction: $E_{\text{total}} < E_1 + E_2$ (violates additivity!)

**Solutions:**
- CCSD: inherently size extensive
- Size extensivity corrections for CI
- Multireference methods: can be corrected

### Computational Complexity

- One-electron integrals: $O(N_{\text{basis}}^4)$ transformation
- Two-electron integral cost: significant bottleneck; tensor factorization possible
- Davidson/Olsen iterative methods: typically 20–100 iterations
- Full CI algorithm scalability: **operation count ~ (N⁶ for CISD-level systems)**

---

## Relevance to OVOS Thesis

### Direct Connections

1. **Active Space = Virtual Space Selection**
   - RAS CI's three-orbital division mirrors OVOS philosophy
   - Core orbitals (frozen) ↔ OVOS-excluded orbitals
   - Active orbitals (full treatment) ↔ OVOS-included orbitals
   - **OVOS insight:** Reduces active space by ~90% → dramatic CI dimension shrinking

2. **CI Truncation ↔ OVOS Truncation**
   - Both reduce exponential configuration space
   - CISD[TQ]: truncates by excitation level
   - OVOS: truncates by orbital importance
   - **Combined:** OVOS pre-selection → smaller active space for CI

3. **Multireference Surgery**
   - OVOS identifies essential references (active space)
   - SOCI then operates on reduced active space
   - **Expected benefit:** OVOS-SOCI hybrid more tractable than full SOCI

4. **Nondynamical Correlation Capture**
   - OVOS precisely targets _where_ nondynamical effects occur (bond-breaking, multireference regions)
   - CI naturally describes these once active space chosen
   - **Synergy:** OVOS identifies active orbitals → CI fully correlates them

### Comparison with Other Truncation Methods

| Method | Strategy | OVOS Application |
|--------|----------|------------------|
| **CISD** | Singles/doubles only | Loses triples essential at stretched geometries |
| **RAS CI** | Three orbital classes | OVOS defines classes; RAS evaluates CI |
| **SOCI** | Multireference CI on active space | OVOS shrinks active space; SOCI then feasible |
| **CCSD(T)** | CC with perturbative triples | Not multireference; fails at dissociation |

---

## Notes & Annotations

### Key Algorithmic Insights

1. **String Representation (Handy, 1980):**
   - Alpha and beta electron indices as separate "strings"
   - Enables vectorized computation of determinant contributions
   - Direct connection to qubit representations in later quantum computing work

2. **Slater's Rules vs Second Quantization:**
   - Slater's rules: practical; used for determinants
   - Second quantization: general; applicable to any basis
   - Both yield identical results; choice depends on coding convenience

3. **Determinant Coupling:**
   - **Same determinant:** Diagonal element only
   - **Differ by 1 spin-orbital:** Includes one- and two-electron terms
   - **Differ by 2 spin-orbitals:** Two-electron term only
   - **Differ by >2:** Zero coupling (Brillouin's theorem for reference)

### Historical Context (as of 1999)

- **Pre-1980:** CSF-based (GUGA approach); conceptually simpler but harder to code
- **1980–1990:** Determinant-based revolution (Handy, Knowles, Olsen)
  - Slater determinants simpler; vectorizability enabled fast implementations
  - Full CI benchmarks became practical for small molecules
- **1990s:** RAS and active space methods flourish
  - Recognition that **active space selection is key bottleneck**
  - OVOS thesis (future work) directly addresses this

### Connections to OVOS Future Work

**Open questions in 1999-level CI:**
1. How do we choose the active space optimally?
2. Can we reduce SOCI computational cost for molecules with >8 electrons?
3. How do multireference methods scale to realistic molecules?

**OVOS answers:**
1. **Optimal active space:** Orbitals with largest electron density variance in virtual space
2. **SOCI reduction:** Pre-screening with OVOS → feasible SOCI; active space from 100→20 orbitals
3. **Scalability:** Hybrid OVOS-SOCI applicable to larger molecules (tested in thesis)

### Terminology Clarification

- **CSF** = Configuration State Function (eigenfunction of Ŝ²)
- **Slater determinant** = Any product of spin orbitals arranged antisymmetrically
- **RAS** = Restricted Active Space (three-orbital strategy)
- **SOCI** = Second-Order CI (multireference CI on active space)
- **Brillouin's theorem:** Reference determinant couple only with singles
- **Size extensivity:** Energy scales linearly with system size (e.g., N molecules → EN for uncorrelated E)

---

## Critical Insight for OVOS

> **The central thesis of Sherrill & Schaefer:** CI methods are "exact" (within basis) but intractable for large systems due to exponential CI space growth. **Solution: reduce active space intelligently.**
>
> **OVOS contribution:** Provides the first systematic, energy-based criterion for intelligent active space reduction, enabling SOCI-level accuracy on molecules where it was previously impossible.
