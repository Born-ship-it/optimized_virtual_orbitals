# Reading List – OVOS Master's Thesis
## `Thesis/articles/`

This folder contains the key reading material for the thesis on  
**Optimized Virtual Orbital Space (OVOS) for Quantum Computing** (UCPH, 2026).

Each entry lists: filename → BibTeX key → description and relevance.

### Subfolder Structure

📂 **`chapters_and_findings/`** — Detailed chapter breakdowns and key findings for each paper
- Contains individual `.md` files for each PDF
- Use these files to document chapters, key findings, and relevant sections
- Each file has sections for chapters, main contributions, relevance to thesis, and annotations
- Update these files as you read each paper to maintain comprehensive reference notes

---

## Primary OVOS Papers

### `OVOS.pdf`
**Key:** `Adamowicz1987`  
*Adamowicz, L. & Bartlett, R. J.*  
"Optimized virtual orbital space for high-level correlated calculations"  
*J. Chem. Phys.* **86**, 6314–6324 (1987). DOI: 10.1063/1.452468  
→ **The foundational paper.** Introduces OVOS, the Hylleraas functional, Newton-Raphson
orbital optimisation, gradient and Hessian expressions. Numerical results for CH₂, C₆H₆,
B₂H₆ and H₂O₂. Must-read and central citation throughout the thesis.

**📝 Detailed Notes:** [chapters_and_findings/Adamowicz1987_OVOS.md](chapters_and_findings/Adamowicz1987_OVOS.md)

---

### `Optimized virtual orbitals for correlated calculations.pdf`
**Key:** `Pitonak2009`  *(fill exact year/journal below)*  
*Pitoňák, M.; Holka, F.; Neogrády, P.; Urban, M.*  
"Optimized virtual orbitals for correlated calculations: Towards large-scale CCSD(T)
calculations of molecular dipole moments and polarizabilities"  
→ Extension of OVOS to CCSD(T) property calculations (dipole moments, polarisabilities).
Shows OVOS performs well with ~50% virtual space reduction. Relevant to background,
theory (OVOS beyond MP2), and outlook (CCSD(T) extension).

**📝 Detailed Notes:** [chapters_and_findings/Pitonak2009_CCSD_T.md](chapters_and_findings/Pitonak2009_CCSD_T.md)

---

### `CC_calc_OVOS.pdf`
**Key:** `Neogrady2011book`  *(fill page range/book details below)*  
*Neogrády, P.; Pitoňák, M.; Granatier, J.; Urban, M.*  
"Coupled Cluster Calculations: OVOS as an Alternative Avenue Towards Treating Still
Larger Molecules" — Chapter 16 in: *Challenges and Advances in Computational Chemistry
and Physics* (2011).  
→ Comprehensive review of using OVOS with high-level CC methods on large molecules.
Useful for Background §CC methods and Outlook §OVOS beyond MP2.

**📝 Detailed Notes:** [chapters_and_findings/Neogrady2011_CC_OVOS.md](chapters_and_findings/Neogrady2011_CC_OVOS.md)

---

### `Cost reduction of high-order coupled-cluster methods via active-space and orbital transformation techniques.pdf`
**Key:** `Rolik2011`  
*Rolik, Z. & Kállay, M.*  
"Cost reduction of high-order coupled-cluster methods via active-space and orbital
transformation techniques"  
*J. Chem. Phys.* **134**, 124111 (2011). DOI: 10.1063/1.3569829  
→ Discusses active-space truncation and orbital transformation for CCSDT and CCSDT(Q).
Provides broader context for virtual space reduction beyond MP2. Relevant to Background
§orbital space methods and Outlook §CCSD(T) scaling.

**📝 Detailed Notes:** [chapters_and_findings/Rolik2011_Cost_Reduction.md](chapters_and_findings/Rolik2011_Cost_Reduction.md)

---

## Orbital-Optimised MP2 and Brueckner Orbitals

### `1807.06185v2.pdf`
**Key:** `Lee2018`  
*Lee, J. & Head-Gordon, M.*  
"Regularized Orbital-Optimized Second-Order Møller-Plesset Perturbation Theory:
A Reliable Fifth-Order Scaling Electron Correlation Model with Orbital Energy
Dependent Regularizers"  
*J. Chem. Theory Comput.* **14**, 5203–5219 (2018). arXiv:1807.06185  
→ **Directly relevant to the Brueckner orbital connection.** Orbital-optimised MP2
(OO-MP2) variationally minimises the MP2 energy with respect to orbital rotations,
making it formally similar to OVOS (but over the full virtual space). OO-MP2 orbitals
are Brueckner-like: the T1 diagnostic tends to zero. This paper is the key citation for
the claim that OVOS MOs approximate Brueckner orbitals. Also relevant for VQE: OO-MP2
orbitals give better overlap with the correlated ground state than HF orbitals.

**📝 Detailed Notes:** [chapters_and_findings/Lee2018_OO-MP2.md](chapters_and_findings/Lee2018_OO-MP2.md)

---

## Quantum Computing – VQE and Hamiltonians

### `Hamiltonian Simulation and Estimation - Juan de Garcia.pdf`
**Key:** `deGracia2024`  *(fill year/venue if published)*  
*de Gracia, J.*  
"From Qubits to Real-World Applications: Hamiltonian Simulation and Estimation"  
Lecture notes / presentation (RISE Research Institutes of Sweden, ~2024).  
→ Accessible introduction to Hamiltonian simulation, QPE, and VQE. Useful as a general
reference when explaining the quantum computing motivation (Introduction §Quantum
Computing Context, Background §Quantum Computing). Not peer-reviewed – use as
supplementary/tutorial reference.

**📝 Detailed Notes:** [chapters_and_findings/deGracia2024_Hamiltonian.md](chapters_and_findings/deGracia2024_Hamiltonian.md)

---

## Previous UCPH Work

### `PREP_project_COVO.pdf`
**Key:** `Clausen2025`  
*Clausen, T. B.*  
"Theory of Correlated Optimized Virtual Orbitals (COVO) and a Preliminary
Implementation of the Method"  
PREP Project, MSc in Quantum Information Science, University of Copenhagen,
October 2025. Supervisors: Prof. S. P. A. Sauer & Asst. Prof. P. W. K. Jensen.  
→ **Your own previous project.** COVO is the correlated extension of OVOS.
Directly motivates this master's thesis. Cite when justifying the choice of topic

**📝 Detailed Notes:** [chapters_and_findings/Clausen2025_COVO.md](chapters_and_findings/Clausen2025_COVO.md)
and the UHF framework. Also provides the UCPH institutional context. Key citation
in Introduction §Motivation.

---

## Standard Textbooks

### `Modern Quantum Chemistry - Attila Szabo, Neil S. Ostlund.pdf`
**Key:** `SzaboOstlund1989`  *(already in bibliography.bib)*  
*Szabo, A. & Ostlund, N. S.* — Dover, 1989.  
→ Standard reference for HF theory, MP2, CI. Cite for any "textbook" quantum chemistry
concept in Background.

### `Molecular Electronic-Structure Theory - T. Helgaker, P. Jørgensen, J. Olsen.pdf`
**Key:** `Helgaker2000`  *(already in bibliography.bib)*  
*Helgaker, T.; Jørgensen, P.; Olsen, J.* — Wiley, 2000.  
→ The definitive graduate text. Cite for orbital spaces, Fock matrix, CC, Brueckner
orbitals (Chapter 4), and MP2 in spin-orbital form.

### `Introduction to Computational Chemistry - Frank Jensen.pdf`
**Key:** `Jensen2017`  *(already in bibliography.bib)*  
*Jensen, F.* — Wiley, 3rd ed., 2017.  
→ Broad computational chemistry reference. Good for introductory citations on basis
sets, HF, and DFT comparisons.

### `The Configuration Interaction Method - C. David Sherrill, Henry F. Schaefer.pdf`
**Key:** `Sherrill1999`  
*Sherrill, C. D. & Schaefer, H. F.*  
"The Configuration Interaction Method: Advances in Highly Correlated Approaches"  
*Adv. Quantum Chem.* **34**, 143–269 (1999). DOI: 10.1016/S0065-3276(08)60532-8  
→ Comprehensive review of CI methods including FCI. Relevant when discussing FCI as
the exact limit and the Brueckner orbital connection (overlap with FCI ground state).
Also useful for Background §CI methods.

---

## Suggested Additional Reading (not in folder yet)

| Topic | Suggested reference |
|---|---|
| Brueckner orbitals (original) | Brueckner, K. A. *Phys. Rev.* **96**, 508 (1954) |
| Brueckner–CCD | Handy, N. C. et al. *Chem. Phys. Lett.* **164**, 185 (1989) |
| VQE original | Peruzzo et al. *Nat. Commun.* **5**, 4213 (2014) ← already in bib |
| Quantum chemistry review | Cao et al. *Chem. Rev.* **119**, 10856 (2019) ← already in bib |
| Jordan-Wigner / qubit mapping | Seeley, Richard, Love *J. Chem. Phys.* **137**, 224109 (2012) |
| FNO comparison | Taube & Bartlett *Collect. Czech. Chem. Commun.* **70**, 837 (2005) ← already in bib |

---

## How to Use This Reading List

### 📚 Workflow

1. **Reference this README** for quick paper summaries, BibTeX keys, and relevance to your thesis
2. **Use `chapters_and_findings/` folder** to maintain detailed notes as you read each paper
3. **Fill in chapter breakdowns** with the main sections and subsections of each paper
4. **Document key findings** relevant to OVOS, VQE, orbital optimization, or quantum computing
5. **Update annotations** with insights, connections to other papers, or questions

### 📝 Template Structure

Each `.md` file in `chapters_and_findings/` includes sections for:
- **Chapters/Sections** — Outline of the paper structure
- **Key Findings** — Main contributions and important results
- **Relevance to Thesis** — How it connects to your work
- **Notes & Annotations** — Your observations and interpretations

### 🔗 Quick Navigation

- Use the **📝 Detailed Notes** links in each paper entry above to jump to the corresponding markdown file
- Keep both README.md and the chapters_and_findings files open side-by-side for efficient reference
- Update files incrementally as you work through the reading list
