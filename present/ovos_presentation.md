---
marp: true
size: 16:9
math: katex
paginate: true
theme: academic
---

<!-- Master's thesis on **Optimized Virtual Orbitals (OVO)** for quantum computing at UCPH. Implementation based on [L. Adamowicz & R. J. Bartlett (1987)](https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level) - minimizes second-order correlation energy (MP2) using orbital rotations 

Outline
1. Motivation
2. Theoretical background
3. OVOS algorithm (reference)
4. Key equations
5. Implementation details
6. Numerical results
7. Limitations & future work
8. Summary

-->

# Optimized Virtual Orbital Space (OVOS)

**Master's Thesis** - UCPH  

---

<!-- Based on article by Adamowicz & Bartlett (1987) -->

**Optimized virtual orbital space for high-level correlated calculations**  
Adamowicz, L. & Bartlett, R. J.  
*J. Chem. Phys.* **86**, 6314-6324 (1987)  
DOI: [10.1063/1.452468](https://doi.org/10.1063/1.452468)


---

<!-- Motivation -->

### Core Concept

Reduce the virtual orbital space dimension from N_VIRT to N'_VIRT < N_VIRT while preserving most correlation energy.

The second-order Hylleraas functional is used to find an optimal rotation of the active virtual space against the nonactive space, to minimize the second-order correlation energy. 