---
marp: true
size: 16:9
math: katex
paginate: true
theme: academic
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Optimized Virtual Orbital Space (OVOS) for Quantum Computing
## PhD Defense Presentation
**Supervisors:** Supervised by Asst. Prof. Phillip W. K. Jensen and Prof. Stephan P. A. Sauer
**Author:** Tobias Born Clausen
**Date:** 11.06.2024

<br>

<!-- ### Based on the work of Adamowicz & Bartlett (1987):
**Optimized virtual orbital space for high-level correlated calculations**  
Adamowicz, L. & Bartlett, R. J. [*J. Chem. Phys.* **86**, 6314-6324 (1987)]
DOI: [10.1063/1.452468](https://doi.org/10.1063/1.452468) -->



<br>
<br>
<br>
<br>

![bottom-right w:350px](images/logo.png)


---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## Motivation

- **Post-HF methods** (MP2, CCSD, CCSD(T)) are essential for chemical accuracy but scale steeply with **virtual orbitals** $V$:
  - MP2: $\mathcal{O}(O^2 V^2)$, CCSD: $\mathcal{O}(O^2 V^4)$, CCSD(T): $\mathcal{O}(O^3 V^4)$
- **Virtual space dominates** computational cost ($V \gg O$)
- **Near‑term quantum devices** (NISQ) have limited qubits & coherence – mapping chemistry requires $2N_{\text{orb}}$ qubits
- **Goal:** Reduce virtual orbitals while preserving correlation energy → **Optimized Virtual Orbital Space (OVOS)** method (Adamowicz & Bartlett, 1987)



---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## Research Objectives

1. **Implement and validate** OVOS using PySCF on a set of small molecules (H$_2$O, CO, HF, NH$_3$, Li$_2$) with 6‑31G and cc‑pVDZ basis sets
2. **Characterize energy recovery** as a function of active virtual orbitals $N'_{\text{virt}}$
3. **Investigate Brueckner orbital condition** – vanishing T1 amplitudes at MP2 level
4. **Quantify benefit** of OVOS orbitals as trial states for **Variational Quantum Eigensolver (VQE)**


---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## Theoretical Background

- **MP2 correlation energy** (full space):
  $$E_{\text{corr}}^{(2)} = \sum_{i>j, a>b} t_{ij}^{ab} \langle ij \| ab \rangle, \quad t_{ij}^{ab} = -\frac{\langle ab \| ij \rangle}{\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j}$$
- **Second‑order Hylleraas functional** $J_2$ – upper bound to $E^{(2)}$:
  $$J_2 = \langle \Psi^{(1)} | H_0 - E^{(0)} | \Psi^{(1)} \rangle + 2 \langle \Psi^{(1)} | V - E^{(1)} | \Psi^{(0)} \rangle$$
- OVOS **minimises $J_2$** with respect to rotations **only between active and inactive virtual orbitals**




---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## Orbital Partitioning & Rotation

- **Three subspaces**: occupied ($i,j$), active virtual ($a,b$), inactive virtual ($e,f$)
- **Unitary rotation** $\mathbf{U} = e^{\mathbf{R}}$ with antisymmetric $\mathbf{R}$:
  $$R = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & R_{ae} \\ 0 & -R_{ae}^T & 0 \end{bmatrix}$$

<br>

- **Gradient** and **Hessian** of $J_2$ derived for Newton‑Raphson update:
  $$G_{ae} = 2\sum_{i>j,b} t_{ij}^{ab} \langle ij \| eb \rangle + 2\sum_{b} D_{ab} f_{eb}, \quad D_{ab} = \sum_{i>j,c} t_{ij}^{ac} t_{ij}^{bc}$$





---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## OVOS Algorithm

```python
1. Perform SCF → initial MO coefficients
2. Transform AO integrals to MO basis
3. Select initial active virtual space
4. Repeat until convergence:
   a. Compute MP1 amplitudes and MP2 energy J₂
   b. Build gradient G and Hessian H (block‑diagonal)
   c. Solve H·ΔR = –G
   d. Form U = exp(ΔR) and rotate MO coefficients
   e. Canonicalize active Fock block
```


---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## Implementation Details

- **PySCF** for integrals, SCF, MO transformations
- **Custom Python class `OVOS`**:
  - Spin‑orbital formalism with interleaved $\alpha$/$\beta$ ordering
  - Efficient **AO → MO integral transformation** for $\langle ij|ab\rangle$ blocks
  - Antisymmetrisation to physicists’ notation
  - Newton‑Raphson with block‑diagonal Hessian (per inactive orbital)
- **Convergence criteria**: $\Delta J_2 < 10^{-8}$ Hartree, $\|\nabla J_2\| < 10^{-4}$, T1 norm < $10^{-8}$
- **Initialisation strategies**: RHF, warm‑start, random unitary sampling

**Example: MP1 amplitude computation**
```python
def _mp1_amplitudes(self, fock_diag, eri_as):
    denom = (fock_diag[occ][:,None,None,None] +
             fock_diag[occ][None,:,None,None] -
             fock_diag[vir][None,None,:,None] -
             fock_diag[vir][None,None,None,:])
    return -eri_as[:nvir_act, :nvir_act, :, :] / denom
```


---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## Computational Setup

| Molecule | Basis sets          | $N_{\text{virt}}$ (cc‑pVDZ) | Active range $N'_{\text{virt}}$ |
|----------|---------------------|-----------------------------|----------------------------------|
| H₂O      | 6‑31G, cc‑pVDZ      | 19                          | 2 – 19 (spin‑orbitals)           |
| CO       | 6‑31G, cc‑pVDZ      | 21                          | 2 – 21                           |
| HF       | 6‑31G, cc‑pVDZ      | 14                          | 2 – 14                           |
| NH₃      | 6‑31G, cc‑pVDZ      | 24                          | 2 – 24                           |
| Li₂      | 6‑31G, cc‑pVDZ      | 25                          | 2 – 25                           |

- All geometries fixed, UHF reference (except RHF for closed‑shell comparison)
- Convergence: max 1000 iterations, level‑shift disabled


---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## Results – Energy Recovery (cc‑pVDZ)

| Molecule | $N_{\text{virt}}$ | $N'_{\text{virt}}$ for 90% recovery | % of full space |
|----------|-------------------|--------------------------------------|------------------|
| H₂O      | 19                | 13                                   | 68%              |
| CO       | 21                | 13                                   | 62%              |
| HF       | 14                | 9                                    | 64%              |
| NH₃      | 24                | 12                                   | 50%              |
| **Li₂**  | 25                | **7**                                | **28%**          |

- **OVOS recovers ~90% MP2 correlation with 50–68% of virtual orbitals** – except Li₂ (28%)
- Approaching full space → convergence difficult (local minima, negative Hessian eigenvalues)



---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->


## Basis Set Dependence (Example: CO)

![width:700px](figures/CO_basis_comparison.png)

- Larger basis (cc‑pVDZ) yields higher energy recovery at same **percentage** of virtuals
- At fixed $N'_{\text{virt}}=5$, 6‑31G recovers more because total $V$ is smaller

---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->


## VQE Reference States – Potential Energy Surfaces

- **VQE without orbital optimisation** (UtUPS ansatz, 1 layer)
- Comparison: UHF, OVOS, UMP2 natural orbitals (NOs)
- **H₂O (6‑31G, 10e, 6 virt)**

![width:700px](figures/H2O_VQE_false.png)

- OVOS orbitals consistently lower energy than UHF, approach UMP2 NOs after “kink”



---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->


## VQE – HF and Li₂

**HF (6‑31G)** – OVOS outperforms UHF except near dissociation

**Li₂ (6‑31G)** – OVOS improves over UHF but UMP2 NOs remain best

**Key observation:**  
- OVOS orbitals often **match or exceed UHF** as VQE starting point  
- **Kinks** correspond to crossing barriers in orbital optimisation landscape  
- UMP2 natural orbitals give lowest energies but require a full MP2 calculation


---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## Orbital‑Optimised VQE (ooVQE) – Iteration Counts

- ooVQE with **previous geometry θ initialisation**  
- **H₂O**: OVOS shows **lower mean iterations** than UHF and UMP2 NOs

![width:550px](figures/H2O_convergence_statistics.png)

- OVOS convergence **more variable** but mean competitive – attractive for resource‑saving




---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## Brueckner Orbital Connection

- Brueckner orbitals satisfy **$t_i^a = 0$** (vanishing singles) – maximises overlap with exact wavefunction
- OVOS stationarity $\nabla_R J_2 = 0$ for **active–inactive rotations** → approximate Brueckner condition in truncated space
- **Observed T1 amplitudes** during OVOS iterations: $10^{-16}$ (similar to HF) – no significant reduction, likely because occupied–virtual rotations were not optimised





---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## Conclusions

✅ **Successful implementation** of OVOS in PySCF, validated on 5 molecules, 2 basis sets  
✅ **~90% MP2 correlation** recovered with **50–68% of virtual orbitals** (Li₂: 28%)  
✅ OVOS orbitals provide **better VQE reference states** than UHF in most cases, competitive with UMP2 natural orbitals  
✅ **Convergence challenges** near full space due to local minima – solved by random initialisation or warm‑start  
✅ Practical value for **NISQ quantum chemistry** – reduces qubit count and circuit depth  





---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->


## Outlook

**Algorithmic improvements:**
- Trust‑region / line‑search for Newton‑Raphson
- GPU acceleration, level‑shifting for indefinite Hessian

**Method extensions:**
- OVOS for **CCSD(T)** and **RHF** reference
- Combine with **natural orbital truncation**

**Quantum computing:**
- Run actual VQE on **quantum hardware** using OVOS orbitals
- Exploit reduced T1 → **UCCD ansatz**, shorter circuits

**Larger systems:**
- Transition metals, periodic systems, basis‑set extrapolation

---

<!-- header: Optimized Virtual Orbital Space (OVOS) -->

## Acknowledgements

- Supervisors Phillip W. K. Jensen and Stephan P. A. Sauer for guidance and support
- Sauer Group members and HQC$^2$ meetings for discussions  

**Code availability:** [github.com/Born-ship-it/optimized_virtual_orbitals](https://github.com/Born-ship-it/optimized_virtual_orbitals)

---


# Thank You!

## **Questions?**

