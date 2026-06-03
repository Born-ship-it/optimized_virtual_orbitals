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
## Master's Thesis Defense
**Author:** Tobias Born Clausen
**Supervisors:** Asst. Prof. Phillip W. K. Jensen and Prof. Stephan P. A. Sauer

<br>
<br>
<br>
<br>
<br>
<br>


![bottom-right w:350px](images/logo.png)


<!--
Notes: Present for 30 mins!!! 
-->

---

<!-- header: 
    <span class="header-left">
        Motivation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Motivation: The Quantum Bottleneck

- **The Challenge:** Post-HF methods (MP2, CCSD) scale steeply $\mathcal{O}(O^2 V^2)$ due to the virtual orbital space $V$.
- **NISQ Constraint:** Near-term quantum devices (VQE) are limited by qubit count and circuit depth.
- **The Insight:** Most virtual orbitals contribute negligibly to electron correlation.
- **Thesis Goal:** Implement the *Optimized Virtual Orbital Space (OVOS)* method (Adamowicz & Bartlett, 1987) to reduce virtual orbitals while preserving correlation energy. Apply this to improve VQE performance.

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Theory
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Theory: Partioning

- **Goal:** Find the optimal virtual subspace for capturing electron correlation.

- ...

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Theory
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Theory: The Hylleraas Functional

- **Framework:** The **Hylleraas functional** is the mathematical target for second-order correlation energy:
  $$E^{(2)} \leq \langle\Psi^{(1)}|H_0 - E^{(0)}|\Psi^{(1)}\rangle + 2\langle\Psi^{(1)}|V - E^{(1)}|\Psi^{(0)}\rangle = J_2$$

- **Functional Form:** Each unique pairs of occupied orbital indices $(i,j)$ defines a functional $J_{ij}^{(2)}$ that depends on the virtual space through the amplitudes $t_{ij}^{ab}$:
  $$
  J_{ij}^{(2)} = \sum_{a>b,c>d} t_{ij}^{ab} t_{ij}^{cd} (f_{ac}\delta_{bd} - f_{ad}\delta_{bc} + f_{bd}\delta_{ac} - f_{bc}\delta_{ad} - (\epsilon_i + \epsilon_j)(\delta_{ac}\delta_{bd} - \delta_{ad}\delta_{bc})) + 2\sum_{a>b} t_{ij}^{ab} \langle ij\|ab\rangle
  $$

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Theory
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Theory: Optimization

**Optimization Strategy:** We perform a unitary rotation of the virtual orbitals to minimize $J_{ij}^{(2)}$ for each occupied pair $(i,j)$.

$$...$$

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Theory
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Theory: Optimization

- **Gradient:** The gradient of $J_{ij}^{(2)}$ with respect to the virtual orbital rotation parameters $\kappa_{ab}$ is computed analytically.

$$...$$

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Theory
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Theory: Optimization

- **Hessian:** The Hessian matrix is approximated using the orbital energy differences, enabling efficient optimization.

$$...$$

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Theory
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Theory: Optimization


- **Newton–Raphson:** We use the Newton–Raphson method to iteratively update the virtual orbitals until convergence.

$$...$$

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Method
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->


## Method: Implementation

**Software:** Implemented and benchmarked OVOS w. PySCF, and SlowQuant.

<div class="algorithm">
  <div class="algorithm-caption">Algorithm 1: OVOS Iterative Optimisation</div>
  <div class="algorithm-step"><span class="keyword">Perform </span> SCF calculation</div>
  <div class="algorithm-step"><span class="keyword">Transform </span> AO integrals to MO basis</div>
  <div class="algorithm-step"><span class="keyword">Select </span> Initial virtual orbital space</div>
  <div class="algorithm-step"><span class="keyword">Repeat </span></div>
  <div class="algorithm-step indent-2"> Compute MP1 amplitudes and MP2 </div>
  <div class="algorithm-step indent-2">Assemble gradient and Hessian </div>
  <div class="algorithm-step indent-2">Solve Newton–Raphson</div>
  <div class="algorithm-step indent-2">Generate Unitary Matrix</div>
  <div class="algorithm-step indent-2">Update MO coefficients </div>
  <div class="algorithm-step indent-2">Canonicalize active Fock block</div>
  <div class="algorithm-step indent-2">Evaluate new MP2 </div>
  <div class="algorithm-step"><span class="keyword">Until</span> convergence </div>
</div>


<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Method
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->


## Method: Implementation

**Perform** SCF calculation to obtain canonical orbitals. 

```python
mf = scf.RHF(mol)

```


<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Method
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->


## Method: Implementation

**Transform** AO integrals to MO basis.

```python
mf = scf.RHF(mol)

```


<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Method
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->


## Method: Implementation

**Select** initial virtual orbital space (canonical HF, previous OVOS, or random).


```python
mf = scf.RHF(mol)

```


<!--
- Speaker Notes: 
-->


---

<!-- header: 
    <span class="header-left">
        Method
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->


## Method: Implementation

**Repeat** until convergence: ...


```python
mf = scf.RHF(mol)

```


<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Computational Details
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->


## Computational Details

- **Molecules:** H₂O, CO, HF, NH₃, Li₂.
- **Basis Set:** 6-31G, cc-pVDZ.
- **Start guesses:** Canonical HF virtual orbitals, previous OVOS, and random.
- **Convergence Criteria:** Energy change < 1e-8 a.u., gradient norm < 1e-6.
- **Software:** PySCF for SCF and integral transformations. SlowQuant for unrestricted wavefunction in VQE.

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Results
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Results: OVOS

**PLOTS - H2O/cc-pVDZ**

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Results
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Results: OVOS

**PLOTS - CO/cc-pVDZ**

<!--
- Speaker Notes: 
-->



---

<!-- header: 
    <span class="header-left">
        Results
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Results: OVOS

**PLOTS - HF/cc-pVDZ**

<!--
- Speaker Notes: 
-->



---

<!-- header: 
    <span class="header-left">
        Results
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Results: OVOS

**PLOTS - NH3/cc-pVDZ**

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Results
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Results: OVOS

**PLOTS - Li2/cc-pVDZ**

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Results
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Results: OVOS Summarized


| Molecule | Fullspace | $N'_{\text{virt}}$ (90% MP2) | % of full space |
|----------|-------------------|------------------------------|-----------------|
| H₂O      | 19                | 13                           | 68%             |
| CO       | 21                | 13                           | 62%             |
| HF       | 14                | 9                            | 64%             |
| NH₃      | 24                | 12                           | 50%             |
| **Li₂**  | 25                | **7**                        | **28%**         |

<br>

**Key Takeaway:** The first 50-68% of orbitals capture the bulk of the correlation.

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Results
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Results: VQE

**PLOTS - H2O/cc-pVDZ**

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Results
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Results: VQE

**PLOTS - HF/cc-pVDZ**

<!--
- Speaker Notes: 
-->


---

<!-- header: 
    <span class="header-left">
        Results
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Results: VQE

**PLOTS - Li2/cc-pVDZ**

<!--
- Speaker Notes: 
-->


---

<!-- header: 
    <span class="header-left">
        Results
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Results: VQE Summarized

![width:600px](figures/PES_HF.png)

- **VQE Bottleneck:** Canonical HF is a poor trial state.
- **OVOS Advantage:** Provides a superior trial state $\rightarrow$ better groundstate, lower energy convergence.
- **OVOS Disadvantage:** Provides an inferior trial state $\rightarrow$ worse groundstate, higher energy convergence.
- **Metric:** Reduced mean iterations for ooVQE.
- **Conclusion:** Classical optimization of virtual orbitals directly reduces quantum circuit depth.

<!--
- Speaker Notes: 
-->

---

<!-- header: 
    <span class="header-left">
        Conclusion
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Conclusion

✅ **Successful implementation** of OVOS.  
✅ **~90% MP2 correlation** recovered with **50–68% of virtual orbitals**.  
✅ OVOS orbitals provide **superior VQE reference states** compared to UHF.  
✅ Practical path to **reducing qubit count and circuit depth** for NISQ chemistry.  


---

<!-- header: 
    <span class="header-left">
        Outlook
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

## Outlook

<div class="columns_2_even">

<div>
<strong>Algorithmic improvements:</strong>

- Initial choice of virtual orbitals
- Trust‑region / line‑search for Newton‑Raphson
- GPU acceleration, level‑shifting for indefinite Hessian

<strong>Method extensions:</strong>

- OVOS for CCSD(T) and RHF reference
- Combine with natural orbital truncation

</div>

<div>
<strong>Quantum computing:</strong>

- Run VQE on quantum hardware using OVOS orbitals
- Explore reduced T1 → UCCD ansatz, shorter circuits
 
<strong>Larger systems:</strong>

- Transition metals
- Periodic systems
- Basis‑sets

</div>

</div>

---

<!-- header: 
    <span class="header-left">
    </span>
    <span class="header-right">
    </span>
-->

# Thank You!
### Questions?

<!--
- Speaker Notes: 
-->
