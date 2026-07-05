---
marp: true
size: 1920x1080
math: katex
paginate: true
theme: academic
fit: False
---

<!-- _class: lead -->
<!-- _paginate: False -->
<!-- paginate: Skip -->

# Optimized Virtual Orbital Space (OVOS) for Quantum Computing
## Master's Thesis Defense
**Author:** Tobias Born Clausen
**Supervisors:** Asst. Prof. Phillip W. K. Jensen and Prof. Stephan P. A. Sauer
**Collaboration:** Molecular Quantum Solution (MQS) ApS
**Date:** 11. June 2026

<br>
<br>
<br>
<br>
<br>
<br>

![bottom-right w:350px](images/logo.png)

<!--
Notes: Welcome!
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
<!-- paginate: True -->

## Motivation: The Quantum Bottleneck

- **NISQ Constraint:** Noisy intermediate-scale quantum (NISQ) devices are limited by qubit count and circuit depth.
- **The Challenge:** Post-HF methods (MP2) scale steeply $\mathcal{O}(O^2 V^2)$ due to the virtual orbital space $V$.
- **The Insight:** Most virtual orbitals contribute negligibly to electron correlation.
- **Thesis Goal:** Implement the *Optimized Virtual Orbital Space (OVOS)* method (Adamowicz & Bartlett, 1987) to reduce virtual orbitals while preserving correlation energy. Benchmark these orbitals for VQE performance.

<div class="reference">
  <div class="reference-item">
    <span class="reference-author">Adamowicz, L. & Bartlett, R. J.</span>
    <span class="reference-title">Optimized virtual orbital space for high-level correlated calculations</span>
    <span class="reference-journal"> [J. Chem. Phys. 86, 6314-6324 (1987)]
DOI: [10.1063/1.452468](https://doi.org/10.1063/1.452468)</span>
  </div>
</div>

<!--
Speaker Notes: 
- NOISY...
- 1000 QUBITS...
- BIG BASIS SET -> LARGE VIRTUAL SPACE -> HIGH COST

- The **computational bottleneck** that motivates everything we do in this thesis.
- That is exactly what the **OVOS method** does...
- I implement OVOS and then ask...
- OVOS -> REDUCED VIRTUAL SPACE -> LOWER COST

**Transition to next slide:**  
"With that motivation in mind, let me now walk you through the **theory** behind OVOS
– how it defines the active virtual space and how we optimise it."
-->

---

<!-- class: title-card -->
<!-- header: 
    <span class="header-left">
    </span>
    <span class="header-right">
    </span>
-->
<!-- paginate: Skip -->

<div class="subtitle">Theory</div>
<div class="meta">Partitioning - Optimisation</div>

<!--
...
-->

---

<!-- class: False -->
<!-- header: 
    <span class="header-left">
        Theory
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

## Theory: Partitioning

- **Index:** The virtual orbital space is partitioned into three subspaces:

<br>

![center w:1050px](images/figures/Ekstra/orbital_partioning_a.png)

<!--
Speaker Notes: 
- OVOS partitions the full spin‑orbital basis into three subspaces...
- Here the orbitals are unrestricted...
-->

---

<!-- class: False -->
<!-- header: 
    <span class="header-left">
        Theory
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

## Theory: Partitioning

- **Index:** The virtual orbital space is partitioned into three subspaces:

<br>

![center w:1050px](images/figures/Ekstra/orbital_partioning_b.png)

<!--
Speaker Notes: ...
-->

---

<!-- class: False -->
<!-- header: 
    <span class="header-left">
        Theory
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

## Theory: Partitioning

- **Index:** The virtual orbital space is partitioned into three subspaces:

<br>

![center w:1050px](images/figures/Ekstra/orbital_partioning_c.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Partitioning
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

- **Goal:** Find the optimal virtual subspace for capturing electron correlation.

<br>

<!--
Speaker Notes: 
- The goal of OVOS is to find the optimal virtual subspace for capturing electron correlation...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Partitioning
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

- **Goal:** Find the optimal virtual subspace for capturing electron correlation.

<br>

![center w:1080px](images/figures/Ekstra/OVOS_number_a.png)

<!--
Speaker Notes: 
- A simple naive approach...
- A&B, Overhead from pre-MP2 calc...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Partitioning
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

- **Goal:** Find the optimal virtual subspace for capturing electron correlation.

<br>

![center w:1080px](images/figures/Ekstra/OVOS_number_b.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Partitioning
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

- **Goal:** Find the optimal virtual subspace for capturing electron correlation.

<br>

![center w:1080px](images/figures/Ekstra/OVOS_number_c.png)

<div class="slide-comment">
N' = N - 1, N - 2, ..., 1 active unoccupied orbital, N' < N
</div>

<!--
Speaker Notes: 
- Add to N' = N-1 ...
- AT N' = N ...
- Optimization...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

## Theory: Optimisation

<br>

![center w:1180px](images/figures/Ekstra/OVOS_loop_a.png)

<!--
Speaker Notes: 
- To give the overveiw of this optimisation loop...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

## Theory: Optimisation

<br>

![center w:1180px](images/figures/Ekstra/OVOS_loop_b.png)

<!--
Speaker Notes: ...
- A broad overview of the optimisation loop...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

## Theory: Optimisation

<br>

![center w:1180px](images/figures/Ekstra/OVOS_loop_c.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

### The Hylleraas Functional

- **Framework:**

$$E^{(2)} \leq \langle\Psi^{(1)}|H_0 - E^{(0)}|\Psi^{(1)}\rangle + 2\langle\Psi^{(1)}|V - E^{(1)}|\Psi^{(0)}\rangle = J^{(2)}$$

<!--
- The second-order Hylleraas functional is used to find an optimal 
rotation of the active virtual space against the nonactive 
space, to minimize the second-order correlation energy.
- The Theory behind the OVOS optimisation ... 
- The second‑order Hylleraas functional J^{(2)} is the variational target that OVOS minimises 
- Orthogonal condition <\Psi^{(1)}|\Psi^{(0)}> = 0
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

### The Hylleraas Functional

- **Framework:**

$$E^{(2)} \leq \langle\Psi^{(1)}|H_0 - E^{(0)}|\Psi^{(1)}\rangle + 2\langle\Psi^{(1)}|V - E^{(1)}|\Psi^{(0)}\rangle = J^{(2)}$$

<div class="slide-comment">
Perturbation theory framework: H = H₀ + V, Eₕ = E₀ + E₁ | Orthogonal: <Ψ¹|Ψ⁰> = 0
</div>

<!--
- The second-order Hylleraas functional is used to find an optimal 
rotation of the active virtual space against the nonactive 
space, to minimize the second-order correlation energy.
- V!!!!
- Energy HF !!!
- Orthonormal condition <\Psi^{(1)}|\Psi^{(0)}> = 0
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

### The Hylleraas Functional

- **Framework:**

$$E^{(2)} \leq \langle\Psi^{(1)}|H_0 - E^{(0)}|\Psi^{(1)}\rangle + 2\langle\Psi^{(1)}|V - E^{(1)}|\Psi^{(0)}\rangle = J^{(2)}$$

- **MP1 amplitude:**

$$
\begin{align*}
    \frac{\partial J^{(2)}}{\partial \Psi^{(1)}} = 0, \quad \Psi^{(1)} = \sum_{i>j,a>b} t_{ij}^{ab} |_{ij}^{ab}\rangle \quad \longrightarrow \quad t_{ij}^{ab} = \frac{\langle ab\|ij\rangle}{\epsilon_i + \epsilon_j - \epsilon_a - \epsilon_b}
\end{align*}
$$


<div class="slide-comment">
Perturbation theory framework: H = H₀ + V, Eₕ = E₀ + E₁ | Orthogonal: <Ψ¹|Ψ⁰> = 0
</div>

<!--
- The second-order Hylleraas functional is used to find an optimal 
rotation of the active virtual space against the nonactive 
space, to minimize the second-order correlation energy.
- Get to MP1 amplitudes by setting the derivative of J^{(2)} w.r.t. Ψ^{(1)} to zero...
- >Ψ¹ is a linear combination of 
- two-electron excited determinants...
- Ψ⁰ SCF reference...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

### The Hylleraas Functional

- **Framework:**

$$E^{(2)} \leq \langle\Psi^{(1)}|H_0 - E^{(0)}|\Psi^{(1)}\rangle + 2\langle\Psi^{(1)}|V - E^{(1)}|\Psi^{(0)}\rangle = J^{(2)}$$

- **Functional Form:**

$$
\begin{align*}
    J^{(2)} = \sum_{i>j} J_{ij}^{(2)}
\end{align*}
$$

<!--
Speaker Notes: ...
- Pair separaible functional...
- Unique pairs...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

### The Hylleraas Functional

- **Framework:** The **Hylleraas functional**

$$E^{(2)} \leq \langle\Psi^{(1)}|H_0 - E^{(0)}|\Psi^{(1)}\rangle + 2\langle\Psi^{(1)}|V - E^{(1)}|\Psi^{(0)}\rangle = J^{(2)}$$

- **Functional Form:**

$$
\begin{align*}
    J_{ij}^{(2)} = \sum_{a>b,c>d} &t_{ij}^{ab} t_{ij}^{cd} \Big(f_{ac}\delta_{bd} - f_{ad}\delta_{bc} + f_{bd}\delta_{ac} - f_{bc}\delta_{ad} \\ &- (\epsilon_i + \epsilon_j)(\delta_{ac}\delta_{bd} - \delta_{ad}\delta_{bc})\Big) + 2\sum_{a>b} t_{ij}^{ab} \langle ij\|ab\rangle
\end{align*}
$$

<!--
Notes: Equations for the Hylleraas functional, and MP2 correlation energy...
Speaker Notes: ...
- Unique Pairs...
- Rotated MO's ... 
- Space...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

### Derivatives

<!--
Speaker Notes: ... 
- J2 -> Landscape...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

### Derivatives

- **Gradient:**

$$
\begin{align*}
    G_{ae} = \frac{\partial J^{(2)}}{\partial R_{ae}} \qquad (N' N_{\text{inact.}})
\end{align*}
$$

<!--
Speaker Notes: ... 
- DRAW: Slope ...
- R_ae: Rotation parameters for active-inactive orbital pairs...
- Stationary point: G = 0 -> Optimal virtual subspace...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

### Derivatives

- **Hessian:**

$$
\begin{align*}
    H_{ae,bf} &= \frac{\partial^2 J^{(2)}}{\partial R_{ae}\ \partial R_{bf}} \qquad (N' N_{\text{inact.}}) \times (N' N_{\text{inact.}})
\end{align*}
$$

<!--
Speaker Notes: ... 
- DRAW: Curvature ...
    - Local minimum, Local maximum, Saddle point...
- H: Block structure...
- H: Diag block (N' X N')
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

### Derivatives

- **Hessian:**

$$
\begin{align*}
    H_{ae,bf} &= \frac{\partial^2 J^{(2)}}{\partial R_{ae}\ \partial R_{bf}} \qquad (N' N_{\text{inact.}}) \times (N' N_{\text{inact.}})
\end{align*}
$$

*Block structure:*
$$
\mathbf{H} =
    \begin{bmatrix}
        \mathbf{H^e}        & 0           & \cdots & 0         \\
        0          & \mathbf{H^f}         & \cdots & 0         \\
        \vdots     & \vdots      & \ddots & \vdots    \\
        0          & 0           & \cdots & \mathbf{H^{N_{\text{inact.}}}}
    \end{bmatrix},
\qquad
\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\quad
$$

<!--
Notes: Gradient and Hessian expression for the Hylleraas functional...
Speaker Notes: ... 
- H: Block structure...
- H: Diag block (N' X N')
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

### Derivatives

- **Hessian:**

$$
\begin{align*}
    H_{ae,bf} &= \frac{\partial^2 J^{(2)}}{\partial R_{ae}\ \partial R_{bf}} \qquad (N' N_{\text{inact.}}) \times (N' N_{\text{inact.}})
\end{align*}
$$

*Block structure:*
$$
\mathbf{H} =
    \begin{bmatrix}
        \mathbf{H^e}        & 0           & \cdots & 0         \\
        0          & \mathbf{H^f}         & \cdots & 0         \\
        \vdots     & \vdots      & \ddots & \vdots    \\
        0          & 0           & \cdots & \mathbf{H^{N_{\text{inact.}}}}
    \end{bmatrix},
\qquad
\mathbf{H^e} = 
    \begin{bmatrix}
        H_{ae,ae}  & H_{ae,be} & \cdots & H_{ae,N'e}   \\
        H_{be,ae}  & H_{be,be} & \cdots & H_{be,N'e}   \\
        \vdots     & \vdots    & \ddots & \vdots       \\
        H_{N'e,ae} & 0         & \cdots & H_{N'e,N'e}
    \end{bmatrix}
$$

<!--
Speaker Notes: ... 
- H: Block structure...
- H: Diag block (N' X N')
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

### Newton–Raphson

**Rotation:**

$$
\begin{align*}
    \mathbf{R} = - \mathbf{G} \cdot \mathbf{H}^{-1}
\end{align*}
$$

<!--
Speaker Notes: ...
- NR: Obtain rotation parameters R...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

### Newton–Raphson

**Rotation:**

$$
\begin{align*}
    \mathbf{R} = - \mathbf{G} \cdot \mathbf{H}^{-1}
\end{align*}
$$

*Block structure:*
$$
\mathbf{R} =
    \begin{bmatrix}
        0  & -\mathbf{R_{ea}} \\
        \mathbf{R_{ae}} & 0
    \end{bmatrix}, \qquad \mathbf{R_{ae}} = -\mathbf{R_{ea}} \qquad  (N' \times N_{\text{inact.}})
$$

<!--
Speaker Notes: ...
- Act-Act orbital != MP2
- Only Act-Inact. orbital rotations...
- Anti-symmetric structure of R...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

### Newton–Raphson

**Rotation:**

$$
\begin{align*}
    \mathbf{R} = - \mathbf{G} \cdot \mathbf{H}^{-1}
\end{align*}
$$

*Block structure:*
$$
\mathbf{R} =
    \begin{bmatrix}
        0  & -\mathbf{R_{ea}} \\
        \mathbf{R_{ae}} & 0
    \end{bmatrix}, \qquad \mathbf{R_{ae}} = -\mathbf{R_{ea}} \qquad  (N' \times N_{\text{inact.}})
$$

*Rotation:*
$$
\phi_a \rightarrow \phi_a' = \phi_a + \sum_e R_{ae} \phi_e - \frac{1}{2} \sum_{b,e} R_{ea} R_{eb} \phi_b + \cdots
$$

<!--
Speaker Notes: ...
- Add onto MO ...
- R_ae: Rotation parameters for active-inactive orbital pairs...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

### Unitary Rotation

**Coefficients:**

$$\mathbf{U} = e^{\mathbf{R}} \quad \longrightarrow \quad \mathbf{U}\mathbf{C}\mathbf{U}^\dagger = \mathbf{C}'\qquad (\mathbf{U}^\dagger \mathbf{U} = \mathbf{I}) $$

<!--
Speaker Notes: ...
- NR: Obtain rotation parameters R...
- UR: Apply rotation to virtual orbitals to get new coefficients C'... 
- UR: Mixes the virtual orbitals, preserving orthonormality...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

### Unitary Rotation

**Coefficients:**

$$\mathbf{U} = e^{\mathbf{R}} \quad \longrightarrow \quad \mathbf{U}\mathbf{C}\mathbf{U}^\dagger = \mathbf{C}'\qquad (\mathbf{U}^\dagger \mathbf{U} = \mathbf{I}) $$

### Canonicalization

**Orbital Canonicalization:**

$$\mathbf{F_{\text{ab}}'} = \mathbf{{C'}_{\text{ab}}^\dagger }\mathbf{F_{\text{ab}}} \mathbf{C'_{\text{ab}}} \quad \longrightarrow \quad \mathbf{F_{\text{ab}}'}\mathbf{C''}=\mathbf{C''}\epsilon$$

<!--
Speaker Notes: ...
- CANONICALIZE -> Well-defined orbital energies
- OC: Ater UR, CANONICALIZE the virtual orbitals
- OC: Maintain orbital energies, ensure stability of optimisation...
- OC: Eigenvectors and eigenvalues of the active Fock block...
- OC: Eigenvalues -> MP1 amplitudes...
-->

---

<!-- header: 
    <span class="header-left">
        Theory, Optimisation
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

### Unitary Rotation

**Coefficients:**

$$\mathbf{U} = e^{\mathbf{R}} \quad \longrightarrow \quad \mathbf{U}\mathbf{C}\mathbf{U}^\dagger = \mathbf{C}'\qquad (\mathbf{U}^\dagger \mathbf{U} = \mathbf{I}) $$

### Canonicalization

**Orbital Canonicalization:**

$$\mathbf{F_{\text{ab}}'} = \mathbf{{C'}_{\text{ab}}^\dagger }\mathbf{F_{\text{ab}}} \mathbf{C'_{\text{ab}}} \quad \longrightarrow \quad \mathbf{F_{\text{ab}}'}\mathbf{C''}=\mathbf{C''}\epsilon$$
$$\epsilon \longrightarrow \text{MP1 amplitudes}$$

<!--
Speaker Notes: ...
- OC: Eigenvalues -> MP1 amplitudes...
-->

---

<!-- class: title-card -->
<!-- header: 
    <span class="header-left">
    </span>
    <span class="header-right">
    </span>
-->
<!-- paginate: Skip -->

<div class="subtitle">Method</div>
<div class="meta">Implementation - Computational Details</div>

<!--
Speaker Notes: ...
-->

---

<!-- class: False -->
<!-- header: 
    <span class="header-left">
        Method
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

## Method: Implementation

**Software:** Implemented and benchmarked OVOS w. PySCF, and SlowQuant.

<!--
Speaker Notes: ...
- PySCF -> Molecule, SCF, integrals
- SlowQuant -> VQE...
-->

---

<!-- class: False -->
<!-- header: 
    <span class="header-left">
        Method, Implementation - OVOS
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

### OVOS

<div class="algorithm">
  <div class="algorithm-caption">Algorithm 1: OVOS Iterative Optimisation</div>
  <div class="algorithm-step"><span class="keyword">Perform </span> SCF calculation</div>
  <div class="algorithm-step"><span class="keyword">Select </span> Initial virtual orbital space</div>
  <div class="algorithm-step"><span class="keyword">Transform </span> AO integrals to MO basis</div>
  <div class="algorithm-step"><span class="keyword">Repeat </span></div>
  <div class="algorithm-step indent-2">Compute MP1 amplitudes and MP2 </div>
  <div class="algorithm-step indent-2">Assemble gradient and Hessian </div>
  <div class="algorithm-step indent-2">Solve Newton–Raphson</div>
  <div class="algorithm-step indent-2">Generate Unitary Matrix</div>
  <div class="algorithm-step indent-2">Update MO coefficients </div>
  <div class="algorithm-step indent-2">Canonicalize active Fock block</div>
  <div class="algorithm-step indent-2">Evaluate new MP2 </div>
  <div class="algorithm-step"><span class="keyword">Until</span> convergence </div>
</div>

<!--
Speaker Notes: ...
- Implemented both restricted and unrestricted orbitals...
-->

---

<!-- header: 
    <span class="header-left">
        Method, Implementation - OVOS
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

**Perform** SCF calculation to obtain initial orbitals and Fock matrix

<pre><code class="language-python">   # OVOS
ovos = OVOS(<span class="hl">mol=mol,                 # Molecule object
            scf=mf,                  # SCF object
            Fao=Fao,                 # AO Fock matrix</span>
            num_opt_virtual_orbs=2,  
            <span class="hl">mo_coeff=mo_coeffs,      # Initial MO coefficients</span>
            init_orbs="RHF",
            verbose=1,
            max_iter=1000,
            conv_energy=1e-8,
            conv_grad=1e-6,
            keep_track_max=50
            ).run(...)
</code></pre>

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Method, Implementation - OVOS
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

**Select** initial virtual orbital space

<pre><code class="language-python">   # OVOS
ovos = OVOS(mol=mol,
            scf=mf,
            Fao=Fao,
            <span class="hl">num_opt_virtual_orbs=2,  # Number of active unoccupied orbitals</span>
            mo_coeff=mo_coeffs,
            init_orbs="RHF",
            verbose=1,
            max_iter=1000,
            conv_energy=1e-8,
            conv_grad=1e-6,
            keep_track_max=50
            ).run(...)
</code></pre>

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Method, Implementation - OVOS
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

**Select** initial virtual orbital space

<pre><code class="language-python">   # OVOS
ovos = OVOS(mol=mol,
            scf=mf,
            Fao=Fao,
            <span class="hl">num_opt_virtual_orbs=2,  # Number of active unoccupied orbitals
            mo_coeff=mo_coeffs,      # Initial MO coefficients</span>
            init_orbs="RHF",
            verbose=1,
            max_iter=1000,
            conv_energy=1e-8,
            conv_grad=1e-6,
            keep_track_max=50
            ).run(...)
</code></pre>

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Method, Implementation - OVOS
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

**Select** initial virtual orbital space

<pre><code class="language-python">   # OVOS
ovos = OVOS(mol=mol,
            scf=mf,
            Fao=Fao,
            <span class="hl">num_opt_virtual_orbs=2,  # Number of active unoccupied orbitals
            mo_coeff=mo_coeffs,      # Initial MO coefficients
            init_orbs="RHF",         # Unrestricted/restricted orbitals</span>
            verbose=1,
            max_iter=1000,
            conv_energy=1e-8,
            conv_grad=1e-6,
            keep_track_max=50
            ).run(...)
</code></pre>

<!--
Speaker Notes: ...
- Restricted: mo_coeffs[0] = mo_coeffs[1] = mf.mo_coeff
- Unrestricted: mo_coeffs[0] = mf.mo_coeff_alpha, mo_coeffs[1] = mf.mo_coeff_beta
- Twice the orbitals... alpha/beta...
-->

---

<!-- header: 
    <span class="header-left">
        Method, Implementation - OVOS
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

**Select** initial virtual orbital space

<pre><code class="language-python">   # OVOS
ovos = OVOS(mol=mol,
            scf=mf,
            Fao=Fao,
            num_opt_virtual_orbs=2,
            mo_coeff=mo_coeffs,
            init_orbs="RHF",
            verbose=1,
            <span class="hl">max_iter=1000,
            conv_energy=1e-8,
            conv_grad=1e-6,
            keep_track_max=50</span>
            ).run(...)
</code></pre>

<!--
Speaker Notes: ...
- Restricted: mo_coeffs[0] = mo_coeffs[1] = mf.mo_coeff
- Unrestricted: mo_coeffs[0] = mf.mo_coeff_alpha, mo_coeffs[1] = mf.mo_coeff_beta
- Twice the orbitals... alpha/beta...
-->

---

<!-- header: 
    <span class="header-left">
        Method, Implementation - OVOS
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

**Select** initial virtual orbital space

<pre><code class="language-python">   # OVOS
ovos = OVOS(...).run(<span class="hl">mo_coeff=mo_coeffs,      # MO coefficients
                     fock_spin=fock_spin      # Fock matrix in spin</span>
                     )
</code></pre>

<!--
Speaker Notes: ...
- INSIDE THE RUN!!!!
-->

---

<!-- header: 
    <span class="header-left">
        Method, Implementation - OVOS
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

**Repeat** until convergence

<br>

![center w:1350px](images/figures/Ekstra/OVOS_flowchart.png)

<!--
Speaker Notes: ...
- INITIAL...
- CONVERGENCE...
- LOOP...
-->

---

<!-- class: False -->
<!-- header: 
    <span class="header-left">
        Method, Implementation - VQE
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

### VQE

<div class="algorithm">
  <div class="algorithm-caption">Algorithm 2: VQE Workflow (Classical)</div>
  <div class="algorithm-step"><span class="keyword">Perform </span> Orbital reference state </div>
  <div class="algorithm-step"><span class="keyword">Select </span> Active space </div>
  <div class="algorithm-step"><span class="keyword">Set </span> Orbital optimization True/False</div>
  <div class="algorithm-step"><span class="keyword">Initialise </span> Theta parameters from previous or random </div>
  <div class="algorithm-step"><span class="keyword">Run </span> Unrestricted wavefunction unitary product state †
 </div>
  <div class="algorithm-step"><span class="keyword">Optimise </span> Theta parameters with classical optimiser BFGS</div>
</div>

$^\dagger$ $|\Psi\rangle = \prod_P \hat{U}_P |\Phi_{\text{ref}}\rangle$

<div class="reference">
  <div class="reference-item">
    <span class="reference-author">Pernille Volsgaard</span>
    <span class="reference-title">SlowQuant: Unrestricted Variational Quantum Eigensolver</span>
    <span class="reference-journal"> <br> [Branch: https://github.com/erikkjellgren/SlowQuant/tree/unrestricted_vqe]</span>
  </div>
</div>

<!--
Speaker Notes: ...
- Q-Gates -> Unitary operators...
- NON-Classical
- Quantum HYBRID algorithm...
- Exact, Compact, Conserve properties of Hamiltonian...
- BFGS stands for Broyden–Fletcher–Goldfarb–Shanno algorithm (quasi-NR method for optimization)...
-->

---

<!-- header: 
    <span class="header-left">
        Method, Implementation - VQE
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->

**Run** Unrestricted wavefunction unitary product space.

<pre><code class="language-python">wf = <span class="hl">Unrestricted</span>WaveFunctionUPS(
        mol.nelectron,
        <span class="hl">((mol.nelectron//2, mol.nelectron//2),        # Electron count by spin
        num_electrons//2+num_opt_virtual_orbs),       # Active space
        mo_coeffs,                                    # Orbtial coefficients</span>
        h_core,
        g_eri,
        "utups",
        {"n_layers": 1},
        include_active_kappa=False)
</code></pre>

<div class="reference">
  <div class="reference-item">
    <span class="reference-author">Pernille Volsgaard</span>
    <span class="reference-title">SlowQuant: Unrestricted Variational Quantum Eigensolver</span>
    <span class="reference-journal"> <br> [Branch: https://github.com/erikkjellgren/SlowQuant/tree/unrestricted_vqe]</span>
  </div>
</div>

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Method, Implementation - VQE
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

**Run** Unrestricted wavefunction unitary product space.

<pre><code class="language-python">wf = UnrestrictedWaveFunctionUPS(
        mol.nelectron,
        ((mol.nelectron//2, mol.nelectron//2),
        num_electrons//2+num_opt_virtual_orbs),
        mo_coeffs,
        h_core,
        g_eri,
        <span class="hl">"utups",                                      # Ansatz type
        {"n_layers": 1},                              # Layers in ansatz</span>
        include_active_kappa=False)
</code></pre>

<div class="reference">
  <div class="reference-item">
    <span class="reference-author">Pernille Volsgaard</span>
    <span class="reference-title">SlowQuant: Unrestricted Variational Quantum Eigensolver</span>
    <span class="reference-journal"> <br> [Branch: https://github.com/erikkjellgren/SlowQuant/tree/unrestricted_vqe]</span>
  </div>
</div>

<!--
Speaker Notes: ...
- Circuit !!!
- Layers improve accuracy, but increase cost...
- Tiled means \hat{U} tiles...
- Accuracy, move qubits by spin...
-->

---

<!-- header: 
    <span class="header-left">
        Method, Implementation - VQE
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

**Run** Unrestricted wavefunction unitary product space.

<pre><code class="language-python">wf = UnrestrictedWaveFunctionUPS(
        mol.nelectron,
        ((mol.nelectron//2, mol.nelectron//2),
        num_electrons//2+num_opt_virtual_orbs),       # Active space
        mo_coeffs,                                    # Orbtial coefficients
        h_core,
        g_eri,
        <span class="hl">"utups",
        {"n_layers": 1},</span>
        include_active_kappa=False)
</code></pre>

![bottom-right w:550px](images/figures/Ekstra/VQE_layers.png)

<div class="reference">
  <div class="reference-item">
    <span class="reference-author">Pernille Volsgaard</span>
    <span class="reference-title">SlowQuant: Unrestricted Variational Quantum Eigensolver</span>
    <span class="reference-journal"> <br> [Branch: https://github.com/erikkjellgren/SlowQuant/tree/unrestricted_vqe]</span>
  </div>
</div>

<!--
Speaker Notes: ...
- Circuit !!!
- Layers improve accuracy, but increase cost...
- Tiled means \hat{U} tiles...
- Accuracy, move qubits by spin...
-->

---

<!-- header: 
    <span class="header-left">
        Method, Implementation - VQE
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

**Optimise:** Theta parameters w. classical optimiser BFGS.

<pre><code class="language-python"> wf.thetas = theta_init
wf.run_wf_optimization_1step(
        "BFGS",
        <span class="hl">orbital_optimization,</span>
        atol=1e-6,
        maxiter)
</code></pre>

<!--
Speaker Notes: ...
- quasi-Newton-Raphson method for optimization...
-->

---

<!-- class: title-card -->
<!-- header: 
    <span class="header-left">
    </span>
    <span class="header-right">
    </span>
-->
<!-- paginate: Skip -->

<div class="subtitle">Results</div>
<div class="meta">OVOS - VQE - ooVQE</div>

<!--
Speaker Notes: ...
-->

---

<!-- class: False -->
<!-- header: 
    <span class="header-left">
        Results, Computational Details
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

## Computational Details

- **Molecules:** H₂O CO, HF, NH₃, Li₂.
- **Basis Set:** 6-31G, cc-pVDZ.
- **Start guesses:** Orbitals from HF, previous OVOS, and random.
- **Convergence Criteria:** Energy change < 1e-8 Ha, gradient norm < 1e-6.

<!-- 
Speaker Notes: ...
Molecules: H2O -> Water ,CO -> Carbon Monoxide, HF -> Hydrogen Fluoride, NH3 -> NH3 = Ammonia
-> Li2 = Dilithiuma. (Different sizes...)
-> Varying spaces...
-> Basis sets: 6-31G (smaller), cc-pVDZ (larger, more accurate)...
-->

---

<!-- class: False -->
<!-- header: 
    <span class="header-left">
        Results, Computational Details
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

## Computational Details

- **Molecules:** ***H₂O***, CO, ***HF***, NH₃, ***Li₂***.
- **Basis Set:** 6-31G, cc-pVDZ.
- **Start guesses:** Orbitals from HF, previous OVOS, and random.
- **Convergence Criteria:** Energy change < 1e-8 Ha, gradient norm < 1e-6.

<!-- 
Speaker Notes: ...
Molecules: H2O -> Water ,CO -> Carbon Monoxide, HF -> Hydrogen Fluoride, NH3 -> NH3 = Ammonia
-> Li2 = Dilithiuma. (Different sizes...)
-> Varying spaces...
-> Basis sets: 6-31G (smaller), cc-pVDZ (larger, more accurate)...
-->

---

<!-- class: False -->
<!-- header: 
    <span class="header-left">
        Results, Computational Details
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

## Computational Details

- **Molecules:** ***H₂O***, CO, ***HF***, NH₃, ***Li₂***.
- **Basis Set:** 6-31G, ***cc-pVDZ***.
- **Start guesses:** Orbitals from HF, previous OVOS, and random.
- **Convergence Criteria:** Energy change < 1e-8 Ha, gradient norm < 1e-6.

<!-- 
Speaker Notes: ...
Molecules: H2O -> Water ,CO -> Carbon Monoxide, HF -> Hydrogen Fluoride, NH3 -> NH3 = Ammonia
-> Li2 = Dilithiuma. (Different sizes...)
-> Varying spaces...
-> Basis sets: 6-31G (smaller), cc-pVDZ (larger, more accurate)...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

## Results: OVOS

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

<!-- ## Results: OVOS -->
<!-- **PLOTS - H2O/cc-pVDZ** -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ_empty.png)

<!--
- Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ_start_guess_a_RHF.png)

<!--
Speaker Notes: ...
- *Points:* MP2 correlation energy at each iteration.
- *Lines:* Amount of recovered correlation energy from initial point to final OVOS solution.
- *Grey Line:* Best of start Guesses (canonical HF, previous OVOS, random).
- *Dashed Lines:* Reference MP2, and orbital optimised MP2 (both with full virtual space).

-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ_start_guess_a_prev.png)

<!--
Speaker Notes: ...
- *Points:* MP2 correlation energy at each iteration.
- *Lines:* Amount of recovered correlation energy from initial point to final OVOS solution.
- *Grey Line:* Best of start Guesses (canonical HF, previous OVOS, random).
- *Dashed Lines:* Reference MP2, and orbital optimised MP2 (both with full virtual space).

-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ_start_guess_a.png)

<!--
Speaker Notes: ...
- *Points:* MP2 correlation energy at each iteration.
- *Lines:* Amount of recovered correlation energy from initial point to final OVOS solution.
- *Grey Line:* Best of start Guesses (canonical HF, previous OVOS, random).
- *Dashed Lines:* Reference MP2, and orbital optimised MP2 (both with full virtual space).

-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ_start_guess_c.png)

<!--
Speaker Notes: ...
- *Points:* MP2 correlation energy at each iteration.
- *Lines:* Amount of recovered correlation energy from initial point to final OVOS solution.
- *Grey Line:* Best of start Guesses (canonical HF, previous OVOS, random).
- *Dashed Lines:* Reference MP2, and orbital optimised MP2 (both with full virtual space).

-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ_start_guess_b.png)

<!--
- Speaker Notes: ...
- *Points:* MP2 correlation energy at each iteration.
- *Lines:* Amount of recovered correlation energy from initial point to final OVOS solution.
- *Grey Line:* Best of start Guesses (canonical HF, previous OVOS, random).
- *Dashed Lines:* Reference MP2, and orbital optimised MP2 (both with full virtual space).

-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ_start_guess_d.png)

<!--
Speaker Notes: ...
- *Points:* MP2 correlation energy at each iteration.
- *Lines:* Amount of recovered correlation energy from initial point to final OVOS solution.
- *Grey Line:* Best of start Guesses (canonical HF, previous OVOS, random).
- *Dashed Lines:* Reference MP2, and orbital optimised MP2 (both with full virtual space).

-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ_start_guess_e.png)

<!--
Speaker Notes: ...
- *Points:* MP2 correlation energy at each iteration.
- *Lines:* Amount of recovered correlation energy from initial point to final OVOS solution.
- *Grey Line:* Best of start Guesses (canonical HF, previous OVOS, random).
- *Dashed Lines:* Reference MP2, and orbital optimised MP2 (both with full virtual space).

-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ_to_low.png)

<!--
Speaker Notes: ...
- Hessian blocks start getting bigger
- 4,5 big recovery...
- iteration counts...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ_to_middle.png)

<!--
Speaker Notes: ...
- Hessian blocks start getting bigger
- 4,5 big recovery...
- iteration counts...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ.png)

<!--
Speaker Notes: ...
- 9,10 big recovery...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/cc-pVDZ/ovos_convergence_H2O_cc-pVDZ_zoom_last.png)

<!--
Speaker Notes: ...
- We clearly see that we do not go below MP2... with RHF OVOS, finding MP2 
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/ovos_basis_set_H2O_empty.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/ovos_basis_set_H2O_unmarked.png)

<!--
Speaker Notes: ...
- Convergence in OVOS solutions -> 90% !!!
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/ovos_basis_set_H2O.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/H2O/ovos_basis_set_H2O_zoom.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** HF ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

![center w:1100px](images/figures/OVOS_PLOTS/HF/cc-pVDZ/ovos_convergence_HF_cc-pVDZ.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** HF ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/HF/ovos_basis_set_HF_unmarked.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** HF ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/HF/ovos_basis_set_HF.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** HF ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/HF/ovos_basis_set_HF_zoom.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** ***HF*** Li2
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

![center w:1100px](images/figures/OVOS_PLOTS/Li2/cc-pVDZ/ovos_convergence_Li2_cc-pVDZ.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** ***HF*** Li2
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/Li2/cc-pVDZ/ovos_convergence_Li2_cc-pVDZ_zoom_bad.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** ***HF*** Li2
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/Li2/cc-pVDZ/ovos_convergence_Li2_cc-pVDZ_zoom_bad_15.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** ***HF*** Li2
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:975px](images/figures/Ekstra/ovos_iterations_Li2_cc-pVDZ_30.png)

<!--
Speaker Notes: ...
- Iteration counts...
- Hessian blocks... 
- Ill-conditioned Hessian...
- Negative values -> saddle point...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** ***HF*** Li2
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/Li2/ovos_basis_set_Li2_unmarked.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** ***HF*** Li2
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/Li2/ovos_basis_set_Li2_unmarked.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** ***HF*** Li2
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/Li2/ovos_basis_set_Li2.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS - ***H2O*** ***HF*** Li2
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1100px](images/figures/OVOS_PLOTS/Li2/ovos_basis_set_Li2_zoom.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, OVOS
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

## Results: OVOS Summarized

| Molecule | Fullspace | $N'_{\text{virt}}$ (90% MP2) | % of full space |
|----------|----------- |------------------------------|-----------------|
| HF       | 14                | 9                            | 64%             |
| H₂O      | 19                | 13                           | 47%             |
| CO       | 21                | 13                           | 62%             |
| NH₃      | 24                | 12                           | 50%             |
| **Li₂**  | **25**                | **7**                        | **28%**         |
| CH₂$^\dagger$ | 70                | ~21                           | ~30%             |

$^\dagger$ Adamowicz & Bartlett (1992) J. Chem. Phys., 86, 1987.

**Key Takeaway:** The first 28-64% of orbitals capture the bulk of the correlation. 

<!--
Speaker Notes: ...
- All molecules
- Bartlett paper...
- Li2 is best!!! 
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE 
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

## VQE

- **Goal:** Assess how OVOS orbitals improve VQE performance.

<br>

![center w:700px](images/figures/Ekstra/PES_HF.png)

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE 
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

## VQE

- **Active space:** Strictly 75% of virtual orbitals -> >95% of correlation energy.

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

![center w:1050px](images/figures/VQE_PLOTS/H2O/VQE_H2O_6-31G_best_of_zoom_oo_False_empty.png)

<div class="slide-comment">
Active space: 5 occupied + 11 virtual orbitals (75% of virtual space).
</div>

<!--
Speaker Notes: ...
- *Points:* VQE energy at each iteration (Previous and Random).
- *Lines:* VQE energy convergence from initial point to final OVOS solution (UHF OVOS, UHF, and UMP2).
- *Dashed Line:* Reference UHF VQE.
- Best of previous and random initial theta parameters
- JUMP POT. SURFACE...
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

![center w:1050px](images/figures/VQE_PLOTS/H2O/VQE_H2O_6-31G_best_of_zoom_oo_False_water.png)

<div class="slide-comment">
Active space: 5 occupied + 11 virtual orbitals (75% of virtual space).
</div>

<!--
Speaker Notes: ...
- *Points:* VQE energy at each iteration (Previous and Random).
- *Lines:* VQE energy convergence from initial point to final OVOS solution (UHF OVOS, UHF, and UMP2).
- *Dashed Line:* Reference UHF VQE.
- Best of previous and random initial theta parameters
- JUMP POT. SURFACE...
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

![center w:1050px](images/figures/VQE_PLOTS/H2O/VQE_H2O_6-31G_best_of_zoom_oo_False.png)

<div class="slide-comment">
Active space: 5 occupied + 11 virtual orbitals (75% of virtual space).
</div>

<!--
Speaker Notes: ...
- *Points:* VQE energy at each iteration (Previous and Random).
- *Lines:* VQE energy convergence from initial point to final OVOS solution (UHF OVOS, UHF, and UMP2).
- *Dashed Line:* Reference UHF VQE.
- Best of previous and random initial theta parameters
- JUMP POT. SURFACE...
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: Hold -->

![center w:1050px](images/figures/VQE_PLOTS/H2O/VQE_H2O_6-31G_best_of_zoom_oo_True.png)

<div class="slide-comment">
Active space: 5 occupied + 11 virtual orbitals (75% of virtual space).
</div>

<!--
Speaker Notes: ...
- Best of previous and random initial theta parameters
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE - H2O ***HF*** ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: Hold -->

![center w:1050px](images/figures/VQE_PLOTS/H2O/VQE_H2O_iterations_to_convergence_statistics%20copy.png)

<div class="slide-comment">
Active space: 5 occupied + 11 virtual orbitals (75% of virtual space).
</div>

<!--
Speaker Notes: ...
- The off poing above, is then the 1 iteration in this plot!
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE - ***H2O*** HF ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

![center w:1050px](images/figures/VQE_PLOTS/HF/VQE_HF_6-31G_best_of_zoom_oo_False.png)

<div class="slide-comment">
Active space: 5 occupied + 10 virtual orbitals (75% of virtual space).
</div>

<!--
Speaker Notes: ...
- Best of previous and random initial theta parameters
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE - ***H2O*** HF ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: Hold -->

![center w:1050px](images/figures/VQE_PLOTS/HF/VQE_HF_6-31G_best_of_zoom_oo_True.png)

<div class="slide-comment">
Active space: 5 occupied + 10 virtual orbitals (75% of virtual space).
</div>

<!--
Speaker Notes: ...
- Best of previous and random initial theta parameters
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE - ***H2O*** HF ***Li2***
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: Hold -->

![center w:1050px](images/figures/VQE_PLOTS/HF/VQE_HF_iterations_to_convergence_statistics%20copy.png)

<div class="slide-comment">
Active space: 5 occupied + 10 virtual orbitals (75% of virtual space).
</div>

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE - ***H2O*** ***HF*** Li2
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

![center w:1050px](images/figures/VQE_PLOTS/Li2/VQE_Li2_6-31G_best_of_zoom_oo_False.png)

<div class="slide-comment">
Active space: 3 occupied + 15 virtual orbitals (75% of virtual space).
</div>

<!--
Speaker Notes: ...
- Best of previous and random initial theta parameters
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE - ***H2O*** ***HF*** Li2
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: hold -->

![center w:1050px](images/figures/VQE_PLOTS/Li2/VQE_Li2_6-31G_best_of_zoom_oo_True.png)

<div class="slide-comment">
Active space: 3 occupied + 15 virtual orbitals (75% of virtual space).
</div>

<!--
Speaker Notes: ...
- Best of previous and random initial theta parameters
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE - ***H2O*** ***HF*** Li2
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: Hold -->

![center w:1050px](images/figures/VQE_PLOTS/Li2/VQE_Li2_iterations_to_convergence_statistics.png)

<div class="slide-comment">
Active space: 3 occupied + 15 virtual orbitals (75% of virtual space).
</div>

<!--
Speaker Notes: ...
-->

---

<!-- header: 
    <span class="header-left">
        Results, VQE
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

## Results: VQE Summarized

- **VQE Bottleneck:** Canonical HF is a poor trial state.
- **OVOS Advantage over UHF:** Superior trial state
$\qquad \qquad \qquad \rightarrow$ better groundstate, lower energy convergence.
- **OVOS Disadvantage over UMP2:** Inferior trial state 
$\qquad \qquad \qquad \rightarrow$ worse groundstate, higher energy convergence.
- **Metric:** Similar mean, big spread, in iterations for ooVQE.
- **Conclusion:** Classical optimisation of virtual orbitals directly reduces quantum circuit depth.

<!--
Speaker Notes: ...
-->

---

<!-- class: title-card -->
<!-- header: 
    <span class="header-left">
    </span>
    <span class="header-right">
    </span>
-->
<!-- paginate: Skip -->

<div class="subtitle">Conclusion & Outlook</div>

<!--
Speaker Notes: ...
-->

---
<!-- class: False -->
<!-- header: 
    <span class="header-left">
        Conclusion
    </span>
    <span class="header-right">
        OVOS
    </span> 
-->
<!-- paginate: True -->

## Conclusion

- **Successful implementation** of OVOS.  
- **~90% MP2 correlation** recovered with **28-64% of virtual orbitals**.  
- OVOS orbitals provide **superior VQE reference states** compared to UHF.  
- Practical path to **reducing qubit count and circuit depth** for NISQ chemistry.  

<!--
Speaker Notes: ...
-->

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

- Ordering of orbitals
- Convergence for Newton‑Raphson

<br>

<strong>Quantum computing:</strong>

- Run VQE on quantum hardware using OVOS orbitals
- Explore reduced T1, UCCSD → UCCD ansatz, shorter circuits

</div>

<div>

<strong>Larger systems:</strong>

- Transition metals
- Periodic systems
- Basis‑sets

</div>

</div>

<!--
Speaker Notes: ...
-->

---

<!-- class: title-card -->
<!-- header: 
    <span class="header-left">
    </span>
    <span class="header-right">
    </span>
-->
<!-- paginate: Skip -->

<div class="subtitle">Thank You!</div>

<!--
Speaker Notes: ...
-->

---

<!-- class: title-card -->
<!-- header: 
    <span class="header-left">
    </span>
    <span class="header-right">
    </span>
-->
<!-- paginate: Skip -->

<div class="subtitle">Thank You!</div>
<div class="meta">Questions?</div>

<!--
Speaker Notes: ...
-->