
<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

## 1️⃣ Deriving $J_2$ from Hylleraas Functional

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

### Deriving the second-order functional $J_2$

**Goal:** Evaluate the Hylleraas variational functional
$$
J_2 = \langle\Phi_1|H_0 - E_0|\Phi_1\rangle + 2\langle\Phi_1|V|\Phi_0\rangle
$$
with the MP2 double-excitation wavefunction
$$
|\Phi_1\rangle = \sum_{I>J}\sum_{A>B} t_{IJ}^{AB} |{}_{IJ}^{AB}\rangle
$$
to obtain the explicit pairwise functional:
$$
\boxed{J_2 = \sum_{I>J} J_{IJ}^{(2)}}
$$

**Strategy:** Evaluate $\langle\Phi_1|H_0|\Phi_1\rangle$ and $\langle\Phi_1|V|\Phi_0\rangle$ separately, then combine.

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->


<div class="columns_2_even">
<div>

#### The setup:

Consider a closed N electron system.

Total Hamiltonian:
$$H = H_0 + V$$
where, $H_0 = \sum_i f_i$, and $V=H-H_0$.

HF energy: 
$$E_{HF} = E_0 + E_1$$
so $E_2$ is the lowest-order correlation.

</div>
<div>

![center w:550](images/spin_orbitals.png)

</div>
</div>


> *Note:* $E_{HF} = E_0 + E_1$. $E_2$ is the lowest-order correlation.

<!---
Total H of the system is partitioned into a zero-order Hamiltonian and Correlation perturabtion operator

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

**Step-by-step**

1) Write the Hylleraas functional,
$$
\begin{aligned}
  J_2 = \langle\Phi_1|H_0-E_0|\Phi_1\rangle + 2\,\langle\Phi_1|V-E_1|\Phi_0\rangle
\end{aligned}
$$
explicitly with the HF determinant, $|\Phi_1\rangle$, and the double‑excitation expansion for $\Phi_1$:
$$
\begin{aligned}
    \ket{\Phi_1} =
    \sum_{I>J, A>B} t_{IJ}^{AB} \ket{_{IJ}^{AB}}
\end{aligned}
$$

2) Evaluate $\langle\Phi_1|H_0 - E_0|\Phi_1\rangle$.

**Key assumption:** Canonical orbitals — $H_0$ is diagonal on each determinant:
$$
H_0\,|{}_{IJ}^{AB}\rangle = (\underbrace{\color{red}{\epsilon_A\ +\ \epsilon_B}}_{\text{virt. eigenval.}} \ \underbrace{\color{green}{-\ \epsilon_I\ -\ \epsilon_J}}_{\text{occ. eigenval.}})\,|{}_{IJ}^{AB}\rangle
$$



> *Note:* Brillouin's theorem and V has max 2-electron terms.

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

<br>

Expand $|\Phi_1\rangle$ as double sum and evaluate:
$$
\begin{aligned}
\langle\Phi_1|H_0|\Phi_1\rangle &= \sum_{I>J}\sum_{A>B}\sum_{C>D} t_{IJ}^{AB} t_{IJ}^{CD} (\color{red}{\epsilon_A+\epsilon_B}\color{green}{-\epsilon_I-\epsilon_J})\langle{}_{IJ}^{AB}|{}_{IJ}^{CD}\rangle
\end{aligned}
$$

Orthonormality of determinants: $\langle{}_{IJ}^{AB}|{}_{IJ}^{CD}\rangle = \delta_{AC}\delta_{BD} - \delta_{AD}\delta_{BC}$ (antisym. overlap).

Expanding the eigenvalues in terms of Fock matrix elements:
$$
\epsilon_A + \epsilon_B = \underbrace{f_{AA}}_{\text{virtual A}} + \underbrace{f_{BB}}_{\text{virtual B}}
$$
For canonical orbitals, the <g>occupied eigenvalues $\epsilon_I, \epsilon_J$</g> are fixed. The <r>virtual Fock elements $f_{AC}$</r> are general (not diagonal):
$$
\begin{aligned}
\langle\Phi_1|H_0|\Phi_1\rangle &= \sum_{I>J}\sum_{A>B}\sum_{C>D} t_{IJ}^{AB} t_{IJ}^{CD} \big[\underbrace{\color{red}{(f_{AC}\delta_{BD} + f_{BD}\delta_{AC} - f_{AD}\delta_{BC} - f_{BC}\delta_{AD})}}_{\text{virtual Fock terms}} \\
&\quad\quad \underbrace{\color{green}{- (\epsilon_I + \epsilon_J)(\delta_{AC}\delta_{BD} - \delta_{AD}\delta_{BC})}}_{\text{occupied energy terms}}\big]
\end{aligned}
$$

<!---
> *Note:* We work in canonical orbitals (diagonal $H_0$), but keep <r>virtual Fock matrix $f_{AB}$ general</r> until the canonical Fock simplification at the end.

...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

<br>

3) Evaluate the second term.

Two‑electron operator, and Excited determinant in operator form:
$$
  {V=\tfrac{1}{4}\sum_{pqrs}\langle pq\|rs\rangle\,a_p^\dagger a_q^\dagger a_s a_r},\quad
  \ket{{}_{ij}^{ab}} = a_a^\dagger a_b^\dagger a_j a_i\,\ket{\Phi_0}.
$$
Matrix element:
$$
\begin{aligned}
  \langle{}_{ij}^{ab}|V|\Phi_0\rangle &= \tfrac{1}{4}\sum_{pqrs}\langle pq\|rs\rangle\,\langle\Phi_0|a_i^\dagger a_j^\dagger a_b a_a\,a_p^\dagger a_q^\dagger a_s a_r\,|\Phi_0\rangle.
\end{aligned}
$$
By Wick's theorem only fully contracted terms survive:
$$\langle\Phi_1|V|\Phi_0\rangle=\sum_{I>J}\color{blue}{\sum_{A>B} t_{IJ}^{AB}\,\langle AB||IJ\rangle}
$$ 

<br>
<br>
<br>
<br>

> *Note:* By Wick's theorem, the only nonzero contractions pair the creation operators $(p,q)$ with the virtual indices $(a,b)$ and the annihilation operators $(r,s)$ with the occupied indices $(i,j)$, yielding the <b>antisymmetrized two-electron integral</b>.

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

<br>

#### ✅ $J_2$ Result Summary

**Pairwise second-order functional:**
$$
\boxed{
\begin{aligned}
  J_{IJ}^{(2)} &= \sum_{A>B} \sum_{C>D} t_{IJ}^{AB} t_{IJ}^{CD} \big[\underbrace{\color{red}{(f_{AC} \,\delta_{BD} - f_{AD}\, \delta_{BC}) + (f_{BD}\,\delta_{AC} - f_{BC}\,\delta_{AD})}}_{\text{virtual Fock contributions}} \\
  &\quad \underbrace{\color{green}{- (\epsilon_{I} + \epsilon_{J})(\delta_{AC}\delta_{BD} - \delta_{AD}\delta_{BC})}}_{\text{occupied energy}}\big] + \underbrace{\color{blue}{2 \sum_{A>B} t_{IJ}^{AB} \langle AB||IJ\rangle}}_{\text{two-electron correlation}}
\end{aligned}
}
$$

**Key features:**
- Quadratic in amplitudes $t_{IJ}^{AB}$ with <r>Fock matrix elements $f_{AC}$</r>
- <g>Occupied eigenvalues $\epsilon_I, \epsilon_J$</g> from canonical orbitals
- Linear in <b>two-electron integrals $\langle AB||IJ\rangle$</b>
- Variational: stationary with respect to $t$ at optimal amplitudes

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

## 2️⃣ Deriving the Gradient $G_{EA}$

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

### Gradient, first‑order derivative of Hylleraas

**Goal:** Obtain the orbital‑rotation gradient:
$$
G_{EA}=\frac{\partial J_2}{\partial R_{EA}} = \underbrace{\color{blue}{2\sum_{I>J}\sum_{B} t_{IJ}^{AB}\,\langle IJ||EB\rangle}}_{\text{two-electron response}} + \underbrace{\color{red}{2\sum_{B} D_{AB}\,f_{EB}}}_{\text{Fock response}}
$$
for rotation parameter $R_{EA}$ (mixing orbital $A$ with orbital $E$).

**Key simplification:** Amplitudes $t_{IJ}^{AB}$ are stationary ($\partial J_2/\partial t = 0$), so only explicit orbital dependence contributes.

The gradient splits into:
- <b>Two-electron contribution</b> $G_{EA}^{(V)}$ from varying $\langle AB||IJ\rangle$
- <r>Fock contribution</r> $G_{EA}^{(f)}$ from varying $f_{AC}$

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

**Step 1: Vary $J_2$ with respect to orbital rotations**

Start from $J_2 = \langle\Phi_1|H_0|\Phi_1\rangle + 2\langle\Phi_1|V|\Phi_0\rangle$ and take the first variation:
$$
\delta J_2 = \langle\Phi_1|\delta H_0|\Phi_1\rangle + 2\langle\Phi_1|\delta V|\Phi_0\rangle
$$


What varies under rotation $R_{EA}$:
- <r>Fock elements</r>: $\delta f_{AC} = \sum_E R_{EA} f_{EC}$ (virtual orbitals rotate)
- <g>Occupied eigenvalues</g>: $\delta\epsilon_I = 0$ (frozen, canonical)
- <b>Two-electron integrals</b>: $\delta\langle AB||IJ\rangle = \sum_P R_{PA}\langle PB||IJ\rangle + \cdots$

**Result:** $G_{EA} = G_{EA}^{(f)} + G_{EA}^{(V)}$ (Fock + two-electron contributions)

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

**Step 2a: Vary the Fock terms**

The $H_0$ contribution has four <r>Fock pieces</r> (from expanding $\langle\Phi_1|H_0|\Phi_1\rangle$):
$$
\sum_{I>J}\sum_{A>B}\sum_{C>D} t_{IJ}^{AB}t_{IJ}^{CD}\big[\color{red}{f_{AC}\delta_{BD} + f_{BD}\delta_{AC} - f_{AD}\delta_{BC} - f_{BC}\delta_{AD}}\big]
$$

Example: Vary the first piece $\color{red}{f_{AC}\delta_{BD}}$:
$$
\begin{aligned}
\delta f_{AC}|_{R_{EA}} &= \underbrace{R_{EA}\color{red}{f_{EC}}}_{\text{rot. couples } E \leftrightarrow A} 
\Rightarrow \quad \delta J_2|_{f_{AC}} = 2R_{EA}\sum_{I>J}\sum_{B,C} t_{IJ}^{AB}t_{IJ}^{CB}\color{red}{f_{EC}}
\end{aligned}
$$

Relabel $C\to B$ in last sum: $\delta J_2|_{f_{AC}} = 2R_{EA}\sum_{I>J}\sum_{B} \underbrace{D_{AB}}_{\text{pair density}}\color{red}{f_{EB}}$

The other three pieces (<r>$f_{BD}$, $-f_{AD}$, $-f_{BC}$</r>) contribute identically by symmetry.
<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

Define the <r>pair-density metric</r> (collects all four Fock pieces):
$$
\boxed{\color{red}{D_{AB} \equiv \sum_{I>J}\sum_{C} t_{IJ}^{AC}t_{IJ}^{BC}}}
$$

Combining all four <r>Fock variations</r> gives the total <r>Fock contribution</r> to gradient:
$$
\boxed{\color{red}{G_{EA}^{(f)} = 2\sum_{B} D_{AB}\,f_{EB}}}
$$

**Physical meaning:** $D_{AB}$ measures the pair-correlation density between virtuals $A$ and $B$ across all occupied pairs $(I,J)$. It weights how Fock rotations $f_{EB}$ contribute to the energy gradient.

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

**Step 2b: Vary the two-electron terms**

The two-electron contribution $2\langle\Phi_1|V|\Phi_0\rangle = 2\sum_{I>J}\sum_{A>B} t_{IJ}^{AB}\color{blue}{\langle AB||IJ\rangle}$ varies as:
$$
\color{blue}{\delta\langle AB||IJ\rangle} = \sum_P R_{PA}\langle PB||IJ\rangle + \sum_P R_{PB}\langle AP||IJ\rangle + \cdots
$$

For rotation $R_{EA}$ (mixing $A\leftrightarrow E$), extract coefficient of $R_{EA}$ from:
$$
2\sum_{I>J}\sum_{A>B} t_{IJ}^{AB}\color{blue}{\delta\langle AB||IJ\rangle}
$$

Collecting terms with $P=E$ on index $A$ yields:
$$
\boxed{\color{blue}{G_{EA}^{(V)} = 2\sum_{I>J}\sum_{B} t_{IJ}^{AB}\,\langle IJ||EB\rangle}}
$$

**Total gradient:** $G_{EA} = \color{red}{G_{EA}^{(f)}} + \color{blue}{G_{EA}^{(V)}}$

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

### ✅ Gradient Result Summary

**Orbital rotation gradient:**
$$
\boxed{
G_{EA} = \underbrace{\color{blue}{2\sum_{I>J}\sum_{B} t_{IJ}^{AB}\,\langle IJ||EB\rangle}}_{\text{two-electron contribution}} + \underbrace{\color{red}{2\sum_{B} D_{AB}\,f_{EB}}}_{\text{Fock contribution}}
}
$$

**Physical interpretation:**
- <b>$G_{EA}^{(V)}$</b>: Response of correlation energy to mixing orbitals $A$ and $E$ via two-electron integrals
- <r>$G_{EA}^{(f)}$</r>: Response via one-electron Fock operator, weighted by pair-density metric $D_{AB}$
- At stationary point: $G_{EA} = 0$ for all virtual–virtual pairs $(E,A)$

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

## 3️⃣ Deriving the Hessian $H_{EA,FB}$

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

### Deriving the Hessian $H_{EA,FB}$

**Goal:** Obtain the second derivative of $J_2$ with respect to two orbital rotations:
$$
H_{EA,FB} = \frac{\partial^2 J_2}{\partial R_{EA} \partial R_{FB}}
$$

**Strategy:** Take two variations of $J_2$ and collect the $R_{EA}R_{FB}$ coefficient.

**Result structure:**
$$
H_{EA,FB} = \color{blue}{\underbrace{H_{EA,FB}^{(V)}}_{\text{two-electron}}} \color{black} + \color{red} \underbrace{H_{EA,FB}^{(f)}}_{\text{Fock}}
$$

- <b>$H^{(V)}$</b>: Two-electron contribution (from varying $\langle AB||IJ\rangle$ twice)
- <r>$H^{(f)}$</r>: Fock contribution (from varying $f_{AC}$ twice)

Both split into **off-diagonal** ($E\neq F$) and **diagonal** ($E=F$) terms.

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

**Step 1a: Two-electron off-diagonal block** ($E \neq F$)

Start from linear two-electron term:
$$
2\sum_{I>J}\sum_{A>B} t_{IJ}^{AB}\color{blue}{\langle AB||IJ\rangle}
$$

Take second variation (both rotations on bra indices):
$$
\color{blue}{\delta^2\langle AB||IJ\rangle} = \sum R_{PA}R_{QB}\langle PQ||IJ\rangle  + \cdots
$$

Extract $R_{EA}R_{FB}$ coefficient with $P=E$, $Q=F$ (both on virtual indices):
$$
2\sum_{I>J}\sum_{A>B} t_{IJ}^{AB} R_{EA}R_{FB} \color{blue}{\langle EF||IJ\rangle}
$$

**Off-diagonal Hessian block:**
$$
\boxed{\color{blue}{H_{EA,FB}^{(V,\text{off})} = 2\sum_{I>J} t_{IJ}^{AB}\langle IJ||EF\rangle}} \quad (E\neq F)
$$

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

**Step 1b: Two-electron diagonal terms** ($E = F$)

When $E=F$, additional contributions arise from "mixed" placements $(P=E, Q=C)$ and $(P=C, Q=E)$:
$$
2\sum_{I>J}\sum_{A>B}\sum_{C} t_{IJ}^{AB} \big(R_{EA}R_{CB}\langle EC||IJ\rangle + R_{CA}R_{EB}\langle CE||IJ\rangle\big)
$$

For rotation $R_{EA} = R_{FA}$ (since $E=F$), relabel indices to obtain:
$$
2\sum_{I>J}\sum_{C} \big[t_{IJ}^{AC}\langle BC||IJ\rangle + t_{IJ}^{CB}\langle CA||IJ\rangle\big] R_{EA}^2
$$

Using antisymmetry $\underbrace{\langle EC||IJ\rangle = -\langle CE||IJ\rangle}_{\text{antisymmetric integral}} = \langle IJ||CE\rangle$:
$$
\boxed{\color{green}{H_{EA,FB}^{(V,\text{diag})} = -\sum_{I>J}\sum_{C} \big[t_{IJ}^{AC}\langle IJ||BC\rangle + t_{IJ}^{CB}\langle IJ||CA\rangle\big]}} \quad (E=F)
$$

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

**Step 2: Fock contribution** (second variation of $f_{AC}$)

Under a unitary rotation $U = e^R$ ($R$ antisymmetric), the Fock matrix transforms as $f' = U^T f U$. The BCH expansion gives:
$$
f' = f + [f, R] + \tfrac{1}{2}[R,[R,f]] + \cdots
$$

The second-order variation is the **double commutator** (not a naive product):
$$
\boxed{\delta^{(2)} f_{AC} = \tfrac{1}{2}[R,[R,f]]_{AC} = \tfrac{1}{2}(R^2 f + f R^2 - 2RfR)_{AC}}
$$

Extracting the $R_{EA}R_{FB}$ coefficient from all four Fock pieces:
$$
\begin{aligned}
E \neq F: &\quad \color{purple}\underbrace{D_{AB}\,f_{EF}}_{\text{off-diagonal Fock}} \quad \color{black}\text{(same as the naive result)}\\

E = F: &\quad \color{green}\underbrace{D_{AB}\,(f_{AA}+f_{BB})}_{\text{energy sum}} \quad \color{black}\text{(the commutator changes the sign vs.\ naive)}
\end{aligned}
$$

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

Combining all four Fock pieces ($f_{AC}$, $f_{BD}$, $-f_{AD}$, $-f_{BC}$) and collecting $R_{EA}R_{FB}$ terms:
$$
\boxed{H_{EA,FB}^{(f)} = \color{green}{D_{AB}(f_{AA}+f_{BB})\delta_{EF}} + \color{purple}{D_{AB}f_{EF}(1-\delta_{EF})}}
$$

where <g>first term</g> is diagonal, <pur>second</pur> is off-diagonal Fock.

**Complete Hessian** (combining all contributions):
$$
\begin{aligned}
H_{EA,FB} &= \underbrace{\color{blue}{2\sum_{I>J} t_{IJ}^{AB}\langle IJ||EF\rangle}}_{\text{two-electron, off-diag}} + \underbrace{\color{green}{-\sum_{I>J}\sum_{C} \big[t_{IJ}^{AC}\langle IJ||BC\rangle + t_{IJ}^{CB}\langle IJ||CA\rangle\big]\delta_{EF}}}_{\text{two-electron, diagonal}} \\
&\quad + \underbrace{\color{green}{D_{AB}(f_{AA}+f_{BB})\delta_{EF}}}_{\text{Fock, diagonal}} + \underbrace{\color{purple}{D_{AB}f_{EF}(1-\delta_{EF})}}_{ \text{Fock, off-diag}}
\end{aligned}
$$

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->


**Structure:**
- When $E\neq F$: <b>two-electron block</b> + <pur>off-diagonal Fock</pur>
- When $E=F$: add <g>diagonal corrections</g> (two-electron + Fock energy sum)
- Matrix is symmetric: $H_{EA,FB} = H_{FB,EA}$

**Physical interpretation:**
- <b>Blue term</b>: direct two-electron response to orbital rotations
- <g>Green terms</g>: diagonal contributions from energy denominators
- <pur>Purple term</pur>: off-diagonal Fock coupling between different rotation pairs

<!---
...

--->

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

### ✅ Hessian Result Summary

**Full orbital rotation Hessian:**
$$
\boxed{
\begin{aligned}
H_{EA,FB} &= \color{blue}{2\sum_{I>J} t_{IJ}^{AB}\,\langle IJ||EF\rangle} \color{green}{- \sum_{I>J}\sum_{C} \Big[ t_{IJ}^{AC}\,\langle IJ||BC\rangle + t_{IJ}^{CB}\,\langle IJ||CA\rangle \Big]\,\delta_{EF}} \\
&\quad + \color{green}{D_{AB}(f_{AA}+f_{BB})\,\delta_{EF}} + \color{purple}{D_{AB}f_{EF}(1-\delta_{EF})}
\end{aligned}
}
$$

**Matrix structure:**

- Block-diagonal in $(E,A)$ pairs when $E \neq F$
- Additional diagonal contributions when $E = F$
- Symmetric: $H_{EA,FB} = H_{FB,EA}$
- Used in Newton–Raphson: determines curvature of $J_2$ landscape

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

## 4️⃣ Newton-Raphson Update

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

### Newton-Raphson orbital optimization

**Goal:** Find optimal orbital rotations to minimize $J_2$ (maximize MP2 correlation energy).

**Strategy:** Use second-order Newton-Raphson method for fast convergence.

**The equation:**
$$\mathbf{R} = -\mathbf{H}^{-1}\mathbf{G}$$

> *Note:* At convergence, $\mathbf{G} \to 0$ so $\mathbf{R} \to 0$ (no further orbital rotation needed).

---

<!-- _header:  OVOS Procedure - Optimization Loop: Newton-Raphson -->

### Constructing the rotation matrix

The rotation parameters $R_{EA}$ form a **skew-symmetric matrix** $\mathbf{R}$:
$$
\mathbf{R} = \begin{pmatrix}
0 & \cdots & -R_{E_1 A_1} & -R_{E_2 A_1} & \cdots \\
\vdots & \ddots & \vdots & \vdots & \\
R_{E_1 A_1} & \cdots & 0 & -R_{E_2 A_2} & \cdots \\
R_{E_2 A_1} & \cdots & R_{E_2 A_2} & 0 & \cdots \\
\vdots & & \vdots & \vdots & \ddots
\end{pmatrix}
$$

**Key properties:**
- Antisymmetry: $R_{AE} = -R_{EA}$ (ensures unitary transformation)
- Unitary rotation: ${\mathbf{U} = \exp(\mathbf{R})}$ (preserves orthonormality)
- New orbitals: $|\phi_i^{\text{new}}\rangle = \sum_j U_{ij} |\phi_j^{\text{old}}\rangle$

<!---
...

--->

---

<!-- _header:  Theoretical Note -->

## Why $(f_{AA}+f_{BB})$, Not $(f_{AA}-f_{BB})$

### Symmetry Verification

The Hessian must be symmetric: $H_{EA,FB} = H_{FB,EA}$. For the **diagonal block** ($E = F$), this means $A \leftrightarrow B$:

- <g>Two-electron diagonal</g>: Relabel dummy index $C$ $\to$ symmetric ✓
- <g>Fock diagonal with $(f_{AA}+f_{BB})$</g>: $D_{AB}(f_{AA}+f_{BB}) \to D_{BA}(f_{BB}+f_{AA})$ = same ✓
- <r>Fock diagonal with $(f_{AA}-f_{BB})$</r>: $D_{AB}(f_{AA}-f_{BB}) \to -D_{AB}(f_{AA}-f_{BB})$ = **antisymmetric** ✗

### Why naive derivations give the wrong sign

A naive formula $\delta^2 f_{AC} = \sum_{P,Q} R_{PA}R_{QC} f_{PQ}$ treats the rotation as $(I+R)^T f (I+R)$, dropping the $\frac{1}{2}R^2$ terms from $e^R$. The correct unitary formula $f' = e^{-R} f\, e^{R}$ gives $\delta^{(2)}f = \frac{1}{2}[R,[R,f]]$, which adds the missing $\frac{1}{2}(R^2 f + fR^2)$ contributions — these flip the sign from $-$ to $+$.

> *Note:* The two-electron integrals are unaffected because the analogous extra terms produce $\langle PP||IJ\rangle = 0$ by antisymmetry.
---

<!-- _header:  Appendix -->

## Appendix: Deriving the Hylleraas Functional from Møller-Plesset Theory

---

<!-- _header:  Appendix: Møller-Plesset Theory -->

### Møller-Plesset Perturbation Theory

**Goal:** Derive the Hylleraas variational functional for MP2 starting from canonical orbitals.

**Setup:** Partition the electronic Hamiltonian:
$$
H = H_0 + \lambda V
$$
where:
- $H_0 = \sum_i f_i$ is the **Fock operator** (zeroth-order Hamiltonian)
- $V = H - H_0$ is the **fluctuation potential** (perturbation)
- $\lambda$ is a perturbation parameter (set to 1 at the end)

**Key assumption:** Work in **canonical orbitals** where $H_0$ is diagonal:
$$
H_0 |\Phi_0\rangle = E_0 |\Phi_0\rangle, \quad H_0 |\Phi_i^a\rangle = (E_0 + \epsilon_a - \epsilon_i) |\Phi_i^a\rangle
$$

---

<!-- _header:  Appendix: Møller-Plesset Theory -->

### Energy and Wavefunction Expansions

Expand the exact energy and wavefunction in powers of $\lambda$:
$$
\begin{aligned}
E &= E_0 + \lambda E_1 + \lambda^2 E_2 + \lambda^3 E_3 + \cdots \\
|\Psi\rangle &= |\Phi_0\rangle + \lambda |\Phi_1\rangle + \lambda^2 |\Phi_2\rangle + \lambda^3 |\Phi_3\rangle + \cdots
\end{aligned}
$$

**Schrödinger equation:**
$$
(H_0 + \lambda V)(|\Phi_0\rangle + \lambda |\Phi_1\rangle + \cdots) = (E_0 + \lambda E_1 + \lambda^2 E_2 + \cdots)(|\Phi_0\rangle + \lambda |\Phi_1\rangle + \cdots)
$$

**Collect terms by order in $\lambda$:**
- $\lambda^0$: $H_0|\Phi_0\rangle = E_0|\Phi_0\rangle$ ✓ (HF equation)
- $\lambda^1$: $H_0|\Phi_1\rangle + V|\Phi_0\rangle = E_0|\Phi_1\rangle + E_1|\Phi_0\rangle$
- $\lambda^2$: $H_0|\Phi_2\rangle + V|\Phi_1\rangle = E_0|\Phi_2\rangle + E_1|\Phi_1\rangle + E_2|\Phi_0\rangle$

---

<!-- _header:  Appendix: Møller-Plesset Theory -->

### First-Order Correction

From the $\lambda^1$ equation, project onto $\langle\Phi_0|$:
$$
\langle\Phi_0|H_0|\Phi_1\rangle + \langle\Phi_0|V|\Phi_0\rangle = E_0\langle\Phi_0|\Phi_1\rangle + E_1\langle\Phi_0|\Phi_0\rangle
$$

Using $\langle\Phi_0|\Phi_1\rangle = 0$ (intermediate normalization) and $\langle\Phi_0|H_0 = E_0\langle\Phi_0|$:
$$
\boxed{E_1 = \langle\Phi_0|V|\Phi_0\rangle}
$$

**Brillouin's theorem:** For canonical HF orbitals, single excitations don't contribute:
$$
\langle\Phi_i^a|V|\Phi_0\rangle = 0
$$
So $|\Phi_1\rangle$ contains only **doubles** (and higher):
$$
|\Phi_1\rangle = \sum_{i<j}\sum_{a<b} t_{ij}^{ab} |\Phi_{ij}^{ab}\rangle + \cdots
$$

---

<!-- _header: Appendix: Møller-Plesset Theory -->

### Second-Order Correction

From the $\lambda^2$ equation, project onto $\langle\Phi_0|$:
$$
\langle\Phi_0|V|\Phi_1\rangle = E_2
$$

This gives the **MP2 energy correction**:
$$
\boxed{E_2 = \langle\Phi_0|V|\Phi_1\rangle = \sum_{i<j}\sum_{a<b} t_{ij}^{ab} \langle\Phi_0|V|\Phi_{ij}^{ab}\rangle}
$$

The amplitudes $t_{ij}^{ab}$ are determined by projecting the $\lambda^1$ equation onto excited determinants:
$$
\langle\Phi_{ij}^{ab}|(H_0 - E_0)|\Phi_1\rangle + \langle\Phi_{ij}^{ab}|V|\Phi_0\rangle = 0
$$

This yields:
$$
t_{ij}^{ab} = \frac{\langle\Phi_{ij}^{ab}|V|\Phi_0\rangle}{E_0 - (E_0 + \epsilon_a + \epsilon_b - \epsilon_i - \epsilon_j)} = \frac{\langle ab||ij\rangle}{\epsilon_i + \epsilon_j - \epsilon_a - \epsilon_b}
$$


---

<!-- _header: Appendix: Hylleraas Functional -->


### The Hylleraas Variational Principle

**Problem:** Standard MP2 amplitudes are **non-variational** 

**Solution:** Hylleraas (1930) introduced a variational functional for perturbation theory:
$$
\boxed{J_2 = \langle\Phi_1|H_0 - E_0|\Phi_1\rangle + 2\langle\Phi_1|V|\Phi_0\rangle}
$$

**Key properties:**
1. **Stationary condition** $\frac{\partial J_2}{\partial t_{ij}^{ab}} = 0$ recovers the exact MP2 amplitude equations
2. **At stationary point:** $J_2 = E_2$ (equals the MP2 energy correction)
3. **Variational:** Can optimize both amplitudes $t_{ij}^{ab}$ AND orbitals simultaneously

**Why multiply by 2?** The factor of 2 in the second term ensures:
$$
{\partial J_2}/{\partial t_{ij}^{ab}}\big|_{\text{stationary}} = 0 \quad \Rightarrow \quad 2\langle\Phi_{ij}^{ab}|(H_0 - E_0)|\Phi_1\rangle + 2\langle\Phi_{ij}^{ab}|V|\Phi_0\rangle = 0
$$
which is the correct amplitude equation (with factor of 2 canceling).

---

<!-- _header:  Appendix: Connection to OVOS -->

### Connection to OVOS

**OVOS extends MP2 by optimizing orbitals:**

Starting from Hylleraas functional $J_2$, OVOS:
1. **Optimizes amplitudes** $t_{ij}^{ab}$ via stationary condition (standard MP2)
2. **Optimizes virtual orbitals** via gradient $G_{EA}$ and Hessian $H_{EA,FB}$
3. Uses **Newton-Raphson** to find optimal orbital rotations

**Result:** Lower correlation energy than standard MP2 with same computational scaling!

**Connection to MP theory:** The Hylleraas functional provides a variational framework that extends MP2 from fixed orbitals to orbital-optimized correlation methods!
