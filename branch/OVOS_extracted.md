# OVOS Method - Implementation Notes

## Reference
**Optimized virtual orbital space for high-level correlated calculations**  
Adamowicz, L. & Bartlett, R. J.  
*J. Chem. Phys.* **86**, 6314-6324 (1987)  
DOI: [10.1063/1.452468](https://doi.org/10.1063/1.452468)

---

## Core Concept

Reduce the virtual orbital space dimension from N_VIRT to N'_VIRT < N_VIRT while preserving most correlation energy. Uses second-order Hylleraas functional to optimize orbital rotations between active and inactive virtual spaces.

**Key Result**: Can recover ~90% of correlation energy with 50% of virtual orbitals, or ~100% when combining OVOS result with exact E₂.

## Method Summary
Optimize a reduced virtual orbital space by rotating active virtuals against inactive virtuals to minimize second-order correlation energy via the Hylleraas functional.


---

## Mathematical Framework

### Second-Order Hylleraas Functional (Equation 2)
Upper bound to E₂:
```
J₂ = ⟨Φ₁|H₀ - E₀|Φ₁⟩ + 2⟨Φ₁|V - E₁|Φ₀⟩
```
- Φ₀: HF reference (intermediately normalized)
- Φ₁: First-order wave function (doubles only)

### Orbital Space Partition
- **Occupied**: i, j indices
- **Active Virtual**: a, b indices (being optimized)
- **Inactive Virtual**: e, f indices (rotate with active)

### Gradient (Equation 12a)
```
G_ea = 2∑_ijb t_ijb^(1) ⟨ij|eb⟩ + 2∑_b D_ab^(2) F_eb
```

### Hessian (Equation 12b)
```
H_ea,fb = 2∑_ij t_ijb^(1) ⟨ij|ef⟩
         - ∑_ijc [t_ijc^(1)⟨ij|bc⟩ + t_icb^(1)⟨ij|ca⟩] δ_ef
         + D_ab^(2) (f_aa - f_bb) δ_ef
         + D_ab^(2) f_ef (1 - δ_ef)
```

Where:
- `t_ijab^(1)` = MP1 amplitudes: `-(⟨ab|ij⟩ - ⟨ab|ji⟩)/(ε_a + ε_b - ε_i - ε_j)`
- `D_ab^(2)` = Second-order density: `∑_ijc t_aic^(1) t_bjc^(1)`

### Newton-Raphson Step (Equation 14)
```
R = -G · H⁻¹
```

### Unitary Transformation (Equation 15)
```
U = exp(R) = X cosh(d) X^T + RX sinh(d) d⁻¹ X^T
```
Where `d² = X^T R² X` (diagonalization of R²)

---

## Implementation Algorithm

1. **SCF Calculation**: Obtain HF orbitals and energy
2. **Integral Transformation**: Compute (ij|ab) integrals in MO basis
3. **Active Space Selection**: Choose N'_VIRT based on orbital contributions to E₂
4. **MP1 Amplitudes**: Calculate t_ijab^(1) for active space
5. **Build G and H**: Construct gradient and Hessian matrices
6. **Solve NR Equation**: R = -G · H⁻¹ (use block structure of H)
7. **Generate U**: Unitary transformation via matrix exponential
8. **Canonicalize**: Diagonalize Fock matrix in rotated active space
9. **Check Convergence**: If E₂ converged, stop; else iterate from step 4

---

## Key Properties

### Invariance
- MBPT/CC methods are invariant to occupied-occupied and virtual-virtual rotations
- OVOS exploits this to rotate virtual space without changing reference

### Pair Separability
Both G and H exhibit electron-pair separability like J₂ itself - enables efficient evaluation.

### Computational Scaling
- Only needs (occ occ|virt virt) integrals
- Much simpler than MCSCF variational optimization
- Main cost: solving NR equation (use block diagonal structure)

---

## Practical Notes

- **Canonical Form**: Active orbitals diagonalize Fock matrix at each iteration
- **Exact E₂**: Calculate full-space E₂ as byproduct, add to OVOS result for ~100% recovery
- **Convergence**: E₂ typically converges in few iterations
- **Block Structure**: Hessian dominated by diagonal blocks H_ea,eb (all active a with one inactive e)

---

## Relation to Your Implementation (Update as implementation continue)

Your `ovos.py` implements:
- `MP2_energy()`: Computes J₂, t_ijab^(1), ERIs, Fock matrix
- `orbital_optimization()`: Builds G (gradient) and H (Hessian), solves NR equation, applies U rotation (with scipy.linalg.expm())
- `run_ovos()`: Iteration loop until E₂ convergence

Active/inactive indexing in your code:
- `active_occ_indices` → i, j
- `active_inocc_indices` → a, b  
- `inactive_indices` → e, f
