# AI Coding Agent Instructions - OVOS Project (Enhanced)

Always show code corrections in chat only. Do not attempt to edit files directly.

## Academic Guidelines
Using generative AI for "limited extent" per UCPH guidelines: idea generation, structure suggestions, literature search, clarifications. Follow good scientific practice - document AI usage, verify all outputs, understand the physics and chemistry.

---

## Project Structure & Code Organization

### Main Files
- **ovos_start_over.py**: Primary OVOS implementation with complete OVOS class
- **ovos.py**: Alternative version (may have different optimization approaches)
- **test_OVOS.py**: Testing and validation module
- **branch/data/**: Stores computed results (JSON format with MP2 energies for different virtual orbital counts)
- **profil/**: Performance profiling results (.prof and .txt files)

### Key Classes & Methods
The `OVOS` class contains the entire algorithm with these primary methods:
1. `__init__()`: Initialization with molecule, number of optimized virtual orbitals, and initial orbital choice
2. `MP2_energy()`: Computes MP2 correlation energy and first-order amplitudes
3. `orbital_optimization()`: Performs orbital rotations to minimize J₂ functional
4. `run_ovos()`: Main loop - iterates between MP2 calculation and orbital optimization until convergence

---

## Project Context
Master's thesis on **Optimized Virtual Orbitals (OVO)** for quantum computing at UCPH. Implementation based on [L. Adamowicz & R. J. Bartlett (1987)](https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level) - minimizes second-order correlation energy (MP2) using orbital rotations.

### Reference
**Optimized virtual orbital space for high-level correlated calculations**  
Adamowicz, L. & Bartlett, R. J.  
*J. Chem. Phys.* **86**, 6314-6324 (1987)  
DOI: [10.1063/1.452468](https://doi.org/10.1063/1.452468)

### Core Concept

Reduce the virtual orbital space dimension from N_VIRT to N'_VIRT < N_VIRT while preserving most correlation energy. Uses second-order Hylleraas functional to optimize orbital rotations between active and inactive virtual spaces.

**Key Result**: Can recover ~90% of correlation energy with 50% of virtual orbitals, or ~100% when combining OVOS result with exact E₂.

#### Method Summary
Optimize a reduced virtual orbital space by rotating active virtuals against inactive virtuals to minimize second-order correlation energy via the Hylleraas functional.

---

## Critical Index Spaces (ESSENTIAL)

The code uses a complex orbital indexing system with **two different orderings**:

### Index Space Definitions (NATURAL/STORAGE ordering)
```
- active_occ_indices (I,J):        Occupied spin-orbitals [0, nelec)
- active_inocc_indices (A,B):      Active virtual orbitals [nelec, nelec+num_opt_virtual_orbs)
- inactive_indices (E,F):          Inactive virtual orbitals [nelec+num_opt_virtual_orbs, tot_spin_orbs)
- virtual_inocc_indices:           Same as inactive_indices
- full_indices:                    All spin-orbitals [0, tot_spin_orbs)
```


---

## Unrestricted Formalism Details

### Alpha/Beta Orbital Structure
- All calculations use **Unrestricted Hartree-Fock (UHF)** 
- Separate molecular orbital coefficients for alpha (spin-up) and beta (spin-down) electrons
- MO coefficients stored as tuple: `mo_coeffs = [mo_coeffs_alpha, mo_coeffs_beta]`
- Each has shape: `(n_basis_functions, n_spatial_orbitals)`

### Spin-Orbital Conversions
Three critical helper methods convert between spatial and spin-orbital bases:

1. **`spatial_to_spin_eri(eri_aaaa, eri_aabb, eri_bbbb)`**: Converts spatial 2-electron integrals to spin-orbital basis
   - Optimized version: `spatial_2_spin_eri_optimized()` uses block assignment for efficiency
   
2. **`spatial_to_spin_fock(Fmo_a, Fmo_b)`**: Builds block-diagonal spin-orbital Fock matrix
   - Optimized version: `spatial_to_spin_fock_optimized()` directly assigns alpha/beta blocks
   
3. **`spatial_to_spin_mo_energy(mo_energy_a, mo_energy_b)`**: Converts spatial MO energies to energy-sorted spin-orbital energies
   - Performs energy-based sorting to define the ENERGY ordering described above

---

## Two-Electron Integrals Convention

### Notation
- **PySCF uses Chemists' notation**: `(ij|kl)` in chemists' notation = `<ik|jl>` in physicists' notation
- Integral values are the same; only the index labeling differs

### Spatial Integral Blocks
Three separate spatial blocks are computed:
- `eri_aaaa`: (αα|αα) alpha-alpha integrals, shape (n_spatial, n_spatial, n_spatial, n_spatial)
- `eri_bbbb`: (ββ|ββ) beta-beta integrals, same shape
- `eri_aabb`: (αα|ββ) alpha-alpha, beta-beta mixed integrals, same shape

**Important**: After `pyscf.ao2mo.kernel()`, integrals must be reshaped from flat 1D array to 4D tensor with above shapes.

### Spin-Orbital Form
After conversion, `eri_spin` has shape `(n_spin, n_spin, n_spin, n_spin)` where:
- Even indices (0,2,4,...) correspond to alpha orbitals
- Odd indices (1,3,5,...) correspond to beta orbitals
- Index spacing allows direct block assignment: `eri_spin[0::2, 0::2, 0::2, 0::2] = eri_aaaa`

**Note**: Code has uncertainty whether `<ab|ij>` in formulas uses chemist vs physicist notation or antisymmetrized integrals. Verify during implementation!

---

## Key Algorithm Variables & Conventions

### Orbital Energies & Fock Matrices
- `eps`: Spin-orbital orbital energies, shape (n_spin,)
- `Fmo_spin`: Spin-orbital Fock matrix in MO basis (energy-ordered), shape (n_spin, n_spin)
- `Fmo_a`, `Fmo_b`: Spatial (alpha, beta) Fock matrices, computed as `C^T F_ao C`

### Amplitudes & Density
- `MP1_amplitudes` or `t_amplitudes`: First-order MP amplitudes used to compute MP2 energy
  - Formula: $t_{ij}^{ab} = - \frac{<ab|ij>}{- \epsilon_i - \epsilon_j + \epsilon_a + \epsilon_b}$
- `D_ab`: Virtual-virtual block of MP2 second-order density matrix
  - Formula: $D_{ab} = \sum_{i>j} \sum_{c} t_{ij}^{ac} t_{ij}^{bc}$

### Functional & Gradients
- `J2`: Second-order Hylleraas functional (MP2 correlation energy)
- `G` (gradient): Derivative of J₂ with respect to orbital rotations $\frac{\partial J_2}{\partial R_{ae}}$
- `H` (Hessian): Second derivative matrix $\frac{\partial^2 J_2}{\partial R_{ae} \partial R_{bf}}$
- Optimization uses Newton's method: $\Delta R = H^{-1} G$

---

## Core Algorithm Workflow

### Step-by-Step Execution (from `run_ovos()`)

1. **Initialization** (`__init__`):
   - Perform UHF calculation to get initial molecular orbitals
   - Compute initial MP2 energy
   - Build index lists for active/inactive orbital spaces
   - Store AO integrals and Fock matrix

2. **Main Loop** (iteration until convergence):
   - **Step (i-ii)**: Transform integrals and matrices from spatial to spin-orbital basis
   - **Step (iii-iv)**: Call `MP2_energy()` to compute:
     - MP2 correlation energy `E_corr`
     - MP1 amplitudes `MP1_amplitudes`
     - Spin-orbital integrals `eri_spin`
     - Spin-orbital Fock matrix `Fmo_spin`
   
   - **Step (v-viii)**: Call `orbital_optimization()` to:
     - Compute gradient `G` of J₂ functional
     - Compute Hessian `H` of J₂ functional
     - Solve Newton step: `dR = -H^{-1} G`
     - Apply orbital rotation: `mo_coeffs = transform(mo_coeffs, dR)`
     - Optionally: Use RLE method for improved stability
     - Return optimized `mo_coeffs`
   
   - **Step (ix)**: Check convergence:
     - Convergence criterion: $|E_{corr}^{(n)} - E_{corr}^{(n-1)}| < 10^{-8}$ Hartree (in ovos_start_over.py)
     - Or: $|E_{corr}^{(n)} - E_{corr}^{(n-1)}| < 10^{-10}$ Hartree (in ovos.py)
     - Max iterations: 500-2500 depending on version

3. **Output** (`run_ovos()` returns):
   - `lst_E_corr`: List of correlation energies at each iteration
   - `lst_iter_counts`: List of iteration counts
   - `mo_coeffs`: Final optimized molecular orbital coefficients

---

## Performance & Optimization Details

### Threading & Reproducibility
```python
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
```
- Forces single-threaded execution for reproducibility
- Essential for debugging and comparing runs

### Computational Complexity
- **Spatial-to-spin ERI conversion**: O(n⁴) in orbital count
- **Hessian assembly**: O(n⁶) with multiple nested loops
- Optimized versions (`spatial_2_spin_eri_optimized()`, `spatial_to_spin_fock_optimized()`) use NumPy block assignment for ~2x speedup

### Profiling Resources
Available in `profil/` directory:
- `.prof` files: Binary cProfile results
- `.txt` files: Human-readable profiling summaries
- Can identify bottlenecks for future optimization

---

## Configuration & Initialization Options

### Initial Orbitals (`init_orbs` parameter)
- **`"UHF"`** (default): Use fresh UHF molecular orbitals from current calculation
- **`np.ndarray`**: Provide pre-computed orbital coefficients (e.g., from previous OVOS run)
  - Format must match: `[alpha_coeffs, beta_coeffs]`
  - Enables orbital continuation for incremental virtual space expansion

### Virtual Orbital Space Expansion
```python
num_opt_virtual_orbs_current = 0
increment = 2  # For closed-shell molecules
while num_opt_virtual_orbs_current < max_opt_virtual_orbs:
    # Run OVOS for current space size
    num_opt_virtual_orbs_current += increment
```

### Sampling Strategies
Three mutually exclusive modes (set only one to `True`):

1. **`use_prev_virt_orbs = True`**:
   - Reuse optimized orbitals from previous num_opt_virtual_orbs as starting guess
   - Faster convergence for incremental expansions
   - Saves optimized orbitals after each successful run

2. **`use_random_unitary_init = True`**:
   - For each num_opt_virtual_orbs, try 500 different random unitary rotations
   - Rotations applied to virtual orbital subspace only
   - Selects best result (lowest J₂) across all 500 attempts
   - More thorough exploration, much slower
   - Uses QR decomposition: `Q, R = np.linalg.qr(random_matrix)`

3. **Default** (both `False`):
   - Simple sequential runs with default UHF initialization
   - Fastest but may get stuck in local minima

---

## Supported Molecules & Basis Sets

### Test Molecules (defined in `ovos_start_over.py`)
```python
find_atom = {
    "H2":  0,  # Bond length 0.74144 Å
    "LiH": 1,  # Bond length 1.595 Å
    "H2O": 2,  # Equilibrium geometry
    "CH2": 3,  # Planar geometry
    "BH3": 4,  # Planar geometry
    "N2":  5   # Bond length 1.10 Å
}
```

### Available Basis Sets
- Minimal: STO-3G, STO-6G, 3-21G
- Standard: 6-31G, DZP, roosdz, anoroosdz
- Correlated: cc-pVDZ, cc-pV5Z, def2-QZVPP, aug-cc-pV5Z, ANO

Basis selection affects virtual orbital count and correlation energy magnitude.

---

## Data Storage & Output Format

### JSON Output Files
Results saved to: `branch/data/{molecule}/{basis}/lst_MP2_{name}.json`

**File naming** (selected by initialization options):
- No prefix: Default sequential runs
- `_prev`: Using `use_prev_virt_orbs = True`
- `_random`: Using `use_random_unitary_init = True`
- `_opt_A/B/C`: If `use_RLE_orbopt = True` with different RLE variants

**File contents** (list of tuples):
```python
[
  [num_opt_virtual_orbs_1, E_corr_1, iterations_1],
  [num_opt_virtual_orbs_2, E_corr_2, iterations_2],
  ...
]
```

### Summary Metrics
Each run prints:
- Total spin-orbitals
- Active occupied/unoccupied orbital counts
- Inactive unoccupied orbital counts
- Convergence status and iteration count
- Final MP2 correlation energy
- Difference from full-space MP2

---

## Verification & Debugging Checks

### MO Coefficient Orthonormality (commented out in code)
```python
norm = C_i.T @ self.S @ C_i
assert np.allclose(norm, np.eye(norm.shape[0]), atol=1e-6)
```
Check: $C^T S C = I$ (MO coefficients are S-orthonormal)

### Fock Matrix Properties
- **Hermiticity**: `assert np.allclose(Fmo_spin, Fmo_spin.T.conj(), atol=1e-10)`
- **Finiteness**: `assert np.all(np.isfinite(Fmo_spin))`
- **Non-zero**: `assert np.count_nonzero(Fmo_spin) > 0`

### Index Space Consistency
- Occupied indices: $\{0, 1, ..., n_{elec}-1\}$
- Active virtual: $\{n_{elec}, ..., n_{elec}+n_{opt}^{virt}-1\}$
- Inactive virtual: $\{n_{elec}+n_{opt}^{virt}, ..., n_{spin}^{tot}-1\}$
- No overlaps between spaces

### Orbital Energy Ordering
After `spatial_to_spin_mo_energy()`, verify:
- Energies are finite and properly sorted
- Occupied orbitals have lowest energies
- Virtual orbitals have higher energies
- Multiplicity: 2 orbitals per spatial orbital (alpha + beta)

---

## Open Questions & Clarifications Needed

### From Code Comments
1. **Integral Notation Ambiguity**: `<ab|ij>` in formulas—confirm whether chemist vs physicist notation, or antisymmetrized form
2. **MP1 Amplitude Formula**: Sign convention in $t_{ij}^{ab} = - \frac{<ab|ij>}{- \epsilon_i - \epsilon_j + \epsilon_a + \epsilon_b}$—verify the minus signs
3. **Density Matrix Definition**: Clarify exact form of virtual-virtual density matrix `D_ab`

### For Future Development
- Numerical stability of Hessian inversion (see RLE method experiments)
- Convergence guarantees for Newton-based optimization with restricted active space
- Comparison with other virtual orbital selection methods
- Extension to restricted (RHF) formalism

---

## Mathematical Notation Reference

### Formulas Implemented

**MP1 Amplitudes**:
The article defines MP1 amplitudes as:
$$t_{ij}^{ab} = - \frac{<ab|ij>}{- \epsilon_i - \epsilon_j + \epsilon_a + \epsilon_b}$$
where $i,j$ are occupied orbitals, $a,b$ are virtual orbitals, and $\epsilon$ are orbital energies.
Note: Am not sure if <ab|ij> means chemist's or physicist's notation, or if it is the antisymmetrized integral.

**MP2 Correlation Energy**:
The article gives the MP2 correlation energy as:
$$J_2 = \sum_{i>j} J_{ij}^{(2)}$$
where
$$J_{ij}^{(2)} = \sum_{a>b} \sum_{c>d} t_{ij}^{ab} t_{ij}^{cd} [(f_{ac} \delta_{bd} - f_{ad} \delta_{bc}) + (f_{bd} \delta_{ac} - f_{bc} \delta_{ad})] - (\epsilon_i + \epsilon_j)(\delta_{ac}\delta_{bd} - \delta_{ad}\delta_{bc}) + 2 \sum_{a>b} t_{ij}^{ab} <ab|ij>$$
where $f_{ac}$ are Fock matrix elements in the MO basis, and $\delta$ is the Kronecker delta, and $<ab|ij>$ are two-electron integrals, and $i,j$ are occupied orbitals, $a,b,c,d$ are virtual orbitals, and $\epsilon$ are orbital energies.
Note: Am not sure if <ab|ij> means chemist's or physicist's notation, or if it is the antisymmetrized integral.

**Gradient of J₂ wrt orbital rotations**:
The gradient of J₂ with respect to orbital rotations between active virtual orbitals $a$ and inactive virtual orbitals $e$ is given by:
$$G_{ea} = \frac{\partial J_2}{\partial R_{ae}} = 2 \sum_{i>j} \sum_{b} t_{ij}^{ab} <ij|eb> + 2 \sum_{b} D_{ab} f_{be}$$
where $D_{ab} = \sum_{i>j} \sum_{c} t_{ij}^{ac} t_{ij}^{bc}$ is the virtual-virtual block of the MP2 second-order density matrix, and $f_{be}$ are Fock matrix elements in the MO basis, and $i,j$ are occupied orbitals, $a,b,c$ are virtual orbitals, and $e,f$ are the inactive orbitals, and $\epsilon$ are orbital energies.
Note: Am not sure if <eb|ij> means chemist's or physicist's notation, or if it is the antisymmetrized integral.

**Hessian of J₂ wrt orbital rotations**:
The Hessian of J₂ with respect to orbital rotations between active virtual orbitals $a$ and inactive virtual orbitals $e$ is given by:
$$H_{ea,fb} = \frac{\partial J_2}{\partial R_{ae} \partial R_{bf}} = 2 \sum_{i>j} t_{ij}^{ab} <ij|ef> - \sum_{i>j} \sum_{c} [t_{ij}^{ac} <ij|bc> + t_{ij}^{bc} <ij|ca>]\delta_ef + D_{ab} (f_{aa} - f_{bb}) \delta_{ef} + D_{ab} f_{ef} (1-\delta_{ef})$$
where $D_{ab} = \sum_{i>j} \sum_{c} t_{ij}^{ac} t_{ij}^{bc}$ is the virtual-virtual block of the MP2 second-order density matrix, and $e,f$ are the inactive orbitals, and $f_{ef}$ are Fock matrix elements in the MO basis, and $i,j$ are occupied orbitals, $a,b,c,d$ are virtual orbitals, and $\epsilon$ are orbital energies.
Note: Am not sure if <ef|ij> means chemist's or physicist's notation, or if it is the antisymmetrized integral.
