# Optimization Options:
## Option A: Cache second_order_density_matrix_element() Results
- [V] Pre-compute all D_AB values before building Hessian
Store in a 2D array instead of recalculating 929k times
- [V] Code change: Build dictionary D_AB_cache once, reuse in hessian()

## Option B: scipy.linalg.solve, self.orbspin, sparse R_matrix
- [V] orbspin array creation (line 352) happens every iteration, could be a class attribute from __init__
- [X] Line 330: R = -1.0*G@np.linalg.inv(H) → R = -1.0*scipy.linalg.solve(H, G)
- [X] sparse R_matrix -> Sparsity of R_matrix (|R_ij| < 1e-06): 83.33%

## Option C: np.eisum, symmetry, cache, precompute, numba
- [X] np.eisum implemented for J_2.
- [V] Hessian symmetry, compute upper triangle and mirror
- [X] Cache ERI slices, In gradient() and hessian(), you repeatedly access the same ERI slices. Pre-extract relevant slices. -> ran only two iterations, found a value instantly... (was lower, -0.0016664714266755923)
- [V] Precompute Index Pairs, lines check if I > J: and if A > B: millions of times. Precompute valid pairs once
- [?] Numba JIT, the innermost loops in MP2_energy are pure NumPy arithmetic