import re

with open('ovos_start_over.py', 'r') as f:
    content = f.read()

old_func = '''def spatial_2_spin_eri_optimized(self, eri_aaaa, eri_aabb, eri_bbbb):
"""
Optimized conversion of spatial ERIs to spin-orbital ERIs.
"""
n_spatial = eri_aaaa.shape[0]
n_spin = 2 * n_spatial

eri_spin = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.float64)

# Fill in the blocks directly
# (αα|αα)
eri_spin[0::2, 0::2, 0::2, 0::2] = eri_aaaa
# (ββ|ββ)
eri_spin[1::2, 1::2, 1::2, 1::2] = eri_bbbb
# (αα|ββ)
eri_spin[0::2, 0::2, 1::2, 1::2] = eri_aabb
# (ββ|αα)
eri_spin[1::2, 1::2, 0::2, 0::2] = eri_aabb.transpose(2,3,0,1)

return eri_spin'''

new_func = '''def spatial_2_spin_eri_optimized(self, eri_aaaa, eri_aabb, eri_bbbb):
"""
Optimized conversion of spatial ERIs to spin-orbital ERIs.

Handles all 6 non-zero spin combinations:
1. (αα|αα) -> eri_aaaa
2. (ββ|ββ) -> eri_bbbb
3. (αα|ββ) -> eri_aabb
4. (ββ|αα) -> eri_aabb transposed
5. (αβ|αβ) -> eri_aabb with special indexing
6. (βα|βα) -> eri_aabb with special indexing
"""
n_spatial = eri_aaaa.shape[0]
n_spin = 2 * n_spatial

eri_spin = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.float64)

# Fill in the blocks directly
# (αα|αα)
eri_spin[0::2, 0::2, 0::2, 0::2] = eri_aaaa
# (ββ|ββ)
eri_spin[1::2, 1::2, 1::2, 1::2] = eri_bbbb
# (αα|ββ)
eri_spin[0::2, 0::2, 1::2, 1::2] = eri_aabb
# (ββ|αα)
eri_spin[1::2, 1::2, 0::2, 0::2] = eri_aabb.transpose(2,3,0,1)

# Cross-spin blocks - need explicit loops due to index mapping
# (αβ|αβ): eri_spin[2p, 2q+1, 2r, 2s+1] = eri_aabb[p, r, q, s]
# (βα|βα): eri_spin[2p+1, 2q, 2r+1, 2s] = eri_aabb[q, s, p, r]
for p in range(n_spatial):
for q in range(n_spatial):
for r in range(n_spatial):
for s in range(n_spatial):
# (αβ|αβ)
eri_spin[2*p, 2*q+1, 2*r, 2*s+1] = eri_aabb[p, r, q, s]
# (βα|βα)
eri_spin[2*p+1, 2*q, 2*r+1, 2*s] = eri_aabb[q, s, p, r]

return eri_spin'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open('ovos_start_over.py', 'w') as f:
        f.write(content)
    print("Successfully patched spatial_2_spin_eri_optimized!")
else:
    print("Could not find the function to patch!")
    # Let's try to find what's there
    import re
    match = re.search(r'def spatial_2_spin_eri_optimized.*?return eri_spin', content, re.DOTALL)
    if match:
        print("Found function at position:", match.start())
        print("First 500 chars:")
        print(repr(match.group(0)[:500]))
