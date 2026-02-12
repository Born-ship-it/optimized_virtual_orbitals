## Reduction of the Hylleraas J₂ functional to the standard MP2 closed form

### Starting point — your formula:

$J_2 = Σ_{i>j} [ Σ_{a>b} Σ_{c>d} t_{ij}^{ab} t_{ij}^{cd} ( f_{ac} δ_{bd} + f_{bd} δ_{ac} - f_{ad} δ_{bc} - f_{bc} δ_{ad} - (ε_i + ε_j)(δ_{ac} δ_{bd} - δ_{ad} δ_{bc}) ) + 2 Σ_{a>b} t_{ij}^{ab} <ab||ij> ]$

Call the first part "Term 1" and the second part "Term 2".


### Step 1: Simplify the bracket via Kronecker deltas

The bracket multiplying $t_{ij}^{ab} t_{ij}^{cd}$ is:

  $B(ab,cd) = f_{ac} δ_{bd} + f_{bd} δ_{ac} - f_{ad} δ_{bc} - f_{bc} δ_{ad}
            - (ε_i + ε_j)(δ_{ac} δ_{bd} - δ_{ad} δ_{bc})$

Since a>b and c>d (both are strictly ordered pairs of virtual indices),
the bracket B(ab,cd) is nonzero only when the Kronecker deltas are satisfied.

Consider what index equalities are possible given a>b and c>d:

  Case (c,d) = (a,b):  i.e. c=a, d=b
    => δ_{ac}=1, δ_{bd}=1, δ_{ad}=0, δ_{bc}=0   (since a>b means a≠b)
    => B = f_{aa}·1 + f_{bb}·1 - 0 - 0 - (ε_i + ε_j)(1·1 - 0)
         = f_{aa} + f_{bb} - ε_i - ε_j

  Case (c,d) = (b,a):  i.e. c=b, d=a
    => This requires c>d, i.e. b>a. But we have a>b, so b>a is impossible.
    => This case never occurs in the restricted sum.

  Any other (c,d):
    => At most one delta can be 1 at a time, but each term in B requires
       TWO deltas to simultaneously be 1 (one on each pair).
       For example, f_{ac} δ_{bd} requires both c≠a generically AND d=b,
       but then the f_{ac} term doesn't collapse further — however,
       the (ε_i+ε_j) terms require BOTH δ_{ac}=1 AND δ_{bd}=1 simultaneously,
       which only happens when (c,d)=(a,b).
       
       More carefully: if d=b but c≠a, we get f_{ac}·1 from the first term,
       and if c=a but d≠b, we get f_{bd}·1 from the second term.
       But these contributions combine with the double sum over (c>d)
       and can be absorbed. Let's verify by expanding fully.

### Step 2: Expand Term 1 carefully

Term 1 = Σ_{i>j} Σ_{a>b} Σ_{c>d} t_{ij}^{ab} t_{ij}^{cd} B(ab,cd)

Split B into sub-terms:

  (i)   Σ_{c>d} t_{ij}^{cd} f_{ac} δ_{bd}
        δ_{bd} forces d=b. With c>d=b, this sums over c>b:
        = Σ_{c>b} t_{ij}^{cb} f_{ac}

  (ii)  Σ_{c>d} t_{ij}^{cd} f_{bd} δ_{ac}
        δ_{ac} forces c=a. With c=a>d, this sums over d<a:
        = Σ_{d<a} t_{ij}^{ad} f_{bd}

  (iii) -Σ_{c>d} t_{ij}^{cd} f_{ad} δ_{bc}
        δ_{bc} forces c=b. With c=b>d:
        = -Σ_{d<b} t_{ij}^{bd} f_{ad}

  (iv)  -Σ_{c>d} t_{ij}^{cd} f_{bc} δ_{ad}
        δ_{ad} forces d=a. With c>d=a:
        = -Σ_{c>a} t_{ij}^{ca} f_{bc}

  (v)   -(ε_i+ε_j) Σ_{c>d} t_{ij}^{cd} (δ_{ac}δ_{bd} - δ_{ad}δ_{bc})
        First part: δ_{ac}δ_{bd} forces (c,d)=(a,b) => t_{ij}^{ab}
        Second part: δ_{ad}δ_{bc} forces (c,d)=(b,a), but b<a so c>d
        is violated => contributes 0
        = -(ε_i+ε_j) t_{ij}^{ab}

So Term 1 = Σ_{i>j} Σ_{a>b} t_{ij}^{ab} [
    Σ_{c>b} t_{ij}^{cb} f_{ac}         ... (i)
  + Σ_{d<a} t_{ij}^{ad} f_{bd}         ... (ii)
  - Σ_{d<b} t_{ij}^{bd} f_{ad}         ... (iii)
  - Σ_{c>a} t_{ij}^{ca} f_{bc}         ... (iv)
  - (ε_i+ε_j) t_{ij}^{ab}             ... (v)
]

### Step 3: Use antisymmetry of t and convert restricted sums to full sums

The amplitudes satisfy: t_{ij}^{ab} = -t_{ij}^{ba}  (antisymmetric in a,b)

For sub-term (i): Σ_{c>b} t_{ij}^{cb} f_{ac}
  In the full (unrestricted) sum over all c: Σ_c t_{ij}^{cb} f_{ac}
  = Σ_{c>b} t_{ij}^{cb} f_{ac} + t_{ij}^{bb} f_{ab} + Σ_{c<b} t_{ij}^{cb} f_{ac}
  
  t_{ij}^{bb} = 0 (antisymmetry with a=b), and
  Σ_{c<b} t_{ij}^{cb} f_{ac} = -Σ_{c<b} t_{ij}^{bc} f_{ac}  (by antisymmetry)
  
  Meanwhile sub-term (iii): -Σ_{d<b} t_{ij}^{bd} f_{ad}
  Relabel d->c: = -Σ_{c<b} t_{ij}^{bc} f_{ac}

  So (i) + (iii) = Σ_{c>b} t_{ij}^{cb} f_{ac} - Σ_{c<b} t_{ij}^{bc} f_{ac}
                  = Σ_{c>b} t_{ij}^{cb} f_{ac} + Σ_{c<b} t_{ij}^{cb} f_{ac}
                  = Σ_{c≠b} t_{ij}^{cb} f_{ac}
                  = Σ_c t_{ij}^{cb} f_{ac}     (since c=b gives 0)

Similarly, (ii) + (iv):
  (ii):  Σ_{d<a} t_{ij}^{ad} f_{bd}
  (iv): -Σ_{c>a} t_{ij}^{ca} f_{bc}, relabel c->d: = -Σ_{d>a} t_{ij}^{da} f_{bd}
                                                     = Σ_{d>a} t_{ij}^{ad} f_{bd}
  So (ii)+(iv) = Σ_{d<a} t_{ij}^{ad} f_{bd} + Σ_{d>a} t_{ij}^{ad} f_{bd}
               = Σ_d t_{ij}^{ad} f_{bd}      (since d=a gives 0)

Therefore:

Term 1 = Σ_{i>j} Σ_{a>b} t_{ij}^{ab} [
    Σ_c t_{ij}^{cb} f_{ac}
  + Σ_d t_{ij}^{ad} f_{bd}
  - (ε_i+ε_j) t_{ij}^{ab}
]

### Step 4: Recognize the Fock operator action

For canonical orbitals (diagonal Fock matrix): f_{ac} = ε_a δ_{ac}

So:
  Σ_c t_{ij}^{cb} f_{ac} = Σ_c t_{ij}^{cb} ε_a δ_{ac} = ε_a t_{ij}^{ab}
  Σ_d t_{ij}^{ad} f_{bd} = Σ_d t_{ij}^{ad} ε_b δ_{bd} = ε_b t_{ij}^{ab}

Therefore:

Term 1 = Σ_{i>j} Σ_{a>b} t_{ij}^{ab} [ ε_a t_{ij}^{ab} + ε_b t_{ij}^{ab} - (ε_i+ε_j) t_{ij}^{ab} ]
       = Σ_{i>j} Σ_{a>b} (t_{ij}^{ab})² (ε_a + ε_b - ε_i - ε_j)

### Step 5: Substitute the amplitude definition

t_{ij}^{ab} = -<ab||ij> / (ε_a + ε_b - ε_i - ε_j)

Define D_{abij} = ε_a + ε_b - ε_i - ε_j for brevity.

(t_{ij}^{ab})² · D_{abij} = [ <ab||ij>² / D_{abij}² ] · D_{abij}
                            = <ab||ij>² / D_{abij}

And for Term 2:
  t_{ij}^{ab} · <ab||ij> = -<ab||ij>² / D_{abij}

### Step 6: Combine Term 1 + Term 2

J_2 = Σ_{i>j} Σ_{a>b} [ <ab||ij>² / D_{abij}  +  2 · (-<ab||ij>² / D_{abij}) ]
    = Σ_{i>j} Σ_{a>b} [ <ab||ij>² / D_{abij} - 2 <ab||ij>² / D_{abij} ]
    = Σ_{i>j} Σ_{a>b} [ -<ab||ij>² / D_{abij} ]
    = Σ_{i>j} Σ_{a>b} t_{ij}^{ab} · <ab||ij>

### Step 7: Convert restricted sum to unrestricted sum

The antisymmetry properties are:
  t_{ij}^{ab} = -t_{ij}^{ba} = -t_{ji}^{ab} = t_{ji}^{ba}
  <ab||ij>    = -<ba||ij>    = -<ab||ji>    = <ba||ji>

Therefore t_{ij}^{ab} <ab||ij> is symmetric under (a<->b) and under (i<->j):
  swapping a,b: (-t)(-<>) = t·<>  ✓
  swapping i,j: (-t)(-<>) = t·<>  ✓

The unrestricted sum counts each (a,b) pair twice (once as a>b, once as b>a)
and each (i,j) pair twice. So:

  Σ_{i>j} Σ_{a>b} t_{ij}^{ab} <ab||ij>  =  (1/4) Σ_{i,j} Σ_{a,b} t_{ij}^{ab} <ab||ij>

### Final result:

  J_2 = (1/4) Σ_{abij} t_{ij}^{ab} <ab||ij>

This is the standard closed-form MP2 correlation energy. ∎

### Implementation:

  J_2 = 0.25 * np.einsum('abij,abij->', t_block, eri_as_block[:nvir_act, :nvir_act, :, :])