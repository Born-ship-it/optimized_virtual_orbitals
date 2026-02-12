## Reduction of the Hylleraas $J_2$ functional to the standard MP2 closed form

### Starting point — the Hylleraas functional:

$$
\begin{aligned}
J_2 &= \sum_{i>j} \Bigg[ \sum_{a>b} \sum_{c>d} t_{ij}^{ab}\, t_{ij}^{cd} \Big( f_{ac}\,\delta_{bd} + f_{bd}\,\delta_{ac} - f_{ad}\,\delta_{bc} - f_{bc}\,\delta_{ad} \\
&\qquad\qquad - (\varepsilon_i + \varepsilon_j)(\delta_{ac}\,\delta_{bd} - \delta_{ad}\,\delta_{bc}) \Big) + 2 \sum_{a>b} t_{ij}^{ab} \langle ab \| ij \rangle \Bigg]
\end{aligned}
$$

Call the first part "Term 1" and the second part "Term 2".

---

### Step 1: Simplify the bracket via Kronecker deltas

The bracket multiplying $t_{ij}^{ab}\, t_{ij}^{cd}$ is:

$$
\begin{aligned}
B(ab,cd) &= f_{ac}\,\delta_{bd} + f_{bd}\,\delta_{ac} - f_{ad}\,\delta_{bc} - f_{bc}\,\delta_{ad} \\
&\quad - (\varepsilon_i + \varepsilon_j)(\delta_{ac}\,\delta_{bd} - \delta_{ad}\,\delta_{bc})
\end{aligned}
$$

Since $a > b$ and $c > d$ (both are strictly ordered pairs of virtual indices), the bracket $B(ab,cd)$ is nonzero only when the Kronecker deltas are satisfied.

Consider what index equalities are possible given $a > b$ and $c > d$:

**Case $(c,d) = (a,b)$**, i.e. $c = a,\; d = b$:

$$
\begin{aligned}
&\delta_{ac} = 1, \quad \delta_{bd} = 1, \quad \delta_{ad} = 0, \quad \delta_{bc} = 0 \quad \text{(since } a > b \text{ means } a \neq b\text{)} \\
&\Rightarrow \quad B = f_{aa} \cdot 1 + f_{bb} \cdot 1 - 0 - 0 - (\varepsilon_i + \varepsilon_j)(1 \cdot 1 - 0) \\
&\phantom{\Rightarrow \quad B} = f_{aa} + f_{bb} - \varepsilon_i - \varepsilon_j
\end{aligned}
$$

**Case $(c,d) = (b,a)$**, i.e. $c = b,\; d = a$:

This requires $c > d$, i.e. $b > a$. But we have $a > b$, so $b > a$ is impossible. This case never occurs in the restricted sum.

**Any other $(c,d)$:**

At most one delta can be 1 at a time, but each term in $B$ requires two deltas to simultaneously be 1. For example, the $(\varepsilon_i + \varepsilon_j)$ terms require both $\delta_{ac} = 1$ and $\delta_{bd} = 1$ simultaneously, which only happens when $(c,d) = (a,b)$.

More carefully: if $d = b$ but $c \neq a$, we get $f_{ac} \cdot 1$ from the first term; if $c = a$ but $d \neq b$, we get $f_{bd} \cdot 1$ from the second term. These contributions survive and are expanded in Step 2.

---

### Step 2: Expand Term 1 carefully

$$
\text{Term 1} = \sum_{i>j} \sum_{a>b} \sum_{c>d} t_{ij}^{ab}\, t_{ij}^{cd}\, B(ab,cd)
$$

Split $B$ into sub-terms by evaluating $\sum_{c>d} t_{ij}^{cd}\, B(ab,cd)$:

$$
\begin{aligned}
\text{(i)} &\quad \sum_{c>d} t_{ij}^{cd}\, f_{ac}\, \delta_{bd} \quad\Rightarrow\quad \delta_{bd} \text{ forces } d = b, \text{ with } c > d = b: \\
&\quad = \sum_{c>b} t_{ij}^{cb}\, f_{ac} \\[8pt]
\text{(ii)} &\quad \sum_{c>d} t_{ij}^{cd}\, f_{bd}\, \delta_{ac} \quad\Rightarrow\quad \delta_{ac} \text{ forces } c = a, \text{ with } c = a > d: \\
&\quad = \sum_{d<a} t_{ij}^{ad}\, f_{bd} \\[8pt]
\text{(iii)} &\quad -\sum_{c>d} t_{ij}^{cd}\, f_{ad}\, \delta_{bc} \quad\Rightarrow\quad \delta_{bc} \text{ forces } c = b, \text{ with } c = b > d: \\
&\quad = -\sum_{d<b} t_{ij}^{bd}\, f_{ad} \\[8pt]
\text{(iv)} &\quad -\sum_{c>d} t_{ij}^{cd}\, f_{bc}\, \delta_{ad} \quad\Rightarrow\quad \delta_{ad} \text{ forces } d = a, \text{ with } c > d = a: \\
&\quad = -\sum_{c>a} t_{ij}^{ca}\, f_{bc} \\[8pt]
\text{(v)} &\quad -(\varepsilon_i + \varepsilon_j) \sum_{c>d} t_{ij}^{cd}\, (\delta_{ac}\,\delta_{bd} - \delta_{ad}\,\delta_{bc}) \\
&\quad \text{First part: } \delta_{ac}\,\delta_{bd} \text{ forces } (c,d) = (a,b) \;\Rightarrow\; t_{ij}^{ab} \\
&\quad \text{Second part: } \delta_{ad}\,\delta_{bc} \text{ forces } (c,d) = (b,a), \text{ but } b < a \text{ so } c > d \text{ is violated} \;\Rightarrow\; 0 \\
&\quad = -(\varepsilon_i + \varepsilon_j)\, t_{ij}^{ab}
\end{aligned}
$$

So:

$$
\begin{aligned}
\text{Term 1} &= \sum_{i>j} \sum_{a>b} t_{ij}^{ab} \bigg[ \sum_{c>b} t_{ij}^{cb}\, f_{ac} + \sum_{d<a} t_{ij}^{ad}\, f_{bd} \\
&\qquad\qquad\qquad\quad - \sum_{d<b} t_{ij}^{bd}\, f_{ad} - \sum_{c>a} t_{ij}^{ca}\, f_{bc} - (\varepsilon_i + \varepsilon_j)\, t_{ij}^{ab} \bigg]
\end{aligned}
$$

---

### Step 3: Use antisymmetry of $t$ and convert restricted sums to full sums

The amplitudes satisfy: $t_{ij}^{ab} = -t_{ij}^{ba}$ (antisymmetric in $a,b$).

**Combining (i) + (iii):**

$$
\begin{aligned}
\text{(i):} \quad &\sum_{c>b} t_{ij}^{cb}\, f_{ac} \\
\text{(iii):} \quad &-\sum_{d<b} t_{ij}^{bd}\, f_{ad} \;\xrightarrow{d \to c}\; -\sum_{c<b} t_{ij}^{bc}\, f_{ac} = \sum_{c<b} t_{ij}^{cb}\, f_{ac} \\[6pt]
\text{(i) + (iii)} &= \sum_{c>b} t_{ij}^{cb}\, f_{ac} + \sum_{c<b} t_{ij}^{cb}\, f_{ac} = \sum_{c \neq b} t_{ij}^{cb}\, f_{ac} = \sum_c t_{ij}^{cb}\, f_{ac}
\end{aligned}
$$

where the last equality uses $t_{ij}^{bb} = 0$ (antisymmetry).

**Combining (ii) + (iv):**

$$
\begin{aligned}
\text{(ii):} \quad &\sum_{d<a} t_{ij}^{ad}\, f_{bd} \\
\text{(iv):} \quad &-\sum_{c>a} t_{ij}^{ca}\, f_{bc} \;\xrightarrow{c \to d}\; -\sum_{d>a} t_{ij}^{da}\, f_{bd} = \sum_{d>a} t_{ij}^{ad}\, f_{bd} \\[6pt]
\text{(ii) + (iv)} &= \sum_{d<a} t_{ij}^{ad}\, f_{bd} + \sum_{d>a} t_{ij}^{ad}\, f_{bd} = \sum_d t_{ij}^{ad}\, f_{bd}
\end{aligned}
$$

where again $t_{ij}^{aa} = 0$.

**Therefore:**

$$
\text{Term 1} = \sum_{i>j} \sum_{a>b} t_{ij}^{ab} \bigg[ \sum_c t_{ij}^{cb}\, f_{ac} + \sum_d t_{ij}^{ad}\, f_{bd} - (\varepsilon_i + \varepsilon_j)\, t_{ij}^{ab} \bigg]
$$

---

### Step 4: Recognize the Fock operator action

For canonical orbitals (diagonal Fock matrix): $f_{ac} = \varepsilon_a\, \delta_{ac}$.

$$
\begin{aligned}
\sum_c t_{ij}^{cb}\, f_{ac} &= \sum_c t_{ij}^{cb}\, \varepsilon_a\, \delta_{ac} = \varepsilon_a\, t_{ij}^{ab} \\[4pt]
\sum_d t_{ij}^{ad}\, f_{bd} &= \sum_d t_{ij}^{ad}\, \varepsilon_b\, \delta_{bd} = \varepsilon_b\, t_{ij}^{ab}
\end{aligned}
$$

Therefore:

$$
\begin{aligned}
\text{Term 1} &= \sum_{i>j} \sum_{a>b} t_{ij}^{ab} \Big[ \varepsilon_a\, t_{ij}^{ab} + \varepsilon_b\, t_{ij}^{ab} - (\varepsilon_i + \varepsilon_j)\, t_{ij}^{ab} \Big] \\
&= \sum_{i>j} \sum_{a>b} \big(t_{ij}^{ab}\big)^2 \big(\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j\big)
\end{aligned}
$$

---

### Step 5: Substitute the amplitude definition

$$
t_{ij}^{ab} = -\frac{\langle ab \| ij \rangle}{\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j}
$$

Define $D_{abij} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$ for brevity.

$$
\begin{aligned}
\big(t_{ij}^{ab}\big)^2 \cdot D_{abij} &= \frac{\langle ab \| ij \rangle^2}{D_{abij}^2} \cdot D_{abij} = \frac{\langle ab \| ij \rangle^2}{D_{abij}} \\[8pt]
t_{ij}^{ab} \cdot \langle ab \| ij \rangle &= -\frac{\langle ab \| ij \rangle^2}{D_{abij}}
\end{aligned}
$$

---

### Step 6: Combine Term 1 + Term 2

$$
\begin{aligned}
J_2 &= \sum_{i>j} \sum_{a>b} \left[ \frac{\langle ab \| ij \rangle^2}{D_{abij}} + 2 \left( -\frac{\langle ab \| ij \rangle^2}{D_{abij}} \right) \right] \\[6pt]
&= \sum_{i>j} \sum_{a>b} \left[ \frac{\langle ab \| ij \rangle^2}{D_{abij}} - \frac{2\,\langle ab \| ij \rangle^2}{D_{abij}} \right] \\[6pt]
&= \sum_{i>j} \sum_{a>b} \left[ -\frac{\langle ab \| ij \rangle^2}{D_{abij}} \right] \\[6pt]
&= \sum_{i>j} \sum_{a>b} t_{ij}^{ab} \cdot \langle ab \| ij \rangle
\end{aligned}
$$

---

### Step 7: Convert restricted sum to unrestricted sum

The antisymmetry properties are:

$$
\begin{aligned}
t_{ij}^{ab} &= -t_{ij}^{ba} = -t_{ji}^{ab} = t_{ji}^{ba} \\
\langle ab \| ij \rangle &= -\langle ba \| ij \rangle = -\langle ab \| ji \rangle = \langle ba \| ji \rangle
\end{aligned}
$$

Therefore the product $t_{ij}^{ab}\, \langle ab \| ij \rangle$ is symmetric under both $(a \leftrightarrow b)$ and $(i \leftrightarrow j)$:

$$
\begin{aligned}
\text{swap } a,b: \quad (-t)(-\langle\rangle) &= t \cdot \langle\rangle \quad \checkmark \\
\text{swap } i,j: \quad (-t)(-\langle\rangle) &= t \cdot \langle\rangle \quad \checkmark
\end{aligned}
$$

The unrestricted sum counts each $(a,b)$ pair twice and each $(i,j)$ pair twice, so:

$$
\sum_{i>j} \sum_{a>b} t_{ij}^{ab}\, \langle ab \| ij \rangle = \frac{1}{4} \sum_{i,j} \sum_{a,b} t_{ij}^{ab}\, \langle ab \| ij \rangle
$$

---

### Final result:

$$
\boxed{\; J_2 = \frac{1}{4} \sum_{a,b,i,j} t_{ij}^{ab}\, \langle ab \| ij \rangle \;}
$$

This is the standard closed-form MP2 correlation energy. $\blacksquare$
