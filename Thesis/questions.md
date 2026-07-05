# Thesis Committee Questions & Answers – OVOS Implementation

## 1. Implementation details · How does your OVOS implementation differ in detail from Adamowicz & Bartlett’s original, especially regarding initial active space selection?

**Answer:**  
Our implementation differs mainly in the **initial active space selection**. Adamowicz & Bartlett originally ranked virtual orbitals by their estimated contribution to the correlation energy (using both diagonal and off‑diagonal MP2 terms). In contrast, we adopt a **simpler sequential selection** based on spin‑orbital indices: after the occupied orbitals, the next \(N'_{\text{virt}}\) spin‑orbitals are taken as active, and the remainder as inactive. This choice is explicitly stated in the *Method* chapter:

> *“In the current implementation, we adopt a simpler approach where the active virtual orbitals are selected sequentially based on their spin-orbital indices.”* (Section \ref{sec:method_algorithm}, “Initial Active Space Selection”)

Other differences noted in the *Conclusion* include: we do not perform a direct comparison with the original code, and our integral evaluation, convergence criteria, and handling of ill‑conditioning are not identical. However, the energy‑recovery behaviour agrees closely (mean deviation 0.35 %), validating our implementation despite these simplifications.

---

## 2. Convergence · What specific strategies did you use to address convergence failures or local minima in challenging cases?

**Answer:**  
We employed several strategies to mitigate convergence problems and local minima:

- **Multiple initialisation strategies** (RHF, warm‑start from a smaller active space, random unitary sampling). The *Results* chapter shows that random sampling or RHF sometimes finds lower‑energy solutions than the warm‑start, especially when the active space approaches full size.
- **Block‑diagonal Hessian approximation** (the “Reduced Linear Equation” or RLE). This simplifies the Newton‑Raphson step and reduces the number of negative eigenvalues, improving stability (see *Theory*, Hessian section).
- **Strict convergence criteria** with a persistence counter (`keep_track_max`). Convergence requires that the energy change, gradient norm, and T1 norm all stay below thresholds for a set number of consecutive iterations (see *Method*, “Convergence and Iteration Control”).
- **Maximum iteration limit** (default 1000) to prevent infinite loops.

**Notably**, we did **not** implement a level‑shift strategy (explicitly stated in *Theory*: “The implementation of OVOS omits the level‑shift strategy”). For future work, we suggest trust‑region or line‑search methods to further improve robustness.

---

## 3. Comparative performance · Why do UMP2 natural orbitals sometimes outperform OVOS orbitals as VQE references, and what factors influence this?

**Answer:**  
UMP2 natural orbitals (NOs) **consistently give lower VQE energies** than OVOS orbitals because they are explicitly constructed to maximise the MP2 correlation energy per orbital. As explained in *Theory* (Section \ref{subsec:QC_OS}), UMP2 NOs diagonalise the one‑particle density matrix derived from the MP2 amplitudes, which directly captures the most important correlation contributions. OVOS, in contrast, optimises a restricted active‑inactive rotation and does not enforce the full stationarity condition for occupied‑virtual rotations.

**Factors that influence the performance gap:**

- **Size of the active space** – For very small \(N'_{\text{virt}}\), OVOS recovers less correlation (e.g., at 30 % virtuals, OVOS recovers 88–96 % vs. UMP2 NOs 96–99 %, Table 1 in *Conclusion*).
- **Molecular system** – H₂O shows the smallest gap (at some geometries OVOS almost matches UMP2 NOs), while HF shows the largest deficit (42 m\(E_h\)).
- **Basis set** – The gap tends to be larger for more diffuse or larger basis sets because UMP2 NOs can exploit more virtual orbitals.
- **Presence of local minima** – In challenging cases (e.g., CO, Li₂ near full space), OVOS may converge to a higher local minimum, whereas UMP2 NOs are obtained by a direct diagonalisation that is not prone to such trapping.

Nevertheless, OVOS orbitals require **fewer qubits** (since they are used directly as the basis, whereas UMP2 NOs still need the same number of virtuals for the same truncation) and give **similar VQE iteration counts** (e.g., 48 vs. 43 for H₂O), making them attractive when qubit reduction is the primary goal.

---

## 4. Spin contamination · How significant is spin contamination in your UHF-based OVOS results, and how might a restricted formalism change outcomes?

**Answer:**  
The thesis uses an **unrestricted Hartree‑Fock (UHF) reference** for generality, which inevitably introduces **spin contamination** – the expectation value \(\langle \hat{S}^2 \rangle\) deviates from the ideal value for a pure spin state. The *Conclusion* explicitly notes:

> *“The use of a UHF reference for closed‑shell molecules introduces spin contamination, potentially affecting the quality of the optimised orbitals. An RHF reference would be cleaner and may lead to more stable convergence for closed‑shell systems.”*

While the thesis does **not quantify** the magnitude of spin contamination in the final OVOS orbitals, it is expected to be non‑negligible for stretched bonds or genuinely open‑shell molecules. For **closed‑shell molecules near equilibrium** (H₂O, CO, HF, NH₃, and Li₂ in its ground state), the contamination is absent because the UHF solution collapses to RHF when symmetry is not broken. However, the use of UHF still doubles the number of virtual spin‑orbitals (though the α and β sets are identical for closed‑shell systems) and may lead to spin‑polarised orbitals if symmetry is artificially broken.

**How a restricted (RHF) formalism would change outcomes:**

- **Cleaner spin state** – The wavefunction would be an eigenfunction of \(\hat{S}^2\), removing a source of error and potentially improving the quality of optimised virtual orbitals.
- **Reduced computational cost** – The number of virtual orbitals in an RHF calculation is half that of UHF (spatial orbitals only), so OVOS would operate on a smaller space, leading to faster iterations and lower memory.
- **Stability** – RHF is less prone to oscillatory convergence for closed‑shell systems, especially when the active space is large.
- **Loss of generality** – RHF cannot describe open‑shell systems (e.g., O₂, radicals), so a UHF reference would still be needed for such cases.

A restricted OVOS implementation is listed as a future extension in the *Outlook*.

---

## 5. Quantum computing impact · Can you quantify the practical qubit and circuit depth savings when using OVOS orbitals in VQE for larger systems?

**Answer:**  
Yes, the savings can be quantified directly from the **reduction in virtual orbital count**:

- **Qubit savings** – For a VQE calculation using a Jordan‑Wigner or Bravyi‑Kitaev mapping, the number of qubits required is proportional to the number of spin‑orbitals. If the original full virtual space has \(N_{\text{virt}}\) orbitals and OVOS retains \(N'_{\text{virt}}\) active virtuals, the qubit count reduces from \(2(N_{\text{occ}} + N_{\text{virt}})\) to \(2(N_{\text{occ}} + N'_{\text{virt}})\). The **saving factor** is roughly \(N_{\text{virt}} / N'_{\text{virt}}\).

  **Example from results** (cc‑pVDZ basis):
  - Li₂: full virtuals = 25, 90 % correlation at \(N'_{\text{virt}} = 7\) (28 %). Saving factor ≈ 3.6×.
  - CO: full virtuals = 21, 90 % at \(N'_{\text{virt}} = 13\) (62 %). Saving factor ≈ 1.6×.
  - H₂O: full virtuals = 19, 90 % at \(N'_{\text{virt}} = 13\) (68 %). Saving factor ≈ 1.5×.
  - HF: full virtuals = 14, 90 % at \(N'_{\text{virt}} = 9\) (64 %). Saving factor ≈ 1.6×.

- **Circuit depth savings** – Two effects:
    1. **Fewer qubits** directly reduces the number of two‑qubit gates in typical VQE ansätze (e.g., the number of Pauli terms scales polynomially with the number of orbitals).
    2. **Reduced T1 amplitudes** – If OVOS orbitals approximate Brueckner orbitals (as argued in *Theory*, Section \ref{subsec:brueckner_background}), the single‑excitation amplitudes become small. This allows replacing the UCCSD ansatz with a cheaper **UCCD** ansatz (only doubles), which reduces the circuit depth by roughly a factor of 2–3 (depending on the implementation). The thesis does not numerically quantify T1 reduction, but it lists this as a key future step.

- **Practical impact for larger systems** – For a molecule with, say, 100 virtual orbitals, using OVOS with \(N'_{\text{virt}} \approx 50\) would halve the qubit requirement and likely reduce the circuit depth by a factor of 2–4, making it feasible on near‑term devices with 50–100 qubits. The *Outlook* explicitly states that such savings are “directly relevant to quantum chemistry on near‑term and fault‑tolerant quantum devices.”

In summary, OVOS can provide **qubit reductions of 1.5× to >3.5×** (with typical 1.5–2× for most molecules) and potentially **comparable circuit depth reductions**, especially if the T1 amplitudes are suppressed enough to allow a UCCD ansatz.

---

## 6. Validation against Adamowicz & Bartlett – How did you ensure that your implementation reproduces their results given the differences in initial active space selection and integral conventions?

**Answer:**  
The conclusion states that agreement is within 0.6 percentage points (mean deviation 0.35%) for energy recovery, which is reassuring. However, the thesis does **not** provide a direct one‑to‑one comparison for the same molecule and basis set (CH₂ with a large basis set is used in the original, while we tested smaller molecules). Table 1 in the conclusion compares Li₂, H₂O, HF – but Adamowicz & Bartlett did not report those molecules. The only direct comparison is a table (Table \ref{tab:comparison_AB} in Results) that lists CH₂ from the original vs. our molecules, which is not an apples‑to‑apples comparison. We acknowledge this limitation: the agreement in energy recovery behaviour is reassuring but does not guarantee that the same local minima are being found. A full numerical validation against the original CH₂ test case was not performed due to differences in basis set availability and computational resources.

**Committee expectation:** A clear statement of which original result was reproduced (or why it was not possible) and a quantitative error bar.

---

## 7. Newton‑Raphson stability – You mention that you did not implement level‑shifting, yet you observed convergence issues for larger active spaces. Could the lack of level‑shifting be the primary cause, and how would you modify the algorithm to guarantee convergence?

**Answer:**  
The lack of level‑shifting likely contributes to convergence issues. The results section reports many cases where the algorithm fails to converge (reaches max iterations) or converges to a poor local minimum (e.g., CO at N′=11–17, Li₂ at N′=12–23). However, the thesis does **not** systematically link these failures to Hessian indefiniteness (e.g., eigenvalue spectra are not shown). Without such diagnostics, the primary cause is uncertain. To guarantee convergence, we would implement an adaptive level‑shift: at each iteration, compute the smallest eigenvalue of each Hessian block; if negative, add a shift \(\lambda = |\lambda_{\min}| + \delta\). We would also consider trust‑region methods.

**Committee expectation:** Quantitative diagnostics (e.g., percentage of iterations where Hessian had negative eigenvalues) and a proposed adaptive level‑shift threshold.

---

## 8. T1 amplitudes and Brueckner character – You claim OVOS orbitals approximate Brueckner orbitals and should reduce T1 amplitudes, but you never actually computed T1 amplitudes for your OVOS orbitals. How can you support this claim without numerical evidence?

**Answer:**  
This is a major gap. The theory section draws a theoretical connection: stationarity of the MP2 energy with respect to active‑inactive rotations implies vanishing of certain T1 components. However, the results section **never** reports T1 norms for any molecule or active space. The convergence criteria include `t1_norm` (mentioned in the convergence code listing), but the values are not presented. The VQE section speculates about reduced circuit depth via UCCD, but no T1 data backs this up. We acknowledge that the claim remains unsupported by numerical evidence and should be treated as a hypothesis for future work.

**Committee expectation:** A table showing T1 norms for OVOS vs. UHF vs. UMP2 natural orbitals for at least one molecule across different active space sizes.

---

## 9. Computational cost – You provide scaling analysis, but you never report actual wall‑clock times or iteration counts for your OVOS runs versus a full MP2 calculation. How much faster is OVOS in practice for your test systems?

**Answer:**  
The thesis reports **iteration counts** (e.g., 797 iterations for NH₃, 668 for Li₂) but no timing data. For small basis sets like 6‑31G, the full MP2 calculation is very fast; OVOS might actually be slower due to many iterations. The text mentions “breakeven analysis” but does not perform one numerically. We recommend future work to benchmark CPU time as a function of \(N'_{\text{virt}}/N_{\text{virt}}\) and to compare against full MP2.

**Committee expectation:** A plot or table of CPU time (or number of integral transformations) for OVOS vs. full MP2 as a function of N′/N_virt, for a representative molecule and basis set.

---

## 10. VQE results – In the VQE potential energy curves, the OVOS orbitals sometimes give **higher** energy than UHF (e.g., HF at 1.32 Å). Can you explain this specific counter‑example? Does it indicate that OVOS can be worse than no optimisation?

**Answer:**  
The conclusion mentions that for HF, “the converged energy was higher than that of the UHF orbitals.” The results section does not analyse why. Possible reasons: spin contamination (HF at 1.32 Å is stretched; UHF becomes spin‑contaminated), convergence to a local minimum in the OVOS optimisation, or the active space truncation (75% of virtuals) being insufficient. Without a systematic investigation (e.g., checking T1 amplitudes, trying a larger active space, or comparing with RHF-based OVOS), this counter‑example weakens the claim that OVOS “generally provides a better reference state.”

**Committee expectation:** A dedicated sub‑section analysing the HF failure, including energy differences, overlap with FCI (if available), and maybe a comparison of orbital shapes.

---

## 11. Warm‑start strategy – You use a warm‑start where converged orbitals from N′ virtuals are used as initial guess for N′+2. How do you handle the mismatch in dimensionality (the new active space has two additional virtuals)? Did you just append canonical virtuals, and does this bias the optimisation?

**Answer:**  
The method section shows pseudo‑code but does **not** explain how the additional virtual orbitals are initialised. When increasing N′ by 2, the two new active orbitals are taken as the next canonical virtuals (by orbital energy) from the previous iteration’s inactive set, appended to the active set. This choice may bias the optimisation because those orbitals are not rotated before inclusion. The results show that warm‑start often gives worse final energies than RHF or random, which might be due to this poor initialisation. A better approach would be to perform a small pre‑optimisation of the expanded active space.

**Committee expectation:** A clear description of the expansion procedure, and ideally a test where the warm‑start is compared to a naive expansion vs. a smarter guess (e.g., based on MP2 natural orbital occupation).

---

## 12. Random unitary sampling – You use 500 random unitary rotations in the virtual space. How did you choose that number? Did you check convergence of the best energy with respect to the number of samples?

**Answer:**  
The number 500 was chosen empirically as a balance between computational cost and coverage of the orbital space. The thesis does **not** report a sensitivity analysis (e.g., best energy vs. number of samples). Without such data, it is unclear whether 500 is sufficient or if 100 would be enough (or 1000 needed). For a challenging case like CO with many local minima, the best energy might still improve with more samples.

**Committee expectation:** A convergence plot of the minimum energy found vs. number of random samples for a challenging case, showing that 500 is in the plateau region.

---

## 13. Basis set comparison – You conclude that “going up in basis set size showed mostly an improvement in correlation energy for small numbers of N′_virt.” However, your figures show that at fixed absolute N′ (e.g., 5 virtuals), the smaller basis set (6‑31G) recovers a **higher fraction** of its full correlation energy than cc‑pVDZ. Which metric is more meaningful for practical use: fraction of full correlation or absolute energy recovered? Please justify.

**Answer:**  
The thesis presents both perspectives but does not take a clear stance. The committee would expect a justification based on the target application. For near‑term quantum devices with a fixed qubit budget, the **absolute number of virtual orbitals** is the constraint, so comparing at fixed N′ is more meaningful. For classical MP2 scaling, the **fraction of virtuals** matters because computational cost scales with N′⁴. The candidate should argue that for quantum computing, the absolute reduction (saving qubits) is the primary metric, while for classical accuracy, the fraction of recovered correlation energy is more relevant.

**Committee expectation:** A clear, application‑driven justification for which metric is used in different parts of the thesis.

---

## 14. VQE iteration count spread – You report that OVOS orbitals show a “bigger spread” in iteration counts compared to UHF and UMP2 natural orbitals. Could this be due to the VQE optimisation getting stuck in different local minima for different geometries? Did you repeat each VQE run multiple times to assess statistical variability?

**Answer:**  
The boxplots show the spread, but the text does not discuss whether each data point is a single run or an average. The methods section says the BFGS algorithm was used and mentions “random initialization of thetas” as a variant, but it is unclear which initialisation was used for the boxplots. The committee would ask: “Did you perform multiple independent runs per geometry to estimate uncertainty?” Without this, the spread might reflect run‑to‑run stochasticity rather than a physical trend.

**Committee expectation:** A clear statement of the number of independent runs per geometry, the random seed policy, and error bars on the boxplots.

---

## 15. Outlook – You propose extending OVOS to CCSD or CCSD(T). How would the gradient and Hessian expressions change? Have you derived them, or is this a hand‑waving future direction?

**Answer:**  
The outlook is vague. The theory chapter only derives the MP2‑based gradient and Hessian. CCSD Lagrangian gradients are considerably more complex and involve solving the Λ equations. The candidate has not derived these expressions. A realistic future direction would require: (1) formulating the CCSD energy Lagrangian, (2) computing the gradient with respect to orbital rotations, which includes response terms from the amplitude equations, and (3) approximating or fully computing the Hessian. This is a substantial project, not a simple extension. The thesis should at least cite literature where this has been done (e.g., OO‑CCSD methods) and outline the expected increase in computational cost.

**Committee expectation:** A brief sketch of the key differences (e.g., the need to solve for perturbed amplitudes, the extra terms from T1 amplitudes) and a realistic estimate of the additional computational cost.

---

## Summary for the Candidate

Your thesis provides a solid implementation and many useful results, but the committee will probe the following **critical gaps**:

1. **Missing validation** against an explicit original test case (e.g., CH₂).
2. **No T1 amplitude data** to support the Brueckner orbital claim.
3. **No timing/performance data** to demonstrate practical speedup.
4. **Insufficient analysis of the HF counter‑example** where OVOS performed worse.
5. **Undocumented warm‑start expansion** (how new virtuals are added).
6. **Lack of justification** for the number of random samples (500).
7. **No uncertainty quantification** for VQE iteration counts.

Prepare detailed, quantitative answers (with additional figures or tables if possible) for each of these points. Good luck!