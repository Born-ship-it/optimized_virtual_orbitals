"""
ovos_trust_region.py

OVOS orbital optimization using a trust‑region Newton step.
The step is obtained by solving (H + μI) r = -g with μ chosen
so that ||r||_2 ≤ trust_radius. This is a standard regularised
Newton method (Levenberg‑Marquardt) and is a convex approach
(as taught in Convex Optimisation I & II).
"""

import numpy as np
import scipy.linalg
from ovos import OVOS  # original base class

class OVOSTrustRegion(OVOS):
    def __init__(self, *args, trust_radius=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.trust_radius = trust_radius
        # Ensure the attribute is initialised (also done in base, but safe)
        self._grad_norm_history = []

    def _newton_step(self, G, H, iteration, start_counting):
        g = G.flatten()
        n = len(g)
        grad_norm = np.linalg.norm(g)
        if grad_norm < 1e-12:
            return np.zeros(n)

        # Regularise H to be positive definite: H + μ I
        # Find μ such that ||r|| <= trust_radius
        # We'll do a simple bisection on μ.
        # First, get eigenvalues to estimate μ_min
        eigvals = np.linalg.eigvalsh(H)
        mu_min = max(0.0, -np.min(eigvals) + 1e-8)  # shift to PD

        # Function to solve for r and its norm
        def solve_for_mu(mu):
            H_reg = H + mu * np.eye(n)
            try:
                r = np.linalg.solve(H_reg, -g)
                return r, np.linalg.norm(r)
            except np.linalg.LinAlgError:
                return None, np.inf

        # Initial μ
        mu = mu_min
        r, rnorm = solve_for_mu(mu)
        if r is None:
            # Fallback: simple gradient step
            return -g / grad_norm * self.trust_radius * 0.1

        if rnorm <= self.trust_radius:
            return r

        # Increase μ until norm <= trust_radius
        mu_high = mu_min + 1.0
        # double until norm <= trust_radius
        for _ in range(20):
            r, rnorm = solve_for_mu(mu_high)
            if r is None:
                mu_high *= 2.0
                continue
            if rnorm <= self.trust_radius:
                break
            mu_high *= 2.0

        # Now bisection between mu and mu_high
        mu_low = mu
        for _ in range(30):
            mu_mid = (mu_low + mu_high) / 2.0
            r, rnorm = solve_for_mu(mu_mid)
            if r is None:
                mu_low = mu_mid
                continue
            if rnorm < self.trust_radius:
                mu_high = mu_mid
            else:
                mu_low = mu_mid
            if abs(rnorm - self.trust_radius) < 1e-6 * self.trust_radius:
                break
        # Final solve with mu_high
        r, _ = solve_for_mu(mu_high)
        if r is None:
            # fallback
            return -g / grad_norm * self.trust_radius * 0.1
        return r