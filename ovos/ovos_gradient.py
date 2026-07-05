"""
ovos_gradient.py

OVOS orbital optimization using gradient descent with a
diminishing step size (no Hessian required). This is a
first‑order method from Convex Optimisation II.
"""

import numpy as np
from ovos import OVOS

class OVOSGradient(OVOS):
    def __init__(self, *args, alpha=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self._grad_norm_history = []   # safe initialisation

    def _newton_step(self, G, H, iteration, start_counting):
        g = G.flatten()
        grad_norm = np.linalg.norm(g)
        if grad_norm < 1e-12:
            return np.zeros_like(g)

        # Diminishing step size: alpha / sqrt(iteration)
        step = self.alpha / np.sqrt(iteration + 1)
        # Limit maximum displacement
        max_step = 5.0
        step = min(step, max_step / (grad_norm + 1e-12))
        return -step * g