"""PDE solvers for the forecast distribution of a regime-switching log-price.

Model: dX_t = m(a_t) dt + sigma(a_t) dW_t,  a_t = Markov chain with generator Q (per year).
F_i(tau, z) = P(X_tau - X_0 <= z | a_0 = i) solves the coupled backward Kolmogorov PDE

    dF_i/dtau = 1/2 sigma_i^2 F_i'' - m_i F_i' + sum_j q_ij F_j,
    F_i(0, z) = 1{z >= 0},  F_i -> 0 (z -> -inf),  F_i -> 1 (z -> +inf).

This is the same coupled system as regime-switching option pricing (Black-Scholes), with
r = 0, the real-world drift instead of r, and a step function instead of the call payoff.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RegimeParams:
    drift: np.ndarray    # m_i, yearly drift of the log-price
    sigma: np.ndarray    # sigma_i, yearly volatility
    Q: np.ndarray        # generator (rows sum to 0), per year

    @property
    def m(self):
        return len(self.sigma)
