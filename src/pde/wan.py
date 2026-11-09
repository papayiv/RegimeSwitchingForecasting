"""Weak Adversarial Network (WAN) for the coupled CDF equation (see pde/__init__.py).

Ansatz (initial and boundary conditions hold exactly):
    F_i(tau, z) = G_i(tau, z) + M(tau, z) * N_i(tau, z),   M = (tau / tau_max) * B(z) * E(tau, z)
    G_i = Phi((z - m_i tau) / (sigma_i sqrt(tau)))  - exact CDF if the regime never switched
    B   = bubble function, 0 at z = +-z_max
    E   = exp(-z^2 / (2 s^2)), s(tau) = k * max(sigma) * sqrt(tau) + s0   - envelope
So the network N learns only the effect of regime switching, and only where the distribution
lives. Without the envelope E the WAN put ~1% of spurious probability far in the tail
(z ~ -0.45 at 21 days): the weak loss hardly sees it, but it inflates the forecast volatility.

Weak form: G_i solves its own single-regime equation exactly, so only the correction
V_i = F_i - G_i and the coupling term remain. With a test function phi_i (0 at z = +-z_max):

    <R_i, phi_i> = integral [ (V_i,tau + m_i V_i,z - sum_j q_ij F_j) phi_i
                              + 1/2 sigma_i^2 V_i,z phi_i,z ] dz dtau

Loss  log( mean_k sum_i <R_i, phi_ik>^2 / sum_i ||phi_ik||^2 );  min over N, max over phi.
Integrals by Gauss-Legendre quadrature (deterministic): nodes clustered near tau = 0 and
around z = 0, where the distribution is concentrated.
"""
import time

import numpy as np
from scipy.stats import norm

from autodiff import Tensor, Adam

DT_DAY = 1 / 252


class MLP:
    """tanh network returning value, d/dtau, d/dz (forward-mode derivatives as tensor ops)."""

    def __init__(self, n_in_scales, n_out, width=32, depth=3, seed=0, out_scale=1.0):
        rng = np.random.default_rng(seed)
        sizes = [2] + [width] * depth + [n_out]
        self.W, self.b = [], []
        for k in range(len(sizes) - 1):
            std = np.sqrt(1.0 / sizes[k]) * (out_scale if k == len(sizes) - 2 else 1.0)
            self.W.append(Tensor(rng.normal(0, std, (sizes[k], sizes[k + 1]))))
            self.b.append(Tensor(np.zeros((1, sizes[k + 1]))))

    @property
    def params(self):
        return self.W + self.b

    def __call__(self, s_tau, s_z, ds_dtau, ds_dz):
        h = Tensor(np.column_stack([s_tau, s_z]))
        z = h @ self.W[0] + self.b[0]
        z_t = self.W[0][0:1, :] * ds_dtau
        z_x = self.W[0][1:2, :] * ds_dz
        for k in range(1, len(self.W)):
            a = z.tanh(); s = 1.0 - a * a
            a_t, a_x = s * z_t, s * z_x
            z = a @ self.W[k] + self.b[k]
            z_t, z_x = a_t @ self.W[k], a_x @ self.W[k]
        return z, z_t, z_x


class WANCdfSolver:
    def __init__(self, params, max_horizon_days=21, z_max=0.6, width=32, depth=3, n_test=8, seed=0,
                 env_k=2.5, env_s0=0.01):
        self.p = params
        self.env_k, self.env_s0 = env_k * float(np.max(params.sigma)), env_s0
        self.tau_max = max_horizon_days * DT_DAY
        self.z_max = z_max
        self.K = n_test
        self.N = MLP(2, params.m, width, depth, seed=seed, out_scale=0.1)
        self.Phi = MLP(2, params.m * n_test, width, depth, seed=seed + 100)
        self.history = []

    # ---- pieces of the ansatz ----------------------------------------------------------
    def _scaled(self, tau, z):
        return 2 * tau / self.tau_max - 1, z / self.z_max, 2 / self.tau_max, 1 / self.z_max

    def _bubble(self, z):
        B = 1 - (z / self.z_max) ** 2
        return B[:, None], (-2 * z / self.z_max**2)[:, None]

    def _G(self, tau, z):
        """Single-regime CDFs (N, m) and their derivatives."""
        t = np.maximum(tau, 1e-10)[:, None]
        s = self.p.sigma[None, :] * np.sqrt(t)
        d = (z[:, None] - self.p.drift[None, :] * t) / s
        pdf = norm.pdf(d) / s
        G = norm.cdf(d)
        G_z = pdf
        G_t = -pdf * (self.p.drift[None, :] + 0.5 * (z[:, None] - self.p.drift[None, :] * t) / t)
        return G, G_t, G_z

    def _mask(self, tau, z):
        """M = (tau/tau_max) * B(z) * E(tau, z) and its derivatives (all known, no network)."""
        B, dB = self._bubble(z)
        t = np.maximum(tau, 1e-10)[:, None]
        zc = z[:, None]
        s = self.env_k * np.sqrt(t) + self.env_s0
        s_t = self.env_k / (2 * np.sqrt(t))
        E = np.exp(-zc**2 / (2 * s**2))
        E_t = E * zc**2 / s**3 * s_t
        E_z = -zc / s**2 * E
        w = t / self.tau_max
        M = w * B * E
        M_t = B * E / self.tau_max + w * B * E_t
        M_z = w * (dB * E + B * E_z)
        return M, M_t, M_z

    def cdf(self, tau, z, as_tensor=False):
        N, N_t, N_z = self.N(*self._scaled(tau, z))
        Mk, Mk_t, Mk_z = self._mask(tau, z)
        V = N * Mk
        V_t = N * Mk_t + N_t * Mk
        V_z = N * Mk_z + N_z * Mk
        G, _, _ = self._G(tau, z)
        if as_tensor:
            return V, V_t, V_z, G
        return np.clip((V + G).value, 0.0, 1.0)

    # ---- quadrature and loss -------------------------------------------------------------
    def quadrature(self, n_tau=24, n_z=24):
        def gl(a, b, n):
            x, w = np.polynomial.legendre.leggauss(n)
            return (a + b) / 2 + (b - a) / 2 * x, (b - a) / 2 * w
        s, ws = gl(0, 1, n_tau)
        t, wt = self.tau_max * s**2, 2 * self.tau_max * s * ws      # clustered near tau = 0
        c = 0.25 * self.z_max                                       # inner panels around z = 0
        zs, wz = zip(*[gl(a, b, n_z) for a, b in
                       [(-self.z_max, -c), (-c, 0), (0, c), (c, self.z_max)]])
        z, wz = np.concatenate(zs), np.concatenate(wz)
        T, Z = np.meshgrid(t, z, indexing="ij")
        return T.ravel(), Z.ravel(), np.outer(wt, wz).ravel()

    def loss(self, tau, z, w):
        p, m = self.p, self.p.m
        V, V_t, V_z, G = self.cdf(tau, z, as_tensor=True)
        F = V + G
        coupling = F @ Tensor(p.Q.T)                              # sum_j q_ij F_j
        A = V_t + V_z * p.drift[None, :] - coupling               # multiplies phi
        D = V_z * (0.5 * p.sigma**2)[None, :]                     # multiplies phi_z
        P, _, P_z = self.Phi(*self._scaled(tau, z))
        B, dB = self._bubble(z)
        W = Tensor(w[:, None])
        total = None
        for k in range(self.K):
            cols = slice(k * m, (k + 1) * m)
            phi = P[:, cols] * B
            phi_z = P[:, cols] * dB + P_z[:, cols] * B
            weak = ((A * phi + D * phi_z) * W).sum(axis=0)
            nrm = (phi * phi * W).sum(axis=0)
            ratio = (weak * weak).sum() / nrm.sum()
            total = ratio if total is None else total + ratio
        return (total * (1.0 / self.K)).log()

    def train(self, n_iter=3000, lr=1e-3, eval_fn=None, eval_every=500, verbose=True):
        opt_n, opt_phi = Adam(self.N.params, lr=lr), Adam(self.Phi.params, lr=lr)
        tau, z, w = self.quadrature()
        t0 = time.time()
        for it in range(1, n_iter + 1):
            frac = it / n_iter
            decay = 0.1 if frac > 0.8 else 0.3 if frac > 0.5 else 1.0
            opt_n.lr = opt_phi.lr = lr * decay                    # decay both players together
            L = self.loss(tau, z, w); L.backward(); opt_phi.step(maximize=True)
            L = self.loss(tau, z, w); L.backward(); opt_n.step()
            if eval_fn is not None and (it % eval_every == 0 or it == 1):
                err = eval_fn(self)
                self.history.append((it, float(L.value), err))
                if verbose:
                    print(f"iter {it:5d}  loss {float(L.value):8.3f}  error {err:.5f}  "
                          f"({time.time() - t0:4.0f}s)")
        self.train_time = time.time() - t0
        return self

    def solve(self, horizons, z):
        """Same output format as solve_cdf_fd: dict h -> (m, len(z))."""
        out = {}
        for h in horizons:
            tau = np.full_like(z, h * DT_DAY)
            zc = np.clip(z, -self.z_max, self.z_max)
            F = self.cdf(tau, zc).T
            F[:, z <= -self.z_max] = 0.0
            F[:, z >= self.z_max] = 1.0
            # monotone rearrangement (Chernozhukov et al. 2009): sort the CDF values along z.
            # A network CDF can wiggle slightly; sorting makes it a valid CDF and never
            # increases its L^p error against the true (monotone) CDF.
            out[h] = np.sort(F, axis=1)
        return out
