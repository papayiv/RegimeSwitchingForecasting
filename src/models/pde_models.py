"""PDE-based forecasting model.

RegimeSwitchingPDE(n_regimes, solver)
    n_regimes = 1 : one equation; drift and volatility from ALL training returns
    n_regimes = 2 : calm/crisis regimes from the HMM; coupled system
    solver = "fd" (finite differences) or "wan" (Weak Adversarial Network)
The PDE is solved once; every test day the forecast mixes the regime CDFs with that day's
filtered regime probabilities (for 1 regime the probability is always 1).
"""
import numpy as np

from .base import ForecastModel, mixture_cdf
import hmm
from pde import RegimeParams
from pde.fd import solve_cdf_fd

DT = 1 / 252


def _on_grid(z_solver, F_solver, z_target):
    """Interpolate solver output (m, nz) onto the evaluation grid."""
    return np.array([np.interp(z_target, z_solver, Fi, left=0.0, right=1.0) for Fi in F_solver])


class RegimeSwitchingPDE(ForecastModel):
    def __init__(self, n_regimes=2, solver="fd", wan_kwargs=None, wan_iters=3000):
        self.m, self.solver = n_regimes, solver
        self.wan_kwargs, self.wan_iters = wan_kwargs or {}, wan_iters
        word = "regime" if n_regimes == 1 else "regimes"
        self.name = f"{n_regimes} {word}, {solver.upper()}"

    def fit(self, train_returns):
        x = np.asarray(train_returns, dtype=float)
        self.hmm = hmm.fit_gaussian_hmm(x, m=self.m)
        drift, sigma, Q = hmm.to_continuous(self.hmm)
        self.params = RegimeParams(drift=drift, sigma=sigma, Q=Q)
        self.n_train = len(x)
        self.wan = None

    def _solve(self, horizons):
        if self.solver == "fd":
            return solve_cdf_fd(self.params, horizons)
        if self.solver == "wan":
            from pde.wan import WANCdfSolver
            if self.wan is None:
                self.wan = WANCdfSolver(self.params, max_horizon_days=max(horizons), **self.wan_kwargs)
                self.wan.train(n_iter=self.wan_iters, verbose=False)
            z = np.linspace(-1, 1, 4001)
            return z, self.wan.solve(horizons, z)
        raise ValueError(self.solver)

    def regime_probs(self, returns, n_test):
        """P(regime on test day t_k | returns up to t_k), parameters fixed from training."""
        r = np.asarray(returns, dtype=float)
        p0 = hmm.filter_probs(r[:-n_test], self.hmm)[-1]       # end of the data before the test
        return hmm.filter_probs(r[-n_test:], self.hmm, p0=p0)

    def predict_cdf(self, returns, n_test, horizons, z_grids):
        z, sol = self._solve(horizons)
        probs = self.regime_probs(returns, n_test)
        return {h: mixture_cdf(probs, _on_grid(z, sol[h], z_grids[h])) for h in horizons}

    def describe(self):
        d = dict(drift=self.params.drift.tolist(), sigma=self.params.sigma.tolist(),
                 Q=self.params.Q.tolist())
        if self.wan is not None:
            d["wan_train_time_s"] = self.wan.train_time
        return d
