"""Example of plugging a NEW algorithm into the template: RiskMetrics EWMA volatility with
a normal distribution. A standard industry benchmark, and only ~20 lines."""
import numpy as np
from scipy.stats import norm

from .base import ForecastModel


class EWMANormal(ForecastModel):
    def __init__(self, lam=0.94):
        self.lam = lam
        self.name = f"EWMA normal (lambda={lam})"

    def fit(self, train_returns):
        x = np.asarray(train_returns, dtype=float)
        self.mean = x.mean()                       # daily mean, from training only
        self.var0 = x.var()

    def predict_cdf(self, returns, n_test, horizons, z_grids):
        r = np.asarray(returns, dtype=float)
        var = self.var0
        v = np.empty(len(r))
        for t, x in enumerate(r):                  # v[t] uses returns up to and including t
            var = self.lam * var + (1 - self.lam) * (x - self.mean) ** 2
            v[t] = var
        v = v[-n_test:]
        return {h: norm.cdf((z_grids[h][None, :] - h * self.mean) / np.sqrt(h * v)[:, None])
                for h in horizons}

    def describe(self):
        return dict(lam=self.lam, daily_mean=self.mean)
