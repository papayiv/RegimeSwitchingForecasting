"""Baseline: historical simulation. The forecast is the empirical distribution of all
h-day returns in the training period, the same every day (ignores regimes)."""
import numpy as np

from .base import ForecastModel


class HistoricalModel(ForecastModel):
    name = "Historical (empirical)"

    def fit(self, train_returns):
        self.train = np.asarray(train_returns, dtype=float)

    def predict_cdf(self, returns, n_test, horizons, z_grids):
        out = {}
        for h in horizons:
            sums = np.convolve(self.train, np.ones(h), mode="valid")        # h-day log-returns
            F = np.searchsorted(np.sort(sums), z_grids[h], side="right") / len(sums)
            out[h] = np.tile(F, (n_test, 1))
        return out
