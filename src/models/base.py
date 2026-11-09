"""The template every forecasting algorithm follows.

A model forecasts the DISTRIBUTION of the future log-return  y = ln(S_{t+h} / S_t)
for several horizons h (in trading days), as a CDF on a grid of z-values.

To add your own algorithm:
    1. subclass ForecastModel,
    2. implement fit(train_returns) and predict_cdf(returns, n_test, horizons, z_grids),
    3. add an instance to the MODELS list in run_backtest.py.

Rule (checked automatically by the backtester): row k of the forecast for the test day t_k
may use returns up to and including day t_k only - never later returns.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    name = "base"

    @abstractmethod
    def fit(self, train_returns: pd.Series) -> None:
        """Estimate everything from the training period (daily log-returns)."""

    @abstractmethod
    def predict_cdf(self, returns: pd.Series, n_test: int, horizons, z_grids) -> dict:
        """returns   : all daily log-returns, training period followed by the test period;
        n_test       : the last n_test entries are the test days t_1..t_n;
        horizons     : list of horizons in trading days, e.g. [1, 5, 21];
        z_grids      : dict h -> 1-d array of z values (log-return levels).
        Returns dict h -> array (n_test, len(z_grids[h])) with P(y <= z | info up to t_k)."""

    def describe(self) -> dict:
        """Optional: fitted parameters, shown in the results."""
        return {}


def mixture_cdf(probs, cdf_per_regime):
    """Forecast CDF = sum_i P(regime i today) * CDF given regime i.
    probs: (n_test, m); cdf_per_regime: (m, n_z) -> (n_test, n_z)."""
    return np.clip(probs @ cdf_per_regime, 0.0, 1.0)
