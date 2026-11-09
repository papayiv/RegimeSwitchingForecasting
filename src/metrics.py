"""Scores for forecasts given as CDFs on a grid.

Forecast errors (main scores):
moments     - mean and standard deviation of the forecast distribution (from the CDF)
point / volatility errors are computed in backtest.py with these
Distribution scores:

crps        - Continuous Ranked Probability Score, lower is better (same units as returns)
log_score   - mean log predictive density at the realised return, higher is better
var_breaches- how often the realised return falls below the alpha-quantile (should be alpha)
kupiec_pvalue - test whether the breach rate equals alpha (small p = rejected)
pit         - F(realised); uniform on [0,1] for a well-calibrated forecast
"""
import numpy as np
from scipy.stats import chi2

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed trapz


def moments(F, z):
    """Mean and standard deviation of each forecast distribution F (n, nz) on grid z.
    Probability mass below z[0] / above z[-1] is put at the grid ends."""
    zm = np.r_[z[0], 0.5 * (z[1:] + z[:-1]), z[-1]]
    p = np.concatenate([F[:, :1], np.diff(F, axis=1), 1 - F[:, -1:]], axis=1)
    mean = p @ zm
    var = p @ zm**2 - mean**2
    return mean, np.sqrt(np.maximum(var, 0))


def crps(F, z, y):
    """F: (n, nz) CDFs on grid z; y: (n,) realised values. Integral of (F - 1{z>=y})^2,
    i.e. the SQUARED L2 DISTANCE between the forecast CDF and the realised step function."""
    H = (z[None, :] >= y[:, None]).astype(float)
    return _trapz((F - H) ** 2, z, axis=1)


def pit(F, z, y):
    return np.array([np.interp(yi, z, Fi) for Fi, yi in zip(F, y)])


def log_score(F, z, y, floor=1e-4):
    """log density at y; density from finite differences of the CDF."""
    f = np.gradient(F, z, axis=1)
    dens = np.array([np.interp(yi, z, fi) for fi, yi in zip(f, y)])
    return np.log(np.maximum(dens, floor))


def quantile(F, z, alpha):
    """alpha-quantile of each forecast (inverse CDF by interpolation)."""
    out = np.empty(len(F))
    for k, Fi in enumerate(F):
        Fm = np.maximum.accumulate(Fi)                       # make monotone for interpolation
        idx = np.searchsorted(Fm, alpha)
        idx = np.clip(idx, 1, len(z) - 1)
        f0, f1 = Fm[idx - 1], Fm[idx]
        out[k] = z[idx - 1] + (alpha - f0) / max(f1 - f0, 1e-12) * (z[idx] - z[idx - 1])
    return out


def kupiec_pvalue(n_breach, n, alpha):
    """Kupiec proportion-of-failures likelihood-ratio test."""
    if n == 0:
        return np.nan
    p_hat = n_breach / n
    def ll(p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return (n - n_breach) * np.log(1 - p) + n_breach * np.log(p)
    lr = -2 * (ll(alpha) - ll(p_hat))
    return float(1 - chi2.cdf(lr, df=1))


def diebold_mariano(loss_a, loss_b, h=1):
    """Is model A's average loss different from model B's? Returns (t-statistic, p-value).
    Negative t = A has lower loss. Newey-West variance with h-1 lags (overlapping forecasts)."""
    from scipy.stats import norm
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d); dc = d - d.mean()
    var = dc @ dc / n
    for lag in range(1, h):
        var += 2 * (1 - lag / h) * (dc[lag:] @ dc[:-lag]) / n
    t = d.mean() / np.sqrt(var / n)
    return float(t), float(2 * (1 - norm.cdf(abs(t))))
