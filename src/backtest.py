"""Backtest engine: fit every model on the training period, forecast the distribution of
h-day returns for every test day, compare the forecasts with what really happened.

Works with ANY model that follows models.base.ForecastModel.

Main scores (forecast errors, all in % of log-return):
  RMSE_mean     L2 error of the point forecast: forecast mean vs realised h-day return
  MAE_median    L1 error of the forecast median
  RMSE_vol      L2 error of the volatility forecast: forecast std vs realised volatility
                sqrt(sum of the h squared daily returns)
Secondary (whole distribution):
  CDF_L2sq      squared L2 distance between forecast CDF and realised step = CRPS
  VaR breaches  how often the return fell below the 1% / 5% forecast quantile
"""
import time

import numpy as np
import pandas as pd

import metrics as M
from data import split

DEFAULT_HORIZONS = [1, 5, 21]
DEFAULT_Z = {1: np.linspace(-0.15, 0.15, 601),
             5: np.linspace(-0.30, 0.30, 601),
             21: np.linspace(-0.60, 0.60, 601)}


def realised(returns, n_test, h):
    """For test day t_k: y = sum of the next h daily log-returns, rv = sqrt(sum of their
    squares) = realised volatility over the horizon. NaN where the future is not available."""
    r = np.asarray(returns, dtype=float)
    pos = np.arange(len(r) - n_test, len(r))
    c1, c2 = np.r_[0.0, np.cumsum(r)], np.r_[0.0, np.cumsum(r**2)]
    y, rv = np.full(n_test, np.nan), np.full(n_test, np.nan)
    ok = pos + h < len(r)
    y[ok] = c1[pos[ok] + h + 1] - c1[pos[ok] + 1]
    rv[ok] = np.sqrt(c2[pos[ok] + h + 1] - c2[pos[ok] + 1])
    return y, rv


def check_no_lookahead(model, returns, n_test, horizons, z_grids, F_full, k=150):
    """Forecasts for the first k test days must not change when later data is removed."""
    cut = len(returns) - n_test + k
    F_cut = model.predict_cdf(returns.iloc[:cut], k, horizons, z_grids)
    return max(float(np.abs(F_cut[h] - F_full[h][:k]).max()) for h in horizons)


def run_backtest(models, returns, test_start="2020-01-01", horizons=DEFAULT_HORIZONS,
                 z_grids=DEFAULT_Z, alphas=(0.01, 0.05), lookahead_check=True, verbose=True):
    train, test = split(returns, test_start)
    n_test = len(test)
    rows, details = [], {}
    for model in models:
        t0 = time.time()
        model.fit(train)
        F = model.predict_cdf(returns, n_test, horizons, z_grids)
        leak = check_no_lookahead(model, returns, n_test, horizons, z_grids, F) if lookahead_check else np.nan
        details[model.name] = {"params": model.describe(), "lookahead_diff": leak}
        for h in horizons:
            y, rv = realised(returns, n_test, h)
            ok = ~np.isnan(y)
            z, Fh, yh, rvh = z_grids[h], F[h][ok], y[ok], rv[ok]
            mean, std = M.moments(Fh, z)
            med = M.quantile(Fh, z, 0.5)
            err = dict(mean=mean - yh, median=med - yh, vol=std - rvh, cdf_l2sq=M.crps(Fh, z, yh))
            row = dict(model=model.name, horizon=h, n=int(ok.sum()),
                       RMSE_mean=100 * np.sqrt(np.mean(err["mean"] ** 2)),
                       MAE_median=100 * np.mean(np.abs(err["median"])),
                       RMSE_vol=100 * np.sqrt(np.mean(err["vol"] ** 2)),
                       CDF_L2sq=100 * err["cdf_l2sq"].mean())
            q = {}
            for a in alphas:
                q[a] = M.quantile(Fh, z, a)
                breach = yh < q[a]
                every_h = np.arange(0, len(yh), h)               # non-overlapping for the test
                row[f"VaR{int(a*100)}% breaches"] = breach.mean()
                row[f"VaR{int(a*100)}% Kupiec p"] = M.kupiec_pvalue(breach[every_h].sum(), len(every_h), a)
            rows.append(row)
            details[model.name][h] = dict(dates=test.index[ok], y=yh, rv=rvh, mean=mean, std=std,
                                          errors=err, pit=M.pit(Fh, z, yh), quantiles=q)
        details[model.name]["time_s"] = time.time() - t0
        if verbose:
            print(f"{model.name:16s} done in {time.time() - t0:6.1f}s   look-ahead check: "
                  f"max change {leak:.1e}")
    return pd.DataFrame(rows), details


LOSSES = {"point (mean), squared error": lambda e: e["mean"] ** 2,
          "volatility, squared error": lambda e: e["vol"] ** 2,
          "CDF, squared L2 (CRPS)": lambda e: e["cdf_l2sq"]}


def compare_pairs(details, pairs, horizons=DEFAULT_HORIZONS):
    """Diebold-Mariano test for each (model A, model B) pair and each loss.
    t < 0: A has the smaller error. Also reports the ratio of the losses A / B."""
    out = []
    for a, b in pairs:
        for h in horizons:
            for loss_name, f in LOSSES.items():
                la, lb = f(details[a][h]["errors"]), f(details[b][h]["errors"])
                t, p = M.diebold_mariano(la, lb, h)
                out.append(dict(A=a, B=b, horizon=h, loss=loss_name,
                                ratio_A_over_B=la.mean() / lb.mean(), DM_t=t, p_value=p))
    return pd.DataFrame(out)
