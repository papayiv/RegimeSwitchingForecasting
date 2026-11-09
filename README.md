# Project "Machine Learning for Regime-Switching Financial Models"

[TODO]здесь будет описание...

## Backtest template

Question: does a 2-regime model (calm/crisis from an HMM) forecast the S&P 500 better than one equation fitted on all data? Train on 2000–2019, test on 2020–2026, horizons 1, 5, 21 trading days.

Comparison pair by pair, same solver inside each pair: **2 regimes FD vs 1 regime FD** and **2 regimes WAN vs 1 regime WAN**.
Main scores are forecast errors (in % of log-return): RMSE of the point forecast (forecast mean vs realised return), MAE of the forecast median, RMSE of the volatility forecast (forecast std vs realised volatility). Secondary: squared L2 distance between forecast CDF and realised outcome (= CRPS), VaR breaches. Pairs are tested with the Diebold-Mariano test.

```
RegimeSwitchingForecasting/
├── data/                       sp500.csv, dtb3.csv, spy_options_*.csv
├── src/
│   ├── data.py                 load returns, train/test split
│   ├── hmm.py                  Gaussian HMM: fit regimes, filter regime probabilities in real time
│   ├── pde/
│   │   ├── __init__.py         the PDE + RegimeParams
│   │   ├── fd.py               finite-difference solver (reference)
│   │   └── wan.py              Weak Adversarial Network solver
│   ├── autodiff.py             tiny NumPy autodiff used by the WAN (no PyTorch needed yet)
│   ├── models/
│   │   ├── base.py             THE TEMPLATE: ForecastModel (fit / predict_cdf)
│   │   ├── pde_models.py       RegimeSwitchingPDE(n_regimes = 1 or 2, solver = "fd" or "wan")
│   │   ├── historical.py       extra baseline (not in the main comparison)
│   │   └── ewma.py             template example of a plug-in model (not in the comparison)
│   ├── metrics.py              moments of the forecast, CRPS, VaR, Kupiec, Diebold-Mariano
│   └── backtest.py             runs any list of models, forecast errors, pair tests, look-ahead check
├── run_backtest.py             command-line run: python run_backtest.py [--no-wan]
├── notebooks/backtest.ipynb    executed notebook with results and plots
└── results/                    metrics.csv, dm_tests.csv, params.json, figs/
```

### The model

Log-price `dX = m(a_t) dt + sigma(a_t) dW`, regime `a_t` = Markov chain with generator Q. The forecast CDF `F_i(tau, z) = P(X_tau - X_0 <= z | regime i)` solves

    dF_i/dtau = 1/2 sigma_i^2 F_i'' - m_i F_i' + sum_j q_ij F_j,   F_i(0, z) = 1{z >= 0}

which is the same coupled system as regime-switching option pricing. It is solved once (FD or WAN). On each test day the forecast is `sum_i P(regime i today) * F_i(h, z)`, where the regime probability is filtered from returns up to that day. Parameters are frozen at the end of 2019.

### Add your own algorithm

1. Subclass `ForecastModel` in `src/models/base.py` and implement `fit` and `predict_cdf`.
2. Add it to `make_models()` in `run_backtest.py` (or pass it to `run_backtest` in the notebook) and say which pair to test in `compare_pairs`.
3. Row k of the forecast may only use returns up to test day k. `backtest.py` checks this and prints the "look-ahead check" (must be 0).

### Run

```bash
pip install -r requirements.txt
python run_backtest.py --no-wan     # FD pair only, seconds
python run_backtest.py              # FD and WAN pairs, ~7 minutes (two WAN trainings)
```
