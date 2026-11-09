"""Run the whole backtest from the command line:

    python run_backtest.py            # FD and WAN models (WAN trains ~3 min per model)
    python run_backtest.py --no-wan   # fast run, FD models only

Comparison: 1 regime vs 2 regimes, separately for each solver (FD and WAN).
Results: results/metrics.csv, results/dm_tests.csv, results/params.json
To test your own algorithm: write a ForecastModel subclass (src/models/base.py)
and add it to make_models() and PAIRS below.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from backtest import run_backtest, compare_pairs
from data import load_returns
from models import RegimeSwitchingPDE

TEST_START = "2020-01-01"


def make_models(with_wan=True):
    solvers = ["fd", "wan"] if with_wan else ["fd"]
    return [RegimeSwitchingPDE(n_regimes=m, solver=s) for s in solvers for m in (1, 2)]


def pairs_for(models):
    names = [m.name for m in models]
    return [(f"2 regimes, {s}", f"1 regime, {s}") for s in ("FD", "WAN")
            if f"2 regimes, {s}" in names]


def main(with_wan=True):
    returns = load_returns()
    models = make_models(with_wan)
    table, details = run_backtest(models, returns, test_start=TEST_START)
    dm = compare_pairs(details, pairs_for(models))
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    table.to_csv(out / "metrics.csv", index=False)
    dm.to_csv(out / "dm_tests.csv", index=False)
    params = {k: {"params": v["params"], "lookahead_diff": v["lookahead_diff"]} for k, v in details.items()}
    (out / "params.json").write_text(json.dumps(params, indent=2))
    print(table.round(4).to_string(index=False))
    print(dm.round(3).to_string(index=False))
    return table, details, dm


if __name__ == "__main__":
    main(with_wan="--no-wan" not in sys.argv)
