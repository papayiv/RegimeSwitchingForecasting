"""Load the S&P 500 series and split it into a training and a test period."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def load_returns(path=DATA / "sp500.csv"):
    """Daily log-returns of the S&P 500 as a pd.Series indexed by date."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    r = np.log(df["close"]).diff()
    r.index = df["date"]
    return r.dropna().rename("logret")


def load_prices(path=DATA / "sp500.csv"):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    return df.set_index("date")["close"]


def split(series, test_start="2020-01-01"):
    """Train = everything before test_start, test = from test_start on."""
    return series[series.index < test_start], series[series.index >= test_start]
