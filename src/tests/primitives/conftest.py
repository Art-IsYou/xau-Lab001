import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(rows, start="2024-01-03 09:00", freq="1min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)
    df.index.name = "time"
    return df


def from_path(points, n, wick=0.2, volume=100.0, start="2024-01-03 09:00", freq="1min"):
    """Piecewise-linear close path through (bar, price) key points."""
    xs, ys = zip(*points)
    closes = np.interp(np.arange(n), xs, ys)
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - wick
    rows = list(zip(opens, highs, lows, closes, np.full(n, volume)))
    return make_ohlcv(rows, start, freq)


def from_hl(highs, lows, start="2024-01-03 09:00", freq="1min", volume=100.0):
    mid = [(h + l) / 2 for h, l in zip(highs, lows)]
    rows = [(m, h, l, m, volume) for m, h, l in zip(mid, highs, lows)]
    return make_ohlcv(rows, start, freq)


def mk_swings(items):
    """items: (kind, price, pivot_pos, confirmed_pos)"""
    return pd.DataFrame(
        [
            {"pivot_pos": p, "pivot_time": pd.NaT, "kind": k, "price": pr, "confirmed_pos": c, "confirmed_time": pd.NaT}
            for k, pr, p, c in items
        ]
    )


@pytest.fixture
def random_walk():
    rng = np.random.default_rng(7)
    n = 600
    closes = 2000 + np.cumsum(rng.normal(0, 0.4, n))
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.2, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.2, n))
    vol = rng.integers(50, 200, n).astype(float)
    return make_ohlcv(list(zip(opens, highs, lows, closes, vol)))
