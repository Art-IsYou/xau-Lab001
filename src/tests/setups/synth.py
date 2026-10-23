"""Hand-built price paths for setup tests.

``long_breakout_scenario`` is a textbook LONG break-Out on M1 bars:

  apex0 (2008.2)  -> drop -> apex1 (2000.2) -> drop -> apex2 (2000.2)  [2-touch resistance]
  -> pullback low -> bounce -> wick below that low that closes back above (sell-side sweep)
  -> consolidation under the level -> high-volume break candle -> candle-over-candle trigger bar

Everything is deterministic. ``mirror`` flips it into the SHORT equivalent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

W = 0.2  # wick added above/below each synthetic bar


def _seg(prev: float, n: int, target: float, vol: float = 100.0):
    out = []
    for i in range(1, n + 1):
        c = prev + (target - prev) * i / n
        o = prev if i == 1 else out[-1][3]
        out.append((o, max(o, c) + W, min(o, c) - W, c, vol))
    return out


def long_breakout_scenario(start="2024-01-02 09:00") -> tuple[pd.DataFrame, dict]:
    bars: list[tuple] = []

    def go(n, target, vol=100.0):
        prev = bars[-1][3] if bars else 2004.0
        bars.extend(_seg(prev, n, target, vol))

    go(12, 2008.0)  # apex0: a lone swing high far above, the future target
    go(14, 1990.0)
    go(10, 2000.0)  # apex1 -> high 2000.2
    go(8, 1994.0)
    go(8, 2000.0)  # apex2 -> high 2000.2 (2nd touch)
    go(8, 1996.5)  # pullback low
    go(4, 1998.5)  # bounce so the low is confirmed before it is swept
    sweep_pos = len(bars)
    bars.append((1998.4, 1998.6, 1995.9, 1998.0, 100.0))  # wick through 1996.3, closes back above
    go(3, 1999.0)
    go(3, 1998.6)
    go(2, 2000.0)  # approach, still closing under the level
    break_pos = len(bars)
    bars.append((2000.0, 2001.5, 1999.95, 2001.4, 350.0))  # the break candle
    bars.append((2001.4, 2002.0, 2001.2, 1999.9 + 2.0, 110.0))  # CoC trigger bar (high >= 2001.5)
    for b in _seg(2001.9, 5, 2004.0):
        bars.append(b)

    idx = pd.date_range(start, periods=len(bars), freq="1min", tz="UTC", name="time")
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close", "volume"], index=idx)
    meta = {"break_pos": break_pos, "sweep_pos": sweep_pos, "level": 2000.2, "target": 2008.2,
            "entry": 2001.5, "stop": 1999.95}
    return df, meta


def mirror(df: pd.DataFrame, c: float = 4000.0) -> pd.DataFrame:
    out = df.copy()
    out["open"], out["close"] = c - df["open"], c - df["close"]
    out["high"], out["low"] = c - df["low"], c - df["high"]
    return out


def trend_series(df: pd.DataFrame, value: int) -> pd.Series:
    return pd.Series(value, index=df.index)


def random_walk(n=6000, seed=7, start="2024-01-02 00:00") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step = rng.normal(0, 0.45, n)
    close = 2000 + np.cumsum(step)
    open_ = np.r_[2000.0, close[:-1]]
    wick = np.abs(rng.normal(0, 0.25, (n, 2)))
    high = np.maximum(open_, close) + wick[:, 0]
    low = np.minimum(open_, close) - wick[:, 1]
    vol = rng.lognormal(4.6, 0.5, n).round()
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC", name="time")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)