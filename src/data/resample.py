"""resample.py - build higher timeframes from clean M1 bars.

Conventions (these prevent look-ahead bias):
  * A bar's timestamp is its OPEN time. A 5-minute bar stamped 10:05 covers
    10:05:00-10:09:59 and is only *known* at 10:10:00. Backtests must not use
    it before then.
  * Empty periods (weekend, daily break) produce NO bar - never filled.
  * By default the last bar is dropped if the data ends before it completed.
  * D1 follows the forex/gold trading day: it rolls at 17:00 New York time.
    All other timeframes align to UTC clock boundaries.
"""
from __future__ import annotations

import pandas as pd

_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
_NY = "America/New_York"
_D1_SHIFT = pd.Timedelta(hours=7)   # 17:00 NY + 7h = midnight

SUPPORTED = list(_MINUTES)

def _agg(df: pd.DataFrame) -> dict:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    if "spread" in df.columns:
        agg["spread"] = "mean"
    return agg

def resample(df: pd.DataFrame, timeframe: str, drop_incomplete_last: bool = True) -> pd.DataFrame:
    """Resample clean M1 bars to `timeframe` (M5, M15, M30, H1, H4, D1)."""
    tf = timeframe.upper()
    if tf not in _MINUTES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Use one of {SUPPORTED}")
    if tf == "M1":
        return df.copy()
    if df.empty:
        return df.copy()

    agg = _agg(df)
    if tf == "D1":
        shifted = df.tz_convert(_NY)
        shifted.index = shifted.index + _D1_SHIFT
        out = shifted.resample("1D", label="left", closed="left").agg(agg)
        out.index = (out.index - _D1_SHIFT).tz_convert("UTC")
    else:
        out = df.resample(f"{_MINUTES[tf]}min", label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open"])
    out.index.name = "time"

    if drop_incomplete_last and len(out):
        # the last M1 bar must be the final minute of the last higher-TF bar
        last_bar_open = out.index[-1]
        last_m1 = df.index[-1]
        if tf == "D1":
            # D1 can't be judged by a fixed length around DST; require next-day cutoff
            end = last_bar_open + pd.Timedelta(days=1)
        else:
            end = last_bar_open + pd.Timedelta(minutes=_MINUTES[tf])
        if last_m1 + pd.Timedelta(minutes=1) < end:
            out = out.iloc[:-1]
    return out