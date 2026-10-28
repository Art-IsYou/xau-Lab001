"""Causal features. Every value at bar t uses only bars <= t (bar t's close is known
when bar t closes; a signal on bar t is traded at the open of bar t+1)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Simple rolling mean of true range, current bar included."""
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def trading_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Day label for a gold 'trading day' that rolls at ~22:00 UTC (the 5pm New York
    roll, ignoring the one-hour DST wobble)."""
    return (index - pd.Timedelta(hours=22)).floor("D")


def h1_trend(df: pd.DataFrame, ema_n: int = 200) -> pd.Series:
    """+1 / -1 = last *completed* H1 close above / below its EMA(ema_n); NaN during warm-up.

    The H1 bar labelled h covers [h, h+1h). After ``shift(1)`` the value attached to
    label h is the one computed through label h-1h, i.e. an hour that has fully closed
    before any M5 bar opening in hour h.
    """
    h1 = df["close"].resample("1h").last().dropna()
    ema = h1.ewm(span=ema_n, adjust=False).mean()
    t = np.sign(h1 - ema).shift(1)
    t.iloc[:ema_n] = np.nan  # EMA warm-up
    return t.reindex(df.index, method="ffill")