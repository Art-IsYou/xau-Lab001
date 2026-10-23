"""Candle anatomy, ATR, and the break-Out candle qualification test.

All functions are row-aligned with ``df`` and use only data up to and
including each bar (``range_in_atr`` deliberately uses the ATR of the *prior*
bars so the candle being judged does not inflate its own yardstick).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._types import LONG, SHORT, check_direction


def candle_parts(df: pd.DataFrame) -> pd.DataFrame:
    """range, body, body_ratio (body/range, NaN on zero-range), wicks, direction."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l
    body = (c - o).abs()
    out = pd.DataFrame(index=df.index)
    out["range"] = rng
    out["body"] = body
    out["body_ratio"] = (body / rng).where(rng > 0)
    out["upper_wick"] = h - np.maximum(o, c)
    out["lower_wick"] = np.minimum(o, c) - l
    out["direction"] = np.sign(c - o).astype(int)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR. NaN until ``period`` bars are available."""
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def range_in_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Candle range as a multiple of the ATR *before* this candle.

    Use it to see how stretched a break candle is. The authoritative
    exhaustion check is still the R:R gate (a huge candle => wide stop =>
    R:R collapses), this is the cheap pre-filter / diagnostic.
    """
    return (df["high"] - df["low"]) / atr(df, period).shift(1)


def body_beyond_fraction(df: pd.DataFrame, level, direction: int) -> pd.Series:
    """Share (0..1) of the candle body that lies beyond ``level``."""
    check_direction(direction)
    o, c = df["open"], df["close"]
    body = (c - o).abs()
    if direction == LONG:
        beyond = (c - np.maximum(o, level)).clip(lower=0)
    else:
        beyond = (np.minimum(o, level) - c).clip(lower=0)
    return (beyond / body).where(body > 0)


def breakout_candle_mask(
    df: pd.DataFrame,
    level,
    direction: int,
    min_body_ratio: float = 0.5,
    min_beyond_frac: float = 0.5,
) -> pd.Series:
    """Which bars qualify as a valid break-Out candle through ``level``.

    LONG : opened at/below level, closed above it, bullish, body >= min_body_ratio
           of the candle range.
    SHORT: mirror image.
    ``min_beyond_frac`` optionally demands that at least that share of the
    *body* sits beyond the level (0.0 = off). ``level`` may be a scalar or a
    Series aligned with ``df``.
    """
    check_direction(direction)
    lvl = level if isinstance(level, pd.Series) else pd.Series(level, index=df.index)
    parts = candle_parts(df)
    body_ok = parts["body_ratio"] >= min_body_ratio
    beyond_ok = body_beyond_fraction(df, lvl, direction) >= min_beyond_frac
    if direction == LONG:
        side = (df["open"] <= lvl) & (df["close"] > lvl) & (parts["direction"] == 1)
    else:
        side = (df["open"] >= lvl) & (df["close"] < lvl) & (parts["direction"] == -1)
    return (side & body_ok & beyond_ok).fillna(False).astype(bool)
