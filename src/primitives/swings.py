"""Swing highs / lows (fractal pivots) with explicit confirmation time.

A pivot at bar ``i`` needs ``right`` bars after it to exist, so it is only
*known* at bar ``i + right``. Every swing therefore carries ``confirmed_pos``.
Rule of thumb for callers: when judging bar ``t``, a swing may be used only if
``confirmed_pos <= t`` (and ``<= t - 1`` if it must pre-date candle ``t``).

Ties: pivot must be strictly beyond the ``left`` bars and at-least-equal to the
``right`` bars, so a run of equal highs yields one pivot (the first).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SWING_COLUMNS = ["pivot_pos", "pivot_time", "kind", "price", "confirmed_pos", "confirmed_time"]


def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """One row per swing: kind is 'high' or 'low'."""
    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")
    h, l = df["high"], df["low"]
    left_max = h.shift(1).rolling(left).max()
    right_max = h[::-1].shift(1).rolling(right).max()[::-1]
    left_min = l.shift(1).rolling(left).min()
    right_min = l[::-1].shift(1).rolling(right).min()[::-1]
    is_high = ((h > left_max) & (h >= right_max)).to_numpy()
    is_low = ((l < left_min) & (l <= right_min)).to_numpy()

    idx = df.index
    parts = []
    for mask, kind, prices in ((is_high, "high", h), (is_low, "low", l)):
        pos = np.flatnonzero(mask)
        parts.append(
            pd.DataFrame(
                {
                    "pivot_pos": pos,
                    "pivot_time": idx[pos],
                    "kind": kind,
                    "price": prices.to_numpy()[pos],
                    "confirmed_pos": pos + right,
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["confirmed_pos", "kind"]).reset_index(drop=True)
    out["confirmed_time"] = idx[out["confirmed_pos"].to_numpy()]
    return out[SWING_COLUMNS]


def swings_confirmed_by(swings: pd.DataFrame, pos: int) -> pd.DataFrame:
    """Swings that are already known at the close of bar ``pos``."""
    return swings[swings["confirmed_pos"] <= pos]


def last_swing_series(df: pd.DataFrame, swings: pd.DataFrame, kind: str) -> pd.Series:
    """Price of the most recent *confirmed* swing of ``kind`` at every bar."""
    if kind not in ("high", "low"):
        raise ValueError("kind must be 'high' or 'low'")
    arr = np.full(len(df), np.nan)
    sel = swings[swings["kind"] == kind]
    arr[sel["confirmed_pos"].to_numpy()] = sel["price"].to_numpy()
    return pd.Series(arr, index=df.index, name=f"last_swing_{kind}").ffill()
