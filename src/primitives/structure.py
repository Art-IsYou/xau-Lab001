"""Higher-timeframe market structure (the 'M15 aligned' gate).

Trend per bar from confirmed swings:
  +1  last swing high is higher than the one before AND last swing low is higher
  -1  both lower
   0  anything else (mixed / range / not enough swings)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .swings import find_swings


def market_structure(
    df: pd.DataFrame, left: int = 2, right: int = 2, swings: pd.DataFrame | None = None
) -> pd.Series:
    """+1 / 0 / -1 at each bar of ``df``, using swings confirmed by that bar."""
    swings = find_swings(df, left, right) if swings is None else swings
    n = len(df)
    signs = {}
    for kind in ("high", "low"):
        sel = swings[swings["kind"] == kind].sort_values("confirmed_pos")
        arr = np.full(n, np.nan)
        if len(sel) > 1:
            arr[sel["confirmed_pos"].to_numpy()[1:]] = np.sign(np.diff(sel["price"].to_numpy()))
        signs[kind] = pd.Series(arr, index=df.index).ffill()
    up = (signs["high"] == 1) & (signs["low"] == 1)
    down = (signs["high"] == -1) & (signs["low"] == -1)
    return pd.Series(np.select([up, down], [1, -1], default=0), index=df.index, name="structure")


def align_htf_to_base(htf_values: pd.Series, base_index: pd.Index, htf_period) -> pd.Series:
    """Project an HTF series onto the base-timeframe index without lookahead.

    An HTF bar stamped at its open time ``T`` is only known at ``T + period``.
    Base bar ``t`` therefore sees the latest HTF value with ``T + period <= t``.
    Base bars before the first available HTF value get NaN.
    """
    s = htf_values.copy()
    s.index = s.index + pd.Timedelta(htf_period)
    return s.reindex(base_index, method="ffill")


def structure_aligned(trend: pd.Series, direction: int) -> pd.Series:
    """True where HTF trend equals the trade direction (NaN / neutral => False)."""
    return (trend == direction).fillna(False).astype(bool)
