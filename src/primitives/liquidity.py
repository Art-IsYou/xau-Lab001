"""Liquidity sweeps: a confirmed swing point is pierced by a wick and price
closes back on the original side.

side = 'sell_side' -> a swing LOW was swept (stops under lows taken). Fuel for a LONG.
side = 'buy_side'  -> a swing HIGH was swept. Fuel for a SHORT.

Each swing can be consumed only once: after it is pierced (swept, or closed
through = a break) it leaves the pool. A swing is only sweepable from the bar
*after* it was confirmed, so there is no lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._types import LONG, check_direction

SWEEP_COLUMNS = ["sweep_pos", "sweep_time", "side", "swept_price", "extreme", "swing_pos"]


def find_sweeps(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    reclaim_bars: int = 1,
    max_age_bars: int | None = 500,
) -> pd.DataFrame:
    """Detect sweeps.

    reclaim_bars: bars allowed (including the piercing bar) for price to close
    back across the swing. 1 = same candle only. ``sweep_pos`` is the bar on
    which the reclaim close happened (the moment the sweep is knowable).
    max_age_bars: swings older than this (from pivot) leave the pool.
    """
    if reclaim_bars < 1:
        raise ValueError("reclaim_bars must be >= 1")
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)

    by_conf: dict[int, list] = {}
    for r in swings.itertuples(index=False):
        by_conf.setdefault(int(r.confirmed_pos), []).append((r.kind, float(r.price), int(r.pivot_pos)))

    lows: list[list] = []  # [price, pivot_pos]
    highs: list[list] = []
    pend_lows: list[list] = []  # [price, pivot_pos, deadline_pos, extreme]
    pend_highs: list[list] = []
    lmax, hmin = -np.inf, np.inf  # highest active swing low / lowest active swing high
    events: list[tuple] = []

    for i in range(n):
        # 1) pending (pierced earlier, waiting for a reclaim close)
        if pend_lows:
            keep = []
            for p in pend_lows:
                p[3] = min(p[3], l[i])
                if c[i] > p[0]:
                    events.append((i, "sell_side", p[0], p[3], p[1]))
                elif i < p[2]:
                    keep.append(p)
            pend_lows = keep
        if pend_highs:
            keep = []
            for p in pend_highs:
                p[3] = max(p[3], h[i])
                if c[i] < p[0]:
                    events.append((i, "buy_side", p[0], p[3], p[1]))
                elif i < p[2]:
                    keep.append(p)
            pend_highs = keep

        # 2) fresh pierces of active swings
        if lows and l[i] < lmax:
            keep = []
            for s in lows:
                if l[i] < s[0]:
                    if c[i] > s[0]:
                        events.append((i, "sell_side", s[0], l[i], s[1]))
                    elif reclaim_bars > 1:
                        pend_lows.append([s[0], s[1], i + reclaim_bars - 1, l[i]])
                else:
                    keep.append(s)
            lows = keep
            lmax = max((s[0] for s in lows), default=-np.inf)
        if highs and h[i] > hmin:
            keep = []
            for s in highs:
                if h[i] > s[0]:
                    if c[i] < s[0]:
                        events.append((i, "buy_side", s[0], h[i], s[1]))
                    elif reclaim_bars > 1:
                        pend_highs.append([s[0], s[1], i + reclaim_bars - 1, h[i]])
                else:
                    keep.append(s)
            highs = keep
            hmin = min((s[0] for s in highs), default=np.inf)

        # 3) swings that become known at the close of bar i are sweepable from i+1
        for kind, price, pivot in by_conf.get(i, ()):
            if kind == "low":
                lows.append([price, pivot])
                lmax = max(lmax, price)
            else:
                highs.append([price, pivot])
                hmin = min(hmin, price)

        # 4) age pruning
        if max_age_bars is not None and i % 64 == 0:
            lows = [s for s in lows if i - s[1] <= max_age_bars]
            highs = [s for s in highs if i - s[1] <= max_age_bars]
            lmax = max((s[0] for s in lows), default=-np.inf)
            hmin = min((s[0] for s in highs), default=np.inf)

    out = pd.DataFrame(events, columns=["sweep_pos", "side", "swept_price", "extreme", "swing_pos"])
    out = out.sort_values("sweep_pos", kind="stable").reset_index(drop=True)
    out["sweep_time"] = df.index[out["sweep_pos"].to_numpy(dtype=int)]
    return out[SWEEP_COLUMNS]


def liquidity_taken_in_window(
    sweeps: pd.DataFrame, direction: int, start_pos: int, end_pos: int
) -> pd.DataFrame:
    """Sweeps that fuel ``direction`` with ``start_pos <= sweep_pos <= end_pos``.

    For a break candle at ``t``, pass ``end_pos = t - 1`` and
    ``start_pos = t - lookback``. Empty result => liquidity NOT taken.
    """
    check_direction(direction)
    side = "sell_side" if direction == LONG else "buy_side"
    m = (sweeps["side"] == side) & (sweeps["sweep_pos"] >= start_pos) & (sweeps["sweep_pos"] <= end_pos)
    return sweeps[m]
