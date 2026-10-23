"""Candle-over-candle (CoC) entry mechanics for the break-Out setup.

LONG : entry stop-order just above the break candle's high, stop just below its low.
SHORT: mirror image.
Buffers are in price units and default to 0 so the caller (config) owns them.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ._types import LONG, check_direction


@dataclass(frozen=True)
class CocTrigger:
    pos: int  # bar on which the entry triggered
    fill_price: float  # trigger price, or the open if the bar gapped through it
    stop_price: float
    stopped_same_bar: bool  # trigger bar also reached the stop; intrabar order unknown


def coc_levels(
    break_high: float,
    break_low: float,
    direction: int,
    entry_buffer: float = 0.0,
    stop_buffer: float = 0.0,
) -> tuple[float, float]:
    """(trigger_price, stop_price) for a break candle."""
    check_direction(direction)
    if direction == LONG:
        return break_high + entry_buffer, break_low - stop_buffer
    return break_low - entry_buffer, break_high + stop_buffer


def coc_trigger(
    df: pd.DataFrame,
    break_pos: int,
    direction: int,
    entry_buffer: float = 0.0,
    stop_buffer: float = 0.0,
    max_wait_bars: int = 3,
) -> CocTrigger | None:
    """First bar after the break candle that trades through its extreme.

    Returns None if the stop level is hit first (setup dead) or nothing
    triggers within ``max_wait_bars``.
    """
    check_direction(direction)
    trig, stop = coc_levels(
        float(df["high"].iloc[break_pos]), float(df["low"].iloc[break_pos]), direction, entry_buffer, stop_buffer
    )
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    end = min(len(df), break_pos + 1 + max_wait_bars)
    for j in range(break_pos + 1, end):
        if direction == LONG:
            if h[j] >= trig:
                return CocTrigger(j, max(trig, o[j]), stop, bool(l[j] <= stop))
            if l[j] <= stop:
                return None
        else:
            if l[j] <= trig:
                return CocTrigger(j, min(trig, o[j]), stop, bool(h[j] >= stop))
            if h[j] >= stop:
                return None
    return None
