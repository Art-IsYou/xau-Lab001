"""Structural levels: clusters of same-side swing points.

``Level.price`` is the *extreme* of the cluster (highest high for resistance,
lowest low for support) because a break only counts once that extreme is
cleared. ``tol`` is in price units (e.g. ``0.25 * atr``).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ._types import LONG, check_direction


@dataclass(frozen=True)
class Level:
    price: float
    kind: str  # "resistance" | "support"
    touches: int
    first_pos: int  # pivot position of earliest touch
    last_pos: int  # pivot position of latest touch
    last_confirmed_pos: int  # bar at which the latest touch became known


def cluster_levels(
    swings: pd.DataFrame,
    as_of_pos: int,
    tol: float,
    min_touches: int = 1,
    max_age_bars: int | None = None,
) -> list[Level]:
    """Levels known at the close of bar ``as_of_pos`` (no lookahead)."""
    known = swings[swings["confirmed_pos"] <= as_of_pos]
    if max_age_bars is not None:
        known = known[known["pivot_pos"] >= as_of_pos - max_age_bars]

    levels: list[Level] = []

    def flush(group, kind):
        if not group:
            return
        prices = [g.price for g in group]
        levels.append(
            Level(
                price=max(prices) if kind == "resistance" else min(prices),
                kind=kind,
                touches=len(group),
                first_pos=min(g.pivot_pos for g in group),
                last_pos=max(g.pivot_pos for g in group),
                last_confirmed_pos=max(g.confirmed_pos for g in group),
            )
        )

    for swing_kind, level_kind in (("high", "resistance"), ("low", "support")):
        sel = known[known["kind"] == swing_kind].sort_values("price")
        group: list = []
        for row in sel.itertuples(index=False):
            if group and row.price - group[0].price > tol:
                flush(group, level_kind)
                group = []
            group.append(row)
        flush(group, level_kind)

    levels = [lv for lv in levels if lv.touches >= min_touches]
    return sorted(levels, key=lambda lv: lv.price)


def level_is_unbroken(df: pd.DataFrame, level: Level, as_of_pos: int) -> bool:
    """True if no bar has *closed* beyond the level since its first touch."""
    seg = df["close"].to_numpy()[level.first_pos + 1 : as_of_pos + 1]
    if level.kind == "resistance":
        return not bool((seg > level.price).any())
    return not bool((seg < level.price).any())


def next_level_beyond(levels: list[Level], price: float, direction: int) -> Level | None:
    """Nearest opposing level beyond ``price`` in the trade direction (a target candidate)."""
    check_direction(direction)
    if direction == LONG:
        cands = [lv for lv in levels if lv.kind == "resistance" and lv.price > price]
        return min(cands, key=lambda lv: lv.price) if cands else None
    cands = [lv for lv in levels if lv.kind == "support" and lv.price < price]
    return max(cands, key=lambda lv: lv.price) if cands else None
