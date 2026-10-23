"""End-to-end: chain the primitives through the break-Out checklist on one
hand-built long scenario. This is a smoke test for composability, not the
setup module itself.

Scenario (M1, London hours): equal highs ~2000.2 (idx 22 & 38) form the key
level, a higher swing high ~2012.2 (idx 5) is the target, the swing low ~1991.8
(idx 30) is swept at idx 49, price grinds up and a high-volume candle breaks
2000.2 at idx 59; CoC triggers at idx 60.
"""
import numpy as np

from primitives import (
    LONG,
    align_htf_to_base,
    breakout_candle_mask,
    cluster_levels,
    coc_trigger,
    find_sweeps,
    find_swings,
    level_is_unbroken,
    liquidity_taken_in_window,
    market_structure,
    min_target_for_rr,
    next_level_beyond,
    passes_rr,
    range_in_atr,
    relative_volume,
    session_mask,
    structure_aligned,
)
from conftest import from_path

BREAK = 59


def build_m1():
    pts = [(0, 2006), (5, 2012), (14, 1996), (22, 2000), (30, 1992), (38, 2000), (46, 1993), (56, 1999.5), (59, 1999.6), (60, 2002.8), (70, 2010), (79, 2011)]
    df = from_path(pts, 80)
    df.iloc[49] = [1994.5, 1995.0, 1991.0, 1994.0, 150.0]  # sell-side sweep of the 1991.8 low
    df.iloc[BREAK] = [1999.6, 2002.6, 1999.4, 2002.2, 320.0]  # break candle
    df.iloc[BREAK + 1] = [2002.2, 2003.0, 2001.8, 2002.8, 150.0]  # CoC bar
    return df


def build_m15():
    pts = [(0, 1980)]
    for m in range(8):
        pts += [(m * 20 + 10, 1990 + 8 * m), (m * 20 + 20, 1984 + 8 * m)]
    return from_path(pts, 161, wick=0.5, start="2024-01-02 00:00", freq="15min")


def test_breakout_checklist_chain():
    df = build_m1()
    swings = find_swings(df, 3, 3)

    # key level: equal highs
    levels_2t = cluster_levels(swings, as_of_pos=BREAK - 1, tol=0.5, min_touches=2)
    res = [lv for lv in levels_2t if lv.kind == "resistance"]
    assert len(res) == 1 and np.isclose(res[0].price, 2000.2)
    level = res[0]
    assert level_is_unbroken(df, level, BREAK - 1)

    # breakout candle qualifies - and it is the only one in the scenario
    mask = breakout_candle_mask(df, level.price, LONG)
    assert mask[mask].index.tolist() == [df.index[BREAK]]

    # participation
    assert relative_volume(df, 20).iloc[BREAK] >= 3.0
    assert range_in_atr(df, 14).iloc[BREAK] > 1.0

    # liquidity taken in the pullback before the break
    sweeps = find_sweeps(df, swings)
    taken = liquidity_taken_in_window(sweeps, LONG, BREAK - 25, BREAK - 1)
    assert taken["sweep_pos"].tolist() == [49]

    # M15 aligned (no lookahead projection onto M1)
    htf = build_m15()
    trend = align_htf_to_base(market_structure(htf), df.index, "15min")
    assert structure_aligned(trend, LONG).iloc[BREAK]

    # major session
    assert session_mask(df.index).iloc[BREAK]

    # CoC entry + R:R
    trig = coc_trigger(df, BREAK, LONG)
    assert trig.pos == BREAK + 1 and not trig.stopped_same_bar
    all_levels = cluster_levels(swings, as_of_pos=trig.pos, tol=0.5)
    target = next_level_beyond(all_levels, trig.fill_price, LONG)
    assert np.isclose(target.price, 2012.2)
    assert target.price >= min_target_for_rr(trig.fill_price, trig.stop_price, LONG, 2.5, expansion=0.2)
    assert passes_rr(trig.fill_price, trig.stop_price, target.price, LONG, min_rr=2.5, expansion=0.2)


def test_chain_rejects_when_liquidity_not_taken():
    df = build_m1()
    df.iloc[49] = [1994.5, 1995.0, 1992.5, 1994.0, 150.0]  # no longer pierces the swing low
    swings = find_swings(df, 3, 3)
    sweeps = find_sweeps(df, swings)
    assert liquidity_taken_in_window(sweeps, LONG, BREAK - 25, BREAK - 1).empty
