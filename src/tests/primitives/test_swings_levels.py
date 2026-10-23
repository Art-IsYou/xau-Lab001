import numpy as np

from primitives import (
    LONG,
    SHORT,
    cluster_levels,
    find_swings,
    last_swing_series,
    level_is_unbroken,
    next_level_beyond,
    swings_confirmed_by,
)
from conftest import from_hl, mk_swings, make_ohlcv


def test_single_swing_high_and_confirmation():
    highs = [10, 11, 12, 15, 12, 11, 10, 9, 8, 7]
    lows = [h - 1 for h in highs]
    df = from_hl(highs, lows)
    s = find_swings(df, 3, 3)
    hi = s[s["kind"] == "high"]
    assert hi["pivot_pos"].tolist() == [3]
    assert hi["price"].tolist() == [15]
    assert hi["confirmed_pos"].tolist() == [6]
    assert s[s["kind"] == "low"].empty


def test_equal_highs_yield_one_pivot():
    highs = [1, 2, 3, 5, 5, 3, 2, 1, 0.5, 0.4]
    df = from_hl(highs, [h - 1 for h in highs])
    s = find_swings(df, 3, 3)
    assert s[s["kind"] == "high"]["pivot_pos"].tolist() == [3]


def test_no_swings_at_edges():
    df = from_hl([1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6])
    assert find_swings(df, 3, 3).empty


def test_last_swing_series_is_nan_until_confirmed():
    highs = [10, 11, 12, 15, 12, 11, 10, 9, 8, 7]
    df = from_hl(highs, [h - 1 for h in highs])
    s = find_swings(df, 3, 3)
    ls = last_swing_series(df, s, "high")
    assert ls.iloc[:6].isna().all()
    assert (ls.iloc[6:] == 15).all()


def test_swings_confirmed_by():
    s = mk_swings([("high", 10, 3, 6), ("low", 5, 8, 11)])
    assert len(swings_confirmed_by(s, 5)) == 0
    assert len(swings_confirmed_by(s, 6)) == 1
    assert len(swings_confirmed_by(s, 11)) == 2


def test_cluster_equal_highs():
    s = mk_swings([("high", 2000.0, 10, 13), ("high", 2000.3, 30, 33), ("low", 1990.0, 20, 23)])
    levels = cluster_levels(s, as_of_pos=40, tol=0.5)
    res = [lv for lv in levels if lv.kind == "resistance"]
    sup = [lv for lv in levels if lv.kind == "support"]
    assert len(res) == 1 and res[0].touches == 2 and res[0].price == 2000.3
    assert res[0].first_pos == 10 and res[0].last_pos == 30
    assert len(sup) == 1 and sup[0].touches == 1
    assert [lv.kind for lv in cluster_levels(s, 40, 0.5, min_touches=2)] == ["resistance"]


def test_cluster_respects_confirmation_and_tolerance():
    s = mk_swings([("high", 2000.0, 10, 13), ("high", 2000.3, 30, 33)])
    assert cluster_levels(s, as_of_pos=20, tol=0.5)[0].touches == 1  # 2nd not known yet
    assert len(cluster_levels(s, as_of_pos=40, tol=0.1)) == 2  # too far apart to merge


def test_max_age():
    s = mk_swings([("high", 2000.0, 10, 13)])
    assert cluster_levels(s, 500, 0.5, max_age_bars=100) == []


def test_level_is_unbroken():
    s = mk_swings([("high", 100.0, 1, 2)])
    lv = cluster_levels(s, 10, 0.5)[0]
    df = make_ohlcv([(95, 96, 94, 95, 1)] * 2 + [(99, 101, 98, 99.5, 1), (99, 101, 98, 101.5, 1)])
    assert level_is_unbroken(df, lv, as_of_pos=2)  # wick above, close below
    assert not level_is_unbroken(df, lv, as_of_pos=3)  # closed above


def test_next_level_beyond():
    s = mk_swings(
        [("high", 2012.0, 5, 8), ("high", 2000.0, 20, 23), ("low", 1990.0, 30, 33), ("low", 1980.0, 40, 43)]
    )
    levels = cluster_levels(s, 100, 0.5)
    assert next_level_beyond(levels, 2002.0, LONG).price == 2012.0
    assert next_level_beyond(levels, 2013.0, LONG) is None
    assert next_level_beyond(levels, 1995.0, SHORT).price == 1990.0
    assert next_level_beyond(levels, 1975.0, SHORT) is None
