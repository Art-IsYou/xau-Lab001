"""Property test: results computed on a truncated history must equal the same
slice of results computed on the full history."""
import pandas as pd

from primitives import (
    atr,
    find_sweeps,
    find_swings,
    last_swing_series,
    market_structure,
    range_in_atr,
    relative_volume,
)

K = 400


def test_series_primitives(random_walk):
    full, pre = random_walk, random_walk.iloc[:K]
    fns = {
        "atr": lambda d: atr(d),
        "range_in_atr": lambda d: range_in_atr(d),
        "relative_volume": lambda d: relative_volume(d),
        "last_swing_high": lambda d: last_swing_series(d, find_swings(d), "high"),
        "last_swing_low": lambda d: last_swing_series(d, find_swings(d), "low"),
        "structure": lambda d: market_structure(d),
    }
    for name, fn in fns.items():
        pd.testing.assert_series_equal(fn(full).iloc[:K], fn(pre), check_names=False, obj=name)


def test_swings_confirmed_prefix(random_walk):
    full, pre = random_walk, random_walk.iloc[:K]
    a = find_swings(full)
    a = a[a["confirmed_pos"] <= K - 1].reset_index(drop=True)
    b = find_swings(pre)
    pd.testing.assert_frame_equal(a, b)


def test_sweeps_prefix(random_walk):
    full, pre = random_walk, random_walk.iloc[:K]
    a = find_sweeps(full, find_swings(full), reclaim_bars=2)
    assert len(a) > 5  # sanity: the walk actually produces sweeps
    a = a[a["sweep_pos"] <= K - 1].reset_index(drop=True)
    b = find_sweeps(pre, find_swings(pre), reclaim_bars=2)
    pd.testing.assert_frame_equal(a, b)
