from primitives import LONG, SHORT, find_sweeps, liquidity_taken_in_window
from conftest import make_ohlcv, mk_swings

FLAT = (105.0, 106.0, 104.0, 105.0, 100.0)


def bars(overrides, n=12):
    rows = [FLAT] * n
    for i, r in overrides.items():
        rows[i] = r
    return make_ohlcv(rows)


LOW_SWING = mk_swings([("low", 100.0, 3, 6)])  # known from bar 6
HIGH_SWING = mk_swings([("high", 110.0, 3, 6)])


def test_sell_side_sweep_same_candle():
    df = bars({8: (104.0, 105.5, 99.0, 101.0, 100.0)})
    ev = find_sweeps(df, LOW_SWING)
    assert len(ev) == 1
    r = ev.iloc[0]
    assert (r.sweep_pos, r.side, r.swept_price, r.extreme, r.swing_pos) == (8, "sell_side", 100.0, 99.0, 3)


def test_no_sweep_before_swing_is_confirmed():
    df = bars({5: (104.0, 105.5, 99.0, 101.0, 100.0)})
    assert find_sweeps(df, LOW_SWING).empty


def test_close_through_is_a_break_not_a_sweep_and_swing_is_consumed():
    df = bars({8: (104.0, 104.5, 99.0, 98.0, 100.0), 9: (98.0, 102.0, 97.0, 101.0, 100.0)})
    assert find_sweeps(df, LOW_SWING).empty
    # bar 9 closes back above 100 but the swing was already consumed at bar 8
    assert find_sweeps(df, LOW_SWING, reclaim_bars=1).empty


def test_reclaim_within_two_bars():
    df = bars({8: (104.0, 104.5, 99.0, 98.5, 100.0), 9: (98.5, 102.0, 98.0, 101.0, 100.0)})
    assert find_sweeps(df, LOW_SWING, reclaim_bars=1).empty
    ev = find_sweeps(df, LOW_SWING, reclaim_bars=2)
    assert len(ev) == 1
    assert ev.iloc[0].sweep_pos == 9
    assert ev.iloc[0].extreme == 98.0  # deepest wick over the pending window


def test_reclaim_window_expires():
    df = bars({8: (104.0, 104.5, 99.0, 98.5, 100.0), 9: (98.5, 99.0, 97.0, 98.0, 100.0), 10: (98, 103, 97.5, 102, 100)})
    assert find_sweeps(df, LOW_SWING, reclaim_bars=2).empty


def test_swing_swept_only_once():
    df = bars({8: (104.0, 105.5, 99.0, 101.0, 100.0), 10: (104.0, 105.5, 99.0, 101.0, 100.0)})
    assert len(find_sweeps(df, LOW_SWING)) == 1


def test_buy_side_sweep():
    df = bars({8: (106.0, 111.0, 105.0, 108.0, 100.0)})
    ev = find_sweeps(df, HIGH_SWING)
    assert len(ev) == 1 and ev.iloc[0].side == "buy_side" and ev.iloc[0].extreme == 111.0


def test_window_filter_and_direction():
    df = bars({8: (104.0, 105.5, 99.0, 101.0, 100.0)})
    ev = find_sweeps(df, LOW_SWING)
    assert not liquidity_taken_in_window(ev, LONG, 0, 10).empty
    assert liquidity_taken_in_window(ev, SHORT, 0, 10).empty  # wrong side
    assert liquidity_taken_in_window(ev, LONG, 0, 7).empty  # not yet happened
    assert liquidity_taken_in_window(ev, LONG, 9, 11).empty  # too old for the window
