import math

import pytest

from primitives import LONG, SHORT, coc_levels, coc_trigger, min_target_for_rr, passes_rr, risk_reward
from conftest import make_ohlcv

BREAK = (1999.6, 2002.6, 1999.4, 2002.2, 300.0)


def test_coc_levels():
    assert coc_levels(2002.6, 1999.4, LONG) == (2002.6, 1999.4)
    assert coc_levels(2002.6, 1999.4, LONG, 0.1, 0.1) == pytest.approx((2002.7, 1999.3))
    assert coc_levels(2002.6, 1999.4, SHORT, 0.1, 0.1) == pytest.approx((1999.3, 2002.7))


def test_trigger_next_bar():
    df = make_ohlcv([BREAK, (2002.2, 2003.0, 2001.8, 2002.8, 100)])
    t = coc_trigger(df, 0, LONG)
    assert (t.pos, t.fill_price, t.stop_price, t.stopped_same_bar) == (1, 2002.6, 1999.4, False)


def test_trigger_gap_fills_at_open():
    df = make_ohlcv([BREAK, (2003.5, 2004.0, 2003.2, 2003.8, 100)])
    assert coc_trigger(df, 0, LONG).fill_price == 2003.5


def test_stop_first_kills_setup():
    df = make_ohlcv([BREAK, (2000.5, 2002.0, 1999.0, 2000.0, 100), (2000, 2005, 1999.5, 2004, 100)])
    assert coc_trigger(df, 0, LONG) is None


def test_same_bar_trigger_and_stop_is_flagged():
    df = make_ohlcv([BREAK, (2002.0, 2003.0, 1999.0, 2001.0, 100)])
    assert coc_trigger(df, 0, LONG).stopped_same_bar


def test_max_wait():
    quiet = (2001.0, 2002.0, 2000.5, 2001.5, 100)
    df = make_ohlcv([BREAK, quiet, quiet, quiet, (2002, 2004, 2001.5, 2003, 100)])
    assert coc_trigger(df, 0, LONG, max_wait_bars=3) is None
    assert coc_trigger(df, 0, LONG, max_wait_bars=4).pos == 4


def test_short_trigger():
    brk = (100.4, 100.6, 97.5, 98.0, 300.0)
    df = make_ohlcv([brk, (98.0, 98.2, 97.0, 97.3, 100)])
    t = coc_trigger(df, 0, SHORT)
    assert (t.pos, t.fill_price, t.stop_price) == (1, 97.5, 100.6)


def test_risk_reward_long_short():
    assert risk_reward(100, 98, 106, LONG) == 3.0
    assert risk_reward(100, 102, 94, SHORT) == 3.0


def test_expansion_hurts_rr():
    assert math.isclose(risk_reward(100, 98, 106, LONG, expansion=0.5), 5.5 / 2.5)
    assert math.isclose(risk_reward(100, 102, 94, SHORT, expansion=0.5), 5.5 / 2.5)


def test_invalid_risk_is_nan_and_fails():
    assert math.isnan(risk_reward(100, 101, 106, LONG))
    assert not passes_rr(100, 101, 106, LONG)


def test_min_target():
    assert min_target_for_rr(100, 98, LONG, 2.5) == 105
    assert min_target_for_rr(100, 102, SHORT, 2.5) == 95
    assert min_target_for_rr(100, 98, LONG, 2.5, expansion=0.5) == 106.75
    t = min_target_for_rr(100, 98, LONG, 2.5)
    assert passes_rr(100, 98, t, LONG, 2.5)
    assert not passes_rr(100, 98, t - 0.01, LONG, 2.5)
