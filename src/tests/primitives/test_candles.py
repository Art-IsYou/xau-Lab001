import numpy as np
import pandas as pd

from primitives import LONG, SHORT, atr, body_beyond_fraction, breakout_candle_mask, candle_parts, range_in_atr
from conftest import make_ohlcv


def test_candle_parts():
    df = make_ohlcv([(10, 14, 9, 13, 1), (10, 10, 10, 10, 1)])
    p = candle_parts(df)
    assert p["range"].iloc[0] == 5
    assert p["body"].iloc[0] == 3
    assert p["body_ratio"].iloc[0] == 0.6
    assert p["upper_wick"].iloc[0] == 1
    assert p["lower_wick"].iloc[0] == 1
    assert p["direction"].iloc[0] == 1
    assert np.isnan(p["body_ratio"].iloc[1])  # zero-range candle


def test_breakout_mask_long():
    level = 100.0
    df = make_ohlcv(
        [
            (99.5, 102.5, 99.4, 102.0, 1),  # valid: opened below, closed above, body 2.5/3.1
            (100.5, 102.5, 100.4, 102.0, 1),  # opened already above level
            (99.0, 105.0, 98.9, 100.4, 1),  # closed above but body ratio ~0.23
            (99.5, 100.2, 99.0, 99.8, 1),  # never closed above
            (101.0, 101.2, 98.0, 98.5, 1),  # bearish
        ]
    )
    assert breakout_candle_mask(df, level, LONG).tolist() == [True, False, False, False, False]


def test_breakout_mask_short():
    level = 100.0
    df = make_ohlcv([(100.5, 100.6, 97.5, 98.0, 1), (99.5, 99.6, 97.0, 98.0, 1)])
    assert breakout_candle_mask(df, level, SHORT).tolist() == [True, False]


def test_body_ratio_threshold_is_inclusive():
    # body 2, range 4 -> exactly 0.5
    df = make_ohlcv([(99.0, 102.0, 98.0, 101.0, 1)])
    assert breakout_candle_mask(df, 100.0, LONG, min_body_ratio=0.5).iloc[0]
    assert not breakout_candle_mask(df, 100.0, LONG, min_body_ratio=0.51).iloc[0]


def test_body_beyond_fraction_option():
    # body 99 -> 101 around level 100: half the body is beyond
    df = make_ohlcv([(99.0, 101.5, 98.8, 101.0, 1)])
    assert body_beyond_fraction(df, 100.0, LONG).iloc[0] == 0.5
    assert breakout_candle_mask(df, 100.0, LONG, min_beyond_frac=0.5).iloc[0]
    assert not breakout_candle_mask(df, 100.0, LONG, min_beyond_frac=0.6).iloc[0]


def test_level_can_be_series():
    df = make_ohlcv([(99.5, 102.5, 99.4, 102.0, 1), (99.5, 102.5, 99.4, 102.0, 1)])
    lvl = pd.Series([100.0, 105.0], index=df.index)
    assert breakout_candle_mask(df, lvl, LONG).tolist() == [True, False]


def test_atr_constant_range():
    rows = [(10, 11, 9, 10, 1)] * 30  # TR = 2 every bar
    df = make_ohlcv(rows)
    a = atr(df, 14)
    assert a.iloc[:13].isna().all()
    assert np.allclose(a.iloc[13:], 2.0)


def test_range_in_atr_uses_prior_atr():
    rows = [(10, 11, 9, 10, 1)] * 20 + [(10, 16, 8, 10, 1)]  # last range 8 vs ATR 2
    df = make_ohlcv(rows)
    assert np.isclose(range_in_atr(df, 14).iloc[-1], 4.0)
