import numpy as np
import pandas as pd

from primitives import (
    LONG,
    SHORT,
    align_htf_to_base,
    market_structure,
    relative_volume,
    session_label,
    session_mask,
    structure_aligned,
    volume_gate,
)
from conftest import from_path, make_ohlcv


def test_uptrend_downtrend_range():
    up = from_path([(0, 100), (5, 110), (10, 104), (15, 115), (20, 108), (25, 120), (30, 112)], 31)
    down = from_path([(0, 120), (5, 110), (10, 116), (15, 105), (20, 112), (25, 100), (30, 108)], 31)
    flat = from_path([(0, 100), (5, 110), (10, 104), (15, 110), (20, 104), (25, 110), (30, 104)], 31)
    assert market_structure(up).iloc[-1] == 1
    assert market_structure(down).iloc[-1] == -1
    assert market_structure(flat).iloc[-1] == 0


def test_structure_unknown_early_is_neutral():
    up = from_path([(0, 100), (5, 110), (10, 104), (15, 115), (20, 108), (25, 120), (30, 112)], 31)
    assert (market_structure(up).iloc[:10] == 0).all()


def test_align_htf_has_no_lookahead():
    s = pd.Series([1, 2, 3], index=pd.date_range("2024-01-02 10:00", periods=3, freq="15min"))
    base = pd.date_range("2024-01-02 10:00", periods=60, freq="1min")
    out = align_htf_to_base(s, base, "15min")
    assert np.isnan(out.loc["2024-01-02 10:14"])
    assert out.loc["2024-01-02 10:15"] == 1
    assert out.loc["2024-01-02 10:29"] == 1
    assert out.loc["2024-01-02 10:30"] == 2


def test_structure_aligned():
    t = pd.Series([1, -1, 0, np.nan])
    assert structure_aligned(t, LONG).tolist() == [True, False, False, False]
    assert structure_aligned(t, SHORT).tolist() == [False, True, False, False]


def test_relative_volume_and_gate():
    rows = [(10, 11, 9, 10, 100.0)] * 20 + [(10, 11, 9, 10, 300.0), (10, 11, 9, 10, 120.0)]
    df = make_ohlcv(rows)
    rv = relative_volume(df, 20)
    assert rv.iloc[:20].isna().all()
    assert rv.iloc[20] == 3.0
    assert volume_gate(df, 20, 1.5).tolist()[-2:] == [True, False]


def test_sessions_dst_and_weekend():
    idx = pd.DatetimeIndex(
        [
            "2024-07-01 07:30",  # 08:30 BST -> london
            "2024-01-02 07:30",  # 07:30 GMT -> before london
            "2024-01-02 13:30",  # 08:30 EST -> new york (and london)
            "2024-01-02 22:30",  # 17:30 EST, after both
            "2024-01-06 10:00",  # Saturday
        ]
    )
    m = session_mask(idx)
    assert m.tolist() == [True, False, True, False, False]
    lab = session_label(idx)
    assert lab.iloc[0] == "london"
    assert lab.iloc[2] == "london+new_york"
    assert lab.iloc[3] == ""


def test_sessions_tz_aware_input():
    idx = pd.DatetimeIndex(["2024-01-02 09:00"], tz="UTC")
    assert session_mask(idx).iloc[0]
