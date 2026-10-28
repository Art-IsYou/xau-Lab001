from dataclasses import replace

import pandas as pd
import pytest

from data import resample
from primitives import LONG, SHORT
from setups import STAGES, BreakoutConfig, scan_breakout
from tests.setups.synth import long_breakout_scenario, mirror, random_walk, trend_series

@pytest.fixture(scope="module")
def scenario():
    return long_breakout_scenario()


def _scan(df, trend_value=1, cfg=None):
    return scan_breakout(df, cfg=cfg or BreakoutConfig(), trend=trend_series(df, trend_value))

def test_long_signal(scenario):
    df, meta = scenario
    res = _scan(df)
    assert len(res.signals) == 1
    s = res.signals[0]
    assert s.direction == LONG
    assert s.break_pos == meta["break_pos"] and s.trigger_pos == meta["break_pos"] + 1
    assert s.level_price == pytest.approx(meta["level"]) and s.level_touches == 2
    assert s.entry_planned == pytest.approx(meta["entry"]) and s.stop == pytest.approx(meta["stop"])
    assert s.fill_price == pytest.approx(meta["entry"])
    assert s.target == pytest.approx(meta["target"])
    assert s.rr_planned == pytest.approx((meta["target"] - meta["entry"]) / (meta["entry"] - meta["stop"]))
    assert s.rel_volume >= 1.5 and s.session == "london"
    assert s.sweep_pos == meta["sweep_pos"] and not s.stopped_same_bar


def test_short_is_the_mirror(scenario):
    df, meta = scenario
    res = _scan(mirror(df), trend_value=-1)
    assert len(res.signals) == 1
    s = res.signals[0]
    assert s.direction == SHORT and s.break_pos == meta["break_pos"]
    assert s.level_price == pytest.approx(4000 - meta["level"])
    assert s.entry_planned == pytest.approx(4000 - meta["entry"])
    assert s.stop == pytest.approx(4000 - meta["stop"])
    assert s.target == pytest.approx(4000 - meta["target"])


def test_wrong_direction_trend_blocks(scenario):
    df, _ = scenario
    assert _scan(df, trend_value=-1).signals == []  # LONG needs +1, SHORT finds no level
    assert _scan(df, trend_value=0).signals == []


def _flat_volume(df, meta):
    d = df.copy()
    d.iloc[meta["break_pos"], d.columns.get_loc("volume")] = 100.0
    return d


def _off_session(df, meta):
    d = df.copy()
    d.index = d.index - pd.Timedelta(hours=7)  # 02:00 UTC
    return d


def _no_trigger(df, meta):
    d = df.copy()
    k = meta["break_pos"] + 1
    d.iloc[k, d.columns.get_loc("high")] = meta["entry"] - 0.1  # never trades through the break high
    d.iloc[k, d.columns.get_loc("close")] = meta["entry"] - 0.3
    for j in (k + 1, k + 2):  # stays under it for the whole wait window
        d.iloc[j, d.columns.get_loc("high")] = meta["entry"] - 0.1
        d.iloc[j, d.columns.get_loc("close")] = meta["entry"] - 0.3
    return d


# (id, data tweak, config overrides, last stage that should still have survivors, first stage that must be empty)
CASES = [
    ("body", None, dict(min_body_ratio=0.99), None, "candle"),
    ("volume", _flat_volume, {}, "candle", "volume"),
    ("session", _off_session, {}, "volume", "session"),
    ("level_touches", None, dict(level_min_touches=3), "htf", "level"),
    ("liquidity", None, dict(liq_lookback_bars=2), "level", "liquidity"),
    ("exhaustion", None, dict(max_range_atr=0.5), "liquidity", "exhaustion"),
    ("rr", None, dict(min_rr=50.0), "target", "rr"),
    ("trigger", _no_trigger, {}, "rr", "trigger"),
]


@pytest.mark.parametrize("name,tweak,overrides,last_ok,first_dead", CASES, ids=[c[0] for c in CASES])
def test_each_gate_kills_the_trade(scenario, name, tweak, overrides, last_ok, first_dead):
    df, meta = scenario
    d = tweak(df, meta) if tweak else df
    res = _scan(d, cfg=replace(BreakoutConfig(), **overrides))
    assert res.signals == []
    f = res.funnel["LONG"]
    assert f[first_dead] == 0
    if last_ok:
        assert f[last_ok] > 0


def test_session_gate_can_be_disabled(scenario):
    df, meta = scenario
    off = _off_session(df, meta)
    res = _scan(off, cfg=replace(BreakoutConfig(), major_session_only=False))
    assert len(res.signals) == 1 and res.signals[0].session == ""


def test_funnel_is_monotonic(scenario):
    df, _ = scenario
    for counts in _scan(df).funnel.values():
        seq = [counts[s] for s in STAGES]
        assert seq == sorted(seq, reverse=True)


def test_gap_through_trigger_is_rechecked(scenario):
    """If the trigger bar gaps far above the trigger price the fill R:R is re-tested."""
    df, meta = scenario
    d = df.copy()
    k = meta["break_pos"] + 1
    gap_open = 2006.5
    for col, v in (("open", gap_open), ("high", 2007.0), ("low", 2006.4), ("close", 2006.9)):
        d.iloc[k, d.columns.get_loc(col)] = v
    res = _scan(d)
    assert res.signals == [] and res.funnel["LONG"]["trigger"] == 1 and res.funnel["LONG"]["fill_rr"] == 0


def test_no_lookahead_truncation():
    """Cutting the data short must not change any earlier signal."""
    import numpy as np

    df = random_walk(8000)
    # alternate LONG / SHORT / neutral regimes so both directions get exercised
    blocks = np.tile([1, -1, 0], len(df) // 300 + 1)
    trend = pd.Series(np.repeat(blocks, 100)[: len(df)], index=df.index)
    cfg = replace(BreakoutConfig(), min_rel_volume=0.0, min_body_ratio=0.3, min_beyond_frac=0.0,
                  level_min_touches=1, min_rr=0.5, major_session_only=False,
                  liq_after_level=False, liq_lookback_bars=80)
    full = scan_breakout(df, cfg=cfg, trend=trend)
    assert {s.direction for s in full.signals} == {LONG, SHORT} and len(full.signals) >= 10

    cut = 5000
    part = scan_breakout(df.iloc[:cut], cfg=cfg, trend=trend.iloc[:cut])
    margin = cfg.max_wait_bars + 2

    def key(res):
        return [(s.break_pos, s.direction, round(s.target, 6), round(s.stop, 6), s.trigger_pos)
                for s in res.signals if s.break_pos < cut - margin]

    assert key(full) == key(part) and len(key(part)) >= 5


def test_htf_path_end_to_end():
    """Real alignment path: M1 -> resample M15 -> structure -> gate."""
    df = random_walk(8000, seed=11)
    m15 = resample(df, "M15")
    cfg = replace(BreakoutConfig(), min_rel_volume=1.0, level_min_touches=1, min_rr=1.0,
                  major_session_only=False, liq_after_level=False, liq_lookback_bars=60)
    res = scan_breakout(df, m15, cfg)
    assert res.funnel["LONG"]["htf"] > 0 and res.funnel["SHORT"]["htf"] > 0
    assert not res.to_frame().empty
    assert list(res.funnel_frame().index[:2]) == ["bars", "candle"]


def test_requires_trend_or_htf():
    df, _ = long_breakout_scenario()
    with pytest.raises(ValueError):
        scan_breakout(df)


def test_scalar_break_candle_matches_primitive():
    """setups.breakout.is_break_candle must stay identical to the vectorized primitive."""
    import numpy as np

    from primitives import breakout_candle_mask
    from setups.breakout import is_break_candle

    rng = np.random.default_rng(3)
    n = 4000
    o = 2000 + rng.normal(0, 1.0, n)
    c = o + rng.normal(0, 1.2, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.4, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.4, n))
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0},
                      index=pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC"))
    level = 2000.3
    for d in (LONG, SHORT):
        for kw in (dict(min_body_ratio=0.5, min_beyond_frac=0.5), dict(min_body_ratio=0.3, min_beyond_frac=0.0)):
            expect = breakout_candle_mask(df, level, d, **kw).to_numpy()
            got = np.array([is_break_candle(o[i], h[i], l[i], c[i], level, d, **kw) for i in range(n)])
            assert (expect == got).all() and expect.sum() > 20