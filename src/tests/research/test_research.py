import numpy as np
import pandas as pd
import pytest

from research import atr, h1_trend, orb, pdhl, run_study, simulate, squeeze, summarize
from research.features import trading_day


def random_walk(days=60, seed=0, vol=0.0004, start="2024-01-02 00:00"):
    n = days * 288
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    ret = rng.normal(0, vol, n)
    close = 2000 * np.exp(np.cumsum(ret))
    opn = np.r_[2000.0, close[:-1]]
    wick = np.abs(rng.normal(0, vol / 2, (2, n))) * close
    df = pd.DataFrame({"open": opn, "high": np.maximum(opn, close) + wick[0],
                       "low": np.minimum(opn, close) - wick[1], "close": close,
                       "volume": 100.0}, index=idx)
    return df


# ---------------------------------------------------------------- simulate: hand-worked
def bars(rows):
    a = np.array(rows, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]  # open, high, low, close


def sim(rows, d, risk, ks=(1.0, 2.0), spread=0.0, H=None):
    o, h, l, c = bars(rows)
    return simulate(o, h, l, c, np.array([0]), np.array([d]), np.array([risk]), ks, spread, H or len(rows))[0]


def test_simulate_long_target_stop_tie_and_timeout():
    # entry 100 (spread 0), risk 1 -> stop 99, 1R = 101, 2R = 102
    win = [(100, 100.5, 99.5, 100.2), (100.2, 102.1, 100.0, 102.0), (102, 102, 101, 101)]
    assert sim(win, 1, 1.0).tolist() == [1.0, 2.0]
    stop = [(100, 100.5, 98.9, 99.0), (99, 103, 99, 103)]
    assert sim(stop, 1, 1.0).tolist() == [-1.0, -1.0]
    tie = [(100, 100.5, 99.5, 100.2), (100.2, 102.5, 98.9, 100.0)]  # stop and 2R inside one bar
    assert sim(tie, 1, 1.0).tolist() == [-1.0, -1.0]
    drift = [(100, 100.4, 99.5, 100.2), (100.2, 100.6, 99.6, 100.5)]  # never resolves -> mark to close
    assert sim(drift, 1, 1.0).tolist() == pytest.approx([0.5, 0.5])


def test_simulate_gap_through_stop_fills_at_open():
    gap = [(100, 100.4, 99.5, 100.2), (98.0, 98.5, 97.5, 98.2)]
    assert sim(gap, 1, 1.0)[0] == pytest.approx(-2.0)


def test_simulate_spread_long_pays_on_entry_short_pays_on_exit():
    rows = [(100, 100.2, 99.9, 100.1), (100.1, 101.3, 100.0, 101.2)]
    # long: entry 100.2, risk 1 -> 1R target 101.2 reached by bar-1 high 101.3
    assert sim(rows, 1, 1.0, ks=(1.0,), spread=0.2)[0] == 1.0
    # short: entry 100 (bid), risk 1 -> stop 101; ask = high + 0.2 -> 100.9+0.2 >= 101 stops it
    rows = [(100, 100.1, 99.8, 99.9), (99.9, 100.9, 99.0, 99.5)]
    assert sim(rows, -1, 1.0, ks=(1.0,), spread=0.2)[0] == -1.0
    # no spread: stop (101) not reached by high 100.9, and low 99.0 reaches the 1R target -> +1
    assert sim(rows, -1, 1.0, ks=(1.0,), spread=0.0)[0] == 1.0


def test_simulate_short_target_needs_ask():
    rows = [(100, 100.1, 99.8, 99.9), (99.9, 100.0, 98.9, 99.0)]
    assert sim(rows, -1, 1.0, ks=(1.0,), spread=0.0)[0] == 1.0  # low 98.9 <= 99
    assert sim(rows, -1, 1.0, ks=(1.0,), spread=0.2)[0] != 1.0  # 98.9 + 0.2 = 99.1 > 99


# ---------------------------------------------------------------- features
def test_atr_matches_manual():
    df = random_walk(days=2)
    a = atr(df, 14)
    manual = max(df.high.iloc[20] - df.low.iloc[20], abs(df.high.iloc[20] - df.close.iloc[19]),
                 abs(df.low.iloc[20] - df.close.iloc[19]))
    assert np.isnan(a.iloc[12]) and np.isfinite(a.iloc[13])
    tr = [max(df.high.iloc[i] - df.low.iloc[i], abs(df.high.iloc[i] - df.close.iloc[i - 1]),
              abs(df.low.iloc[i] - df.close.iloc[i - 1])) for i in range(7, 21)]
    assert a.iloc[20] == pytest.approx(np.mean(tr)) and manual == pytest.approx(tr[-1])


def test_trading_day_rolls_at_22_utc():
    idx = pd.DatetimeIndex(["2024-01-02 21:55", "2024-01-02 22:00", "2024-01-03 21:55"], tz="UTC")
    d = trading_day(idx)
    assert d[0] != d[1] and d[1] == d[2]


# ---------------------------------------------------------------- no lookahead in candidates
@pytest.mark.parametrize("name,fn", [
    ("orb_london", lambda d, a: orb(d, a, start="08:00", tz="Europe/London", min_width_atr=0, max_width_atr=1e9)),
    ("orb_ny", lambda d, a: orb(d, a, start="09:30", tz="America/New_York", min_width_atr=0, max_width_atr=1e9)),
    ("squeeze", lambda d, a: squeeze(d, a, pct_window=300, pct=0.4)),
    ("pdhl", lambda d, a: pdhl(d, a)),
])
def test_candidates_do_not_look_ahead(name, fn):
    df = random_walk(days=40, seed=3)
    full = fn(df, atr(df))
    assert len(full) > 5, f"{name}: synthetic data produced too few events to test"
    for cut in (len(df) // 3, len(df) // 2, 3 * len(df) // 4):
        part = fn(df.iloc[:cut], atr(df.iloc[:cut]))
        want = full[full["idx"] < cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), want)


def test_h1_trend_is_causal():
    df = random_walk(days=30, seed=5)
    t_full = h1_trend(df, ema_n=20)
    cut = len(df) // 2
    t_part = h1_trend(df.iloc[:cut], ema_n=20)
    both = t_part.dropna().index
    assert (t_full.loc[both] == t_part.loc[both]).all()


# ---------------------------------------------------------------- the study itself
def test_null_is_calibrated_on_a_random_walk():
    """Random entry bars on a driftless walk should rarely look significant."""
    df = random_walk(days=60, seed=11)
    a = atr(df)
    ps = []
    for s in range(12):
        rng = np.random.default_rng(100 + s)
        idx = rng.integers(300, len(df) - 200, 60)
        d = rng.choice([-1, 1], 60)
        close = df.close.to_numpy()
        stop = close[idx] - d * 2.0 * a.to_numpy()[idx]
        ev = pd.DataFrame({"idx": idx, "direction": d, "stop_level": stop})
        st = run_study(df, a, ev, n_controls=100, horizon=48, seed=s)
        ps.append(summarize(st, n_boot=1000)["p"].min())
    assert sum(p < 0.01 for p in ps) <= 1  # min over 4 targets, 12 trials


def test_planted_edge_is_detected():
    df = random_walk(days=60, seed=21)
    rng = np.random.default_rng(7)
    sig = np.sort(rng.choice(np.arange(300, len(df) - 200, 40), 60, replace=False))
    d = rng.choice([-1, 1], len(sig))
    a = atr(df)
    # plant momentum: after each signal bar, push price d * 0.4 ATR per bar for 6 bars
    c = df.close.to_numpy().copy()
    shift = np.zeros(len(df))
    for i, di in zip(sig, d):
        shift[i + 1:] += di * 0.4 * a.iloc[i] * np.minimum(np.arange(len(df) - i - 1) + 1, 6)[: len(df) - i - 1] / 6
    c2 = c + shift
    o2 = np.r_[c2[0], c2[:-1]]
    df2 = pd.DataFrame({"open": o2, "high": np.maximum(o2, c2) + 0.05, "low": np.minimum(o2, c2) - 0.05,
                        "close": c2, "volume": 1.0}, index=df.index)
    a2 = atr(df2)
    ev = pd.DataFrame({"idx": sig, "direction": d, "stop_level": df2.close.to_numpy()[sig] - d * 2.0 * a2.to_numpy()[sig]})
    res = summarize(run_study(df2, a2, ev, n_controls=150, horizon=48, seed=2), n_boot=2000)
    assert (res["edge_R"] > 0).all() and res["p"].min() < 0.01


def test_summarize_mask_and_empty():
    df = random_walk(days=30, seed=2)
    a = atr(df)
    idx = np.arange(400, 4000, 100)
    ev = pd.DataFrame({"idx": idx, "direction": 1, "stop_level": df.close.to_numpy()[idx] - 3 * a.to_numpy()[idx]})
    st = run_study(df, a, ev, n_controls=20, horizon=24)
    assert summarize(st, np.zeros(len(st["obs"]), bool))["n"].eq(0).all()
    half = np.arange(len(st["obs"])) < len(st["obs"]) // 2
    assert summarize(st, half, n_boot=200)["n"].iloc[0] == half.sum()