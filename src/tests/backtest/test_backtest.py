"""Backtest engine tests. Every expected number is worked out by hand in the comments."""
import numpy as np
import pandas as pd
import pytest

from backtest import BacktestConfig, breakdowns, run_backtest, summarize

LONG, SHORT = 1, -1


def make_df(bars):
    idx = pd.date_range("2024-01-02 05:00", periods=len(bars), freq="1min", tz="UTC")
    return pd.DataFrame(bars, index=idx, columns=["open", "high", "low", "close"]).assign(volume=100.0)


def sig(direction=LONG, break_pos=0, trigger_pos=1, fill=100.0, stop=99.0, target=102.5,
        same_bar=False, **extra):
    return {"direction": direction, "break_pos": break_pos, "trigger_pos": trigger_pos,
            "fill_price": fill, "stop": stop, "target": target, "stopped_same_bar": same_bar, **extra}


def frame(*rows):
    return pd.DataFrame(list(rows))


NOCOST = BacktestConfig(spread=0.0)

# bar 0 = break candle, bar 1 = trigger bar (entry 100, never touches 99), bar 2 drifts up
LONG_BARS = [
    (99.5, 100.5, 99.4, 100.4),
    (100.0, 100.2, 99.9, 100.1),
    (100.1, 101.0, 100.0, 100.9),
]


def test_long_target_no_cost():
    df = make_df(LONG_BARS + [(100.9, 102.6, 100.8, 102.5)])
    res = run_backtest(df, frame(sig()), NOCOST)
    t = res.trades.iloc[0]
    # risk 1.0, reward 2.5 -> +2.5R; lots = 10000*1% / (1.0*100) = 1.00; pnl = 2.5*100*1 = 250
    assert t.exit_reason == "target" and t.r == pytest.approx(2.5)
    assert t.lots == pytest.approx(1.0) and t.pnl_usd == pytest.approx(250.0)
    assert t.equity_after == pytest.approx(10_250.0)
    assert df.index[3] == t.exit_time


def test_long_pays_spread_on_entry():
    df = make_df(LONG_BARS + [(100.9, 102.6, 100.8, 102.5)])
    t = run_backtest(df, frame(sig()), BacktestConfig(spread=0.2)).trades.iloc[0]
    # entry = ask = 100.2, risk 1.2, target 102.5 -> reward 2.3 -> 2.3/1.2 R
    assert t.entry == pytest.approx(100.2) and t.r == pytest.approx(2.3 / 1.2)
    assert t.rr_net == pytest.approx(2.3 / 1.2)
    # lots = 100 / (1.2*100) = 0.8333 -> floor to 0.83 ; pnl = 2.3*100*0.83
    assert t.lots == pytest.approx(0.83) and t.pnl_usd == pytest.approx(2.3 * 100 * 0.83)


def test_long_stop():
    df = make_df(LONG_BARS + [(100.9, 101.0, 98.9, 99.2)])
    t = run_backtest(df, frame(sig()), NOCOST).trades.iloc[0]
    assert t.exit_reason == "stop" and t.r == pytest.approx(-1.0) and t.pnl_usd == pytest.approx(-100.0)


def test_same_bar_stop_and_target_resolves_to_stop():
    df = make_df(LONG_BARS + [(100.9, 102.6, 98.9, 100.0)])
    t = run_backtest(df, frame(sig()), NOCOST).trades.iloc[0]
    assert t.exit_reason == "stop" and t.r == pytest.approx(-1.0) and bool(t.same_bar_conflict)


def test_stop_gap_fills_at_open_not_stop():
    df = make_df(LONG_BARS + [(98.0, 98.5, 97.5, 98.2)])
    t = run_backtest(df, frame(sig()), NOCOST).trades.iloc[0]
    # exits at the open 98.0 -> (98-100)/1 = -2R, lots 1.0 -> -200
    assert t.exit == pytest.approx(98.0) and t.r == pytest.approx(-2.0) and t.pnl_usd == pytest.approx(-200.0)


def test_slippage_hurts_entry_and_stop_fill():
    df = make_df(LONG_BARS + [(100.9, 101.0, 98.9, 99.2)])
    t = run_backtest(df, frame(sig()), BacktestConfig(spread=0.0, slippage=0.1)).trades.iloc[0]
    # entry 100.1, stop exit 98.9 -> risk 1.1, loss 1.2 -> -1.0909R
    assert t.entry == pytest.approx(100.1) and t.exit == pytest.approx(98.9)
    assert t.r == pytest.approx(-1.2 / 1.1)


def test_stop_inside_trigger_bar_is_a_loss_even_if_flagged_only():
    bars = [(99.5, 100.5, 99.4, 100.4), (100.0, 100.2, 99.9, 100.1), (100.1, 101, 100, 100.9)]
    t = run_backtest(make_df(bars), frame(sig(same_bar=True)), NOCOST).trades.iloc[0]
    assert t.exit_reason == "stop_trigger_bar" and t.r == pytest.approx(-1.0) and t.bars_held == 0
    # and detected from the data alone when the flag is absent
    bars[1] = (100.0, 100.2, 98.8, 99.5)
    t = run_backtest(make_df(bars), frame(sig(same_bar=False)), NOCOST).trades.iloc[0]
    assert t.exit_reason == "stop_trigger_bar"


def test_target_inside_trigger_bar_is_ignored():
    bars = [(99.5, 100.5, 99.4, 100.4), (100.0, 103.0, 99.9, 102.0), (102.0, 102.1, 101.9, 102.0)]
    t = run_backtest(make_df(bars), frame(sig()), NOCOST).trades.iloc[0]
    assert t.exit_reason == "eod" and t.exit_time == make_df(bars).index[2]
    assert t.r == pytest.approx(2.0)  # closed at 102.0


def test_short_stop_uses_ask_so_spread_triggers_it_early():
    bars = [(100.5, 100.6, 99.6, 99.7), (100.0, 100.1, 99.8, 99.9),
            (99.9, 100.7, 99.0, 99.5),  # 100.7 + 0.2 = 100.9 < 101 -> alive
            (99.5, 100.85, 99.4, 100.0)]  # 100.85 + 0.2 = 101.05 >= 101 -> stopped
    s = frame(sig(SHORT, fill=100.0, stop=101.0, target=97.5))
    t = run_backtest(make_df(bars), s, BacktestConfig(spread=0.2)).trades.iloc[0]
    assert t.exit_reason == "stop" and t.exit_time == make_df(bars).index[3]
    assert t.entry == pytest.approx(100.0)  # short sells at the bid
    assert t.r == pytest.approx(-1.0)
    # same data with no spread: the 100.85 high does not reach 101 -> not stopped
    t0 = run_backtest(make_df(bars), s, NOCOST).trades.iloc[0]
    assert t0.exit_reason == "eod"


def test_short_target_needs_ask_to_reach_it():
    bars = [(100.5, 100.6, 99.6, 99.7), (100.0, 100.1, 99.8, 99.9),
            (99.9, 100.0, 97.4, 97.8),  # 97.4 + 0.2 = 97.6 > 97.5 -> not yet
            (97.8, 97.9, 97.3, 97.4)]  # 97.3 + 0.2 = 97.5 <= 97.5 -> target
    s = frame(sig(SHORT, fill=100.0, stop=101.0, target=97.5))
    t = run_backtest(make_df(bars), s, BacktestConfig(spread=0.2)).trades.iloc[0]
    assert t.exit_reason == "target" and t.exit_time == make_df(bars).index[3]
    assert t.r == pytest.approx(2.5)


def test_position_size_floors_to_lot_step_and_respects_min_lot():
    df = make_df(LONG_BARS + [(100.9, 102.6, 100.8, 102.5)])
    s = frame(sig(stop=99.1, target=103.0))
    cfg = BacktestConfig(spread=0.0, risk_pct=0.5)
    # risk 0.9 -> 50 / (0.9*100) = 0.5556 -> 0.55 lots
    assert run_backtest(df, s, cfg).trades.iloc[0].lots == pytest.approx(0.55)
    tiny = BacktestConfig(spread=0.0, initial_equity=50.0)  # 0.5 / 100 = 0.005 < 0.01
    res = run_backtest(df, frame(sig()), tiny)
    assert res.trades.empty and res.skipped.iloc[0].reason == "below_min_lot"


def test_one_position_at_a_time():
    flat = (100.3, 100.5, 100.0, 100.3)
    bars = [(99.5, 100.5, 99.4, 100.4), (100.0, 100.2, 99.9, 100.1), flat, flat,
            (100.3, 102.6, 100.2, 102.5), flat, flat, flat]
    s = frame(sig(),  # A: exits on bar 4
              sig(break_pos=2, trigger_pos=3),  # B: placed while A is open -> dropped
              sig(break_pos=4, trigger_pos=5))  # C: placed on A's exit bar -> allowed
    res = run_backtest(make_df(bars), s, NOCOST)
    assert len(res.trades) == 2 and list(res.skipped.reason) == ["overlap"]
    assert list(res.trades.signal_id) == [0, 2]


def test_min_net_rr_filter_uses_spread_adjusted_rr():
    df = make_df(LONG_BARS + [(100.9, 102.6, 100.8, 102.5)])
    res = run_backtest(df, frame(sig()), BacktestConfig(spread=0.2, min_net_rr=2.0))
    assert res.trades.empty and res.skipped.iloc[0].reason == "net_rr"  # 1.917 < 2.0
    assert len(run_backtest(df, frame(sig()), BacktestConfig(spread=0.2, min_net_rr=1.9)).trades) == 1


def test_time_stop_exits_at_close_of_last_allowed_bar():
    df = make_df(LONG_BARS + [(100.9, 102.6, 100.8, 102.5)])
    t = run_backtest(df, frame(sig()), BacktestConfig(spread=0.0, max_hold_bars=1)).trades.iloc[0]
    assert t.exit_reason == "time" and t.exit == pytest.approx(100.9) and t.r == pytest.approx(0.9)


def test_compounding_vs_fixed_risk():
    win = (100.9, 102.6, 100.8, 102.5)
    flat = (100.3, 100.5, 100.0, 100.3)
    bars = LONG_BARS[:2] + [win, (100.0, 100.5, 99.9, 100.1), (100.0, 100.2, 99.9, 100.1), win]
    s = frame(sig(), sig(break_pos=2, trigger_pos=4))
    comp = run_backtest(make_df(bars), s, NOCOST).trades
    fixed = run_backtest(make_df(bars), s, BacktestConfig(spread=0.0, compounding=False)).trades
    assert comp.lots.iloc[1] == pytest.approx(1.02)  # 10250*1%/100 = 1.0250 -> 1.02
    assert fixed.lots.iloc[1] == pytest.approx(1.0)


def test_summarize_and_breakdowns():
    df = make_df(LONG_BARS + [(100.9, 102.6, 100.8, 102.5)])
    res = run_backtest(df, frame(sig(session="london", range_in_atr=1.0)), NOCOST)
    s = summarize(res.trades)
    assert s["trades"] == 1 and s["win_rate"] == 1.0 and s["expectancy_r"] == pytest.approx(2.5)
    assert set(breakdowns(res.trades)) >= {"direction", "month", "entry_hour_utc", "session"}

    t = pd.DataFrame({"r": [2.5, -1.0, -1.0, 2.5], "equity_after": [10250, 10150, 10050, 10300],
                      "bars_held": [3, 2, 2, 5], "same_bar_conflict": [False] * 4,
                      "exit_reason": ["target", "stop", "stop", "target"]})
    m = summarize(t, 10_000.0)
    assert m["profit_factor"] == pytest.approx(5.0 / 2.0) and m["win_rate"] == 0.5
    assert m["expectancy_r"] == pytest.approx(0.75) and m["total_r"] == pytest.approx(3.0)
    assert m["max_drawdown_r"] == pytest.approx(2.0) and m["max_losing_streak"] == 2
    assert m["max_drawdown_pct"] == pytest.approx((10250 - 10050) / 10250 * 100)
    assert summarize(pd.DataFrame()) == {"trades": 0}


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        run_backtest(make_df(LONG_BARS), pd.DataFrame({"direction": [1]}), NOCOST)


