"""Backtest engine for break-Out signals.

Input contract
--------------
``df``       the same OHLCV frame (UTC index, *bid* prices as exported by MT5) that was
             passed to ``scan_breakout`` -- signal positions index into it.
``signals``  ``ScanResult.to_frame()``; needs direction, break_pos, trigger_pos,
             fill_price, stop, target (stopped_same_bar is used when present).

What the engine owns (breakout.py deliberately leaves it to this layer)
-----------------------------------------------------------------------
* Costs: MT5 exports are bid-only, so the ask is modelled as ``bid + spread``.
    LONG  buys at the ask, sells at the bid  -> pays the spread on entry.
    SHORT sells at the bid, buys at the ask  -> pays the spread on exit
          (a short's stop/target are tested against ``bid + spread``).
* Pessimistic intrabar rules (M1 bars carry no tick order):
    - stop and target inside the same bar -> the stop wins;
    - the trigger bar can only stop you out (its target is ignored), and a bar the
      signal flags ``stopped_same_bar`` is a loss;
    - a stop that gaps is filled at the open (worse), a target that gaps at the target.
* Exit strictly at the stop or the structural target -- no averaging in, no moving stops.
* One position at a time. An order placed at the close of ``break_pos`` is dropped if a
  position is still open then.
* Fixed-fractional sizing in lots (XAU/USD: 1 lot = ``contract_size`` oz).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

LONG, SHORT = 1, -1  # same sign convention as primitives / rr.py

REQUIRED = ("direction", "break_pos", "trigger_pos", "fill_price", "stop", "target")
PASSTHROUGH = ("session", "rel_volume", "range_in_atr", "level_touches", "level_price",
               "rr_planned", "rr_fill")


@dataclass(frozen=True)
class BacktestConfig:
    """Prices are in price units (USD per oz for XAU/USD)."""

    spread: float = 0.15  # constant spread; 15 points on a 2-digit gold quote = $0.15
    slippage: float = 0.0  # extra adverse price on entry and on stop fills
    commission_per_lot: float = 0.0  # round-turn, account currency (0 on Exness standard)
    initial_equity: float = 10_000.0
    risk_pct: float = 1.0  # % of equity risked per trade (playbook: 0.5-1)
    compounding: bool = True  # risk % of current equity, else of initial equity
    contract_size: float = 100.0  # oz per lot
    lot_step: float = 0.01
    min_lot: float = 0.01
    max_hold_bars: int | None = None  # time stop; None = hold until stop/target/end of data
    min_net_rr: float | None = None  # drop trades whose R:R *after spread* is below this


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    skipped: pd.DataFrame
    config: BacktestConfig


def _floor_lots(raw: float, step: float) -> float:
    return round(math.floor(raw / step + 1e-9) * step, 8)


def _exit_after_entry(opn, high, low, close, i, d, stop, target, stopped_same_bar, cfg):
    """Exit for a position entered inside bar ``i``. Returns (pos, price, reason, conflict)."""
    s, k = cfg.spread, cfg.slippage
    n = len(close)

    # --- trigger bar: only the stop can be known to come after the entry ---
    stop_here = (low[i] <= stop) if d == LONG else (high[i] + s >= stop)
    if stopped_same_bar or stop_here:
        px = stop - k if d == LONG else stop + k
        return i, px, "stop_trigger_bar", False

    start = i + 1
    end = n if cfg.max_hold_bars is None else min(n, start + cfg.max_hold_bars)
    if end > start:
        lo, hi, op = low[start:end], high[start:end], opn[start:end]
        if d == LONG:
            stop_m, tgt_m = lo <= stop, hi >= target
        else:
            stop_m, tgt_m = hi + s >= stop, lo + s <= target
        ev = stop_m | tgt_m
        if ev.any():
            e = int(ev.argmax())
            j = start + e
            conflict = bool(stop_m[e] and tgt_m[e])
            if stop_m[e]:  # stop wins any tie
                if d == LONG:
                    px = op[e] if op[e] <= stop else stop - k
                else:
                    ask_open = op[e] + s
                    px = ask_open if ask_open >= stop else stop + k
                return j, px, "stop", conflict
            return j, target, "target", False

    # no stop/target: time stop or end of data, closed at the last bar's close
    j = max(end - 1, i)
    px = close[j] if d == LONG else close[j] + s
    reason = "time" if (cfg.max_hold_bars is not None and start + cfg.max_hold_bars <= n) else "eod"
    return j, px, reason, False


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, cfg: BacktestConfig | None = None) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    missing = [c for c in REQUIRED if c not in signals.columns]
    if missing:
        raise ValueError(f"signals is missing columns: {missing}")

    opn = df["open"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    idx = df.index
    n = len(df)

    sig = signals.sort_values(["break_pos", "trigger_pos"], kind="stable")
    extras = [c for c in PASSTHROUGH if c in sig.columns]
    same_bar_flag = "stopped_same_bar" in sig.columns

    trades: list[dict] = []
    skipped: list[dict] = []
    equity = cfg.initial_equity
    busy_until = -1  # bar position of the last exit

    def skip(sid, row, reason):
        skipped.append({"signal_id": sid, "break_time": idx[int(row["break_pos"])], "reason": reason})

    for sid, row in sig.iterrows():
        bp, tp, d = int(row["break_pos"]), int(row["trigger_pos"]), int(row["direction"])
        if tp >= n:
            skip(sid, row, "trigger_out_of_range")
            continue
        if bp < busy_until:
            skip(sid, row, "overlap")
            continue
        if equity <= 0:
            skip(sid, row, "ruin")
            continue

        stop, target = float(row["stop"]), float(row["target"])
        raw_fill = float(row["fill_price"])
        entry = raw_fill + cfg.spread + cfg.slippage if d == LONG else raw_fill - cfg.slippage

        risk_px = (entry - stop) * d
        if not risk_px > 0:
            skip(sid, row, "bad_risk")
            continue
        rr_net = (target - entry) * d / risk_px
        if cfg.min_net_rr is not None and not rr_net >= cfg.min_net_rr:
            skip(sid, row, "net_rr")
            continue

        base = equity if cfg.compounding else cfg.initial_equity
        lots = _floor_lots(base * cfg.risk_pct / 100.0 / (risk_px * cfg.contract_size), cfg.lot_step)
        if lots < cfg.min_lot:
            skip(sid, row, "below_min_lot")
            continue

        j, exit_px, reason, conflict = _exit_after_entry(
            opn, high, low, close, tp, d, stop, target,
            bool(row["stopped_same_bar"]) if same_bar_flag else False, cfg,
        )
        busy_until = j

        r = (exit_px - entry) * d / risk_px
        pnl = (exit_px - entry) * d * cfg.contract_size * lots - cfg.commission_per_lot * lots
        equity += pnl
        trades.append({
            "signal_id": sid, "direction": d,
            "break_time": idx[bp], "entry_time": idx[tp], "exit_time": idx[j],
            "entry": entry, "stop": stop, "target": target, "exit": exit_px,
            "risk_px": risk_px, "rr_net": rr_net, "exit_reason": reason,
            "bars_held": j - tp, "r": r, "lots": lots, "pnl_usd": pnl, "equity_after": equity,
            "same_bar_conflict": conflict,
            **{c: row[c] for c in extras},
        })

    return BacktestResult(
        trades=pd.DataFrame(trades),
        skipped=pd.DataFrame(skipped, columns=["signal_id", "break_time", "reason"]),
        config=cfg,
    )