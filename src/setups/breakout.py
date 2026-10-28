"""break-Out setup: high-volume momentum continuation through a key structural level.

Playbook (setup - break-Out.pdf), mapped onto primitives
---------------------------------------------------------
* Break candle qualifies (opened on the near side of the level, closed beyond it,
  body >= 50% of range, >= 50% of the body beyond the level) -> ``breakout_candle_mask``
* The level is a *key* structural level: a cluster of >= ``level_min_touches`` swing
  points that nothing has closed through yet               -> ``cluster_levels`` / ``level_is_unbroken``
* M15 structure aligned with the break direction            -> ``market_structure`` / ``align_htf_to_base``
* Liquidity taken during the prior consolidation/pullback   -> ``find_sweeps`` / ``liquidity_taken_in_window``
* Participation: relative volume on the break candle, and (by default) major session only
                                                            -> ``volume_gate`` / ``session_mask``
* Entry candle-over-candle, stop beyond the break candle    -> ``coc_levels`` / ``coc_trigger``
* R:R >= 2.5 holds even with entry expansion                -> ``risk_reward``
* Exhaustion candle => wide stop => R:R collapses => skipped by the R:R gate
  (``max_range_atr`` is an optional cheap pre-filter, off by default).

Decision vs. fill
-----------------
Every *qualification* gate uses only data up to and including the break candle's
close. The CoC trigger (bars after the break) is the order-fill simulation, so a
signal carries both ``break_pos`` (when the decision is knowable) and
``trigger_pos`` (when the entry fills). A backtester must place the stop-order at
the close of ``break_pos``.

Not handled here (it is trade management, not detection): "Range after the Break"
=> exit strictly at the stop, never average in. The backtest layer owns that.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from primitives import (
    LONG,
    MAJOR_SESSIONS,
    SHORT,
    align_htf_to_base,
    atr,
    candle_parts,
    cluster_levels,
    coc_levels,
    coc_trigger,
    find_sweeps,
    find_swings,
    level_is_unbroken,
    liquidity_taken_in_window,
    market_structure,
    next_level_beyond,
    range_in_atr,
    relative_volume,
    risk_reward,
    session_label,
    session_mask,
    structure_aligned,
)

# Gate order. The funnel reports how many break candidates survive each gate.
STAGES = (
    "candle",  # right direction + body ratio
    "volume",  # relative volume on the break candle
    "session",  # inside a major session (if major_session_only)
    "htf",  # M15 structure aligned
    "level",  # breaks an unbroken key level with >= min_beyond_frac of its body
    "liquidity",  # sweep in the pullback window
    "exhaustion",  # optional range_in_atr cap
    "target",  # a structural target exists beyond the entry
    "rr",  # planned R:R >= min_rr (with expansion)
    "trigger",  # CoC entry fills within max_wait_bars
    "fill_rr",  # R:R still >= min_rr at the actual fill (gap protection)
)


@dataclass(frozen=True)
class BreakoutConfig:
    """Every threshold in one place. Prices are in price units (USD for XAU/USD).

    Defaults are starting points taken from the playbook where it states a number
    (50% body, 50% beyond, R:R 2.5) and educated guesses elsewhere. Tune them in
    the backtest, not here.
    """

    # swings / levels (on the entry timeframe)
    swing_left: int = 3
    swing_right: int = 3
    atr_period: int = 14
    level_tol_atr: float = 0.25  # cluster tolerance = this * ATR before the break candle
    level_min_touches: int = 2  # "key" level = at least this many swing points
    level_max_age_bars: int | None = 600

    # break candle
    min_body_ratio: float = 0.5
    min_beyond_frac: float = 0.3
    max_range_atr: float | None = None  # optional exhaustion pre-filter (R:R gate is authoritative)

    # participation
    volume_window: int = 20
    min_rel_volume: float = 1.5
    major_session_only: bool = True
    sessions: tuple = MAJOR_SESSIONS

    # higher-timeframe alignment
    htf_period: str = "15min"
    htf_swing_left: int = 2
    htf_swing_right: int = 2

    # liquidity
    liq_lookback_bars: int = 30  # sweep must occur within this many bars before the break candle
    liq_reclaim_bars: int = 1
    liq_after_level: bool = True  # sweep must come after the level's latest touch (i.e. in the pullback)
    sweep_max_age_bars: int | None = 500

    # entry / risk
    entry_buffer: float = 0.0
    stop_buffer: float = 0.0
    max_wait_bars: int = 3
    min_rr: float = 2.5
    rr_expansion: float = 0.0  # assumed adverse slippage on entry for the planned R:R
    target_min_touches: int = 1  # any swing level can serve as a target

    directions: tuple = (LONG, SHORT)


@dataclass(frozen=True)
class BreakoutSignal:
    direction: int
    break_pos: int
    break_time: pd.Timestamp
    trigger_pos: int
    trigger_time: pd.Timestamp
    level_price: float
    level_touches: int
    entry_planned: float
    fill_price: float
    stop: float
    target: float
    rr_planned: float
    rr_fill: float
    rel_volume: float
    range_in_atr: float
    session: str
    sweep_pos: int
    swept_price: float
    stopped_same_bar: bool  # trigger bar also reached the stop; intrabar order unknown


@dataclass
class ScanResult:
    signals: list
    funnel: dict  # {"LONG": {"bars": n, "candle": n, ...}, "SHORT": {...}}

    def to_frame(self) -> pd.DataFrame:
        if not self.signals:
            return pd.DataFrame(columns=[f for f in BreakoutSignal.__dataclass_fields__])
        return pd.DataFrame([asdict(s) for s in self.signals])

    def funnel_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.funnel)


def is_break_candle(o: float, h: float, l: float, c: float, level: float, direction: int,
                    min_body_ratio: float, min_beyond_frac: float) -> bool:
    """Scalar twin of ``primitives.breakout_candle_mask`` for one candle.

    The primitive is vectorized over a whole DataFrame; calling it once per
    (candidate candle, level) costs ~10 ms of pandas overhead each, which made a
    scan of a few thousand M1 bars take a minute. tests/test_breakout.py asserts
    this stays identical to the primitive on random candles.
    """
    rng, body = h - l, abs(c - o)
    if not (rng > 0 and body > 0):
        return False
    if body / rng < min_body_ratio:
        return False
    if direction == LONG:
        return bool(o <= level and c > level and c > o and (c - max(o, level)) / body >= min_beyond_frac)
    return bool(o >= level and c < level and c < o and (min(o, level) - c) / body >= min_beyond_frac)


def _swing_window(swings: pd.DataFrame, conf: np.ndarray, hi_pos: int, max_age: int | None) -> pd.DataFrame:
    """Swings confirmed by ``hi_pos`` (and not older than ``max_age``). ``swings`` is
    sorted by ``confirmed_pos``, so this is a cheap slice instead of a full filter."""
    hi = int(np.searchsorted(conf, hi_pos, side="right"))
    lo = 0 if max_age is None else int(np.searchsorted(conf, hi_pos - max_age, side="left"))
    return swings.iloc[lo:hi]


def scan_breakout(
    df: pd.DataFrame,
    htf_df: pd.DataFrame | None = None,
    cfg: BreakoutConfig | None = None,
    *,
    trend: pd.Series | None = None,
) -> ScanResult:
    """Find every break-Out signal in ``df`` (entry-timeframe OHLCV, UTC index).

    htf_df: M15 OHLCV used for the structure gate. Ignored if ``trend`` is given.
    trend:  precomputed HTF trend (+1/0/-1) aligned to ``df.index``; pass it when
            scanning repeatedly (parameter sweeps) so M15 structure is computed once.
    """
    cfg = cfg or BreakoutConfig()
    if trend is None:
        if htf_df is None:
            raise ValueError("provide htf_df (M15 bars) or a precomputed trend series")
        trend = align_htf_to_base(
            market_structure(htf_df, cfg.htf_swing_left, cfg.htf_swing_right), df.index, cfg.htf_period
        )
    else:
        trend = trend.reindex(df.index)

    n = len(df)
    parts = candle_parts(df)
    atr_prev = atr(df, cfg.atr_period).shift(1).to_numpy(float)
    rng_atr = range_in_atr(df, cfg.atr_period).to_numpy(float)
    relvol = relative_volume(df, cfg.volume_window).to_numpy(float)
    in_session = session_mask(df.index, cfg.sessions).to_numpy(bool)
    labels = session_label(df.index, cfg.sessions).to_numpy(object)
    swings = find_swings(df, cfg.swing_left, cfg.swing_right)
    conf = swings["confirmed_pos"].to_numpy()
    sweeps = find_sweeps(df, swings, cfg.liq_reclaim_bars, cfg.sweep_max_age_bars)
    opn = df["open"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    clo = df["close"].to_numpy(float)

    signals: list[BreakoutSignal] = []
    funnel: dict = {}

    for d in cfg.directions:
        name = "LONG" if d == LONG else "SHORT"
        counts = {"bars": n, **{s: 0 for s in STAGES}}

        # ---- vectorized gates (cheap, no level search needed) ----
        m = ((parts["direction"] == d) & (parts["body_ratio"] >= cfg.min_body_ratio)).to_numpy(bool)
        counts["candle"] = int(m.sum())
        m = m & (relvol >= cfg.min_rel_volume)
        counts["volume"] = int(m.sum())
        if cfg.major_session_only:
            m = m & in_session
        counts["session"] = int(m.sum())
        m = m & structure_aligned(trend, d).to_numpy(bool)
        counts["htf"] = int(m.sum())

        # ---- per-candidate gates ----
        for t in np.flatnonzero(m):
            t = int(t)
            if t < 1 or not np.isfinite(atr_prev[t]):
                continue

            sub = _swing_window(swings, conf, t - 1, cfg.level_max_age_bars)
            # Cheap necessary condition: a level's price is one of its swing prices, so the
            # candle must have crossed at least one same-side swing between its open and close.
            sp = sub["price"].to_numpy(float)
            sk = sub["kind"].to_numpy()
            if d == LONG:
                crossed = ((sk == "high") & (sp >= opn[t]) & (sp < clo[t])).any()
            else:
                crossed = ((sk == "low") & (sp <= opn[t]) & (sp > clo[t])).any()
            if not crossed:
                continue
            levels = cluster_levels(sub, t - 1, cfg.level_tol_atr * atr_prev[t], 1, cfg.level_max_age_bars)
            kind = "resistance" if d == LONG else "support"
            cands = [lv for lv in levels if lv.kind == kind and lv.touches >= cfg.level_min_touches]
            cands.sort(key=lambda lv: lv.price, reverse=(d == LONG))  # most-cleared level first
            level = None
            for lv in cands:
                if is_break_candle(opn[t], high[t], low[t], clo[t], lv.price, d,
                                   cfg.min_body_ratio, cfg.min_beyond_frac) and level_is_unbroken(df, lv, t - 1):
                    level = lv
                    break
            if level is None:
                continue
            counts["level"] += 1

            sw = liquidity_taken_in_window(sweeps, d, t - cfg.liq_lookback_bars, t - 1)
            if cfg.liq_after_level:
                sw = sw[sw["sweep_pos"] >= level.last_pos]
            if sw.empty:
                continue
            sweep = sw.iloc[-1]
            counts["liquidity"] += 1

            if cfg.max_range_atr is not None and not rng_atr[t] <= cfg.max_range_atr:
                continue
            counts["exhaustion"] += 1

            entry, stop = coc_levels(high[t], low[t], d, cfg.entry_buffer, cfg.stop_buffer)
            target_levels = [lv for lv in levels if lv.touches >= cfg.target_min_touches]
            target = next_level_beyond(target_levels, entry, d)
            if target is None:
                continue
            counts["target"] += 1

            rr_planned = risk_reward(entry, stop, target.price, d, cfg.rr_expansion)
            if not rr_planned >= cfg.min_rr:  # NaN (non-positive risk) fails too
                continue
            counts["rr"] += 1

            trig = coc_trigger(df, t, d, cfg.entry_buffer, cfg.stop_buffer, cfg.max_wait_bars)
            if trig is None:
                continue
            counts["trigger"] += 1

            rr_fill = risk_reward(trig.fill_price, trig.stop_price, target.price, d)
            if not rr_fill >= cfg.min_rr:
                continue
            counts["fill_rr"] += 1

            signals.append(
                BreakoutSignal(
                    direction=d,
                    break_pos=t,
                    break_time=df.index[t],
                    trigger_pos=trig.pos,
                    trigger_time=df.index[trig.pos],
                    level_price=level.price,
                    level_touches=level.touches,
                    entry_planned=entry,
                    fill_price=trig.fill_price,
                    stop=trig.stop_price,
                    target=target.price,
                    rr_planned=float(rr_planned),
                    rr_fill=float(rr_fill),
                    rel_volume=float(relvol[t]),
                    range_in_atr=float(rng_atr[t]),
                    session=str(labels[t]),
                    sweep_pos=int(sweep["sweep_pos"]),
                    swept_price=float(sweep["swept_price"]),
                    stopped_same_bar=trig.stopped_same_bar,
                )
            )
        funnel[name] = counts

    signals.sort(key=lambda s: (s.break_pos, s.direction))
    return ScanResult(signals, funnel)