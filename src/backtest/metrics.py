"""Performance metrics. Everything is expressed in R first (price-based, after costs);
dollars and drawdown-% come from the engine's compounding equity column."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _profit_factor(r: np.ndarray) -> float:
    gain, loss = r[r > 0].sum(), -r[r < 0].sum()
    if loss > 0:
        return float(gain / loss)
    return float("inf") if gain > 0 else float("nan")


def _max_losing_streak(r: np.ndarray) -> int:
    best = cur = 0
    for x in r:
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    return best


def bootstrap_mean_ci(r: np.ndarray, n_boot: int = 5000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI for mean R. With few trades this is wide -- that is the point."""
    if len(r) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def summarize(trades: pd.DataFrame, initial_equity: float = 10_000.0) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["r"].to_numpy(float)
    cum = np.cumsum(r)
    dd_r = (np.maximum.accumulate(np.r_[0.0, cum])[1:] - cum).max()
    eq = np.r_[initial_equity, trades["equity_after"].to_numpy(float)]
    dd_pct = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    lo, hi = bootstrap_mean_ci(r)
    wins, losses = r[r > 0], r[r < 0]
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "expectancy_r": float(r.mean()),
        "expectancy_r_ci95": (lo, hi),
        "total_r": float(r.sum()),
        "avg_win_r": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss_r": float(losses.mean()) if len(losses) else float("nan"),
        "profit_factor": _profit_factor(r),
        "max_drawdown_r": float(dd_r),
        "max_drawdown_pct": float(dd_pct),
        "max_losing_streak": _max_losing_streak(r),
        "final_equity": float(eq[-1]),
        "return_pct": float((eq[-1] / initial_equity - 1) * 100),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "same_bar_conflicts": int(trades["same_bar_conflict"].sum()),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }


def _basic(r: pd.Series) -> dict:
    a = r.to_numpy(float)
    return {"trades": len(a), "win_rate": float((a > 0).mean()), "avg_r": float(a.mean()),
            "total_r": float(a.sum()), "profit_factor": _profit_factor(a)}


def breakdown(trades: pd.DataFrame, key) -> pd.DataFrame:
    """Per-group trade count / win rate / avg R / total R / PF. ``key`` is a column
    name or a Series aligned to ``trades``."""
    if trades.empty:
        return pd.DataFrame()
    k = trades[key] if isinstance(key, str) else key
    rows = {g: _basic(grp["r"]) for g, grp in trades.groupby(k, observed=True)}
    return pd.DataFrame(rows).T.astype({"trades": int})


def breakdowns(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The cuts that tell an edge from a lucky stretch."""
    if trades.empty:
        return {}
    t = trades
    out = {
        "direction": breakdown(t, t["direction"].map({1: "LONG", -1: "SHORT"})),
        "month": breakdown(t, t["entry_time"].dt.strftime("%Y-%m")),
        "entry_hour_utc": breakdown(t, t["entry_time"].dt.hour),
        "exit_reason": breakdown(t, "exit_reason"),
    }
    if "session" in t:
        out["session"] = breakdown(t, "session")
    if "range_in_atr" in t and t["range_in_atr"].nunique() >= 3 and len(t) >= 9:
        out["break_range_atr_tercile"] = breakdown(
            t, pd.qcut(t["range_in_atr"], 3, labels=["small", "mid", "large"], duplicates="drop")
        )
    return out