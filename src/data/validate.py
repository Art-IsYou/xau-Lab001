"""validate.py - integrity checks for standardized M1 data.

Policy (important for a trading system):
  * REMOVE rows that are provably corrupt (duplicates, NaN/non-positive prices,
    impossible OHLC). Each removal is counted in the report.
  * FLAG but never alter things that may be real market behaviour (spikes,
    gaps). Gaps are NOT forward-filled: inventing bars would create fake
    candles that setups could trade on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]


@dataclass
class ValidationReport:
    rows_in: int = 0
    rows_out: int = 0
    unsorted: bool = False
    duplicates_removed: int = 0
    nan_rows_removed: int = 0
    non_positive_removed: int = 0
    bad_ohlc_removed: int = 0
    negative_volume_removed: int = 0
    gaps: pd.DataFrame = field(default_factory=pd.DataFrame)   # start, end, missing_bars, kind
    spikes: pd.DataFrame = field(default_factory=pd.DataFrame)  # flagged bars (kept in data)

    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out

    def summary(self) -> str:
        lines = [
            f"rows: {self.rows_in} in -> {self.rows_out} out ({self.rows_removed} removed)",
            f"  unsorted input: {self.unsorted} (sorted)",
            f"  duplicate timestamps removed: {self.duplicates_removed}",
            f"  NaN rows removed: {self.nan_rows_removed}",
            f"  non-positive price rows removed: {self.non_positive_removed}",
            f"  impossible OHLC rows removed: {self.bad_ohlc_removed}",
            f"  negative volume rows removed: {self.negative_volume_removed}",
        ]
        if len(self.gaps):
            counts = self.gaps["kind"].value_counts().to_dict()
            lines.append(f"  gaps (kept, not filled): {counts}")
        else:
            lines.append("  gaps: none")
        lines.append(f"  spike bars flagged (kept): {len(self.spikes)}")
        return "\n".join(lines)


def _classify_gap(delta: pd.Timedelta) -> str:
    if delta > pd.Timedelta(hours=40):
        return "weekend"
    if delta > pd.Timedelta(hours=2):
        return "unexpected_long"       # holiday / missing data - inspect
    if delta > pd.Timedelta(minutes=10):
        return "break"                 # daily break or a thin-market stretch
    return "short"                     # a few missing minutes, usually no ticks


def find_gaps(df: pd.DataFrame, bar: pd.Timedelta = pd.Timedelta(minutes=1)) -> pd.DataFrame:
    """Return every place where consecutive bars are further apart than `bar`."""
    if len(df) < 2:
        return pd.DataFrame(columns=["start", "end", "missing_bars", "kind"])
    idx = df.index.to_series()
    delta = idx.diff()
    mask = delta > bar
    out = pd.DataFrame({
        "start": idx.shift(1)[mask].array,
        "end": idx[mask].array,
        "missing_bars": (delta[mask] / bar - 1).astype(int).to_numpy(),
        "kind": [_classify_gap(d) for d in delta[mask]],
    })
    return out.reset_index(drop=True)


def find_spikes(df: pd.DataFrame, threshold: float = 15.0, window: int = 500) -> pd.DataFrame:
    """Flag bars whose close-to-close move is `threshold` x the typical move.

    'Typical' is a rolling median of absolute returns (robust to the spikes
    themselves). Real news spikes will show up here too - that is intended;
    they are flagged for review, not removed.
    """
    ret = df["close"].diff().abs()
    typical = ret.rolling(window, min_periods=50).median()
    typical = typical.where(typical > 0)
    score = ret / typical
    flagged = df[score > threshold].copy()
    flagged["spike_score"] = score[score > threshold]
    return flagged


def validate(df: pd.DataFrame, spike_threshold: float = 15.0):
    """Return (clean_df, ValidationReport). Input is never modified."""
    rep = ValidationReport(rows_in=len(df))
    d = df.copy()

    rep.unsorted = not d.index.is_monotonic_increasing
    if rep.unsorted:
        d = d.sort_index()

    dup = d.index.duplicated(keep="last")
    rep.duplicates_removed = int(dup.sum())
    d = d[~dup]

    nan = d[OHLC].isna().any(axis=1)
    rep.nan_rows_removed = int(nan.sum())
    d = d[~nan]

    nonpos = (d[OHLC] <= 0).any(axis=1)
    rep.non_positive_removed = int(nonpos.sum())
    d = d[~nonpos]

    hi_ok = d["high"] >= d[["open", "close", "low"]].max(axis=1)
    lo_ok = d["low"] <= d[["open", "close", "high"]].min(axis=1)
    bad = ~(hi_ok & lo_ok)
    rep.bad_ohlc_removed = int(bad.sum())
    d = d[~bad]

    if "volume" in d.columns:
        negv = d["volume"] < 0
        rep.negative_volume_removed = int(negv.sum())
        d = d[~negv]
        d["volume"] = d["volume"].fillna(0)

    rep.rows_out = len(d)
    rep.gaps = find_gaps(d)
    rep.spikes = find_spikes(d, threshold=spike_threshold)
    return d, rep