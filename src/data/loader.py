"""loader.py - read MT5 CSV exports into a standard, UTC-indexed DataFrame.

Standard output schema (every data source must produce exactly this):
    index   : tz-aware UTC DatetimeIndex named "time" = bar OPEN time
    columns : open, high, low, close, volume  (+ spread if the file has it)
    volume  : tick volume (spot gold has no real centralized volume)

The loader only reads and normalizes. Cleaning lives in validate.py and
timeframe conversion in resample.py.
"""
from __future__ import annotations

import csv
import re
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

OHLC = ["open", "high", "low", "close"]
STANDARD_COLUMNS = OHLC + ["volume"]

# Accepted header names (lower-cased, <> stripped) -> standard name
_ALIASES = {
    "time": "time", "datetime": "time", "timestamp": "time", "date": "date",
    "open": "open", "high": "high", "low": "low", "close": "close",
    "volume": "volume", "tickvol": "volume", "tick_volume": "volume",
    "tickvolume": "volume",
    "vol": "real_volume", "real_volume": "real_volume",
    "spread": "spread",
}


def _clean_header(name: str) -> str:
    return re.sub(r"[<>\s]", "", str(name)).lower()


def _sniff_delimiter(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        sample = fh.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _parse_source_tz(source_tz):
    """'UTC', '+2' / '-5' / 'UTC+2' (fixed hours), or an IANA name."""
    if source_tz is None:
        return None
    if isinstance(source_tz, (int, float)):
        return timedelta(hours=float(source_tz))
    s = str(source_tz).strip()
    if s.upper() in ("UTC", "GMT", "Z"):
        return timedelta(0)
    m = re.fullmatch(r"(?:UTC|GMT)?\s*([+-]\d{1,2})(?::?(\d{2}))?", s, re.I)
    if m:
        hours = int(m.group(1))
        minutes = int(m.group(2) or 0)
        sign = -1 if m.group(1).startswith("-") else 1
        return timedelta(hours=hours, minutes=sign * minutes)
    return ZoneInfo(s)  # raises if unknown - fail loudly


def _to_utc(naive: pd.Series, tz) -> pd.DatetimeIndex:
    """Convert naive broker timestamps to UTC given a fixed offset or ZoneInfo."""
    if isinstance(tz, timedelta):
        return pd.DatetimeIndex(naive - tz).tz_localize("UTC")
    return pd.DatetimeIndex(naive).tz_localize(
        tz, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")


def read_raw(path) -> pd.DataFrame:
    """Parse a CSV into the standard columns with a NAIVE 'time' column.

    Handles: comma/semicolon/tab delimiters, header names in any case, the MT5
    native export (<DATE> <TIME> <TICKVOL> <VOL> <SPREAD>) and the simple
    'time,open,high,low,close,volume' layout.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=_sniff_delimiter(path), encoding="utf-8-sig")
    df.columns = [_ALIASES.get(_clean_header(c), _clean_header(c)) for c in df.columns]

    if "time" not in df.columns:
        if "date" in df.columns:
            df["time"] = df["date"].astype(str)   # DATE column already holds date+time
        else:
            raise ValueError(f"{path.name}: no time/datetime column. Found: {list(df.columns)}")
    elif "date" in df.columns:
        # both present: native MT5 export -> combine
        df["time"] = df["date"].astype(str) + " " + df["time"].astype(str)
    df = df.drop(columns=[c for c in ("date",) if c in df.columns])

    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}. Found: {list(df.columns)}")

    # MT5 writes dates as 2024.01.02 - normalize to ISO before parsing
    t = df["time"].astype(str).str.strip().str.replace(r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True)
    df["time"] = pd.to_datetime(t, errors="coerce")
    bad_time = int(df["time"].isna().sum())
    if bad_time:
        df = df.dropna(subset=["time"])

    keep = STANDARD_COLUMNS + (["spread"] if "spread" in df.columns else [])
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df[["time"] + keep].reset_index(drop=True)
    out.attrs["unparseable_time_rows"] = bad_time
    return out


def detect_utc_offset(naive_times: pd.Series, min_gaps: int = 3):
    """Estimate the broker-time offset from UTC (in hours) from the weekly close.

    The FX/gold week ends Friday 17:00 New York time (22:00 UTC in winter,
    21:00 UTC in summer), so the last bar before each weekend gap is stamped
    16:59 New York. We find the weekend gaps (> 40h without bars), compare each
    last-bar-before-the-gap with that expected moment, and take the most
    common whole-hour difference.

    The Friday close is used rather than the Sunday reopen because brokers
    disagree on when gold reopens (17:00 vs 18:00 New York).

    Returns (offset_hours, agreement): agreement is the fraction of weekends
    voting for that offset. Returns (None, 0.0) if there are fewer than
    `min_gaps` weekends (roughly 3+ weeks of M1 data).
    """
    t = pd.Series(pd.to_datetime(naive_times)).sort_values().reset_index(drop=True)
    gap = t.diff().shift(-1)                      # time from this bar to the next one
    last_bars = t[gap > pd.Timedelta(hours=40)]
    ny = ZoneInfo("America/New_York")
    votes = []
    for ts in last_bars:
        if ts.weekday() not in (4, 5):            # broker-time Friday, or Saturday if far east of UTC
            continue
        friday = (ts - pd.Timedelta(days=ts.weekday() - 4)).normalize()
        expected = (pd.Timestamp(friday.year, friday.month, friday.day, 16, 59, tzinfo=ny)
                    .tz_convert("UTC").tz_localize(None))
        votes.append(round((ts - expected).total_seconds() / 3600))
    if len(votes) < min_gaps:
        return None, 0.0
    s = pd.Series(votes)
    mode = int(s.mode().iloc[0])
    return mode, float((s == mode).mean())


def load_csv(path, source_tz="auto") -> pd.DataFrame:
    """Load one MT5 CSV export as a standard UTC-indexed DataFrame.

    source_tz:
      "auto"       - infer the broker's fixed UTC offset from weekend reopens.
                     Raises if it can't be inferred confidently: a wrong
                     timezone silently ruins every session-based setup.
      "UTC", "+2", "Europe/Athens", ... - explicit; use this once you know it.
    """
    raw = read_raw(path)
    if raw.empty:
        raise ValueError(f"{Path(path).name}: no rows")

    if source_tz == "auto":
        offset, agreement = detect_utc_offset(raw["time"])
        if offset is None:
            raise ValueError(
                "Cannot auto-detect timezone (needs ~3+ weekends of data). "
                "Pass source_tz explicitly, e.g. load_csv(path, source_tz='UTC').")
        if agreement < 0.7:
            raise ValueError(
                f"Timezone auto-detect is inconsistent (best guess UTC{offset:+d}, "
                f"only {agreement:.0%} agreement - the broker may follow DST). "
                "Pass source_tz explicitly.")
        tz = timedelta(hours=offset)
    else:
        tz = _parse_source_tz(source_tz)

    idx = _to_utc(raw["time"], tz)
    df = raw.drop(columns=["time"])
    df.index = idx
    df.index.name = "time"
    df = df[~df.index.isna()]
    df.attrs["source_tz"] = source_tz
    return df
