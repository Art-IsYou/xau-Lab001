"""Candidate breakout entry rules, run on closed M5 bars.

Each returns a DataFrame with columns ``idx`` (position of the signal bar, whose CLOSE
triggered it), ``direction`` (+1/-1) and ``stop_level`` (absolute price of the
structural stop). The study enters at the *next* bar's open, so nothing here may look
past bar ``idx``. Defaults are priors, not fitted values; do not sweep them on the
sample you evaluate on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import trading_day

LONG, SHORT = 1, -1
COLS = ["idx", "direction", "stop_level"]


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLS).astype({"idx": int, "direction": int, "stop_level": float})


def orb(df, atr14, *, start="08:00", tz="Europe/London", range_minutes=30, window_hours=3,
        min_width_atr=2.0, max_width_atr=12.0, stop_mode="opposite", bar_minutes=5) -> pd.DataFrame:
    """Opening-range breakout. The range is [start, start+range_minutes) in *local* time
    (``tz``, so DST is handled); the first M5 close beyond it within ``window_hours``
    after the range is the signal. One event per day.

    London cash open: start="08:00", tz="Europe/London".
    US equity open:   start="09:30", tz="America/New_York".
    """
    loc = df.index.tz_convert(tz)
    mins = (loc.hour * 60 + loc.minute).to_numpy()
    s0 = int(start[:2]) * 60 + int(start[3:5])
    s1, s2 = s0 + range_minutes, s0 + range_minutes + window_hours * 60
    need = range_minutes // bar_minutes
    hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    a = atr14.to_numpy(float)
    rows = []
    for _, g in pd.Series(np.arange(len(df))).groupby(loc.normalize()):
        pos = g.to_numpy()
        m = mins[pos]
        rpos = pos[(m >= s0) & (m < s1)]
        if len(rpos) < need:
            continue
        rhi, rlo = hi[rpos].max(), lo[rpos].min()
        a_end = a[rpos[-1]]
        if not np.isfinite(a_end) or a_end <= 0:
            continue
        if not (min_width_atr <= (rhi - rlo) / a_end <= max_width_atr):
            continue
        for p in pos[(m >= s1) & (m < s2)]:
            if cl[p] > rhi:
                rows.append((p, LONG, rlo if stop_mode == "opposite" else (rhi + rlo) / 2))
                break
            if cl[p] < rlo:
                rows.append((p, SHORT, rhi if stop_mode == "opposite" else (rhi + rlo) / 2))
                break
    return _frame(rows)


def squeeze(df, atr14, *, n=24, pct_window=2000, pct=0.2, cooldown=24) -> pd.DataFrame:
    """Volatility-compression breakout. The box is the last ``n`` bars' high/low; it is
    'compressed' when box height / ATR ranks in the lowest ``pct`` of the trailing
    ``pct_window`` boxes. Signal: first close beyond a compressed box. Stop: the box's
    opposite edge."""
    hh = df["high"].rolling(n).max().shift(1)  # box = bars t-n .. t-1
    ll = df["low"].rolling(n).min().shift(1)
    box = (hh - ll) / atr14.shift(1)
    comp = box.rolling(pct_window, min_periods=pct_window // 2).rank(pct=True) <= pct
    cl = df["close"]
    long_, short_ = (comp & (cl > hh)).to_numpy(), (comp & (cl < ll)).to_numpy()
    hh_, ll_ = hh.to_numpy(float), ll.to_numpy(float)
    rows, last = [], -10**9
    for p in np.flatnonzero(long_ | short_):
        if p - last < cooldown:
            continue
        rows.append((p, LONG, ll_[p]) if long_[p] else (p, SHORT, hh_[p]))
        last = p
    return _frame(rows)


def pdhl(df, atr14, *, stop_atr=2.0, start_hour=7, end_hour=20) -> pd.DataFrame:
    """Previous-trading-day high/low break: first M5 close that *crosses* PDH (long) or
    PDL (short) inside [start_hour, end_hour) UTC, once per direction per day.
    Stop: ``stop_atr`` x ATR from the signal close (volatility stop, not structural)."""
    day = trading_day(df.index)
    pdh = day.map(df["high"].groupby(day).max().shift(1)).to_numpy(float)
    pdl = day.map(df["low"].groupby(day).min().shift(1)).to_numpy(float)
    cl, a = df["close"].to_numpy(float), atr14.to_numpy(float)
    hours, codes = df.index.hour.to_numpy(), pd.factorize(day)[0]
    done: set = set()
    rows = []
    for p in range(1, len(df)):
        if not (start_hour <= hours[p] < end_hour) or not np.isfinite(a[p]):
            continue
        if cl[p] > pdh[p] >= cl[p - 1] and (codes[p], LONG) not in done:
            done.add((codes[p], LONG))
            rows.append((p, LONG, cl[p] - stop_atr * a[p]))
        elif cl[p] < pdl[p] <= cl[p - 1] and (codes[p], SHORT) not in done:
            done.add((codes[p], SHORT))
            rows.append((p, SHORT, cl[p] + stop_atr * a[p]))
    return _frame(rows)