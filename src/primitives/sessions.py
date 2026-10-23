"""Trading-session masks defined in local wall-clock time (DST-correct).

Naive indexes are assumed to be in ``source_tz`` (default UTC). Make sure the
data layer hands over UTC, or pass the right ``source_tz``.
Session windows below are defaults, not facts: tune them in config.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SessionDef:
    name: str
    tz: str
    start: str  # local "HH:MM", inclusive
    end: str  # local "HH:MM", exclusive


LONDON = SessionDef("london", "Europe/London", "08:00", "16:30")
NEW_YORK = SessionDef("new_york", "America/New_York", "08:00", "17:00")
MAJOR_SESSIONS = (LONDON, NEW_YORK)


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _aware(index, source_tz: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(source_tz) if idx.tz is None else idx


def _inside(idx: pd.DatetimeIndex, s: SessionDef) -> np.ndarray:
    loc = idx.tz_convert(s.tz)
    m = np.asarray(loc.hour) * 60 + np.asarray(loc.minute)
    a, b = _minutes(s.start), _minutes(s.end)
    inside = (m >= a) & (m < b) if a < b else (m >= a) | (m < b)
    return inside & (np.asarray(loc.day_of_week) < 5)


def session_mask(index, sessions=MAJOR_SESSIONS, source_tz: str = "UTC") -> pd.Series:
    """True where the bar falls inside ANY of ``sessions`` (weekdays only)."""
    idx = _aware(index, source_tz)
    out = np.zeros(len(idx), dtype=bool)
    for s in sessions:
        out |= _inside(idx, s)
    return pd.Series(out, index=index)


def session_label(index, sessions=MAJOR_SESSIONS, source_tz: str = "UTC") -> pd.Series:
    """'london', 'new_york', 'london+new_york' (overlap) or '' per bar."""
    idx = _aware(index, source_tz)
    labels = np.full(len(idx), "", dtype=object)
    for s in sessions:
        hit = _inside(idx, s)
        labels = np.where(hit, np.where(labels == "", s.name, labels + "+" + s.name), labels)
    return pd.Series(labels, index=index)
