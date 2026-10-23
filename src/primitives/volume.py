"""Participation measures. MT5 ``volume`` is tick volume, which is fine as a
relative-participation proxy."""
from __future__ import annotations

import pandas as pd


def relative_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume divided by the mean volume of the previous ``window`` bars
    (current bar excluded). NaN until ``window`` prior bars exist."""
    base = df["volume"].shift(1).rolling(window, min_periods=window).mean()
    return df["volume"] / base.where(base > 0)


def volume_gate(df: pd.DataFrame, window: int = 20, min_mult: float = 1.5) -> pd.Series:
    """True where relative volume >= ``min_mult``."""
    return (relative_volume(df, window) >= min_mult).fillna(False).astype(bool)
