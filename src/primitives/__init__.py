"""Primitives: small, pure, individually testable building blocks.

Conventions
-----------
* Input is an OHLCV DataFrame with a sorted, unique DatetimeIndex and columns
  open/high/low/close/volume (the data layer's output).
* ``*_pos`` values are integer bar positions (``df.iloc`` positions).
* No lookahead: a value at bar ``t`` only depends on bars ``<= t``. Swings
  carry ``confirmed_pos``; HTF data is shifted by one HTF period.
* Direction is ``LONG = 1`` / ``SHORT = -1``.
* Thresholds are explicit keyword arguments; nothing is hidden in globals, so
  the config layer can feed them in later.
"""
from ._types import LONG, SHORT
from .candles import (
    atr,
    body_beyond_fraction,
    breakout_candle_mask,
    candle_parts,
    range_in_atr,
    true_range,
)
from .entry import CocTrigger, coc_levels, coc_trigger
from .levels import Level, cluster_levels, level_is_unbroken, next_level_beyond
from .liquidity import find_sweeps, liquidity_taken_in_window
from .rr import min_target_for_rr, passes_rr, risk_reward
from .sessions import LONDON, MAJOR_SESSIONS, NEW_YORK, SessionDef, session_label, session_mask
from .structure import align_htf_to_base, market_structure, structure_aligned
from .swings import find_swings, last_swing_series, swings_confirmed_by
from .volume import relative_volume, volume_gate

__all__ = [n for n in dir() if not n.startswith("_")]
