"""Risk:reward arithmetic. ``expansion`` models adverse slippage on entry
(the 'CoC entry results in slight market expansion' clause): the entry is
assumed to fill that many price units worse than planned."""
from __future__ import annotations

from ._types import check_direction


def risk_reward(entry: float, stop: float, target: float, direction: int, expansion: float = 0.0) -> float:
    check_direction(direction)
    e = entry + direction * expansion
    risk = (e - stop) * direction
    reward = (target - e) * direction
    if risk <= 0:
        return float("nan")
    return reward / risk


def min_target_for_rr(entry: float, stop: float, direction: int, min_rr: float, expansion: float = 0.0) -> float:
    """Closest target price that still delivers ``min_rr``."""
    check_direction(direction)
    e = entry + direction * expansion
    risk = (e - stop) * direction
    return e + direction * min_rr * risk


def passes_rr(
    entry: float, stop: float, target: float, direction: int, min_rr: float = 2.5, expansion: float = 0.0
) -> bool:
    return bool(risk_reward(entry, stop, target, direction, expansion) >= min_rr)
