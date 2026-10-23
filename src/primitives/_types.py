"""Shared constants for the primitives layer."""
from __future__ import annotations

LONG = 1
SHORT = -1


def check_direction(direction: int) -> int:
    if direction not in (LONG, SHORT):
        raise ValueError("direction must be 1 (LONG) or -1 (SHORT)")
    return direction
