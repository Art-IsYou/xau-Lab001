"""Setups: playbook checklists turned into detectors, built only from primitives.

Only break-Out is implemented for now; break-n-Retest and Fake-Out follow once
break-Out has a final, backtested algorithm.
"""
from .breakout import STAGES, BreakoutConfig, BreakoutSignal, ScanResult, scan_breakout

__all__ = ["STAGES", "BreakoutConfig", "BreakoutSignal", "ScanResult", "scan_breakout"]