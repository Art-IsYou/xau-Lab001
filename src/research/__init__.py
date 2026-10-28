"""Event-study framework: does a candidate entry rule beat matched random entries?"""
from .candidates import orb, pdhl, squeeze
from .features import atr, h1_trend, trading_day
from .study import run_study, simulate, summarize

__all__ = ["orb", "pdhl", "squeeze", "atr", "h1_trend", "trading_day", "run_study", "simulate", "summarize"]