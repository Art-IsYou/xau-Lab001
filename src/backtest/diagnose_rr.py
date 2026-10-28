# diagnose_rr.py  ->  from repo root: PYTHONPATH=src python diagnose_rr.py
from dataclasses import replace
import pandas as pd
from backtest.run import load_data
from setups.breakout import BreakoutConfig, scan_breakout

df, htf = load_data("data/raw/XAUUSD_1m.csv", "1min")

for touches in (1, 2, 3):
    cfg = replace(BreakoutConfig(), min_rr=0.0, target_min_touches=touches)
    s = scan_breakout(df, htf_df=htf, cfg=cfg).to_frame()
    print(f"\n=== target_min_touches={touches}  signals={len(s)} ===")
    if s.empty:
        continue
    s["risk"] = (s.fill_price - s.stop) * s.direction
    s["reward"] = (s.target - s.fill_price) * s.direction
    for d, name in ((1, "LONG"), (-1, "SHORT")):
        g = s[s.direction == d]
        if g.empty:
            print(name, "none"); continue
        print(f"{name} n={len(g)}  median risk ${g.risk.median():.2f}  "
              f"median reward ${g.reward.median():.2f}  "
              f"rr>=1: {(g.rr_fill>=1).mean():.0%}  rr>=2.5: {(g.rr_fill>=2.5).mean():.0%}")