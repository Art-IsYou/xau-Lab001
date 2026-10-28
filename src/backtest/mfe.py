# src/backtest/mfe.py  ->  PYTHONPATH=src python src/backtest/mfe.py
import sys
import numpy as np, pandas as pd
from backtest.run import load_data
from setups.breakout import BreakoutConfig, scan_breakout

SPREAD = 0.15
TF = sys.argv[1] if len(sys.argv) > 1 else "1min"
df, htf = load_data("data/raw/XAUUSD_1m.csv", TF)
sig = scan_breakout(df, htf_df=htf, cfg=BreakoutConfig()).to_frame()
hi, lo = df.high.to_numpy(), df.low.to_numpy()

rows = []
for s in sig.itertuples():
    d, i = s.direction, s.trigger_pos
    entry = s.fill_price + SPREAD if d == 1 else s.fill_price
    risk = (entry - s.stop) * d
    mfe, stopped = 0.0, False
    for j in range(i, min(i + 1500, len(df))):
        hit = lo[j] <= s.stop if d == 1 else hi[j] + SPREAD >= s.stop
        if hit:                      # stop wins ties; its bar's high/low is not counted
            stopped = True
            break
        if j > i:
            fav = (hi[j] - entry) / risk if d == 1 else (entry - (lo[j] + SPREAD)) / risk
            mfe = max(mfe, fav)
    rows.append((d, mfe, stopped))
m = pd.DataFrame(rows, columns=["d", "mfe", "stopped"])

out = []
for k in (0.5, 1, 1.5, 2, 2.5, 3, 4):
    for name, g in (("ALL", m), ("LONG", m[m.d == 1]), ("SHORT", m[m.d == -1])):
        p = (g.mfe >= k).mean()
        out.append((k, name, len(g), round(p, 2), round(p * k - (1 - p), 3)))
print(pd.DataFrame(out, columns=["target_R", "set", "n", "p_reach", "expectancy_R"])
      .pivot(index="target_R", columns="set", values=["p_reach", "expectancy_R"]))
print("\nmedian MFE before stop (R):", m.groupby("d").mfe.median().round(2).to_dict())