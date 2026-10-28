"""Run every candidate through the matched-control event study.

    PYTHONPATH=src python -m research.run --csv data/raw/XAUUSD_1m.csv --oos-from 2024-12-01

The candidates and their parameters are fixed in advance (see candidates.py). Do not
re-run with tweaked parameters until something looks good: with 8 variants x 4 targets
a handful of p < 0.05 results is expected by chance alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import orb, pdhl, squeeze
from .features import atr, h1_trend
from .study import run_study, summarize

VARIANTS = {
    "orb_london": lambda df, a: orb(df, a, start="08:00", tz="Europe/London"),
    "orb_newyork": lambda df, a: orb(df, a, start="09:30", tz="America/New_York"),
    "squeeze": lambda df, a: squeeze(df, a),
    "pdhl": lambda df, a: pdhl(df, a),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--spread", type=float, default=0.15)
    p.add_argument("--oos-from", required=True, help="UTC date; events from here are the holdout")
    p.add_argument("--horizon", type=int, default=96, help="M5 bars to hold (96 = 8h)")
    p.add_argument("--out", default="results/research")
    a = p.parse_args()

    from backtest.run import load_data  # your data-layer adapter

    df, _ = load_data(a.csv, "5min")
    atr14 = atr(df)
    trend = h1_trend(df)
    cut = pd.Timestamp(a.oos_from, tz="UTC")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{df.index[0]} -> {df.index[-1]}  M5 bars {len(df):,}  spread {a.spread}  holdout from {a.oos_from}")

    tables = []
    for name, fn in VARIANTS.items():
        ev = fn(df, atr14)
        for flavour in ("all", "with_trend"):
            e = ev if flavour == "all" else ev[ev["direction"].to_numpy() == trend.to_numpy()[ev["idx"].to_numpy()]]
            label = f"{name}/{flavour}"
            if len(e) < 10:
                print(f"\n{label}: only {len(e)} events, skipped")
                continue
            st = run_study(df, atr14, e, spread=a.spread, horizon=a.horizon)
            oos = np.asarray(st["times"] >= cut)
            for part, mask in (("in_sample", ~oos), ("holdout", oos), ("all", None)):
                t = summarize(st, mask)
                t.insert(0, "part", part)
                t.insert(0, "variant", label)
                tables.append(t)
            print(f"\n=== {label}   events {len(st['obs'])} (holdout {int(oos.sum())}) ===")
            show = pd.concat([x for x in tables[-3:]])
            print(show.drop(columns="variant").round(3).to_string(index=False))

    if tables:
        pd.concat(tables).to_csv(out / "event_study.csv", index=False)
    n_tests = len(VARIANTS) * 2 * 4
    print(f"\n{n_tests} variant/target tests per part: expect ~{0.05 * n_tests:.0f} to show p<0.05 by chance.")
    print("Only trust a result with edge_R > 0 and p < 0.01 in the pooled sample AND edge_R > 0 in the holdout.")
    print(f"written to {out}/event_study.csv")


if __name__ == "__main__":
    main()