"""Scan -> backtest -> report.

    python -m backtest.run --csv data/raw/XAUUSD_M1.csv --spread 0.15 --risk-pct 1 \
        --oos-from 2025-06-01 --out results/breakout_v1

Run from the xau-setups/ root. Outputs: trades.csv, skipped.csv, funnel.csv, summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest import BacktestConfig, breakdowns, run_backtest, summarize


def load_data(csv_path: str, tf: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ADAPTER -- the only place that touches your data layer.

    Return (entry_df, htf_m15_df), both UTC-indexed OHLCV. The two imports below are
    placeholders: rename them to the real functions in data/loader.py and
    data/resample.py.
    """
    from data.loader import load_csv  # noqa: <- rename to your loader function
    from data.resample import resample  # noqa: <- rename to your resample function

    m1 = load_csv(csv_path)
    entry = m1 if tf == "1min" else resample(m1, tf)
    return entry, resample(m1, "M15")


def _fmt(summary: dict) -> str:
    if summary.get("trades", 0) == 0:
        return "  no trades"
    lo, hi = summary["expectancy_r_ci95"]
    return "\n".join([
        f"  trades            {summary['trades']}",
        f"  win rate          {summary['win_rate']:.1%}",
        f"  expectancy        {summary['expectancy_r']:+.3f} R   (95% CI {lo:+.3f} .. {hi:+.3f})",
        f"  total             {summary['total_r']:+.1f} R",
        f"  avg win / loss    {summary['avg_win_r']:+.2f} R / {summary['avg_loss_r']:+.2f} R",
        f"  profit factor     {summary['profit_factor']:.2f}",
        f"  max drawdown      {summary['max_drawdown_r']:.1f} R  /  {summary['max_drawdown_pct']:.1f}%",
        f"  max losing streak {summary['max_losing_streak']}",
        f"  equity            {summary['final_equity']:,.0f}  ({summary['return_pct']:+.1f}%)",
        f"  exits             {summary['exit_reasons']}",
        f"  same-bar stop/target conflicts (resolved as stop): {summary['same_bar_conflicts']}",
    ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="MT5 M1 export")
    p.add_argument("--tf", default="1min", help="entry timeframe the scan runs on (default 1min)")
    p.add_argument("--spread", type=float, default=0.15, help="price units; 15 points = 0.15")
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--commission", type=float, default=0.0, help="round-turn per lot")
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--risk-pct", type=float, default=1.0)
    p.add_argument("--no-compound", action="store_true")
    p.add_argument("--max-hold", type=int, default=None, help="time stop in bars")
    p.add_argument("--min-net-rr", type=float, default=None, help="skip trades whose R:R after spread is lower")
    p.add_argument("--oos-from", default=None, help="UTC date; trades from here are reported separately")
    p.add_argument("--out", default="results/latest")
    a = p.parse_args()

    from setups.breakout import BreakoutConfig, scan_breakout  # imported late: needs repo root on sys.path

    df, htf = load_data(a.csv, a.tf)
    scan = scan_breakout(df, htf_df=htf, cfg=BreakoutConfig())
    cfg = BacktestConfig(
        spread=a.spread, slippage=a.slippage, commission_per_lot=a.commission,
        initial_equity=a.equity, risk_pct=a.risk_pct, compounding=not a.no_compound,
        max_hold_bars=a.max_hold, min_net_rr=a.min_net_rr,
    )
    res = run_backtest(df, scan.to_frame(), cfg)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / "trades.csv", index=False)
    res.skipped.to_csv(out / "skipped.csv", index=False)
    scan.funnel_frame().to_csv(out / "funnel.csv")

    print(f"data {df.index[0]} -> {df.index[-1]}  ({len(df):,} bars)   spread {a.spread}  risk {a.risk_pct}%")
    print("\nfunnel\n" + scan.funnel_frame().to_string())
    print(f"\nsignals {len(scan.signals)}  traded {len(res.trades)}  skipped {len(res.skipped)} "
          f"{res.skipped['reason'].value_counts().to_dict() if len(res.skipped) else ''}")

    summaries = {"all": summarize(res.trades, a.equity)}
    print("\nALL\n" + _fmt(summaries["all"]))
    if a.oos_from and not res.trades.empty:
        cut = pd.Timestamp(a.oos_from, tz="UTC")
        ins, oos = res.trades[res.trades["entry_time"] < cut], res.trades[res.trades["entry_time"] >= cut]
        summaries["in_sample"], summaries["out_of_sample"] = summarize(ins, a.equity), summarize(oos, a.equity)
        print(f"\nIN-SAMPLE (< {a.oos_from})\n" + _fmt(summaries["in_sample"]))
        print(f"\nOUT-OF-SAMPLE (>= {a.oos_from})\n" + _fmt(summaries["out_of_sample"]))

    for name, tbl in breakdowns(res.trades).items():
        print(f"\nby {name}\n" + tbl.round(3).to_string())
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()