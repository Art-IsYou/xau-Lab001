"""Event study with a matched random control.

Why a control: the earlier random-walk baseline (P(reach k before stop) = 1/(1+k))
ignores drift. Gold rose through most of the sample, so *any* long entry looks good
and any short looks bad. Here each event is compared with random entries that share
its direction, its UTC hour of day, and its risk in ATR terms, so drift, intraday
seasonality and volatility regime are all held fixed. What is left is the entry rule.

Trade model (same conventions as backtest/engine.py; bars are bid prices):
  * signal on bar t's close -> fill at bar t+1's open (long pays the ask = bid+spread);
  * stop wins any same-bar tie; a gap through the stop fills at the open;
  * a target needs the ask (short) / bid (long) to reach it;
  * unresolved after ``horizon`` bars -> marked to the last close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(opn, high, low, close, e_idx, d, risk, ks, spread, horizon):
    """R outcome of fixed-risk trades entered at the open of bar ``e_idx``.

    e_idx, d, risk are 1-D arrays (one trade each); returns an array (n, len(ks)).
    Requires e_idx + horizon <= len(close).
    """
    e_idx, d, risk = np.asarray(e_idx), np.asarray(d), np.asarray(risk, float)
    n, H = len(e_idx), horizon
    ix = e_idx[:, None] + np.arange(H)
    O, Hi, Lo, C = opn[ix], high[ix], low[ix], close[ix]
    is_long = (d == 1)[:, None]
    entry = np.where(d == 1, O[:, 0] + spread, O[:, 0])
    stop = entry - d * risk

    stop_hit = np.where(is_long, Lo <= stop[:, None], Hi + spread >= stop[:, None])
    stop_idx = np.where(stop_hit.any(1), stop_hit.argmax(1), H)

    j = np.minimum(stop_idx, H - 1)
    Oj = O[np.arange(n), j]
    exit_long = np.where(j > 0, np.minimum(Oj, stop), stop)  # gap through the stop fills at the open
    exit_short = np.where(j > 0, np.maximum(Oj + spread, stop), stop)
    r_stop = np.where(d == 1, (exit_long - entry) / risk, (entry - exit_short) / risk)
    r_time = np.where(d == 1, (C[:, -1] - entry) / risk, (entry - (C[:, -1] + spread)) / risk)

    out = np.empty((n, len(ks)))
    for c, k in enumerate(ks):
        tgt_hit = np.where(is_long, Hi >= (entry + k * risk)[:, None], Lo + spread <= (entry - k * risk)[:, None])
        tgt_idx = np.where(tgt_hit.any(1), tgt_hit.argmax(1), H)
        out[:, c] = np.where(tgt_idx < stop_idx, k, np.where(stop_idx < H, r_stop, r_time))
    return out


def run_study(df, atr14, events, *, spread=0.15, ks=(1.0, 1.5, 2.0, 3.0), horizon=96,
              min_risk_mult=4.0, n_controls=200, seed=0):
    """Simulate every event and ``n_controls`` matched random entries per event.

    Returns dict(times, direction, obs[n,K], ctrl[n,m,K], ks). Events whose risk is
    below ``min_risk_mult * spread`` (cost would dominate) or that are too close to
    the end of the data are dropped.
    """
    rng = np.random.default_rng(seed)
    opn, hi, lo, cl = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    a = atr14.to_numpy(float)
    N = len(df)
    min_risk = min_risk_mult * spread

    idx = events["idx"].to_numpy(int)
    d = events["direction"].to_numpy(int)
    stop_level = events["stop_level"].to_numpy(float)
    e = idx + 1
    ok = e + horizon <= N
    entry = np.where(d == 1, opn[np.minimum(e, N - 1)] + spread, opn[np.minimum(e, N - 1)])
    risk = (entry - stop_level) * d
    ok &= np.isfinite(a[idx]) & (risk >= min_risk)
    idx, d, e, risk = idx[ok], d[ok], e[ok], risk[ok]
    risk_atr = risk / a[idx]

    obs = simulate(opn, hi, lo, cl, e, d, risk, ks, spread, horizon)

    hours = df.index.hour.to_numpy()
    valid = np.isfinite(a) & (np.arange(N) + 1 + horizon <= N)
    pools = {h: np.flatnonzero((hours == h) & valid) for h in np.unique(hours)}
    ctrl = np.empty((len(idx), n_controls, len(ks)))
    for i in range(len(idx)):
        pool = pools[hours[idx[i]]]
        pick = rng.choice(pool, size=n_controls, replace=True)
        r = np.maximum(risk_atr[i] * a[pick], min_risk)
        ctrl[i] = simulate(opn, hi, lo, cl, pick + 1, np.full(n_controls, d[i]), r, ks, spread, horizon)
    return {"times": df.index[idx], "direction": d, "obs": obs, "ctrl": ctrl, "ks": tuple(ks)}


def summarize(study, mask=None, n_boot=4000, seed=1) -> pd.DataFrame:
    """Per target multiple: events' mean R vs the matched-control mean R, and the
    one-sided p-value that random entries (one control drawn per event, resampled
    ``n_boot`` times) would do at least as well."""
    obs, ctrl, ks = study["obs"], study["ctrl"], study["ks"]
    if mask is not None:
        obs, ctrl = obs[mask], ctrl[mask]
    n, m, K = ctrl.shape if len(obs) else (0, 0, len(ks))
    if n == 0:
        return pd.DataFrame({"k": ks, "n": 0})
    rng = np.random.default_rng(seed)
    rows_ix = np.arange(n)[None, :]
    null = np.concatenate([
        ctrl[rows_ix, rng.integers(0, m, (min(500, n_boot - s), n))].mean(axis=1)
        for s in range(0, n_boot, 500)
    ])
    o = obs.mean(0)
    p = (1 + (null >= o).sum(0)) / (len(null) + 1)
    return pd.DataFrame({
        "k": ks, "n": n,
        "win_rate": [(obs[:, c] == k).mean() for c, k in enumerate(ks)],
        "mean_R": o, "control_R": null.mean(0), "edge_R": o - null.mean(0), "p": p,
    })