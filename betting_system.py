"""FightNight betting system — apply the KISS rule and settle.

Rule (at the OPENING line; pre-registered spec: FLIP 2u, GAP 1u):
  FLIP: model picks the market underdog AND model prob >= 0.60  -> 2 units
  GAP : model agrees on favorite AND prices them >= 100 American pts stronger  -> 1 unit
Pre-filter: skip bouts whose opening implied-prob sum is outside [1.00, 1.12].

Requires: data/model_only_preds.csv, data/bfo_joined.csv, data/fights_v2.csv
Usage: python betting_system.py [year|all]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

FLIP_CONF = 0.60
GAP_PTS = 100
FLIP_UNITS = 2     # pre-registered spec: FLIP at 2u, GAP at 1u
GAP_UNITS = 1

am = lambda dec: np.where(dec >= 2, (dec - 1) * 100, -100 / (dec - 1))
amp = lambda q: np.where(q >= .5, -100 * q / (1 - q), 100 * (1 - q) / q)


def load():
    M = pd.read_csv("data/model_only_preds.csv")
    J = pd.read_csv("data/bfo_joined.csv")
    fv = pd.read_csv("data/fights_v2.csv")[["fight_id", "event_date"]]
    D = M.merge(J, on="fight_id").merge(fv, on="fight_id")
    D = D.dropna(subset=["y_a", "a_open", "b_open"])
    imp = 1 / D.a_open + 1 / D.b_open
    D = D[(imp >= 1.0) & (imp <= 1.12)].copy()      # odds sanity
    o1, o2, p = D.a_open.values, D.b_open.values, D.p_model_a.values
    mkt1 = (1 / o1) >= (1 / o2)
    mod1 = p >= .5
    D["mkt1"], D["mod1"] = mkt1, mod1
    D["flip"] = mkt1 != mod1
    D["conf"] = np.where(mod1, p, 1 - p)
    D["gap_pts"] = np.where(mkt1, am(o1), am(o2)) - np.where(mkt1, amp(p), amp(1 - p))
    return D.sort_values("event_date")


def settle(units, sub):
    dec = np.where(sub.mod1.values, sub.a_open.values, sub.b_open.values)
    won = (sub.y_a.values == 1) == sub.mod1.values
    return np.where(won, units * (dec - 1), -float(units))


def run(D):
    fl = D[D.flip & (D.conf >= FLIP_CONF)].copy()
    gp = D[(~D.flip) & (D.gap_pts >= GAP_PTS)].copy()
    fl["u"], fl["rule"], fl["pnl"] = FLIP_UNITS, "FLIP", settle(FLIP_UNITS, fl)
    gp["u"], gp["rule"], gp["pnl"] = GAP_UNITS, "GAP", settle(GAP_UNITS, gp)
    return pd.concat([fl, gp]).sort_values("event_date")


def report(B, label):
    staked = B.u.sum()
    pnl = B.pnl.sum()
    w, l = int((B.pnl > 0).sum()), int((B.pnl < 0).sum())
    r = B.pnl / B.u
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan")
    print(f"\n{label}: {len(B)} bets  {w}-{l}  staked {staked:.0f}u  "
          f"P&L {pnl:+.1f}u (${pnl*100:+,.0f})  ROI {pnl/staked:+.1%}  t={t:.2f}")
    for rl, g in B.groupby("rule"):
        print(f"    {rl}: {len(g)} bets  {int((g.pnl>0).sum())}-{int((g.pnl<0).sum())}  "
              f"ROI {g.pnl.sum()/g.u.sum():+.1%}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    D = load()
    if arg != "all":
        D = D[D.year == int(arg)]
    B = run(D)
    report(B, f"BETTING SYSTEM ({arg})")
