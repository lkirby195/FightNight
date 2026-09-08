"""FightNight betting system — apply the KISS rule and settle.

Rule (flat 1 unit at the OPENING line):
  FLIP: model picks the market underdog AND model prob >= 0.65
  GAP : model agrees on favorite AND prices them >= 100 American pts stronger
Pre-filter: skip bouts whose opening implied-prob sum is outside [1.00, 1.12].

Requires: data/model_only_preds.csv, data/bfo_joined.csv, data/fights_v2.csv
Usage: python betting_system.py [year|all] [--write-ledger]

--write-ledger: after printing the summary, write every triggered bet for the
requested year(s) to data/system_ledger.csv -- the ledger of record. That file
is GENERATED, never hand-edited. clean_close is the last pre-fight paired tick
(card_settle.paired_summary); bouts whose raw BFO close is already a clean book
use it as-is, the rest are re-read from the BFO line history (cache_bfo/).
"""
from __future__ import annotations

import argparse
import re
import unicodedata

import numpy as np
import pandas as pd

FLIP_CONF = 0.65
GAP_PTS = 100
FLIP_UNITS = 1     # flat; set 2 for the higher-variance variant
GAP_UNITS = 1

LEDGER = "data/system_ledger.csv"
LEDGER_COLS = ["event_date", "event", "fight_id", "fighter", "rule", "units",
               "open_line", "clean_close", "clv_pts", "result", "pnl"]

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


def _lastn(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    p = re.sub(r"[^a-z ]", "", s.lower()).split()
    return p[-1] if p else ""


def _bfo_mu(bl, r):
    """BFO matchup id for one bet: same date (+-3d), same opening prices, same
    surnames. -> (mu, a_is_f1). Fails loudly rather than guess."""
    c = bl[(bl.d - pd.Timestamp(r.event_date)).abs().dt.days <= 3]
    c = c[((c.f1_open == r.a_open) & (c.f2_open == r.b_open)) |
          ((c.f1_open == r.b_open) & (c.f2_open == r.a_open))]
    names = {_lastn(r.fighter_a), _lastn(r.fighter_b)}
    if len(c):
        c = c[c.apply(lambda x: {_lastn(x.fighter1), _lastn(x.fighter2)} == names,
                      axis=1).astype(bool)]
    if len(c) != 1:
        raise SystemExit(f"--write-ledger: {len(c)} BFO matchups for {r.fight_id} "
                         f"{r.fighter_a} vs {r.fighter_b} {r.event_date}")
    x = c.iloc[0]
    if _lastn(r.fighter_a) != _lastn(r.fighter_b):
        return int(x.mu), _lastn(x.fighter1) == _lastn(r.fighter_a)
    return int(x.mu), bool(x.f1_open == r.a_open)


def _clean_close(bl, r):
    """(a_close, b_close) of the last pre-fight paired tick (card_settle rule)."""
    from card_settle import OR_HI, OR_LO, paired_summary
    if OR_LO <= 1 / r.a_close + 1 / r.b_close <= OR_HI:
        return r.a_close, r.b_close            # raw close is already a clean book
    from scrape_bfo import ggd
    mu, a_is_f1 = _bfo_mu(bl, r)
    s = paired_summary(ggd(mu, 1), ggd(mu, 2))
    if s is None:
        raise SystemExit(f"--write-ledger: no BFO ticks for mu {mu} ({r.fight_id})")
    _, f1c, _, f2c = s[:4]
    return (f1c, f2c) if a_is_f1 else (f2c, f1c)


def write_ledger(B, path=LEDGER):
    """Write every triggered bet in B to `path`, sorted by date then bout order.
    open_line / clean_close are American odds of the bet side; clv_pts is the
    bet side's vig-free close minus open, in prob points; pnl is in units."""
    fv = pd.read_csv("data/fights_v2.csv")[["fight_id", "event_name", "bout_order",
                                            "fighter_a", "fighter_b"]]
    bl = pd.read_csv("data/bfo_lines.csv")
    bl["d"] = pd.to_datetime(bl.event_date)
    rows = []
    for _, r in B.merge(fv, on="fight_id").iterrows():
        ac, bc = _clean_close(bl, r)
        pick_a = bool(r.mod1)
        o, c = (r.a_open, ac) if pick_a else (r.b_open, bc)
        qo = (1 / o) / (1 / r.a_open + 1 / r.b_open)
        qc = (1 / c) / (1 / ac + 1 / bc)
        rows.append(dict(event_date=r.event_date, event=r.event_name,
                         fight_id=r.fight_id,
                         fighter=r.fighter_a if pick_a else r.fighter_b,
                         rule=r.rule, units=int(r.u),
                         open_line=int(round(float(am(o)))),
                         clean_close=int(round(float(am(c)))),
                         clv_pts=round((qc - qo) * 100, 2),
                         result="WIN" if r.pnl > 0 else "LOSS",
                         pnl=round(float(r.pnl), 4), _bo=r.bout_order))
    L = (pd.DataFrame(rows, columns=LEDGER_COLS + ["_bo"])
         .sort_values(["event_date", "_bo"]).drop(columns="_bo"))
    L.to_csv(path, index=False)
    print(f"wrote {path}: {len(L)} rows  staked {L.units.sum()}u  P&L {L.pnl.sum():+.1f}u")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("year", nargs="?", default="all", help="calendar year or 'all'")
    ap.add_argument("--write-ledger", action="store_true",
                    help=f"write the triggered bets to {LEDGER} (ledger of record)")
    a = ap.parse_args()
    D = load()
    if a.year != "all":
        D = D[D.year == int(a.year)]
    B = run(D)
    report(B, f"BETTING SYSTEM ({a.year})")
    if a.write_ledger:
        write_ledger(B)
