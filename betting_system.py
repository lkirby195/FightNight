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
live=1 marks bets whose card report was run before the event (see LIVE_CARDS);
live=0 rows are backfilled after the fact and are never quoted as results.
placeable_line is the bet side's price at the EARLIEST data/placeable_lines.csv
capture for that bout (card_report.py writes one when it reports a future
event): the price on the board when the pick went on record. live=1 rows with
a placeable line settle pnl there; every other row settles at open_line.
pnl_at_open keeps the open-line settlement for all rows; clv_pts is always
open -> clean_close.
Two summaries are printed: LIVE RECORD (live=1 only, P&L at placeable and at
open) and FULL SIM (all bets, at open).
Placement policy v1 (frozen 2026-09-09) is a layer on top of the signal rules:
every signal is still logged at flat 1u (units, pnl, pnl_at_open); stake_units()
decides from the pre-fight RDs (data/prefight_rd.csv, written by prefight_rd.py)
and the bet side's price whether it is placed (placed, stake_u) and
pnl_placed = stake_u * pnl per unit. A PLACED (policy v1) block follows each
summary.
"""
from __future__ import annotations

import argparse
import os
import re
import unicodedata

import numpy as np
import pandas as pd

FLIP_CONF = 0.65
GAP_PTS = 100
FLIP_UNITS = 1     # flat; set 2 for the higher-variance variant
GAP_UNITS = 1

# Placement policy v1 — frozen 2026-09-09
# A layer on top of the signal rules above, which stay frozen: every signal is
# still logged at flat 1u; the policy decides which are placed and at what
# stake, from the pre-fight rating deviations (RD, data/prefight_rd.csv) of the
# backed fighter and the opponent and from the backed side's American price
# (placeable_line where captured, else open_line).
SKIP_LINE_MAX = 250        # bet side +250 or longer -> not placed
UNKNOWN_RD = 160           # both fighters RD > this -> not placed
RAMP_LO, RAMP_HI = 130, 200


def stake_units(rd_self, rd_opp, line):
    """Stake in units for one signal (0.0 = not placed). Full 1u up to a
    backed-side RD of RAMP_LO, falling linearly to 0.5u at RAMP_HI and floored
    there."""
    if line > SKIP_LINE_MAX:
        return 0.0
    if rd_self > UNKNOWN_RD and rd_opp > UNKNOWN_RD:
        return 0.0
    return float(np.clip(1 - (rd_self - RAMP_LO) / (RAMP_HI - RAMP_LO) * 0.5, 0.5, 1.0))

# Live record: every card from LIVE_FROM through LIVE_THROUGH was reported
# pre-event, plus the dates in LIVE_CARDS. ADD EACH NEW CARD'S DATE TO LIVE_CARDS
# WHEN ITS REPORT IS RUN PRE-EVENT; anything else is backfilled (live=0).
LIVE_FROM, LIVE_THROUGH = "2026-01-01", "2026-07-25"
LIVE_CARDS = {"2026-08-22",   # UFC Sacramento (no bets)
              "2026-08-29",   # UFC Shanghai
              "2026-09-05",   # UFC Paris
              "2026-09-12"}   # Noche UFC Glendale (no bets)

LEDGER = "data/system_ledger.csv"
PLACEABLE = "data/placeable_lines.csv"      # written by card_report.py
PREFIGHT_RD = "data/prefight_rd.csv"        # written by prefight_rd.py
LEDGER_COLS = ["event_date", "event", "fight_id", "fighter", "rule", "live", "units",
               "rd_self", "rd_opp", "placed", "stake_u", "pnl_placed",
               "open_line", "placeable_line", "clean_close", "clv_pts", "result",
               "method", "round", "pnl", "pnl_at_open"]

am = lambda dec: np.where(dec >= 2, (dec - 1) * 100, -100 / (dec - 1))
amp = lambda q: np.where(q >= .5, -100 * q / (1 - q), 100 * (1 - q) / q)


def load():
    M = pd.read_csv("data/model_only_preds.csv")
    J = pd.read_csv("data/bfo_joined.csv")
    fv = pd.read_csv("data/fights_v2.csv")[["fight_id", "event_date", "event_name",
                                            "bout_order", "fighter_a", "fighter_b",
                                            "method", "round"]].rename(columns={"round": "rnd"})
    rd = pd.read_csv(PREFIGHT_RD)               # fight_id, rd_a, rd_b
    D = (M.merge(J, on="fight_id").merge(fv, on="fight_id")
          .merge(rd, on="fight_id", how="left"))
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


def is_live(event_date: str) -> bool:
    return LIVE_FROM <= event_date <= LIVE_THROUGH or event_date in LIVE_CARDS


def run(D):
    fl = D[D.flip & (D.conf >= FLIP_CONF)].copy()
    gp = D[(~D.flip) & (D.gap_pts >= GAP_PTS)].copy()
    fl["u"], fl["rule"], fl["pnl"] = FLIP_UNITS, "FLIP", settle(FLIP_UNITS, fl)
    gp["u"], gp["rule"], gp["pnl"] = GAP_UNITS, "GAP", settle(GAP_UNITS, gp)
    B = pd.concat([fl, gp]).sort_values("event_date")
    B["live"] = B.event_date.map(is_live).astype(int)
    B["pnl_at_open"] = B.pnl
    B["placeable"] = placeable_prices(B)        # decimal; NaN when none captured
    has = B.placeable.notna()
    B.loc[has, "pnl"] = np.where(B.pnl_at_open[has] > 0,
                                 B.u[has] * (B.placeable[has] - 1),
                                 -B.u[has].astype(float))
    # placement policy v1: a pure function of the columns above, every row
    if B.rd_a.isna().any():
        raise SystemExit(f"{PREFIGHT_RD} has no row for {int(B.rd_a.isna().sum())} "
                         f"bets: run prefight_rd.py")
    B["rd_self"] = np.where(B.mod1, B.rd_a, B.rd_b)
    B["rd_opp"] = np.where(B.mod1, B.rd_b, B.rd_a)
    bet_dec = np.where(has, B.placeable, np.where(B.mod1, B.a_open, B.b_open))
    B["bet_line"] = np.rint(am(bet_dec)).astype(int)    # bet side, American
    B["stake_u"] = [round(stake_units(a, b, c), 2)
                    for a, b, c in zip(B.rd_self, B.rd_opp, B.bet_line)]
    B["placed"] = (B.stake_u > 0).astype(int)
    B["pnl_placed"] = B.stake_u * B.pnl / B.u
    return B


def _summ(B, col, tag=""):
    pnl, r = B[col].sum(), B[col] / B.u
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan")
    return (f"P&L{tag} {pnl:+.1f}u (${pnl*100:+,.0f})  ROI {pnl/B.u.sum():+.1%}  "
            f"t={t:.2f}")


def report(B, label, placeable=False):
    """One summary block. placeable=True (LIVE RECORD) settles at the placeable
    line where one was captured and also prints the open-line figure; otherwise
    every bet is settled at open (pure sim)."""
    print()
    if len(B) == 0:
        print(f"{label}: 0 bets")
        _report_placed(B, "pnl")
        return
    col = "pnl" if placeable else "pnl_at_open"
    w, l = int((B[col] > 0).sum()), int((B[col] < 0).sum())
    head = f"{label}: {len(B)} bets  {w}-{l}  staked {B.u.sum():.0f}u  "
    if placeable:
        print(head + _summ(B, "pnl", " (placeable)"))
        print("    " + _summ(B, "pnl_at_open", " (open)"))
    else:
        print(head + _summ(B, col))
    for rl, g in B.groupby("rule"):
        print(f"    {rl}: {len(g)} bets  {int((g[col]>0).sum())}-{int((g[col]<0).sum())}  "
              f"ROI {g[col].sum()/g.u.sum():+.1%}")
    _report_placed(B, col)


def _report_placed(B, col):
    """PLACED (policy v1) block under a summary: the signals in B the policy
    places, at their policy stake, settled the same way as that summary
    (col = pnl for LIVE RECORD, pnl_at_open for FULL SIM)."""
    P = B[B.placed == 1] if len(B) else B
    head = f"PLACED (policy v1): {len(P)} placed of {len(B)} signals"
    if len(P) == 0:
        print(head)
        return
    pp = P.stake_u * P[col] / P.u               # P&L at the policy stake
    r = P[col] / P.u                            # per-bet return on stake
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan")
    print(f"{head}  {int((pp > 0).sum())}-{int((pp < 0).sum())}  "
          f"staked {P.stake_u.sum():.1f}u  P&L {pp.sum():+.1f}u (${pp.sum()*100:+,.0f})  "
          f"ROI {pp.sum()/P.stake_u.sum():+.1%}  t={t:.2f}")
    for rl, g in P.groupby("rule"):
        gp = g.stake_u * g[col] / g.u
        print(f"    {rl}: {len(g)} placed  {int((gp>0).sum())}-{int((gp<0).sum())}  "
              f"ROI {gp.sum()/g.stake_u.sum():+.1%}")


def _lastn(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    p = re.sub(r"[^a-z ]", "", s.lower().replace("-", " ")).split()
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


def placeable_prices(B):
    """Decimal price of the bet side at the EARLIEST placeable_lines.csv capture
    for each live=1 bet (the price on the board when the pick went on record).
    NaN for live=0 rows and for live rows with no capture (every row before
    2026-09-08, when captures began)."""
    out = pd.Series(np.nan, index=B.index)
    if not os.path.exists(PLACEABLE):
        return out
    pl = pd.read_csv(PLACEABLE)
    if len(pl) == 0:
        return out
    pl["d"] = pd.to_datetime(pl.event_date)
    bl = pd.read_csv("data/bfo_lines.csv")
    bl["d"] = pd.to_datetime(bl.event_date)
    for i, r in B[B.live == 1].iterrows():
        if not ((pl.d - pd.Timestamp(r.event_date)).abs().dt.days <= 3).any():
            continue                            # nothing captured for this card
        mu, a_is_f1 = _bfo_mu(bl, r)
        c = pl[pl.mu == mu].sort_values("captured_at")
        if len(c):
            x = c.iloc[0]
            out.loc[i] = float(x.f1_line) if bool(r.mod1) == a_is_f1 else float(x.f2_line)
    return out


def write_ledger(B, path=LEDGER):
    """Write every triggered bet in B to `path`, sorted by date then bout order.
    open_line / placeable_line / clean_close are American odds of the bet side;
    clv_pts is the bet side's vig-free close minus open, in prob points; pnl
    (placeable where captured, else open) and pnl_at_open are in units.
    rd_self / rd_opp are the pre-fight RDs of the bet side and the opponent,
    placed / stake_u the placement-policy decision and pnl_placed its P&L
    (stake_u * pnl per unit); method / round come from fights_v2."""
    bl = pd.read_csv("data/bfo_lines.csv")
    bl["d"] = pd.to_datetime(bl.event_date)
    rows = []
    for _, r in B.iterrows():
        ac, bc = _clean_close(bl, r)
        pick_a = bool(r.mod1)
        o, c = (r.a_open, ac) if pick_a else (r.b_open, bc)
        qo = (1 / o) / (1 / r.a_open + 1 / r.b_open)
        qc = (1 / c) / (1 / ac + 1 / bc)
        rows.append(dict(event_date=r.event_date, event=r.event_name,
                         fight_id=r.fight_id,
                         fighter=r.fighter_a if pick_a else r.fighter_b,
                         rule=r.rule, live=int(r.live), units=int(r.u),
                         rd_self=round(float(r.rd_self), 2),
                         rd_opp=round(float(r.rd_opp), 2),
                         placed=int(r.placed), stake_u=round(float(r.stake_u), 2),
                         pnl_placed=round(float(r.pnl_placed), 4),
                         open_line=int(round(float(am(o)))),
                         placeable_line=("" if pd.isna(r.placeable)
                                         else int(round(float(am(r.placeable))))),
                         clean_close=int(round(float(am(c)))),
                         clv_pts=round((qc - qo) * 100, 2),
                         result="WIN" if r.pnl > 0 else "LOSS",
                         method=r["method"], pnl=round(float(r.pnl), 4),
                         pnl_at_open=round(float(r.pnl_at_open), 4),
                         _bo=r.bout_order, **{"round": int(r["rnd"])}))
    L = (pd.DataFrame(rows, columns=LEDGER_COLS + ["_bo"])
         .sort_values(["event_date", "_bo"]).drop(columns="_bo"))
    L.to_csv(path, index=False)
    print(f"wrote {path}: {len(L)} rows  staked {L.units.sum()}u  "
          f"P&L {L.pnl.sum():+.1f}u (at open {L.pnl_at_open.sum():+.1f}u)  "
          f"placed {int(L.placed.sum())} staked {L.stake_u.sum():.1f}u "
          f"P&L {L.pnl_placed.sum():+.1f}u")


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
    report(B[B.live == 1], f"LIVE RECORD ({a.year})", placeable=True)
    report(B, f"FULL SIM ({a.year})")
    if a.write_ledger:
        write_ledger(B)
