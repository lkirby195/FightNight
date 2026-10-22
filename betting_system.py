"""FightNight betting system — apply the KISS rule and settle.

Rule (flat 1 unit at the OPENING line), decided by signals.classify():
  FLIP: model picks the market underdog AND model prob >= 0.65
  GAP : model agrees on favorite (or the opens are a pick'em) AND prices them
        >= 100 American pts stronger
Pre-filter: skip bouts whose opening implied-prob sum is outside [1.00, 1.12].
Every threshold, the tie-break, the debut rule and the placement stake live in
signals.py; this module joins the data, runs the classifier over every bout
and settles. Nothing here decides a signal.

Requires: data/model_only_preds.csv, data/bfo_joined.csv, data/fights_v2.csv,
data/fighters_v2.csv, data/prefight_rd.csv
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
every signal is still logged at flat 1u (units, pnl, pnl_at_open); the stake
(signals.stake_units, from the pre-fight RDs in data/prefight_rd.csv and the
bet side's price of record) decides whether it is placed (placed, stake_u) and
pnl_placed = stake_u * pnl per unit. A PLACED (policy v1) block follows each
summary.
Opener validity (2026-09-30): signals evaluate at the VALIDATED open
(scrape_bfo.validated_open; a_open / b_open in bfo_joined.csv), and so does
the pre-filter. A bout whose opener was corrected within the hour carries
open_suspect=1 with the raw first ticks in a_open_raw / b_open_raw; a signal
that fires only against that raw tick is still logged, as a VOID row
(units 0, placed 0, stake_u 0, reason "void: suspect opener") that no summary
counts -- the same idea as card_report.OFF_CARD: visible, never settled.
Debut rule (ruled 2026-10-22): a bout is in the ledger only when both fighters
were on file with a UFC bout before the event (signals.first_bout_dates). A
bout with a debutant gets no projection, no signal and no row; the rule is
decided from fights_v2, never from a per-bout list. (It replaces the 2026-09-30
live-flag debut rule, which kept such rows as live=0.)
On-record rule (2026-09-30): cards reported before OPEN_RULE_FROM were
evaluated pre-event at the raw first tick, so on those cards a live signal
must also fire at the raw opener (same pick); a signal that exists only under
the validated open there was never on record and is live=0 (backfilled).
pm_p_at_capture: the Polymarket price of the backed side at the earliest
placeable capture, for placed rows whose capture recorded one (polymarket.py).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import signals
from names import bfo_lastn, lastn as _lastn
# The constants are defined once, in signals.py; re-exported here for the
# callers and tests that import them from betting_system.
from signals import (FLIP_CONF, FLIP_UNITS, GAP_PTS, GAP_UNITS, RAMP_HI, RAMP_LO,  # noqa: F401
                     SKIP_LINE_MAX, UNKNOWN_RD, VOID_SUSPECT, stake_units)

# Live record: every card from LIVE_FROM through LIVE_THROUGH was reported
# pre-event, plus the dates in LIVE_CARDS. ADD EACH NEW CARD'S DATE TO LIVE_CARDS
# WHEN ITS REPORT IS RUN PRE-EVENT; anything else is backfilled (live=0).
LIVE_FROM, LIVE_THROUGH = "2026-01-01", "2026-07-25"
OPEN_RULE_FROM = "2026-09-30"   # card reports evaluate the validated open from here
LIVE_CARDS = {"2026-08-22",   # UFC Sacramento (no bets)
              "2026-08-29",   # UFC Shanghai
              "2026-09-05",   # UFC Paris
              "2026-09-12",   # Noche UFC Glendale (no bets)
              "2026-09-19",   # UFC 331 (FLIP Joshua van, 1.00u)
              "2026-10-24"}   # UFC 333 (FLIP Jacob Malkoun, 0.92u)
# Not live, settled as backfill (no pre-event capture reached main):
# 2026-09-26 Vegas 121, 2026-10-03 UFC 332, 2026-10-10 Allen/Duncan,
# 2026-10-17 Edmonton.

LEDGER = "data/system_ledger.csv"
PLACEABLE = "data/placeable_lines.csv"      # written by card_report.py
PREFIGHT_RD = "data/prefight_rd.csv"        # written by prefight_rd.py
LEDGER_COLS = ["event_date", "event", "fight_id", "fighter", "rule", "live", "units",
               "rd_self", "rd_opp", "placed", "stake_u", "pnl_placed",
               "open_line", "placeable_line", "clean_close", "clv_pts", "result",
               "method", "round", "pnl", "pnl_at_open", "reason", "pm_p_at_capture"]
B_COLS = ["event_date", "rule", "live", "u", "pnl", "pnl_at_open", "placed", "stake_u"]

am = lambda dec: np.where(dec >= 2, (dec - 1) * 100, -100 / (dec - 1))
amp = lambda q: np.where(q >= .5, -100 * q / (1 - q), 100 * (1 - q) / q)


def load():
    """One row per bout with a model-only prediction, a BFO opening pair and
    a result: the inputs signals.classify needs, plus the debut-rule flags
    (known_a / known_b / projectable) from fights_v2 and fighters_v2."""
    M = pd.read_csv("data/model_only_preds.csv")
    J = pd.read_csv("data/bfo_joined.csv")
    fv_all = pd.read_csv("data/fights_v2.csv")
    fv = fv_all[["fight_id", "event_date", "event_name", "bout_order", "fighter_a_id",
                 "fighter_a", "fighter_b_id", "fighter_b", "method", "round"]
                ].rename(columns={"round": "rnd"})
    rd = pd.read_csv(PREFIGHT_RD)               # fight_id, rd_a, rd_b
    D = (M.merge(J, on="fight_id").merge(fv, on="fight_id")
          .merge(rd, on="fight_id", how="left"))
    D = D.dropna(subset=["y_a", "a_open", "b_open"])
    for c, src in (("open_suspect", 0), ("a_open_raw", "a_open"), ("b_open_raw", "b_open")):
        if c not in D:                          # bfo_joined.csv from before 2026-09-30
            D[c] = src if isinstance(src, int) else D[src]
    D["open_suspect"] = D.open_suspect.fillna(0).astype(int)
    D["a_open_raw"] = D.a_open_raw.fillna(D.a_open)
    D["b_open_raw"] = D.b_open_raw.fillna(D.b_open)
    first = signals.first_bout_dates(fv_all)
    known = set(pd.read_csv("data/fighters_v2.csv").fighter_id)
    D["known_a"] = [fid in known and signals.on_record(fid, d, first)
                    for fid, d in zip(D.fighter_a_id, D.event_date)]
    D["known_b"] = [fid in known and signals.on_record(fid, d, first)
                    for fid, d in zip(D.fighter_b_id, D.event_date)]
    D["projectable"] = D.known_a & D.known_b
    return D.sort_values("event_date")


def is_live(event_date: str) -> bool:
    return LIVE_FROM <= event_date <= LIVE_THROUGH or event_date in LIVE_CARDS


def _decision(r, dec_a=None, dec_b=None):
    """signals.classify for one row of load()'s frame, fighter_a as side 1."""
    return signals.classify(float(r.p_model_a), float(r.a_open), float(r.b_open),
                            float(r.rd_a), float(r.rd_b),
                            known1=bool(r.known_a), known2=bool(r.known_b),
                            o1_raw=float(r.a_open_raw), o2_raw=float(r.b_open_raw),
                            open_suspect=bool(r.open_suspect),
                            dec1=dec_a, dec2=dec_b)


def run(D):
    """Classify every bout of D with signals.classify -> the ledger frame B:
    one row per signal (u = 1) or VOID (u = 0, reason set), with the price of
    record, the policy-v1 stake and the live flag. A bout the debut rule
    excludes (projectable False) has no row."""
    if D.rd_a.isna().any():
        raise SystemExit(f"{PREFIGHT_RD} has no row for {int(D.rd_a.isna().sum())} "
                         f"bouts: run prefight_rd.py")
    # pass 1: which bouts fire at all (the stake is the only thing the price of
    # record changes, so it can wait for pass 2)
    D = D[[_decision(r).rule is not None for r in D.itertuples(index=False)]].copy()
    if len(D) == 0:
        return pd.DataFrame(columns=B_COLS)
    # price of record: the earliest capture for bouts on live cards that have
    # one (fails loudly when such a bout cannot be matched to its BFO series)
    D["dec_a"], D["dec_b"], D["pm_a"] = capture_prices(D)
    rows = []
    for r in D.itertuples(index=False):
        dec_a = None if pd.isna(r.dec_a) else float(r.dec_a)
        dec_b = None if pd.isna(r.dec_b) else float(r.dec_b)
        d = _decision(r, dec_a, dec_b)
        mod1 = d.side == 1
        u = 0 if d.void else (FLIP_UNITS if d.rule == "FLIP" else GAP_UNITS)
        won = (int(r.y_a) == 1) == mod1
        # a VOID row is logged at the raw opener it fired on
        a_open, b_open = (r.a_open_raw, r.b_open_raw) if d.void else (r.a_open, r.b_open)
        dec_open = a_open if mod1 else b_open
        pnl_open = 0.0 if d.void else (u * (dec_open - 1) if won else -float(u))
        placeable = dec_a if mod1 else dec_b
        if d.void or placeable is None:
            pnl, placeable = pnl_open, np.nan
        else:
            pnl = u * (placeable - 1) if won else -float(u)
        pm_p = np.nan if pd.isna(r.pm_a) else (float(r.pm_a) if mod1 else 1 - float(r.pm_a))
        raw = signals.rule_at(float(r.p_model_a), float(r.a_open_raw), float(r.b_open_raw))
        on_rec = r.event_date >= OPEN_RULE_FROM or (raw is not None and raw[1] == d.side)
        rows.append(dict(r._asdict(), rule=d.rule, mod1=mod1, conf=d.conf, u=u,
                         a_open=a_open, b_open=b_open, reason=d.reason, void=d.void,
                         pnl=pnl, pnl_at_open=pnl_open, placeable=placeable, pm_p=pm_p,
                         on_record=on_rec, live=int(is_live(r.event_date) and on_rec),
                         rd_self=r.rd_a if mod1 else r.rd_b, rd_opp=r.rd_b if mod1 else r.rd_a,
                         bet_line=d.line, stake_u=d.stake, placed=int(d.placed),
                         pnl_placed=(d.stake * pnl / u) if u else 0.0))
    return pd.DataFrame(rows).sort_values("event_date")


def capture_prices(D):
    """(decimal price of fighter a, of fighter b, Polymarket price of fighter a)
    at the EARLIEST placeable_lines.csv capture of each bout of D that lies on
    a live card with captures (the price on the board when the pick went on
    record). NaN elsewhere: every row before 2026-09-08, when captures began,
    and every backfilled card. The Polymarket price is NaN as well when the
    capture recorded none (pm_p_self, fighter1's price, from 2026-09-30)."""
    nan = lambda: pd.Series(np.nan, index=D.index, dtype=float)
    da, db, pm = nan(), nan(), nan()
    if not os.path.exists(PLACEABLE):
        return da, db, pm
    pl = pd.read_csv(PLACEABLE)
    if len(pl) == 0:
        return da, db, pm
    if "pm_p_self" not in pl:
        pl["pm_p_self"] = np.nan
    pl["d"] = pd.to_datetime(pl.event_date)
    bl = None
    for i, r in D.iterrows():
        if not is_live(r.event_date):
            continue
        if not ((pl.d - pd.Timestamp(r.event_date)).abs().dt.days <= 3).any():
            continue                            # nothing captured for this card
        if bl is None:
            bl = pd.read_csv("data/bfo_lines.csv")
            bl["d"] = pd.to_datetime(bl.event_date)
        mu, a_is_f1 = _bfo_mu(bl, r)
        c = pl[pl.mu == mu].sort_values("captured_at")
        if len(c):
            x = c.iloc[0]
            f1, f2 = float(x.f1_line), float(x.f2_line)
            da.loc[i], db.loc[i] = (f1, f2) if a_is_f1 else (f2, f1)
            if pd.notna(x.pm_p_self):
                pm.loc[i] = float(x.pm_p_self) if a_is_f1 else 1 - float(x.pm_p_self)
    return da, db, pm


def counted(B):
    """The rows a summary counts: every signal that is not VOID."""
    return B[B.u > 0]


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
    n_void = int((B.u == 0).sum()) if len(B) else 0
    B = counted(B) if len(B) else B
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
    if n_void:
        print(f"    VOID (not counted): {n_void} ({VOID_SUSPECT})")
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


def _bfo_mu(bl, r):
    """BFO matchup id for one bout: same date (+-3d), same opening prices
    (validated or raw), same surnames (names.py: BFO aliases applied,
    generational suffixes dropped). -> (mu, a_is_f1). Fails loudly rather
    than guess."""
    c = bl[(bl.d - pd.Timestamp(r.event_date)).abs().dt.days <= 3]
    f1r = c.f1_open_raw if "f1_open_raw" in c else c.f1_open
    f2r = c.f2_open_raw if "f2_open_raw" in c else c.f2_open
    c = c[((c.f1_open == r.a_open) & (c.f2_open == r.b_open)) |
          ((c.f1_open == r.b_open) & (c.f2_open == r.a_open)) |
          ((f1r == r.a_open) & (f2r == r.b_open)) |
          ((f1r == r.b_open) & (f2r == r.a_open))]
    surnames = {_lastn(r.fighter_a), _lastn(r.fighter_b)}
    if len(c):
        c = c[c.apply(lambda x: {bfo_lastn(x.fighter1), bfo_lastn(x.fighter2)} == surnames,
                      axis=1).astype(bool)]
    if len(c) != 1:
        raise SystemExit(f"--write-ledger: {len(c)} BFO matchups for {r.fight_id} "
                         f"{r.fighter_a} vs {r.fighter_b} {r.event_date}")
    x = c.iloc[0]
    if _lastn(r.fighter_a) != _lastn(r.fighter_b):
        return int(x.mu), bfo_lastn(x.fighter1) == _lastn(r.fighter_a)
    return int(x.mu), bool(x.f1_open == r.a_open or
                           x.get("f1_open_raw", x.f1_open) == r.a_open)


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
    open_line / placeable_line / clean_close are American odds of the bet side;
    clv_pts is the bet side's vig-free close minus open, in prob points; pnl
    (placeable where captured, else open) and pnl_at_open are in units.
    rd_self / rd_opp are the pre-fight RDs of the bet side and the opponent,
    placed / stake_u the placement-policy decision and pnl_placed its P&L
    (stake_u * pnl per unit); method / round come from fights_v2. VOID rows
    (reason set, units 0) are logged at the raw opener they fired on, result
    VOID, and settle nothing; pm_p_at_capture is the Polymarket price of the
    backed side at the earliest capture, placed rows only."""
    bl = pd.read_csv("data/bfo_lines.csv")
    bl["d"] = pd.to_datetime(bl.event_date)
    rows = []
    for _, r in B.iterrows():
        ac, bc = _clean_close(bl, r)
        pick_a = bool(r.mod1)
        o, c = (r.a_open, ac) if pick_a else (r.b_open, bc)
        qo = (1 / o) / (1 / r.a_open + 1 / r.b_open)
        qc = (1 / c) / (1 / ac + 1 / bc)
        void = int(r.u) == 0
        won = (r.y_a == 1) == pick_a
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
                         result="VOID" if void else ("WIN" if won else "LOSS"),
                         method=r["method"], pnl=round(float(r.pnl), 4),
                         pnl_at_open=round(float(r.pnl_at_open), 4),
                         reason=r.reason,
                         pm_p_at_capture=("" if (pd.isna(r.pm_p) or int(r.placed) != 1)
                                          else round(float(r.pm_p), 4)),
                         _bo=r.bout_order, **{"round": int(r["rnd"])}))
    L = (pd.DataFrame(rows, columns=LEDGER_COLS + ["_bo"])
         .sort_values(["event_date", "_bo"]).drop(columns="_bo"))
    L.to_csv(path, index=False)
    print(f"wrote {path}: {len(L)} rows  staked {L.units.sum()}u  "
          f"P&L {L.pnl.sum():+.1f}u (at open {L.pnl_at_open.sum():+.1f}u)  "
          f"placed {int(L.placed.sum())} staked {L.stake_u.sum():.1f}u "
          f"P&L {L.pnl_placed.sum():+.1f}u  void {int((L.units == 0).sum())}")


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
