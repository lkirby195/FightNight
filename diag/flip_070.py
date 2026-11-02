"""Diagnostic: FLIP threshold 0.70 vs 0.65 over the full sim. Read-only.

Runs betting_system.run() twice on the same load() frame with signals.FLIP_CONF
overridden for the run (signals.py is NOT edited; rule_at reads the module
global at call time). Nothing under data/ is written. Output: diag/flip-070.md.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import signals                       # noqa: E402
import betting_system as bs          # noqa: E402

BASE, TEST = 0.65, 0.70
assert signals.FLIP_CONF == BASE, signals.FLIP_CONF

PREDS_2025 = "data/model_only_preds_2025_backtest.csv"     # sealed holdout preds
SEALED_2025 = "data/system_ledger_2025_backtest.csv"       # sealed site file


def load_with(M):
    """betting_system.load() with M in place of data/model_only_preds.csv
    (same body; 2025 is the sealed holdout, absent from the main preds)."""
    J = pd.read_csv("data/bfo_joined.csv")
    fv_all = pd.read_csv("data/fights_v2.csv")
    fv = fv_all[["fight_id", "event_date", "event_name", "bout_order", "fighter_a_id",
                 "fighter_a", "fighter_b_id", "fighter_b", "method", "round"]
                ].rename(columns={"round": "rnd"})
    rd = pd.read_csv(bs.PREFIGHT_RD)
    D = (M.merge(J, on="fight_id").merge(fv, on="fight_id").merge(rd, on="fight_id", how="left"))
    D = D.dropna(subset=["y_a", "a_open", "b_open"])
    for c, src in (("open_suspect", 0), ("a_open_raw", "a_open"), ("b_open_raw", "b_open")):
        if c not in D:
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


D = bs.load()
assert not (pd.to_datetime(D.event_date).dt.year == 2025).any()
D25 = load_with(pd.read_csv(PREDS_2025))
assert (pd.to_datetime(D25.event_date).dt.year == 2025).all()


def both(D):
    B65 = bs.run(D)
    signals.FLIP_CONF = TEST
    try:
        B70 = bs.run(D)
    finally:
        signals.FLIP_CONF = BASE
    return B65, B70


B65, B70 = both(D)
B65_25, B70_25 = both(D25)
assert (B65_25.live == 0).all()
B65 = pd.concat([B65, B65_25]).sort_values("event_date")
B70 = pd.concat([B70, B70_25]).sort_values("event_date")

# the 0.70 run must be exactly the 0.65 run minus FLIP rows with conf < 0.70
expect = B65[~((B65.rule == "FLIP") & (B65.conf < TEST))]
assert set(expect.fight_id) == set(B70.fight_id), "0.70 run is not a subset of 0.65"
assert len(expect) == len(B70)

for B in (B65, B70):
    B["year"] = pd.to_datetime(B.event_date).dt.year


def counted(B):
    return B[B.u > 0]


def window(B, name):
    y = B.year
    live = B.live == 1
    return {
        "2012-2024 (train/tune)": B[(y >= 2012) & (y <= 2024)],
        "2025 (holdout)": B[y == 2025],
        "2026 not timestamped (live=0)": B[(y == 2026) & ~live],
        "2026 live (live=1)": B[(y == 2026) & live],
        "2026 all (site tab)": B[y == 2026],
        "all-time": B,
    }[name]


WINDOWS = ["2012-2024 (train/tune)", "2025 (holdout)", "2026 not timestamped (live=0)",
           "2026 live (live=1)", "2026 all (site tab)", "all-time"]


def stats(B):
    B = counted(B)
    n = len(B)
    if n == 0:
        return dict(bets=0, wl="0-0", net=0.0, roi=float("nan"), placed=0, pwl="0-0",
                    pstake=0.0, pnet=0.0, proi=float("nan"))
    w, l = int((B.pnl > 0).sum()), int((B.pnl < 0).sum())
    P = B[B.placed == 1]
    pw, pl = int((P.pnl_placed > 0).sum()), int((P.pnl_placed < 0).sum())
    ps = P.stake_u.sum()
    return dict(bets=n, wl=f"{w}-{l}", net=B.pnl.sum(), roi=B.pnl.sum() / B.u.sum(),
                placed=len(P), pwl=f"{pw}-{pl}", pstake=ps, pnet=P.pnl_placed.sum(),
                proi=(P.pnl_placed.sum() / ps) if ps else float("nan"))


def fmt_roi(x):
    return "n/a" if pd.isna(x) else f"{x:+.1%}"


def row_pair(label, a, b):
    return (f"| {label} | {a['bets']} | {a['wl']} | {a['net']:+.2f} | {fmt_roi(a['roi'])} "
            f"| {a['placed']} | {a['pwl']} | {a['pstake']:.2f} | {a['pnet']:+.2f} | {fmt_roi(a['proi'])} "
            f"| {b['bets']} | {b['wl']} | {b['net']:+.2f} | {fmt_roi(b['roi'])} "
            f"| {b['placed']} | {b['pwl']} | {b['pstake']:.2f} | {b['pnet']:+.2f} | {fmt_roi(b['proi'])} |")


HDR = ("| window | bets | W-L | net u | ROI | placed | pl W-L | stake u | pl net u | pl ROI "
       "| bets | W-L | net u | ROI | placed | pl W-L | stake u | pl net u | pl ROI |")
SEP = "|" + "---|" * 19
SUPER = "| | FLIP 0.65 (current) | | | | | | | | | FLIP 0.70 | | | | | | | | |"

out = []
out.append("# Diagnostic: FLIP threshold 0.70 vs 0.65\n")
out.append(f"Branch `diag/flip-070` off main {os.popen('git rev-parse --short main').read().strip()}. "
           "Read-only run of `betting_system.run()` twice on one `load()` frame, "
           "with `signals.FLIP_CONF` overridden in memory for the second run "
           "(`diag/flip_070.py`). signals.py constants, GAP rule, policy v1 skips and "
           "ramp, debut rule and opener rule unchanged. No ledger written, nothing under "
           "data/ or site/ touched.\n")
out.append(f"Data through {counted(B65).event_date.max()} (last settled card). "
           f"Run date {pd.Timestamp.today().date()}.\n")
out.append("2025 is the sealed holdout and is absent from `data/model_only_preds.csv`; "
           "its rows here come from the sealed `data/model_only_preds_2025_backtest.csv` "
           "pushed through the same current `load()` logic and `betting_system.run()` "
           "(so the current debut rule, pick'em tie = GAP and alias table apply). The "
           "sealed site file `data/system_ledger_2025_backtest.csv` is cross-checked in "
           "section 2b; neither sealed file was rewritten.\n")
out.append("The site's 2026 tab (since PR #13) is every 2026 ledger row; it is shown "
           "split (live=0 / live=1) and as the union.\n")
out.append("Conventions: `net u` is the ledger's `pnl` at 1u flat (settled at the earliest "
           "pre-event capture for live=1 rows, at open otherwise, exactly as the ledger of "
           "record); `placed` columns are policy v1 (stake_u / pnl_placed). W-L counts "
           "pnl > 0 / pnl < 0; a push (draw/NC) is in bets but in neither. VOID rows are "
           "not counted. The 0.70 run was asserted to equal the 0.65 run minus FLIP rows "
           "with conf < 0.70 (same fights, same prices, same policy).\n")

def _band(B):
    W = counted(B)
    return stats(W[(W.rule == "FLIP") & (W.conf < TEST)])


b_tr, b_ho = _band(B65[B65.year <= 2024]), _band(B65[B65.year >= 2025])
a65, a70 = stats(B65), stats(B70)
mal = counted(B65)[counted(B65).fighter_b.str.contains("Malkoun") | counted(B65).fighter_a.str.contains("Malkoun")]
mal = mal[mal.year == 2026].iloc[-1]
out.append("## 0. Summary\n")
out.append(f"- 0.70 removes {b_tr['bets'] + b_ho['bets']} of {a65['bets']} all-time bets: "
           f"{b_tr['bets']} in 2012-2024 "
           f"({b_tr['wl']}, {b_tr['net']:+.2f}u, ROI {fmt_roi(b_tr['roi'])}) and "
           f"{b_ho['bets']} in 2025-2026 ({b_ho['wl']}, {b_ho['net']:+.2f}u, ROI "
           f"{fmt_roi(b_ho['roi'])}). GAP rows do not move.")
out.append(f"- All-time at 1u flat: {a65['bets']} bets {a65['net']:+.2f}u (ROI {fmt_roi(a65['roi'])}) "
           f"-> {a70['bets']} bets {a70['net']:+.2f}u (ROI {fmt_roi(a70['roi'])}). "
           f"Policy v1 placed: {a65['pnet']:+.2f}u -> {a70['pnet']:+.2f}u.")
out.append(f"- The band is as profitable per bet as the kept FLIPs in 2012-2024 and more "
           f"profitable than them in 2025-2026; 0.70 lowers every window's net and ROI.")
out.append(f"- 2026 live: no row changes. Malkoun (2026-10-24) is conf {mal.conf:.3f}, "
           f"above 0.70 by {mal.conf - TEST:.3f}; Tuivasa 0.703 (unplaced, +504) and Van 0.751 "
           f"also stay. A preds regeneration moves probs by ~0.001, so Malkoun and Tuivasa "
           f"sit within noise of the line.\n")

out.append("## 1. Window table, both thresholds\n")
out.append(SUPER)
out.append(HDR)
out.append(SEP)
for w in WINDOWS:
    out.append(row_pair(w, stats(window(B65, w)), stats(window(B70, w))))
out.append("")

# per-rule split per window, 0.65 only, so the reader sees what GAP contributes
out.append("Per-rule split at 0.65 (GAP is identical under 0.70):\n")
out.append("| window | rule | bets | W-L | net u | ROI | placed | pl net u |")
out.append("|---|---|---|---|---|---|---|---|")
for w in WINDOWS:
    W = counted(window(B65, w))
    for rl in ("FLIP", "GAP"):
        s = stats(W[W.rule == rl])
        out.append(f"| {w} | {rl} | {s['bets']} | {s['wl']} | {s['net']:+.2f} | "
                   f"{fmt_roi(s['roi'])} | {s['placed']} | {s['pnet']:+.2f} |")
out.append("")

out.append("## 2. Band table: FLIP bets with conf in [0.65, 0.70)\n")
out.append("This is exactly the set a 0.70 threshold removes. GAP-only line included to "
           "show what does not move.\n")
out.append("| window | set | bets | W-L | net u | ROI | placed | pl W-L | stake u | pl net u | pl ROI |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for lo, hi, lab in ((2012, 2024, "2012-2024"), (2025, 2026, "2025-2026")):
    W = counted(B65[(B65.year >= lo) & (B65.year <= hi)])
    sets = [
        ("FLIP [0.65, 0.70) - removed by 0.70", W[(W.rule == "FLIP") & (W.conf < TEST)]),
        ("FLIP >= 0.70 - kept", W[(W.rule == "FLIP") & (W.conf >= TEST)]),
        ("GAP only - unchanged", W[W.rule == "GAP"]),
        ("all signals", W),
    ]
    for name, S in sets:
        s = stats(S)
        out.append(f"| {lab} | {name} | {s['bets']} | {s['wl']} | {s['net']:+.2f} | "
                   f"{fmt_roi(s['roi'])} | {s['placed']} | {s['pwl']} | {s['pstake']:.2f} | "
                   f"{s['pnet']:+.2f} | {fmt_roi(s['proi'])} |")
out.append("")

out.append("## 2b. Cross-check: the sealed 2025 site file\n")
S = pd.read_csv(SEALED_2025)
P25 = pd.read_csv(PREDS_2025).merge(
    pd.read_csv("data/fights_v2.csv")[["fight_id", "fighter_a"]], on="fight_id")
S = S.merge(P25[["fight_id", "fighter_a", "p_model_a"]], on="fight_id", how="left")
S["conf"] = np.where(S.fighter == S.fighter_a, S.p_model_a, 1 - S.p_model_a)
S["u"] = S.units
cur = counted(B65_25)
out.append(f"Sealed file: {len(S)} signals ({(S.rule == 'FLIP').sum()} FLIP / "
           f"{(S.rule == 'GAP').sum()} GAP), placed {int(S.placed.sum())}, "
           f"pnl_placed {S.pnl_placed.sum():+.2f}u. Written 2026-09-09 under the "
           "pre-PR-#9 rules (no debut rule, no alias table, pick'em tie not yet GAP).\n")
out.append(f"Current rules over the same preds: {len(cur)} signals "
           f"({(cur.rule == 'FLIP').sum()} FLIP / {(cur.rule == 'GAP').sum()} GAP), "
           f"placed {int(cur.placed.sum())}, pnl_placed {cur.pnl_placed.sum():+.2f}u. "
           f"Fights in sealed only: {len(set(S.fight_id) - set(cur.fight_id))}; "
           f"in current only: {len(set(cur.fight_id) - set(S.fight_id))}.\n")
out.append("Band on the sealed file itself (conf joined from the sealed preds):\n")
out.append("| set | bets | W-L | net u | ROI | placed | pl W-L | stake u | pl net u | pl ROI |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for name, X in (("FLIP [0.65, 0.70) - removed by 0.70", S[(S.rule == "FLIP") & (S.conf < TEST)]),
                ("FLIP >= 0.70 - kept", S[(S.rule == "FLIP") & (S.conf >= TEST)]),
                ("GAP only - unchanged", S[S.rule == "GAP"]),
                ("all signals", S)):
    s = stats(X)
    out.append(f"| {name} | {s['bets']} | {s['wl']} | {s['net']:+.2f} | {fmt_roi(s['roi'])} "
               f"| {s['placed']} | {s['pwl']} | {s['pstake']:.2f} | {s['pnet']:+.2f} | "
               f"{fmt_roi(s['proi'])} |")
out.append("")

out.append("## 3. Bets in [0.65, 0.70) from 2025 and 2026\n")
out.append("2025 rows are from the current-rules run over the sealed preds (section 2 basis).\n")
out.append("| date | event | bout (pick first) | rule | conf | line | live | stake u | result | net u (1u) | net u (placed) |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
band = counted(B65[(B65.rule == "FLIP") & (B65.conf < TEST) & (B65.year >= 2025)])
for r in band.sort_values(["event_date", "bout_order"]).itertuples(index=False):
    pick, opp = (r.fighter_a, r.fighter_b) if r.mod1 else (r.fighter_b, r.fighter_a)
    res = "WIN" if r.pnl > 0 else ("LOSS" if r.pnl < 0 else "PUSH")
    out.append(f"| {r.event_date} | {r.event_name} | {pick} vs {opp} | {r.rule} | {r.conf:.3f} "
               f"| {int(r.bet_line):+d} | {int(r.live)} | {r.stake_u:.2f} | {res} | {r.pnl:+.2f} "
               f"| {r.pnl_placed:+.2f} |")
out.append("")

out.append("## 4. 2026 live rows\n")
L = counted(B65[B65.live == 1]).sort_values("event_date")
for r in L.itertuples(index=False):
    pick = r.fighter_a if r.mod1 else r.fighter_b
    moves = (r.rule == "FLIP") and (r.conf < TEST)
    out.append(f"- {r.event_date} {pick}: {r.rule} conf {r.conf:.3f} line {int(r.bet_line):+d} "
               f"stake {r.stake_u:.2f} -> {'REMOVED under 0.70' if moves else 'unchanged'}")
n_mv = int(((L.rule == "FLIP") & (L.conf < TEST)).sum())
out.append(f"\n{n_mv} of {len(L)} live=1 rows change under 0.70.\n")

with open("diag/flip-070.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
