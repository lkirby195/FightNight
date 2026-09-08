"""Replay check (merge gate): replay data/fights_v2.csv through engine.py with
best_params.json and compare the result to the frozen v1 ratings in
data/ratings.csv.

Replays every fight dated on or before the last fight in ratings.csv (the v1
freeze date), applying best_params exactly as card_report.build_states does.
Prints fighters compared / missing / mismatches / max delta and exits non-zero
if any frozen fighter is missing from the replay or max |rating delta| > 0.05
(v1 ratings are rounded to 0.1, so 0.05 is the rounding tolerance).

Usage: python replay_check.py            # from the repo root
"""
from __future__ import annotations

import csv
import json
import sys

import engine

TOL = 0.05


def apply_params(bp: dict) -> None:
    engine.TAU = bp["tau"]
    engine.SIGMA0 = bp["sigma0"]
    for k, v in [("U-DEC", bp["s_udec"]), ("M-DEC", bp["s_udec"]),
                 ("S-DEC", bp["s_sdec"]), ("DQ", bp["s_sdec"])]:
        engine.S_TABLE[k] = (v, 1 - v)


def main() -> int:
    apply_params(json.load(open("best_params.json")))
    with open("data/ratings.csv", encoding="utf-8", newline="") as fh:
        frozen = {r["fighter_id"]: r for r in csv.DictReader(fh)}
    cutoff = max(r["last_fight"] for r in frozen.values())
    with open("data/fights_v2.csv", encoding="utf-8", newline="") as fh:
        fights = list(csv.DictReader(fh))
    fights.sort(key=lambda r: (r["event_date"], int(r["bout_order"])))

    eng = engine.Engine()
    n = 0
    for f in fights:
        if f["event_date"] > cutoff:
            continue
        eng.process({"event_date": f["event_date"],
                     "fighter_a_id": f["fighter_a_id"], "fighter_a": f["fighter_a"],
                     "fighter_b_id": f["fighter_b_id"], "fighter_b": f["fighter_b"],
                     "outcome": f["outcome"], "method": f["method"],
                     "a_missed_weight": f.get("a_mw", ""),
                     "b_missed_weight": f.get("b_mw", "")})
        n += 1

    missing, deltas = [], []
    for fid, r in frozen.items():
        fo = eng.fighters.get(fid)
        if fo is None:
            missing.append(r["name"])
            continue
        deltas.append((abs(fo.rating - float(r["rating"])),
                       abs(fo.rd - float(r["rd"])), fo.name))
    max_r = max(deltas, key=lambda d: d[0]) if deltas else (0.0, 0.0, "-")
    max_rd = max(d[1] for d in deltas) if deltas else 0.0
    mismatches = [d for d in deltas if d[0] > TOL + 1e-9]

    print(f"replayed {n} fights through {cutoff} -> {len(eng.fighters)} fighters")
    print(f"fighters compared: {len(deltas)} / {len(frozen)} in data/ratings.csv"
          f"  missing: {len(missing)}")
    print(f"mismatches (|rating delta| > {TOL}): {len(mismatches)}")
    print(f"max |rating delta|: {max_r[0]:.4f} ({max_r[2]})   max |rd delta|: {max_rd:.4f}")
    for name in missing[:10]:
        print(f"  missing: {name}")
    for d in sorted(mismatches, reverse=True)[:10]:
        print(f"  mismatch: {d[2]} rating off by {d[0]:.3f}")
    ok = not missing and not mismatches
    print("REPLAY CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
