"""Pre-fight RD table: data/prefight_rd.csv (fight_id, rd_a, rd_b).

Deterministic replay of data/fights_v2.csv through engine.py with
best_params.json, the same replay as replay_check.py and
card_report.build_states. For every fight, rd_a / rd_b are the time-inflated
pre-fight rating deviations of the two fighters on the public scale
(engine.SCALE * phi after engine.inflate), i.e. the uncertainty each carried
into the bout, taken before the bout is rated. Debutants carry 350.
betting_system.py joins this by fight_id for the placement policy.

Usage: python prefight_rd.py            # from the repo root
"""
from __future__ import annotations

import csv
import json
from datetime import date

import engine
from replay_check import apply_params

OUT = "data/prefight_rd.csv"


def prefight_rd(eng: engine.Engine, fid: str, name: str, d: date) -> float:
    """Public-scale RD of a fighter going into a bout on date d, exactly the
    pre-fight snapshot engine.process takes (Step 0) before updating."""
    fo = eng.fighter(fid, name)             # same object process() will use
    n = (d - fo.last_fight).days / engine.YEAR if fo.last_fight else 0.0
    return engine.SCALE * engine.inflate(fo.phi, fo.sigma, max(n, 0.0))


def main() -> None:
    apply_params(json.load(open("best_params.json")))
    with open("data/fights_v2.csv", encoding="utf-8", newline="") as fh:
        fights = list(csv.DictReader(fh))
    fights.sort(key=lambda r: (r["event_date"], int(r["bout_order"])))

    eng = engine.Engine()
    rows = []
    for f in fights:
        d = date.fromisoformat(f["event_date"])
        rd_a = prefight_rd(eng, f["fighter_a_id"], f["fighter_a"], d)
        rd_b = prefight_rd(eng, f["fighter_b_id"], f["fighter_b"], d)
        rows.append((f["fight_id"], f"{rd_a:.2f}", f"{rd_b:.2f}"))
        eng.process({"event_date": f["event_date"],
                     "fighter_a_id": f["fighter_a_id"], "fighter_a": f["fighter_a"],
                     "fighter_b_id": f["fighter_b_id"], "fighter_b": f["fighter_b"],
                     "outcome": f["outcome"], "method": f["method"],
                     "a_missed_weight": f.get("a_mw", ""),
                     "b_missed_weight": f.get("b_mw", "")})

    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fight_id", "rd_a", "rd_b"])
        w.writerows(rows)
    both = sum(1 for _, a, b in rows if float(a) > 160 and float(b) > 160)
    print(f"wrote {OUT}: {len(rows)} fights  ({len(eng.fighters)} fighters replayed; "
          f"{both} bouts with both RD > 160)")


if __name__ == "__main__":
    main()
