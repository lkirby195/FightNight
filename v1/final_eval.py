import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(os.path.join(_ROOT, "data"))  # all data reads/writes live in data/
"""Final: tuned params + missed-weight flags -> full replay, leaderboard,
market benchmark, ratings export."""
import csv, math, json
from datetime import date
import engine as E
from engine import Engine, R0

best = json.load(open("best_params.json"))
E.TAU = best["tau"]
E.SIGMA0 = best["sigma0"]
E.S_TABLE = {
    "KO/TKO": (1.00, 0.00), "SUB": (1.00, 0.00),
    "U-DEC": (best["s_udec"], 1 - best["s_udec"]),
    "M-DEC": (best["s_udec"], 1 - best["s_udec"]),
    "S-DEC": (best["s_sdec"], 1 - best["s_sdec"]),
    "DQ":    (best["s_sdec"], 1 - best["s_sdec"]),
}
print("params:", best)

fights = list(csv.DictReader(open("fights.csv")))
fights.sort(key=lambda f: (f["event_date"], int(f["bout_order"])))
mw_n = sum(1 for f in fights if f["a_missed_weight"] == "1" or f["b_missed_weight"] == "1")
print(f"missed-weight fights in data: {mw_n}")

eng = Engine()
names = {}
for f in fights:
    names[f["fighter_a_id"]] = f["fighter_a"]
    names[f["fighter_b_id"]] = f["fighter_b"]
    eng.process(f)

# export predictions with names for market join
with open("preds.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["date", "p_a", "outcome", "a_name", "b_name", "a_fights", "b_fights"])
    for p in eng.predictions:
        w.writerow([p["date"].isoformat(), f"{p['p_a']:.6f}", p["outcome"],
                    names[p["a"]], names[p["b"]], p["a_fights"], p["b_fights"]])

# leaderboard
today = date(2026, 7, 30)
active = [f for f in eng.fighters.values()
          if f.last_fight and (today - f.last_fight).days <= 548
          and f.fights >= 3 and f.rd < 150]
active.sort(key=lambda x: x.rating, reverse=True)
print(f"\nTOP 20 ACTIVE (tuned, 18mo, 3+ fights, RD<150)")
for i, f in enumerate(active[:20], 1):
    print(f"{i:>3} {f.name:<26} {f.rating:>6.0f} ±{2*f.rd:>3.0f}  {f.wins}-{f.losses}")

with open("ratings.csv", "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["fighter_id", "name", "rating", "rd", "sigma", "ufc_fights",
                "ufc_wins", "ufc_losses", "last_fight"])
    for f in sorted(eng.fighters.values(), key=lambda x: x.rating, reverse=True):
        w.writerow([f.fid, f.name, round(f.rating, 1), round(f.rd, 1),
                    round(f.sigma, 5), f.fights, f.wins, f.losses, f.last_fight])
print("\nratings.csv + preds.csv written")
