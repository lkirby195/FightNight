import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(os.path.join(_ROOT, "data"))  # all data reads/writes live in data/
"""Full-history replay + leaderboard + backtest."""
import csv, math, json, os
from datetime import date
import engine as E
from engine import Engine, R0

# Load tuned parameters if present (falls back to spec priors)
if os.path.exists("best_params.json"):
    _p = json.load(open("best_params.json"))
    E.TAU = _p["tau"]
    E.SIGMA0 = _p["sigma0"]
    E.S_TABLE = {
        "KO/TKO": (1.00, 0.00), "SUB": (1.00, 0.00),
        "U-DEC": (_p["s_udec"], 1 - _p["s_udec"]),
        "M-DEC": (_p["s_udec"], 1 - _p["s_udec"]),
        "S-DEC": (_p["s_sdec"], 1 - _p["s_sdec"]),
        "DQ":    (_p["s_sdec"], 1 - _p["s_sdec"]),
    }
    print("loaded tuned params:", _p)

def main():
    fights = list(csv.DictReader(open("fights.csv")))
    fights.sort(key=lambda f: (f["event_date"], int(f["bout_order"])))
    eng = Engine()
    for f in fights:
        eng.process(f)

    # ---- Leaderboard: active (fought in last 18 months), 3+ UFC fights ----
    today = date(2026, 7, 30)
    active = [f for f in eng.fighters.values()
              if f.last_fight and (today - f.last_fight).days <= 548 and f.fights >= 3 and f.rd < 150]
    active.sort(key=lambda f: f.rating, reverse=True)
    print(f"\n{'='*74}\nTOP 25 ACTIVE (active 18mo, 3+ UFC fights, RD < 150)\n{'='*74}")
    print(f"{'#':>3} {'Fighter':<26} {'Rating':>7} {'RD':>5} {'Rec':>7} {'Last fight':>11}")
    for i, f in enumerate(active[:25], 1):
        rec = f"{f.wins}-{f.losses}"
        print(f"{i:>3} {f.name:<26} {f.rating:>7.0f} {f.rd:>5.0f} {rec:>7} {str(f.last_fight):>11}")

    # ---- Full ratings export ----
    with open("ratings.csv", "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["fighter_id", "name", "rating", "rd", "sigma", "ufc_fights",
                    "ufc_wins", "ufc_losses", "last_fight"])
        for f in sorted(eng.fighters.values(), key=lambda x: x.rating, reverse=True):
            w.writerow([f.fid, f.name, round(f.rating, 1), round(f.rd, 1),
                        round(f.sigma, 5), f.fights, f.wins, f.losses, f.last_fight])

    # ---- Backtest: 2020+ decided fights, both fighters with >=1 prior UFC fight ----
    def logloss_glicko():
        ll, n, correct = 0.0, 0, 0
        for p in eng.predictions:
            if p["date"] < date(2020, 1, 1) or p["outcome"] != "A_WIN":
                continue
            if p["a_fights"] < 1 or p["b_fights"] < 1:
                continue
            prob = min(max(p["p_a"], 1e-9), 1 - 1e-9)
            ll += -math.log(prob)
            correct += prob > 0.5
            n += 1
        return ll / n, correct / n, n

    # vanilla Elo K=32 baseline, same fights
    def elo_baseline():
        R = {}
        ll, n, correct = 0.0, 0, 0
        counts = {}
        for f in sorted(list(csv.DictReader(open("fights.csv"))),
                        key=lambda x: (x["event_date"], int(x["bout_order"]))):
            a, b = f["fighter_a_id"], f["fighter_b_id"]
            ra, rb = R.get(a, R0), R.get(b, R0)
            e_a = 1 / (1 + 10 ** ((rb - ra) / 400))
            d = date.fromisoformat(f["event_date"])
            if f["outcome"] == "A_WIN":
                if d >= date(2020, 1, 1) and counts.get(a, 0) >= 1 and counts.get(b, 0) >= 1:
                    prob = min(max(e_a, 1e-9), 1 - 1e-9)
                    ll += -math.log(prob)
                    correct += prob > 0.5
                    n += 1
                s_a, s_b = 1.0, 0.0
            elif f["outcome"] == "DRAW":
                s_a = s_b = 0.5
            else:
                continue
            R[a] = ra + 32 * (s_a - e_a)
            R[b] = rb + 32 * (s_b - (1 - e_a))
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
        return ll / n, correct / n, n

    gl, ga, gn = logloss_glicko()
    el, ea, en = elo_baseline()
    print(f"\n{'='*74}\nBACKTEST - 2020-present decided fights, both fighters with UFC history\n{'='*74}")
    print(f"{'Model':<28} {'Log-loss':>9} {'Accuracy':>9} {'N':>6}")
    print(f"{'Coin flip':<28} {0.6931:>9.4f} {0.500:>9.1%} {gn:>6}")
    print(f"{'Vanilla Elo (K=32)':<28} {el:>9.4f} {ea:>9.1%} {en:>6}")
    print(f"{'Glicko-2 adapted (ours)':<28} {gl:>9.4f} {ga:>9.1%} {gn:>6}")
    print("\n(Parameters at spec priors - zero tuning. Market closing line ~0.62 log-loss.)")

if __name__ == "__main__":
    main()
