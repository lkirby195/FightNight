import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(os.path.join(_ROOT, "data"))  # all data reads/writes live in data/
"""Grid search over tunable parameters per spec section 9.
Train: log-loss on 2014-01-01..2019-12-31 (both fighters with UFC history).
Test (evaluated ONCE at the end): 2020-01-01+.
"""
import csv, math, itertools, sys
from datetime import date
import engine as E
from engine import Engine

FIGHTS = list(csv.DictReader(open("fights.csv")))
FIGHTS.sort(key=lambda f: (f["event_date"], int(f["bout_order"])))
TRAIN = (date(2014, 1, 1), date(2020, 1, 1))
TEST = (date(2020, 1, 1), date(2027, 1, 1))

def run(params, window):
    # inject params into engine module
    E.TAU = params["tau"]
    E.SIGMA0 = params["sigma0"]
    E.S_TABLE = {
        "KO/TKO": (1.00, 0.00), "SUB": (1.00, 0.00),
        "U-DEC": (params["s_udec"], 1 - params["s_udec"]),
        "M-DEC": (params["s_udec"], 1 - params["s_udec"]),
        "S-DEC": (params["s_sdec"], 1 - params["s_sdec"]),
        "DQ":    (params["s_sdec"], 1 - params["s_sdec"]),
    }
    eng = Engine()
    for f in FIGHTS:
        eng.process(f)
    ll, n, correct = 0.0, 0, 0
    lo, hi = window
    for p in eng.predictions:
        if not (lo <= p["date"] < hi) or p["outcome"] != "A_WIN":
            continue
        if p["a_fights"] < 1 or p["b_fights"] < 1:
            continue
        prob = min(max(p["p_a"], 1e-9), 1 - 1e-9)
        ll += -math.log(prob)
        correct += prob > 0.5
        n += 1
    return ll / n, correct / n, n

if __name__ == "__main__":
    grid = {
        "tau":    [0.2, 0.4, 0.6, 0.9],
        "sigma0": [0.04, 0.06, 0.09],
        "s_udec": [0.85, 0.90, 0.95, 1.00],
        "s_sdec": [0.65, 0.75, 0.85],
    }
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    print(f"{len(combos)} combos", flush=True)
    results = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        ll, acc, n = run(params, TRAIN)
        results.append((ll, acc, params))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(combos)} best so far: {min(r[0] for r in results):.5f}", flush=True)
    results.sort(key=lambda x: x[0])
    print("\nTOP 5 (train log-loss):")
    for ll, acc, p in results[:5]:
        print(f"  {ll:.5f} acc={acc:.3f} {p}")
    best = results[0][2]
    # ONE test-set evaluation for tuned params + baseline priors
    print("\n== TEST SET (2020+) ==", flush=True)
    for label, p in [("tuned", best),
                     ("priors", {"tau": 0.5, "sigma0": 0.06, "s_udec": 0.90, "s_sdec": 0.75})]:
        ll, acc, n = run(p, TEST)
        print(f"{label:<8} log-loss={ll:.5f} acc={acc:.3f} n={n} {p}")
    import json
    json.dump(best, open("best_params.json", "w"))
