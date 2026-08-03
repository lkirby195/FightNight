import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(os.path.join(_ROOT, "data"))  # all data reads/writes live in data/
"""Join historical closing odds to our fights; benchmark market vs model.
Tier 1: exact normalized-name match (date +/- 1 day).
Tier 2: fuzzy per-date match (token overlap, handles name order/changes/diminutives).
"""
import csv, math, unicodedata
from datetime import date, timedelta
from unicodedata import combining

def norm(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not combining(c)).lower().replace("-", " ").replace(".", "").strip()

def toks(s):
    return set(t for t in s.split() if len(t) >= 3)

def sim(a, b):
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    # prefix credit for diminutives (jim/jimmy, luci/lucie)
    if inter == 0:
        for x in ta:
            for y in tb:
                if x.startswith(y[:4]) or y.startswith(x[:4]):
                    inter = 0.6
    return inter / min(len(ta), len(tb))

def implied(a):
    a = float(a)
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)

def load_market():
    exact, per_date = {}, {}
    for r in csv.DictReader(open("odds-raw.csv")):
        if not (r.get("R_odds") and r.get("B_odds") and r.get("date")):
            continue
        try:
            pr, pb = implied(r["R_odds"]), implied(r["B_odds"])
        except ValueError:
            continue
        rn, bn = norm(r["R_fighter"]), norm(r["B_fighter"])
        rec = (pr / (pr + pb), rn, bn)
        exact[(r["date"], frozenset([rn, bn]))] = rec
        per_date.setdefault(r["date"], []).append(rec)
    return exact, per_date

def match(exact, per_date, d_iso, na, nb):
    d0 = date.fromisoformat(d_iso)
    days = [(d0 + timedelta(days=dd)).isoformat() for dd in (0, -1, 1)]
    for d in days:
        k = (d, frozenset([na, nb]))
        if k in exact:
            p_red, rn, bn = exact[k]
            return p_red if rn == na else 1 - p_red
    # tier 2: fuzzy within the same dates
    best, best_score = None, 0.0
    for d in days:
        for p_red, rn, bn in per_date.get(d, []):
            s1 = min(sim(na, rn), sim(nb, bn))   # straight
            s2 = min(sim(na, bn), sim(nb, rn))   # crossed
            if max(s1, s2) > best_score:
                best_score = max(s1, s2)
                best = p_red if s1 >= s2 else 1 - p_red
    return best if best_score >= 0.5 else None

def evaluate():
    exact, per_date = load_market()
    ll_m = ll_g = 0.0
    acc_m = acc_g = n = unmatched = 0
    pairs = []
    for p in csv.DictReader(open("preds.csv")):
        if p["date"] < "2020-01-01" or p["outcome"] != "A_WIN":
            continue
        if int(p["a_fights"]) < 1 or int(p["b_fights"]) < 1:
            continue
        pm = match(exact, per_date, p["date"], norm(p["a_name"]), norm(p["b_name"]))
        if pm is None:
            unmatched += 1
            continue
        pg = float(p["p_a"])
        pairs.append((pm, pg))
        for prob, is_model in ((pm, False), (pg, True)):
            prob = min(max(prob, 1e-9), 1 - 1e-9)
            if is_model:
                ll_g += -math.log(prob); acc_g += prob > 0.5
            else:
                ll_m += -math.log(prob); acc_m += prob > 0.5
        n += 1
    print(f"matched {n} fights, {unmatched} unmatched (coverage gaps in odds dataset)")
    print(f"{'Closing line (vig-free)':<26} log-loss={ll_m/n:.5f} acc={acc_m/n:.3f}")
    print(f"{'Our model':<26} log-loss={ll_g/n:.5f} acc={acc_g/n:.3f}")
    def bll(w):
        return sum(-math.log(min(max(w*pm + (1-w)*pg, 1e-9), 1-1e-9)) for pm, pg in pairs) / len(pairs)
    best_w = min(((bll(w/20), w/20) for w in range(10, 21)))
    print(f"optimal blend: w_market={best_w[1]:.2f} log-loss={best_w[0]:.5f}")

if __name__ == "__main__":
    evaluate()
