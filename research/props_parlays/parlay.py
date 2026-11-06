"""Parlay vs straights on the frozen placed ledger. Read-only.
Windows: 2012-2022 (primary), 2023-2024 (confirm). 2025+ never read."""
import sys
import numpy as np
import pandas as pd

from pathlib import Path
import socket


def _no_network(*a, **k):
    raise RuntimeError("props_parlays: network access is not allowed")


socket.socket.connect = _no_network
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parents[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else str(HERE)
L = pd.read_csv(f"{REPO}/data/system_ledger.csv")
L = L[(L.event_date < "2025-01-01") & (L.result != "VOID")].copy()


def dec(line):
    return 1 + line / 100 if line > 0 else 1 + 100 / abs(line)


L["d"] = L.open_line.apply(dec)
L["win"] = (L.result == "WIN").astype(int)
# settle price check: 1u flat pnl on wins must equal d-1
w = L[L.win == 1]
assert np.allclose(w.d - 1, w.pnl_at_open, atol=1e-3), "price mismatch"

P = L[L.placed == 1].sort_values(["event_date", "fight_id"]).reset_index(drop=True)
P["blk"] = np.where(P.event_date < "2023-01-01", "2012-22", "2023-24")


def straights(df, stake="flat"):
    s = np.ones(len(df)) if stake == "flat" else df.stake_u.values
    pnl = np.where(df.win == 1, s * (df.d - 1), -s)
    return s.sum(), pnl.sum(), pnl


def parlay(groups, stake_fn):
    rows = []
    for g in groups:
        s = stake_fn(g)
        hit = g.win.all()
        price = g.d.prod()
        rows.append((len(g), s, s * (price - 1) if hit else -s, hit, price))
    return pd.DataFrame(rows, columns=["legs", "stake", "pnl", "hit", "price"])


def maxdd(pnl):
    c = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0], c]))[1:]
    return (peak - c).max()


out = []
for blk, df in P.groupby("blk"):
    st, pn, arr = straights(df)
    out.append(dict(block=blk, structure="straights 1u", bets=len(df), staked=st, pnl=pn,
                    roi=pn / st, hit=df.win.mean(), maxdd=maxdd(arr)))
    for k in (2, 3):
        idx = np.arange(len(df)) // k
        groups = [g for _, g in df.groupby(idx) if len(g) == k]
        # equal money: a k-leg parlay stakes k units, same as k straights
        R = parlay(groups, lambda g: float(len(g)))
        out.append(dict(block=blk, structure=f"{k}-leg rolling, {k}u", bets=len(R),
                        staked=R.stake.sum(), pnl=R.pnl.sum(), roi=R.pnl.sum() / R.stake.sum(),
                        hit=R.hit.mean(), maxdd=maxdd(R.pnl.values)))
    # same-card parlays: only cards with 2+ placed legs
    sc = [g for _, g in df.groupby("event_date") if len(g) >= 2]
    if sc:
        R = parlay(sc, lambda g: float(len(g)))
        cards = pd.concat(sc)
        st2, pn2, _ = straights(cards)
        out.append(dict(block=blk, structure="same-card straights", bets=len(cards), staked=st2,
                        pnl=pn2, roi=pn2 / st2, hit=cards.win.mean(), maxdd=np.nan))
        out.append(dict(block=blk, structure="same-card parlay", bets=len(R), staked=R.stake.sum(),
                        pnl=R.pnl.sum(), roi=R.pnl.sum() / R.stake.sum(), hit=R.hit.mean(),
                        maxdd=np.nan))
emp = pd.DataFrame(out)
pd.set_option("display.width", 200)
print(emp.round(3).to_string(index=False))
print("\nplaced legs per card (2012-24):", P.groupby("event_date").size().value_counts().sort_index().to_dict())
print("median leg decimal price:", P.d.median().round(3), " mean:", P.d.mean().round(3))

# ---- Monte Carlo seasons: legs drawn from the empirical price mix ----
rng = np.random.default_rng(20261105)
prices = P.d.values
N_SIM, LEGS = 100_000, 12          # 12 legs ~ one season (169 placed / 13 yrs = 13)
rows = []
for r in (0.20, 0.10, 0.05, 0.0, -0.03):
    d = rng.choice(prices, size=(N_SIM, LEGS))
    p = np.clip((1 + r) / d, 0, 0.99)
    win = rng.random((N_SIM, LEGS)) < p
    s_pnl = np.where(win, d - 1, -1).sum(1)          # 12u staked
    res = {"true leg ROI": r, "straights mean": s_pnl.mean(), "straights P(loss)": (s_pnl < 0).mean(),
           "straights p5": np.percentile(s_pnl, 5)}
    for k in (2, 3):
        W = win.reshape(N_SIM, LEGS // k, k).all(2)
        D = d.reshape(N_SIM, LEGS // k, k).prod(2)
        pp = np.where(W, k * (D - 1), -k).sum(1)     # same 12u staked
        res[f"{k}-leg mean"] = pp.mean()
        res[f"{k}-leg P(loss)"] = (pp < 0).mean()
        res[f"{k}-leg p5"] = np.percentile(pp, 5)
    rows.append(res)
mc = pd.DataFrame(rows)
print("\nMonte Carlo, 12u staked per season (12 straights vs 6 two-leg vs 4 three-leg):")
print(mc.round(2).to_string(index=False))

# ---- analytic: parlay ROI vs true leg edge, with single-book haircut ----
print("\nAnalytic n-leg ROI = ((1+r)(1-h))^n - 1   (h = price haircut from losing line-shopping)")
tab = []
for r in (0.20, 0.10, 0.05, 0.02, 0.0):
    for h in (0.0, 0.02, 0.04):
        x = (1 + r) * (1 - h)
        tab.append(dict(r=r, h=h, straight=x - 1, two=x**2 - 1, three=x**3 - 1))
print(pd.DataFrame(tab).round(3).to_string(index=False))

emp.to_csv(OUT + "/parlay_empirical.csv", index=False, lineterminator="\n")
mc.to_csv(OUT + "/parlay_montecarlo.csv", index=False, lineterminator="\n")
