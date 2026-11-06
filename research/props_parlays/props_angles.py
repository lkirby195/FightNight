"""Two prop-derived angles. Read-only; 2025+ never read.
A) Judging-noise test: in bouts likely to go the distance, is the moneyline favorite
   overpriced? Pre-registered comparison: dog ROI (top GTD tercile) - dog ROI (bottom
   tercile). Tercile cutpoints fixed on 2012-17 walk-forward predictions. Primary block
   2018-22, confirm 2023-24.
B) Method skew of our signals: do our winners win by decision more (or less) than the
   bout's walk-forward P(goes the distance) predicts?"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pathlib import Path
import socket


def _no_network(*a, **k):
    raise RuntimeError("props_parlays: network access is not allowed")


socket.socket.connect = _no_network
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parents[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else str(HERE)
D = pd.read_csv(f"{OUT}/bouts_gtd.csv")
D = D[(D.outcome == "A_WIN") & D.p_gtd_m2.notna() & D.a_open.notna() & D.b_open.notna()].copy()
# fighter_a is always the winner in fights_v2; a_* prices are the winner's prices.
D["fav_won"] = D.pa > 0.5
D["fav_p"] = np.where(D.fav_won, D.pa, 1 - D.pa)          # vig-free close of the favorite
# dog price at close / open (raw book price, what you could bet)
D["dog_close"] = np.where(D.fav_won, D.b_close, D.a_close)
D["dog_open"] = np.where(D.fav_won, D.b_open, D.a_open)
D["dog_win"] = ~D.fav_won
D = D[(D.pa != 0.5)]
for c in ("close", "open"):
    D[f"dog_pnl_{c}"] = np.where(D.dog_win, D[f"dog_{c}"] - 1, -1.0)

cut = D[D.year <= 2017].p_gtd_m2.quantile([1 / 3, 2 / 3]).values
D["terc"] = np.select([D.p_gtd_m2 <= cut[0], D.p_gtd_m2 <= cut[1]], ["low", "mid"], "high")
print("tercile cutpoints (P GTD, fixed on 2012-17):", cut.round(3))

rng = np.random.default_rng(7)


def table(df):
    g = df.groupby("terc").agg(n=("gtd", "size"), gtd_rate=("gtd", "mean"),
                               fav_implied=("fav_p", "mean"), fav_won=("fav_won", "mean"),
                               dog_roi_close=("dog_pnl_close", "mean"),
                               dog_roi_open=("dog_pnl_open", "mean"))
    g["fav_gap_pp"] = 100 * (g.fav_won - g.fav_implied)
    return g.loc[["low", "mid", "high"]]


def boot_diff(df, col, B=20000):
    hi = df[df.terc == "high"][col].values
    lo = df[df.terc == "low"][col].values
    obs = hi.mean() - lo.mean()
    bs = np.array([rng.choice(hi, len(hi)).mean() - rng.choice(lo, len(lo)).mean() for _ in range(B)])
    return obs, np.percentile(bs, [2.5, 97.5])


pd.set_option("display.width", 200)
for blk, (a, b) in {"2012-17 (cutpoint fit)": (2012, 2017), "2018-22 PRIMARY": (2018, 2022),
                    "2023-24 confirm": (2023, 2024)}.items():
    df = D[(D.year >= a) & (D.year <= b)]
    print(f"\n== {blk}  n={len(df)}")
    print(table(df).round(3).to_string())
    for col in ("dog_pnl_open", "dog_pnl_close"):
        obs, ci = boot_diff(df, col)
        print(f"  high-minus-low {col}: {obs:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")

# ---- B) method skew of frozen signals ----
L = pd.read_csv(f"{REPO}/data/system_ledger.csv")
L = L[(L.event_date < "2023-01-01") & L.result.isin(["WIN", "LOSS"])]
G = pd.read_csv(f"{OUT}/bouts_gtd.csv")[["fight_id", "p_gtd_m2", "gtd"]]
M = L.merge(G, on="fight_id", how="inner")
DEC = {"U-DEC", "S-DEC", "M-DEC"}
M["dec"] = M.method.isin(DEC)
print("\n== B) method skew, frozen signals 2012-22 (all signals, placed or not)")
for name, df in [("all", M)] + list(M.groupby("rule")):
    w = df[df.result == "WIN"]
    l = df[df.result == "LOSS"]
    print(f"  {name:4s} wins n={len(w):3d}: by decision {w.dec.mean():.3f} vs expected {w.p_gtd_m2.mean():.3f} | "
          f"losses n={len(l):3d}: by decision {l.dec.mean():.3f} vs expected {l.p_gtd_m2.mean():.3f} | "
          f"all bouts GTD {df.dec.mean():.3f} vs {df.p_gtd_m2.mean():.3f}")
# win-rate split by how the fight ended: where does the edge live?
M["fin"] = ~M.dec
M["d"] = M.open_line.apply(lambda x: 1 + x / 100 if x > 0 else 1 + 100 / abs(x))
M["pnl1"] = np.where(M.result == "WIN", M.d - 1, -1.0)
print("\n  1u P&L contribution by how the bout ended (2012-22 signals):")
print(M.groupby("dec").agg(n=("pnl1", "size"), win=("result", lambda s: (s == "WIN").mean()),
                           pnl_u=("pnl1", "sum")).rename(index={True: "decision", False: "finish"}).round(3).to_string())
# would the signal side by-decision be a better ticket? need prop prices; show the shares only

# ---- price-controlled check (the tercile table above is confounded with price) ----
D["lp"] = np.log(D.fav_p / (1 - D.fav_p))
print("\n== A') price-controlled: logit(fav_won) ~ logit(close price) + P(GTD)")
for blk, (a, b) in {"2012-17": (2012, 2017), "2018-22": (2018, 2022), "2023-24": (2023, 2024)}.items():
    d = D[(D.year >= a) & (D.year <= b)]
    m = sm.Logit(d.fav_won.astype(int), sm.add_constant(d[["lp", "p_gtd_m2"]])).fit(disp=0)
    print(f"  {blk}: n={len(d)}  coef P(GTD) {m.params.p_gtd_m2:+.3f} (se {m.bse.p_gtd_m2:.3f}, p {m.pvalues.p_gtd_m2:.2f})")
