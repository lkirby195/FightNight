"""NOT pre-registered. Two follow-ups to the totals_edge confirm result, run after
the gate failed. Reads per_fight_dev_confirm.csv only (2015-2024); 2025+ never read.

A) Line movement with controls. The pre-registered slope of (q_close - q_open)
   on (m - q_open) is confounded: if the open is noisy and drifts back toward
   the close, the slope is positive even for a model with no information. Refit
   with q_open as a covariate, and with a null "model" (the training-period base
   rate) in place of M2.
B) Market calibration. Recalibrating the market alone (fit on dev) beat the
   stack on confirm. Show the recalibration, and mean market vs realized by year.

    python research/totals_edge/posthoc.py [OUT]
"""
import socket
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _no_network(*a, **k):
    raise RuntimeError("posthoc: network access is not allowed")


socket.socket.connect = _no_network
sys.dont_write_bytecode = True
# sklearn 1.8+ warns that penalty=None is deprecated; behaviour is unchanged
warnings.filterwarnings("ignore", category=FutureWarning)
HERE = Path(__file__).resolve().parent
OUT = sys.argv[1] if len(sys.argv) > 1 else str(HERE)
P = pd.read_csv(f"{OUT}/per_fight_dev_confirm.csv")
assert P.year.max() <= 2024
L = lambda p: np.log(p / (1 - p))  # noqa: E731


def ols(y, X):
    X = np.c_[np.ones(len(y)), X]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return beta[1:], beta[1:] / se[1:]


rows = []
for (t, blk), g in P.assign(blk=np.where(P.year <= 2022, "dev 2015-22", "confirm 2023-24")).groupby(["target", "blk"], sort=False):
    g = g.dropna(subset=["q_open"])
    dq = (g.q - g.q_open).values
    base = P[(P.target == t) & (P.year < (2015 if blk.startswith("dev") else 2023))].y.mean()
    base = base if not np.isnan(base) else g.y.mean()
    (b1,), (t1,) = ols(dq, (g.m - g.q_open).values[:, None])
    (b0,), (t0,) = ols(dq, (base - g.q_open).values[:, None])
    (bm, bq), (tm, tq) = ols(dq, np.c_[g.m.values, g.q_open.values])
    rows.append(dict(target=t, block=blk, n=len(g), slope_prereg=b1, t_prereg=t1,
                     slope_null_model=b0, t_null=t0, coef_m_given_open=bm, t_m_given_open=tm,
                     coef_open=bq, t_open=tq))
A = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("A) line movement (q_close - q_open), OLS")
print(A.round(4).to_string(index=False))

print("\nB) market calibration: logit(P) = a + b logit(q), fit on dev 2015-22")
from sklearn.linear_model import LogisticRegression  # noqa: E402
for t, g in P.groupby("target", sort=False):
    d = g[g.year <= 2022]
    lr = LogisticRegression(penalty=None, max_iter=5000).fit(L(d.q.values)[:, None], d.y)
    print(f"  {t:20s} a {lr.intercept_[0]:+.3f}  b {lr.coef_[0][0]:.3f}")
    by = g.groupby("year").agg(n=("y", "size"), market=("q", "mean"), realized=("y", "mean"), model=("m", "mean"))
    by["mkt_minus_real_pp"] = 100 * (by.market - by.realized)
    print(by.round(3).to_string())
A.to_csv(f"{OUT}/posthoc_line_movement.csv", index=False, lineterminator="\n")

# C) The calibration table says the long-fight side (Over / goes the distance) is
# underpriced most years. What does betting it blind return at the raw BFO mean
# price, with the vig? 1u flat on every row, side 1 = Over / Yes.
print("\nC) blind long-fight side at raw BFO mean prices (1u flat), and the short side for contrast")
out = []
for t, g in P.groupby("target", sort=False):
    for price, (c1, c2) in {"close": ("p1_close_clean", "p2_close_clean"), "open": ("p1_open", "p2_open")}.items():
        h = g.dropna(subset=["q_open"]) if price == "open" else g
        for blk, (a, b) in {"2015-18": (2015, 2018), "2019-22": (2019, 2022), "2023-24": (2023, 2024)}.items():
            x = h[(h.year >= a) & (h.year <= b)]
            long_pnl = np.where(x.y == 1, x[c1] - 1, -1.0)
            short_pnl = np.where(x.y == 0, x[c2] - 1, -1.0)
            out.append(dict(target=t, price=price, years=blk, n=len(x),
                            long_roi=long_pnl.mean(), long_t=long_pnl.mean() / (long_pnl.std(ddof=1) / np.sqrt(len(x))),
                            short_roi=short_pnl.mean(), years_long_positive=None))
        yr = h.assign(pnl=np.where(h.y == 1, h[c1] - 1, -1.0)).groupby("year").pnl.mean()
        out.append(dict(target=t, price=price, years="2015-24 all", n=len(h),
                        long_roi=np.where(h.y == 1, h[c1] - 1, -1.0).mean(), long_t=np.nan,
                        short_roi=np.where(h.y == 0, h[c2] - 1, -1.0).mean(),
                        years_long_positive=f"{int((yr > 0).sum())}/{len(yr)}"))
C = pd.DataFrame(out)
print(C.round(4).to_string(index=False))
C.to_csv(f"{OUT}/posthoc_blind_long.csv", index=False, lineterminator="\n")
