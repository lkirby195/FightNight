"""Walk-forward evaluation: Stage 2 vs recalibrated closing market, 2012-2024.

For each year Y: Stage 2 refit on fights < Y; stacker fit only on
OUT-of-sample predictions from prior years; evaluated on year Y.
2025+ is never touched (sealed holdout).

Outputs: data/walkforward.csv (per-fight loss differences) and a console report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features
import stage2

PASSC = ["d_ss_acc", "d_ss_def", "d_td_acc", "d_td_def", "d_ctrl15",
         "d_pace15", "d_head_share", "d_leg_share", "d_ground_share",
         "d_ctrled15"]
C_REG = 0.3
LG = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))


def fit_stage2(Xc, y, mask):
    m = make_pipeline(StandardScaler(with_mean=False),
                      LogisticRegression(C=C_REG, max_iter=3000, fit_intercept=False))
    m.fit(np.vstack([Xc[mask].values, -Xc[mask].values]),
          np.concatenate([y[mask], 1 - y[mask]]))
    return m


def fit_stack(P, cols):
    Z = P[cols].values
    S = LogisticRegression(fit_intercept=False, C=1e6)
    S.fit(np.vstack([Z, -Z]), np.concatenate([P.y, 1 - P.y]))
    return S


def main():
    df, X, y = stage2.load()
    Xc = X.drop(columns=PASSC)
    mkt = pd.read_csv("data/market.csv").set_index("fight_id").p_market_a
    d = df.copy()
    d["p_mkt"] = d.fight_id.map(mkt)
    fl = d.fight_id.map(features.flip)
    d.loc[fl, "p_mkt"] = 1 - d.loc[fl, "p_mkt"]
    d["y"] = y

    pool, per_year = [], []
    for Y in range(2010, 2025):
        trm = (d.date < f"{Y}-01-01").values
        evm = ((d.date >= f"{Y}-01-01") & (d.date < f"{Y+1}-01-01")
               & d.p_mkt.notna()).values
        if evm.sum() == 0:
            continue
        m = fit_stage2(Xc, y, trm)
        rec = pd.DataFrame({"year": Y,
                            "lm": LG(d.p_mkt[evm].values),
                            "lp": LG(m.predict_proba(Xc[evm].values)[:, 1]),
                            "y": d.y[evm].values})
        if Y >= 2012 and sum(len(p) for p in pool) >= 300:
            P = pd.concat(pool)
            Sa, Sb = fit_stack(P, ["lm"]), fit_stack(P, ["lm", "lp"])
            pa = Sa.predict_proba(rec[["lm"]].values)[:, 1]
            pb = Sb.predict_proba(rec[["lm", "lp"]].values)[:, 1]
            q = lambda p: np.clip(p, 1e-9, 1 - 1e-9)
            la = -(rec.y * np.log(q(pa)) + (1 - rec.y) * np.log(1 - q(pa)))
            lb = -(rec.y * np.log(q(pb)) + (1 - rec.y) * np.log(1 - q(pb)))
            per_year.append(pd.DataFrame({"year": Y, "diff": la.values - lb.values,
                                          "w_model": Sb.coef_[0][1]}))
        pool.append(rec)

    r = pd.concat(per_year)
    r.to_csv("data/walkforward.csv", index=False)
    print(r.groupby("year").agg(n=("diff", "size"), gain=("diff", "mean"),
                                w_model=("w_model", "first")).round(4).to_string())
    diff = r["diff"].values
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"\nPOOLED: n={len(diff)}  gain {diff.mean():+.5f}  t={diff.mean()/se:.2f}")


if __name__ == "__main__":
    main()
