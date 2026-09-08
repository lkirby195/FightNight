"""CLV evaluation: pure model-only picks vs BFO opening lines, graded at close.

Requires:
  data/bfo_lines.csv        (scrape_bfo.py)
  data/fights_v2.csv        (scrape_v2.py)
  data/model_only_preds.csv (this script regenerates if absent)

CRITICAL: predictions must contain NO market input. Stacked probabilities
incorporate the closing line and make any CLV measurement circular.

Outputs: data/bfo_joined.csv, data/clv_eval_full.csv + console report.
"""
from __future__ import annotations

import re
import unicodedata

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
YEARS = list(range(2012, 2025)) + [2026]        # 2025 sealed


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def lastn(s: str) -> str:
    p = norm(s).split()
    return p[-1] if p else ""


def gen_model_preds():
    df, X, y = stage2.load()
    Xc = X.drop(columns=PASSC)
    fl = df.fight_id.map(features.flip)
    out = []
    for Y in YEARS:
        trm = (df.date < f"{Y}-01-01").values
        evm = ((df.date >= f"{Y}-01-01") & (df.date < f"{Y+1}-01-01")).values
        m = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(C=0.3, max_iter=3000, fit_intercept=False))
        m.fit(np.vstack([Xc[trm].values, -Xc[trm].values]),
              np.concatenate([y[trm], 1 - y[trm]]))
        pm = m.predict_proba(Xc[evm].values)[:, 1]
        out.append(pd.DataFrame({
            "fight_id": df.fight_id[evm].values, "year": Y,
            "p_model_a": np.where(fl[evm].values, 1 - pm, pm),
            "y_a": np.where(fl[evm].values, 1 - y[evm], y[evm])}))
    M = pd.concat(out)
    M.to_csv("data/model_only_preds.csv", index=False)
    return M


def join_bfo():
    b = pd.read_csv("data/bfo_lines.csv", parse_dates=["event_date"]).drop_duplicates("mu")
    f = pd.read_csv("data/fights_v2.csv", parse_dates=["event_date"])
    f = f[f.event_date >= "2010-01-01"].copy()
    for frame, c1, c2 in ((b, "fighter1", "fighter2"), (f, "fighter_a", "fighter_b")):
        frame["pair"] = frame.apply(lambda r: tuple(sorted([norm(r[c1]), norm(r[c2])])), axis=1)
        frame["lpair"] = frame.apply(lambda r: tuple(sorted([lastn(r[c1]), lastn(r[c2])])), axis=1)
    rows = []
    for _, r in b.iterrows():
        c = f[(f.pair == r.pair) & (abs((f.event_date - r.event_date).dt.days) <= 3)]
        if len(c) == 0:
            c = f[(f.lpair == r.lpair) & (abs((f.event_date - r.event_date).dt.days) <= 3)]
        if len(c) == 1:
            c = c.iloc[0]
            f1a = (norm(r.fighter1) == norm(c.fighter_a)) or (lastn(r.fighter1) == lastn(c.fighter_a))
            rows.append(dict(
                fight_id=c.fight_id,
                a_open=r.f1_open if f1a else r.f2_open,
                a_close=r.f1_close if f1a else r.f2_close,
                b_open=r.f2_open if f1a else r.f1_open,
                b_close=r.f2_close if f1a else r.f1_close))
    J = pd.DataFrame(rows).drop_duplicates("fight_id")
    J.to_csv("data/bfo_joined.csv", index=False)
    return J


def main():
    import os
    M = (pd.read_csv("data/model_only_preds.csv")
         if os.path.exists("data/model_only_preds.csv") else gen_model_preds())
    J = join_bfo()
    D = M.merge(J, on="fight_id")
    inv = lambda x: 1.0 / x
    D["qa_o"] = inv(D.a_open) / (inv(D.a_open) + inv(D.b_open))
    D["qa_c"] = inv(D.a_close) / (inv(D.a_close) + inv(D.b_close))
    D["pick_a"] = D.p_model_a > D.qa_o
    D["clv"] = np.where(D.pick_a, D.qa_c - D.qa_o, (1 - D.qa_c) - (1 - D.qa_o))
    D["dis"] = D.p_model_a - D.qa_o
    D["mv"] = D.qa_c - D.qa_o
    T = lambda x: np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x)))
    print(f"{'era':12s}{'n':>6}{'CLV':>8}{'t':>7}{'slope':>7}")
    for lo, hi in [(2012, 2016), (2017, 2021), (2022, 2026)]:
        s = D[(D.year >= lo) & (D.year <= hi)]
        print(f"{lo}-{hi:10d}{len(s):>6}{s.clv.mean()*100:>+8.2f}{T(s.clv):>7.2f}"
              f"{np.polyfit(s.dis, s.mv, 1)[0]:>7.2f}")
    print(f"{'POOLED':12s}{len(D):>6}{D.clv.mean()*100:>+8.2f}{T(D.clv):>7.2f}"
          f"{np.polyfit(D.dis, D.mv, 1)[0]:>7.2f}")
    D.to_csv("data/clv_eval_full.csv", index=False)


if __name__ == "__main__":
    main()
