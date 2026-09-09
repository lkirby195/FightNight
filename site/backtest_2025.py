"""One-shot 2025 backtest for the public site (sealed; approved 2026-09-09).

Writes, once:
  data/model_only_preds_2025_backtest.csv   walk-forward Stage 2 probabilities
                                            for 2025 (fit on fights < 2025-01-01,
                                            exactly clv_eval.gen_model_preds)
  data/system_ledger_2025_backtest.csv      betting_system rule (frozen
                                            2026-09-08) + placement policy v1
                                            (frozen 2026-09-09) over those
                                            probabilities, settled at the
                                            opening line (live=0)

Both files are a sealed read for the site's 2025 tab: display only, never an
input to model selection, never regenerated. This script refuses to run if
either file exists. It does not touch data/model_only_preds.csv or
data/system_ledger.csv.

Usage: python site/backtest_2025.py        # once, from the repo root
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import betting_system as bs
import features
import stage2
from clv_eval import PASSC

YEAR = 2025
PREDS = Path("data/model_only_preds_2025_backtest.csv")
LEDGER = Path("data/system_ledger_2025_backtest.csv")


def preds_2025() -> pd.DataFrame:
    df, X, y = stage2.load()
    Xc = X.drop(columns=PASSC)
    fl = df.fight_id.map(features.flip)
    trm = (df.date < f"{YEAR}-01-01").values
    evm = ((df.date >= f"{YEAR}-01-01") & (df.date < f"{YEAR + 1}-01-01")).values
    m = make_pipeline(StandardScaler(with_mean=False),
                      LogisticRegression(C=0.3, max_iter=3000, fit_intercept=False))
    m.fit(np.vstack([Xc[trm].values, -Xc[trm].values]),
          np.concatenate([y[trm], 1 - y[trm]]))
    pm = m.predict_proba(Xc[evm].values)[:, 1]
    return pd.DataFrame({"fight_id": df.fight_id[evm].values, "year": YEAR,
                         "p_model_a": np.where(fl[evm].values, 1 - pm, pm),
                         "y_a": np.where(fl[evm].values, 1 - y[evm], y[evm])})


def load_with(M: pd.DataFrame) -> pd.DataFrame:
    """betting_system.load() with M in place of data/model_only_preds.csv."""
    J = pd.read_csv("data/bfo_joined.csv")
    fv = pd.read_csv("data/fights_v2.csv")[["fight_id", "event_date", "event_name",
                                            "bout_order", "fighter_a", "fighter_b",
                                            "method", "round"]].rename(columns={"round": "rnd"})
    rd = pd.read_csv(bs.PREFIGHT_RD)
    D = (M.merge(J, on="fight_id").merge(fv, on="fight_id").merge(rd, on="fight_id", how="left"))
    D = D.dropna(subset=["y_a", "a_open", "b_open"])
    imp = 1 / D.a_open + 1 / D.b_open
    D = D[(imp >= 1.0) & (imp <= 1.12)].copy()
    o1, o2, p = D.a_open.values, D.b_open.values, D.p_model_a.values
    mkt1 = (1 / o1) >= (1 / o2)
    mod1 = p >= .5
    D["mkt1"], D["mod1"] = mkt1, mod1
    D["flip"] = mkt1 != mod1
    D["conf"] = np.where(mod1, p, 1 - p)
    D["gap_pts"] = (np.where(mkt1, bs.am(o1), bs.am(o2))
                    - np.where(mkt1, bs.amp(p), bs.amp(1 - p)))
    return D.sort_values("event_date")


def main() -> None:
    for p in (PREDS, LEDGER):
        if p.exists():
            raise SystemExit(f"{p} exists: the 2025 backtest is sealed, not regenerated")
    M = preds_2025()
    M.to_csv(PREDS, index=False)
    print(f"wrote {PREDS}: {len(M)} rows")
    B = bs.run(load_with(M))
    assert (B.live == 0).all(), "2025 rows must be backtest (live=0)"
    bs.report(B, f"BACKTEST ({YEAR})")
    bs.write_ledger(B, path=str(LEDGER))


if __name__ == "__main__":
    main()
