"""Can we predict fight duration better than weight class + the moneyline?
Read-only. Walk-forward annual refit; train < Y, predict Y.
Evaluation blocks: 2015-2022 and 2023-2024. 2025+ never read."""
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, brier_score_loss

from pathlib import Path
import socket


def _no_network(*a, **k):
    raise RuntimeError("props_parlays: network access is not allowed")


socket.socket.connect = _no_network
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parents[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else str(HERE)
F = pd.read_csv(f"{REPO}/data/fights_v2.csv")
F = F[F.event_date < "2025-01-01"].sort_values(["event_date", "bout_order"]).reset_index(drop=True)
F["year"] = F.event_date.str[:4].astype(int)
DEC = {"U-DEC", "S-DEC", "M-DEC"}
FIN = {"KO/TKO", "SUB"}
F["valid"] = F.method.isin(DEC | FIN)
F["gtd"] = F.method.isin(DEC).astype(int)
mm, ss = F.time.fillna("0:00").str.split(":", expand=True).astype(int).T.values
F["elapsed"] = (F["round"] - 1) * 5 + mm + ss / 60          # minutes
F["mx"] = F.groupby("event_id").bout_order.transform("max")
F["sched5"] = ((F.title_bout == 1) | ((F.bout_order == F.mx) & (F.year >= 2012))).astype(int)
F["women"] = F.weight_class.str.startswith("Women").astype(int)
WC = ["Heavyweight", "Light Heavyweight", "Middleweight", "Welterweight", "Lightweight",
      "Featherweight", "Bantamweight", "Flyweight"]
for w in WC:
    F["wc_" + w[:4]] = F.weight_class.str.replace("Women's ", "").eq(w).astype(int)

# ---- strictly pre-fight rolling fighter stats (updated AFTER each fight) ----
S = defaultdict(lambda: dict(n=0, dec=0, finw=0, w=0, finl=0, l=0, kol=0, mins=0.0,
                             kd=0, kda=0, sl=0, sa=0, sub=0))
PRI = dict(dec=0.45, k=4)
rows = []
for r in F.itertuples():
    feats = {}
    for side, me, op in (("a", "a", "b"), ("b", "b", "a")):
        s = S[getattr(r, f"fighter_{me}_id")]
        n, mins = s["n"], max(s["mins"], 1e-9)
        feats[side] = dict(
            n=n,
            dec=(s["dec"] + PRI["dec"] * PRI["k"]) / (n + PRI["k"]),
            finw=(s["finw"] + 0.5 * 3) / (s["w"] + 3),
            finl=(s["finl"] + 0.5 * 3) / (s["l"] + 3),
            kol=s["kol"],
            avgmin=(s["mins"] + 10 * 3) / (n + 3),
            kd15=15 * (s["kd"] + 0.3) / (mins + 15) if n else 0.3,
            kda15=15 * (s["kda"] + 0.3) / (mins + 15) if n else 0.3,
            slpm=(s["sl"] + 3.5 * 15) / (mins + 15),
            sapm=(s["sa"] + 3.5 * 15) / (mins + 15),
            sub15=15 * (s["sub"] + 0.5) / (mins + 15),
        )
    a, b = feats["a"], feats["b"]
    rows.append(dict(
        dec_sum=a["dec"] + b["dec"], dec_min=min(a["dec"], b["dec"]),
        finw_max=max(a["finw"], b["finw"]), finl_max=max(a["finl"], b["finl"]),
        kol_max=max(a["kol"], b["kol"]), avgmin_sum=a["avgmin"] + b["avgmin"],
        kd_sum=a["kd15"] + b["kd15"], kda_sum=a["kda15"] + b["kda15"],
        pace=a["slpm"] + b["slpm"] + a["sapm"] + b["sapm"], sub_sum=a["sub15"] + b["sub15"],
        exp_min=min(a["n"], b["n"]), debut_any=int(min(a["n"], b["n"]) == 0)))
    if r.valid or r.method in ("DQ",):
        for me, op in (("a", "b"), ("b", "a")):
            s = S[getattr(r, f"fighter_{me}_id")]
            won = (r.outcome == "A_WIN") == (me == "a") and r.outcome in ("A_WIN", "B_WIN")
            lost = r.outcome in ("A_WIN", "B_WIN") and not won
            s["n"] += 1
            s["dec"] += int(r.method in DEC)
            s["w"] += int(won); s["l"] += int(lost)
            s["finw"] += int(won and r.method in FIN)
            s["finl"] += int(lost and r.method in FIN)
            s["kol"] += int(lost and r.method == "KO/TKO")
            s["mins"] += r.elapsed
            s["kd"] += getattr(r, f"{me}_kd") or 0; s["kda"] += getattr(r, f"{op}_kd") or 0
            s["sl"] += getattr(r, f"{me}_sig_str") or 0; s["sa"] += getattr(r, f"{op}_sig_str") or 0
            s["sub"] += getattr(r, f"{me}_sub_att") or 0
X = pd.concat([F, pd.DataFrame(rows)], axis=1)

# ---- market: vig-free close, symmetric mismatch only (orientation-free) ----
B = pd.read_csv(f"{REPO}/data/bfo_joined.csv")
for c in ("a_close", "b_close", "a_open", "b_open"):
    B[c] = pd.to_numeric(B[c], errors="coerce")
B = B.dropna(subset=["a_close", "b_close"])
ov = 1 / B.a_close + 1 / B.b_close
B = B[(ov >= 1.0) & (ov <= 1.15)]
B["pa"] = (1 / B.a_close) / (1 / B.a_close + 1 / B.b_close)
X = X.merge(B[["fight_id", "pa", "a_close", "b_close", "a_open", "b_open"]], on="fight_id", how="left")
X["mis"] = (X.pa - 0.5).abs()
X["mis2"] = X.mis ** 2
FD = pd.read_csv(f"{REPO}/data/features_v2.csv")[["fight_id", "rating_diff", "rd_x", "rd_y"]]
X = X.merge(FD, on="fight_id", how="left")
X["rdiff_abs"] = X.rating_diff.abs()
X["rd_sum"] = X.rd_x + X.rd_y

BASE = ["sched5", "women", "year"] + [c for c in X if c.startswith("wc_")]
MKT = ["mis", "mis2"]
FTR = ["dec_sum", "dec_min", "finw_max", "finl_max", "kol_max", "avgmin_sum", "kd_sum",
       "kda_sum", "pace", "sub_sum", "exp_min", "debut_any", "rdiff_abs", "rd_sum"]
MODELS = {"M0 base": BASE, "M1 base+market": BASE + MKT, "M2 base+market+fighter": BASE + MKT + FTR,
          "M3 base+fighter (no market)": BASE + FTR}

D = X[X.valid & X.pa.notna() & (X.year >= 2008)].copy()


def walk(df, target, years):
    preds = {m: pd.Series(np.nan, index=df.index) for m in MODELS}
    for Y in years:
        tr, te = df[df.year < Y], df[df.year == Y]
        if te.empty:
            continue
        for m, cols in MODELS.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            clf.fit(tr[cols], tr[target])
            preds[m].loc[te.index] = clf.predict_proba(te[cols])[:, 1]
    return pd.DataFrame(preds)


def report(df, P, target, label):
    out = []
    for blk, (y0, y1) in {"2015-22": (2015, 2022), "2023-24": (2023, 2024)}.items():
        idx = df.index[(df.year >= y0) & (df.year <= y1)]
        y = df.loc[idx, target]
        rate = df.loc[df.year < y0, target].mean()
        row = dict(target=label, block=blk, n=len(idx), base_rate=y.mean(),
                   ll_const=log_loss(y, np.full(len(y), rate)))
        for m in MODELS:
            row["ll " + m.split()[0]] = log_loss(y, P.loc[idx, m])
        row["brier M1"] = brier_score_loss(y, P.loc[idx, "M1 base+market"])
        row["brier M2"] = brier_score_loss(y, P.loc[idx, "M2 base+market+fighter"])
        out.append(row)
    return out


YEARS = range(2012, 2025)
res = []
Pg = walk(D, "gtd", YEARS)
res += report(D, Pg, "gtd", "GTD")
# over 2.5 rounds, 3-round fights only
D3 = D[D.sched5 == 0].copy()
D3["o25"] = (D3.elapsed > 12.5).astype(int)
Po = walk(D3, "o25", YEARS)
res += report(D3, Po, "o25", "O2.5 (3-rd)")
# over 1.5 rounds, 3-round fights
D3["o15"] = (D3.elapsed > 7.5).astype(int)
Po1 = walk(D3, "o15", YEARS)
res += report(D3, Po1, "o15", "O1.5 (3-rd)")
R = pd.DataFrame(res)
pd.set_option("display.width", 250)
print(R.round(4).to_string(index=False))

# coefficients of M2 fit on 2008-2022 (for the narrative only)
tr = D[D.year <= 2022]
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(tr[MODELS["M2 base+market+fighter"]], tr.gtd)
coef = pd.Series(clf[-1].coef_[0], index=MODELS["M2 base+market+fighter"]).sort_values()
print("\nM2 standardized coefs (GTD, fit 2008-22):\n", coef.round(3).to_string())

# calibration of the walk-forward M2 GTD predictions, 2015-2022
cal = D[(D.year >= 2015) & (D.year <= 2022)].assign(p=Pg["M2 base+market+fighter"])
cal["bin"] = pd.qcut(cal.p, 5)
print("\nM2 GTD calibration 2015-22:\n", cal.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("gtd", "mean"), n=("gtd", "size")).round(3).to_string())

# save per-bout walk-forward predictions for the downstream studies
D["p_gtd_m1"] = Pg["M1 base+market"]
D["p_gtd_m2"] = Pg["M2 base+market+fighter"]
D[["fight_id", "event_date", "year", "fighter_a", "fighter_b", "outcome", "method", "round",
   "elapsed", "sched5", "weight_class", "gtd", "pa", "a_close", "b_close", "a_open", "b_open",
   "p_gtd_m1", "p_gtd_m2"]].to_csv(f"{OUT}/bouts_gtd.csv", index=False, lineterminator="\n")
R.to_csv(f"{OUT}/finish_logloss.csv", index=False, lineterminator="\n")
print("\nGTD base rate by year:", D.groupby("year").gtd.mean().round(3).to_dict())
