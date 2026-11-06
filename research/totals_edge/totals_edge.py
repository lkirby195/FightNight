"""Totals edge test -- see PREREG.md (committed before any run).

Default run: dev 2015-22 and confirm 2023-24. Nothing dated 2025-01-01 or later
is loaded.
--holdout: the one-time read of 2025-01-01 .. 2026-11-01. It recomputes the
confirm gate on pre-2025 data first, refuses if the gate fails or if
HOLDOUT_READ.md already exists, and writes HOLDOUT_READ.md when done.

    python research/totals_edge/totals_edge.py [REPO] [OUT] [--holdout]
"""
import socket
import warnings
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _no_network(*a, **k):
    raise RuntimeError("totals_edge: network access is not allowed")


socket.socket.connect = _no_network
sys.dont_write_bytecode = True
# sklearn 1.8+ warns that penalty=None is deprecated; behaviour is unchanged
warnings.filterwarnings("ignore", category=FutureWarning)

HOLDOUT = "--holdout" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
HERE = Path(__file__).resolve().parent
REPO = ARGS[0] if ARGS else str(HERE.parents[1])
OUT = ARGS[1] if len(ARGS) > 1 else str(HERE)
PRE_CUT, HOLD_END = "2025-01-01", "2026-11-02"
MARKER = Path(OUT) / "HOLDOUT_READ.md"

DEC = {"U-DEC", "S-DEC", "M-DEC"}
FIN = {"KO/TKO", "SUB"}
WC = ["Heavyweight", "Light Heavyweight", "Middleweight", "Welterweight", "Lightweight",
      "Featherweight", "Bantamweight", "Flyweight"]
PRI = dict(dec=0.45, k=4)
FTR = ["dec_sum", "dec_min", "finw_max", "finl_max", "kol_max", "avgmin_sum", "kd_sum",
       "kda_sum", "pace", "sub_sum", "exp_min", "debut_any", "rdiff_abs", "rd_sum"]
MKT = ["mis", "mis2"]
TARGETS = {  # name -> (BFO prop, line, 3-round only, outcome column, half-round tie point)
    "O2.5 (3-rd) PRIMARY": ("total", 2.5, True, "o25", 12.5),
    "O1.5 (3-rd)": ("total", 1.5, True, "o15", 7.5),
    "GTD": ("gtd", None, False, "gtd", None),
}
PRIMARY = "O2.5 (3-rd) PRIMARY"
BLOCKS = {"dev 2015-22": (2015, 2022), "confirm 2023-24": (2023, 2024),
          "holdout 2025-26": (2025, 2026)}
L = lambda p: np.log(p / (1 - p))  # noqa: E731
S = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def build(cut):
    """M2's training frame, finish.py feature code verbatim, fights before `cut`."""
    F = pd.read_csv(f"{REPO}/data/fights_v2.csv")
    F = F[F.event_date < cut].sort_values(["event_date", "bout_order"]).reset_index(drop=True)
    F["year"] = F.event_date.str[:4].astype(int)
    F["valid"] = F.method.isin(DEC | FIN)
    F["gtd"] = F.method.isin(DEC).astype(int)
    mm, ss = F.time.fillna("0:00").str.split(":", expand=True).astype(int).T.values
    F["elapsed"] = (F["round"] - 1) * 5 + mm + ss / 60
    F["mx"] = F.groupby("event_id").bout_order.transform("max")
    F["sched5"] = ((F.title_bout == 1) | ((F.bout_order == F.mx) & (F.year >= 2012))).astype(int)
    F["women"] = F.weight_class.str.startswith("Women").astype(int)
    for w in WC:
        F["wc_" + w[:4]] = F.weight_class.str.replace("Women's ", "").eq(w).astype(int)

    St = defaultdict(lambda: dict(n=0, dec=0, finw=0, w=0, finl=0, l=0, kol=0, mins=0.0,
                                  kd=0, kda=0, sl=0, sa=0, sub=0))
    rows = []
    for r in F.itertuples():
        feats = {}
        for side in ("a", "b"):
            s = St[getattr(r, f"fighter_{side}_id")]
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
                s = St[getattr(r, f"fighter_{me}_id")]
                won = (r.outcome == "A_WIN") == (me == "a") and r.outcome in ("A_WIN", "B_WIN")
                lost = r.outcome in ("A_WIN", "B_WIN") and not won
                s["n"] += 1
                s["dec"] += int(r.method in DEC)
                s["w"] += int(won)
                s["l"] += int(lost)
                s["finw"] += int(won and r.method in FIN)
                s["finl"] += int(lost and r.method in FIN)
                s["kol"] += int(lost and r.method == "KO/TKO")
                s["mins"] += r.elapsed
                s["kd"] += getattr(r, f"{me}_kd") or 0
                s["kda"] += getattr(r, f"{op}_kd") or 0
                s["sl"] += getattr(r, f"{me}_sig_str") or 0
                s["sa"] += getattr(r, f"{op}_sig_str") or 0
                s["sub"] += getattr(r, f"{me}_sub_att") or 0
    X = pd.concat([F, pd.DataFrame(rows)], axis=1)

    B = pd.read_csv(f"{REPO}/data/bfo_joined.csv")
    for c in ("a_close", "b_close", "a_close_clean", "b_close_clean"):
        B[c] = pd.to_numeric(B[c], errors="coerce")
    # clean close where present (2026 only: in-play ticks), else the raw close
    B["ac"] = B.a_close_clean.fillna(B.a_close)
    B["bc"] = B.b_close_clean.fillna(B.b_close)
    B = B.dropna(subset=["ac", "bc"])
    ov = 1 / B.ac + 1 / B.bc
    B = B[(ov >= 1.0) & (ov <= 1.15)]
    B["pa"] = (1 / B.ac) / (1 / B.ac + 1 / B.bc)
    X = X.merge(B[["fight_id", "pa"]], on="fight_id", how="left")
    X["mis"] = (X.pa - 0.5).abs()
    X["mis2"] = X.mis ** 2
    FD = pd.read_csv(f"{REPO}/data/features_v2.csv")[["fight_id", "rating_diff", "rd_x", "rd_y"]]
    X = X.merge(FD, on="fight_id", how="left")
    X["rdiff_abs"] = X.rating_diff.abs()
    X["rd_sum"] = X.rd_x + X.rd_y
    X["o25"] = (X.elapsed > 12.5).astype(int)
    X["o15"] = (X.elapsed > 7.5).astype(int)
    D = X[X.valid & X.pa.notna() & (X.year >= 2008)].copy()
    return D, [c for c in X if c.startswith("wc_")]


def m2_preds(D, wc, target, three_rd, holdout):
    """Walk-forward M2 P(target). Annual refit 2015-2024; holdout: one fit on < 2025."""
    df = D[D.sched5 == 0] if three_rd else D
    cols = ["sched5", "women", "year"] + wc + MKT + FTR
    out = pd.Series(np.nan, index=df.index)
    fits = [(Y, df.year < Y, df.year == Y) for Y in range(2015, 2025)]
    if holdout:
        fits.append(("hold", df.year < 2025, df.year >= 2025))
    for _, tr, te in fits:
        if not te.any():
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(df.loc[tr, cols], df.loc[tr, target])
        out[te] = clf.predict_proba(df.loc[te, cols])[:, 1]
    return df.assign(m=out)


def prices(prop, line, cut):
    P = pd.read_csv(f"{REPO}/data/bfo_props_joined.csv")
    P = P[(P.event_date < cut) & (P.prop == prop)]
    P = P[P.line == line] if line is not None else P
    P = P.sort_values(["fight_id", "p1_ticks"], ascending=[True, False]).drop_duplicates("fight_id")
    ovc = 1 / P.p1_close_clean + 1 / P.p2_close_clean
    ovo = 1 / P.p1_open + 1 / P.p2_open
    P = P[(ovc >= 1.0) & (ovc <= 1.15)].copy()
    P["q"] = (1 / P.p1_close_clean) / (1 / P.p1_close_clean + 1 / P.p2_close_clean)
    P["q_open"] = np.where((ovo >= 1.0) & (ovo <= 1.15), (1 / P.p1_open) / (1 / P.p1_open + 1 / P.p2_open), np.nan)
    return P[["fight_id", "q", "q_open", "p1_open", "p2_open", "p1_close_clean", "p2_close_clean"]]


def eval_rows(D, wc, name, cut, holdout):
    prop, line, three_rd, tgt, tie = TARGETS[name]
    df = m2_preds(D, wc, tgt, three_rd, holdout)
    if tie is not None:
        df = df[np.abs(df.elapsed - tie) > 1e-9]
    E = df[["fight_id", "year", tgt, "m"]].rename(columns={tgt: "y"}).merge(
        prices(prop, line, cut), on="fight_id", how="inner")
    return E[E.m.notna() & (E.year >= 2015)].reset_index(drop=True)


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_lr(Xm, y):
    return LogisticRegression(penalty=None, max_iter=5000).fit(Xm, y)


def blend_w(y, q, m):
    g = np.round(np.arange(0, 1.0001, 0.01), 2)
    losses = [ll(y, S(w * L(q) + (1 - w) * L(m))).mean() for w in g]
    return float(g[int(np.argmin(losses))])


def evaluate(Efit, Eev, rng):
    """Stack/recal fit on Efit, scored on Eev; blend w* fit inside Eev."""
    rec = fit_lr(L(Efit.q.values)[:, None], Efit.y)
    stk = fit_lr(np.c_[L(Efit.q.values), L(Efit.m.values)], Efit.y)
    y, q, m = Eev.y.values, Eev.q.values, Eev.m.values
    p_rec = rec.predict_proba(L(q)[:, None])[:, 1]
    p_stk = stk.predict_proba(np.c_[L(q), L(m)])[:, 1]
    d = ll(y, p_rec) - ll(y, p_stk)
    w = blend_w(y, q, m)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
    ws = np.array([blend_w(y[i], q[i], m[i]) for i in idx])
    b, c = stk.coef_[0]
    # line movement: (q_close - q_open) on (m - q_open)
    mv = Eev.dropna(subset=["q_open"])
    xo, yo = (mv.m - mv.q_open).values, (mv.q - mv.q_open).values
    sl = np.polyfit(xo, yo, 1)[0]
    res = yo - np.polyval(np.polyfit(xo, yo, 1), xo)
    sl_se = np.sqrt(res.var(ddof=2) / ((xo - xo.mean()) ** 2).sum())
    return dict(n=len(y), base_rate=y.mean(), ll_market=ll(y, q).mean(), ll_model=ll(y, m).mean(),
                ll_recal=ll(y, p_rec).mean(), ll_stack=ll(y, p_stk).mean(),
                gain=d.mean(), gain_t=d.mean() / (d.std(ddof=1) / np.sqrt(len(d))),
                stack_b_mkt=b, stack_c_model=c, model_share=c / (b + c),
                blend_w=w, blend_w_lo=np.percentile(ws, 2.5), blend_w_hi=np.percentile(ws, 97.5),
                move_n=len(mv), move_slope=sl, move_t=sl / sl_se)


def bets(Efit, Eev, thr=0.03):
    """Blend-at-open: stack on q_open fit on Efit; bet EV > thr at the raw open."""
    f = Efit.dropna(subset=["q_open"])
    e = Eev.dropna(subset=["q_open"]).copy()
    stk = fit_lr(np.c_[L(f.q_open.values), L(f.m.values)], f.y)
    p = stk.predict_proba(np.c_[L(e.q_open.values), L(e.m.values)])[:, 1]
    ev_o, ev_u = p * e.p1_open - 1, (1 - p) * e.p2_open - 1
    over = ev_o >= ev_u
    ev = np.where(over, ev_o, ev_u)
    e = e[ev > thr].assign(over=over[ev > thr])
    win = np.where(e.over, e.y == 1, e.y == 0)
    pnl = np.where(win, np.where(e.over, e.p1_open, e.p2_open) - 1, -1.0)
    clv = np.where(e.over, e.q - e.q_open, e.q_open - e.q)
    n = len(e)
    if n == 0:
        return dict(bets=0)
    return dict(bets=n, over_share=e.over.mean(), hit=win.mean(), roi=pnl.mean(),
                roi_t=pnl.mean() / (pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                clv_pts=100 * clv.mean(), clv_t=clv.mean() / (clv.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan)


def blk(E, name):
    a, b = BLOCKS[name]
    return E[(E.year >= a) & (E.year <= b)]


def run(cut, holdout):
    D, wc = build(cut)
    out, per_fight = [], []
    for t in TARGETS:
        E = eval_rows(D, wc, t, cut, holdout)
        dev, conf = blk(E, "dev 2015-22"), blk(E, "confirm 2023-24")
        plan = [("dev 2015-22", dev, dev, 1), ("confirm 2023-24", dev, conf, 2)]
        if holdout:
            plan = [("holdout 2025-26", pd.concat([dev, conf]), blk(E, "holdout 2025-26"), 3)]
        for bname, fit, ev, seed in plan:
            r = dict(target=t, block=bname, **evaluate(fit, ev, np.random.default_rng(7 + seed)))
            r.update({"bet_" + k: v for k, v in bets(fit, ev).items()})
            out.append(r)
        keep = E if holdout else E[E.year <= 2024]
        per_fight.append(keep.assign(target=t))
    return pd.DataFrame(out), pd.concat(per_fight)


def gate(R):
    c = R[(R.target == PRIMARY) & (R.block == "confirm 2023-24")].iloc[0]
    return bool(c.gain > 0 and c.blend_w < 1.0), c


def show(R):
    pd.set_option("display.width", 250)
    cols = ["target", "block", "n", "base_rate", "ll_market", "ll_model", "ll_recal", "ll_stack",
            "gain", "gain_t", "model_share", "blend_w", "blend_w_lo", "blend_w_hi",
            "move_slope", "move_t", "bet_bets", "bet_hit", "bet_roi", "bet_roi_t", "bet_clv_pts", "bet_clv_t"]
    print(R[[c for c in cols if c in R]].round(4).to_string(index=False))


if __name__ == "__main__":
    if not HOLDOUT:
        R, PF = run(PRE_CUT, False)
        show(R)
        ok, c = gate(R)
        print(f"\nGATE (primary, confirm): gain {c.gain:+.4f} (t {c.gain_t:.2f}), blend w* {c.blend_w:.2f}"
              f" -> {'PASS' if ok else 'FAIL'}")
        R.to_csv(f"{OUT}/results_dev_confirm.csv", index=False, lineterminator="\n")
        PF.to_csv(f"{OUT}/per_fight_dev_confirm.csv", index=False, lineterminator="\n")
        sys.exit(0)

    if MARKER.exists():
        sys.exit(f"refusing: {MARKER} exists -- the holdout has already been read once")
    R0, _ = run(PRE_CUT, False)
    ok, c = gate(R0)
    if not ok:
        sys.exit(f"refusing: confirm gate FAILED (gain {c.gain:+.4f}, w* {c.blend_w:.2f}); holdout stays unread")
    R, PF = run(HOLD_END, True)
    show(R)
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    R.to_csv(f"{OUT}/results_holdout.csv", index=False, lineterminator="\n")
    PF[PF.year >= 2025].to_csv(f"{OUT}/per_fight_holdout.csv", index=False, lineterminator="\n")
    MARKER.write_text(
        f"# Holdout read\n\nRead once on {date.today().isoformat()} at commit {head}.\n"
        f"Window: 2025-01-01 to {HOLD_END} (exclusive). Results: results_holdout.csv.\n"
        "Do not rerun; totals_edge.py --holdout refuses while this file exists.\n",
        newline="\n")
