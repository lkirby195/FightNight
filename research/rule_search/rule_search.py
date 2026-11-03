"""Rule-search study, 2008-2022. Read-only.

Question: if a grid of simple betting rules is searched across long history,
does the search find real edges or only luck? research/fade_buckets showed the
fade-the-favourite idea fails and turned up one post-hoc standout cell; this
scores the search procedure itself against a null in which no rule has an edge.

Bouts, prices and definitions are fade_buckets': its per_bout(), tag() and
universe() are imported and called, not copied, and the loader below is its
load() over the wider window. Model probabilities are walk-forward: the frozen
preds where they cover a year (2012-2022), fade_buckets' year-by-year refit
(year Y fit on bouts dated before Y-01-01, features cut at 2023-01-01) for the
years they do not. The grid (105 rules), STANDOUT and the pass bar were fixed
in the brief before this ran; GAP and FLIP (signals.classify) are benchmarks.

Window 2008-01-01 <= event_date < 2023-01-01. Nothing dated 2023 or later is
loaded from any results file (fights_v2 and features are cut on read, every
other file is cut to the window's fight_ids on read); 2023-24 is reserved to
confirm whatever survives and 2025 stays sealed. Sockets are blocked. Nothing
outside research/rule_search/ is written. Fixed seed: two runs are identical.

Run: python research/rule_search/rule_search.py   (any cwd)
Outputs: coverage.csv, rules.csv, rule_blocks.csv, walkforward.csv,
walkforward_pnl.png, null_dist.csv, null_summary.csv, null_max_t.png.
"""
from __future__ import annotations

import importlib.util
import os
import socket
import sys


def _no_network(*a, **k):
    raise RuntimeError("rule_search: network access is not allowed")


socket.socket.connect = _no_network
sys.dont_write_bytecode = True                       # no __pycache__ in research/fade_buckets/

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline           # noqa: E402
from sklearn.preprocessing import StandardScaler     # noqa: E402

import betting_system as bs                          # noqa: E402
import clv_eval                                      # noqa: E402
import features                                      # noqa: E402
import signals                                       # noqa: E402
import stage2                                        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "fade_buckets", os.path.join(ROOT, "research", "fade_buckets", "fade_buckets.py"))
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)

LO, HI = "2008-01-01", "2023-01-01"                  # window: LO <= event_date < HI
Y0, Y1 = 2008, 2022
YEARS = list(range(Y0, Y1 + 1))
FROZEN = "data/model_only_preds_frozen_2026-09-08.csv"
OUT = HERE

SEED = 20261103
N_REPS = 1000
REFIT_TOL = 0.005                                    # refit vs frozen, max |dp| per year
B1_MIN = 300                                         # B1 merges into B2 below this
MIN_BLOCK_N = 30                                     # pass bar 3
WF_YEARS = list(range(2011, 2023))
WF_MIN_PRIOR = 30
STANDOUT_T = 1.65
RD_KNOWN = signals.UNKNOWN_RD                        # 160
CAP = signals.SKIP_LINE_MAX                          # +250
SOFT, SHARP = (2008, 2017), (2018, 2022)
FAV_T = (3, 5, 8, 11, 15)
DOG_T = (5, 10, 15, 20, 25)
BLUE = fb.COLORS["KK"]


def fail(msg):
    raise SystemExit(f"STOP: {msg}")


# ---- inputs -----------------------------------------------------------------

def refit_preds():
    """fade_buckets' walk-forward refit (clv_eval.gen_model_preds: year Y fit
    on bouts dated before Y-01-01, predict Y) from features cut at HI, for
    every window year. Same matrix, pipeline and orientation as the frozen
    file; build_matrix centres and median-fills over the frame it is given,
    which here ends at HI."""
    df = pd.read_csv("data/features_v2.csv", parse_dates=["date"])
    df = df[df.date < HI]
    assert df.date.max() < pd.Timestamp(HI)
    df = df[df.label != 0.5].copy()
    X = stage2.build_matrix(df).drop(columns=clv_eval.PASSC)
    y = df.label.astype(int).values
    fl = df.fight_id.map(features.flip).values
    out = []
    for Y in YEARS:
        tr = (df.date < f"{Y}-01-01").values
        ev = ((df.date >= f"{Y}-01-01") & (df.date < f"{Y + 1}-01-01")).values
        m = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(C=0.3, max_iter=3000, fit_intercept=False))
        m.fit(np.vstack([X[tr].values, -X[tr].values]), np.concatenate([y[tr], 1 - y[tr]]))
        pm = m.predict_proba(X[ev].values)[:, 1]
        out.append(pd.DataFrame({"fight_id": df.fight_id[ev].values, "year": Y,
                                 "p_model_a": np.where(fl[ev], 1 - pm, pm),
                                 "y_a": np.where(fl[ev], 1 - y[ev], y[ev])}))
    return pd.concat(out, ignore_index=True)


def model_preds(ids, ev_year):
    """Frozen preds for the years they cover, the refit for the rest, and the
    refit-vs-frozen diff on every overlap year."""
    F = pd.read_csv(FROZEN)
    F = F[F.fight_id.isin(ids)].copy()
    R = refit_preds()
    R = R[R.fight_id.isin(ids)].copy()
    for name, P in (("frozen", F), ("refit", R)):
        if not (P.year.values == ev_year[P.fight_id].values).all():
            fail(f"{name} preds: year column is not the event year")
    fy = sorted(int(y) for y in F.year.unique())
    rows = []
    for Y in fy:
        f = F[F.year == Y].set_index("fight_id")
        r = R[R.year == Y].set_index("fight_id")
        if set(f.index) != set(r.index):
            fail(f"{Y}: frozen and refit cover different bouts")
        if not (f.y_a.astype(int) == r.y_a[f.index].astype(int)).all():
            fail(f"{Y}: frozen and refit disagree on the result")
        d = (f.p_model_a - r.p_model_a[f.index]).abs()
        rows.append(dict(year=Y, n=len(f), max_dp=d.max(), mean_dp=d.mean()))
    M = pd.concat([F.assign(src="frozen"), R[~R.year.isin(fy)].assign(src="refit")],
                  ignore_index=True)
    return M[["fight_id", "year", "p_model_a", "y_a", "src"]], pd.DataFrame(rows)


def build_frame(fv_hist, fv, J, rd, M):
    """fade_buckets.load()'s frame over this window and these preds."""
    cols = ["fight_id", "event_date", "event_name", "bout_order", "fighter_a_id",
            "fighter_a", "fighter_b_id", "fighter_b", "method", "round"]
    D = (M.merge(J, on="fight_id").merge(fv[cols].rename(columns={"round": "rnd"}),
                                         on="fight_id")
          .merge(rd, on="fight_id", how="left"))
    D = D.dropna(subset=["y_a", "a_open", "b_open"])
    D["open_suspect"] = D.open_suspect.fillna(0).astype(int)
    D["a_open_raw"] = D.a_open_raw.fillna(D.a_open)
    D["b_open_raw"] = D.b_open_raw.fillna(D.b_open)
    first = signals.first_bout_dates(fv_hist)
    known = set(pd.read_csv("data/fighters_v2.csv").fighter_id)
    D["known_a"] = [f in known and signals.on_record(f, d, first)
                    for f, d in zip(D.fighter_a_id, D.event_date)]
    D["known_b"] = [f in known and signals.on_record(f, d, first)
                    for f, d in zip(D.fighter_b_id, D.event_date)]
    D["projectable"] = D.known_a & D.known_b
    if D.rd_a.isna().any():
        fail(f"{bs.PREFIGHT_RD} has no row for {int(D.rd_a.isna().sum())} bouts")
    D = D.sort_values(["event_date", "bout_order", "fight_id"]).reset_index(drop=True)
    assert D.event_date.max() < HI and D.event_date.min() >= LO
    D["yr"] = D.event_date.str[:4].astype(int)
    return D


def exclusion_masks(D):
    """fade_buckets.universe()'s exclusions, in its order, as cumulative masks."""
    m1 = D.projectable
    m2 = m1 & D.prefilter
    m3 = m2 & ~D.pickem
    m4 = m3 & (D.gap != 0)
    return dict(debut=~m1, prefilter=m1 & ~D.prefilter, pickem=m2 & D.pickem,
                gap0=m3 & (D.gap == 0)), m4


def coverage(fv, J, M, D, U, ex_masks):
    fvy = fv.assign(yr=fv.event_date.str[:4].astype(int))
    Jf = J.merge(fvy[["fight_id", "yr", "outcome"]], on="fight_id")
    op = Jf[Jf.a_open.notna() & Jf.b_open.notna()]
    cl = Jf[Jf.a_close.notna() & Jf.b_close.notna()]
    Mw = M.merge(fvy[["fight_id", "yr"]], on="fight_id")
    rows = []
    for Y in YEARS:
        o = op[op.yr == Y]
        u = U[U.yr == Y]
        d = D.yr == Y
        src = Mw[Mw.yr == Y].src.unique()
        rows.append(dict(
            year=Y, bouts=int((fvy.yr == Y).sum()), open_2sided=len(o),
            p_mod=int((Mw.yr == Y).sum()), p_mod_source="/".join(sorted(src)) or "-",
            draws_open=int((o.outcome == "DRAW").sum()), ncs_open=int((o.outcome == "NC").sum()),
            priced_with_p_mod=int(d.sum()),
            ex_debut=int((ex_masks["debut"] & d).sum()),
            ex_prefilter=int((ex_masks["prefilter"] & d).sum()),
            ex_pickem=int((ex_masks["pickem"] & d).sum()),
            ex_gap0=int((ex_masks["gap0"] & d).sum()),
            usable=len(u), kk=int((u.tier == "KK").sum()),
            close_2sided=int((cl.yr == Y).sum()),
            usable_with_clean_close=int(u.clv_pp.notna().sum())))
    C = pd.DataFrame(rows)
    other = C.open_2sided - C.draws_open - C.ncs_open - C.priced_with_p_mod
    if (other != 0).any():
        fail(f"bouts with an open and no model probability that are not draws/NCs: "
             f"{dict(zip(C.year[other != 0], other[other != 0]))}")
    ex = C[["ex_debut", "ex_prefilter", "ex_pickem", "ex_gap0"]].sum(axis=1)
    if not (C.priced_with_p_mod - ex == C.usable).all():
        fail("coverage: priced - exclusions != usable")
    return C


# ---- rules ------------------------------------------------------------------

RD_TXT = {"none": "", "dog": ", dog RD <= 160", "fav": ", fav RD <= 160",
          "both": ", both RD <= 160"}
PICK_TXT = {"any": "", "dog": ", model picks dog (p_mod < 0.50)",
            "dog60": ", model picks dog at 0.60+ (p_mod <= 0.40)"}
CAP_TXT = {"none": "", "250": ", dog <= +250"}


def rule_list():
    R = []
    for T in FAV_T:
        for rdf in ("none", "dog", "both"):
            R.append(dict(family="FAV", in_grid=1, side="fav", T=T, pick="", cap="", rd=rdf,
                          definition=f"bet fav when gap >= {T} pp" + RD_TXT[rdf]))
    for T in DOG_T:
        for pick in ("any", "dog", "dog60"):
            for cap in ("none", "250"):
                for rdf in ("none", "fav", "both"):
                    R.append(dict(family="DOG", in_grid=1, side="dog", T=T, pick=pick,
                                  cap=cap, rd=rdf,
                                  definition=f"bet dog when gap <= -{T} pp" + PICK_TXT[pick]
                                  + CAP_TXT[cap] + RD_TXT[rdf]))
    nf = sum(r["family"] == "FAV" for r in R)
    for i, r in enumerate(R):
        r["id"] = f"F{i + 1:02d}" if r["family"] == "FAV" else f"D{i + 1 - nf:02d}"
    R.append(dict(id="STANDOUT", family="STANDOUT", in_grid=0, side="dog", T="", pick="dog",
                  cap="", rd="", definition="bet dog when -25 < gap <= -20 pp and p_mod < 0.50 "
                  "(found on 2018-22; fresh test 2008-17)"))
    for rl in ("GAP", "FLIP"):
        R.append(dict(id=rl, family="BENCHMARK", in_grid=0, side="pick", T="", pick="",
                      cap="", rd="", definition=f"frozen {rl} rule via signals.classify, 1u at "
                      "open (VOID rows excluded; pick'ems included, as in the sim)"))
    T = pd.DataFrame(R)[["id", "family", "in_grid", "side", "T", "pick", "cap", "rd",
                         "definition"]]
    if T.in_grid.sum() != 105 or T.id.duplicated().any():
        fail("rule grid is not 105 distinct rules")
    return T


def rule_bets(r, D, in_u):
    """(row indices into D in date order, bet-on-favourite flag per bet)."""
    rd_f, rd_d = D.rd_fav <= RD_KNOWN, D.rd_dog <= RD_KNOWN
    true = pd.Series(True, index=D.index)
    if r.family == "FAV":
        rd = {"none": true, "dog": rd_d, "both": rd_f & rd_d}[r.rd]
        m = in_u & (D.gap >= r.T) & rd
        sf = np.ones(int(m.sum()), bool)
    elif r.family == "DOG":
        rd = {"none": true, "fav": rd_f, "both": rd_f & rd_d}[r.rd]
        pick = {"any": true, "dog": D.p_mod < 0.50, "dog60": D.p_mod <= 0.40}[r.pick]
        cap = {"none": true, "250": D.dog_am <= CAP}[r.cap]
        m = in_u & (D.gap <= -r.T) & pick & cap & rd
        sf = np.zeros(int(m.sum()), bool)
    elif r.family == "STANDOUT":
        m = in_u & (D.gap > -25) & (D.gap <= -20) & (D.p_mod < 0.50)
        sf = np.zeros(int(m.sum()), bool)
    else:
        m = D.tag == r.id
        g = D[m]
        sf = ((g.tag_side == 1) == (g.a_open < g.b_open)).values
    idx = np.flatnonzero(m.values)
    # every bet is on the model side (the side fade_buckets' clv_pp is taken on)
    if not (sf == (D.gap.values[idx] > 0)).all():
        fail(f"rule {r.id}: a bet is not on the model side")
    return idx, sf


# ---- scoring (real data = 1 replicate) --------------------------------------

def rule_year_sums(PF, PD, bets, yidx):
    """PF / PD: (reps x bouts) P&L of 1u at open on the favourite / dog of
    every bout of D. -> n (rules x years), and S (sum), Q (sum of squares),
    W (wins), each (reps x rules x years)."""
    reps, R, NY = PF.shape[0], len(bets), len(YEARS)
    n = np.zeros((R, NY), int)
    S, Q = np.zeros((reps, R, NY)), np.zeros((reps, R, NY))
    W = np.zeros((reps, R, NY), int)
    for k, (idx, sf) in enumerate(bets):
        X = np.where(sf[None, :], PF[:, idx], PD[:, idx])
        b = np.searchsorted(yidx[idx], np.arange(NY + 1))
        for j in range(NY):
            s, e = b[j], b[j + 1]
            n[k, j] = e - s
            if e > s:
                x = X[:, s:e]
                S[:, k, j], Q[:, k, j], W[:, k, j] = x.sum(1), (x * x).sum(1), (x > 0).sum(1)
    return n, S, Q, W


def tstat(n, S, Q):
    """ROI, SE and t of 1u bets from count, sum and sum of squares (sample SD,
    ddof 1). NaN where undefined (n < 2 or zero spread)."""
    n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        roi = S / n
        var = np.maximum((Q - S * S / n) / (n - 1), 0.0)
        se = np.sqrt(var / n)
        t = roi / se
    roi = np.where(n > 0, roi, np.nan)
    se = np.where(n > 1, se, np.nan)
    t = np.where((n > 1) & (se > 0), t, np.nan)
    return roi, se, t


def year_cols(lo, hi):
    return [Y - Y0 for Y in YEARS if lo <= Y <= hi]


def agg(n, S, Q, W, lo, hi):
    c = year_cols(lo, hi)
    return n[:, c].sum(1), S[:, :, c].sum(2), Q[:, :, c].sum(2), W[:, :, c].sum(2)


def walk_forward(n, S, Q, W, G):
    """For each year Y of WF_YEARS: among the first G rules (the grid) with at
    least WF_MIN_PRIOR bets dated before Y, the one with the highest pooled t
    over those bets (ties -> first in rules.csv order), bet in year Y.
    -> per-year arrays (reps x len(WF_YEARS)), selection -1 = no bets."""
    n, S, Q, W = n[:G], S[:, :G], Q[:, :G], W[:, :G]
    reps = S.shape[0]
    cn, cS, cQ = np.cumsum(n, 1), np.cumsum(S, 2), np.cumsum(Q, 2)
    r = np.arange(reps)
    keys = ("sel", "tied", "prior_n", "prior_roi", "prior_t", "n", "S", "Q", "W")
    out = {k: [] for k in keys}
    for Y in WF_YEARS:
        j = Y - Y0
        pn, pS, pQ = cn[:, j - 1], cS[:, :, j - 1], cQ[:, :, j - 1]
        proi, _, pt = tstat(pn, pS, pQ)
        elig = pn >= WF_MIN_PRIOR
        if not elig.any():
            z = np.zeros(reps)
            for k, v in zip(keys, (z - 1, z, z, z * np.nan, z * np.nan, z, z, z, z)):
                out[k].append(v)
            continue
        key = np.where(elig[None, :], np.where(np.isfinite(pt), pt, -1e300), -np.inf)
        sel = key.argmax(1)
        out["sel"].append(sel)
        out["tied"].append((key == key[r, sel][:, None]).sum(1))
        out["prior_n"].append(pn[sel])
        out["prior_roi"].append(proi[r, sel])
        out["prior_t"].append(pt[r, sel])
        out["n"].append(n[sel, j])
        out["S"].append(S[r, sel, j])
        out["Q"].append(Q[r, sel, j])
        out["W"].append(W[r, sel, j])
    return {k: np.stack(v, 1) for k, v in out.items()}


def score(PF, PD, bets, yidx, blocks, G):
    """Everything Steps 2-3 need from one P&L draw (real or null)."""
    n, S, Q, W = rule_year_sums(PF, PD, bets, yidx)
    res = dict(n=n, S=S, Q=Q, W=W, block={})
    for name, lo, hi in blocks:
        res["block"][name] = agg(n, S, Q, W, lo, hi)
    for name, (lo, hi) in (("pooled", (Y0, Y1)), ("soft", SOFT), ("sharp", SHARP)):
        res[name] = agg(n, S, Q, W, lo, hi)
    res["wf"] = walk_forward(n, S, Q, W, G)
    return res


def summary_stats(res, G, blocks):
    """Per replicate: (a) max pooled t over the grid and its rule, (b) grid rules
    with ROI > 0 in every block, (c) walk-forward OOS ROI, the benchmarks'
    pooled ROI and the grid's mean pooled ROI."""
    pn, pS, pQ, _ = res["pooled"]
    roi, _, t = tstat(pn, pS, pQ)
    tg = np.where(np.isfinite(t[:, :G]), t[:, :G], -np.inf)
    pos = np.ones((pS.shape[0], G), bool)
    for name, _, _ in blocks:
        bn, bS, bQ, _ = res["block"][name]
        broi, _, _ = tstat(bn, bS, bQ)
        pos &= np.nan_to_num(broi[:, :G], nan=-1.0) > 0
    wf = res["wf"]
    wn, wS, wQ = wf["n"].sum(1), wf["S"].sum(1), wf["Q"].sum(1)
    wroi, _, wt = tstat(wn, wS, wQ)
    return dict(max_t=tg.max(1), max_t_rule=tg.argmax(1), n_pos_all=pos.sum(1),
                wf_n=wn, wf_net=wS, wf_roi=wroi, wf_t=wt,
                gap_roi=roi[:, G + 1], flip_roi=roi[:, G + 2],
                grid_mean_roi=np.nanmean(roi[:, :G], 1))


def pnl_matrices(fav_won, o_fav, o_dog):
    PF = np.where(fav_won, o_fav[None, :] - 1, -1.0)
    PD = np.where(~fav_won, o_dog[None, :] - 1, -1.0)
    return PF, PD


def bet_pnl(PF, PD, idx, sf):
    return np.where(sf[None, :], PF[:, idx], PD[:, idx])


def num(v, dp=2):
    v = float(v)
    return round(v, dp) if np.isfinite(v) else np.nan


# ---- sanity checks ------------------------------------------------------------

def check_fade_frame(U):
    """2018-22 bouts == fade_buckets' own bucketed per-bout frame (its load(),
    tag(), per_bout(), universe()), exactly, on every column both share."""
    fD, _, _ = fb.load()
    fU, _ = fb.universe(fb.per_bout(fb.tag(fD)))
    mine = U[U.event_date >= "2018-01-01"].sort_values("fight_id").reset_index(drop=True)
    theirs = fU.sort_values("fight_id").reset_index(drop=True)
    if len(mine) != len(theirs) or not (mine.fight_id.values == theirs.fight_id.values).all():
        return False, f"bout sets differ: {len(mine)} here vs {len(theirs)} in fade_buckets"
    shared = [c for c in theirs.columns if c in mine.columns]
    bad = [c for c in shared
           if not ((mine[c] == theirs[c]) | (mine[c].isna() & theirs[c].isna())).all()]
    return not bad, (f"{len(theirs)} bouts, {len(shared)} shared columns, exact match on "
                     f"{len(shared) - len(bad)}; mismatched: {', '.join(bad) or 'none'}")


def check_sim(D, rules, bets, PF, PD):
    """GAP/FLIP bets here vs betting_system.run() on the same frame and vs the
    committed ledger, 2012-2022, 1u at open: count, W-L, net, ROI."""
    lines, ok = [], True
    D12 = D[D.yr >= 2012]
    Bc = bs.counted(bs.run(D12))
    L = pd.read_csv(bs.LEDGER)
    L = L[(L.event_date >= "2012-01-01") & (L.event_date < HI) & (L.units > 0)]
    for rl in ("GAP", "FLIP"):
        idx, sf = bets[int(np.flatnonzero(rules.id.values == rl)[0])]
        keep = D.yr.values[idx] >= 2012
        x = bet_pnl(PF, PD, idx[keep], sf[keep])[0]
        mine = (len(x), int((x > 0).sum()), int((x < 0).sum()), x.sum())
        b = Bc[Bc.rule == rl]
        sim = (len(b), int((b.pnl_at_open > 0).sum()), int((b.pnl_at_open < 0).sum()),
               b.pnl_at_open.sum())
        lg = L[L.rule == rl]
        led = (len(lg), int((lg.result == "WIN").sum()), int((lg.result == "LOSS").sum()),
               lg.pnl_at_open.sum())
        same_ids = (set(D.fight_id.values[idx[keep]]) == set(b.fight_id)
                    == set(lg.fight_id))
        good = (mine[:3] == sim[:3] and abs(mine[3] - sim[3]) < 1e-9 and same_ids
                and led[:3] == sim[:3] and abs(led[3] - sim[3]) <= 1e-4 * max(sim[0], 1))
        ok &= good
        fmt = lambda v: f"{v[0]} {v[1]}-{v[2]} net {v[3]:+.2f}u ROI {v[3] / v[0]:+.2%}"
        lines.append(f"{rl:4s} here {fmt(mine)} | bs.run {fmt(sim)} | ledger {fmt(led)} "
                     f"| same fight_ids {same_ids} -> {'PASS' if good else 'FAIL'}")
    return ok, lines


def check_vectorised(rules, bets, res, PF, PD):
    """The summed path (rule_year_sums -> tstat) == a direct per-rule
    computation on the real P&L."""
    pn, pS, _, _ = res["pooled"]
    _, _, t = tstat(*res["pooled"][:3])
    worst = 0.0
    for k, (idx, sf) in enumerate(bets):
        x = bet_pnl(PF, PD, idx, sf)[0]
        if len(x) != pn[k] or abs(x.sum() - pS[0, k]) > 1e-9:
            return False, rules.id[k]
        if len(x) > 1 and x.std(ddof=1) > 0:
            worst = max(worst, abs(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) - t[0, k]))
    return worst < 1e-9, f"max |t diff| {worst:.1e}"


# ---- tables -------------------------------------------------------------------

def rule_table(rules, res, blocks, D, bets, G, null_max_t, q95):
    rows = []
    pn, pS, pQ, pW = res["pooled"]
    for k, r in enumerate(rules.itertuples(index=False)):
        row = dict(id=r.id, family=r.family, definition=r.definition)
        broi, bns = [], []
        for name, _, _ in blocks:
            bn, bS, bQ, bW = res["block"][name]
            roi, se, t = tstat(bn[k], bS[0, k], bQ[0, k])
            row.update({f"{name}_n": int(bn[k]),
                        f"{name}_wl": f"{int(bW[0, k])}-{int(bn[k] - bW[0, k])}",
                        f"{name}_roi_pct": num(roi * 100), f"{name}_se_pct": num(se * 100),
                        f"{name}_t": num(t)})
            broi.append(float(roi))
            bns.append(int(bn[k]))
        roi, se, t = tstat(pn[k], pS[0, k], pQ[0, k])
        row.update(pooled_n=int(pn[k]), pooled_wl=f"{int(pW[0, k])}-{int(pn[k] - pW[0, k])}",
                   pooled_net_u=num(pS[0, k], 3), pooled_roi_pct=num(roi * 100),
                   pooled_se_pct=num(se * 100), pooled_t=num(t))
        fin = [v for v in broi if np.isfinite(v)]
        row.update(blocks_pos=sum(v > 0 for v in fin),
                   worst_block_roi_pct=num(min(fin) * 100) if fin else np.nan)
        for era in ("soft", "sharp"):
            en, eS, eQ, _ = res[era]
            eroi, _, et = tstat(en[k], eS[0, k], eQ[0, k])
            row.update({f"{era}_n": int(en[k]), f"{era}_roi_pct": num(eroi * 100),
                        f"{era}_t": num(et)})
        c = D.clv_pp.values[bets[k][0]]
        c = c[~np.isnan(c)]
        row.update(clv_n=len(c), mean_clv_pp=num(c.mean()) if len(c) else np.nan)
        if k < G:
            b1 = all(np.isfinite(v) and v > 0 for v in broi)
            b2 = bool(np.isfinite(t) and t > q95)
            b3 = all(v >= MIN_BLOCK_N for v in bns)
            row.update(null_pct_max_t=num(100 * np.mean(null_max_t < t), 1) if np.isfinite(t)
                       else np.nan, bar1_all_blocks_pos=int(b1), bar2_t_above_null95=int(b2),
                       bar3_min30_each_block=int(b3), advances=int(b1 and b2 and b3))
        rows.append(row)
    return pd.DataFrame(rows)


def wf_table(rules, wf, D, bets, PF, PD, ov):
    """Real walk-forward: one row per year, a total row, and the OOS bets."""
    rows, bl = [], []
    cum = 0.0
    for i, Y in enumerate(WF_YEARS):
        s = int(wf["sel"][0, i])
        if s < 0:
            rows.append(dict(year=str(Y), rule_id="", definition="no rule had 30 prior bets",
                             n=0, wl="0-0", net_u=0.0, cum_net_u=num(cum, 3)))
            continue
        idx, sf = bets[s]
        keep = D.yr.values[idx] == Y
        x = bet_pnl(PF, PD, idx[keep], sf[keep])[0]
        if len(x) != wf["n"][0, i] or abs(x.sum() - wf["S"][0, i]) > 1e-9:
            fail(f"walk-forward {Y}: bets do not reproduce the summed path")
        cum += x.sum()
        roi, _, t = tstat(len(x), x.sum(), (x * x).sum())
        rows.append(dict(year=str(Y), rule_id=rules.id[s], definition=rules.definition[s],
                         n_tied=int(wf["tied"][0, i]), prior_n=int(wf["prior_n"][0, i]),
                         prior_roi_pct=num(wf["prior_roi"][0, i] * 100),
                         prior_t=num(wf["prior_t"][0, i]), n=len(x),
                         wl=f"{int((x > 0).sum())}-{int((x < 0).sum())}", net_u=num(x.sum(), 3),
                         roi_pct=num(roi * 100), t=num(t), cum_net_u=num(cum, 3)))
        bl.append(pd.DataFrame({"year": Y, "rule": rules.id[s], "pnl": x,
                                "exp": 1 / ov[idx[keep]] - 1}))
    B = pd.concat(bl, ignore_index=True)
    x = B.pnl.values
    roi, se, t = tstat(len(x), x.sum(), (x * x).sum())
    rows.append(dict(year=f"{WF_YEARS[0]}-{WF_YEARS[-1]}", rule_id="OOS total",
                     definition="all out-of-sample bets", n=len(x),
                     wl=f"{int((x > 0).sum())}-{int((x < 0).sum())}", net_u=num(x.sum(), 3),
                     roi_pct=num(roi * 100), t=num(t), cum_net_u=num(cum, 3)))
    W = pd.DataFrame(rows)
    for c in ("n_tied", "prior_n"):
        W[c] = W[c].astype("Int64")
    return W, B, dict(n=len(x), roi=float(roi), se=float(se), t=float(t),
                                       net=float(x.sum()), w=int((x > 0).sum()),
                                       l=int((x < 0).sum()))


# ---- charts -------------------------------------------------------------------

def wf_chart(B, path, title, subtitle):
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150, facecolor=fb.SURFACE)
    fb._style(ax)
    x = np.arange(1, len(B) + 1)
    cum, ce = B.pnl.cumsum().values, B.exp.cumsum().values
    ax.axhline(0, color=fb.INK, linewidth=1)
    starts = B.groupby("year", sort=False).head(1).index.values
    for s in starts[1:]:
        ax.axvline(s + 0.5, color=fb.GRID, linewidth=0.8, zorder=0)
    ax.plot(x, ce, color=fb.INK2, linewidth=1.5, linestyle=(0, (4, 3)),
            label="expected if the opening price is right (the vig)")
    ax.plot(x, cum, color=BLUE, linewidth=2, label="actual, out of sample")
    for yv, txt, col in ((cum[-1], f"{cum[-1]:+.1f}u", fb.INK), (ce[-1], f"{ce[-1]:+.1f}u", fb.INK2)):
        ax.annotate(txt, (x[-1], yv), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=col)
    ends = np.append(starts[1:], len(B))
    mids = (starts + ends) / 2 + 0.5
    g = B.groupby("year", sort=False)
    labels = [f"{Y}\n{r}\n{n}" for (Y, r), n in zip(g.rule.first().items(), g.size().values)]
    ax.set_xticks(mids, labels)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlim(0, len(B) * 1.06)
    ax.set_ylabel("cumulative units (1u flat at open)", color=fb.INK2, fontsize=9.5)
    ax.set_xlabel("year / rule picked from earlier years / bets that year (bets in date order)",
                  color=fb.INK2, fontsize=9)
    fig.text(0.06, 0.95, title, fontsize=13, color=fb.INK, weight="bold")
    fig.text(0.06, 0.912, subtitle, fontsize=9, color=fb.INK2)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], loc="upper left", bbox_to_anchor=(0.055, 0.895),
               frameon=False, fontsize=8.5, ncol=2, labelcolor=fb.INK)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.83, bottom=0.16)
    fig.savefig(path, facecolor=fb.SURFACE, metadata={"Software": None})
    plt.close(fig)


def null_chart(null_t, real_t, real_id, q95, pct, path, G):
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150, facecolor=fb.SURFACE)
    fb._style(ax)
    lo = np.floor(min(null_t.min(), real_t) * 10) / 10 - 0.1
    hi = np.ceil(max(null_t.max(), real_t) * 10) / 10 + 0.1
    ax.hist(null_t, bins=np.arange(lo, hi + 0.1, 0.1), color=BLUE, edgecolor=fb.SURFACE,
            linewidth=0.8)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.16)
    ax.set_xlim(lo, hi + 0.9)                        # room for the right-hand label
    ax.axvline(q95, color=fb.INK2, linewidth=1.2, linestyle=(0, (4, 2)))
    ax.axvline(real_t, color=fb.INK, linewidth=1.8)
    marks = sorted([(q95, f"null 95th percentile {q95:.2f}", fb.INK2),
                    (real_t, f"real best {real_t:.2f} ({real_id})\nnull percentile {pct:.1f}",
                     fb.INK)])
    for (xv, txt, col), (ha, dx) in zip(marks, (("right", -5), ("left", 5))):
        ax.annotate(txt, (xv, top * 1.1), xytext=(dx, 0), textcoords="offset points",
                    ha=ha, va="top", fontsize=8.5, color=col)
    ax.set_xlabel(f"largest pooled t among the {G} grid rules", color=fb.INK2, fontsize=9.5)
    ax.set_ylabel("replicates per 0.1 bin", color=fb.INK2, fontsize=9.5)
    fig.text(0.07, 0.94, "Best rule found by the search when no rule has an edge",
             fontsize=13, color=fb.INK, weight="bold")
    fig.text(0.07, 0.895, f"{len(null_t):,} null replicates: every winner redrawn from the "
             "vig-free opening price; prices, model and RD unchanged, 2008-2022",
             fontsize=9, color=fb.INK2)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.12)
    fig.savefig(path, facecolor=fb.SURFACE, metadata={"Software": None})
    plt.close(fig)


# ---- main ---------------------------------------------------------------------

def pct_below(null, real):
    return 100 * float(np.mean(null < real))


def main():
    fv_hist = pd.read_csv("data/fights_v2.csv")
    fv_hist = fv_hist[fv_hist.event_date < HI]
    assert fv_hist.event_date.max() < HI
    fv = fv_hist[fv_hist.event_date >= LO]
    ids = set(fv.fight_id)
    ev_year = fv.set_index("fight_id").event_date.str[:4].astype(int)
    J = pd.read_csv("data/bfo_joined.csv")
    J = J[J.fight_id.isin(ids)]
    rd = pd.read_csv(bs.PREFIGHT_RD)
    rd = rd[rd.fight_id.isin(ids)]
    M, rdiff = model_preds(ids, ev_year)
    D = fb.per_bout(fb.tag(build_frame(fv_hist, fv, J, rd, M)))
    ex_masks, keep = exclusion_masks(D)
    U, ex = fb.universe(D)
    if not (D.fight_id[keep].values == U.fight_id.values).all():
        fail("exclusion masks do not reproduce fade_buckets.universe()")
    in_u = keep
    assert D.event_date.max() < HI and U.event_date.max() < HI

    # Step 0: coverage, before anything else
    C = coverage(fv, J, M, D, U, ex_masks)
    print(f"window {LO} <= event_date < {HI}; seed {SEED}; {N_REPS} null replicates")
    print("\nSTEP 0 - coverage by year")
    print(C.to_string(index=False))
    tot = C.drop(columns=["year", "p_mod_source"]).sum()
    print("total  " + "  ".join(f"{k} {v}" for k, v in tot.items()))
    print(f"\nexclusions in order (fade_buckets.universe): " +
          ", ".join(f"{k} {v}" for k, v in ex.items()) +
          f"; draws {tot.draws_open} and NCs {tot.ncs_open} have no model probability")
    print(f"\nrefit vs {FROZEN} on the overlap years (tolerance max |dp| < {REFIT_TOL}):")
    for r in rdiff.itertuples(index=False):
        print(f"    {r.year}  n {r.n:3d}  max {r.max_dp:.4f}  mean {r.mean_dp:.5f}  "
              f"{'ok' if r.max_dp < REFIT_TOL else 'OVER'}")
    over = rdiff[rdiff.max_dp >= REFIT_TOL]
    print(f"    overall max {rdiff.max_dp.max():.4f}; "
          + (f"OVER the tolerance in {', '.join(map(str, over.year))}" if len(over)
             else "within the tolerance every year")
          + f"; refit used for {', '.join(map(str, sorted(M[M.src == 'refit'].year.unique())))}")

    b1 = int(C[C.year <= 2012].usable.sum())
    if b1 < B1_MIN:
        blocks = [("B12", 2008, 2017), ("B3", 2018, 2022)]
        print(f"\nB1 has {b1} usable bouts (< {B1_MIN}): merged into B2")
    else:
        blocks = [("B1", 2008, 2012), ("B2", 2013, 2017), ("B3", 2018, 2022)]
        print(f"\nblocks: " + ", ".join(f"{n} {lo}-{hi} ({int(C[(C.year >= lo) & (C.year <= hi)].usable.sum())} usable)"
                                     for n, lo, hi in blocks) + f"; B1 >= {B1_MIN}, not merged")

    # Step 1: rules
    rules = rule_list()
    G = int(rules.in_grid.sum())
    bets = [rule_bets(r, D, in_u) for r in rules.itertuples(index=False)]
    first_with = {}
    for rid, (idx, _) in zip(rules.id, bets):
        first_with.setdefault(idx.tobytes(), rid)
    rules["same_bets_as"] = [first_with[idx.tobytes()] if first_with[idx.tobytes()] != rid
                             else "" for rid, (idx, _) in zip(rules.id, bets)]
    n_sets = len({idx.tobytes() for idx, _ in bets[:G]})
    print(f"\nSTEP 1 - {G} grid rules (15 FAV + 90 DOG), plus STANDOUT and the GAP/FLIP "
          f"benchmarks = {len(rules)} rows in rules.csv; the {G} grid rules are {n_sets} "
          f"distinct bet sets (identical by construction: "
          + ", ".join(f"{r.id}={r.same_bets_as}" for r in rules.itertuples() if r.same_bets_as)
          + ")")

    # what the null assumes: winners at the vig-free open's probability
    fw = ((U.y_a == 1) == (U.a_open < U.b_open))
    print(f"\n[null premise] favourites won {int(fw.sum())} of {len(U)} usable bouts; "
          f"{U.p_mkt.sum():.1f} expected at the vig-free open (sd "
          f"{np.sqrt((U.p_mkt * (1 - U.p_mkt)).sum()):.1f})")
    band = pd.cut(U.p_mkt, [0.5, 0.567, 0.609, 0.653, 0.70, 1.0])
    cal = pd.DataFrame({"band": band, "fw": fw, "p": U.p_mkt}).groupby("band", observed=True)
    for b, g in cal:
        print(f"    p_mkt {b}: n {len(g):4d}  fav won {g.fw.mean():.3f}  expected {g.p.mean():.3f}")

    yidx = (D.yr - Y0).values
    fav_a = (D.a_open < D.b_open).values
    o_fav, o_dog, p_mkt = D.o_fav.values, D.o_dog.values, D.p_mkt.values
    ov = 1 / o_fav + 1 / o_dog
    fav_won = ((D.y_a == 1).values == fav_a)[None, :]
    PF, PD = pnl_matrices(fav_won, o_fav, o_dog)
    real = score(PF, PD, bets, yidx, blocks, G)

    fails = []
    ok, msg = check_vectorised(rules, bets, real, PF, PD)
    print(f"\n[internal] summed path == direct per-rule computation: {msg} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("summed path != direct computation")
    ok, msg = check_fade_frame(U)
    print(f"[sanity 1] 2018-22 bouts vs fade_buckets' per-bout frame: {msg} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("2018-22 bouts do not reproduce fade_buckets")
    ok, lines = check_sim(D, rules, bets, PF, PD)
    print("[sanity 2] GAP/FLIP 2012-2022, 1u at open: here vs betting_system.run() vs ledger")
    for ln in lines:
        print("    " + ln)
    if not ok:
        fails.append("GAP/FLIP do not reproduce betting_system's sim")
    if fails:
        fail("; ".join(fails) + " - nothing written")

    # Step 4: null
    rng = np.random.default_rng(SEED)
    fav_won0 = rng.random((N_REPS, len(D))) < p_mkt[None, :]
    PF0, PD0 = pnl_matrices(fav_won0, o_fav, o_dog)
    del fav_won0
    null = summary_stats(score(PF0, PD0, bets, yidx, blocks, G), G, blocks)
    del PF0, PD0
    rs = summary_stats(real, G, blocks)

    exp_roi = np.array([np.mean(1 / ov[idx] - 1) for idx, _ in bets[:G]])
    vig_u = float(np.mean(1 - 1 / ov[in_u.values]))
    null_grid = float(null["grid_mean_roi"].mean())
    ok = null_grid < 0 and abs(null_grid - exp_roi.mean()) < 0.0025
    print(f"[sanity 3] null mean pooled ROI over the {G} grid rules {null_grid:+.3%} vs "
          f"expected at the vig-free open {exp_roi.mean():+.3%} (mean 1/overround - 1 over "
          f"each rule's bets); universe average vig {vig_u:.3%} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fail("null mean ROI is not negative and near the vig - nothing written")

    q95 = float(np.quantile(null["max_t"], 0.95))
    T = rule_table(rules, real, blocks, D, bets, G, null["max_t"], q95)
    W, WB, wr = wf_table(rules, real["wf"], D, bets, PF, PD, ov)
    if abs(wr["roi"] - rs["wf_roi"][0]) > 1e-12:
        fail("walk-forward table != summed path")
    wf95 = float(np.quantile(null["wf_roi"], 0.95))

    ND = pd.DataFrame({"rep": np.arange(1, N_REPS + 1),
                       "max_t": np.round(null["max_t"], 4),
                       "max_t_rule": rules.id.values[null["max_t_rule"]],
                       "n_rules_pos_all_blocks": null["n_pos_all"],
                       "wf_n": null["wf_n"].astype(int), "wf_net_u": np.round(null["wf_net"], 4),
                       "wf_roi_pct": np.round(null["wf_roi"] * 100, 4),
                       "wf_t": np.round(null["wf_t"], 4),
                       "gap_roi_pct": np.round(null["gap_roi"] * 100, 4),
                       "flip_roi_pct": np.round(null["flip_roi"] * 100, 4),
                       "grid_mean_roi_pct": np.round(null["grid_mean_roi"] * 100, 4)})
    srows = []
    for name, key, scale in (("max pooled t (grid)", "max_t", 1),
                             ("rules positive in every block", "n_pos_all", 1),
                             ("walk-forward OOS ROI %", "wf_roi", 100),
                             ("GAP pooled ROI %", "gap_roi", 100),
                             ("FLIP pooled ROI %", "flip_roi", 100),
                             ("grid mean pooled ROI %", "grid_mean_roi", 100)):
        nv, rv = null[key] * scale, float(rs[key][0]) * scale
        srows.append(dict(metric=name, real=num(rv, 4), null_mean=num(nv.mean(), 4),
                          null_p50=num(np.quantile(nv, 0.5), 4),
                          null_p95=num(np.quantile(nv, 0.95), 4),
                          pct_null_below_real=num(pct_below(nv, rv), 1),
                          share_null_ge_real=num(np.mean(nv >= rv), 3)))
    S = pd.DataFrame(srows)

    # write
    C.to_csv(os.path.join(OUT, "coverage.csv"), index=False)
    rules.to_csv(os.path.join(OUT, "rules.csv"), index=False)
    T.to_csv(os.path.join(OUT, "rule_blocks.csv"), index=False)
    W.to_csv(os.path.join(OUT, "walkforward.csv"), index=False)
    ND.to_csv(os.path.join(OUT, "null_dist.csv"), index=False)
    S.to_csv(os.path.join(OUT, "null_summary.csv"), index=False)
    wf_chart(WB, os.path.join(OUT, "walkforward_pnl.png"),
             f"Walk-forward rule search, out of sample {WF_YEARS[0]}-{WF_YEARS[-1]}",
             f"Each year bets the grid rule with the best pooled t on earlier years "
             f"(>= {WF_MIN_PRIOR} prior bets) · {wr['n']} bets, {wr['w']}-{wr['l']}, "
             f"ROI {wr['roi']:+.1%}, t {wr['t']:+.2f} · null 95th percentile ROI {wf95:+.1%}")
    real_t, real_id = float(rs["max_t"][0]), rules.id[int(rs["max_t_rule"][0])]
    null_chart(null["max_t"], real_t, real_id, q95, pct_below(null["max_t"], real_t),
               os.path.join(OUT, "null_max_t.png"), G)

    # report
    grid = T[T.family.isin(["FAV", "DOG"])]
    bl = [n for n, _, _ in blocks]
    print(f"\nSTEP 2 - top 10 grid rules by pooled t (null max-t 95th percentile {q95:.2f})")
    top = grid.sort_values(["pooled_t", "id"], ascending=[False, True]).head(10)
    cols = (["id", "pooled_n", "pooled_roi_pct", "pooled_t"] + [f"{b}_roi_pct" for b in bl]
            + [f"{b}_n" for b in bl] + ["blocks_pos", "null_pct_max_t", "advances"])
    print(top[cols].to_string(index=False))
    for r in top.itertuples(index=False):
        print(f"    {r.id}: {r.definition}")
    print(f"\npass bar: {int(grid.bar1_all_blocks_pos.sum())} rules positive in every block, "
          f"{int(grid.bar2_t_above_null95.sum())} with pooled t > {q95:.2f}, "
          f"{int(grid.bar3_min30_each_block.sum())} with >= {MIN_BLOCK_N} bets in every block "
          f"-> {int(grid.advances.sum())} of {G} rules advance")
    adv = grid[grid.advances == 1].merge(rules[["id", "same_bets_as"]], on="id")
    for r in adv.itertuples(index=False):
        print(f"    {r.id}: {r.definition}; pooled n {r.pooled_n} ROI {r.pooled_roi_pct:+.2f}% "
              f"t {r.pooled_t:+.2f}; " + ", ".join(f"{b} {getattr(r, f'{b}_roi_pct'):+.2f}% "
                                                   f"(n {getattr(r, f'{b}_n')})" for b in bl)
              + (f"; same bets as {r.same_bets_as}" if r.same_bets_as else ""))
    so = T[T.id == "STANDOUT"].iloc[0]
    sn, sS, sQ, sW = real["soft"]
    k = int(np.flatnonzero(rules.id.values == "STANDOUT")[0])
    sroi, sse, st = (float(v) for v in tstat(sn[k], sS[0, k], sQ[0, k]))
    s_pass = sroi > 0 and st >= STANDOUT_T
    print(f"\nSTANDOUT ({so.definition})")
    print(f"    fresh test B1+B2 (2008-17): n {int(sn[k])}  {int(sW[0, k])}-{int(sn[k] - sW[0, k])}  "
          f"ROI {sroi:+.2%}  SE {sse:.2%}  t {st:+.2f}  -> "
          f"{'ADVANCES' if s_pass else 'does not advance'} (bar: ROI > 0 and t >= {STANDOUT_T})")
    print("    by block: " + "; ".join(f"{b} n {so[f'{b}_n']} {so[f'{b}_wl']} ROI {so[f'{b}_roi_pct']:+.2f}%"
                                      for b in bl) + "  (B3 = where it was found)")
    print("\nSTEP 3 - walk-forward selections")
    print(W[["year", "rule_id", "n_tied", "prior_n", "prior_t", "n", "wl", "net_u", "roi_pct",
             "cum_net_u"]].to_string(index=False))
    print(f"    OOS {wr['n']} bets {wr['w']}-{wr['l']} net {wr['net']:+.2f}u ROI {wr['roi']:+.2%} "
          f"SE {wr['se']:.2%} t {wr['t']:+.2f}; null 95th percentile ROI {wf95:+.2%}; "
          f"procedure {'WORKS' if wr['roi'] > wf95 else 'does not beat the null'}")
    print("\nSTEP 4 - real values against the null")
    print(S.to_string(index=False))
    for rl in ("GAP", "FLIP"):
        r = T[T.id == rl].iloc[0]
        print(f"    {rl}: n {r.pooled_n} {r.pooled_wl} ROI {r.pooled_roi_pct:+.2f}% t {r.pooled_t:+.2f}; "
              + "; ".join(f"{b} n {r[f'{b}_n']} ROI {r[f'{b}_roi_pct']:+.2f}%" for b in bl))
    print(f"\nwrote {', '.join(sorted(f for f in os.listdir(OUT) if f.endswith(('.csv', '.png'))))}")


if __name__ == "__main__":
    main()
