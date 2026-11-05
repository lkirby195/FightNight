"""Fade-bucket research, 2018-2022. Read-only.

Question (a reviewer's proposal): when the model is much less bullish than the
market on the market favourite (model 52% / market 73%), does betting the dog
pay? That quadrant is the mirror image of GAP and was never tested.

Every bout with a two-sided opening price and a model probability is put in a
gap bucket (gap = model minus vig-free market on the opening favourite, in
probability points) and bet 1u flat on the model side at its opening price.
Results by bucket x RD tier, the same for the "model agrees" subset
(p_mod >= 0.50), one pre-registered fade cell, and the gap histogram.

Inputs are the ledger's: the frozen model-only preds (walk-forward, checked
below), the validated opens and the closes in data/bfo_joined.csv, the
pre-fight RDs of placement policy v1 (data/prefight_rd.csv) and the debut rule
from fights_v2. GAP/FLIP tags come from betting_system._decision, i.e.
signals.classify. Nothing outside research/fade_buckets/ is written, no
network is used (sockets are blocked below), and no row dated 2023-01-01 or
later is loaded from any results file: 2023-24 is reserved to confirm a
pre-registered rule and 2025 stays sealed.

Run: python research/fade_buckets/fade_buckets.py   (any cwd; ~30 s)
Outputs: bucket_table.csv, bucket_roi.png, bucket_table_agree.csv,
bucket_roi_agree.png, fade_cell.csv, gap_hist.png.
"""
from __future__ import annotations

import os
import socket
import sys


def _no_network(*a, **k):
    raise RuntimeError("fade_buckets: network access is not allowed")


socket.socket.connect = _no_network

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
from card_settle import OR_HI, OR_LO                 # noqa: E402  clean-book bounds

LO, HI = "2018-01-01", "2023-01-01"                  # window: LO <= event_date < HI
FROZEN = "data/model_only_preds_frozen_2026-09-08.csv"
PREDS = FROZEN if os.path.exists(FROZEN) else "data/model_only_preds.csv"
OUT = HERE

RD_KNOWN = signals.UNKNOWN_RD                        # 160: RD <= this is "known"
TIERS = ["KK", "KU", "UU", "ALL"]
BUCKETS = ["<=-25", "(-25,-20]", "(-20,-15]", "(-15,-10]", "(-10,-5]", "(-5,0)",
           "[0,5)", "[5,10)", "[10,15)", ">=15"]


def fail(msg):
    raise SystemExit(f"STOP: {msg}")


def window_ids(fv):
    return set(fv[(fv.event_date >= LO) & (fv.event_date < HI)].fight_id)


# ---- load (window filter at read time) -------------------------------------

def load():
    """betting_system.load()'s frame for the window only: preds x bfo_joined x
    fights_v2 x prefight_rd, with the debut-rule flags. fights_v2 is cut at HI
    on read (earlier years stay for the debut rule's history); every other
    file is cut to the window's fight_ids on read."""
    fv_hist = pd.read_csv("data/fights_v2.csv")
    fv_hist = fv_hist[fv_hist.event_date < HI]
    ids = window_ids(fv_hist)
    M = pd.read_csv(PREDS)
    M = M[M.fight_id.isin(ids)]
    J = pd.read_csv("data/bfo_joined.csv")
    J = J[J.fight_id.isin(ids)]
    rd = pd.read_csv(bs.PREFIGHT_RD)
    rd = rd[rd.fight_id.isin(ids)]
    fv = fv_hist[fv_hist.fight_id.isin(ids)]
    assert fv_hist.event_date.max() < HI and fv.event_date.min() >= LO
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
    D = D.sort_values(["event_date", "bout_order"]).reset_index(drop=True)
    assert D.event_date.max() < HI, D.event_date.max()
    assert D.event_date.min() >= LO, D.event_date.min()
    return D, fv, J


# ---- check 0: the window's probabilities are walk-forward ------------------

def walk_forward_check():
    """Refit Stage 2 exactly as clv_eval.gen_model_preds does (year Y fit on
    fights dated before Y-01-01, predict Y) from features cut at HI, and
    compare to PREDS. An in-sample fit (one fit on everything before HI) is
    the contrast. The refit cannot equal the file to the last digit because
    build_matrix centres two interaction terms and median-fills over the
    frame it is given; the gap to the in-sample fit is what matters."""
    df = pd.read_csv("data/features_v2.csv", parse_dates=["date"])
    df = df[df.date < HI]
    assert df.date.max() < pd.Timestamp(HI)
    df = df[df.label != 0.5].copy()
    X = stage2.build_matrix(df).drop(columns=clv_eval.PASSC)
    y = df.label.astype(int).values
    fl = df.fight_id.map(features.flip).values

    def fit(mask):
        m = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(C=0.3, max_iter=3000, fit_intercept=False))
        m.fit(np.vstack([X[mask].values, -X[mask].values]),
              np.concatenate([y[mask], 1 - y[mask]]))
        return m

    def p_a(m, mask):
        pm = m.predict_proba(X[mask].values)[:, 1]
        return pd.Series(np.where(fl[mask], 1 - pm, pm), index=df.fight_id[mask].values)

    wf = []
    for Y in range(int(LO[:4]), int(HI[:4])):
        ev = ((df.date >= f"{Y}-01-01") & (df.date < f"{Y + 1}-01-01")).values
        wf.append(p_a(fit((df.date < f"{Y}-01-01").values), ev))
    wf = pd.concat(wf)
    ev = (df.date >= LO).values
    ins = p_a(fit((df.date < HI).values), ev)
    M = pd.read_csv(PREDS)
    M = M[M.fight_id.isin(wf.index)].set_index("fight_id")
    if len(M) != len(wf):
        fail(f"walk-forward check: {PREDS} has {len(M)} of {len(wf)} window bouts")
    if not (M.year == pd.to_datetime(df.set_index("fight_id").date[M.index]).dt.year).all():
        fail(f"walk-forward check: {PREDS} year column is not the event year")
    d_wf = (M.p_model_a - wf[M.index]).abs()
    d_in = (M.p_model_a - ins[M.index]).abs()
    print(f"[check 0] walk-forward: {PREDS} vs a Stage 2 refit per year (fit < Y-01-01, "
          f"features cut at {HI}), n={len(M)}")
    print(f"          walk-forward refit |dp| max {d_wf.max():.4f} mean {d_wf.mean():.5f}; "
          f"in-sample fit (< {HI}) |dp| max {d_in.max():.4f} mean {d_in.mean():.5f}")
    if not (d_wf.max() < 0.01 and d_wf.mean() < d_in.mean() / 5):
        fail("the window's probabilities do not reproduce as walk-forward predictions")
    print("          PASS: the probabilities are the walk-forward (out-of-sample) fits")
    return dict(n=len(M), wf_max=d_wf.max(), wf_mean=d_wf.mean(),
                in_max=d_in.max(), in_mean=d_in.mean())


# ---- per-bout frame ---------------------------------------------------------

def vigfree(o_self, o_other):
    return (1 / o_self) / (1 / o_self + 1 / o_other)


def bucket_of(g):
    """Index into BUCKETS for an array of gaps (pp); asserts exactly one hit."""
    conds = [g <= -25, (g > -25) & (g <= -20), (g > -20) & (g <= -15),
             (g > -15) & (g <= -10), (g > -10) & (g <= -5), (g > -5) & (g < 0),
             (g >= 0) & (g < 5), (g >= 5) & (g < 10), (g >= 10) & (g < 15), g >= 15]
    hits = np.vstack(conds).astype(int)
    if not (hits.sum(axis=0) == 1).all():
        fail("a bout lands in zero or several gap buckets")
    return hits.argmax(axis=0)


def tag(D):
    """GAP / FLIP / VOID tag and the picked side (1 = fighter_a) from the
    shared classifier, for every bout of the window frame."""
    rule, side = [], []
    for r in D.itertuples(index=False):
        d = bs._decision(r)                  # signals.classify, fighter_a as side 1
        rule.append("VOID" if d.void else (d.rule or ""))
        side.append(d.side if d.rule else 0)
    D = D.copy()
    D["tag"], D["tag_side"] = rule, side
    return D


def per_bout(D):
    """Add the bucket fields to every bout (excluded bouts keep NaN gaps)."""
    D = D.copy()
    fav_a = D.a_open < D.b_open
    D["pickem"] = D.a_open == D.b_open
    ov = 1 / D.a_open + 1 / D.b_open
    D["prefilter"] = (ov >= signals.OVERROUND_LO) & (ov <= signals.OVERROUND_HI)
    D["o_fav"] = np.where(fav_a, D.a_open, D.b_open)
    D["o_dog"] = np.where(fav_a, D.b_open, D.a_open)
    D["p_mkt"] = vigfree(D.o_fav, D.o_dog)
    D["p_mod"] = np.where(fav_a, D.p_model_a, 1 - D.p_model_a)
    D["gap"] = (D.p_mod - D.p_mkt) * 100
    D["rd_fav"] = np.where(fav_a, D.rd_a, D.rd_b)
    D["rd_dog"] = np.where(fav_a, D.rd_b, D.rd_a)
    unk = (D.rd_a > RD_KNOWN).astype(int) + (D.rd_b > RD_KNOWN).astype(int)
    D["tier"] = np.array(["KK", "KU", "UU"])[unk]
    D["dog_am"] = [int(round(signals.american(o))) for o in D.o_dog]
    # the model side: favourite when the model is above the market, dog below
    side_fav = D.gap > 0
    D["model_side"] = np.where(side_fav, "fav", "dog")
    D["o_side"] = np.where(side_fav, D.o_fav, D.o_dog)
    fav_won = (D.y_a == 1) == fav_a
    D["won"] = np.where(side_fav, fav_won, ~fav_won)
    D["pnl"] = np.where(D.won, D.o_side - 1, -1.0)
    D["side_is_a"] = np.where(side_fav, fav_a, ~fav_a)
    # CLV on the model side: vig-free close minus open, in pp. Close = the
    # clean close where card_settle wrote one, else the raw last tick when it
    # is a clean book (the ledger's _clean_close rule); otherwise none (the
    # ledger would re-read the BFO series; this script does no network).
    ca = D.a_close_clean.where(D.a_close_clean.notna(), D.a_close)
    cb = D.b_close_clean.where(D.b_close_clean.notna(), D.b_close)
    clean = D.a_close_clean.notna() | ((1 / ca + 1 / cb >= OR_LO) & (1 / ca + 1 / cb <= OR_HI))
    qo = np.where(D.side_is_a, vigfree(D.a_open, D.b_open), vigfree(D.b_open, D.a_open))
    qc = np.where(D.side_is_a, vigfree(ca, cb), vigfree(cb, ca))
    D["clv_pp"] = np.where(clean, (qc - qo) * 100, np.nan)
    return D


def universe(D):
    """The bucketed bouts and the exclusion counts (in the order applied)."""
    ex = {}
    U = D
    for name, keep in (("debut (debut rule: no projection)", lambda U: U.projectable),
                       ("odds-sanity pre-filter fails", lambda U: U.prefilter),
                       ("exact pick'em at open", lambda U: ~U.pickem),
                       ("gap exactly 0 (no model side)", lambda U: U.gap != 0)):
        k = keep(U)
        ex[name] = int((~k).sum())
        U = U[k]
    U = U.copy()
    U["bucket"] = np.array(BUCKETS)[bucket_of(U.gap.values)]
    return U, ex


# ---- stats ------------------------------------------------------------------

def stats(g):
    n = len(g)
    w, l = int(g.won.sum()), int((~g.won.astype(bool)).sum())
    r = g.pnl
    roi = r.mean() if n else np.nan
    se = r.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    mdec = g.o_side.mean() if n else np.nan
    return dict(n=n, wl=f"{w}-{l}", w=w, l=l, net_u=round(r.sum(), 3),
                roi_pct=round(roi * 100, 2) if n else np.nan,
                roi_se_pct=round(se * 100, 2) if n > 1 else np.nan,
                t=round(roi / se, 2) if n > 1 and se > 0 else np.nan,
                mean_open_dec=round(mdec, 3) if n else np.nan,
                mean_open_us=int(round(signals.american(mdec))) if n else np.nan,
                clv_n=int(g.clv_pp.notna().sum()),
                mean_clv_pp=round(g.clv_pp.mean(), 2) if g.clv_pp.notna().any() else np.nan,
                gap_tagged=int((g.tag == "GAP").sum()),
                flip_tagged=int((g.tag == "FLIP").sum()))


def bucket_table(U):
    rows = []
    for b in BUCKETS:
        side = "dog" if b in BUCKETS[:6] else "fav"
        for t in TIERS:
            g = U[U.bucket == b] if t == "ALL" else U[(U.bucket == b) & (U.tier == t)]
            rows.append(dict(bucket=b, tier=t, model_side=side, **stats(g)))
    T = pd.DataFrame(rows)
    T["mean_open_us"] = T.mean_open_us.astype("Int64")
    # partition check: bucket n's add up per tier, tiers add up to ALL
    for t in TIERS:
        tot = len(U) if t == "ALL" else int((U.tier == t).sum())
        if T[T.tier == t].n.sum() != tot:
            fail(f"bucket n's for tier {t} sum to {T[T.tier == t].n.sum()}, not {tot}")
    by_b = T[T.tier != "ALL"].groupby("bucket", sort=False).n.sum()
    if not (by_b.values == T[T.tier == "ALL"].n.values).all():
        fail("KK + KU + UU != ALL in some bucket")
    return T


# ---- charts -----------------------------------------------------------------

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"ALL": "#3d3c39", "KK": "#2a78d6", "KU": "#eb6834", "UU": "#1baf7a"}
LABELS = {"ALL": "ALL", "KK": "KK  both RD ≤ 160", "KU": "KU  one RD > 160",
          "UU": "UU  both RD > 160"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def roi_chart(T, path, title, subtitle, footnote=""):
    fig, ax = plt.subplots(figsize=(12, 6.6), dpi=150, facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(BUCKETS))
    order = ["ALL", "KK", "KU", "UU"]
    off = dict(zip(order, (-0.27, -0.09, 0.09, 0.27)))
    # y range from the data (+-1 SE), never past -100 / +150; longer bars are clipped
    se0 = T.roi_se_pct.fillna(0)
    lo = max(-100.0, np.floor(((T.roi_pct - se0).min() - 5) / 10) * 10)
    hi = min(150.0, np.ceil(((T.roi_pct + se0).max() + 5) / 10) * 10)
    for t in order:
        g = T[T.tier == t].set_index("bucket").reindex(BUCKETS)
        xs, ys, se = x + off[t], g.roi_pct.values, g.roi_se_pct.values
        ok = g.n.values > 0
        ax.plot(xs[ok], ys[ok], color=COLORS[t], linewidth=2 if t != "ALL" else 2.4,
                marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=LABELS[t], zorder=3 if t == "ALL" else 2)
        for xi, yi, si, ni in zip(xs, ys, se, g.n.values):
            if ni == 0:
                continue
            if not np.isnan(si):
                a, b = max(yi - si, lo), min(yi + si, hi)
                ax.plot([xi, xi], [a, b], color=COLORS[t], linewidth=1, alpha=0.55, zorder=1)
                if yi + si > hi:
                    ax.plot(xi, hi, marker="^", markersize=4, color=COLORS[t], alpha=0.8)
                if yi - si < lo:
                    ax.plot(xi, lo, marker="v", markersize=4, color=COLORS[t], alpha=0.8)
            ax.annotate(f"{ni}", (xi, min(max(yi, lo), hi)), xytext=(0, -11),
                        textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
    ax.axhline(0, color=INK, linewidth=1)
    head = (hi - lo) * 0.09                  # headroom for the two region labels
    ax.axvline(5.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(5.35, hi + head * 0.6, "← model below market: bet the dog", ha="right",
            va="center", fontsize=8.5, color=INK2)
    ax.text(5.65, hi + head * 0.6, "model above market: bet the favorite →", ha="left",
            va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, BUCKETS)
    ax.set_xlim(-0.6, len(BUCKETS) - 0.4)
    ax.set_ylim(lo - (hi - lo) * 0.04, hi + head)
    ax.set_ylabel("ROI on the model side (%)", color=INK2, fontsize=9.5)
    ax.set_xlabel("gap = model minus vig-free market, P(opening favorite), pp",
                  color=INK2, fontsize=9.5)
    fig.text(0.055, 0.955, title, fontsize=13, color=INK, weight="bold")
    fig.text(0.055, 0.918, subtitle, fontsize=9, color=INK2)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.05, 0.9), frameon=False,
               fontsize=8.5, ncol=4, labelcolor=INK)
    if footnote:
        fig.text(0.985, 0.012, footnote, ha="right", va="bottom", fontsize=8, color=INK2)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.1)
    fig.savefig(path, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def gap_hist(U, path, mean, median):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=SURFACE)
    _style(ax)
    g = U.gap.values
    bins = np.arange(np.floor(g.min()), np.ceil(g.max()) + 1, 1.0)
    ax.hist(g, bins=bins, color=COLORS["KK"], edgecolor=SURFACE, linewidth=0.6)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.06)
    ax.axvline(0, color=INK, linewidth=1)
    ax.axvline(mean, color=INK, linewidth=1.2, linestyle=(0, (4, 2)),
               label=f"mean {mean:+.2f} pp")
    ax.axvline(median, color=INK2, linewidth=1.2, linestyle=(0, (1, 2)),
               label=f"median {median:+.2f} pp")
    ax.text(0.5, top * 1.03, "model = market", ha="left", va="top", fontsize=8, color=INK2)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax.set_xlabel("gap = model minus vig-free market, P(opening favorite), pp",
                  color=INK2, fontsize=9.5)
    ax.set_ylabel("bouts per 1-pp bin", color=INK2, fontsize=9.5)
    neg = (g < 0).mean()
    fig.text(0.07, 0.94, "How far the model sits from the market on the favorite, 2018-2022",
             fontsize=13, color=INK, weight="bold")
    fig.text(0.07, 0.895, f"n = {len(g):,} bouts; the model is below the market on the "
             f"favorite in {neg:.0%} of them", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.12)
    fig.savefig(path, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


# ---- main -------------------------------------------------------------------

def p_at_most(w, probs):
    """P(at most w wins) for independent bets with win probabilities `probs`
    (Poisson-binomial CDF, exact by convolution)."""
    dist = np.array([1.0])
    for q in probs:
        dist = np.convolve(dist, [1 - q, q])
    return float(dist[:w + 1].sum())


def show(T, tiers):
    cols = ["bucket", "tier", "model_side", "n", "wl", "net_u", "roi_pct", "roi_se_pct",
            "t", "mean_open_dec", "mean_open_us", "clv_n", "mean_clv_pp",
            "gap_tagged", "flip_tagged"]
    print(T[T.tier.isin(tiers)][cols].to_string(index=False))


def sim_stats(B):
    out = {}
    for rl, g in B.groupby("rule"):
        out[rl] = (len(g), int((g.pnl_at_open > 0).sum()), int((g.pnl_at_open < 0).sum()),
                   round(g.pnl_at_open.sum(), 6))
    return out


def main():
    print(f"window {LO} <= event_date < {HI}; model probability file: {PREDS}")
    wf = walk_forward_check()

    D, fv, J = load()
    fails = []

    # accounting from the window's bouts to the bucketed universe
    n_fv = len(fv)
    jw = J.merge(fv[["fight_id", "outcome"]], on="fight_id")
    jw = jw[jw.a_open.notna() & jw.b_open.notna()]
    draws = int((jw.outcome == "DRAW").sum())
    ncs = int((jw.outcome == "NC").sum())
    no_pred_other = len(jw) - draws - ncs - len(D)
    D = per_bout(tag(D))
    U, ex = universe(D)

    print(f"\naccounting: {n_fv} window bouts in fights_v2; {len(jw)} with a two-sided "
          f"BFO open; excluded {draws} draws + {ncs} NCs (no model probability: stage2 "
          f"drops draws, features never emits an NC) -> {len(D)} with a model probability")
    if no_pred_other != 0:
        fails.append(f"{no_pred_other} non-draw/NC bouts with an open have no model probability")
    for k, v in ex.items():
        print(f"    excluded {v:4d}  {k}")
    print(f"    -> {len(U)} bouts bucketed")

    # sanity check 1: GAP/FLIP tags reproduce betting_system's sim
    B = bs.run(D)
    Bc = bs.counted(B)
    if (B.live != 0).any():
        fails.append("a 2018-2022 row came out live=1")
    T_ = D[D.tag.isin(["GAP", "FLIP"])]
    mine = {}
    for rl, g in T_.groupby("tag"):
        in_u = g[g.fight_id.isin(U.fight_id)]
        u = U.set_index("fight_id").loc[in_u.fight_id]
        if not ((u.side_is_a.values) == (in_u.tag_side.values == 1)).all():
            fails.append(f"{rl}: the classifier's pick differs from the model side in a bucket")
        out_u = g[~g.fight_id.isin(U.fight_id)]
        o = np.where(out_u.tag_side == 1, out_u.a_open, out_u.b_open)
        w_out = (out_u.y_a == 1).values == (out_u.tag_side == 1).values
        pnl = np.concatenate([u.pnl.values, np.where(w_out, o - 1, -1.0)])
        mine[rl] = (len(g), int((pnl > 0).sum()), int((pnl < 0).sum()), round(pnl.sum(), 6),
                    len(out_u))
    sim = sim_stats(Bc)
    L = pd.read_csv(bs.LEDGER)
    L = L[(L.event_date >= LO) & (L.event_date < HI) & (L.units > 0)]
    led = {rl: (len(g), int((g.result == "WIN").sum()), int((g.result == "LOSS").sum()),
                round(g.pnl_at_open.sum(), 6)) for rl, g in L.groupby("rule")}
    n_void = int((D.tag == "VOID").sum())
    print("\n[check 1] GAP/FLIP tags vs betting_system.run() on the same frame "
          "(1u at open), and vs the committed ledger")
    for rl in ("FLIP", "GAP"):
        m, s, lg = mine.get(rl), sim.get(rl), led.get(rl)
        roi = lambda v: f"{v[3] / v[0]:+.2%}" if v else "-"
        print(f"    {rl:4s} tags {m[0]} {m[1]}-{m[2]} net {m[3]:+.2f}u ROI {roi(m)} "
              f"({m[4]} of them pick'ems, outside the buckets) | sim {s[0]} {s[1]}-{s[2]} "
              f"net {s[3]:+.2f}u ROI {roi(s)} | ledger {lg[0]} {lg[1]}-{lg[2]} "
              f"net {lg[3]:+.2f}u")
        if m[:3] != s[:3] or abs(m[3] - s[3]) > 1e-9:
            fails.append(f"{rl}: tags {m[:4]} != sim {s}")
        # the ledger stores pnl_at_open rounded to 4 dp
        if lg[:3] != s[:3] or abs(lg[3] - s[3]) > 1e-4 * s[0]:
            fails.append(f"{rl}: ledger {lg} != sim {s} (preds file differs from the ledger's)")
    if set(T_.fight_id) != set(Bc.fight_id):
        fails.append("tagged fight_ids != sim fight_ids")
    print(f"    VOID tags: {n_void} (sim VOID rows: {int((B.u == 0).sum())})")
    # CLV column vs the ledger's clv_pts (same side, same clean close; 2 dp there)
    cx = U[U.tag.isin(["GAP", "FLIP"])].merge(L[["fight_id", "clv_pts"]], on="fight_id")
    dclv = (cx.clv_pp - cx.clv_pts).abs()
    print(f"    CLV on the {len(cx)} bucketed tagged bouts vs ledger clv_pts: "
          f"{int(cx.clv_pp.notna().sum())} with a CLV, max |diff| {dclv.max():.4f} pp")
    if cx.clv_pp.isna().any() or not dclv.max() <= 0.006:
        fails.append("CLV on tagged bouts does not match the ledger's clv_pts")

    # tables (check 2 runs inside bucket_table)
    T = bucket_table(U)
    A = U[U.p_mod >= 0.50]
    TA = bucket_table(A)
    if TA.flip_tagged.sum() != 0:
        fails.append("a FLIP-tagged bout is in the model-agrees subset")
    print(f"\n[check 2] partition: every bout in exactly one bucket; per-tier bucket n's sum to "
          f"the tier; KK+KU+UU = ALL; ALL = {len(U)} = {len(D)} - "
          f"{sum(ex.values())} excluded (agree subset: {len(A)})")
    print(f"    tiers: " + ", ".join(f"{t} {int((U.tier == t).sum())}" for t in TIERS[:3]))
    tagged_u = int(U.tag.isin(["GAP", "FLIP"]).sum())
    pick_tagged = sum(v[4] for v in mine.values())
    if T[T.tier == "ALL"][["gap_tagged", "flip_tagged"]].sum().sum() + pick_tagged != len(Bc):
        fails.append("bucketed tags + pick'em tags != sim bets")
    print(f"    tagged bouts bucketed {tagged_u} + pick'em {pick_tagged} = sim {len(Bc)}")

    if fails:
        for f in fails:
            print("FAIL:", f)
        fail("sanity check failed; nothing written")
    print("    PASS (checks 1 and 2)")

    # pre-registered fade cell (defined in the brief before any look)
    base = (U.p_mod >= 0.45) & (U.p_mod <= 0.60) & (U.p_mkt >= 0.70)
    rdk = (U.rd_a <= RD_KNOWN) & (U.rd_b <= RD_KNOWN)
    cap = U.dog_am <= signals.SKIP_LINE_MAX
    cells = [("pre-registered", "p_mod in [0.45,0.60], p_mkt >= 0.70, both RD <= 160, "
              "dog open <= +250", base & rdk & cap),
             ("(a) no RD filter", "p_mod in [0.45,0.60], p_mkt >= 0.70, dog open <= +250",
              base & cap),
             ("(b) no price cap", "p_mod in [0.45,0.60], p_mkt >= 0.70, both RD <= 160",
              base & rdk)]
    rows = []
    for name, defn, m in cells:
        g = U[m]
        if not (g.model_side == "dog").all():
            fail(f"fade cell {name}: a bout's model side is not the dog")
        s = stats(g)
        # with zero wins the sample SE is 0 and t undefined; these say how
        # surprising the W-L is: wins expected at the vig-free open and at the
        # model's price, and P(this many wins or fewer) if the open were right
        s.update(mkt_exp_w=round((1 - g.p_mkt).sum(), 2),
                 mod_exp_w=round((1 - g.p_mod).sum(), 2),
                 p_mkt_le_w=round(p_at_most(s["w"], 1 - g.p_mkt.values), 4))
        rows.append(dict(cell=name, definition=defn, **s))
    F = pd.DataFrame(rows)
    F["mean_open_us"] = F.mean_open_us.astype("Int64")

    gm, gmed = U.gap.mean(), U.gap.median()

    # write
    T.to_csv(os.path.join(OUT, "bucket_table.csv"), index=False, lineterminator="\n")
    TA.to_csv(os.path.join(OUT, "bucket_table_agree.csv"), index=False, lineterminator="\n")
    F.to_csv(os.path.join(OUT, "fade_cell.csv"), index=False, lineterminator="\n")
    sub = ("1u flat on the model side at its opening price · bars ±1 SE "
           "(arrow = clipped at the axis) · number under each point = bouts")
    roi_chart(T, os.path.join(OUT, "bucket_roi.png"),
              f"ROI of betting the model side, by gap bucket, 2018-2022 (n = {len(U):,})", sub)
    roi_chart(TA, os.path.join(OUT, "bucket_roi_agree.png"),
              f"Model still picks the favorite (p_mod ≥ 0.50), 2018-2022 (n = {len(A):,})",
              sub, footnote="Favorite-side buckets are the same bouts as the full table: "
              "gap > 0 implies p_mod > p_mkt > 0.50.")
    gap_hist(U, os.path.join(OUT, "gap_hist.png"), gm, gmed)

    print("\nbucket_table (all tiers)")
    show(T, TIERS)
    print("\nbucket_table_agree (all tiers)")
    show(TA, TIERS)
    print("\npooled by model side (ALL tiers)")
    for name, X_ in (("full", U), ("agree", A)):
        for side in ("dog", "fav"):
            s = stats(X_[X_.model_side == side])
            print(f"    {name:5s} {side}: n {s['n']}  {s['wl']}  net {s['net_u']:+.2f}u  "
                  f"ROI {s['roi_pct']:+.2f}%  SE {s['roi_se_pct']:.2f}  t {s['t']:+.2f}  "
                  f"CLV {s['mean_clv_pp']:+.2f} pp")
    print("\nfade_cell")
    for _, r in F.iterrows():
        print(f"    {r.cell}: {r.definition}")
    print(F.drop(columns="definition").to_string(index=False))
    print(f"\ngap over {len(U)} bouts: mean {gm:+.2f} pp, median {gmed:+.2f} pp, "
          f"sd {U.gap.std():.2f} pp; model below market on the favorite in "
          f"{(U.gap < 0).mean():.1%}")
    print(f"\nwrote {', '.join(sorted(f for f in os.listdir(OUT) if not f.endswith('.py')))}")


if __name__ == "__main__":
    main()
