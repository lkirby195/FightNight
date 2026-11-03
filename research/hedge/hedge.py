"""Hedge study, 2012-2022, on the frozen signals. Read-only.

Question: the model's proven edge is line movement (CLV), while its profit
column is noisy. If a signal is bet at the open and hedged at the close when
the line has moved our way, does CLV become locked profit, and what does that
cost in total P&L? This is an execution study. The bets are the frozen GAP and
FLIP signals (signals.classify, through betting_system._decision) and
placement policy v1, unchanged; only what happens at the close varies.

Bouts, prices and the universe are rule_search's: its model_preds() and
build_frame() over 2012-2022, with fade_buckets' tag(), per_bout() and
universe() on top (imported and called, not copied). Opens are the validated
opens betting_system settles at: every bet in the window is backfilled
(live=0), so it settles at open. Closes are the clean two-sided close in
bfo_joined.csv: a/b_close_clean where card_settle wrote one, else the raw
close when it is already a clean book (overround in [OR_LO, OR_HI]). That is
the ledger's own _clean_close rule and fade_buckets' CLV rule. No window row
has a clean-close column value (card_settle's backfill covers 2026), so in
practice it is the raw close wherever that is a clean book. A bet without a
clean close is never hedged.

Lock math. Stake s on side X at open decimal d_o; the hedge on the other side
Y is struck at decimal c. A full lock hedges h = s*d_o/c on Y and locks
L = s*(d_o - 1) - h whichever side wins; L > 0 iff 1/d_o + 1/c < 1.
Policies (pre-registered): P0 never hedge; P1 full lock when L >= 0; P2a/b/c
full lock when L >= 0.02s / 0.05s / 0.10s; P3 always full hedge, even at a
negative L; P4 half hedge (h/2) when L >= 0. Hedge-leg execution: a book at
the BFO close (the best price across books) with 0, 1 or 2 pp added to Y's
implied probability; or, HYPOTHETICALLY, an exchange at Y's vig-free close
with commission k on the hedge leg's winnings, i.e. an effective price of
1 + (c_fair - 1)(1 - k). The hedge decision is taken at the executable price.

Window 2012-01-01 <= event_date < 2023-01-01, the committed sim's. fights_v2
is cut at 2023-01-01 on read, every other results file to the window's
fight_ids (the ledger to the window's dates); nothing dated 2023 or later is
printed, plotted or used. Sockets are blocked. No random draws are made, so
two runs are byte-identical without a seed. Nothing outside research/hedge/
is written.

Run: python research/hedge/hedge.py   (any cwd; about 10 s)
Outputs: hedge_results.csv, hedge_bets.csv, lock_hist.png, cum_pnl.png.
"""
from __future__ import annotations

import importlib.util
import os
import socket
import sys


def _no_network(*a, **k):
    raise RuntimeError("hedge: network access is not allowed")


socket.socket.connect = _no_network
sys.dont_write_bytecode = True                       # no __pycache__ in research/rule_search/

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

import betting_system as bs                          # noqa: E402
import signals                                       # noqa: E402
from card_settle import OR_HI, OR_LO                 # noqa: E402  clean-book bounds

_spec = importlib.util.spec_from_file_location(
    "rule_search", os.path.join(ROOT, "research", "rule_search", "rule_search.py"))
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)
fb = rs.fb

LO, HI = "2012-01-01", "2023-01-01"                  # window: LO <= event_date < HI
YEARS = list(range(2012, 2023))
OUT = HERE
RULES = ("GAP", "FLIP")

# pre-registered: id -> (rule, threshold on L per 1u of stake, hedge fraction)
POLICIES = {
    "P0": ("never hedge", np.inf, 0.0),
    "P1": ("full lock when L >= 0", 0.0, 1.0),
    "P2a": ("full lock when L >= 0.02s", 0.02, 1.0),
    "P2b": ("full lock when L >= 0.05s", 0.05, 1.0),
    "P2c": ("full lock when L >= 0.10s", 0.10, 1.0),
    "P3": ("always full hedge at close", -np.inf, 1.0),
    "P4": ("half hedge (h/2) when L >= 0", 0.0, 0.5),
}
HX = " (hypothetical)"
# hedge-leg execution: (label, kind, pp added / commission, hypothetical)
EXECUTIONS = [("book +0pp", "book", 0.0, 0), ("book +1pp", "book", 1.0, 0),
              ("book +2pp", "book", 2.0, 0), ("exchange k=0%" + HX, "exchange", 0.0, 1),
              ("exchange k=2%" + HX, "exchange", 0.02, 1),
              ("exchange k=5%" + HX, "exchange", 0.05, 1)]
BAR_SETTINGS = ("book +2pp", "exchange k=2%" + HX)
BAR_DD_CUT, BAR_PNL_CUT = 0.30, 0.25                 # adoption bar 2 and 3
P0_COLOR, P1_COLOR, P3_COLOR = fb.COLORS["KK"], fb.COLORS["KU"], fb.COLORS["UU"]
NO_LOCK = "#a3a29d"


def fail(msg):
    raise SystemExit(f"STOP: {msg}")


# ---- lock math --------------------------------------------------------------

def dec(line):
    """American -> decimal."""
    return 1 + line / 100 if line > 0 else 1 + 100 / -line


def us(d):
    """Decimal -> American, rounded; "" when there is no price."""
    return int(round(signals.american(float(d)))) if np.isfinite(d) else ""


def hedge_price(c_x, c_y, kind, x):
    """Decimal the hedge on Y is struck at. book: Y's close with x pp added to
    its implied probability. exchange (hypothetical): Y's vig-free close,
    1 + c_y/c_x, with commission x on the hedge leg's winnings. NaN where there
    is no clean close (or a slipped price at or below 1)."""
    c_x, c_y = np.asarray(c_x, float), np.asarray(c_y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "book":
            c = 1 / (1 / c_y + x / 100)
        else:
            c = 1 + (c_y / c_x) * (1 - x)
    return np.where(np.isfinite(c) & (c > 1), c, np.nan)


def lock(s, d_o, c):
    """Full-lock hedge stake on Y and the locked P&L, for stake s on X at d_o."""
    h = s * d_o / c
    return h, s * (d_o - 1) - h


def settle(s, d_o, c, hs, x_won):
    """P&L of s on X at d_o plus hs on Y at c (hs = 0: no hedge) -> (P&L,
    payoff if X wins, payoff if Y wins)."""
    hs = np.asarray(hs, float)
    win_x = s * (d_o - 1) - hs
    win_y = np.where(hs > 0, hs * (np.nan_to_num(c, nan=1.0) - 1), 0.0) - s
    return np.where(x_won, win_x, win_y), win_x, win_y


def apply_policy(pol, s, d_o, c):
    """-> (hedged mask, hedge stake, full-lock L) under policy pol."""
    _, thr, frac = POLICIES[pol]
    h, L = lock(s, d_o, c)
    with np.errstate(invalid="ignore"):
        hedged = np.isfinite(c) & (L / s >= thr) & (frac > 0)
    return hedged, np.where(hedged, frac * h, 0.0), L


def selftest():
    """The brief's worked lock, before anything else, plus the sign rule and
    outcome-independence at every execution setting."""
    d_o, c = dec(150), dec(-135)
    h, L = lock(1.0, d_o, c)
    pay, _, _ = settle(1.0, d_o, c, h, np.array([True, False]))
    ok1 = abs(h - 1.436) < 0.001 and abs(L - 0.064) < 0.001
    ok2 = abs(pay[0] - pay[1]) < 1e-9 and abs(pay[0] - L) < 1e-9
    print(f"[unit test] +150 open, Y -135 close, s=1: h {h:.4f} (want 1.436 +-0.001), "
          f"L {L:+.4f} (want +0.064 +-0.001), payoff X wins {pay[0]:+.12f} / Y wins "
          f"{pay[1]:+.12f}, |diff| {abs(pay[0] - pay[1]):.1e} -> "
          f"{'PASS' if ok1 and ok2 else 'FAIL'}")
    g = np.round(np.arange(1.02, 8.0, 0.01), 2)
    Do, C = np.meshgrid(g, g)
    _, LL = lock(1.0, Do, C)
    m = 1 - 1 / Do - 1 / C
    away = np.abs(m) > 1e-12
    ok3 = bool(((LL > 0) == (m > 0))[away].all())
    print(f"[unit test] L > 0 iff 1/d_o + 1/c < 1 on a {len(g)}x{len(g)} price grid: "
          f"{'PASS' if ok3 else 'FAIL'}")
    worst = 0.0
    for _, kind, x, _ in EXECUTIONS:
        for cx, cy in ((dec(-150), dec(130)), (dec(-135), dec(115)), (dec(240), dec(-300))):
            ch = hedge_price(cx, cy, kind, x)
            hh, _ = lock(1.0, d_o, ch)
            _, wx, wy = settle(1.0, d_o, ch, hh, True)
            worst = max(worst, abs(float(wx) - float(wy)))
    ok4 = worst < 1e-9
    slip = float(hedge_price(dec(135), dec(-135), "book", 2.0))
    ok5 = abs(1 / slip - (1 / dec(-135) + 0.02)) < 1e-12
    print(f"[unit test] full lock outcome-independent at every execution setting "
          f"(max |diff| {worst:.1e}); -135 +2pp -> {us(slip)} -> "
          f"{'PASS' if ok4 and ok5 else 'FAIL'}")
    if not (ok1 and ok2 and ok3 and ok4 and ok5):
        fail("lock-math unit test failed - nothing run")


# ---- inputs -------------------------------------------------------------------

def load():
    """rule_search's frame over 2012-2022 (frozen preds every year), tagged and
    bucketed by fade_buckets, with the clean close on both sides (a_cl, b_cl;
    NaN where there is none)."""
    fv_hist = pd.read_csv("data/fights_v2.csv")
    fv_hist = fv_hist[fv_hist.event_date < HI]
    assert fv_hist.event_date.max() < HI, fv_hist.event_date.max()
    fv = fv_hist[fv_hist.event_date >= LO]
    ids = set(fv.fight_id)
    ev_year = fv.set_index("fight_id").event_date.str[:4].astype(int)
    J = pd.read_csv("data/bfo_joined.csv")
    J = J[J.fight_id.isin(ids)]
    rd = pd.read_csv(bs.PREFIGHT_RD)
    rd = rd[rd.fight_id.isin(ids)]
    M, _ = rs.model_preds(ids, ev_year)
    if (M.src != "frozen").any():
        fail("a window bout has no frozen prediction")
    D = fb.per_bout(fb.tag(rs.build_frame(fv_hist, fv, J, rd, M)))
    assert D.event_date.max() < HI and D.event_date.min() >= LO, (D.event_date.min(),
                                                                  D.event_date.max())
    col = D.a_close_clean.notna() & D.b_close_clean.notna()
    ca = D.a_close_clean.where(col, D.a_close)
    cb = D.b_close_clean.where(col, D.b_close)
    ov = 1 / ca + 1 / cb
    clean = col | ((ov >= OR_LO) & (ov <= OR_HI))
    D["a_cl"], D["b_cl"] = ca.where(clean), cb.where(clean)
    D["close_src"] = np.where(col, "clean column", np.where(clean, "raw (clean book)", "none"))
    return D


def bet_frame(G, side_a, rule, stake):
    """One row per bet: X = the side bet at the open, Y = the other side."""
    side_a = np.asarray(side_a, bool)
    pick = lambda a, b: np.where(side_a, G[a].values, G[b].values)
    return pd.DataFrame({
        "event_date": G.event_date.values, "event": G.event_name.values,
        "fight_id": G.fight_id.values, "bout_order": G.bout_order.values, "yr": G.yr.values,
        "rule": rule, "bet_on": pick("fighter_a", "fighter_b"),
        "opponent": pick("fighter_b", "fighter_a"),
        "x_won": (G.y_a.values == 1) == side_a,
        "d_o": pick("a_open", "b_open"), "d_oy": pick("b_open", "a_open"),
        "c_x": pick("a_cl", "b_cl"), "c_y": pick("b_cl", "a_cl"),
        "rd_self": pick("rd_a", "rd_b"), "rd_opp": pick("rd_b", "rd_a"),
        "stake": np.asarray(stake, float)})


def u1_bets(D):
    """Frozen GAP/FLIP signals (VOID rows excluded, pick'ems included, as in the
    sim) with the policy-v1 stake from the shared classifier at the open."""
    S = D[D.tag.isin(RULES)]
    dd = [bs._decision(r) for r in S.itertuples(index=False)]
    if [d.rule for d in dd] != list(S.tag) or [d.side for d in dd] != list(S.tag_side):
        fail("signals.classify disagrees with fade_buckets.tag()")
    return bet_frame(S, S.tag_side.values == 1, S.tag.values, [d.stake for d in dd])


def u2_bets(D):
    """Every usable bout (rule_search's exclusions), 1u on the model side."""
    U, ex = fb.universe(D)
    _, keep = rs.exclusion_masks(D)
    if not (D.fight_id[keep].values == U.fight_id.values).all():
        fail("exclusion masks do not reproduce fade_buckets.universe()")
    return bet_frame(U, U.side_is_a.values, "", np.ones(len(U))), ex


# ---- scoring --------------------------------------------------------------------

def metrics(pnl, s, hedged, hs, yr, p0):
    n = len(pnl)
    sd = pnl.std(ddof=1)
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    ys = pd.Series(pnl).groupby(np.asarray(yr)).sum().sort_index()
    r = hs[hedged] / s[hedged]
    return dict(n_bets=n, pct_hedged=100 * hedged.mean(),
                mean_hedge_per_u=r.mean() if len(r) else np.nan,
                max_hedge_per_u=r.max() if len(r) else np.nan,
                staked_u=s.sum(), pnl_u=pnl.sum(), roi_pct=100 * pnl.sum() / s.sum(),
                mean_pnl_per_bet=pnl.mean(), sd_pnl_per_bet=sd,
                t=pnl.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan,
                max_dd_u=float((peak - cum).max()),
                worst_year=int(ys.idxmin()), worst_year_pnl_u=float(ys.min()),
                losing_years=int((ys < 0).sum()), years=len(ys),
                n_locked=int(hedged.sum()), pnl_locked_u=pnl[hedged].sum(),
                pnl_locked_if_ridden_u=p0[hedged].sum(), n_ridden=int((~hedged).sum()),
                pnl_ridden_u=pnl[~hedged].sum())


def run_grid(views):
    """views: (universe, staking, bet frame). -> the metrics grid and every
    (universe, execution, policy) -> (P&L, hedged, hedge stake, L, payoff if X
    wins, payoff if Y wins), all at the actual stake."""
    rows, keep = [], {}
    for name, staking, B in views:
        s, d_o, x_won = B.stake.values, B.d_o.values, B.x_won.values
        p0 = settle(s, d_o, np.nan, 0.0, x_won)[0]
        no_close = int(np.isnan(B.c_y.values).sum())
        for label, kind, x, hyp in EXECUTIONS:
            c = hedge_price(B.c_x.values, B.c_y.values, kind, x)
            for pol, (rule, _, _) in POLICIES.items():
                hedged, hs, L = apply_policy(pol, s, d_o, c)
                pnl, wx, wy = settle(s, d_o, c, hs, x_won)
                keep[(name, label, pol)] = dict(pnl=pnl, hedged=hedged, hs=hs, L=L, wx=wx, wy=wy)
                rows.append(dict(universe=name, staking=staking, policy=pol, policy_rule=rule,
                                 execution=label, hypothetical=hyp,
                                 **metrics(pnl, s, hedged, hs, B.yr.values, p0),
                                 n_no_clean_close=no_close))
    return pd.DataFrame(rows), keep


def adoption(R):
    """The pre-registered bar on U1-placed, per policy, at each bar setting."""
    R = R.copy()
    for c in ("bar1_t_above_p0", "bar2_dd_30pct_below", "bar3_pnl_within_25pct", "adopt"):
        R[c] = ""
    for ex in BAR_SETTINGS:
        m = (R.universe == "U1-placed") & (R.execution == ex)
        p0 = R[m & (R.policy == "P0")].iloc[0]
        for i in R.index[m & (R.policy != "P0")]:
            r = R.loc[i]
            b = (r.t > p0.t, r.max_dd_u <= (1 - BAR_DD_CUT) * p0.max_dd_u,
                 r.pnl_u >= p0.pnl_u - BAR_PNL_CUT * abs(p0.pnl_u))
            R.loc[i, ["bar1_t_above_p0", "bar2_dd_30pct_below", "bar3_pnl_within_25pct"]] = \
                ["PASS" if v else "FAIL" for v in b]
            R.loc[i, "adopt"] = "PASS" if all(b) else "FAIL"
    return R


# ---- sanity checks ----------------------------------------------------------------

def check_sim(D, B1, keep):
    """P0 on U1-flat and U1-placed vs betting_system.run() on the same frame and
    vs the committed ledger, 2012-2022: count, W-L, P&L, and the same bets,
    sides and stakes."""
    run = bs.counted(bs.run(D))
    if (run.live != 0).any():
        fail("a window bet is live (it would settle at a capture, not the open)")
    L = pd.read_csv(bs.LEDGER)
    L = L[(L.event_date >= LO) & (L.event_date < HI) & (L.units > 0)]
    flat = keep[("U1-flat", "book +0pp", "P0")]["pnl"]
    placed = keep[("U1-placed", "book +0pp", "P0")]["pnl"]
    pm = B1.stake.values > 0
    wl = lambda x: (len(x), int((x > 0).sum()), int((x < 0).sum()), float(np.sum(x)))
    lines, ok = [], True
    for view, mine, sim, led in (
            ("U1-flat", wl(flat), wl(run.pnl_at_open.values), wl(L.pnl_at_open.values)),
            ("U1-placed", wl(placed), wl(run[run.placed == 1].pnl_placed.values),
             wl(L[L.placed == 1].pnl_placed.values))):
        good = (mine[:3] == sim[:3] == led[:3] and abs(mine[3] - sim[3]) < 1e-9
                and abs(led[3] - sim[3]) <= 0.5e-4 * sim[0])
        ok &= good
        f = lambda v: f"{v[0]} bets {v[1]}-{v[2]} {v[3]:+.4f}u"
        lines.append(f"{view:9s} here {f(mine)} | bs.run {f(sim)} | ledger {f(led)} "
                     f"-> {'PASS' if good else 'FAIL'}")
    tt = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    lines.append(f"    P0 U1-placed t {tt(placed):.3f} on P&L per bet at the stake (this study's "
                 f"t); betting_system's PLACED t on per-unit return "
                 f"{tt(placed / B1.stake.values[pm]):.3f}")
    for rl in RULES:
        x = flat[B1.rule.values == rl]
        lines.append(f"    {rl}: {len(x)} bets {int((x > 0).sum())}-{int((x < 0).sum())} "
                     f"{x.sum():+.2f}u (1u at open)")
    mine = B1.assign(side=B1.bet_on).set_index("fight_id")
    sim = run.assign(side=np.where(run.mod1, run.fighter_a, run.fighter_b)).set_index("fight_id")
    led = L.set_index("fight_id")
    ids_ok = set(mine.index) == set(sim.index) == set(led.index)
    if ids_ok:
        f_ok = (mine.side == sim.side[mine.index]).all() and \
               (mine.side == led.fighter[mine.index]).all()
        st_ok = (np.abs(mine.stake - sim.stake_u[mine.index]) < 1e-12).all() and \
                (np.abs(mine.stake - led.stake_u[mine.index]) < 1e-9).all()
        pl_ok = (mine.stake > 0).eq(led.placed[mine.index] == 1).all()
    else:
        f_ok = st_ok = pl_ok = False
    good = ids_ok and f_ok and st_ok and pl_ok
    ok &= good
    lines.append(f"same fight_ids {ids_ok}, same side {f_ok}, same stake {st_ok}, same placed "
                 f"set {pl_ok} (here vs bs.run vs ledger) -> {'PASS' if good else 'FAIL'}")
    # the placed P&L is the per-1u P&L times the stake, policy by policy
    worst = max(float(np.max(np.abs(keep[("U1-placed", ex, p)]["pnl"]
                                    - B1.stake.values[pm] * keep[("U1-flat", ex, p)]["pnl"][pm])))
                for ex, _, _, _ in EXECUTIONS for p in POLICIES)
    lines.append(f"U1-placed P&L == stake x U1-flat P&L per 1u, every policy and setting: "
                 f"max |diff| {worst:.1e} -> {'PASS' if worst < 1e-12 else 'FAIL'}")
    ok &= worst < 1e-12
    return ok, lines, led


def check_close(B1, led):
    """The close used here == the ledger's clean_close (bet side, American) and
    clv_pts, on every U1 bet that has one."""
    has = np.isfinite(B1.c_x.values)
    g = B1[has]
    lc = led.loc[g.fight_id]
    am = np.array([us(c) for c in g.c_x], dtype=int)
    qo = (1 / g.d_o) / (1 / g.d_o + 1 / g.d_oy)
    qc = (1 / g.c_x) / (1 / g.c_x + 1 / g.c_y)
    clv = (qc - qo).values * 100                     # the ledger rounds to 2 dp
    a_ok = (am == lc.clean_close.values).all()
    c_ok = (np.abs(clv - lc.clv_pts.values) <= 0.005 + 1e-9).all()
    return a_ok and c_ok, (f"{len(g)} U1 bets with a clean close: close == ledger clean_close "
                           f"{a_ok}, CLV pts == ledger clv_pts {c_ok}")


def check_p3_lock(keep):
    worst, n = 0.0, 0
    for (_, _, pol), k in keep.items():
        if pol == "P3":
            h = k["hedged"]
            n += int(h.sum())
            if h.any():
                worst = max(worst, float(np.max(np.abs(k["wx"][h] - k["wy"][h]))))
    return worst < 1e-9, (f"{n} hedged bet-settings under P3 (3 views x 6 settings): max "
                          f"|payoff if X wins - payoff if Y wins| {worst:.1e}")


def check_p3_clv(B2, keep):
    """U2, P3 at the BFO close: the locked P&L is the open bet's expected return
    at the vig-free close (CLV in price terms) less the vig the hedge pays at
    the close (hedge stake x the closing hold). The brief's approximation uses
    the unweighted hold."""
    k = keep[("U2", "book +0pp", "P3")]
    h = k["hedged"]
    g = B2[h]
    ov = 1 / g.c_x.values + 1 / g.c_y.values
    clv = g.d_o.values * (1 / g.c_x.values) / ov - 1
    hold = 1 - 1 / ov
    hu = g.d_o.values / g.c_y.values
    exact = clv - hu * hold
    err = float(np.max(np.abs(k["pnl"][h] - exact)))
    qo = (1 / g.d_o) / (1 / g.d_o + 1 / g.d_oy)
    clv_pp = 100 * ((1 / g.c_x.values) / ov - qo.values).mean()
    m_all, m_h = k["pnl"].mean(), k["pnl"][h].mean()
    approx = clv.mean() - hold.mean()
    ok = m_all < 0 and err < 1e-9
    lines = [f"P3 mean P&L per bet {m_all:+.4%} over all {len(B2)} U2 bets "
             f"({m_h:+.4%} over the {int(h.sum())} hedged)",
             f"mean CLV in price terms (d_o x vig-free close prob - 1) {clv.mean():+.4%} "
             f"(mean CLV {clv_pp:+.3f} pp); average closing vig (hold, 1 - 1/overround) "
             f"{hold.mean():.4%}",
             f"brief's approximation CLV - vig {approx:+.4%}: gap to P3 (hedged) "
             f"{m_h - approx:+.4%}, because the hedge stake averages {hu.mean():.3f}u per 1u, "
             f"not 1u",
             f"exact: CLV - mean(hedge stake x hold) {exact.mean():+.4%}; per-bet identity "
             f"max |diff| {err:.1e}"]
    return ok, lines, dict(p3=m_all, p3_hedged=m_h, clv=clv.mean(), hold=hold.mean(),
                           approx=approx, exact=exact.mean(), hu=hu.mean(), clv_pp=clv_pp)


# ---- charts -----------------------------------------------------------------------

def lock_chart(L, n_all, n_none, path):
    """Histogram of the full-lock L per 1u at the BFO close, U1-flat."""
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=150, facecolor=fb.SURFACE)
    fb._style(ax)
    w, xmax = 0.02, 1.1                              # bets beyond xmax are noted, not drawn
    shown = L[L <= xmax]
    k0, k1 = int(np.floor(shown.min() / w)), int(np.ceil(shown.max() / w))
    edges = np.array([k * w for k in range(k0, k1 + 1)])
    cnt, _ = np.histogram(shown, bins=edges)
    pos = edges[:-1] >= 0
    ax.bar(edges[:-1], cnt, width=w, align="edge", color=np.where(pos, P0_COLOR, NO_LOCK),
           edgecolor=fb.SURFACE, linewidth=0.8)
    top = cnt.max()
    ax.set_ylim(0, top * 1.42)
    ax.set_xlim(edges[0] - w, xmax)
    ax.axvline(0, color=fb.INK, linewidth=1.6)
    for xv in (0.02, 0.05, 0.10):
        ax.axvline(xv, color=fb.INK2, linewidth=0.9, linestyle=(0, (2, 2)))
    ax.annotate("P2a / P2b / P2c thresholds: 0.02, 0.05, 0.10", (0.10, top * 1.38),
                xytext=(5, 0), textcoords="offset points", ha="left", va="top", fontsize=8,
                color=fb.INK2)
    npos = int((L > 0).sum())
    med = float(np.median(L[L > 0])) if npos else np.nan
    ax.annotate(f"lock available (L > 0)\n{npos} of {n_all} bets ({npos / n_all:.0%}), "
                f"median +{med:.3f}u", (0.10, top * 1.17), xytext=(5, 0),
                textcoords="offset points", ha="left", va="center", fontsize=9, color=fb.INK)
    ax.annotate(f"no lock (L <= 0)\n{len(L) - npos} bets", (0, top * 1.17), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="center", fontsize=9, color=fb.INK2)
    off = np.sort(L[L > xmax])
    if len(off):
        ax.annotate(f"{len(off)} off the chart at " + ", ".join(f"+{v:.2f}u" for v in off)
                    + " ->", (xmax, top * 0.12), xytext=(-4, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=fb.INK2)
    ax.set_xlabel("L: profit locked by a full hedge at the close, per 1u bet at the open",
                  color=fb.INK2, fontsize=9.5)
    ax.set_ylabel(f"bets per {w:.2f}u bin", color=fb.INK2, fontsize=9.5)
    fig.text(0.07, 0.945, "How often the line moves far enough to lock a profit",
             fontsize=13, color=fb.INK, weight="bold")
    fig.text(0.07, 0.902, f"{n_all} frozen GAP/FLIP signals 2012-2022, 1u at the open, full "
             f"hedge at the BFO close (best price, no slippage); "
             + (f"{n_none} without a clean close not shown" if n_none
                else "every bet has a clean close"), fontsize=9, color=fb.INK2)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.11)
    fig.savefig(path, facecolor=fb.SURFACE, metadata={"Software": None})
    plt.close(fig)


def cum_chart(B, curves, path, subtitle):
    """Cumulative U1-placed P&L in bet order, one line per policy."""
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150, facecolor=fb.SURFACE)
    fb._style(ax)
    x = np.arange(1, len(B) + 1)
    yr = B.yr.values
    starts = np.flatnonzero(np.r_[True, yr[1:] != yr[:-1]])
    for s in starts[1:]:
        ax.axvline(s + 0.5, color=fb.GRID, linewidth=0.8, zorder=0)
    ax.axhline(0, color=fb.INK, linewidth=1)
    ends = []
    for name, rule, col, y in curves:
        cy = np.cumsum(y)
        ax.plot(x, cy, color=col, linewidth=2, label=f"{name}  {rule}")
        ends.append([cy[-1], f"{name} {cy[-1]:+.1f}u"])
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ends.sort()
    for i in range(1, len(ends)):                    # keep end labels apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for yv, txt in ends:
        ax.annotate(txt, (x[-1], yv), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=fb.INK)
    mids = (starts + np.r_[starts[1:], len(B)]) / 2 + 0.5
    ax.set_xticks(mids, [str(y) for y in yr[starts]])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlim(0, len(B) * 1.08)
    ax.set_ylabel("cumulative units at the policy-v1 stake", color=fb.INK2, fontsize=9.5)
    ax.set_xlabel("placed bets in date order", color=fb.INK2, fontsize=9)
    fig.text(0.06, 0.95, "Hedging the placed bets at the close, with 2 pp book slippage",
             fontsize=13, color=fb.INK, weight="bold")
    for i, line in enumerate(subtitle):
        fig.text(0.06, 0.912 - 0.03 * i, line, fontsize=9, color=fb.INK2)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.055, 0.905 - 0.03 * len(subtitle)),
               frameon=False, fontsize=8.5, ncol=3, labelcolor=fb.INK)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.8, bottom=0.1)
    fig.savefig(path, facecolor=fb.SURFACE, metadata={"Software": None})
    plt.close(fig)


# ---- output -------------------------------------------------------------------------

ROUND = dict(pct_hedged=2, mean_hedge_per_u=4, max_hedge_per_u=4, staked_u=2, pnl_u=4,
             roi_pct=3, mean_pnl_per_bet=5, sd_pnl_per_bet=5, t=3, max_dd_u=4,
             worst_year_pnl_u=4, pnl_locked_u=4, pnl_locked_if_ridden_u=4, pnl_ridden_u=4)


def bets_table(B1, keep):
    rows = pd.DataFrame({
        "event_date": B1.event_date, "event": B1.event, "fight_id": B1.fight_id,
        "rule": B1.rule, "bet_on": B1.bet_on, "opponent": B1.opponent,
        "result": np.where(B1.x_won, "WIN", "LOSS"),
        "placed": (B1.stake > 0).astype(int), "stake_u": B1.stake,
        "rd_self": B1.rd_self.round(2), "rd_opp": B1.rd_opp.round(2),
        "open_bet": [us(d) for d in B1.d_o], "open_opp": [us(d) for d in B1.d_oy],
        "close_bet": [us(d) for d in B1.c_x], "close_opp": [us(d) for d in B1.c_y],
        "open_bet_dec": B1.d_o.round(6), "open_opp_dec": B1.d_oy.round(6),
        "close_bet_dec": B1.c_x.round(6), "close_opp_dec": B1.c_y.round(6),
        "clean_close": np.isfinite(B1.c_x).astype(int)})
    qo = (1 / B1.d_o) / (1 / B1.d_o + 1 / B1.d_oy)
    qc = (1 / B1.c_x) / (1 / B1.c_x + 1 / B1.c_y)
    rows["clv_pp"] = ((qc - qo) * 100).round(4)
    for tag, ex in (("0pp", "book +0pp"), ("2pp", "book +2pp")):
        k = keep[("U1-flat", ex, "P3")]
        rows[f"hedge_u_{tag}"] = np.round(k["hs"], 6)
        rows[f"L_{tag}"] = np.round(k["L"], 6)
    for tag, ex in (("0pp", "book +0pp"), ("2pp", "book +2pp")):
        for p in POLICIES:
            rows[f"pnl_{p}_{tag}"] = np.round(keep[("U1-flat", ex, p)]["pnl"], 6)
    return rows


def show(R, view, ex):
    T = R[(R.universe == view) & (R.execution == ex)]
    cols = ["policy", "n_bets", "pct_hedged", "mean_hedge_per_u", "pnl_u", "roi_pct",
            "sd_pnl_per_bet", "t", "max_dd_u", "worst_year", "worst_year_pnl_u",
            "losing_years", "n_locked", "pnl_locked_u", "pnl_locked_if_ridden_u"]
    return T[cols].round(ROUND).to_string(index=False)


def main():
    print(f"window {LO} <= event_date < {HI}; no random draws")
    selftest()
    D = load()
    B1 = u1_bets(D)
    B1p = B1[B1.stake > 0].reset_index(drop=True)
    B2, ex = u2_bets(D)
    for B in (B1, B2):
        assert B.event_date.max() < HI and B.event_date.min() >= LO
    src = D.close_src.value_counts()
    print(f"\nframe {len(D)} bouts with an open and a frozen probability; close source: "
          + ", ".join(f"{k} {v}" for k, v in src.items()))
    print(f"U1 {len(B1)} signals ({', '.join(f'{r} {int((B1.rule == r).sum())}' for r in RULES)}), "
          f"{len(B1p)} placed by policy v1; U2 {len(B2)} usable bouts (exclusions: "
          + ", ".join(f"{k} {v}" for k, v in ex.items()) + ")")
    nc = B1[np.isnan(B1.c_x.values)]
    print(f"U1 signals without a clean close (never hedged): {len(nc)}")
    for r in nc.itertuples(index=False):
        print(f"    {r.event_date} {r.fight_id} {r.rule} {r.bet_on} vs {r.opponent} "
              f"(stake {r.stake:.2f}u)")
    print(f"U2 bouts without a clean close: {int(np.isnan(B2.c_x.values).sum())}")

    views = [("U1-flat", "1u flat", B1.assign(stake=1.0)), ("U1-placed", "policy v1", B1p),
             ("U2", "1u flat", B2)]
    R, keep = run_grid(views)

    fails = []
    ok, lines, led = check_sim(D, B1, keep)
    print("\n[sanity 1] P0 vs betting_system.run() on the same frame vs the committed "
          "ledger, 2012-2022")
    for ln in lines:
        print("    " + ln)
    if not ok:
        fails.append("P0 does not reproduce betting_system's sim")
    ok, msg = check_close(B1, led)
    print(f"[sanity 1b] {msg} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("the close here is not the ledger's clean close")
    ok, msg = check_p3_lock(keep)
    print(f"[sanity 2] {msg} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("a P3 hedge is not outcome-independent")
    ok, lines, s3 = check_p3_clv(B2, keep)
    print("[sanity 3] U2, P3 at the BFO close (book +0pp)")
    for ln in lines:
        print("    " + ln)
    print(f"    P3 mean negative and the per-bet identity exact -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("P3 on U2 is not CLV net of the closing vig")
    if fails:
        fail("; ".join(fails) + " - nothing written")

    R = adoption(R)
    T = bets_table(B1, keep)
    R.round(ROUND).to_csv(os.path.join(OUT, "hedge_results.csv"), index=False,
                          lineterminator="\n")
    T.to_csv(os.path.join(OUT, "hedge_bets.csv"), index=False, lineterminator="\n")

    L0 = keep[("U1-flat", "book +0pp", "P3")]["L"]
    L2 = keep[("U1-flat", "book +2pp", "P3")]["L"]
    fin = np.isfinite(L0)
    lock_chart(L0[fin], len(B1), int((~fin).sum()), os.path.join(OUT, "lock_hist.png"))
    ex2 = "book +2pp"
    curves = [(p, POLICIES[p][0], c, keep[("U1-placed", ex2, p)]["pnl"])
              for p, c in (("P0", P0_COLOR), ("P1", P1_COLOR), ("P3", P3_COLOR))]
    sub = R[(R.universe == "U1-placed") & (R.execution == ex2)].set_index("policy")
    cum_chart(B1p, curves, os.path.join(OUT, "cum_pnl.png"),
              [f"{len(B1p)} placed GAP/FLIP bets 2012-2022 at the policy-v1 stake, bet at the "
               f"open; the hedge leg at the BFO close with 2 pp added to its implied probability",
               "  ·  ".join(f"{p}: t {sub.t[p]:.2f}, max drawdown {sub.max_dd_u[p]:.1f}u"
                            for p in ("P0", "P1", "P3"))])

    # report
    print("\nLOCK AVAILABILITY (U1-flat, full lock per 1u)")
    for tag, L in (("BFO close (+0pp)", L0), ("book +2pp", L2)):
        p = L[np.isfinite(L) & (L > 0)]
        print(f"    {tag}: positive lock on {len(p)} of {len(B1)} bets ({len(p) / len(B1):.1%}); "
              f"median L among them {np.median(p):+.4f}u, mean {p.mean():+.4f}u; "
              f">= 0.02u {int((p >= 0.02).sum())}, >= 0.05u {int((p >= 0.05).sum())}, "
              f">= 0.10u {int((p >= 0.10).sum())}")
    for ex in ("book +0pp", "book +2pp", "exchange k=2%" + HX):
        print(f"\nU1-placed, hedge at {ex}")
        print(show(R, "U1-placed", ex))
    print("\nADOPTION BAR (U1-placed): t above P0's, max drawdown >= 30% below P0's, "
          "P&L no more than 25% below P0's")
    for ex in BAR_SETTINGS:
        A = R[(R.universe == "U1-placed") & (R.execution == ex) & (R.policy != "P0")]
        print(f"    {ex}:")
        for r in A.itertuples(index=False):
            print(f"        {r.policy:4s} t {r.bar1_t_above_p0}  drawdown {r.bar2_dd_30pct_below}  "
                  f"P&L {r.bar3_pnl_within_25pct}  -> {r.adopt}")
    print("\nU2 headline (1u on the model side, every usable bout)")
    for ex in ("book +0pp", "book +2pp", "exchange k=2%" + HX):
        T2 = R[(R.universe == "U2") & (R.execution == ex) & R.policy.isin(["P0", "P1", "P3"])]
        print(f"    {ex}: " + "; ".join(
            f"{r.policy} {r.pnl_u:+.1f}u ROI {r.roi_pct:+.2f}% t {r.t:+.2f} DD {r.max_dd_u:.1f}u"
            for r in T2.itertuples(index=False)))
    print(f"\nwrote {', '.join(sorted(f for f in os.listdir(OUT) if f.endswith(('.csv', '.png'))))}")


if __name__ == "__main__":
    main()
