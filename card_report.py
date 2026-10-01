"""Card report: model projections + open/close lines + CLV for one UFC event.

Usage: python card_report.py <bfo-event-slug> <event-date YYYY-MM-DD>

Builds fighter states from all fights STRICTLY BEFORE event-date (leakage-safe),
projects every bout with the frozen Stage 2 model (trained <2026), pulls
BFO opening/closing mean lines, grades CLV, and joins results if the event
is in data/fights_v2.csv.

For an event dated today or later the BFO fetches bypass the disk cache and
the current line of every bout is recorded in data/placeable_lines.csv (see
record_placeable): the price that could actually be bet when the pick went on
record. betting_system.py settles live bets at the earliest such capture.

Opens are the validated opens (scrape_bfo.validated_open: the first tick
unless it was corrected within the hour). A signal that fires only against
the raw first tick of such a bout prints "void: suspect opener" and is never
a bet; betting_system.py logs it as a VOID row.

Polymarket (polymarket.py) is a second price source, never a signal source.
The PM column is fighter1's Polymarket price; the capture records it with the
market's liquidity and slug (pm_p_self, pm_liquidity, pm_slug; empty when
there is no market). For a future card, bouts Polymarket lists that BFO does
not yet have are projected too -- in their own Stage 2 frame, so the BFO
bouts' projections are what they would be without them -- with the BFO
columns blank and "no BFO line" in the signal column: no signal can fire
without a BFO open.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import engine
import features
import polymarket
import stage2
from betting_system import (FLIP_CONF, GAP_PTS, SKIP_LINE_MAX, VOID_SUSPECT, am as am_dec,
                            amp as amp_prob, stake_units)
from names import bfo_norm, make_finder, norm
from scrape_bfo import bout_summary, fetch, ggd

PASSC = ["d_ss_acc", "d_ss_def", "d_td_acc", "d_td_def", "d_ctrl15",
         "d_pace15", "d_head_share", "d_leg_share", "d_ground_share",
         "d_ctrled15"]


def build_states(cutoff: date):
    bp = json.load(open("best_params.json"))
    engine.TAU = bp["tau"]
    engine.SIGMA0 = bp["sigma0"]
    for k, v in [("U-DEC", bp["s_udec"]), ("M-DEC", bp["s_udec"]),
                 ("S-DEC", bp["s_sdec"]), ("DQ", bp["s_sdec"])]:
        engine.S_TABLE[k] = (v, 1 - v)
    fights, meta, details = features.load()
    eng = engine.Engine()
    car = defaultdict(features.Career)
    for f in fights:
        d = date.fromisoformat(f["event_date"])
        if d >= cutoff:
            continue
        aid, bid = f["fighter_a_id"], f["fighter_b_id"]
        outcome, method = f["outcome"], f["method"]
        mins = features.duration_min(f["round"], f["time"])
        num = lambda k: float(f[k]) if f[k] not in ("", None) else 0.0
        won_a, drew = outcome == "A_WIN", outcome == "DRAW"
        if outcome != "NC" and method not in ("Overturned", "CNC"):
            det = details.get(f["fight_id"])
            da = db = None
            if det:
                dv = lambda k: float(det[k]) if det.get(k) not in ("", None) else None
                da = {"ss_l": dv("a_ss_l"), "ss_a": dv("a_ss_a"),
                      "opp_ss_l": dv("b_ss_l"), "opp_ss_a": dv("b_ss_a"),
                      "td_a": dv("a_td_a"), "opp_td_l": dv("b_td_l"),
                      "opp_td_a": dv("b_td_a"), "ctrl_s": dv("a_ctrl_s"),
                      "opp_ctrl_s": dv("b_ctrl_s"), "head_l": dv("a_head_l"),
                      "leg_l": dv("a_leg_l"), "ground_l": dv("a_ground_l")}
                db = {"ss_l": dv("b_ss_l"), "ss_a": dv("b_ss_a"),
                      "opp_ss_l": dv("a_ss_l"), "opp_ss_a": dv("a_ss_a"),
                      "td_a": dv("b_td_a"), "opp_td_l": dv("a_td_l"),
                      "opp_td_a": dv("a_td_a"), "ctrl_s": dv("b_ctrl_s"),
                      "opp_ctrl_s": dv("a_ctrl_s"), "head_l": dv("b_head_l"),
                      "leg_l": dv("b_leg_l"), "ground_l": dv("b_ground_l")}
            car[aid].update(mins, num("a_sig_str"), num("b_sig_str"), num("a_kd"),
                            num("b_kd"), num("a_td"), num("b_td"), num("a_sub_att"),
                            won_a, drew, method, d, da)
            car[bid].update(mins, num("b_sig_str"), num("a_sig_str"), num("b_kd"),
                            num("a_kd"), num("b_td"), num("a_td"), num("b_sub_att"),
                            not won_a and not drew, drew, method, d, db)
        eng.process({"event_date": f["event_date"], "fighter_a_id": aid,
                     "fighter_a": f["fighter_a"], "fighter_b_id": bid,
                     "fighter_b": f["fighter_b"], "outcome": outcome,
                     "method": method, "a_missed_weight": f.get("a_mw", ""),
                     "b_missed_weight": f.get("b_mw", "")})
    return eng, car, meta


def get_card(slug, fresh=False):
    """Bouts with opening and latest mean lines from the BFO event page.
    fresh=True bypasses the disk cache (future events: the latest tick must be
    the price on the board now, not whenever the page was first cached).
    Opens are validated (scrape_bfo.bout_summary); open_suspect and the
    raw first ticks ride along."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(fetch(f"https://www.bestfightodds.com/events/{slug}",
                               refresh=fresh), "lxml")
    bouts = []
    for tr in soup.select("tr[id^=mu-]"):
        mid = tr["id"].split("-")[1]
        if not mid.isdigit():
            continue
        a1 = tr.select_one("a[href^='/fighters/']")
        tr2 = tr.find_next_sibling("tr")
        a2 = tr2.select_one("a[href^='/fighters/']") if tr2 else None
        if a1 and a2:
            mu = int(mid)
            s = bout_summary(ggd(mu, 1, refresh=fresh), ggd(mu, 2, refresh=fresh))
            if s:
                bouts.append(dict(mu=mu, f1=a1.get_text(strip=True),
                                  f2=a2.get_text(strip=True),
                                  **{k: s[k] for k in ("f1_open", "f1_close", "f2_open",
                                                       "f2_close", "open_suspect",
                                                       "f1_open_raw", "f2_open_raw")}))
    return bouts


# Bouts BestFightOdds still lists that are not on the live card (withdrawals,
# cancellations). Keyed by (event date, BFO matchup id), so an entry can only
# ever affect the one card it was added for. Dropped before the report table,
# the signals and the placeable capture alike: a cancelled bout must not reach
# data/placeable_lines.csv, which is append-only.
OFF_CARD = {
    ("2026-09-19", 44891),   # UFC 331 Ortega / Moicano: Ortega withdrew days
                             # out (suspected eye injury), Moicano moved to a
                             # later Fight Night
    ("2026-09-26", 44970),   # UFC Vegas 121 Gall / Dumas: Gall out, Dumas
                             # fought Luis Hernandez (own BFO matchup); BFO
                             # kept the old bout listed, Polymarket refunded it
    ("2026-09-26", 44977),   # UFC Vegas 121 Amaya / Machado: Machado out,
                             # Amaya fought Tina Black; BFO never listed the
                             # replacement bout and kept this one
}


PLACEABLE = "data/placeable_lines.csv"
PLACEABLE_COLS = ["event_date", "bfo_slug", "mu", "fighter1", "fighter2",
                  "f1_line", "f2_line", "captured_at",
                  "pm_p_self", "pm_liquidity", "pm_slug"]   # since 2026-09-30


def record_placeable(slug, ev_date_s, bouts, pm_by_mu=None, path=PLACEABLE):
    """Append the current line of every bout of a FUTURE event to `path`: the
    price that could actually be bet when the pick went on record
    (betting_system.py settles live bets at the EARLIEST capture per bout).
    Append-only: rows already on file are never rewritten or replaced. One row
    per bout per run, stamped with the run's captured_at; a bout whose lines
    are identical to a row already on file (same mu, f1_line, f2_line) is
    skipped, so a re-run that finds nothing moved adds nothing. Never called
    for past events. pm_p_self / pm_liquidity / pm_slug: fighter1's Polymarket
    price (fighter2's is 1 - it), the moneyline's liquidity and the event
    slug at the same moment; empty when there is no market."""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    seen = set()
    have = os.path.exists(path) and os.path.getsize(path) > 0
    if have:
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh)
            if rd.fieldnames != PLACEABLE_COLS:
                raise SystemExit(f"{path} has columns {rd.fieldnames}; expected "
                                 f"{PLACEABLE_COLS}")
            for r in rd:
                seen.add((r["mu"], float(r["f1_line"]), float(r["f2_line"])))
    new = [b for b in bouts
           if (str(b["mu"]), float(b["f1_close"]), float(b["f2_close"])) not in seen]
    pm_by_mu = pm_by_mu or {}
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PLACEABLE_COLS)
        if not have:
            w.writeheader()
        for b in new:
            rec = pm_by_mu.get(b["mu"])
            w.writerow(dict(event_date=ev_date_s, bfo_slug=slug, mu=str(b["mu"]),
                            fighter1=b["f1"], fighter2=b["f2"],
                            f1_line=b["f1_close"], f2_line=b["f2_close"],
                            captured_at=now,
                            pm_p_self="" if not rec or rec["p_a"] is None else rec["p_a"],
                            pm_liquidity=("" if not rec or rec["liquidity"] is None
                                          else rec["liquidity"]),
                            pm_slug=rec["slug"] if rec else ""))
    print(f"placeable lines -> {path}: {len(new)} appended, "
          f"{len(bouts) - len(new)} unchanged ({now})")


def earliest_capture(path=PLACEABLE):
    """mu -> (f1_line, f2_line) decimal of the earliest capture on file: the
    price the ledger settles a live bet at (betting_system.placeable_prices)."""
    out = {}
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return out
    with open(path, newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["captured_at"])
    for r in rows:
        out.setdefault(int(r["mu"]), (float(r["f1_line"]), float(r["f2_line"])))
    return out


def fit_model():
    df0, X0, y0 = stage2.load()
    Xc = X0.drop(columns=PASSC)
    tr = (df0.date < "2026-01-01").values
    m = make_pipeline(StandardScaler(with_mean=False),
                      LogisticRegression(C=0.3, max_iter=3000, fit_intercept=False))
    m.fit(np.vstack([Xc[tr].values, -Xc[tr].values]),
          np.concatenate([y0[tr], 1 - y0[tr]]))
    return m


def pm_lookup(bouts, ev_date_s, future):
    """Polymarket moneyline per BFO bout (mu -> polymarket record oriented
    f1-first, or None) and, for a future card, the bouts Polymarket lists for
    the date that matched no BFO bout. Network trouble is printed, never
    fatal: Polymarket is a second price source, not a dependency."""
    by_mu, extra = {}, []
    for b in bouts:
        try:
            by_mu[b["mu"]] = polymarket.find_bout(b["f1"], b["f2"], ev_date_s, refresh=future)
        except Exception as e:  # noqa: BLE001
            print(f"  PM ERR {b['f1']} / {b['f2']}: {e!r}")
            by_mu[b["mu"]] = None
    if future:
        try:
            listed = polymarket.card_bouts(ev_date_s, refresh=True)
        except Exception as e:  # noqa: BLE001
            print(f"  PM ERR card listing: {e!r}")
            listed = []
        matched = {r["slug"] for r in by_mu.values() if r}
        extra = [r for r in listed if r["slug"] not in matched]
    return by_mu, extra


def project(items, find, snap, model, ev_date_s):
    """items: (key, name1, name2) -> (preds {key: P(name1)}, rds {key: (rd1, rd2)},
    ids {key: (id1, id2)}) for the items whose fighters both resolve. One
    Stage 2 frame per call (build_matrix centres two terms per frame)."""
    rows, keys, rds, ids = [], [], {}, {}
    for key, n1, n2 in items:
        i1, i2 = find(n1), find(n2)
        if not (i1 and i2):
            continue
        ids[key] = (i1, i2)
        p1, s1 = snap(i1)
        p2, s2 = snap(i2)
        rows.append(dict(fight_id=str(key), date=ev_date_s, method="",
                         weight_class="", title=0, label=1,
                         p_glicko=engine.predict(p1, p2),
                         rating_diff=(p1.mu - p2.mu) * engine.SCALE,
                         rd_x=p1.phi * engine.SCALE, rd_y=p2.phi * engine.SCALE,
                         **{f"x_{k}": v for k, v in s1.items()},
                         **{f"y_{k}": v for k, v in s2.items()}))
        keys.append(key)
        rds[key] = (p1.phi * engine.SCALE, p2.phi * engine.SCALE)
    if not rows:
        return {}, rds, ids
    R = pd.DataFrame(rows)
    X = stage2.build_matrix(R).drop(columns=PASSC)
    p = model.predict_proba(X.values)[:, 1]
    return dict(zip(keys, (float(x) for x in p))), rds, ids


def eval_rule(p, o1, o2):
    """The betting_system rule at one opening pair -> (rule, picked_is_f1,
    conf) or None (pre-filter, FLIP, GAP; the frozen constants)."""
    if not 1.0 <= 1 / o1 + 1 / o2 <= 1.12:
        return None                             # odds-sanity pre-filter
    mkt1, mod1 = (1 / o1) >= (1 / o2), p >= .5
    conf = p if mod1 else 1 - p
    gap = float(am_dec(o1 if mkt1 else o2)) - float(amp_prob(p if mkt1 else 1 - p))
    if mkt1 != mod1 and conf >= FLIP_CONF:
        return "FLIP", mod1, conf
    if mkt1 == mod1 and gap >= GAP_PTS:
        return "GAP", mod1, conf
    return None


def signals(bouts, preds, rds, future):
    """mu -> signal for every projected bout the rule fires on at the
    validated open, with the placement policy v1 stake (stake_units) next to
    it. The stake's line is the price the ledger will settle at: the earliest
    capture on file for the bout, else the current line for a future card
    (this run's capture), else open. A suspect-opener bout where only the
    raw first tick fires gets a VOID entry (tag "void: suspect opener",
    stake 0): logged, never bet."""
    cap = earliest_capture()
    out = {}
    for b in bouts:
        if b["mu"] not in preds:
            continue
        p = preds[b["mu"]]
        rd1, rd2 = rds[b["mu"]]
        hit = eval_rule(p, b["f1_open"], b["f2_open"])
        if hit:
            rule, mod1, conf = hit
            rd_self, rd_opp = (rd1, rd2) if mod1 else (rd2, rd1)
            c = cap.get(b["mu"])
            if c is not None:
                dec = c[0] if mod1 else c[1]
            elif future:
                dec = b["f1_close"] if mod1 else b["f2_close"]
            else:
                dec = b["f1_open"] if mod1 else b["f2_open"]
            line = int(round(float(am_dec(dec))))
            u = stake_units(rd_self, rd_opp, line)
            if u > 0:
                tag = f"{u:.2f}u"
            elif line > SKIP_LINE_MAX:
                tag = f"skip: {line:+d}"
            else:
                tag = "skip: both unknown"
            opn = int(round(float(am_dec(b["f1_open"] if mod1 else b["f2_open"]))))
            out[b["mu"]] = dict(rule=rule, who=b["f1"] if mod1 else b["f2"], conf=conf,
                                opn=opn, line=line, rd_self=rd_self, rd_opp=rd_opp,
                                tag=tag, stake=u, void=False)
        elif b.get("open_suspect"):
            raw = eval_rule(p, b["f1_open_raw"], b["f2_open_raw"])
            if raw:
                rule, mod1, conf = raw
                rd_self, rd_opp = (rd1, rd2) if mod1 else (rd2, rd1)
                opn = int(round(float(am_dec(b["f1_open_raw"] if mod1 else b["f2_open_raw"]))))
                out[b["mu"]] = dict(rule=rule, who=b["f1"] if mod1 else b["f2"], conf=conf,
                                    opn=opn, line=opn, rd_self=rd_self, rd_opp=rd_opp,
                                    tag=VOID_SUSPECT, stake=0.0, void=True)
    return out


def main(slug, ev_date_s):
    ev_date = date.fromisoformat(ev_date_s)
    eng, car, meta = build_states(ev_date)
    # BFO name -> fighter id: aliases, suffix strip, exact / joined / fuzzy (names.py)
    find = make_finder((fid, fo.name) for fid, fo in eng.fighters.items())

    def snap(fid):
        fo = eng.fighters[fid]
        n = (ev_date - fo.last_fight).days / engine.YEAR if fo.last_fight else 0.0
        pre = engine.PreState(fo.mu, engine.inflate(fo.phi, fo.sigma, max(n, 0)),
                              fo.sigma)
        mt = meta.get(fid, {})
        return pre, car[fid].snapshot(ev_date, mt.get("dob"), mt.get("height"),
                                      mt.get("reach"), mt.get("stance"))

    future = ev_date >= date.today()
    bouts = get_card(slug, fresh=future)        # live prices for a future card
    off = [b for b in bouts if (ev_date_s, b["mu"]) in OFF_CARD]
    for b in off:
        print(f"off card: dropped {b['f1']} / {b['f2']} (mu {b['mu']})")
    bouts = [b for b in bouts if (ev_date_s, b["mu"]) not in OFF_CARD]
    pm_by_mu, pm_extra = pm_lookup(bouts, ev_date_s, future)
    model = fit_model()
    preds, rds, ids = project([(b["mu"], b["f1"], b["f2"]) for b in bouts],
                              find, snap, model, ev_date_s)
    keep = [b for b in bouts if b["mu"] in preds]
    xpreds, _, xids = project([(r["slug"], r["outcomes"][0], r["outcomes"][1])
                               for r in pm_extra], find, snap, model, ev_date_s)

    # results if the event is in our data: by fighter id where both sides were
    # found (spelling-proof: "Dooho Choi" vs BFO "Doo Ho Choi"), else by name pair
    fv = pd.read_csv("data/fights_v2.csv")
    fv = fv[fv.event_date == ev_date_s]
    win, win_id = {}, {}
    for _, f in fv.iterrows():
        if f.outcome == "A_WIN":
            win[tuple(sorted([norm(f.fighter_a), norm(f.fighter_b)]))] = \
                (norm(f.fighter_a), f.method)
            win_id[frozenset((f.fighter_a_id, f.fighter_b_id))] = \
                (f.fighter_a_id, norm(f.fighter_a), f.method)

    def result(n1, n2, i12):
        """(result text, f1 won or None) for a bout."""
        r = win_id.get(frozenset(i12)) if i12 else None
        if r:
            won1, r = r[0] == i12[0], r[1:]
        else:
            r = win.get(tuple(sorted([bfo_norm(n1), bfo_norm(n2)])))
            won1 = bool(r) and r[0] == bfo_norm(n1)
        return (f"{r[0].split()[-1].title()} ({r[1]})" if r else "-"), (won1 if r else None)

    am = lambda d: f"+{round((d-1)*100)}" if d >= 2 else f"{round(-100/(d-1))}"
    amp = lambda p: f"-{round(100*p/(1-p))}" if p >= .5 else f"+{round(100*(1-p)/p)}"
    vf = lambda d1, d2: (1/d1) / (1/d1 + 1/d2)
    pm_txt = lambda rec: (f"{rec['p_a']:.0%}" if rec and rec.get("p_a") is not None else "-")
    sig = signals(keep, preds, rds, future)
    print(f"{'FIGHT':40s}{'MODEL %':>13}{'FAIR':>12}{'OPEN':>12}{'CLOSE':>12}{'PM':>6}"
          f"{'CLV':>8}  {'RESULT':18s} SIGNAL")
    clvs, w, n = [], 0, 0
    for b in bouts:
        rt, won1 = result(b["f1"], b["f2"], ids.get(b["mu"]))
        line = f"{am(b['f1_open'])}/{am(b['f2_open'])}"
        cl = f"{am(b['f1_close'])}/{am(b['f2_close'])}"
        pmc = pm_txt(pm_by_mu.get(b["mu"]))
        s = sig.get(b["mu"])
        stxt = f"{s['rule']} {s['tag']}" if s else ("(suspect opener)" if b.get("open_suspect") else "")
        if b["mu"] not in preds:
            print(f"{b['f1']+' / '+b['f2']:40s}{'debut-no proj':>13}{'':>12}"
                  f"{line:>12}{cl:>12}{pmc:>6}{'-':>8}  {rt:18s} {stxt}")
            continue
        p = preds[b["mu"]]
        qo, qc = vf(b["f1_open"], b["f2_open"]), vf(b["f1_close"], b["f2_close"])
        clv = (qc - qo) if p > qo else (qo - qc)
        clvs.append(clv)
        ok = ""
        if won1 is not None:
            n += 1
            hit = (p > 0.5) == won1
            w += hit
            ok = " W" if hit else " L"
        print(f"{b['f1']+' / '+b['f2']:40s}{f'{p:.0%}/{1-p:.0%}':>13}"
              f"{amp(p)+'/'+amp(1-p):>12}{line:>12}{cl:>12}{pmc:>6}{clv*100:>+7.1f}p"
              f"  {rt + ok:18s} {stxt}")
    for r in pm_extra:                          # Polymarket-only bouts: BFO columns blank
        n1, n2 = r["outcomes"]
        rt, won1 = result(n1, n2, xids.get(r["slug"]))
        pmc = pm_txt(r)
        if r["slug"] not in xpreds:
            print(f"{n1+' / '+n2:40s}{'debut-no proj':>13}{'':>12}{'':>12}{'':>12}{pmc:>6}"
                  f"{'-':>8}  {rt:18s} no BFO line")
            continue
        p = xpreds[r["slug"]]
        print(f"{n1+' / '+n2:40s}{f'{p:.0%}/{1-p:.0%}':>13}{amp(p)+'/'+amp(1-p):>12}"
              f"{'':>12}{'':>12}{pmc:>6}{'-':>8}  {rt:18s} no BFO line")
    for b in bouts:
        if b.get("open_suspect"):
            print(f"  suspect opener: {b['f1']} / {b['f2']} (mu {b['mu']}) raw "
                  f"{am(b['f1_open_raw'])}/{am(b['f2_open_raw'])} -> open "
                  f"{am(b['f1_open'])}/{am(b['f2_open'])}")
    if clvs:
        print(f"\ncard CLV: {np.mean(clvs)*100:+.2f} pts over {len(clvs)} projected"
              + (f" | picks {w}-{n-w}" if n else ""))
    n_pm = sum(1 for r in pm_by_mu.values() if r) + len(pm_extra)
    print(f"Polymarket: {n_pm} bouts with a market ({sum(1 for r in pm_by_mu.values() if r)} "
          f"matched to BFO, {len(pm_extra)} not on BFO); BFO: {len(bouts)} bouts")
    print_signals(sig)
    if future:
        record_placeable(slug, ev_date_s, bouts, pm_by_mu)


def print_signals(sig):
    """The rule-fired picks (signals) with the placement policy v1 stake next
    to each, VOID entries last."""
    live = [s for s in sig.values() if not s["void"]]
    void = [s for s in sig.values() if s["void"]]
    placed = sum(1 for s in live if s["stake"] > 0)
    print()
    print(f"signals at open (FLIP >= {FLIP_CONF:.2f} / GAP >= {GAP_PTS} pts): {len(live)}"
          f"  placed (policy v1): {placed}" + (f"  void: {len(void)}" if void else ""))
    for s in live + void:
        print(f"  {s['rule']:4s} {s['who']:28s} {s['conf']:.0%}  open {s['opn']:+d}  "
              f"line {s['line']:+d}  RD {s['rd_self']:.0f}/{s['rd_opp']:.0f}  {s['tag']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
