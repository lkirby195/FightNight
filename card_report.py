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
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import date, datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import engine
import features
import stage2
from scrape_bfo import fetch, ggd, series_summary

PASSC = ["d_ss_acc", "d_ss_def", "d_td_acc", "d_td_def", "d_ctrl15",
         "d_pace15", "d_head_share", "d_leg_share", "d_ground_share",
         "d_ctrled15"]


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower().replace("-", " ")).strip()


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
    the price on the board now, not whenever the page was first cached)."""
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
            s1 = series_summary(ggd(mu, 1, refresh=fresh))
            s2 = series_summary(ggd(mu, 2, refresh=fresh))
            if s1 and s2:
                bouts.append(dict(mu=mu, f1=a1.get_text(strip=True),
                                  f2=a2.get_text(strip=True),
                                  f1_open=s1[0], f1_close=s1[1],
                                  f2_open=s2[0], f2_close=s2[1]))
    return bouts


PLACEABLE = "data/placeable_lines.csv"
PLACEABLE_COLS = ["event_date", "bfo_slug", "mu", "fighter1", "fighter2",
                  "f1_line", "f2_line", "captured_at"]


def record_placeable(slug, ev_date_s, bouts, path=PLACEABLE):
    """Append the current line of every bout of a FUTURE event to `path`: the
    price that could actually be bet when the pick went on record
    (betting_system.py settles live bets at the EARLIEST capture per bout).
    Append-only: rows already on file are never rewritten or replaced. One row
    per bout per run, stamped with the run's captured_at; a bout whose lines
    are identical to a row already on file (same mu, f1_line, f2_line) is
    skipped, so a re-run that finds nothing moved adds nothing. Never called
    for past events."""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    seen = set()
    have = os.path.exists(path) and os.path.getsize(path) > 0
    if have:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                seen.add((r["mu"], float(r["f1_line"]), float(r["f2_line"])))
    new = [b for b in bouts
           if (str(b["mu"]), float(b["f1_close"]), float(b["f2_close"])) not in seen]
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PLACEABLE_COLS)
        if not have:
            w.writeheader()
        for b in new:
            w.writerow(dict(event_date=ev_date_s, bfo_slug=slug, mu=str(b["mu"]),
                            fighter1=b["f1"], fighter2=b["f2"],
                            f1_line=b["f1_close"], f2_line=b["f2_close"],
                            captured_at=now))
    print(f"placeable lines -> {path}: {len(new)} appended, "
          f"{len(bouts) - len(new)} unchanged ({now})")


def fit_model():
    df0, X0, y0 = stage2.load()
    Xc = X0.drop(columns=PASSC)
    tr = (df0.date < "2026-01-01").values
    m = make_pipeline(StandardScaler(with_mean=False),
                      LogisticRegression(C=0.3, max_iter=3000, fit_intercept=False))
    m.fit(np.vstack([Xc[tr].values, -Xc[tr].values]),
          np.concatenate([y0[tr], 1 - y0[tr]]))
    return m


def main(slug, ev_date_s):
    ev_date = date.fromisoformat(ev_date_s)
    eng, car, meta = build_states(ev_date)
    byname, byjoined = {}, {}
    for fid, fo in eng.fighters.items():
        byname.setdefault(norm(fo.name), fid)
        byjoined.setdefault(norm(fo.name).replace(" ", ""), fid)

    def find(name):
        n = norm(name)
        if n in byname:
            return byname[n]
        if n.replace(" ", "") in byjoined:      # BFO "Sangcha-An" vs "Sangcha'an"
            return byjoined[n.replace(" ", "")]
        t = n.split()
        c = [fid for nm, fid in byname.items()
             if nm.endswith(" " + t[-1]) and nm.split()[0][:3] == t[0][:3]]
        return c[0] if len(c) == 1 else None

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
    rows, keep = [], []
    for b in bouts:
        i1, i2 = find(b["f1"]), find(b["f2"])
        if not (i1 and i2):
            continue
        p1, s1 = snap(i1)
        p2, s2 = snap(i2)
        rows.append(dict(fight_id=b["mu"], date=ev_date_s, method="",
                         weight_class="", title=0, label=1,
                         p_glicko=engine.predict(p1, p2),
                         rating_diff=(p1.mu - p2.mu) * engine.SCALE,
                         rd_x=p1.phi * engine.SCALE, rd_y=p2.phi * engine.SCALE,
                         **{f"x_{k}": v for k, v in s1.items()},
                         **{f"y_{k}": v for k, v in s2.items()}))
        keep.append(b)
    R = pd.DataFrame(rows)
    X = stage2.build_matrix(R).drop(columns=PASSC)
    R["p"] = fit_model().predict_proba(X.values)[:, 1]
    preds = dict(zip(R.fight_id.astype(int), R.p))

    # results if the event is in our data
    fv = pd.read_csv("data/fights_v2.csv")
    fv = fv[fv.event_date == ev_date_s]
    win = {}
    for _, f in fv.iterrows():
        if f.outcome == "A_WIN":
            win[tuple(sorted([norm(f.fighter_a), norm(f.fighter_b)]))] = \
                (norm(f.fighter_a), f.method)

    am = lambda d: f"+{round((d-1)*100)}" if d >= 2 else f"{round(-100/(d-1))}"
    amp = lambda p: f"-{round(100*p/(1-p))}" if p >= .5 else f"+{round(100*(1-p)/p)}"
    vf = lambda d1, d2: (1/d1) / (1/d1 + 1/d2)
    print(f"{'FIGHT':40s}{'MODEL %':>13}{'FAIR':>12}{'OPEN':>12}{'CLOSE':>12}"
          f"{'CLV':>8}  RESULT")
    clvs, w, n = [], 0, 0
    for b in bouts:
        key = tuple(sorted([norm(b["f1"]), norm(b["f2"])]))
        r = win.get(key)
        rt = f"{r[0].split()[-1].title()} ({r[1]})" if r else "-"
        line = f"{am(b['f1_open'])}/{am(b['f2_open'])}"
        cl = f"{am(b['f1_close'])}/{am(b['f2_close'])}"
        if b["mu"] not in preds:
            print(f"{b['f1']+' / '+b['f2']:40s}{'debut-no proj':>13}{'':>12}"
                  f"{line:>12}{cl:>12}{'-':>8}  {rt}")
            continue
        p = preds[b["mu"]]
        qo, qc = vf(b["f1_open"], b["f2_open"]), vf(b["f1_close"], b["f2_close"])
        clv = (qc - qo) if p > qo else (qo - qc)
        clvs.append(clv)
        ok = ""
        if r:
            n += 1
            hit = (p > 0.5) == (r[0] == norm(b["f1"]))
            w += hit
            ok = " W" if hit else " L"
        print(f"{b['f1']+' / '+b['f2']:40s}{f'{p:.0%}/{1-p:.0%}':>13}"
              f"{amp(p)+'/'+amp(1-p):>12}{line:>12}{cl:>12}{clv*100:>+7.1f}p"
              f"  {rt}{ok}")
    if clvs:
        print(f"\ncard CLV: {np.mean(clvs)*100:+.2f} pts over {len(clvs)} projected"
              + (f" | picks {w}-{n-w}" if n else ""))
    if future:
        record_placeable(slug, ev_date_s, bouts)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
