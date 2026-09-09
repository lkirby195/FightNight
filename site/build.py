"""SPLIT DECISION: static site generator for the FightNight public record.

Usage:  python site/build.py [--as-of YYYY-MM-DD] [--out DIR]

Read-only over the repo's data files:
  data/fights_v2.csv, data/prefight_rd.csv, data/system_ledger.csv,
  data/system_ledger_2025_backtest.csv, data/model_only_preds.csv,
  data/model_only_preds_2025_backtest.csv, data/bfo_joined.csv,
  data/bfo_lines.csv, data/placeable_lines.csv, data/clv_eval_full.csv,
  best_params.json, site/division_overrides.csv
Writes site/out/ (the deployable static site) and site/state/ranks_prev.json
(the rank snapshot behind the movement column).

Run card_report.py for the upcoming event first: the This-week page is built
from its output, data/placeable_lines.csv, and projects each captured bout
with the same state replay and Stage 2 fit card_report uses, so the numbers
on the page are the numbers that were printed when the pick went on record.

The build fails if any output file contains a string from LINT.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import card_report as cr
import engine
import stage2
from betting_system import (FLIP_CONF, GAP_PTS, RAMP_HI, RAMP_LO, SKIP_LINE_MAX,
                            UNKNOWN_RD, is_live, stake_units)
from replay_check import apply_params

BRAND = "SPLIT DECISION"
UNIT = 100                      # dollars per unit
TOP_N = 15                      # rows per ranking table
ELIGIBLE_MONTHS = 18            # last fight within this many months
RISING_RD = 160                 # "Rising" = RD above this
COIN = (0.48, 0.52)             # model p in this band = coin flip
RULE_FREEZE, POLICY_FREEZE = "2026-09-08", "2026-09-09"
LINT = ["lkirby", "Logan", "github.com/lkirby195"]

DIVISIONS = ["Flyweight", "Bantamweight", "Featherweight", "Lightweight",
             "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
             "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
             "Women's Featherweight"]
NON_DIV = {"Catch Weight", "Open Weight", "Super Heavyweight"}
METHOD = {"KO/TKO": "KO/TKO", "SUB": "Submission", "U-DEC": "Unanimous decision",
          "S-DEC": "Split decision", "M-DEC": "Majority decision", "DQ": "DQ",
          "Overturned": "Overturned", "CNC": "No contest", "Other": "Other"}
LEDGER_PUBLIC_DROP = {"live", "units", "pnl_at_open"}

DATA = ROOT / "data"
OUT_DEFAULT = SITE / "out"
STATE = SITE / "state" / "ranks_prev.json"
OVERRIDES = SITE / "division_overrides.csv"


# ----------------------------------------------------------------- helpers --
def vigfree(d1: float, d2: float) -> tuple[float, float]:
    a, b = 1 / d1, 1 / d2
    return a / (a + b), b / (a + b)


def american(dec: float) -> int:
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


def fair_line(p: float) -> int:
    return int(round(-100 * p / (1 - p))) if p >= .5 else int(round(100 * (1 - p) / p))


def sline(x) -> str:
    return "" if x is None or x == "" or (isinstance(x, float) and np.isnan(x)) \
        else f"{int(x):+d}"


def pair(x1, x2) -> str:
    return f"{sline(x1)} / {sline(x2)}" if x1 is not None and x2 is not None else ""


def pct(q: float) -> int:
    return int(round(q * 100))


def money(units: float, signed: bool = False) -> str:
    d = int(round(units * UNIT))
    if d < 0:
        return f"−${-d:,}"
    return f"+${d:,}" if signed and d > 0 else f"${d:,}"


def surname(name: str) -> str:
    t = name.split()
    if len(t) > 1 and t[-1].rstrip(".").lower() in ("jr", "sr", "ii", "iii"):
        t = t[:-1]
    return t[-1] if t else name


def day_text(d: date) -> str:
    return f"{d:%a %b} {d.day}, {d.year}"


def short_day(d: date) -> str:
    return f"{d:%a %b} {d.day}"


def months_ago(d: date, n: int) -> date:
    y, m = d.year, d.month - n
    while m <= 0:
        y, m = y - 1, m + 12
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def market_text(q1: float, q2: float, n1: str, n2: str):
    """('Name 62%', favourite name) from a vig-free pair; ('Even 50%', None)."""
    p1, p2 = pct(q1), pct(q2)
    if p1 == p2:
        return "Even 50%", None
    return (f"{surname(n1)} {p1}%", n1) if p1 > p2 else (f"{surname(n2)} {p2}%", n2)


def event_display(fights_name: str | None, slug: str | None, titles: dict) -> str:
    """Public event name: fights_v2 name, else the BFO page title, else the slug."""
    if fights_name:
        return fights_name
    t = titles.get(slug, "")
    t = t.split(" | ")[0]
    if " for " in t:
        t = t.split(" for ")[0]
    if t.endswith(" Odds"):
        t = t[:-5]
    if t:
        return t
    s = re.sub(r"-\d+$", "", slug or "next card").replace("-", " ").title()
    return s.replace("Ufc", "UFC")


# ------------------------------------------------------------ model replay --
class Model:
    """Year models (Stage 2 fit on fights strictly before Jan 1 of the year,
    exactly clv_eval.gen_model_preds) and per-date state replays
    (card_report.build_states), both cached."""

    def __init__(self):
        self._loaded = None
        self._models: dict[int, object] = {}
        self._states: dict[date, tuple] = {}

    def _data(self):
        if self._loaded is None:
            df0, X0, y0 = stage2.load()
            self._loaded = (df0, X0.drop(columns=cr.PASSC), y0)
        return self._loaded

    def year_model(self, year: int):
        if year not in self._models:
            df0, Xc, y0 = self._data()
            tr = (df0.date < f"{year}-01-01").values
            m = make_pipeline(StandardScaler(with_mean=False),
                              LogisticRegression(C=0.3, max_iter=3000, fit_intercept=False))
            m.fit(np.vstack([Xc[tr].values, -Xc[tr].values]),
                  np.concatenate([y0[tr], 1 - y0[tr]]))
            self._models[year] = m
        return self._models[year]

    def states(self, ev: date):
        if ev not in self._states:
            self._states[ev] = cr.build_states(ev)
        return self._states[ev]

    def snap(self, ev: date, fid: str, name: str):
        eng, car, meta = self.states(ev)
        fo = eng.fighter(fid, name)          # debutant -> default state
        n = (ev - fo.last_fight).days / engine.YEAR if fo.last_fight else 0.0
        pre = engine.PreState(fo.mu, engine.inflate(fo.phi, fo.sigma, max(n, 0)), fo.sigma)
        mt = meta.get(fid, {})
        return pre, car[fid].snapshot(ev, mt.get("dob"), mt.get("height"),
                                      mt.get("reach"), mt.get("stance"))

    def project(self, ev: date, bouts: list[tuple[str, str, str, str]], model=None):
        """bouts: (id1, name1, id2, name2) in card order -> list of
        (p1, rd1, rd2). The whole card is one frame, as in card_report.main."""
        rows = []
        for i1, n1, i2, n2 in bouts:
            p1, s1 = self.snap(ev, i1, n1)
            p2, s2 = self.snap(ev, i2, n2)
            rows.append(dict(fight_id=f"{i1}-{i2}", date=ev.isoformat(), method="",
                             weight_class="", title=0, label=1,
                             p_glicko=engine.predict(p1, p2),
                             rating_diff=(p1.mu - p2.mu) * engine.SCALE,
                             rd_x=p1.phi * engine.SCALE, rd_y=p2.phi * engine.SCALE,
                             **{f"x_{k}": v for k, v in s1.items()},
                             **{f"y_{k}": v for k, v in s2.items()}))
        if not rows:
            return []
        R = pd.DataFrame(rows)
        X = stage2.build_matrix(R).drop(columns=cr.PASSC)
        m = model or self.year_model(ev.year)
        p = m.predict_proba(X.values)[:, 1]
        return [(float(p[i]), float(R.rd_x[i]), float(R.rd_y[i])) for i in range(len(R))]

    def finder(self, ev: date):
        """Name -> fighter id over the fighters with history before ev
        (card_report.main's find)."""
        eng, _, _ = self.states(ev)
        byname, byjoined = {}, {}
        for fid, fo in eng.fighters.items():
            byname.setdefault(cr.norm(fo.name), fid)
            byjoined.setdefault(cr.norm(fo.name).replace(" ", ""), fid)

        def find(name):
            n = cr.norm(name)
            if n in byname:
                return byname[n]
            if n.replace(" ", "") in byjoined:
                return byjoined[n.replace(" ", "")]
            t = n.split()
            c = [fid for nm, fid in byname.items()
                 if nm.endswith(" " + t[-1]) and nm.split()[0][:3] == t[0][:3]]
            return c[0] if len(c) == 1 else None
        return find


# ------------------------------------------------------------------ inputs --
class Data:
    def __init__(self, as_of: date):
        self.as_of = as_of
        self.fights = read_csv(DATA / "fights_v2.csv")
        self.fights.sort(key=lambda r: (r["event_date"], int(r["bout_order"])))
        self.data_through = max(f["event_date"] for f in self.fights)
        self.preds = {r["fight_id"]: float(r["p_model_a"])
                      for r in read_csv(DATA / "model_only_preds.csv")}
        self.preds_bt = {r["fight_id"]: float(r["p_model_a"])
                         for r in read_csv(DATA / "model_only_preds_2025_backtest.csv")}
        self.lines = {r["fight_id"]: {k: float(r[k]) for k in
                                      ("a_open", "a_close", "b_open", "b_close")}
                      for r in read_csv(DATA / "bfo_joined.csv")}
        self.rd = {r["fight_id"]: (float(r["rd_a"]), float(r["rd_b"]))
                   for r in read_csv(DATA / "prefight_rd.csv")}
        self.ledger = read_csv(DATA / "system_ledger.csv")
        self.ledger_bt = read_csv(DATA / "system_ledger_2025_backtest.csv")
        for r in self.ledger + self.ledger_bt:
            for k in ("stake_u", "pnl_placed", "pnl", "clv_pts", "rd_self", "rd_opp"):
                r[k] = float(r[k]) if r[k] != "" else None
            r["placed"] = int(r["placed"])
            r["live"] = int(r["live"])
        # placeable captures: per mu sorted by captured_at, per event in card order
        self.captures: dict[int, list[dict]] = defaultdict(list)
        self.capture_events: dict[str, dict] = {}
        for r in read_csv(DATA / "placeable_lines.csv"):
            mu = int(r["mu"])
            self.captures[mu].append(r)
            ev = self.capture_events.setdefault(
                r["event_date"], dict(slug=r["bfo_slug"], mus=[], posted=r["captured_at"]))
            if mu not in ev["mus"]:
                ev["mus"].append(mu)
            ev["posted"] = min(ev["posted"], r["captured_at"])
        for c in self.captures.values():
            c.sort(key=lambda r: r["captured_at"])
        # BFO line file: opening lines by mu, page titles by slug
        self.bfo = {}
        self.titles = {}
        for r in read_csv(DATA / "bfo_lines.csv"):
            self.bfo[int(r["mu"])] = r
            self.titles.setdefault(r["slug"], r["event_title"])
        self.overrides = {r["fighter_id"]: r["division"] for r in read_csv(OVERRIDES)
                          if r.get("fighter_id")}

    def capture_index(self, ev_date: str) -> dict[frozenset, int]:
        """Name-pair -> mu for captures within a day of ev_date."""
        d = date.fromisoformat(ev_date)
        out = {}
        for cd, ev in self.capture_events.items():
            if abs((date.fromisoformat(cd) - d).days) <= 1:
                for mu in ev["mus"]:
                    r = self.captures[mu][0]
                    n1, n2 = cr.norm(r["fighter1"]), cr.norm(r["fighter2"])
                    out[frozenset((n1, n2))] = mu
                    out.setdefault(frozenset((n1.split()[-1], n2.split()[-1])), mu)
        return out

    def capture_for(self, index: dict, a: str, b: str):
        """(mu, a_is_f1) for a fights_v2 bout, or None."""
        na, nb = cr.norm(a), cr.norm(b)
        mu = index.get(frozenset((na, nb)))
        if mu is None:
            mu = index.get(frozenset((na.split()[-1], nb.split()[-1])))
        if mu is None:
            return None
        f1 = cr.norm(self.captures[mu][0]["fighter1"])
        a_is_f1 = f1 == na or f1.split()[-1] == na.split()[-1]
        return mu, a_is_f1


# ---------------------------------------------------------------- rankings --
def replay_all(fights):
    apply_params(json.load(open(ROOT / "best_params.json")))
    eng = engine.Engine()
    hist = defaultdict(list)
    for f in fights:
        eng.process({"event_date": f["event_date"],
                     "fighter_a_id": f["fighter_a_id"], "fighter_a": f["fighter_a"],
                     "fighter_b_id": f["fighter_b_id"], "fighter_b": f["fighter_b"],
                     "outcome": f["outcome"], "method": f["method"],
                     "a_missed_weight": f.get("a_mw", ""),
                     "b_missed_weight": f.get("b_mw", "")})
        d = date.fromisoformat(f["event_date"])
        for me, opp, won in ((f["fighter_a_id"], f["fighter_b"], f["outcome"] == "A_WIN"),
                             (f["fighter_b_id"], f["fighter_a"], False)):
            if f["outcome"] == "A_WIN":
                res = "W" if won else "L"
            elif f["outcome"] == "DRAW":
                res = "D"
            else:
                res = "NC"
            hist[me].append(dict(date=d, res=res, opp=opp, wc=f["weight_class"]))
    return eng, hist


def division_of(fid: str, hist: list[dict], overrides: dict) -> str | None:
    if fid in overrides:
        return overrides[fid]
    last3 = hist[-3:]
    wcs = [h["wc"] for h in last3 if h["wc"] not in NON_DIV]
    if not wcs:
        return None
    c = Counter(wcs)
    best = max(c.values())
    for h in reversed(last3):                   # most recent among the tied
        if c.get(h["wc"]) == best:
            return h["wc"]
    return None


def last_fight_text(h: dict) -> str:
    verb = {"W": "Beat", "L": "Lost to", "D": "Drew with", "NC": "No contest vs"}[h["res"]]
    return f"{verb} {h['opp']}"


def movement(prev: int | None, rank: int) -> str:
    if prev is None:
        return "new"
    if prev == rank:
        return "–"
    return f"▲{prev - rank}" if prev > rank else f"▼{rank - prev}"


def build_rankings(data: Data):
    eng, hist = replay_all(data.fights)
    cutoff = months_ago(data.as_of, ELIGIBLE_MONTHS)
    cands = []
    for fid, fo in eng.fighters.items():
        if not fo.last_fight or fo.last_fight < cutoff:
            continue
        h = hist[fid]
        cands.append(dict(fid=fid, name=fo.name, rating=fo.rating, rd=fo.rd,
                          score=fo.rating - fo.rd,
                          division=division_of(fid, h, data.overrides),
                          last3=[x["res"] for x in h[-3:]],
                          last=last_fight_text(h[-1]), last_date=short_day(h[-1]["date"])))
    # previous ranks (rotated only when the data moves)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get("data_through") == data.data_through:
        keep_prev = state.get("prev")           # same data: keep comparing to it
    else:
        keep_prev = ({"data_through": state["data_through"], "ranks": state["ranks"]}
                     if state else None)
    prev = keep_prev["ranks"] if keep_prev else None    # None: no earlier build
    lists, ranks_now = [], {}
    groups = [("p4p", "Pound for pound", cands)] + \
             [(re.sub(r"[^a-z]+", "-", d.lower()).strip("-"), d,
               [c for c in cands if c["division"] == d]) for d in DIVISIONS]
    for key, label, pool in groups:
        rows = sorted(pool, key=lambda c: -c["score"])[:TOP_N]
        if not rows:
            continue
        pr = prev.get(key, {}) if prev is not None else None
        ranks_now[key] = {r["fid"]: i for i, r in enumerate(rows, 1)}
        lists.append(dict(key=key, label=label, rows=[
            dict(r, rank=i, move=movement(pr.get(r["fid"]), i) if pr is not None else "–",
                 rating_i=int(round(r["rating"])), rd_i=int(round(r["rd"])))
            for i, r in enumerate(rows, 1)]))
    p4p = lists[0]["rows"]
    explainer = None
    if len(p4p) > 1 and p4p[0]["rating"] < p4p[1]["rating"]:
        explainer = dict(one=p4p[0], two=p4p[1])
    rising = sorted([c for c in cands if c["rd"] > RISING_RD],
                    key=lambda c: -c["rating"])[:5]
    rising = [dict(c, rating_i=int(round(c["rating"])), rd_i=int(round(c["rd"])))
              for c in rising]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(dict(data_through=data.data_through, as_of=data.as_of.isoformat(),
                                     ranks=ranks_now, prev=keep_prev), indent=1))
    return dict(lists=lists, explainer=explainer, rising=rising,
                eligible_since=day_text(cutoff), n_eligible=len(cands))


# --------------------------------------------------------------- this week --
def pick_of(p1: float, n1: str, n2: str):
    """(text, side, coin) for P(f1) = p1; side is the picked name or None."""
    if COIN[0] <= p1 <= COIN[1]:
        return "Coin flip · 50%", None, True
    if p1 >= .5:
        return f"{surname(n1)} {pct(p1)}%", n1, False
    return f"{surname(n2)} {pct(1 - p1)}%", n2, False


def signal_at_open(p1: float, o1: float, o2: float):
    """betting_system rule at the opening line -> (rule, picked_is_f1) or None."""
    if not 1.0 <= 1 / o1 + 1 / o2 <= 1.12:
        return None
    mkt1, mod1 = (1 / o1) >= (1 / o2), p1 >= .5
    conf = p1 if mod1 else 1 - p1
    am = lambda d: (d - 1) * 100 if d >= 2 else -100 / (d - 1)
    amp = lambda q: -100 * q / (1 - q) if q >= .5 else 100 * (1 - q) / q
    gap = am(o1 if mkt1 else o2) - amp(p1 if mkt1 else 1 - p1)
    if mkt1 != mod1 and conf >= FLIP_CONF:
        return "FLIP", mod1
    if mkt1 == mod1 and gap >= GAP_PTS:
        return "GAP", mod1
    return None


def build_this_week(data: Data, model: Model):
    future = sorted(d for d in data.capture_events if d >= data.as_of.isoformat())
    if not future:
        nxt = sorted((r["event_date"], r["slug"]) for r in data.bfo.values()
                     if r["event_date"] > data.as_of.isoformat())
        name = event_display(None, nxt[0][1] if nxt else None, data.titles)
        return dict(event=None, next_name=name)
    ev_date = future[0]
    ev = data.capture_events[ev_date]
    d = date.fromisoformat(ev_date)
    find = model.finder(d)
    eng = model.states(d)[0]
    disp = lambda fid, bfo_name: eng.fighters[fid].name if fid else bfo_name
    bouts, proj_in = [], []
    for mu in ev["mus"]:
        first, last = data.captures[mu][0], data.captures[mu][-1]
        i1, i2 = find(first["fighter1"]), find(first["fighter2"])
        b = dict(mu=mu, f1=disp(i1, first["fighter1"]), f2=disp(i2, first["fighter2"]),
                 i1=i1, i2=i2, first=first, last=last, main=not bouts)
        if i1 and i2:
            proj_in.append((i1, b["f1"], i2, b["f2"]))
            b["proj_idx"] = len(proj_in) - 1
        bouts.append(b)
    proj = model.project(d, proj_in, model=cr.fit_model())
    rows, n_bets, stakes = [], 0, []
    for b in bouts:
        f1, f2 = b["f1"], b["f2"]
        q1, q2 = vigfree(float(b["last"]["f1_line"]), float(b["last"]["f2_line"]))
        odds_text, fav = market_text(q1, q2, f1, f2)
        row = dict(f1=f1, f2=f2, main=b["main"], odds_text=odds_text,
                   num=dict(placeable=pair(american(float(b["first"]["f1_line"])),
                                           american(float(b["first"]["f2_line"]))),
                            now=pair(american(float(b["last"]["f1_line"])),
                                     american(float(b["last"]["f2_line"])))))
        if "proj_idx" not in b:
            row.update(pick_text="No pick · UFC debut", tag=None,
                       bet_text="No bet", bet_kind="none")
            rows.append(row)
            continue
        p1, rd1, rd2 = proj[b["proj_idx"]]
        pick_text, side, coin = pick_of(p1, f1, f2)
        disagreed = side is not None and fav is not None and side != fav
        bf = data.bfo.get(b["mu"])
        if bf:
            o1, o2 = float(bf["f1_open"]), float(bf["f2_open"])
        else:
            print(f"  ! no opening line in bfo_lines.csv for mu {b['mu']} "
                  f"{f1} vs {f2}: using the earliest capture as open")
            o1, o2 = float(b["first"]["f1_line"]), float(b["first"]["f2_line"])
        sig = signal_at_open(p1, o1, o2)
        rule, stake, line, rd_self, rd_opp = None, 0.0, None, None, None
        if sig:
            rule, mod1 = sig
            line_dec = float(b["first"]["f1_line"] if mod1 else b["first"]["f2_line"])
            line = american(line_dec)
            rd_self, rd_opp = (rd1, rd2) if mod1 else (rd2, rd1)
            stake = round(stake_units(rd_self, rd_opp, line), 2)
        if stake > 0:
            n_bets += 1
            stakes.append(stake)
            bet_text, kind = f"{money(stake)} on {surname(side)}", "bet"
        elif disagreed:
            bet_text, kind = "Disagreed · not enough to bet", "disagree"
        else:
            bet_text, kind = "No bet · we agree", "agree"
        row.update(pick_text=pick_text, tag="Disagreed" if disagreed else None,
                   bet_text=bet_text, bet_kind=kind)
        row["num"].update(model_line=pair(fair_line(p1), fair_line(1 - p1)),
                          open=pair(american(o1), american(o2)),
                          rule=rule or "", stake_u=f"{stake:.2f}" if rule else "",
                          rd=f"{rd_self:.0f} / {rd_opp:.0f}" if rule else
                             f"{rd1:.0f} / {rd2:.0f}")
        rows.append(row)
    posted = datetime.fromisoformat(ev["posted"])
    if n_bets == 0:
        summary = ("No bets this week. Where our numbers and the odds disagree, "
                   "they don't disagree by enough to put money down.")
    elif n_bets == 1:
        summary = f"One bet this week: {money(stakes[0])}."
    else:
        summary = f"{n_bets} bets this week, {money(sum(stakes))} in play."
    return dict(event=dict(name=event_display(None, ev["slug"], data.titles),
                           date_text=day_text(d), posted_text=short_day(posted.date()),
                           summary=summary, bouts=rows, n_bets=n_bets),
                stake_note=stake_note())


def stake_note() -> str:
    return (f"Stakes run from {money(0.5)} to {money(1)} a fight, set by how well "
            f"the numbers know the fighter we're backing. We pass when the fighter "
            f"is {SKIP_LINE_MAX:+d} or longer, or when neither fighter has much of "
            f"a track record.")


# ------------------------------------------------------------------ record --
def build_cards(data: Data, model: Model, year: str, preds: dict, ledger: list,
                live_only: bool):
    """Finished-card blocks for one year, newest first."""
    by_fid = defaultdict(list)
    for r in ledger:
        by_fid[r["fight_id"]].append(r)
    events = {}
    for f in data.fights:
        if f["event_date"][:4] != year:
            continue
        if live_only and not is_live(f["event_date"]):
            continue
        events.setdefault((f["event_date"], f["event_name"]), []).append(f)
    cards = []
    for (ev_date, ev_name), bouts in sorted(events.items(), reverse=True):
        d = date.fromisoformat(ev_date)
        bouts = sorted(bouts, key=lambda f: -int(f["bout_order"]))
        # model p for every bout: preds file, else engine replay (draws, NCs)
        missing = [f for f in bouts if f["fight_id"] not in preds]
        if missing:
            proj = model.project(d, [(f["fighter_a_id"], f["fighter_a"],
                                      f["fighter_b_id"], f["fighter_b"]) for f in bouts])
            p_all = {f["fight_id"]: proj[i][0] for i, f in enumerate(bouts)}
        cidx = data.capture_index(ev_date)
        fights, bets, hits, total = [], [], 0, 0
        for f in bouts:
            a, b, fid = f["fighter_a"], f["fighter_b"], f["fight_id"]
            p_a = preds.get(fid, p_all.get(fid) if missing else None)
            cap = data.capture_for(cidx, a, b)
            ln = data.lines.get(fid)
            qa = qb = None
            placeable = ""
            if cap:
                mu, a_is_f1 = cap
                c0 = data.captures[mu][0]
                d1, d2 = float(c0["f1_line"]), float(c0["f2_line"])
                if not a_is_f1:
                    d1, d2 = d2, d1
                qa, qb = vigfree(d1, d2)
                placeable = pair(american(d1), american(d2))
            elif ln:
                qa, qb = vigfree(ln["a_open"], ln["b_open"])
            odds_text, fav = (market_text(qa, qb, a, b) if qa is not None
                              else ("—", None))
            pick_text, side, coin = pick_of(p_a, a, b) if p_a is not None \
                else ("No pick", None, False)
            disagreed = side is not None and fav is not None and side != fav
            if f["outcome"] == "A_WIN" and side is not None:
                total += 1
                won = side == a
                hits += won
                mark, mark_kind = ("✓", "hit") if won else ("✗", "miss")
            elif f["outcome"] == "DRAW":
                mark, mark_kind = "Draw", "void"
            elif f["outcome"] == "NC":
                mark, mark_kind = "No contest", "void"
            else:
                mark, mark_kind = ("Coin flip" if coin else "—"), "void"
            sig = by_fid.get(fid, [None])[0]
            rd = data.rd.get(fid)
            fights.append(dict(
                f1=a, f2=b, main=f is bouts[0], pick_text=pick_text,
                tag="Disagreed" if disagreed else None, odds_text=odds_text,
                mark=mark, mark_kind=mark_kind,
                result=f"{METHOD.get(f['method'], f['method'])} R{f['round']}",
                num=dict(model_line=pair(fair_line(p_a), fair_line(1 - p_a)) if p_a is not None else "",
                         open=pair(american(ln["a_open"]), american(ln["b_open"])) if ln else "",
                         placeable=placeable,
                         close=sline(sig["clean_close"]) if sig else "",
                         clv=f"{sig['clv_pts']:+.1f}" if sig else "",
                         rule=sig["rule"] if sig else "",
                         stake_u=f"{sig['stake_u']:.2f}" if sig else "",
                         rd=f"{rd[0]:.0f} / {rd[1]:.0f}" if rd else "")))
            for r in by_fid.get(fid, []):
                if r["placed"] != 1:
                    continue
                pick_is_a = r["fighter"] == a
                p_pick = p_a if pick_is_a else 1 - p_a
                won = r["pnl_placed"] > 0
                bets.append(dict(
                    stake=money(r["stake_u"]), pick=f"{r['fighter']} · {pct(p_pick)}%",
                    opp=b if pick_is_a else a, odds_text=odds_text,
                    result=(f"Won {money(r['pnl_placed'])}" if won
                            else f"Lost {money(-r['pnl_placed'])}"),
                    won=won, how=f"{METHOD.get(r['method'], r['method'])} R{r['round']}",
                    num=dict(model_line=sline(fair_line(p_pick)), open=sline(r["open_line"]),
                             placeable=sline(r["placeable_line"]), close=sline(r["clean_close"]),
                             clv=f"{r['clv_pts']:+.1f}", rule=r["rule"],
                             stake_u=f"{r['stake_u']:.2f}",
                             rd=f"{r['rd_self']:.0f} / {r['rd_opp']:.0f}")))
        w = sum(1 for x in bets if x["won"])
        net = sum(r["pnl_placed"] for r in by_fid_rows(by_fid, bouts) if r["placed"] == 1)
        live = is_live(ev_date)
        posted = data.capture_events.get(ev_date) or next(
            (e for cd, e in data.capture_events.items()
             if abs((date.fromisoformat(cd) - d).days) <= 1), None)
        if posted:
            posted_text = "Picks posted " + short_day(
                datetime.fromisoformat(posted["posted"]).date())
        elif live:
            posted_text = "Picks on record before the event"
        else:
            posted_text = "Backtest · settled at opening odds, no money down"
        cards.append(dict(key=f"{year}-{ev_date}", name=ev_name, date=ev_date,
                          date_text=day_text(d), posted_text=posted_text, live=live,
                          bets=bets, w=w, l=len(bets) - w, net=money(net, signed=True),
                          net_u=net, fights=fights, hits=hits, total=total))
    return cards


def by_fid_rows(by_fid, bouts):
    for f in bouts:
        yield from by_fid.get(f["fight_id"], [])


def tiles(rows: list[dict]) -> dict:
    placed = [r for r in rows if r["placed"] == 1]
    w = sum(1 for r in placed if r["pnl_placed"] > 0)
    net = sum(r["pnl_placed"] for r in placed)
    return dict(bets=len(placed), w=w, l=len(placed) - w, net=money(net, signed=True),
                pos=net >= 0)


def build_record(data: Data, model: Model, out: Path):
    live26 = [r for r in data.ledger if r["live"] == 1 and r["event_date"] >= "2026"]
    bt25 = [r for r in data.ledger_bt if r["event_date"][:4] == "2025"]
    cards26 = build_cards(data, model, "2026", data.preds, live26, live_only=True)
    cards25 = build_cards(data, model, "2025", data.preds_bt, bt25, live_only=False)
    tabs = [dict(key="2026", label="2026", badge="LIVE", tip="Picks on record before each event; money down",
                 tiles=tiles(live26), cards=cards26),
            dict(key="2025", label="2025", badge="BACKTEST",
                 tip="Settled at opening odds, no money down", tiles=tiles(bt25), cards=cards25),
            dict(key="all", label="All", badge=None, tip=None, tiles=tiles(live26 + bt25),
                 cards=cards26 + cards25)]
    # public ledger copy
    cols = [c for c in (data.ledger or data.ledger_bt)[0].keys() if c not in LEDGER_PUBLIC_DROP]
    (out / "record").mkdir(parents=True, exist_ok=True)
    with open(out / "record" / "ledger.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(live26 + bt25, key=lambda r: r["event_date"]):
            w.writerow({k: ("" if r[k] is None else r[k]) for k in cols})
    return dict(tabs=tabs, cards={c["key"]: c for c in cards26 + cards25},
                n_ledger=len(live26) + len(bt25))


# ------------------------------------------------------------ how it works --
def build_how(data: Data) -> dict:
    D = pd.read_csv(DATA / "clv_eval_full.csv")
    D = D[D.year >= 2018]
    q = lambda p: np.clip(p, 1e-9, 1 - 1e-9)
    ll = lambda p: float(-np.mean(D.y_a * np.log(q(p)) + (1 - D.y_a) * np.log(1 - q(p))))
    slope = float(np.polyfit(D.dis, D.mv, 1)[0])
    ratings_freeze = max(r["last_fight"] for r in read_csv(DATA / "ratings.csv"))
    return dict(n=len(D), years=f"{int(D.year.min())}–{int(D.year.max())}",
                clv=D.clv.mean() * 100, move10=slope * 10,
                ll_model=ll(D.p_model_a), ll_open=ll(D.qa_o), ll_close=ll(D.qa_c),
                ratings_freeze=ratings_freeze, rule_freeze=RULE_FREEZE,
                policy_freeze=POLICY_FREEZE, flip=FLIP_CONF, gap=GAP_PTS,
                skip_line=SKIP_LINE_MAX, unknown_rd=UNKNOWN_RD, ramp=(RAMP_LO, RAMP_HI))


# -------------------------------------------------------------------- main --
def lint(out: Path):
    bad = []
    for p in out.rglob("*"):
        if p.is_file():
            t = p.read_text(encoding="utf-8", errors="ignore")
            for s in LINT:
                if s in t:
                    bad.append(f"{p.relative_to(out)}: {s!r}")
    if bad:
        raise SystemExit("anonymity lint FAILED:\n  " + "\n  ".join(bad))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", default=date.today().isoformat(),
                    help="build date (eligibility window, next event); default today")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    a = ap.parse_args()
    as_of = date.fromisoformat(a.as_of)
    out = Path(a.out)

    data = Data(as_of)
    model = Model()
    env = Environment(loader=FileSystemLoader(SITE / "templates"), autoescape=True,
                      trim_blocks=True, lstrip_blocks=True)
    base = dict(brand=BRAND, as_of=day_text(as_of), data_through=day_text(
        date.fromisoformat(data.data_through)))

    print("rankings ...")
    rankings = build_rankings(data)
    print("this week ...")
    week = build_this_week(data, model)
    print("record ...")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    record = build_record(data, model, out)
    how = build_how(data)

    pages = {"rankings/index.html": ("rankings.html", dict(rankings, nav="rankings")),
             "this-week/index.html": ("this_week.html", dict(week, nav="this-week")),
             "index.html": ("this_week.html", dict(week, nav="this-week")),
             "record/index.html": ("record.html", dict(record, nav="record")),
             "how-it-works/index.html": ("how_it_works.html", dict(how, nav="how"))}
    for rel, (tpl, ctx) in pages.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(env.get_template(tpl).render(**base, **ctx), encoding="utf-8")
    shutil.copy(SITE / "static" / "style.css", out / "style.css")
    lint(out)
    p4p = ", ".join(r["name"] for r in rankings["lists"][0]["rows"][:8])
    print(f"wrote {out}: {len(pages)} pages, {record['n_ledger']} ledger rows")
    print(f"P4P top 8: {p4p}")
    ev = week["event"]
    print("this week: " + (f"{ev['name']} {ev['date_text']}: {ev['n_bets']} bets"
                           if ev else f"no capture yet (next: {week['next_name']})"))
    print("anonymity lint: PASS")


if __name__ == "__main__":
    main()
