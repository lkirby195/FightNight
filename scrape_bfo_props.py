"""BestFightOdds prop lines: rounds totals, goes the distance, method.

Probe findings (2026-11-05, noche-ufc-4318 and ufc-182-jones-vs-cormier-864):
- Event pages carry props in the odds table as <tr class="pr"> rows. Each has a
  chart button td.but-sip whose data-li is [side, mu, pt, tn]: side 1/2
  (Over/Under, Yes/No, "X wins by ..."/"Not X ..."), mu = the matchup id (same
  ids as data/bfo_lines.csv), pt = prop type, tn = 0 for fight-level props and
  1/2 for fighter props in BFO page order (= bfo_lines fighter1/fighter2).
- /api/ggd?m={mu}&p={side}&pt={pt}&tn={tn} serves that side's cross-book mean
  line history, same base64+ROT47 encoding as the moneyline series. Archived
  events show no book prices in the table, but the history is there: UFC 182
  (2015-01-03) has 22 ticks per side on O/U 4.5, July 2014 to fight night.
- Type codes seen: 1 goes to decision; 2-5 fight starts round N; 6 draw;
  8 by TKO/KO; 9 by submission; 10 inside distance; 11 by decision; 13+ wins
  in round N; 32 O/U 1.5; 33 O/U 2.5; 35 O/U 4.5. Rows are classified by their
  label, never by code, so an unseen code is skipped rather than mislabelled.

Open/close use scrape_bfo.bout_summary (validated paired open, last-tick close),
so props and moneylines share one open rule. Side 1 is always Over / Yes /
"fighter wins by X"; side 2 is Under / No / "not that".

In-play: BFO prunes in-play ticks from older events. For events under ~4 months
old the last moneyline tick can be in-play (see card_settle). Rows carry
recent=1 when the event was within 120 days of the scrape. Checked on the
2026-11-05 scrape (688 recent rows, 220 bouts, 97 of them with moneyline
in-play ticks): no prop tick lands at or after the moneyline's in-play onset,
and no prop's last tick is more than 10 min past the last clean pre-fight
moneyline tick. Books pull props at the bell. --join still writes the
card_settle clean close (p1/p2_close_clean) for every row; it differs from the
raw last tick on 9 older rows (out-of-band closes) and on no recent row.

BFO /api/ggd returns HTTP 500 every time for some goes-the-distance histories
(of bouts with a total: under 2% a year in 2015-21, 4-5% in 2022-23
and 2025-26, 13% in 2024). Those props are tried once
(no backoff), skipped, and retried on the next run.

Polite: scrape_bfo.fetch (0.45 s spacing, disk cache cache_bfo/, retries;
no retry on 500 for /api/ggd).
Resumable: a (mu, pt, tn) already in the output is never fetched or written
twice. Event list = the UFC slugs already in data/bfo_lines.csv.

Usage:
  python scrape_bfo_props.py START_YEAR END_YEAR [--types totals,gtd,method]
      scrape events with START_YEAR <= year < END_YEAR -> data/bfo_props.csv
      (default types: totals,gtd)
  python scrape_bfo_props.py --join
      data/bfo_props.csv -> data/bfo_props_joined.csv (fight_id; fighter props
      oriented to fights_v2 a/b) + a coverage report
"""
from __future__ import annotations

import base64
import csv
import hashlib
import html as H
import json
import os
import re
import sys
from datetime import date, datetime

import pandas as pd
from bs4 import BeautifulSoup

import scrape_bfo as sb

OUT = "data/bfo_props.csv"
JOINED = "data/bfo_props_joined.csv"
COLS = ["slug", "event_date", "mu", "fighter1", "fighter2", "prop", "line", "tn", "pt",
        "label1", "label2", "p1_open", "p1_close", "p2_open", "p2_close",
        "p1_ticks", "p2_ticks", "t_open", "t_close", "open_suspect",
        "p1_open_raw", "p2_open_raw", "recent"]

RE_OVER = re.compile(r"^Over (\d+)½ rounds?$")
RE_UNDER = re.compile(r"^Under (\d+)½ rounds?$")
RE_GTD_Y = re.compile(r"^Fight goes to (?:the )?decision$")
RE_GTD_N = re.compile(r"^Fight (?:doesn't|does not) go to (?:the )?decision$")
RE_METHOD = re.compile(r"^(?P<who>.+) wins (?:by (?P<m>TKO/KO|submission|decision)|"
                       r"(?P<itd>inside (?:the )?distance))$")
METHOD = {"TKO/KO": "ko", "submission": "sub", "decision": "dec"}


def gpd(mu: int, side: int, pt: int, tn: int):
    """One side's prop line history (decoded ggd JSON), or None when empty."""
    # BFO answers 500 every time for some prop histories (~2-15% of GTD props);
    # one attempt, no backoff, so we don't keep hitting a failing endpoint.
    raw = sb.fetch(f"{sb.BASE}/api/ggd?m={mu}&p={side}&pt={pt}&tn={tn}", no_retry=(500,))
    return _decode(raw)


def cached_gpd(mu: int, side: int, pt: int, tn: int):
    """gpd from the disk cache only; None when the series was never cached.
    Never touches the network (join() calls this for every row)."""
    url = f"{sb.BASE}/api/ggd?m={mu}&p={side}&pt={pt}&tn={tn}"
    path = os.path.join(sb.CACHE, hashlib.sha1(url.encode()).hexdigest())
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return _decode(fh.read().decode("utf-8", "replace"))


def _decode(raw: str):
    raw = re.sub(r"[^A-Za-z0-9+/=]", "", raw)
    if not raw:
        return None
    try:
        return json.loads(sb.rot47(base64.b64decode(raw + "=" * (-len(raw) % 4))
                                   .decode("latin-1")))
    except (ValueError, UnicodeDecodeError):
        return None


def label(th) -> str:
    # BFO double-escapes the ½ entity in prop labels ("1&amp;#189;")
    return H.unescape(H.unescape(th.get_text(" ", strip=True))) if th else ""


def prop_rows(slug: str):
    """-> {(mu, pt, tn): {1: label, 2: label}} for every prop row on the page."""
    soup = BeautifulSoup(sb.fetch(f"{sb.BASE}/events/{slug}"), "lxml")
    out: dict = {}
    for tr in soup.select("tr.pr"):
        btn = tr.select_one("td.but-sip")
        if btn is None:
            continue
        try:
            side, mu, pt, tn = json.loads(btn["data-li"])
        except (ValueError, KeyError, TypeError):
            continue
        out.setdefault((int(mu), int(pt), int(tn)), {})[int(side)] = label(tr.find("th"))
    return out


def classify(l1: str, l2: str, tn: int, types: set):
    """-> (prop, line) or None. Side 1 must be the positive side."""
    m1, m2 = RE_OVER.match(l1), RE_UNDER.match(l2)
    if "totals" in types and m1 and m2 and m1.group(1) == m2.group(1):
        return "total", int(m1.group(1)) + 0.5
    if "gtd" in types and RE_GTD_Y.match(l1) and RE_GTD_N.match(l2):
        return "gtd", None
    if "method" in types and tn in (1, 2):
        m = RE_METHOD.match(l1)
        if m:
            return ("itd" if m.group("itd") else METHOD[m.group("m")]), None
    return None


def scrape(y0: int, y1: int, types: set):
    b = pd.read_csv(sb.OUT)
    b["year"] = b.event_date.str[:4].astype(int)
    b = b[(b.year >= y0) & (b.year < y1)].drop_duplicates("mu")
    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT, usecols=["mu", "pt", "tn"])
        done = set(zip(d.mu, d.pt, d.tn))
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
    if new:
        w.writeheader()
    today = date.today()
    events = b.groupby("slug", sort=False)
    order = b.groupby("slug").event_date.min().sort_values().index
    n_ev = len(order)
    written = skipped = empty = 0
    for i, slug in enumerate(order, 1):
        ev = events.get_group(slug)
        bouts = ev.set_index("mu")
        try:
            rows = prop_rows(slug)
        except Exception as e:  # noqa: BLE001
            print(f"  PAGE ERR {slug}: {e}", flush=True)
            continue
        ev_date = ev.event_date.iloc[0]
        recent = int((today - datetime.strptime(ev_date, "%Y-%m-%d").date()).days < 120)
        n_ev_rows = 0
        for (mu, pt, tn), labs in rows.items():
            if mu not in bouts.index or 1 not in labs or 2 not in labs:
                continue
            c = classify(labs[1], labs[2], tn, types)
            if c is None:
                continue
            if (mu, pt, tn) in done:
                skipped += 1
                continue
            try:
                s = sb.bout_summary(gpd(mu, 1, pt, tn), gpd(mu, 2, pt, tn))
            except Exception as e:  # noqa: BLE001
                print(f"  GGD ERR mu={mu} pt={pt} tn={tn}: {e}", flush=True)
                continue
            if s is None:
                empty += 1
                continue
            r = bouts.loc[mu]
            w.writerow({"slug": slug, "event_date": ev_date, "mu": mu,
                        "fighter1": r.fighter1, "fighter2": r.fighter2,
                        "prop": c[0], "line": c[1], "tn": tn, "pt": pt,
                        "label1": labs[1], "label2": labs[2],
                        "p1_open": s["f1_open"], "p1_close": s["f1_close"],
                        "p2_open": s["f2_open"], "p2_close": s["f2_close"],
                        "p1_ticks": s["f1_ticks"], "p2_ticks": s["f2_ticks"],
                        "t_open": s["t_open"], "t_close": s["t_close"],
                        "open_suspect": s["open_suspect"],
                        "p1_open_raw": s["f1_open_raw"], "p2_open_raw": s["f2_open_raw"],
                        "recent": recent})
            done.add((mu, pt, tn))
            written += 1
            n_ev_rows += 1
        fh.flush()
        print(f"[{i}/{n_ev}] {ev_date} {slug}: +{n_ev_rows} rows "
              f"(total new {written}, skipped {skipped}, empty {empty})", flush=True)
    fh.close()
    print(f"done {y0}-{y1 - 1}: {written} new rows, {skipped} already on file, "
          f"{empty} props with no two-sided history", flush=True)


def mu_map():
    """mu -> (fight_id, f1_is_a), by the same matching as clv_eval.join_bfo
    (KEYS cascade within +/-3 days). Read-only: the moneyline join already
    enforces the fail-loud name rule over the same bfo_lines bouts."""
    from clv_eval import KEYS
    b = pd.read_csv(sb.OUT, parse_dates=["event_date"]).drop_duplicates("mu")
    f = pd.read_csv("data/fights_v2.csv", parse_dates=["event_date"])
    f = f[f.event_date >= "2010-01-01"].copy()
    for k, fn, bfn, h in KEYS:
        b[k] = b.apply(lambda r: tuple(sorted([bfn(r.fighter1, h), bfn(r.fighter2, h)])), axis=1)
        f[k] = f.apply(lambda r: tuple(sorted([fn(r.fighter_a, h), fn(r.fighter_b, h)])), axis=1)
    out = {}
    for _, r in b.iterrows():
        near = f[abs((f.event_date - r.event_date).dt.days) <= 3]
        c = near.iloc[0:0]
        for k, _, _, _ in KEYS:
            c = near[near[k] == r[k]]
            if len(c):
                break
        if len(c) == 1:
            c = c.iloc[0]
            f1a = any(bfn(r.fighter1, h) == fn(c.fighter_a, h) for _, fn, bfn, h in KEYS)
            out[int(r.mu)] = (c.fight_id, f1a)
    return out


def join():
    P = pd.read_csv(OUT)
    M = mu_map()
    P["fight_id"] = P.mu.map(lambda m: M.get(int(m), (None, None))[0])
    f1a = P.mu.map(lambda m: M.get(int(m), (None, None))[1])
    # fighter props: which fights_v2 side is the named fighter
    P["side"] = None
    fp = P.tn.isin([1, 2]) & f1a.notna()
    P.loc[fp, "side"] = [("a" if (tn == 1) == bool(x) else "b")
                         for tn, x in zip(P.loc[fp, "tn"], f1a[fp])]
    # In-play-safe close, same rule as card_settle.paired_summary: the last
    # paired tick whose overround is in [1.00, 1.15]. BFO keeps in-play ticks
    # for ~4 months after an event, so p1_close/p2_close (raw last tick) can be
    # mid-fight prices on recent cards. Computed from the disk cache only.
    from card_settle import paired_summary
    clean = []
    for r in P.itertuples():
        s1, s2 = (cached_gpd(int(r.mu), sd, int(r.pt), int(r.tn)) for sd in (1, 2))
        s = paired_summary(s1, s2) if s1 and s2 else None
        clean.append((s[1], s[3], s[5], s[6]) if s else (None, None, None, None))
    P[["p1_close_clean", "p2_close_clean", "t_close_clean", "inplay_ticks_dropped"]] = clean
    J = P[P.fight_id.notna()].drop(columns=["slug"])
    J.to_csv(JOINED, index=False, lineterminator="\n")
    # Coverage against our fight records (not BFO's bout list, which includes
    # cancelled and re-booked matchups). ml = fight has a joined moneyline.
    # Counts only; no outcomes are read here.
    F = pd.read_csv("data/fights_v2.csv", usecols=["fight_id", "event_date"])
    F = F[F.event_date >= str(P.event_date.min())].copy()
    F["year"] = F.event_date.str[:4]
    F["ml"] = F.fight_id.isin(pd.read_csv("data/bfo_joined.csv", usecols=["fight_id"]).fight_id)
    for p in ("total", "gtd"):
        F[p] = F.fight_id.isin(J.loc[J.prop == p, "fight_id"])
    cov = F.groupby("year").agg(fights=("fight_id", "size"), ml=("ml", "mean"),
                                total=("total", "mean"), gtd=("gtd", "mean"))
    cov["total|ml"] = F[F.ml].groupby("year").total.mean()
    cov["gtd|ml"] = F[F.ml].groupby("year").gtd.mean()
    pd.set_option("display.width", 200)
    print("share of fights_v2 fights with each price:")
    print(cov.round(3).to_string())
    tot = P[P.prop == "total"]
    print("\ntotals lines:", tot.line.value_counts().sort_index().to_dict())
    print(f"joined rows: {len(J)} of {len(P)} ({len(J) / len(P):.1%})")
    moved = (P.p1_close_clean != P.p1_close) | (P.p2_close_clean != P.p2_close)
    print("rows whose clean close differs from the raw last tick, by recent flag:",
          P.assign(moved=moved).groupby("recent").moved.agg(["sum", "size"]).to_dict("index"))
    print("rows with no cached series (clean close missing):", int(P.p1_close_clean.isna().sum()))
    return cov


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--join" in sys.argv:
        join()
    else:
        types = {"totals", "gtd"}
        for a in sys.argv:
            if a.startswith("--types="):
                types = set(a.split("=", 1)[1].split(","))
        if "--types" in sys.argv:
            types = set(sys.argv[sys.argv.index("--types") + 1].split(","))
            args = [a for a in args if a != sys.argv[sys.argv.index("--types") + 1]]
        y0, y1 = (int(args[0]), int(args[1])) if len(args) >= 2 else (2015, 2027)
        scrape(y0, y1, types)
