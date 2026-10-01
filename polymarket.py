"""Polymarket: a second price source for UFC bouts. Never a signal source.

The signal rules and the placement policy read BestFightOdds only; this module
records what the Polymarket moneyline said alongside, so the two prices can be
compared after the fact (data/pm_clv_2026.csv) and a bout Polymarket lists
before BestFightOdds does can still be projected (card_report "no BFO line").

Endpoints (all public, no key):
  GET gamma-api.polymarket.com/public-search?q=<surname>&limit_per_type=30
      events (each with its markets) matching the text; resolved ones included.
  GET gamma-api.polymarket.com/events?tag_slug=ufc&closed=false&limit=100
      every open UFC event: the full-card listing for an upcoming date.
  GET clob.polymarket.com/prices-history?market=<tokenId>&interval=max&fidelity=60
      hourly price history of one outcome token, through resolution.

A bout is one event whose slug ends in -YYYY-MM-DD (the fight date) holding
one "moneyline" market: two outcomes that are fighter names (the other
markets on the event are props: Yes/No, Over/Under). Names are matched with
names.norm (suffixes dropped, ASCII-folded); an outcome given as a bare
surname ("Cejudo") matches on the surname alone.

Disk-cached in cache_pm/ (sha1 of the URL, like the other scrapers), 0.45 s
spacing. A future card's prices move, so card_report refreshes them; resolved
markets are stable and come from the cache.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests

import names

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CACHE = "cache_pm"
HISTORY_DIR = os.path.join("data", "pm_history")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
SEARCH_LIMIT = 30          # public-search pages 5 events by default
PROPS = {"yes", "no", "over", "under"}

S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept": "application/json"})
_last = [0.0]


def fetch_json(url: str, refresh: bool = False):
    """Cached GET -> parsed JSON. refresh=True re-downloads and overwrites the
    cached copy (live markets, whose prices move)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, hashlib.sha1(url.encode()).hexdigest())
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    gap = time.time() - _last[0]
    if gap < 0.45:
        time.sleep(0.45 - gap)
    _last[0] = time.time()
    for attempt in range(5):
        try:
            r = S.get(url, timeout=30)
            r.raise_for_status()
            js = r.json()
            with open(path, "wb") as fh:
                fh.write(json.dumps(js).encode("utf-8"))
            return js
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def _jl(x):
    """gamma serialises list fields (outcomes, outcomePrices, clobTokenIds) as
    JSON strings; accept either form."""
    if x is None:
        return []
    return json.loads(x) if isinstance(x, str) else list(x)


def _num(x):
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def slug_date(slug: str) -> str | None:
    """'ufc-rao-rau2-2026-09-26' -> '2026-09-26'; None when the slug carries no date."""
    m = re.search(r"-(\d{4}-\d{2}-\d{2})$", slug or "")
    return m.group(1) if m else None


def moneyline(event: dict) -> dict | None:
    """The event's fighter-vs-fighter market: exactly two outcomes, neither a
    prop label. None for a props-only or "who fights next" event."""
    for m in event.get("markets", []) or []:
        o = _jl(m.get("outcomes"))
        if len(o) == 2 and not {str(x).strip().lower() for x in o} & PROPS:
            return m
    return None


def same_fighter(outcome: str, name: str) -> bool:
    """An outcome label names the fighter: full normalised match, else the same
    surname with the same first three letters of the first name, or a bare
    surname outcome ("Cejudo")."""
    o, n = names.norm(outcome), names.norm(name)
    if o == n or (o and o.replace(" ", "") == n.replace(" ", "")):
        return True
    ot, nt = o.split(), n.split()
    if not ot or not nt or ot[-1] != nt[-1]:
        return False
    return len(ot) == 1 or ot[0][:3] == nt[0][:3]


def match_outcomes(outcomes: list[str], a: str, b: str):
    """-> (index of a, index of b) in outcomes, or None."""
    if len(outcomes) != 2:
        return None
    for ia, ib in ((0, 1), (1, 0)):
        if same_fighter(outcomes[ia], a) and same_fighter(outcomes[ib], b):
            return ia, ib
    return None


def bout_record(event: dict, market: dict, ia: int = 0, ib: int = 1) -> dict:
    """One flat record for a (event, moneyline) pair, oriented so that `a` is
    outcome ia and `b` outcome ib."""
    outcomes = [str(x) for x in _jl(market.get("outcomes"))]
    prices = [_num(x) for x in _jl(market.get("outcomePrices"))]
    tokens = [str(x) for x in _jl(market.get("clobTokenIds"))]
    pad = lambda xs: (xs + [None, None])[:2]
    prices, tokens = pad(prices), pad(tokens)
    return dict(slug=event.get("slug", ""), title=event.get("title", ""),
                market_id=str(market.get("id", "")),
                outcomes=outcomes, outcome_prices=prices, clob_token_ids=tokens,
                a_idx=ia, b_idx=ib, p_a=prices[ia], p_b=prices[ib],
                token_a=tokens[ia], token_b=tokens[ib],
                liquidity=_num(market.get("liquidity")) if _num(market.get("liquidity")) is not None
                else _num(event.get("liquidity")),
                volume=_num(market.get("volume")) if _num(market.get("volume")) is not None
                else _num(event.get("volume")),
                closed=bool(market.get("closed")), end_date=market.get("endDate"))


def search(q: str, refresh: bool = False) -> list[dict]:
    js = fetch_json(f"{GAMMA}/public-search?q={quote(q)}&limit_per_type={SEARCH_LIMIT}",
                    refresh=refresh)
    return js.get("events", []) if isinstance(js, dict) else []


def find_bout(fighter_a: str, fighter_b: str, event_date: str, refresh: bool = False):
    """The Polymarket moneyline for fighter_a vs fighter_b on event_date, or
    None. Searches a's surname, then b's, then the full names; an event
    qualifies when its slug ends in -<event_date> and its moneyline names both
    fighters. The record is oriented a-first (p_a, token_a)."""
    seen = set()
    for q in (names.lastn(fighter_a), names.lastn(fighter_b), fighter_a, fighter_b):
        q = (q or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        for ev in search(q, refresh=refresh):
            if slug_date(ev.get("slug", "")) != event_date:
                continue
            m = moneyline(ev)
            if not m:
                continue
            mo = match_outcomes([str(x) for x in _jl(m.get("outcomes"))], fighter_a, fighter_b)
            if mo:
                return bout_record(ev, m, *mo)
    return None


def card_bouts(event_date: str, refresh: bool = True) -> list[dict]:
    """Every open UFC event on Polymarket dated event_date (slug suffix) that
    carries a moneyline, in Polymarket's outcome order (a = outcome 0)."""
    js = fetch_json(f"{GAMMA}/events?tag_slug=ufc&closed=false&limit=100", refresh=refresh)
    out = []
    for ev in js if isinstance(js, list) else []:
        if slug_date(ev.get("slug", "")) != event_date:
            continue
        m = moneyline(ev)
        if m:
            out.append(bout_record(ev, m, 0, 1))
    return out


def prices_history(token_id: str, refresh: bool = False) -> list[tuple[int, float]]:
    """Hourly (t_seconds, price) points for one outcome token, oldest first."""
    js = fetch_json(f"{CLOB}/prices-history?market={token_id}&interval=max&fidelity=60",
                    refresh=refresh)
    pts = [(int(p["t"]), float(p["p"])) for p in (js.get("history", []) if isinstance(js, dict) else [])]
    pts.sort()
    return pts


def price_at(hist: list[tuple[int, float]], t: float):
    """Price of the last point at or before t (seconds); None before the first point."""
    p = None
    for tt, pp in hist:
        if tt > t:
            break
        p = pp
    return p


# ------------------------------------------------------------- backfill --
def _ts(iso: str) -> float:
    d = datetime.fromisoformat(iso)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def earliest_capture_time(event_date: str, a: str, b: str, path="data/placeable_lines.csv"):
    """captured_at (ISO) of the earliest placeable_lines.csv row for the bout
    (same surnames, event within 3 days), or None."""
    if not os.path.exists(path):
        return None
    want = {names.lastn(a), names.lastn(b)}
    d0 = datetime.fromisoformat(event_date)
    best = None
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if abs((datetime.fromisoformat(r["event_date"]) - d0).days) > 3:
                continue
            if {names.bfo_lastn(r["fighter1"]), names.bfo_lastn(r["fighter2"])} != want:
                continue
            if best is None or r["captured_at"] < best:
                best = r["captured_at"]
    return best


def bfo_close_time(bl, row) -> float | None:
    """Seconds of BFO's last pre-fight paired tick for a ledger bout (the
    card_settle overround rule, the instant the ledger's clean_close is read
    at), from the cached tick series; None when the bout cannot be matched."""
    from betting_system import _bfo_mu
    from card_settle import paired_summary
    from scrape_bfo import ggd
    try:
        mu, _ = _bfo_mu(bl, row)
    except SystemExit:
        return None
    s = paired_summary(ggd(mu, 1), ggd(mu, 2))
    return None if s is None else s[5] / 1000.0


def backfill(year: int, ledger="data/system_ledger.csv", out="data/pm_clv_%d.csv"):
    """For every ledger bout of `year` with a resolved Polymarket market: save
    both outcomes' hourly price histories under data/pm_history/<slug>.json
    and write data/pm_clv_<year>.csv with the pick's price at our capture time
    (the earliest capture; else the event date 00:00 UTC minus 48 h) and its
    last pre-fight price (the Polymarket price at BFO's last clean pre-fight
    tick, the instant the ledger's clean_close is read at). Read-only on every
    other file."""
    import pandas as pd
    os.makedirs(HISTORY_DIR, exist_ok=True)
    with open("data/fights_v2.csv", newline="", encoding="utf-8") as fh:
        fv = {r["fight_id"]: r for r in csv.DictReader(fh)}
    with open(ledger, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["event_date"][:4] == str(year)]
    J = pd.read_csv("data/bfo_joined.csv").set_index("fight_id")
    bl = pd.read_csv("data/bfo_lines.csv")
    bl["d"] = pd.to_datetime(bl.event_date)
    out_rows, n_market, n_hist = [], 0, 0
    print(f"{'date':11s}{'pick':24s}{'rule':5s}{'PM slug':30s}{'cap':>7}{'pre':>7}"
          f"{'PM CLV':>8}{'BFO CLV':>9}  result")
    for r in rows:
        f = fv[r["fight_id"]]
        a, b = f["fighter_a"], f["fighter_b"]
        rec = find_bout(a, b, r["event_date"])
        tag = ""
        if rec is None:
            tag = "no market"
        elif not rec["closed"]:
            tag = "market open"
        if tag:
            print(f"{r['event_date']:11s}{r['fighter'][:23]:24s}{r['rule']:5s}{tag}")
            continue
        n_market += 1
        hist = {}
        for o, tok in zip(rec["outcomes"], rec["clob_token_ids"]):
            hist[o] = prices_history(tok) if tok else []
        with open(os.path.join(HISTORY_DIR, rec["slug"] + ".json"), "w", encoding="utf-8") as fh:
            json.dump(dict(slug=rec["slug"], market_id=rec["market_id"],
                           event_date=r["event_date"], fighter_a=a, fighter_b=b,
                           outcomes=rec["outcomes"], clob_token_ids=rec["clob_token_ids"],
                           history={o: [[t, p] for t, p in h] for o, h in hist.items()}),
                      fh, indent=0)
        pick_is_a = r["fighter"] == a
        h = hist[rec["outcomes"][rec["a_idx"] if pick_is_a else rec["b_idx"]]]
        cap = earliest_capture_time(r["event_date"], a, b)
        if cap:
            t_cap, cap_src = _ts(cap), "capture"
        else:
            t_cap = (datetime.fromisoformat(r["event_date"]).replace(tzinfo=timezone.utc)
                     - timedelta(hours=48)).timestamp()
            cap_src = "event-48h"
        jrow = J.loc[r["fight_id"]] if r["fight_id"] in J.index else None
        t_pre = None
        if jrow is not None:
            row = pd.Series(dict(event_date=r["event_date"], fight_id=r["fight_id"],
                                 fighter_a=a, fighter_b=b,
                                 a_open=jrow.a_open, b_open=jrow.b_open))
            t_pre = bfo_close_time(bl, row)
        p_cap = price_at(h, t_cap) if h else None
        p_pre = price_at(h, t_pre) if (h and t_pre is not None) else None
        n_hist += bool(h)
        clv = (p_pre - p_cap) * 100 if (p_pre is not None and p_cap is not None) else None
        fmt = lambda p: f"{p*100:6.1f}%" if p is not None else "     -"
        note = "" if h else "  (no history served)"
        print(f"{r['event_date']:11s}{r['fighter'][:23]:24s}{r['rule']:5s}{rec['slug'][:29]:30s}"
              f"{fmt(p_cap)}{fmt(p_pre)}{(f'{clv:+7.1f}p' if clv is not None else '       -'):>8}"
              f"{float(r['clv_pts']):+8.1f}p  {r['result']}{note}")
        iso = lambda t: ("" if t is None else
                         datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds"))
        out_rows.append(dict(
            event_date=r["event_date"], event=r["event"], fight_id=r["fight_id"],
            fighter=r["fighter"], rule=r["rule"], live=r["live"], placed=r["placed"],
            pm_slug=rec["slug"], pm_market_id=rec["market_id"], pm_history_points=len(h),
            t_capture=iso(t_cap), capture_source=cap_src,
            pm_p_at_capture="" if p_cap is None else round(p_cap, 4),
            t_prefight=iso(t_pre),
            pm_p_prefight="" if p_pre is None else round(p_pre, 4),
            pm_clv_pts="" if clv is None else round(clv, 2),
            bfo_clv_pts=r["clv_pts"], result=r["result"]))
    path = out % year
    cols = ["event_date", "event", "fight_id", "fighter", "rule", "live", "placed",
            "pm_slug", "pm_market_id", "pm_history_points", "t_capture", "capture_source",
            "pm_p_at_capture", "t_prefight", "pm_p_prefight", "pm_clv_pts", "bfo_clv_pts",
            "result"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    n = [x for x in out_rows if x["pm_clv_pts"] != ""]
    print(f"\n{n_market} of {len(rows)} ledger bouts have a resolved market, {n_hist} of those "
          f"a served price history, {len(n)} with both prices", end="")
    if n:
        pm = sum(x["pm_clv_pts"] for x in n) / len(n)
        bf = sum(float(x["bfo_clv_pts"]) for x in n) / len(n)
        print(f": mean PM CLV {pm:+.2f} pts vs mean BFO CLV {bf:+.2f} pts on the same picks")
    else:
        print()
    print(f"wrote {path}: {len(out_rows)} rows; histories in {HISTORY_DIR}/")
    return out_rows


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "backfill":
        backfill(int(a[1]))
    elif len(a) == 3:
        rec = find_bout(a[0], a[1], a[2], refresh=True)
        print(json.dumps(rec, indent=1) if rec else "no market")
    else:
        sys.exit("usage: polymarket.py backfill <year> | polymarket.py <fighter_a> <fighter_b> <YYYY-MM-DD>")
