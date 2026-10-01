"""BestFightOdds crawler: opening + closing mean lines per fighter per fight.

Pipeline: sitemap-events.xml -> UFC event pages (matchup ids, fighter names,
event date) -> /api/ggd?m={mu}&p={1,2} -> decode base64+ROT47 -> JSON series
of (timestamp_ms, decimal_odds) for the cross-book mean line.

Open = the validated open (validated_open): the bout's first paired tick
(both sides' price at the first instant both have one), UNLESS the next
paired tick lands within OPEN_WINDOW_MS of it and, on either side, crosses
even money or moves more than OPEN_JUMP implied-probability points -- a
data-entry opener (sides swapped, wrong price) corrected within the hour.
Then that corrected pair is the open and the bout carries open_suspect=1;
the raw first ticks are kept (f1_open_raw, f2_open_raw). The check is paired
because BFO moves both sides at the same instants and a bettor sees a book,
not a side: Perez / Dumont 2026-09-26 opened -200/+169 and was -106/-110 two
minutes later, which only the Dumont side "crosses". Close = last tick. One
function (bout_summary) serves scrape, card_report and card_settle, so every
consumer sees one open.

The event date is the schema.org SportsEvent "startDate" in the event page's
JSON-LD, never the sitemap <lastmod>: BFO serves one event under several
alias slugs whose lastmods differ (UFC 331 landed on three dates that way),
and lastmod runs a day late on many cards. A page with no parseable date
raises ValueError(slug) and the event is skipped, loudly.

Polite: 0.45s spacing, disk cache (cache_bfo/), resumable. Incremental by
matchup id: a bout already in data/bfo_lines.csv (or already written this run
under another slug for the same event) is never written twice.

Usage: python scrape_bfo.py [start_year end_year] [--rebuild]
  --rebuild first rewrites every row already on file from its cached series
  (fetching any series the cache lacks), row order and identity columns
  untouched, and lists the bouts flagged open_suspect; the incremental pass
  then runs as usual. Run it once after a change to the open rule.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

BASE = "https://www.bestfightodds.com"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CACHE = "cache_bfo"
os.makedirs(CACHE, exist_ok=True)
os.makedirs("data", exist_ok=True)

S = requests.Session()
S.headers.update({"User-Agent": UA})
_last = [0.0]


def fetch(url: str, binary=False, refresh=False) -> str:
    """Cached GET. refresh=True re-downloads and overwrites the cached copy
    (card_report uses it for future events, whose lines are still moving)."""
    key = hashlib.sha1(url.encode()).hexdigest()
    path = os.path.join(CACHE, key)
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as fh:
            b = fh.read()
        return b if binary else b.decode("utf-8", "replace")
    gap = time.time() - _last[0]
    if gap < 0.45:
        time.sleep(0.45 - gap)
    _last[0] = time.time()
    for attempt in range(5):
        try:
            r = S.get(url, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as fh:
                fh.write(r.content)
            return r.content if binary else r.text
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def rot47(s: str) -> str:
    return "".join(chr(33 + (ord(c) - 33 + 47) % 94) if 33 <= ord(c) <= 126 else c
                   for c in s)


def ggd(mu: int, p: int, refresh=False):
    raw = fetch(f"{BASE}/api/ggd?m={mu}&p={p}", refresh=refresh)
    raw = re.sub(r"[^A-Za-z0-9+/=]", "", raw)
    dec = base64.b64decode(raw + "=" * (-len(raw) % 4)).decode("latin-1")
    return json.loads(rot47(dec))


# UFC MMA cards only: "ufc" must be a hyphen-delimited token of the slug
# (ufc-…, noche-ufc-…). Feeder and grappling brands that share the token are
# excluded (road-to-ufc-…, ufc-bjj-…, ufc-fight-pass-invitational-…).
UFC_SLUG = re.compile(r"^(?!road-to-ufc-)(?!ufc-bjj)(?!ufc-fight-pass-)"
                      r"(?:[^-]+-)*ufc(?:-|$)")


def ufc_event_slugs():
    """-> sorted (slug, lastmod) pairs for every UFC event in the sitemap."""
    xml = fetch(f"{BASE}/sitemap-events.xml")
    ent = re.findall(r"<loc>https://www\.bestfightodds\.com/events/([^<]+)</loc>"
                     r"\s*<lastmod>([^<]+)</lastmod>", xml)
    return sorted({e for e in ent if UFC_SLUG.match(e[0])})


# Event date: the schema.org SportsEvent JSON-LD carried by every event page,
#     "startDate": "2026-09-19",
# The meta description ("... on September 19, 2026") carries it too; the old
# month-name regex missed that form because of the comma, so every date came
# from the sitemap lastmod. Pinned by tests/test_scrape_bfo.py against a saved
# copy of the UFC 331 page.
DATE_RE = re.compile(r'"startDate":\s*"(\d{4}-\d{2}-\d{2})"')


def parse_event(slug: str):
    """-> (title, event_date ISO, bouts) from the BFO event page. Raises
    ValueError(slug) when the page carries no parseable event date."""
    html = fetch(f"{BASE}/events/{slug}")
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    m = DATE_RE.search(html)
    if not m:
        raise ValueError(slug)              # no date on the page: never guess
    try:
        ev_date = datetime.strptime(m.group(1), "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise ValueError(slug) from None
    bouts = []
    for tr in soup.select("tr[id^=mu-]"):
        mid = tr["id"].split("-")[1]
        if not mid.isdigit():
            continue  # event-prop rows use mu-e#### ids
        mu = int(mid)
        a1 = tr.select_one("a[href^='/fighters/']")
        tr2 = tr.find_next_sibling("tr")
        a2 = tr2.select_one("a[href^='/fighters/']") if tr2 else None
        if a1 and a2:
            bouts.append({"mu": mu,
                          "f1": a1.get_text(strip=True),
                          "f2": a2.get_text(strip=True)})
    return title, ev_date, bouts


# Opener validity (2026-09-30). BFO's first tick is occasionally a data-entry
# slip that the book corrects minutes later: Demopoulos / Jauregui 2026-09-26
# opened -850 / +596 and flipped to +596 / -850 eight minutes on. A signal
# evaluated against that first tick is a signal against a price nobody could
# bet, so the open is the corrected tick when the correction is immediate.
OPEN_WINDOW_MS = 60 * 60 * 1000     # the next tick must land within an hour
OPEN_JUMP = 0.40                    # ... crossing even money, or moving > 40 pts


def _pts(js):
    """Sorted (timestamp_ms, decimal) ticks of one side's ggd JSON."""
    out = []
    for srs in js if isinstance(js, list) else [js]:
        for pt in srs.get("data", []):
            t, v = (pt.get("x"), pt.get("y")) if isinstance(pt, dict) else (pt[0], pt[1])
            if v is not None:
                out.append((t, float(v)))
    out.sort()
    return out


def paired_ticks(p1, p2):
    """(t, v1, v2) at every instant either side ticks, each side's latest
    price at or before t, from the first instant both sides have one."""
    ts = sorted({t for t, _ in p1} | {t for t, _ in p2})
    pairs, i, j, v1, v2 = [], 0, 0, None, None
    for t in ts:
        while i < len(p1) and p1[i][0] <= t:
            v1 = p1[i][1]
            i += 1
        while j < len(p2) and p2[j][0] <= t:
            v2 = p2[j][1]
            j += 1
        if v1 is not None and v2 is not None:
            pairs.append((t, v1, v2))
    return pairs


def _corrected(v0, v1):
    """The move v0 -> v1 on one side crosses even money or exceeds OPEN_JUMP."""
    return (v0 >= 2.0) != (v1 >= 2.0) or abs(1.0 / v0 - 1.0 / v1) > OPEN_JUMP


def validated_open(p1, p2):
    """-> (f1_open, f2_open, t_open, suspect, f1_raw, f2_raw) for a bout's two
    sorted tick series. The first paired tick, unless the next paired tick
    lands within OPEN_WINDOW_MS and is a correction (_corrected) on either
    side: then that pair is the open and suspect is True. None when the two
    sides never both have a price."""
    pairs = paired_ticks(p1, p2)
    if not pairs:
        return None
    t0, a0, b0 = pairs[0]
    if len(pairs) >= 2:
        t1, a1, b1 = pairs[1]
        if t1 - t0 <= OPEN_WINDOW_MS and (_corrected(a0, a1) or _corrected(b0, b1)):
            return a1, b1, t1, True, a0, b0
    return a0, b0, t0, False, a0, b0


def series_summary(js):
    """-> (open_dec, close_dec, n_ticks, t_open, t_close) of ONE side's ggd
    JSON, raw: first tick, last tick. None when it has no ticks. The bout's
    validated open is bout_summary / validated_open."""
    pts = _pts(js)
    if not pts:
        return None
    return pts[0][1], pts[-1][1], len(pts), pts[0][0], pts[-1][0]


def bout_summary(js1, js2):
    """-> the bfo_lines row fields for one matchup from both sides' ggd JSON
    (validated opens, last-tick closes, tick counts, timestamps, open_suspect
    and the raw first ticks), or None when either side has no ticks."""
    p1, p2 = _pts(js1), _pts(js2)
    if not p1 or not p2:
        return None
    vo = validated_open(p1, p2)
    if vo is None:
        return None
    o1, o2, t_o, suspect, r1, r2 = vo
    return {"f1_open": o1, "f1_close": p1[-1][1],
            "f2_open": o2, "f2_close": p2[-1][1],
            "f1_ticks": len(p1), "f2_ticks": len(p2),
            "t_open": t_o, "t_close": p1[-1][0],
            "open_suspect": int(suspect),
            "f1_open_raw": r1, "f2_open_raw": r2}


def summarize(mu: int, refresh=False):
    """-> bout_summary of one matchup's two series, or None."""
    return bout_summary(ggd(mu, 1, refresh=refresh), ggd(mu, 2, refresh=refresh))


COLS = ["slug", "event_title", "event_date", "mu", "fighter1", "fighter2",
        "f1_open", "f1_close", "f2_open", "f2_close",
        "f1_ticks", "f2_ticks", "t_open", "t_close",
        "open_suspect", "f1_open_raw", "f2_open_raw"]

OUT = "data/bfo_lines.csv"


def _am(dec: float) -> str:
    return f"+{round((dec - 1) * 100)}" if dec >= 2 else f"{round(-100 / (dec - 1))}"


def rebuild(start_year=2010, end_year=2027, out_path=OUT):
    """Rewrite every row of out_path dated within [start_year, end_year) from
    its series (cache_bfo/, fetched when absent): opens, closes, tick counts,
    timestamps and the open_suspect / raw-open columns. Row order and the
    identity columns (slug, title, date, mu, names) are untouched. Prints
    what changed and every flagged bout."""
    with open(out_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n_open = n_close = n_flag = n_err = 0
    flagged = []
    for i, r in enumerate(rows, 1):
        if not (start_year <= int(r["event_date"][:4]) < end_year):
            for k in COLS:
                r.setdefault(k, "")
            continue
        try:
            s = summarize(int(r["mu"]))
        except Exception as e:  # noqa: BLE001
            print(f"  GGD ERR mu={r['mu']}: {e}", flush=True)
            s = None
        if s is None:                       # series gone from BFO: row kept as it was,
            n_err += 1                      # opens taken as clean
            print(f"  kept as-is (no series) {r['event_date']} {r['fighter1']} / "
                  f"{r['fighter2']} (mu {r['mu']})", flush=True)
            r["open_suspect"] = r.get("open_suspect") or "0"
            r["f1_open_raw"] = r.get("f1_open_raw") or r["f1_open"]
            r["f2_open_raw"] = r.get("f2_open_raw") or r["f2_open"]
            continue
        old_open = (r.get("f1_open"), r.get("f2_open"))
        old_close = (r.get("f1_close"), r.get("f2_close"))
        r.update({k: str(v) for k, v in s.items()})
        n_open += old_open != (r["f1_open"], r["f2_open"])
        n_close += old_close != (r["f1_close"], r["f2_close"])
        if s["open_suspect"]:
            n_flag += 1
            flagged.append(r)
        if i % 500 == 0:
            print(f"  rebuilt {i}/{len(rows)}", flush=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"rebuilt {out_path}: {len(rows)} rows; opens changed {n_open}, closes changed "
          f"{n_close}, open_suspect {n_flag}, series errors {n_err}")
    for r in flagged:
        print(f"  SUSPECT {r['event_date']} {r['fighter1']} / {r['fighter2']} (mu {r['mu']}): "
              f"raw {_am(float(r['f1_open_raw']))}/{_am(float(r['f2_open_raw']))} -> "
              f"open {_am(float(r['f1_open']))}/{_am(float(r['f2_open']))}")
    return flagged


def main(start_year=2022, end_year=2027, out_path=OUT):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            done = {r["mu"] for r in rd}
            if rd.fieldnames != COLS:
                raise SystemExit(f"{out_path} has columns {rd.fieldnames}; expected {COLS}. "
                                 f"Run `scrape_bfo.py {start_year} {end_year} --rebuild` once.")
    slugs = ufc_event_slugs()
    print(f"UFC events in sitemap: {len(slugs)}", flush=True)
    mode = "a" if done else "w"
    n_done = 0
    with open(out_path, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        if not done:
            w.writeheader()
        for si, (slug, lastmod) in enumerate(slugs):
            if not (start_year <= int(lastmod[:4]) < end_year):
                continue
            try:
                title, ev_date, bouts = parse_event(slug)
            except Exception as e:  # noqa: BLE001
                print(f"  EV ERR {slug}: {e!r}", flush=True)
                continue
            for b in bouts:
                if str(b["mu"]) in done:
                    continue
                try:
                    s = summarize(b["mu"])
                except Exception as e:  # noqa: BLE001
                    print(f"  GGD ERR mu={b['mu']}: {e}", flush=True)
                    continue
                if s is None:
                    continue
                w.writerow({"slug": slug, "event_title": title, "event_date": ev_date,
                            "mu": b["mu"], "fighter1": b["f1"], "fighter2": b["f2"], **s})
                if s["open_suspect"]:
                    print(f"  SUSPECT opener {ev_date} {b['f1']} / {b['f2']} (mu {b['mu']}): "
                          f"raw {_am(s['f1_open_raw'])}/{_am(s['f2_open_raw'])} -> "
                          f"open {_am(s['f1_open'])}/{_am(s['f2_open'])}", flush=True)
                done.add(str(b["mu"]))   # one row per matchup even if the
                n_done += 1              # sitemap lists the event twice
                if n_done % 100 == 0:
                    fh.flush()
                    print(f"  {n_done} bouts ({slug})", flush=True)
    print(f"done: {n_done} new bouts -> {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    a = args or ["2022", "2027"]
    if "--rebuild" in flags:
        rebuild(int(a[0]), int(a[1]))
    main(int(a[0]), int(a[1]))
