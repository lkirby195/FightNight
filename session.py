"""BestFightOdds crawler: opening + closing mean lines per fighter per fight.

Pipeline: sitemap-events.xml -> UFC event pages (matchup ids, fighter names,
event date) -> /api/ggd?m={mu}&p={1,2} -> decode base64+ROT47 -> JSON series
of (timestamp_ms, decimal_odds) for the cross-book mean line.

First tick = opening line, last tick = closing line.

Polite: 0.45s spacing, disk cache (cache_bfo/), resumable.
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


def fetch(url: str, binary=False) -> str:
    key = hashlib.sha1(url.encode()).hexdigest()
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
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


def ggd(mu: int, p: int):
    raw = fetch(f"{BASE}/api/ggd?m={mu}&p={p}")
    raw = re.sub(r"[^A-Za-z0-9+/=]", "", raw)
    dec = base64.b64decode(raw + "=" * (-len(raw) % 4)).decode("latin-1")
    return json.loads(rot47(dec))


def ufc_event_slugs():
    xml = fetch(f"{BASE}/sitemap-events.xml")
    ent = re.findall(r"<loc>https://www\.bestfightodds\.com/events/(ufc[^<]*)</loc>"
                     r"\s*<lastmod>([^<]+)</lastmod>", xml)
    return sorted(set(ent))


DATE_RE = re.compile(r"(January|February|March|April|May|June|July|August|"
                     r"September|October|November|December)\s+\d{1,2}\w*\s+\d{4}")


def parse_event(slug: str):
    html = fetch(f"{BASE}/events/{slug}")
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    m = DATE_RE.search(html)
    ev_date = None
    if m:
        txt = re.sub(r"(\d)(st|nd|rd|th)", r"\1", m.group(0))
        try:
            ev_date = datetime.strptime(txt, "%B %d %Y").date().isoformat()
        except ValueError:
            pass
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


def series_summary(js):
    """-> (open_dec, close_dec, n_ticks, t_open, t_close) from ggd JSON."""
    pts = []
    for srs in js if isinstance(js, list) else [js]:
        for pt in srs.get("data", []):
            t, v = (pt.get("x"), pt.get("y")) if isinstance(pt, dict) else (pt[0], pt[1])
            if v is not None:
                pts.append((t, float(v)))
    if not pts:
        return None
    pts.sort()
    return pts[0][1], pts[-1][1], len(pts), pts[0][0], pts[-1][0]


COLS = ["slug", "event_title", "event_date", "mu", "fighter1", "fighter2",
        "f1_open", "f1_close", "f2_open", "f2_close",
        "f1_ticks", "f2_ticks", "t_open", "t_close"]


def main(start_year=2022, end_year=2027):
    out_path = "data/bfo_lines.csv"
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            done = {r["mu"] for r in csv.DictReader(fh)}
    slugs = ufc_event_slugs()
    print(f"UFC events in sitemap: {len(slugs)}", flush=True)
    mode = "a" if done else "w"
    n_done = 0
    with open(out_path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        if not done:
            w.writeheader()
        for si, (slug, lastmod) in enumerate(slugs):
            if not (start_year <= int(lastmod[:4]) < end_year):
                continue
            try:
                title, ev_date, bouts = parse_event(slug)
            except Exception as e:  # noqa: BLE001
                print(f"  EV ERR {slug}: {e}", flush=True)
                continue
            ev_date = ev_date or lastmod[:10]
            for b in bouts:
                if str(b["mu"]) in done:
                    continue
                try:
                    s1 = series_summary(ggd(b["mu"], 1))
                    s2 = series_summary(ggd(b["mu"], 2))
                except Exception as e:  # noqa: BLE001
                    print(f"  GGD ERR mu={b['mu']}: {e}", flush=True)
                    continue
                if not (s1 and s2):
                    continue
                w.writerow({"slug": slug, "event_title": title, "event_date": ev_date,
                            "mu": b["mu"], "fighter1": b["f1"], "fighter2": b["f2"],
                            "f1_open": s1[0], "f1_close": s1[1],
                            "f2_open": s2[0], "f2_close": s2[1],
                            "f1_ticks": s1[2], "f2_ticks": s2[2],
                            "t_open": s1[3], "t_close": s1[4]})
                n_done += 1
                if n_done % 100 == 0:
                    fh.flush()
                    print(f"  {n_done} bouts ({slug})", flush=True)
    print(f"done: {n_done} new bouts -> {out_path}")


if __name__ == "__main__":
    a = sys.argv[1:] or ["2022", "2027"]
    main(int(a[0]), int(a[1]))
