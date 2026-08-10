"""UFC Stats v2 scrape: per-fight stats + fighter attributes.

Pass A (fast, 782 requests): event tables -> fights_v2.csv
    per fighter per fight: KD, sig strikes landed, TD, sub attempts
    plus weight class, method, round, time, bout order, title flag.
Pass B (2.7k requests): fighter pages -> fighters_v2.csv
    DOB, height, reach, stance.
Pass C (optional, 9k requests): fight-details pages -> fight_stats_v2.csv
    strike ATTEMPTS, control time, head/body/leg, distance/clinch/ground.

All raw HTML is disk-cached under cache/, so re-runs cost nothing.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import sys
from datetime import datetime

from bs4 import BeautifulSoup

from session import UFCStatsSession

CACHE = "cache"
OUT = "data"
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

SESS = UFCStatsSession(delay=0.35)


def fetch(url: str) -> str:
    key = hashlib.sha1(url.encode()).hexdigest()
    path = os.path.join(CACHE, key + ".html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    html = SESS.get(url)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return html


def _id(href: str) -> str:
    return href.rstrip("/").split("/")[-1]


def _int(txt: str):
    txt = (txt or "").strip()
    return int(txt) if txt.isdigit() else None


# ---------------------------------------------------------------- pass A
def scrape_events() -> list[dict]:
    soup = BeautifulSoup(fetch("/statistics/events/completed?page=all"), "lxml")
    events = []
    for tr in soup.select("tr.b-statistics__table-row"):
        a = tr.select_one("a.b-link")
        d = tr.select_one("span.b-statistics__date")
        if not (a and d):
            continue
        tds = tr.select("td")
        events.append({
            "event_id": _id(a["href"]),
            "event_name": a.get_text(strip=True),
            "event_date": datetime.strptime(d.get_text(strip=True), "%B %d, %Y").date().isoformat(),
            "location": tds[1].get_text(strip=True) if len(tds) > 1 else "",
        })
    events.sort(key=lambda e: e["event_date"])
    return events


def scrape_event_fights(event: dict) -> list[dict]:
    soup = BeautifulSoup(fetch(f"http://ufcstats.com/event-details/{event['event_id']}"), "lxml")
    rows = soup.select("tr.b-fight-details__table-row")
    fights, order = [], 0
    for tr in rows:
        link = tr.get("data-link")
        if not link:
            continue
        cols = tr.select("td.b-fight-details__table-col")
        if len(cols) < 10:
            continue
        txt = [[p.get_text(strip=True) for p in c.select("p")] for c in cols]
        fighter_links = [a["href"] for a in cols[1].select("a.b-link")]
        if len(fighter_links) < 2:
            continue
        belt = any("belt" in (img.get("src") or "") for img in cols[6].select("img"))
        wclass = txt[6][0] if txt[6] else ""
        result = txt[0][0].lower() if txt[0] else ""
        order += 1
        fights.append({
            "fight_id": _id(link),
            "event_id": event["event_id"],
            "event_date": event["event_date"],
            "event_name": event["event_name"],
            "bout_order": order,
            "fighter_a_id": _id(fighter_links[0]),
            "fighter_a": txt[1][0],
            "fighter_b_id": _id(fighter_links[1]),
            "fighter_b": txt[1][1],
            "outcome": {"win": "A_WIN", "draw": "DRAW", "nc": "NC"}.get(result, result.upper()),
            "method": (txt[7][0] if txt[7] else ""),
            "method_detail": (txt[7][1] if len(txt[7]) > 1 else ""),
            "round": _int(txt[8][0] if txt[8] else ""),
            "time": (txt[9][0] if txt[9] else ""),
            "weight_class": wclass,
            "title_bout": int(belt or "title" in wclass.lower()),
            "a_kd": _int(txt[2][0] if txt[2] else ""),
            "b_kd": _int(txt[2][1] if len(txt[2]) > 1 else ""),
            "a_sig_str": _int(txt[3][0] if txt[3] else ""),
            "b_sig_str": _int(txt[3][1] if len(txt[3]) > 1 else ""),
            "a_td": _int(txt[4][0] if txt[4] else ""),
            "b_td": _int(txt[4][1] if len(txt[4]) > 1 else ""),
            "a_sub_att": _int(txt[5][0] if txt[5] else ""),
            "b_sub_att": _int(txt[5][1] if len(txt[5]) > 1 else ""),
        })
    return fights


# ---------------------------------------------------------------- pass B
_MONTHS = "%b %d, %Y"


def scrape_fighter(fid: str) -> dict:
    soup = BeautifulSoup(fetch(f"http://ufcstats.com/fighter-details/{fid}"), "lxml")
    name_el = soup.select_one("span.b-content__title-highlight")
    rec_el = soup.select_one("span.b-content__title-record")
    row = {"fighter_id": fid,
           "name": name_el.get_text(strip=True) if name_el else "",
           "record": rec_el.get_text(strip=True).replace("Record:", "").strip() if rec_el else ""}
    for li in soup.select("li.b-list__box-list-item"):
        parts = li.get_text(" ", strip=True).split(":", 1)
        if len(parts) != 2:
            continue
        k, v = parts[0].strip().lower(), " ".join(parts[1].split())
        if k == "height":
            m = re.match(r"(\d+)'\s*(\d+)", v)
            row["height_in"] = int(m.group(1)) * 12 + int(m.group(2)) if m else None
        elif k == "reach":
            m = re.match(r"(\d+)", v)
            row["reach_in"] = int(m.group(1)) if m else None
        elif k == "stance":
            row["stance"] = v or None
        elif k == "dob":
            try:
                row["dob"] = datetime.strptime(v, _MONTHS).date().isoformat()
            except ValueError:
                row["dob"] = None
    return row


FIGHT_COLS = ["fight_id", "event_id", "event_date", "event_name", "bout_order",
              "fighter_a_id", "fighter_a", "fighter_b_id", "fighter_b",
              "outcome", "method", "method_detail", "round", "time",
              "weight_class", "title_bout",
              "a_kd", "b_kd", "a_sig_str", "b_sig_str",
              "a_td", "b_td", "a_sub_att", "b_sub_att"]
FIGHTER_COLS = ["fighter_id", "name", "record", "height_in", "reach_in", "stance", "dob"]


def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(which="A"):
    if which in ("A", "AB"):
        events = scrape_events()
        print(f"events: {len(events)}", flush=True)
        fights = []
        for i, ev in enumerate(events, 1):
            fights += scrape_event_fights(ev)
            if i % 50 == 0:
                print(f"  {i}/{len(events)} events, {len(fights)} fights", flush=True)
        write_csv(f"{OUT}/fights_v2.csv", FIGHT_COLS, fights)
        print(f"wrote {OUT}/fights_v2.csv  n={len(fights)}")
    if which in ("B", "AB"):
        ids = set()
        with open(f"{OUT}/fights_v2.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ids.add(r["fighter_a_id"])
                ids.add(r["fighter_b_id"])
        ids = sorted(ids)
        print(f"fighters: {len(ids)}", flush=True)
        rows = []
        for i, fid in enumerate(ids, 1):
            rows.append(scrape_fighter(fid))
            if i % 200 == 0:
                print(f"  {i}/{len(ids)}", flush=True)
        write_csv(f"{OUT}/fighters_v2.csv", FIGHTER_COLS, rows)
        print(f"wrote {OUT}/fighters_v2.csv  n={len(rows)}")



# ---------------------------------------------------------------- pass C
_XY = re.compile(r"(\d+)\s+of\s+(\d+)")


def _pair(cell_texts):
    """['54 of 113','39 of 63'] -> [(54,113),(39,63)]; '---'/plain ints handled."""
    out = []
    for t in cell_texts[:2]:
        m = _XY.search(t or "")
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
        elif (t or "").strip().isdigit():
            out.append((int(t), None))
        else:
            out.append((None, None))
    while len(out) < 2:
        out.append((None, None))
    return out


def _ctrl_sec(t):
    try:
        mm, ss = t.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return None


def scrape_fight_details(fight_id: str) -> dict | None:
    soup = BeautifulSoup(fetch(f"http://ufcstats.com/fight-details/{fight_id}"), "lxml")
    tables = soup.select("table")
    if len(tables) < 3:
        return None  # very old fights have no stats tables
    row = {"fight_id": fight_id}
    # table 0: totals
    tds = tables[0].select("tbody tr")[0].select("td")
    cell = lambda i: [p.get_text(strip=True) for p in tds[i].select("p")]
    names = cell(0)
    (a_ss, b_ss) = _pair(cell(2))
    (a_ts, b_ts) = _pair(cell(4))
    (a_td, b_td) = _pair(cell(5))
    row.update(fd_a=names[0], fd_b=names[1] if len(names) > 1 else "",
               a_ss_l=a_ss[0], a_ss_a=a_ss[1], b_ss_l=b_ss[0], b_ss_a=b_ss[1],
               a_ts_l=a_ts[0], a_ts_a=a_ts[1], b_ts_l=b_ts[0], b_ts_a=b_ts[1],
               a_td_l=a_td[0], a_td_a=a_td[1], b_td_l=b_td[0], b_td_a=b_td[1])
    rev = cell(8)
    row["a_rev"], row["b_rev"] = _int(rev[0]), _int(rev[1] if len(rev) > 1 else "")
    ctrl = cell(9)
    row["a_ctrl_s"] = _ctrl_sec(ctrl[0])
    row["b_ctrl_s"] = _ctrl_sec(ctrl[1] if len(ctrl) > 1 else "")
    # table 2: sig-strike breakdown
    tds2 = tables[2].select("tbody tr")[0].select("td")
    cell2 = lambda i: [p.get_text(strip=True) for p in tds2[i].select("p")]
    for idx, tag in [(3, "head"), (4, "body"), (5, "leg"),
                     (6, "dist"), (7, "clinch"), (8, "ground")]:
        (ax, bx) = _pair(cell2(idx))
        row[f"a_{tag}_l"], row[f"a_{tag}_a"] = ax
        row[f"b_{tag}_l"], row[f"b_{tag}_a"] = bx
    return row


DETAIL_COLS = ["fight_id", "fd_a", "fd_b",
               "a_ss_l", "a_ss_a", "b_ss_l", "b_ss_a",
               "a_ts_l", "a_ts_a", "b_ts_l", "b_ts_a",
               "a_td_l", "a_td_a", "b_td_l", "b_td_a",
               "a_rev", "b_rev", "a_ctrl_s", "b_ctrl_s"] + \
              [f"{s}_{t}_{m}" for t in ("head", "body", "leg", "dist", "clinch", "ground")
               for s in ("a", "b") for m in ("l", "a")]


def main_c():
    done = set()
    out_path = f"{OUT}/fight_details_v2.csv"
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            done = {r["fight_id"] for r in csv.DictReader(fh)}
    ids = []
    with open(f"{OUT}/fights_v2.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["fight_id"] not in done:
                ids.append(r["fight_id"])
    print(f"pass C: {len(ids)} to fetch ({len(done)} done)", flush=True)
    mode = "a" if done else "w"
    with open(out_path, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=DETAIL_COLS, extrasaction="ignore")
        if not done:
            w.writeheader()
        for i, fid in enumerate(ids, 1):
            try:
                row = scrape_fight_details(fid)
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {fid}: {e}", flush=True)
                continue
            if row:
                w.writerow(row)
            if i % 250 == 0:
                fh.flush()
                print(f"  {i}/{len(ids)}", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "A"
    main_c() if arg == "C" else main(arg)
