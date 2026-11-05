"""UFC Stats v2 scrape: per-fight stats + fighter attributes.

Pass A (fast, 782 requests): event tables -> fights_v2.csv
    per fighter per fight: KD, sig strikes landed, TD, sub attempts
    plus weight class, method, round, time, bout order, title flag.
Pass B (2.7k requests): fighter pages -> fighters_v2.csv
    DOB, height, reach, stance.
Pass C (optional, 9k requests): fight-details pages -> fight_stats_v2.csv
    strike ATTEMPTS, control time, head/body/leg, distance/clinch/ground.
Mode `new` (the weekly cadence, cadence.py settle): passes A and B for what is
    new since data/fights_v2.csv -- the completed-events list fetched live,
    only events not on file, only fighters not on file -- then pass C. Rows
    already on file are left byte-for-byte as they are, so it costs a handful
    of requests and needs no cache/. A full `A` / `B` run is still the way to
    pick up a historical correction (an overturned result).

All raw HTML is disk-cached under cache/, so re-runs cost nothing -- except
an event page dated today or later, which is never cached (its results are
not in yet; a copy taken then would serve a results-less card on Sunday,
as UFC Vegas 123 did on 2026-11-02), and pass A refuses to write a card
whose every bout has a blank winner (check_results).
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


def fetch(url: str, cache: bool = True) -> str:
    """Cached GET. cache=False fetches live and neither reads nor writes the
    cache: for a page still changing (an event not yet fought). Anything
    that pulls an upcoming event page by hand (the Thursday gate) must pass
    cache=False too."""
    key = hashlib.sha1(url.encode()).hexdigest()
    path = os.path.join(CACHE, key + ".html")
    if cache and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    html = SESS.get(url)
    if cache:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
    return html


def _today() -> str:
    return datetime.now().date().isoformat()


def _id(href: str) -> str:
    return href.rstrip("/").split("/")[-1]


def _int(txt: str):
    txt = (txt or "").strip()
    return int(txt) if txt.isdigit() else None


# ---------------------------------------------------------------- pass A
def scrape_events(fresh: bool = False) -> list[dict]:
    """Completed events, oldest first. fresh=True reads the list live (the
    cached copy never learns about a new card)."""
    return _parse_events(fetch("/statistics/events/completed?page=all", cache=not fresh))


def upcoming_events() -> list[dict]:
    """Upcoming events (always live), soonest first."""
    return _parse_events(fetch("/statistics/events/upcoming?page=all", cache=False))


def _parse_events(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
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


def scrape_event_fights(event: dict, cache: bool | None = None) -> list[dict]:
    # an event dated today or later has no results yet: never cache its page.
    # cache=False forces a live read (mode `new`, the Thursday card check).
    if cache is None:
        cache = event["event_date"] < _today()
    soup = BeautifulSoup(fetch(f"http://ufcstats.com/event-details/{event['event_id']}",
                               cache=cache), "lxml")
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
    # UFC Stats lists the main event first. Number bouts chronologically
    # (1 = first bout of the night, N = main event) to match data/fights_v2.csv
    # and the frozen v1 replay order: tournament-era fighters fought several
    # times per night, so within-card order changes the Glicko replay.
    n = len(fights)
    for f in fights:
        f["bout_order"] = n + 1 - f["bout_order"]
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
              "a_td", "b_td", "a_sub_att", "b_sub_att",
              "a_mw", "b_mw"]   # missed-weight flags: not on UFC Stats, carried over
FIGHTER_COLS = ["fighter_id", "name", "record", "height_in", "reach_in", "stance", "dob"]


def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class ResultsPending(RuntimeError):
    """A card UFC Stats lists as completed has a bout with no result yet."""


def read_rows(path):
    """-> (fieldnames, rows as str dicts) of a CSV on file."""
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def main_new(today: str | None = None) -> list[dict]:
    """Incremental scrape for the weekly cadence. Pass A for the completed
    events not in data/fights_v2.csv (list and pages read live), pass B for
    the fighters not in data/fighters_v2.csv, then pass C (incremental by
    fight id). Rows on file are rewritten exactly as read; new fights carry
    a_mw / b_mw = 0 like a full pass A. Refuses, before writing anything, a
    new card with any bout lacking a result (ResultsPending: not posted yet,
    try again later) or one dated before the last card on file (a late
    listing: run the full pass A). -> the new events, oldest first."""
    today = today or _today()
    fpath, dpath = f"{OUT}/fights_v2.csv", f"{OUT}/fighters_v2.csv"
    cols, rows = read_rows(fpath)
    if cols != FIGHT_COLS:
        raise SystemExit(f"{fpath} has columns {cols}; expected {FIGHT_COLS}")
    have = {r["event_id"] for r in rows}
    last = max(r["event_date"] for r in rows)
    new = [e for e in scrape_events(fresh=True)
           if e["event_id"] not in have and e["event_date"] < today]
    if not new:
        print(f"scrape new: nothing new on UFC Stats (last card on file {last})")
        return []
    early = [e for e in new if e["event_date"] < last]
    if early:
        raise RuntimeError("scrape new: " + "; ".join(f"{e['event_name']} {e['event_date']}"
                           for e in early) + f" is dated before the last card on file "
                           f"({last}); run the full `scrape_v2.py A` instead")
    fights = []
    for ev in new:
        fights += scrape_event_fights(ev, cache=False)
    check_results(fights)
    pending = [f for f in fights if not f["outcome"]]
    if pending:
        raise ResultsPending("results not posted yet for " + "; ".join(
            f"{f['fighter_a']} vs {f['fighter_b']} ({f['event_name']} {f['event_date']})"
            for f in pending))
    for f in fights:
        f["a_mw"], f["b_mw"] = "0", "0"
    write_csv(fpath, FIGHT_COLS, rows + fights)
    print(f"scrape new: +{len(fights)} bouts from " + "; ".join(
        f"{e['event_name']} {e['event_date']}" for e in new))
    dcols, drows = read_rows(dpath)
    if dcols != FIGHTER_COLS:
        raise SystemExit(f"{dpath} has columns {dcols}; expected {FIGHTER_COLS}")
    known = {r["fighter_id"] for r in drows}
    ids = sorted({f[k] for f in fights for k in ("fighter_a_id", "fighter_b_id")} - known)
    drows += [scrape_fighter(fid) for fid in ids]
    drows.sort(key=lambda r: r["fighter_id"])
    write_csv(dpath, FIGHTER_COLS, drows)
    print(f"scrape new: +{len(ids)} fighters")
    main_c()
    return new


def check_results(fights: list[dict]) -> None:
    """Refuse a card whose every bout has a blank winner: the event page was
    read before UFC Stats posted the results (a pre-event copy in cache/, or
    a card listed under completed events too early). Raises RuntimeError
    naming the events; nothing is written."""
    by_ev: dict[tuple, list[str]] = {}
    for f in fights:
        by_ev.setdefault((f["event_id"], f["event_name"], f["event_date"]), []).append(f["outcome"])
    bad = [k for k, outs in by_ev.items() if outs and all(o == "" for o in outs)]
    if bad:
        names = "; ".join(f"{n} {d} ({eid})" for eid, n, d in bad)
        raise RuntimeError(f"results missing for every bout of: {names} -- the event page "
                           f"was read before the results were posted; delete its copy under "
                           f"{CACHE}/ (sha1 of http://ufcstats.com/event-details/<event_id>) "
                           f"and run pass A again")


def main(which="A"):
    if which in ("A", "AB"):
        events = scrape_events()
        print(f"events: {len(events)}", flush=True)
        fights = []
        for i, ev in enumerate(events, 1):
            fights += scrape_event_fights(ev)
            if i % 50 == 0:
                print(f"  {i}/{len(events)} events, {len(fights)} fights", flush=True)
        check_results(fights)
        # Missed-weight flags (a_mw/b_mw, 0/1) are not on UFC Stats. Carry them
        # over from the existing file by fight_id so a re-scrape does not drop
        # them; fights not in the previous file default to 0 (unverified).
        prev = {}
        if os.path.exists(f"{OUT}/fights_v2.csv"):
            with open(f"{OUT}/fights_v2.csv", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    prev[r["fight_id"]] = (r.get("a_mw") or "0", r.get("b_mw") or "0")
        for f in fights:
            f["a_mw"], f["b_mw"] = prev.get(f["fight_id"], ("0", "0"))
        n_mw = sum((f["a_mw"] == "1") + (f["b_mw"] == "1") for f in fights)
        write_csv(f"{OUT}/fights_v2.csv", FIGHT_COLS, fights)
        print(f"wrote {OUT}/fights_v2.csv  n={len(fights)}  missed-weight flags carried over: {n_mw}")
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
    if arg == "C":
        main_c()
    elif arg == "new":
        main_new()
    else:
        main(arg)
