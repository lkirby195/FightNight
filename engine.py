import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(os.path.join(_ROOT, "data"))  # all data reads/writes live in data/
import csv, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from bs4 import BeautifulSoup
from session import new_session, get

BASE = "http://ufcstats.com"
tls = threading.local()

def sess():
    if not hasattr(tls, "s"):
        tls.s = new_session()
    return tls.s

def parse_listing():
    html = get(sess(), BASE + "/statistics/events/completed?page=all")
    soup = BeautifulSoup(html, "lxml")
    events = []
    for row in soup.select("tr.b-statistics__table-row"):
        a = row.select_one("a.b-link")
        d = row.select_one("span.b-statistics__date")
        if a and d:
            date = datetime.strptime(d.get_text(strip=True), "%B %d, %Y").date()
            events.append({"url": a["href"], "name": a.get_text(strip=True), "date": date})
    return events

def parse_event(ev):
    html = get(sess(), ev["url"])
    soup = BeautifulSoup(html, "lxml")
    fights = []
    rows = soup.select("tr.b-fight-details__table-row[data-link]")
    # rows are main event first; bout_order 1 = first fight of the night
    total = len(rows)
    for idx, row in enumerate(rows):
        cols = row.select("td")
        if len(cols) < 10:
            continue
        flags = [i.get_text(strip=True).lower() for i in cols[0].select("i, a i")]
        flag_text = cols[0].get_text(" ", strip=True).lower()
        links = cols[1].select("a")
        if len(links) < 2:
            continue
        fA = {"id": links[0]["href"].rstrip("/").split("/")[-1], "name": links[0].get_text(strip=True)}
        fB = {"id": links[1]["href"].rstrip("/").split("/")[-1], "name": links[1].get_text(strip=True)}
        method_ps = [p.get_text(strip=True) for p in cols[7].select("p")]
        method = method_ps[0] if method_ps else cols[7].get_text(strip=True)
        detail = method_ps[1] if len(method_ps) > 1 else ""
        wc = cols[6].get_text(strip=True)
        if "nc" in flag_text.split():
            outcome = "NC"
        elif "draw" in flag_text:
            outcome = "DRAW"
        elif "win" in flag_text:
            outcome = "A_WIN"  # first-listed fighter won
        else:
            outcome = "UNKNOWN"
        fights.append({
            "event_name": ev["name"], "event_date": ev["date"].isoformat(),
            "bout_order": total - idx,  # 1 = earliest bout
            "fighter_a_id": fA["id"], "fighter_a": fA["name"],
            "fighter_b_id": fB["id"], "fighter_b": fB["name"],
            "outcome": outcome, "method": method, "method_detail": detail,
            "weight_class": wc,
        })
    return fights

def main():
    events = parse_listing()
    today = datetime(2026, 7, 30).date()
    events = [e for e in events if e["date"] <= today]
    print(f"{len(events)} completed events")
    all_fights, errors = [], []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(parse_event, e): e for e in events}
        done = 0
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                all_fights.extend(fut.result())
            except Exception as ex_:
                errors.append((e["name"], str(ex_)))
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(events)} events, {len(all_fights)} fights")
    all_fights.sort(key=lambda f: (f["event_date"], f["bout_order"]))
    with open("fights.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_fights[0].keys()))
        w.writeheader()
        w.writerows(all_fights)
    print(f"DONE: {len(all_fights)} fights, {len(errors)} errors")
    for name, err in errors[:10]:
        print("ERR", name, err)

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{time.time()-t0:.0f}s")
