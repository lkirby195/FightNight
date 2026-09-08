import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(os.path.join(_ROOT, "data"))  # all data reads/writes live in data/
import csv, re, json, os, sys, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, threading

API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "MMARatingsResearch/0.1 (research contact)"}
CACHE = "mw_cache.json"
tls = threading.local()

def sess():
    if not hasattr(tls, "s"):
        tls.s = requests.Session()
    return tls.s

def wiki_get(params):
    import time, random
    for attempt in range(4):
        try:
            return sess().get(API, params=params, headers=UA, timeout=20).json()
        except ValueError:
            time.sleep(1.0 * (attempt + 1) + random.random())
    return None

def wiki_text(title):
    r = wiki_get({"action": "query", "prop": "extracts", "explaintext": 1,
                  "format": "json", "titles": title, "redirects": 1})
    if not r:
        return ""
    return next(iter(r["query"]["pages"].values())).get("extract", "")

MISS_RE = re.compile(r"(weighed in at [\d.]+.{0,80}?(over|above)|missed weight|failed to make weight)", re.I)

def extract(event_name):
    m = re.match(r"(UFC \d+)", event_name)
    title = m.group(1) if m else event_name
    txt = wiki_text(title)
    if len(txt) < 500:
        r = wiki_get({"action": "query", "list": "search", "srsearch": event_name,
                      "format": "json", "srlimit": 1})
        hits = r["query"]["search"] if r else []
        if hits:
            txt = wiki_text(hits[0]["title"])
    sents = re.split(r"(?<=[.])\s+", txt)
    return [s for s in sents if MISS_RE.search(s)]

def main(chunk_size=140):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    fights = list(csv.DictReader(open("fights.csv")))
    ev_names = sorted(set(f["event_name"] for f in fights if f["event_date"] >= "2013-01-01"))
    todo = [e for e in ev_names if e not in cache][:chunk_size]
    print(f"{len(ev_names)} events total, {len(cache)} cached, doing {len(todo)}")
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(extract, e): e for e in todo}
        done = 0
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                cache[e] = fut.result()
            except Exception:
                pass  # retry next chunk
            done += 1
            if done % 20 == 0:
                json.dump(cache, open(CACHE, "w"))
    json.dump(cache, open(CACHE, "w"))
    n_sent = sum(len(v) for v in cache.values())
    print(f"cached: {len(cache)}/{len(ev_names)} events, {n_sent} miss-sentences")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 140)
