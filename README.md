import hashlib, re, requests

BASE = "http://ufcstats.com"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

def solve_challenge(html, session):
    m = re.search(r'nonce="([^"]+)"', html)
    t = re.search(r"new Array\((\d+)\+1\)\.join\('0'\)", html)
    if not m:
        return False
    nonce = m.group(1)
    zeros = "0" * int(t.group(1)) if t else "00"
    n = 0
    while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(zeros):
        n += 1
    r = session.post(BASE + "/__c", data={"nonce": nonce, "n": n}, headers=UA, timeout=30)
    return r.status_code < 300

def get(session, url, retries=3):
    for _ in range(retries):
        r = session.get(url, headers=UA, timeout=30)
        if "Checking your browser" in r.text[:600]:
            solve_challenge(r.text, session)
            continue
        if r.status_code == 200:
            return r.text
    raise RuntimeError(f"failed: {url}")

def new_session():
    s = requests.Session()
    r = s.get(BASE + "/statistics/events/completed", headers=UA, timeout=30)
    if "Checking your browser" in r.text[:600]:
        solve_challenge(r.text, s)
    return s

if __name__ == "__main__":
    s = new_session()
    html = get(s, BASE + "/statistics/events/completed?page=all")
    print(len(html), "chars")
    print("event links:", html.count("event-details"))
