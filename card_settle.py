"""Card settle: card_report with in-play-safe closing lines.

Problem: BFO keeps live in-play ticks in the /api/ggd series for ~4 months
after an event. `scrape_bfo.series_summary` takes the last tick unconditionally,
so recent cards' "closing" lines are really mid-fight prices (e.g. -2500 on
the fighter who is winning round 3). Historical events (>4 months old) are
pruned server-side and are clean.

Fix: walk BOTH sides' tick series jointly. For each timestamp, take each side's
latest price at-or-before that instant. The closing line is the LAST paired
tick whose two-sided overround  1/d1 + 1/d2  lies in [OR_LO, OR_HI]. In-play
ticks blow the overround far outside that band (one side collapses to ~1.02
while the other lags), so they are skipped and the last pre-fight book wins.

Opening line is unchanged (first tick per side) so historical CLV records are
untouched.

Usage:  python card_settle.py <bfo-event-slug> <event-date YYYY-MM-DD>
        (same CLI as card_report.py; use this for any event < ~4 months old)
"""
import sys
import card_report
from scrape_bfo import fetch, ggd

OR_LO, OR_HI = 1.00, 1.15


def _pts(js):
    out = []
    for srs in js if isinstance(js, list) else [js]:
        for pt in srs.get("data", []):
            t, v = (pt.get("x"), pt.get("y")) if isinstance(pt, dict) else (pt[0], pt[1])
            if v is not None:
                out.append((t, float(v)))
    out.sort()
    return out


def paired_summary(js1, js2):
    """-> (f1_open, f1_close, f2_open, f2_close, n_pairs, t_close, n_skipped)
    or None if either side has no ticks."""
    p1, p2 = _pts(js1), _pts(js2)
    if not p1 or not p2:
        return None
    ts = sorted({t for t, _ in p1} | {t for t, _ in p2})
    # latest value of each side at or before t
    pairs, i, j, v1, v2 = [], 0, 0, None, None
    for t in ts:
        while i < len(p1) and p1[i][0] <= t:
            v1 = p1[i][1]; i += 1
        while j < len(p2) and p2[j][0] <= t:
            v2 = p2[j][1]; j += 1
        if v1 is not None and v2 is not None:
            pairs.append((t, v1, v2))
    if not pairs:
        return None
    skipped = 0
    for t, d1, d2 in reversed(pairs):
        orr = 1.0 / d1 + 1.0 / d2
        if OR_LO <= orr <= OR_HI:
            return p1[0][1], d1, p2[0][1], d2, len(pairs), t, skipped
        skipped += 1
    # no clean pair at all: fall back to first pair, flag loudly
    t, d1, d2 = pairs[0]
    print(f"  ! no paired tick with overround in [{OR_LO},{OR_HI}]; "
          f"using first pair", file=sys.stderr)
    return p1[0][1], d1, p2[0][1], d2, len(pairs), t, skipped


def get_card_clean(slug, fresh=False):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(fetch(f"https://www.bestfightodds.com/events/{slug}",
                               refresh=fresh), "lxml")
    bouts, total_skipped = [], 0
    for tr in soup.select("tr[id^=mu-]"):
        mid = tr["id"].split("-")[1]
        if not mid.isdigit():
            continue
        a1 = tr.select_one("a[href^='/fighters/']")
        tr2 = tr.find_next_sibling("tr")
        a2 = tr2.select_one("a[href^='/fighters/']") if tr2 else None
        if not (a1 and a2):
            continue
        mu = int(mid)
        s = paired_summary(ggd(mu, 1, refresh=fresh), ggd(mu, 2, refresh=fresh))
        if s is None:
            continue
        f1o, f1c, f2o, f2c, _, _, skipped = s
        total_skipped += skipped
        bouts.append(dict(mu=mu, f1=a1.get_text(strip=True), f2=a2.get_text(strip=True),
                          f1_open=f1o, f1_close=f1c, f2_open=f2o, f2_close=f2c))
    if total_skipped:
        print(f"[card_settle] dropped {total_skipped} in-play ticks across "
              f"{len(bouts)} bouts", file=sys.stderr)
    return bouts


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    card_report.get_card = get_card_clean   # swap in the clean fetcher
    card_report.main(sys.argv[1], sys.argv[2])
