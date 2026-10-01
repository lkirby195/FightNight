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

Opening line is the validated open (scrape_bfo.validated_open: the first
paired tick unless the next paired tick within an hour moves either side by
more than 40 implied-probability points, in which case that corrected pair
is the open and the bout is flagged open_suspect) -- the same open every
other consumer of the series uses.

Off-card bouts (card_report.OFF_CARD, keyed by event date and BFO matchup id)
are left out before any line history is pulled, so a cancelled bout neither
reaches the report nor counts in the in-play tick tally.

Usage:  python card_settle.py <bfo-event-slug> <event-date YYYY-MM-DD>
        (same CLI as card_report.py; use this for any event < ~4 months old)
"""
import functools
import sys

import card_report
from scrape_bfo import _pts, fetch, ggd, paired_ticks, validated_open

OR_LO, OR_HI = 1.00, 1.15


def paired_summary(js1, js2):
    """-> (f1_open, f1_close, f2_open, f2_close, n_pairs, t_close, n_skipped,
    open_suspect, f1_open_raw, f2_open_raw) or None if either side has no
    ticks. Opens are the validated opens (scrape_bfo.validated_open); the raw
    first ticks follow so a caller can tell what a suspect opener said."""
    p1, p2 = _pts(js1), _pts(js2)
    if not p1 or not p2:
        return None
    pairs = paired_ticks(p1, p2)          # latest value of each side at or before t
    if not pairs:
        return None
    o1, o2, _, suspect, r1, r2 = validated_open(p1, p2)
    opens = (o1, o2, int(suspect), r1, r2)
    skipped = 0
    for t, d1, d2 in reversed(pairs):
        orr = 1.0 / d1 + 1.0 / d2
        if OR_LO <= orr <= OR_HI:
            return (opens[0], d1, opens[1], d2, len(pairs), t, skipped) + opens[2:]
        skipped += 1
    # no clean pair at all: fall back to first pair, flag loudly
    t, d1, d2 = pairs[0]
    print(f"  ! no paired tick with overround in [{OR_LO},{OR_HI}]; "
          f"using first pair", file=sys.stderr)
    return (opens[0], d1, opens[1], d2, len(pairs), t, skipped) + opens[2:]


def get_card_clean(slug, fresh=False, off=frozenset()):
    """card_report.get_card with paired_summary closes. `off`: BFO matchup ids
    to leave out before their series are pulled (main() derives it from
    card_report.OFF_CARD for the event date)."""
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
        if mu in off:
            print(f"off card: dropped {a1.get_text(strip=True)} / "
                  f"{a2.get_text(strip=True)} (mu {mu})")
            continue
        s = paired_summary(ggd(mu, 1, refresh=fresh), ggd(mu, 2, refresh=fresh))
        if s is None:
            continue
        f1o, f1c, f2o, f2c, _, _, skipped, suspect, f1r, f2r = s
        total_skipped += skipped
        bouts.append(dict(mu=mu, f1=a1.get_text(strip=True), f2=a2.get_text(strip=True),
                          f1_open=f1o, f1_close=f1c, f2_open=f2o, f2_close=f2c,
                          open_suspect=suspect, f1_open_raw=f1r, f2_open_raw=f2r))
    if total_skipped:
        print(f"[card_settle] dropped {total_skipped} in-play ticks across "
              f"{len(bouts)} bouts", file=sys.stderr)
    return bouts


def main(slug, ev_date_s):
    """card_report.main with the in-play-safe fetcher, minus the bouts
    card_report.OFF_CARD lists for this date (card_report.main drops them
    again afterwards, as a backstop)."""
    off = frozenset(mu for d, mu in card_report.OFF_CARD if d == ev_date_s)
    card_report.get_card = functools.partial(get_card_clean, off=off)
    card_report.main(slug, ev_date_s)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
