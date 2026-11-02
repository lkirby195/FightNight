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

The clean close of every bout settled here is written back to
data/bfo_lines.csv (f1_close_clean / f2_close_clean, with
inplay_ticks_dropped) next to the raw last-tick close, which stays as it is;
the site shows the clean close and grades CLV at it. `--backfill <year>`
does the same for every row of that year dated before today, from the
cached series. The ledger keeps computing its own clean close
(betting_system._clean_close) and does not read these columns.

Usage:  python card_settle.py <bfo-event-slug> <event-date YYYY-MM-DD>
        (same CLI as card_report.py; use this for any event < ~4 months old)
        python card_settle.py --backfill <year>
"""
import csv
import datetime
import functools
import sys

import card_report
import scrape_bfo
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
                          open_suspect=suspect, f1_open_raw=f1r, f2_open_raw=f2r,
                          inplay_dropped=skipped))
    if total_skipped:
        print(f"[card_settle] dropped {total_skipped} in-play ticks across "
              f"{len(bouts)} bouts", file=sys.stderr)
    return bouts


CLEAN_COLS = ("f1_close_clean", "f2_close_clean", "inplay_ticks_dropped")


def write_clean_closes(bouts, path=scrape_bfo.OUT):
    """Write each bout's clean close (f1_close / f2_close of a get_card_clean
    bout: the last paired tick that is a clean book) and its in-play tick
    count to the bfo_lines row with the same mu. Every other column and row
    is left as it is; a file from before these columns existed gets them
    (blank elsewhere). A bout with no row is reported and skipped: run the
    BFO scrape first. Returns (rows written, mus without a row)."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_mu = {int(b["mu"]): b for b in bouts}
    seen, n = set(), 0
    for r in rows:
        for k in scrape_bfo.COLS:
            r.setdefault(k, "")
        b = by_mu.get(int(r["mu"]))
        if b is None:
            continue
        r["f1_close_clean"], r["f2_close_clean"] = str(b["f1_close"]), str(b["f2_close"])
        r["inplay_ticks_dropped"] = str(int(b["inplay_dropped"]))
        seen.add(int(r["mu"]))
        n += 1
    missing = sorted(set(by_mu) - seen)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=scrape_bfo.COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"clean closes -> {path}: {n} rows"
          + (f"; no bfo_lines row for mu {missing} (run the BFO scrape, then settle again)"
             if missing else ""))
    return n, missing


def backfill(year: int, path=scrape_bfo.OUT, today: str | None = None):
    """Clean closes for every bfo_lines row of `year` dated before today (a
    settled card), each from its own series (cache_bfo/, fetched when
    absent). Rows dated today or later are left alone: their series are
    still moving; a row whose series cannot be read is reported and left
    blank. Returns the bouts written."""
    today = today or datetime.date.today().isoformat()
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bouts, n_inplay, n_differ = [], 0, 0
    for r in rows:
        if r["event_date"][:4] != str(year) or r["event_date"] >= today:
            continue
        mu = int(r["mu"])
        try:
            s = paired_summary(ggd(mu, 1), ggd(mu, 2))
        except Exception as e:  # noqa: BLE001  (a series BFO no longer serves)
            print(f"  GGD ERR mu={mu} {r['event_date']} {r['fighter1']} / {r['fighter2']}: {e!r}")
            s = None
        if s is None:
            print(f"  no series: {r['event_date']} {r['fighter1']} / {r['fighter2']} (mu {mu})")
            continue
        _, f1c, _, f2c, _, _, skipped, _, _, _ = s
        bouts.append(dict(mu=mu, f1_close=f1c, f2_close=f2c, inplay_dropped=skipped))
        n_inplay += skipped
        n_differ += (f1c, f2c) != (float(r["f1_close"]), float(r["f2_close"]))
    n, _ = write_clean_closes(bouts, path)
    print(f"backfill {year}: {n} rows written; clean close differs from the raw close on "
          f"{n_differ}; {n_inplay} in-play ticks dropped")
    return bouts


def main(slug, ev_date_s):
    """card_report.main with the in-play-safe fetcher, minus the bouts
    card_report.OFF_CARD lists for this date (card_report.main drops them
    again afterwards, as a backstop); then the clean closes of the bouts it
    reported go to bfo_lines (write_clean_closes)."""
    off = frozenset(mu for d, mu in card_report.OFF_CARD if d == ev_date_s)
    card_report.get_card = functools.partial(get_card_clean, off=off)
    bouts = card_report.main(slug, ev_date_s)
    if bouts:
        write_clean_closes(bouts)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--backfill":
        backfill(int(sys.argv[2]))
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
