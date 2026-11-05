"""Weekly cadence, automated: the Thursday report and the Sunday settle.

    python cadence.py report [--final] [--dry-run] [--commit-msg FILE]
    python cadence.py settle [--final] [--commit-msg FILE]

The same two commands site/README.md describes, with the hand steps taken out:
which card, which BFO slug, which bouts are really on the card, whether the
book is priced, whether the results are posted. Every call is safe to repeat:
a run with nothing to do changes nothing and says why. GitHub Actions runs it
on a schedule (.github/workflows/cadence.yml), commits what changed and pushes
to main; Vercel deploys main.

report (Thursday, Friday as the fallback)
  1. The next card on the UFC Stats upcoming list. Nothing within
     REPORT_HORIZON_DAYS -> nothing to do. A capture already on file for its
     date -> nothing to do (placeable_lines.csv is append-only and the first
     capture is the price of record; a second run must not re-post picks).
  2. The card itself from UFC Stats, read live: the authority on what is on it.
  3. The BFO event pages: the events sitemap (refreshed), every page with a
     recent lastmod read live, kept when its own date is within a day of the
     UFC Stats date and at least one of its bouts is on the UFC Stats card.
     Since 2026-11 BFO often splits one card across two pages, one dated the
     Saturday and one the Sunday; all of them are the card, best page first.
  4. Each bout BFO prices is matched to a UFC Stats bout by fighter (names.py:
     aliases, exact or joined match; a spelling near-miss of the actual
     opponent still matches). A BFO bout that matches none is off the card --
     a withdrawal, a phantom listing -- and so is a second listing of a bout
     already matched; both go to card_report --off, so they never reach the
     capture. This replaces adding OFF_CARD rows by hand.
  5. Gate: at least MIN_PRICED of the UFC Stats bouts priced on BFO, else no
     post (the Thursday rule: never post a half card). On the --final run the
     gate failing is an error, so somebody hears about it.
  6. card_report.py <slug>[,<slug>] <UFC Stats date> --off ... (capture),
     then site/build.py.

settle (Sunday, Monday as the fallback)
  1. scrape_v2.py new: the completed cards not on file. None -> nothing to do.
     Results not posted yet -> nothing to do (an error on --final).
  2. features.py, scrape_bfo.py.
  3. card_settle.py for each new card at its UFC Stats date, slugs from the
     Thursday capture (else the BFO rows dated within a day), --off for BFO
     bouts that did not happen. It writes the clean closes to bfo_lines, so
     it comes before clv_eval, which carries them into bfo_joined.
  4. clv_eval.py, prefight_rd.py, betting_system.py all --write-ledger,
     replay_check.py (the merge gate: a FAIL stops the run before anything
     is committed), site/build.py.

Generated files keep the line endings their committed copies have (the repo
mixes CRLF and LF by history; a Linux run would otherwise rewrite every line
of a CRLF file).

Exit status: 0 when the run finished or had nothing to do, non-zero on any
error. The workflow opens an issue on a non-zero exit.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta

REPORT_HORIZON_DAYS = 6       # report the next card only when it is this close
MIN_PRICED = 0.80             # share of UFC Stats bouts BFO must price
SITEMAP_LOOKBACK_DAYS = 21    # BFO pages with a lastmod this recent are candidates
PLACEABLE = "data/placeable_lines.csv"
GENERATED = ("data", "site/out", "site/state")

PY = sys.executable


# ------------------------------------------------------------------ output --
def say(msg: str = "") -> None:
    """Print, and add to the GitHub Actions job summary when there is one."""
    print(msg, flush=True)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


def run(*args: str, summary: bool = False) -> str:
    """Run a pipeline script from the repo root; stop the cadence if it fails.
    summary=True also puts its output in the job summary as a code block."""
    cmd = [PY, "-u", *args]
    print(f"\n$ {' '.join(['python', *args])}", flush=True)
    lines = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace") as p:
        for line in p.stdout:                 # stream it: long steps should not look hung
            print(line, end="", flush=True)
            lines.append(line)
    out = "".join(lines)
    if summary:
        say(f"\n```\n$ python {' '.join(args)}\n{out.rstrip()}\n```\n")
    if p.returncode:
        raise SystemExit(f"cadence: `python {' '.join(args)}` exited {p.returncode}")
    return out


def short_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d:%b} {d.day}"


# --------------------------------------------------------------- matching --
def match_card(bfo_bouts, card):
    """Split BFO bouts (dicts with mu, f1, f2) against a UFC Stats card
    (scrape_v2.scrape_event_fights / fights_v2 rows). -> (on, off, matched):
    `matched` is the set of UFC Stats fight_ids some BFO bout matched. A BFO
    bout is on the card when both names resolve to the two fighters of one
    UFC Stats bout, or when one name is exact and the other a near-miss of
    that bout's other fighter (names.near_miss_pair). A second BFO listing of
    a bout already matched is off too ("duplicate listing"): one capture per
    fight. Each off bout carries `why`."""
    import names
    roster = {}
    for f in card:
        roster[f["fighter_a_id"]] = f["fighter_a"]
        roster[f["fighter_b_id"]] = f["fighter_b"]
    resolve = names.make_resolver(roster.items())
    by_pair = {frozenset((f["fighter_a_id"], f["fighter_b_id"])): f["fight_id"] for f in card}
    on, off, matched = [], [], set()
    for b in bfo_bouts:
        try:
            i1, i2 = resolve(b["f1"], b["f2"])
        except names.NameJoinError:
            i1 = i2 = None
        fid = by_pair.get(frozenset((i1, i2))) if i1 and i2 else None
        if fid is None:
            fid = next((f["fight_id"] for f in card if names.near_miss_pair(
                b["f1"], b["f2"], f["fighter_a"], f["fighter_b"])), None)
        if fid is None:
            off.append(dict(b, why="not on the UFC Stats card"))
        elif fid in matched:
            off.append(dict(b, why="duplicate listing"))
        else:
            on.append(b)
            matched.add(fid)
    return on, off, matched


def card_bouts(slugs, fetch_card):
    """BFO bouts of one card over several pages, in page order, a matchup id
    seen twice kept once (card_report.main reads them the same way)."""
    out, seen = [], set()
    for slug in slugs:
        for b in fetch_card(slug):
            if b["mu"] not in seen:
                seen.add(b["mu"])
                out.append(dict(b, slug=slug))
    return out


def bfo_candidates(ev_date: str, card):
    """-> [(n_matched, slug, title, bfo_date)] for the BFO pages dated within
    a day of ev_date, best first. Sitemap and pages read live."""
    import scrape_bfo as sb
    sb.fetch(f"{sb.BASE}/sitemap-events.xml", refresh=True)
    d0 = date.fromisoformat(ev_date)
    floor = (min(d0, date.today()) - timedelta(days=SITEMAP_LOOKBACK_DAYS)).isoformat()
    out = []
    for slug, lastmod in sb.ufc_event_slugs():
        if lastmod[:10] < floor:
            continue
        try:
            sb.fetch(f"{sb.BASE}/events/{slug}", refresh=True)
            title, bdate, bouts = sb.parse_event(slug)
        except Exception as e:  # noqa: BLE001 -- undated or unreachable page
            print(f"  BFO page {slug} skipped ({e!r})")
            continue
        if abs((date.fromisoformat(bdate) - d0).days) > 1:
            continue
        _, _, matched = match_card(bouts, card)
        if matched:
            out.append((len(matched), slug, title, bdate))
    return sorted(out, key=lambda t: (-t[0], len(t[1]), t[1]))


def captured_dates(path=PLACEABLE) -> dict[str, list[str]]:
    """event_date -> the BFO slugs of its captures, in capture order."""
    out = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in sorted(csv.DictReader(fh), key=lambda r: r["captured_at"]):
                slugs = out.setdefault(r["event_date"], [])
                if r["bfo_slug"] not in slugs:
                    slugs.append(r["bfo_slug"])
    return out


# -------------------------------------------------------------- line ends --
def restore_eol(roots=GENERATED) -> list[str]:
    """Give every modified tracked file under `roots` the line endings of its
    committed copy (all CRLF or all LF). -> the paths rewritten."""
    changed = subprocess.run(["git", "diff", "--name-only", "--", *roots],
                             stdout=subprocess.PIPE, text=True, check=True).stdout.split()
    if not changed:
        return []
    eol = {}
    for line in subprocess.run(["git", "ls-files", "--eol", "--", *changed],
                               stdout=subprocess.PIPE, text=True, check=True).stdout.splitlines():
        info, path = line.split("\t", 1)
        eol[path] = info.split()[0]          # i/lf, i/crlf, i/mixed, i/-text, i/none
    fixed = []
    for path in changed:
        want = eol.get(path)
        if want not in ("i/lf", "i/crlf") or not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            raw = fh.read()
        lf = raw.replace(b"\r\n", b"\n")
        new = lf.replace(b"\n", b"\r\n") if want == "i/crlf" else lf
        if new != raw:
            with open(path, "wb") as fh:
                fh.write(new)
            fixed.append(path)
    return fixed


def finish(commit_msg_path: str | None, subject: str, body: list[str]) -> None:
    fixed = restore_eol()
    if fixed:
        print(f"line endings matched to the committed copies: {', '.join(fixed)}")
    if commit_msg_path:
        with open(commit_msg_path, "w", encoding="utf-8") as fh:
            fh.write(subject + "\n\n" + "\n".join(body) + "\n")
    say(f"\n**{subject}**")


# ----------------------------------------------------------------- report --
def cmd_report(a) -> int:
    import card_report
    import scrape_v2
    today = date.today()
    nxt = [e for e in scrape_v2.upcoming_events() if e["event_date"] >= today.isoformat()]
    if not nxt:
        say("Report: no upcoming card on UFC Stats. Nothing to do.")
        return 0
    ev = nxt[0]
    ev_date = ev["event_date"]
    if date.fromisoformat(ev_date) > today + timedelta(days=REPORT_HORIZON_DAYS):
        say(f"Report: next card is {ev['event_name']} on {ev_date}, more than "
            f"{REPORT_HORIZON_DAYS} days out. Nothing to do.")
        return 0
    if ev_date in captured_dates():
        say(f"Report: {ev['event_name']} ({ev_date}) is already captured. Nothing to do.")
        return 0

    card = scrape_v2.scrape_event_fights(ev, cache=False)
    if not card:
        raise SystemExit(f"cadence: UFC Stats lists no bouts for {ev['event_name']} {ev_date}")
    cands = bfo_candidates(ev_date, card)
    if not cands:
        msg = (f"Report: no BFO page matches {ev['event_name']} ({ev_date}) yet.")
        say(msg)
        if a.final:
            raise SystemExit("cadence: " + msg)
        return 0
    slugs = [c[1] for c in cands]
    pages = ", ".join(f"`{c[1]}` (dated {c[3]})" for c in cands)
    bouts = card_bouts(slugs, lambda sl: card_report.get_card(sl, fresh=True))
    on, off, matched = match_card(bouts, card)
    share = len(matched) / len(card)
    say(f"## {ev['event_name']} — {ev_date}")
    say(f"BFO {'pages' if len(cands) > 1 else 'page'} {pages}; UFC Stats lists {len(card)} "
        f"bouts, BFO prices {len(matched)} of them ({share:.0%}).")
    for b in off:
        say(f"- kept out of the capture ({b['why']}): {b['f1']} / {b['f2']} (mu {b['mu']})")
    unpriced = [f for f in card if f["fight_id"] not in matched]
    for f in unpriced:
        say(f"- not priced yet: {f['fighter_a']} vs {f['fighter_b']}")
    if share < MIN_PRICED:
        msg = (f"Report: only {share:.0%} of the card is priced (need {MIN_PRICED:.0%}). "
               f"Not posting a half card.")
        say(msg)
        if a.final:
            raise SystemExit("cadence: " + msg)
        return 0
    if a.dry_run:
        say("Dry run: no capture, no build.")
        return 0

    slug = ",".join(slugs)
    args = ["card_report.py", slug, ev_date]
    if off:
        args += ["--off", ",".join(str(b["mu"]) for b in off)]
    out = run(*args, summary=True)
    m = re.search(r"signals at open .*?: (\d+)\s+placed \(policy v1\): (\d+)", out)
    n_sig, n_placed = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    run("site/build.py")
    sig_txt = ("no bets" if n_placed == 0 else f"{n_placed} placed of {n_sig} signals"
               if n_sig is not None else "signals: see report")
    subject = (f"Thursday capture: {ev['event_name']} ({short_date(ev_date)}), "
               f"{len(matched)} bouts, {sig_txt}")
    body = [f"Automated (cadence.py report). card_report.py {slug} {ev_date} at "
            f"{datetime.now().astimezone():%Y-%m-%d %H:%M %Z}; BFO dates "
            + "; ".join(f"{c[1]} {c[3]}" for c in cands) + ".",
            f"UFC Stats lists {len(card)} bouts; BFO prices {len(matched)} ({share:.0%})."]
    if off:
        body.append("Kept out of the capture: " + "; ".join(
            f"{b['f1']} / {b['f2']} (mu {b['mu']}, {b['why']})" for b in off) + ".")
    body.append("Site rebuilt.")
    finish(a.commit_msg, subject, body)
    return 0


# ----------------------------------------------------------------- settle --
def settle_slugs(ev_date: str, card) -> list[str]:
    """The BFO slugs to settle a card at: the Thursday capture's, else every
    slug with bfo_lines rows dated within a day that match the card, most
    matches first."""
    cap = captured_dates().get(ev_date)
    if cap:
        return cap
    d0 = date.fromisoformat(ev_date)
    pages = {}
    with open("data/bfo_lines.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if abs((date.fromisoformat(r["event_date"]) - d0).days) <= 1:
                pages.setdefault(r["slug"], []).append(
                    dict(mu=int(r["mu"]), f1=r["fighter1"], f2=r["fighter2"]))
    ranked = sorted(((len(match_card(b, card)[2]), s) for s, b in pages.items()),
                    key=lambda t: (-t[0], t[1]))
    return [s for n, s in ranked if n]


def bust_bfo_cache(ev_dates) -> None:
    """Drop the cached BFO sitemap, and the event page and line series of
    every bout captured for these dates: copies taken before the fight (the
    Thursday capture writes them) would hand scrape_bfo a pre-fight close.
    A no-op on a fresh checkout (CI), where cache_bfo/ starts empty."""
    import hashlib
    import scrape_bfo as sb
    urls = [f"{sb.BASE}/sitemap-events.xml"]
    if os.path.exists(PLACEABLE):
        with open(PLACEABLE, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["event_date"] in ev_dates:
                    urls.append(f"{sb.BASE}/events/{r['bfo_slug']}")
                    urls += [f"{sb.BASE}/api/ggd?m={r['mu']}&p={p}" for p in (1, 2)]
    for url in set(urls):
        path = os.path.join(sb.CACHE, hashlib.sha1(url.encode()).hexdigest())
        if os.path.exists(path):
            os.remove(path)


def cmd_settle(a) -> int:
    import scrape_bfo as sb
    import scrape_v2
    try:
        new = scrape_v2.main_new()
    except scrape_v2.ResultsPending as e:
        say(f"Settle: {e}. Nothing written; try again later.")
        if a.final:
            raise SystemExit(f"cadence: {e}")
        return 0
    if not new:
        say("Settle: no new completed card on UFC Stats. Nothing to do.")
        return 0
    bust_bfo_cache([ev["event_date"] for ev in new])
    t = date.today()
    y0 = t.year - (t.month == 1)             # sitemap lastmod years to read: this one,
    y1 = t.year + 1 + (t.month == 12)        # plus the neighbour across New Year
    run("features.py")
    run("scrape_bfo.py", str(y0), str(y1))

    # card_settle writes the clean closes into bfo_lines, so it runs before
    # clv_eval, which carries them into bfo_joined (the site's Close / CLV)
    _, rows = scrape_v2.read_rows("data/fights_v2.csv")
    body, names_ = ["Automated (cadence.py settle)."], []
    for ev in new:
        card = [r for r in rows if r["event_id"] == ev["event_id"]]
        names_.append(f"{ev['event_name']} ({short_date(ev['event_date'])}), {len(card)} bouts")
        slugs = settle_slugs(ev["event_date"], card)
        if not slugs:
            say(f"- {ev['event_name']}: no BFO lines found; results recorded, no card settle.")
            body.append(f"{ev['event_name']}: no BFO lines found, not settled on BFO.")
            continue

        def page_bouts(sl):
            sb.fetch(f"{sb.BASE}/events/{sl}", refresh=True)
            return sb.parse_event(sl)[2]
        _, off, _ = match_card(card_bouts(slugs, page_bouts), card)
        slug = ",".join(slugs)
        args = ["card_settle.py", slug, ev["event_date"]]
        if off:
            args += ["--off", ",".join(str(b["mu"]) for b in off)]
        out = run(*args, summary=True)
        m = re.search(r"picks (\d+)-(\d+)", out)
        body.append(f"card_settle.py {slug} {ev['event_date']}"
                    + (f": picks {m.group(1)}-{m.group(2)}" if m else "")
                    + (f"; off the card: " + ", ".join(f"{b['f1']} / {b['f2']}" for b in off)
                       if off else "") + ".")

    run("clv_eval.py")
    run("prefight_rd.py")
    run("betting_system.py", "all", "--write-ledger")
    run("replay_check.py", summary=True)
    run("site/build.py")
    body.append("Ledger regenerated (betting_system.py all --write-ledger); replay PASS; "
                "site rebuilt.")
    finish(a.commit_msg, "Sunday settle: " + "; ".join(names_), body)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["report", "settle"])
    ap.add_argument("--final", action="store_true",
                    help="last scheduled try of the week: 'not ready yet' is an error")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only: resolve the card and check the gate, write nothing")
    ap.add_argument("--commit-msg", help="write the commit message here when files changed")
    a = ap.parse_args(argv)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())
    return cmd_report(a) if a.command == "report" else cmd_settle(a)


if __name__ == "__main__":
    sys.exit(main())
