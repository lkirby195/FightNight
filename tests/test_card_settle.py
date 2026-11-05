"""card_settle: off-card bouts are dropped before their line histories are
pulled, the paired-tick close ignores in-play ticks, and the clean close is
written back to bfo_lines next to the raw one."""
import csv
import functools
from pathlib import Path

import pytest

import card_report
import card_settle
import scrape_bfo

FIXTURE = Path(__file__).parent / "fixtures" / "bfo_ufc-331-4302.html"
MUS = [41067, 44399, 44890, 44397, 44892, 44756, 44398, 44893, 44894, 44758,
       44757, 44759, 44891]


@pytest.fixture
def served(monkeypatch):
    """The saved UFC 331 page and a synthetic clean series per (mu, side);
    returns the list of (mu, side) pulls."""
    html = FIXTURE.read_text(encoding="utf-8")
    monkeypatch.setattr(card_settle, "fetch", lambda url, **kw: html)
    calls = []

    def ggd(mu, p, refresh=False):
        calls.append((mu, p))
        o = 1.8 if p == 1 else 2.1                    # overround 1.03: clean book
        return [{"data": [[1789000000000, o], [1789600000000, o]]}]
    monkeypatch.setattr(card_settle, "ggd", ggd)
    return calls


def test_off_card_bout_is_skipped_before_its_series_is_pulled(served, capsys):
    bouts = card_settle.get_card_clean("ufc-331-4302", off=frozenset({44891}))
    assert [b["mu"] for b in bouts] == [m for m in MUS if m != 44891]
    assert not any(mu == 44891 for mu, _ in served)
    assert "off card: dropped Brian Ortega / Renato Moicano (mu 44891)" in capsys.readouterr().out


def test_without_off_every_bout_is_pulled(served):
    bouts = card_settle.get_card_clean("ufc-331-4302")
    assert [b["mu"] for b in bouts] == MUS
    assert sorted(served) == sorted((mu, p) for mu in MUS for p in (1, 2))


def test_main_takes_off_card_from_card_report_for_that_date_only(monkeypatch):
    monkeypatch.setattr(card_report, "OFF_CARD",
                        {("2026-09-19", 44891), ("2026-01-01", 41067)})
    monkeypatch.setattr(card_report, "get_card", card_report.get_card)  # restored after
    ran = []
    monkeypatch.setattr(card_report, "main", lambda slug, d, **kw: ran.append((slug, d)))
    card_settle.main("ufc-331-4302", "2026-09-19")
    assert ran == [("ufc-331-4302", "2026-09-19")]
    installed = card_report.get_card
    assert isinstance(installed, functools.partial)
    assert installed.func is card_settle.get_card_clean
    assert installed.keywords == {"off": frozenset({44891})}     # not the 2026-01-01 entry


RAW_COLS = scrape_bfo.COLS[:-3]          # the file before the clean-close columns


def _row(mu, date, f1c, f2c):
    r = dict.fromkeys(RAW_COLS, "")
    r.update(slug="s", event_title="t", event_date=date, mu=str(mu), fighter1="A", fighter2="B",
             f1_open="2.6", f2_open="1.54", f1_close=str(f1c), f2_close=str(f2c),
             f1_ticks="3", f2_ticks="3", t_open="1", t_close="3", open_suspect="0",
             f1_open_raw="2.6", f2_open_raw="1.54")
    return r


def _write(path, rows, cols=RAW_COLS):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def test_write_clean_closes_fills_the_matched_rows_and_leaves_the_rest(tmp_path, capsys):
    path = tmp_path / "bfo_lines.csv"
    _write(path, [_row(1, "2026-01-10", 5.3, 1.556), _row(2, "2026-01-10", 1.9, 1.95)])
    n, missing = card_settle.write_clean_closes(
        [dict(mu=1, f1_close=2.44, f2_close=1.61, inplay_dropped=1),
         dict(mu=9, f1_close=1.0, f2_close=1.0, inplay_dropped=0)], path)
    assert (n, missing) == (1, [9])
    assert "no bfo_lines row for mu [9]" in capsys.readouterr().out
    cols, rows = _read(path)
    assert cols == scrape_bfo.COLS                       # columns added once
    r1, r2 = rows
    assert (r1["f1_close"], r2["f1_close"]) == ("5.3", "1.9")       # raw untouched
    assert (r1["f1_close_clean"], r1["f2_close_clean"], r1["inplay_ticks_dropped"]) == ("2.44", "1.61", "1")
    assert (r2["f1_close_clean"], r2["f2_close_clean"], r2["inplay_ticks_dropped"]) == ("", "", "")
    assert path.read_bytes().count(b"\r\n") == 3           # the file's own line ending


def test_main_writes_the_clean_closes_of_the_bouts_card_report_reported(monkeypatch):
    bouts = [dict(mu=44397, f1_close=2.44, f2_close=1.61, inplay_dropped=1)]
    monkeypatch.setattr(card_report, "get_card", card_report.get_card)
    monkeypatch.setattr(card_report, "main", lambda slug, d, **kw: bouts)
    written = []
    monkeypatch.setattr(card_settle, "write_clean_closes", lambda b: written.append(b))
    card_settle.main("ufc-331-4302", "2026-09-19")
    assert written == [bouts]
    written.clear()
    monkeypatch.setattr(card_report, "main", lambda slug, d, **kw: None)   # nothing reported
    card_settle.main("ufc-331-4302", "2026-09-19")
    assert written == []


def test_backfill_writes_settled_rows_of_the_year_only(tmp_path, monkeypatch):
    path = tmp_path / "bfo_lines.csv"
    _write(path, [_row(1, "2026-01-10", 5.3, 1.556),       # settled: in-play raw close
                  _row(2, "2026-11-02", 5.3, 1.556),       # today: left alone
                  _row(3, "2025-12-06", 5.3, 1.556)])      # other year
    monkeypatch.setattr(card_settle, "ggd", lambda mu, p, refresh=False: (
        [{"data": [[1, 2.60], [2, 2.44], [3, 5.30]]}] if p == 1
        else [{"data": [[1, 1.54], [2, 1.61], [3, 1.556]]}]))
    out = card_settle.backfill(2026, path, today="2026-11-02")
    assert out == [dict(mu=1, f1_close=2.44, f2_close=1.61, inplay_dropped=1)]
    _, rows = _read(path)
    assert [r["f1_close_clean"] for r in rows] == ["2.44", "", ""]
    assert [r["inplay_ticks_dropped"] for r in rows] == ["1", "", ""]
    assert [r["f1_close"] for r in rows] == ["5.3", "5.3", "5.3"]


def test_get_card_clean_reports_the_in_play_ticks_per_bout(served):
    bouts = card_settle.get_card_clean("ufc-331-4302")
    assert all(b["inplay_dropped"] == 0 for b in bouts)


def test_paired_summary_takes_the_last_clean_book_not_the_in_play_tick():
    s1 = [{"data": [[1, 2.60], [2, 2.44], [3, 5.30]]}]      # Pantoja: open +160, close +144, in-play +430
    s2 = [{"data": [[1, 1.54], [2, 1.61], [3, 1.556]]}]     # van: 1/5.3 + 1/1.556 = 0.83 -> in-play
    f1o, f1c, f2o, f2c, n, t, skipped, suspect, f1r, f2r = card_settle.paired_summary(s1, s2)
    assert (f1o, f2o) == (2.60, 1.54)
    assert (f1c, f2c, t, skipped) == (2.44, 1.61, 2, 1)
    assert n == 3
    assert (suspect, f1r, f2r) == (0, 2.60, 1.54)       # a clean opener: raw == validated
