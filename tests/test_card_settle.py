"""card_settle: off-card bouts are dropped before their line histories are
pulled, and the paired-tick close ignores in-play ticks."""
import functools
from pathlib import Path

import pytest

import card_report
import card_settle

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
    monkeypatch.setattr(card_report, "main", lambda slug, d: ran.append((slug, d)))
    card_settle.main("ufc-331-4302", "2026-09-19")
    assert ran == [("ufc-331-4302", "2026-09-19")]
    installed = card_report.get_card
    assert isinstance(installed, functools.partial)
    assert installed.func is card_settle.get_card_clean
    assert installed.keywords == {"off": frozenset({44891})}     # not the 2026-01-01 entry


def test_paired_summary_takes_the_last_clean_book_not_the_in_play_tick():
    s1 = [{"data": [[1, 2.60], [2, 2.44], [3, 5.30]]}]      # Pantoja: open +160, close +144, in-play +430
    s2 = [{"data": [[1, 1.54], [2, 1.61], [3, 1.556]]}]     # van: 1/5.3 + 1/1.556 = 0.83 -> in-play
    f1o, f1c, f2o, f2c, n, t, skipped, suspect, f1r, f2r = card_settle.paired_summary(s1, s2)
    assert (f1o, f2o) == (2.60, 1.54)
    assert (f1c, f2c, t, skipped) == (2.44, 1.61, 2, 1)
    assert n == 3
    assert (suspect, f1r, f2r) == (0, 2.60, 1.54)       # a clean opener: raw == validated
