"""scrape_bfo: the event date comes from the page, never the sitemap lastmod.

Fixture: tests/fixtures/bfo_ufc-331-4302.html, a saved copy of the UFC 331
event page (see the comment at its top). BFO served that card under three
alias slugs whose sitemap lastmods were 2026-09-10, 2026-09-19 and 2026-09-20;
the old month-name DATE_RE matched nothing on the page and the lastmod
fallback dated the 13 bouts three different ways.
"""
import csv
import re
from pathlib import Path

import pytest

import scrape_bfo

FIXTURE = Path(__file__).parent / "fixtures" / "bfo_ufc-331-4302.html"
SLUGS = ["ufc-331-4302", "ufc-331-4303", "ufc-331-4358"]
MUS = [41067, 44399, 44890, 44397, 44892, 44756, 44398, 44893, 44894, 44758,
       44757, 44759, 44891]


@pytest.fixture
def page() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def serve(monkeypatch, html: str):
    """Every event-page fetch returns `html`; anything else is a test error."""
    def fetch(url, binary=False, refresh=False):
        assert url.startswith(f"{scrape_bfo.BASE}/events/"), url
        return html
    monkeypatch.setattr(scrape_bfo, "fetch", fetch)


def test_date_re_pins_the_json_ld_start_date(page):
    m = scrape_bfo.DATE_RE.search(page)
    assert m and m.group(1) == "2026-09-19"
    # the page's other date forms are not what the regex reads
    assert "September 19, 2026" in page and "September 19th" in page


def test_parse_event_reads_date_and_13_bouts_from_the_page(monkeypatch, page):
    serve(monkeypatch, page)
    title, ev_date, bouts = scrape_bfo.parse_event("ufc-331-4302")
    assert ev_date == "2026-09-19"
    assert title == "UFC 331: Van vs. Pantoja 2 Odds"
    assert [b["mu"] for b in bouts] == MUS
    assert (bouts[0]["f1"], bouts[0]["f2"]) == ("Alexandre Pantoja", "Joshua van")
    assert (bouts[2]["f1"], bouts[2]["f2"]) == ("Doo Ho Choi", "Patricio Freire")


@pytest.mark.parametrize("mutate", [
    lambda h: re.sub(r'"startDate":\s*"[^"]*",?', "", h),          # field removed
    lambda h: h.replace('"startDate": "2026-09-19"', '"startDate": ""'),
    lambda h: h.replace('"startDate": "2026-09-19"', '"startDate": "2026-13-45"'),
])
def test_parse_event_raises_on_a_page_without_a_parseable_date(monkeypatch, page, mutate):
    html = mutate(page)
    assert html != page
    serve(monkeypatch, html)
    with pytest.raises(ValueError) as e:
        scrape_bfo.parse_event("ufc-331-4303")
    assert str(e.value) == "ufc-331-4303"       # the slug, so the log names the page


def test_three_alias_slugs_yield_one_dated_set_of_rows(monkeypatch, tmp_path, page):
    """One card in the sitemap three times, three different lastmods, one
    byte-identical page: 13 rows, 13 distinct mu, one date (the page's), one
    slug (the first in sort order), and no lastmod anywhere."""
    lastmods = {"ufc-331-4302": "2026-09-19", "ufc-331-4303": "2026-09-10",
                "ufc-331-4358": "2026-09-20"}
    monkeypatch.setattr(scrape_bfo, "ufc_event_slugs",
                        lambda: sorted(lastmods.items()))
    serve(monkeypatch, page)
    calls = []

    def ggd(mu, p, refresh=False):
        calls.append((mu, p))
        o = 1.5 + (mu % 10) / 10 + p / 100        # distinct per (mu, side)
        return [{"data": [[1789000000000, o], [1789600000000, o + 0.1]]}]
    monkeypatch.setattr(scrape_bfo, "ggd", ggd)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    scrape_bfo.main(2026, 2027)

    rows = list(csv.DictReader(open(tmp_path / "data" / "bfo_lines.csv", newline="")))
    assert len(rows) == 13
    assert sorted({int(r["mu"]) for r in rows}) == sorted(MUS)
    assert {r["event_date"] for r in rows} == {"2026-09-19"}
    assert {r["slug"] for r in rows} == {"ufc-331-4302"}
    assert {r["event_title"] for r in rows} == {"UFC 331: Van vs. Pantoja 2 Odds"}
    for r in rows:
        mu = int(r["mu"])
        assert float(r["f1_open"]) == pytest.approx(1.5 + (mu % 10) / 10 + 0.01)
        assert float(r["f2_open"]) == pytest.approx(1.5 + (mu % 10) / 10 + 0.02)
    assert sorted(calls) == sorted((mu, p) for mu in MUS for p in (1, 2))  # pulled once
    text = (tmp_path / "data" / "bfo_lines.csv").read_text()
    assert "2026-09-10" not in text and "2026-09-20" not in text
