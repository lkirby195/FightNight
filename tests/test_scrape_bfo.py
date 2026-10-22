"""scrape_bfo: the event date comes from the page, never the sitemap lastmod.

Fixtures (tests/fixtures/, see the comment at the top of each):
  bfo_ufc-331-4302.html  UFC 331 page saved 2026-09-22. Carries the schema.org
      SportsEvent JSON-LD that DATE_RE reads. BFO served that card under three
      alias slugs whose sitemap lastmods were 2026-09-10, 2026-09-19 and
      2026-09-20; the old month-name DATE_RE matched nothing on the page and
      the lastmod fallback dated the 13 bouts three different ways.
  bfo_ufc-vegas-121-4368.html  UFC Vegas 121 page saved 2026-10-22. No JSON-LD
      anywhere on the page (none served since October 2026); the date is the
      meta description's "on September 26, 2026" (META_DATE_RE).
"""
import csv
import re
from pathlib import Path

import pytest

import scrape_bfo

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "bfo_ufc-331-4302.html"
FIXTURE_V121 = FIXTURES / "bfo_ufc-vegas-121-4368.html"
SLUGS = ["ufc-331-4302", "ufc-331-4303", "ufc-331-4358"]
MUS = [41067, 44399, 44890, 44397, 44892, 44756, 44398, 44893, 44894, 44758,
       44757, 44759, 44891]
MUS_V121 = [44968, 44969, 45050, 44976, 44977, 44975, 44973, 44979, 44978,
            44974, 44972, 44971, 44970]


@pytest.fixture
def page() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def page_v121() -> str:
    return FIXTURE_V121.read_text(encoding="utf-8")


def serve(monkeypatch, html: str):
    """Every event-page fetch returns `html`; anything else is a test error."""
    def fetch(url, binary=False, refresh=False):
        assert url.startswith(f"{scrape_bfo.BASE}/events/"), url
        return html
    monkeypatch.setattr(scrape_bfo, "fetch", fetch)


def strip_json_ld(html: str) -> str:
    """The page with its JSON-LD startDate field removed."""
    out = re.sub(r'"startDate":\s*"[^"]*",?', "", html)
    assert out != html and '"startDate"' not in out
    return out


def strip_meta(html: str) -> str:
    """The page with an empty meta description."""
    out = re.sub(r'<meta name="description" content="[^"]*"',
                 '<meta name="description" content=""', html)
    assert out != html
    return out


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


def test_json_ld_decides_when_present_even_if_the_meta_description_differs(monkeypatch, page):
    html = page.replace("on September 19, 2026", "on September 20, 2026")
    assert html != page
    serve(monkeypatch, html)
    assert scrape_bfo.parse_event("ufc-331-4302")[1] == "2026-09-19"


def test_meta_description_dates_a_page_with_no_json_ld(monkeypatch, page):
    html = strip_json_ld(page)
    assert scrape_bfo.META_DATE_RE.search(html).group(1) == "September 19, 2026"
    serve(monkeypatch, html)
    title, ev_date, bouts = scrape_bfo.parse_event("ufc-331-4302")
    assert ev_date == "2026-09-19"                 # same date the JSON-LD gave
    assert [b["mu"] for b in bouts] == MUS


def test_vegas_121_page_has_no_json_ld_and_dates_from_the_meta_description(monkeypatch, page_v121):
    assert '"startDate"' not in page_v121 and "ld+json" not in page_v121
    assert scrape_bfo.DATE_RE.search(page_v121) is None
    assert scrape_bfo.META_DATE_RE.search(page_v121).group(1) == "September 26, 2026"
    serve(monkeypatch, page_v121)
    title, ev_date, bouts = scrape_bfo.parse_event("ufc-vegas-121-4368")
    assert ev_date == "2026-09-26"
    assert title.startswith("UFC Vegas 121 Odds")
    assert title.endswith("Sep 26, 2026 | Best Fight Odds")
    assert [b["mu"] for b in bouts] == MUS_V121
    assert (bouts[0]["f1"], bouts[0]["f2"]) == ("Raoni Barcelos", "Raul Rosas Jr")
    assert (bouts[-1]["f1"], bouts[-1]["f2"]) == ("Mickey Gall", "Sedriques Dumas")


@pytest.mark.parametrize("mutate", [
    lambda h: strip_meta(strip_json_ld(h)),                      # neither form
    lambda h: h.replace('"startDate": "2026-09-19"', '"startDate": ""'),
    lambda h: h.replace('"startDate": "2026-09-19"', '"startDate": "2026-13-45"'),
    lambda h: strip_json_ld(h).replace("on September 19, 2026", "on Sept 19, 2026"),
])
def test_parse_event_raises_on_a_page_without_a_parseable_date(monkeypatch, page, mutate):
    """No JSON-LD and no readable meta date; a JSON-LD field that is present
    but empty or invalid (the field decides whenever it is there, the meta
    description is not consulted); no JSON-LD and a meta date in a form the
    regex does not read."""
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
