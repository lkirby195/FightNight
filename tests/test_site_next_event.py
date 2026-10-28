"""This Week's empty state names the next UFC card from BFO, found the way
card_report reaches a card (events sitemap, then the page's own date), and
falls back to a plain sentence when nothing resolves. No network here."""
import importlib.util
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_build", ROOT / "site" / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

AS_OF = date(2026, 10, 28)
PAGES = {   # slug -> (title, page date)
    "ufc-vegas-123-4427": ("UFC Vegas 123 Odds – Nov 1, 2026 | Best Fight Odds", "2026-11-01"),
    "ufc-334-4433": ("UFC 334 Odds – Nov 14, 2026 | Best Fight Odds", "2026-11-14"),
    "ufc-333-4321": ("UFC 333: Volkanovski vs Evloev Odds", "2026-10-24"),
    "noche-ufc-4299": None,          # stale page, no date
}
SITEMAP = [("ufc-333-4321", "2026-10-24"), ("ufc-334-4433", "2026-11-14"),
           ("ufc-vegas-123-4427", "2026-11-01"), ("noche-ufc-4299", "2026-10-30"),
           ("ufc-100-137", "2009-07-11")]


def fake_bfo(monkeypatch, sitemap=SITEMAP, pages=PAGES, fetched=None):
    monkeypatch.setattr(build.sb, "fetch", lambda url, **kw: (fetched.append(url)
                                                               if fetched is not None else None))
    monkeypatch.setattr(build.sb, "ufc_event_slugs", lambda: sitemap)

    def parse_event(slug):
        page = pages.get(slug)
        if page is None:
            raise ValueError(slug)
        return page[0], page[1], [{"mu": 1, "f1": "A", "f2": "B"}]
    monkeypatch.setattr(build.sb, "parse_event", parse_event)


def test_earliest_upcoming_page_wins_and_settled_or_old_pages_are_not_fetched(monkeypatch):
    fetched = []
    fake_bfo(monkeypatch, fetched=fetched)
    nx = build.next_event(AS_OF, known={"ufc-333-4321"})
    assert nx == dict(name="UFC Vegas 123", date="2026-11-01",
                      date_text="Sun Nov 1, 2026", slug="ufc-vegas-123-4427")
    # the settled card and the 2009 page were never requested
    assert not any("ufc-333-4321" in u or "ufc-100-137" in u for u in fetched)


def test_offline_resolves_to_none(monkeypatch):
    def boom():
        raise OSError("no route to host")
    monkeypatch.setattr(build.sb, "fetch", lambda url, **kw: None)
    monkeypatch.setattr(build.sb, "ufc_event_slugs", boom)
    assert build.next_event(AS_OF, known=set()) is None


def test_undated_page_is_skipped_and_nothing_upcoming_is_none(monkeypatch):
    fake_bfo(monkeypatch, sitemap=[("noche-ufc-4299", "2026-10-30"),
                                   ("ufc-333-4321", "2026-10-24")])
    assert build.next_event(AS_OF, known=set()) is None


def test_event_display_strips_bfo_title_furniture():
    titles = {"a": "UFC 334 Odds – Nov 14, 2026 | Best Fight Odds",
              "b": "UFC 332 for October 4 | Best Fight Odds",
              "c": "UFC 333: Volkanovski vs Evloev Odds"}
    assert build.event_display(None, "a", titles) == "UFC 334"
    assert build.event_display(None, "b", titles) == "UFC 332"
    assert build.event_display(None, "c", titles) == "UFC 333: Volkanovski vs Evloev"
    assert build.event_display("UFC Stats name", "a", titles) == "UFC Stats name"
