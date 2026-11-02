"""scrape_v2: an event page dated today or later is never cached, and pass A
refuses a card whose every bout has a blank winner (the 2026-11-02 trap: the
Thursday gate had cached UFC Vegas 123's page pre-event, Sunday's pass A wrote
the card with no results)."""
import pytest

import scrape_v2 as sv


def test_fetch_cache_false_neither_reads_nor_writes_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "CACHE", str(tmp_path))
    served = ["<html>v1</html>"]
    monkeypatch.setattr(sv.SESS, "get", lambda url: served[-1])
    assert sv.fetch("/x", cache=False) == "<html>v1</html>"
    assert list(tmp_path.iterdir()) == []                   # nothing written
    assert sv.fetch("/x") == "<html>v1</html>"
    assert len(list(tmp_path.iterdir())) == 1               # cached now
    served.append("<html>v2</html>")
    assert sv.fetch("/x") == "<html>v1</html>"              # cache hit
    assert sv.fetch("/x", cache=False) == "<html>v2</html>"  # live, cache not read
    assert sv.fetch("/x") == "<html>v1</html>"              # ... and not overwritten


def test_event_page_dated_today_or_later_is_fetched_without_the_cache(monkeypatch):
    monkeypatch.setattr(sv, "_today", lambda: "2026-10-31")
    calls = []

    def fetch(url, cache=True):
        calls.append((url, cache))
        return "<html></html>"
    monkeypatch.setattr(sv, "fetch", fetch)
    for d in ("2026-10-30", "2026-10-31", "2026-11-07"):
        assert sv.scrape_event_fights({"event_id": "e1", "event_name": "x", "event_date": d}) == []
    assert [c for _, c in calls] == [True, False, False]
    assert all(u == "http://ufcstats.com/event-details/e1" for u, _ in calls)


def _ev(**kw):
    base = dict(event_id="0a66de3a797cf73a", event_name="UFC Fight Night: Moicano vs. Nolan",
                event_date="2026-10-31", fight_id="f", outcome="")
    return dict(base, **kw)


def test_check_results_raises_when_every_winner_of_a_card_is_blank():
    blank = [_ev(fight_id=str(i)) for i in range(12)]
    with pytest.raises(RuntimeError, match=r"Moicano vs. Nolan 2026-10-31 \(0a66de3a797cf73a\)"):
        sv.check_results(blank)
    # one result in: not the blank-card case (a card mid-posting is not refused)
    sv.check_results(blank[:11] + [_ev(fight_id="x", outcome="A_WIN")])
    # a good card does not cover for a blank one
    good = [dict(_ev(), event_id="b", event_name="UFC 333", event_date="2026-10-24",
                 outcome="A_WIN")]
    sv.check_results(good)
    with pytest.raises(RuntimeError, match="0a66de3a797cf73a"):
        sv.check_results(good + blank)
    sv.check_results([])
