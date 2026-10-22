"""polymarket.py: the moneyline of an event, name matching against outcome
labels, find_bout / card_bouts over a saved public-search response, and the
two price-history readings the 2026 backfill uses.

Fixture: tests/fixtures/pm_search_talbott.json, three events trimmed from
GET gamma-api.polymarket.com/public-search?q=Talbott on 2026-09-30: the open
UFC 332 Figueiredo / Talbott event (moneyline + props), the resolved
Cejudo / Talbott event whose outcomes are bare surnames, and a Yes/No-only
"who fights next" event."""
import json
from pathlib import Path

import pytest

import polymarket as pm

FIXTURE = Path(__file__).parent / "fixtures" / "pm_search_talbott.json"


@pytest.fixture
def served(monkeypatch):
    js = json.loads(FIXTURE.read_text(encoding="utf-8"))
    calls = []

    def fetch_json(url, refresh=False):
        calls.append(url)
        if "/public-search" in url:
            return js
        if "/events?tag_slug=ufc" in url:
            return js["events"]
        raise AssertionError(url)
    monkeypatch.setattr(pm, "fetch_json", fetch_json)
    return calls


def test_slug_date():
    assert pm.slug_date("ufc-dei-pay-2026-10-03") == "2026-10-03"
    assert pm.slug_date("ufc-raul-rosas-jr-next-fight") is None


def test_moneyline_is_the_two_fighter_market_and_props_are_not(served):
    js = json.loads(FIXTURE.read_text(encoding="utf-8"))
    by = {e["slug"]: e for e in js["events"]}
    m = pm.moneyline(by["ufc-dei-pay-2026-10-03"])
    assert json.loads(m["outcomes"]) == ["Deiveson Figueiredo", "Payton Talbott"]
    assert pm.moneyline(by["who-will-merab-dvalishivili-fight-next"]) is None


def test_same_fighter_handles_suffixes_surnames_and_near_misses():
    assert pm.same_fighter("Raul Rosas Jr.", "Raul Rosas Jr")
    assert pm.same_fighter("Cejudo", "Henry Cejudo")             # bare surname outcome
    assert pm.same_fighter("Sangcha'an", "Sangcha-An")
    assert not pm.same_fighter("Deiveson Figueiredo", "Daniel Figueiredo")
    assert not pm.same_fighter("Talbott", "Payton Talbot")


def test_find_bout_matches_both_names_and_the_slug_date_a_first(served):
    r = pm.find_bout("Deiveson Figueiredo", "Payton Talbott", "2026-10-03")
    assert r["slug"] == "ufc-dei-pay-2026-10-03" and r["market_id"] == "4740048"
    assert (r["a_idx"], r["b_idx"]) == (0, 1)
    assert (r["p_a"], r["p_b"]) == (0.165, 0.835)
    assert r["token_a"] and r["token_b"] and r["token_a"] != r["token_b"]
    assert r["liquidity"] > 0 and r["closed"] is False
    assert served[0].endswith("/public-search?q=figueiredo&limit_per_type=30")


def test_find_bout_orients_to_the_caller_and_reads_bare_surname_outcomes(served):
    r = pm.find_bout("Payton Talbott", "Deiveson Figueiredo", "2026-10-03")
    assert (r["a_idx"], r["b_idx"]) == (1, 0) and (r["p_a"], r["p_b"]) == (0.835, 0.165)
    r = pm.find_bout("Henry Cejudo", "Payton Talbott", "2025-12-06")
    assert r["slug"] == "ufc-hen-pay-2025-12-06" and r["closed"] is True
    assert r["outcomes"] == ["Cejudo", "Talbott"] and r["p_a"] == 0.0


def test_find_bout_is_none_on_the_wrong_date_or_fighter(served):
    assert pm.find_bout("Deiveson Figueiredo", "Payton Talbott", "2026-10-04") is None
    assert pm.find_bout("Deiveson Figueiredo", "Marlon Vera", "2026-10-03") is None


def test_card_bouts_lists_every_dated_moneyline_for_the_date(served):
    bouts = pm.card_bouts("2026-10-03")
    assert [b["slug"] for b in bouts] == ["ufc-dei-pay-2026-10-03"]
    assert bouts[0]["outcomes"] == ["Deiveson Figueiredo", "Payton Talbott"]
    assert pm.card_bouts("2025-12-06") == [{**b} for b in pm.card_bouts("2025-12-06")]  # closed event still dated


def test_price_at_reads_the_last_point_at_or_before_the_instant():
    h = [(1, 0.5), (2, 0.42), (3, 0.415), (4, 0.0005), (5, 0.0)]
    assert pm.price_at(h, 2.5) == 0.42
    assert pm.price_at(h, 3) == 0.415                   # at the instant itself
    assert pm.price_at(h, 0.5) is None                  # before the first point
    assert pm.price_at(h, 99) == 0.0
    assert pm.price_at([], 3) is None
