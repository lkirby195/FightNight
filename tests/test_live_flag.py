"""live=1 is capture proof (ruled 2026-10-26): a ledger row is live iff
data/placeable_lines.csv holds a capture for that bout -- same event date,
both fighters -- whose captured_at is strictly before the event date.
Nothing else grants it: not LIVE_CARDS, not the event date, not a freeze
window."""
import csv
from types import SimpleNamespace

import pandas as pd
import pytest

import betting_system as bs

COLS = ["event_date", "bfo_slug", "mu", "fighter1", "fighter2", "f1_line", "f2_line",
        "captured_at", "pm_p_self", "pm_liquidity", "pm_slug"]


def placeable(tmp_path, rows):
    p = tmp_path / "placeable_lines.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(dict({c: "" for c in COLS}, **r))
    return str(p)


def bout(event_date="2026-10-24", a="Jacob Malkoun", b="Abubakar Vagaev"):
    return SimpleNamespace(event_date=event_date, fighter_a=a, fighter_b=b)


def cap(captured_at, event_date="2026-10-24", f1="Abubakar Vagaev", f2="Jacob Malkoun"):
    return dict(event_date=event_date, bfo_slug="ufc-333-4321", mu="45382", fighter1=f1,
                fighter2=f2, f1_line="1.654", f2_line="2.28", captured_at=captured_at)


def test_a_capture_strictly_before_the_event_makes_the_bout_live(tmp_path):
    idx = bs.capture_index(placeable(tmp_path, [cap("2026-10-22T10:56:50-06:00")]))
    assert bs.is_live(bout(), idx)
    # the capture names are in BFO order and spelling; the bout in UFC Stats order
    assert bs.is_live(bout(a="Abubakar Vagaev", b="Jacob Malkoun"), idx)


def test_a_capture_on_the_event_date_or_after_is_not_proof(tmp_path):
    same_day = bs.capture_index(placeable(tmp_path, [cap("2026-10-24T00:30:00-06:00")]))
    assert not bs.is_live(bout(), same_day)
    after = bs.capture_index(placeable(tmp_path, [cap("2026-10-25T09:00:00-06:00")]))
    assert not bs.is_live(bout(), after)
    # the earliest capture decides: one pre-event row among later ones is enough
    mixed = bs.capture_index(placeable(tmp_path, [cap("2026-10-25T09:00:00-06:00"),
                                                   cap("2026-10-23T18:00:00-06:00")]))
    assert bs.is_live(bout(), mixed)


def test_a_bout_with_no_capture_is_not_live(tmp_path):
    idx = bs.capture_index(placeable(tmp_path, [cap("2026-10-22T10:56:50-06:00")]))
    assert not bs.is_live(bout(a="Alexander Volkanovski", b="Movsar Evloev"), idx)
    assert not bs.is_live(bout(event_date="2026-10-25"), idx)         # same names, other date
    assert not bs.is_live(bout(), {})                                  # empty file


def test_a_live_cards_date_without_a_capture_is_not_live(tmp_path):
    assert "2026-09-19" in bs.LIVE_CARDS                              # the list still exists
    idx = bs.capture_index(placeable(tmp_path, [cap("2026-10-22T10:56:50-06:00")]))
    assert not bs.is_live(bout(event_date="2026-09-19", a="Joshua Van", b="Alexandre Pantoja"), idx)
    # and a date outside every old window IS live with a capture
    idx2 = bs.capture_index(placeable(tmp_path, [cap("2027-03-01T12:00:00-06:00", event_date="2027-03-06")]))
    assert bs.is_live(bout(event_date="2027-03-06"), idx2)


def test_run_marks_live_from_captures_only(tmp_path, monkeypatch):
    """The ledger frame: the same signal is live=1 with a pre-event capture and
    live=0 without one; LIVE_CARDS plays no part."""
    monkeypatch.setattr(bs, "_captures", bs.capture_index(
        placeable(tmp_path, [cap("2026-10-22T10:56:50-06:00")])))
    nan = lambda D: pd.Series(float("nan"), index=D.index, dtype=float)
    monkeypatch.setattr(bs, "capture_prices", lambda D: (nan(D), nan(D), nan(D)))
    row = dict(fight_id="f1", event_date="2026-10-24", event_name="UFC 333", bout_order=4,
               fighter_a_id="A", fighter_a="Abubakar Vagaev", fighter_b_id="B",
               fighter_b="Jacob Malkoun", method="U-DEC", rnd=3, year=2026, y_a=1,
               p_model_a=0.29, a_open=1.645, b_open=2.32, a_open_raw=1.645, b_open_raw=2.32,
               open_suspect=0, rd_a=281.0, rd_b=141.0, known_a=True, known_b=True,
               projectable=True)
    B = bs.run(pd.DataFrame([row]))
    assert len(B) == 1 and B.iloc[0].rule == "FLIP" and int(B.iloc[0].live) == 1
    B2 = bs.run(pd.DataFrame([dict(row, event_date="2026-09-19")]))   # in LIVE_CARDS, no capture
    assert int(B2.iloc[0].live) == 0
