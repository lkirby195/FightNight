"""SHADOW_D61, pre-registered 2026-11-03: a shadow tag, logged only. It fires
when the model picks the market underdog and P_model(fav) - P_open(fav) <=
-0.20 and FLIP does not fire. It never changes a signal, is computed only when
asked (the ledger and the report ask only for events from 2026-01-01), is never
placed or counted, and is live only with capture proof for an event after
2026-11-03."""
import csv
import dataclasses
import itertools

import numpy as np
import pandas as pd
import pytest

import betting_system as bs
import card_report
import signals as sg

FAV, DOG = 1.4, 3.2                 # f1 -250 / f2 +220
Q1 = (1 / FAV) / (1 / FAV + 1 / DOG)  # vig-free P(f1) at the open, 0.698


def grid():
    prices = [1.1, 1.25, 1.4, 1.6, 1.9, 2.2, 3.2, 4.0, 6.0]
    for p1 in np.linspace(0.05, 0.95, 37):
        for o1, o2 in itertools.product(prices, prices):
            yield float(p1), o1, o2


def test_shadow_constants():
    assert (sg.SHADOW_D61, sg.SHADOW_GAP) == ("SHADOW_D61", -0.20)
    assert (sg.SHADOW_FROM, sg.SHADOW_FREEZE) == ("2026-01-01", "2026-11-03")
    assert not sg.shadow_window("2025-12-31") and sg.shadow_window("2026-01-01")


def test_fires_on_the_dog_when_the_model_sits_20_pts_under_the_market():
    s = sg.classify(Q1 - 0.21, FAV, DOG, 100, 100, shadow=True).shadow
    assert (s.tag, s.side, s.open_line, s.line) == (sg.SHADOW_D61, 2, 220, 220)
    assert s.conf == pytest.approx(1 - (Q1 - 0.21)) and s.gap == pytest.approx(-0.21)
    flipped = sg.classify(1 - (Q1 - 0.21), DOG, FAV, 100, 100, shadow=True).shadow
    assert flipped.side == 1 and flipped.gap == pytest.approx(-0.21)


def test_no_shadow_short_of_the_gap_or_when_the_model_picks_the_favourite():
    assert sg.classify(Q1 - 0.19, FAV, DOG, 100, 100, shadow=True).shadow is None
    # P_model(fav) exactly 0.50 is not a dog pick, however far under the market
    assert sg.classify(0.50, 1.25, 4.0, 100, 100, shadow=True).shadow is None


def test_no_shadow_at_a_pickem_outside_the_pre_filter_or_for_a_debut():
    assert sg.classify(0.30, 1.9, 1.9, 100, 100, shadow=True).shadow is None
    assert sg.classify(0.45, 1.2, 2.5, 100, 100, shadow=True).shadow is None   # overround 1.23
    assert sg.classify(Q1 - 0.21, FAV, DOG, 100, 100, known2=False,
                       shadow=True).shadow is None


def test_flip_excludes_it_and_gap_never_co_fires():
    d = sg.classify(0.30, FAV, DOG, 100, 100, shadow=True)       # P(dog) 0.70
    assert d.signal == ("FLIP", 2) and d.shadow is None
    hits = 0
    for p1, o1, o2 in grid():
        d = sg.classify(p1, o1, o2, 100, 100, shadow=True)
        if d.shadow is not None:
            hits += 1
            assert d.rule is None
    assert hits > 0                              # the grid reaches the tag


def test_signal_fields_are_identical_with_and_without_the_shadow():
    for p1, o1, o2 in grid():
        for kw in ({}, dict(o1_raw=o2, o2_raw=o1, open_suspect=True)):
            plain = sg.classify(p1, o1, o2, 120, 170, **kw)
            asked = sg.classify(p1, o1, o2, 120, 170, shadow=True, **kw)
            assert plain.shadow is None
            assert dataclasses.replace(asked, shadow=None) == plain


def test_would_place_and_shadow_stake_follow_policy_v1():
    # dog at +260: policy v1 would skip it; the RD ramp is recorded anyway
    far = sg.classify(0.45, 1.3, 3.6, 100, 165, shadow=True).shadow
    assert (far.line, far.skip, far.would_place, far.stake) == (260, "skip: +260", False, 0.75)
    unk = sg.classify(Q1 - 0.21, FAV, DOG, 170, 170, shadow=True).shadow
    assert (unk.skip, unk.would_place, unk.stake) == ("skip: both unknown", False, 0.71)
    # the price of record moves the line (and the +250 test), never the tag
    cap = sg.classify(0.45, 1.3, 3.6, 100, 165, dec2=3.3, shadow=True).shadow
    assert (cap.open_line, cap.line, cap.would_place) == (260, 230, True)
    assert sg.ramp_units(sg.RAMP_LO) == 1.0 and sg.ramp_units(sg.RAMP_HI + 50) == 0.5
    assert sg.stake_units(165, 100, 220) == sg.ramp_units(165)


COLS = ["event_date", "bfo_slug", "mu", "fighter1", "fighter2", "f1_line", "f2_line",
        "captured_at", "pm_p_self", "pm_liquidity", "pm_slug"]


def bout(fight_id, event_date, p_model_a=0.45, y_a=0):
    return dict(fight_id=fight_id, event_date=event_date, event_name="UFC X", bout_order=1,
                fighter_a_id="A", fighter_a="Alan Fav", fighter_b_id="B", fighter_b="Bob Dog",
                method="KO/TKO", rnd=1, year=int(event_date[:4]), y_a=y_a,
                p_model_a=p_model_a, a_open=FAV, b_open=DOG, a_open_raw=FAV,
                b_open_raw=DOG, open_suspect=0, rd_a=100.0, rd_b=150.0,
                known_a=True, known_b=True, projectable=True)


def test_ledger_window_live_rule_and_never_counted(tmp_path, monkeypatch):
    p = tmp_path / "placeable_lines.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for d in ("2026-10-24", "2026-11-14"):          # each captured two days before
            w.writerow(dict({c: "" for c in COLS}, event_date=d, fighter1="Alan Fav",
                            fighter2="Bob Dog", captured_at=f"{d[:8]}{int(d[8:]) - 2:02d}T10:00"))
    monkeypatch.setattr(bs, "_captures", bs.capture_index(str(p)))
    full = lambda D, x: pd.Series(x, index=D.index, dtype=float)
    monkeypatch.setattr(bs, "capture_prices", lambda D: (full(D, 1.5), full(D, 3.0),
                                                         full(D, np.nan)))
    seen = []
    real = sg.shadow_at
    monkeypatch.setattr(sg, "shadow_at", lambda p1, o1, o2: seen.append(p1) or real(p1, o1, o2))
    D = pd.DataFrame([bout("old", "2025-12-31", p_model_a=0.44), bout("pre", "2026-10-24"),
                      bout("post", "2026-11-14"), bout("nocap", "2026-11-21")])
    S = bs.run_shadow(D)
    assert 0.44 not in seen                      # the 2025 bout is never evaluated
    assert list(S.fight_id) == ["pre", "post", "nocap"]
    assert list(S.live) == [0, 1, 0]             # captured, but on or before the freeze
    assert (S.rule == sg.SHADOW_D61).all() and (S.u == 1).all()
    assert (S.placed == 0).all() and (S.stake_u == 0).all() and (S.pnl_placed == 0).all()
    assert list(S.would_place) == [1, 1, 1] and list(S.shadow_stake) == [0.86] * 3
    post = S[S.fight_id == "post"].iloc[0]       # live: settles at its capture
    assert post.placeable == 3.0 and post.pnl == pytest.approx(2.0)
    assert post.pnl_at_open == pytest.approx(DOG - 1)
    pre = S[S.fight_id == "pre"].iloc[0]         # not live: at open
    assert np.isnan(pre.placeable) and pre.pnl == pytest.approx(DOG - 1)
    assert len(bs.run(D)) == 0                   # never a ledger row, never counted


def test_report_section_is_separate_and_titled(capsys):
    card_report.print_shadows({1: dict(tag=sg.SHADOW_D61, who="Rizvan Kuniev", conf=0.53,
                                       gap=-0.234, opn=222, line=104, rd_self=232, rd_opp=95,
                                       would_place=True, skip="", stake=0.5),
                               2: dict(tag=sg.SHADOW_D61, who="Niko Price", conf=0.55,
                                       gap=-0.21, opn=275, line=275, rd_self=99, rd_opp=192,
                                       would_place=False, skip="skip: +275", stake=1.0)})
    out = capsys.readouterr().out
    assert "\nShadow - not a bet (SHADOW_D61, logged only): 2\n" in out
    assert "would place 0.50u" in out and "would skip: +275" in out
