"""signals.classify: the one signal path. The frozen rule (2026-09-08), the
pre-filter, the pick'em tie-break and the debut rule (both ruled 2026-10-22),
opener validity (2026-09-30) and the placement policy v1 stake (2026-09-09),
each pinned where it is decided."""
import pandas as pd
import pytest

import signals as sg

PK = 1.90909                        # -110 a side: a pick'em


def test_constants_carry_the_frozen_values():
    assert (sg.FLIP_CONF, sg.GAP_PTS, sg.FLIP_UNITS, sg.GAP_UNITS) == (0.65, 100, 1, 1)
    assert (sg.OVERROUND_LO, sg.OVERROUND_HI) == (1.00, 1.12)
    assert (sg.SKIP_LINE_MAX, sg.UNKNOWN_RD, sg.RAMP_LO, sg.RAMP_HI) == (250, 160, 130, 200)
    assert sg.VOID_SUSPECT == "void: suspect opener"


def test_pre_filter_is_the_overround_band():
    assert sg.rule_at(0.90, 1.5, 3.0) is not None          # sum 1.000: in
    assert sg.rule_at(0.90, 1.5, 2.8) is not None          # 1.024: in
    assert sg.rule_at(0.90, 1.3, 1.5) is None              # 1.436: out
    assert sg.classify(0.90, 1.3, 1.5, 100, 100).rule is None


def test_flip_needs_the_market_underdog_at_sixty_five():
    # f1 +150 (2.5), f2 -175 (1.571): the model on f1
    r = sg.rule_at(0.70, 2.5, 1.571)
    assert r[:2] == ("FLIP", 1) and r[2] == pytest.approx(0.70)
    assert sg.rule_at(0.649, 2.5, 1.571) is None
    assert sg.rule_at(0.65, 2.5, 1.571)[0] == "FLIP"       # at the threshold
    # same bout seen from the other side: side 2 is the pick
    assert sg.rule_at(0.30, 1.571, 2.5)[:2] == ("FLIP", 2)


def test_gap_needs_a_hundred_points_over_the_favourite_fair_line():
    # f1 -200 (1.5), model 80% -> fair -400: gap 200
    r = sg.rule_at(0.80, 1.5, 2.6)
    assert r[:2] == ("GAP", 1) and r[3] == pytest.approx(200)
    # model 60% -> fair -150: gap 50
    assert sg.rule_at(0.60, 1.5, 2.6) is None
    # on the underdog at 60%: not a FLIP (below 0.65), not a GAP (not the favourite)
    assert sg.rule_at(0.40, 1.5, 2.6) is None


def test_pick_em_has_no_market_favourite_so_the_gap_test_applies():
    """Price / Shahbazyan 2026-10-10: -110 / -110, model 81% on Price. The
    report (Price listed second) said FLIP, the ledger (Price listed first)
    said GAP. One answer now, whichever side is listed first: GAP."""
    assert sg.market_favourite(PK, PK) is None
    a = sg.rule_at(0.814, PK, PK)                          # Price as fighter1
    b = sg.rule_at(1 - 0.814, PK, PK)                      # Price as fighter2
    assert a[0] == b[0] == "GAP" and (a[1], b[1]) == (1, 2)
    assert a[3] == pytest.approx(b[3]) and a[3] > sg.GAP_PTS
    # at 66% the gap over -110 is 84 points: no signal on either side, where the
    # old orientation-dependent test fired a FLIP when the pick was listed second
    assert sg.rule_at(0.66, PK, PK) is None and sg.rule_at(0.34, PK, PK) is None
    assert sg.rule_at(0.68, PK, PK)[0] == "GAP"            # 103 points


def test_debut_rule_both_fighters_must_be_on_record():
    d = sg.classify(0.80, 1.5, 2.6, 100, 350, known2=False)
    assert d == sg.Decision(projectable=False, reason=sg.REASON_DEBUT)
    assert d.signal is None and d.stake == 0.0 and not d.placed
    assert sg.classify(0.80, 1.5, 2.6, 350, 100, known1=False).reason == sg.REASON_DEBUT
    assert sg.classify(0.80, 1.5, 2.6, 100, 100).projectable
    assert sg.classify(None, 1.5, 2.6, 100, 100).reason == sg.REASON_NO_PROJECTION


def test_first_bout_dates_and_on_record():
    fv = pd.DataFrame(dict(event_date=["2026-03-01", "2026-06-01", "2026-06-01"],
                           fighter_a_id=["A", "A", "C"], fighter_b_id=["B", "C", "D"]))
    first = sg.first_bout_dates(fv)
    assert first == {"A": "2026-03-01", "B": "2026-03-01", "C": "2026-06-01", "D": "2026-06-01"}
    assert sg.on_record("A", "2026-06-01", first)
    assert not sg.on_record("C", "2026-06-01", first)      # debuts that night
    assert sg.on_record("C", "2026-06-02", first)
    assert not sg.on_record("Z", "2026-06-02", first)      # never on file


def test_signal_decision_carries_the_price_of_record_and_the_stake():
    # GAP on f1 at open -200; the price of record (capture) is -222 (1.45)
    d = sg.classify(0.80, 1.5, 2.6, 150, 100, dec1=1.45, dec2=2.7)
    assert d.signal == ("GAP", 1) and (d.open_line, d.line) == (-200, -222)
    assert d.stake == pytest.approx(round(1 - (150 - 130) / 70 * 0.5, 2))   # 0.86u
    assert d.placed and d.skip == "" and not d.void and d.reason == ""
    # no price of record: the open is the line
    assert sg.classify(0.80, 1.5, 2.6, 100, 100).line == -200


def test_policy_v1_skips_are_in_the_decision():
    far = sg.classify(0.70, 3.6, 1.3, 100, 100)             # FLIP at +260
    assert far.signal == ("FLIP", 1) and far.stake == 0.0 and far.skip == "skip: +260"
    unk = sg.classify(0.80, 1.5, 2.6, 170, 170)             # both RD > 160
    assert unk.signal == ("GAP", 1) and unk.stake == 0.0 and unk.skip == "skip: both unknown"
    assert sg.classify(0.80, 1.5, 2.6, 170, 100).stake > 0  # one side known enough


def test_stake_units_pins_policy_v1():
    assert sg.stake_units(sg.RAMP_LO, 100, -110) == 1.0
    assert sg.stake_units(sg.RAMP_HI, 100, -110) == 0.5
    assert sg.stake_units(sg.RAMP_HI + 100, 100, -110) == 0.5
    assert sg.stake_units((sg.RAMP_LO + sg.RAMP_HI) / 2, 100, -110) == pytest.approx(0.75)
    assert sg.stake_units(100, 100, sg.SKIP_LINE_MAX) == 1.0
    assert sg.stake_units(100, 100, sg.SKIP_LINE_MAX + 1) == 0.0
    assert sg.stake_units(sg.UNKNOWN_RD + 1, sg.UNKNOWN_RD + 1, -110) == 0.0
    assert sg.stake_units(350, 100, +150) == 0.5
    assert sg.stake_units(121, 100, -138) == 1.0            # Joshua Van, UFC 331
    assert sg.stake_units(103, 210, 504) == 0.0             # Tai Tuivasa, UFC 331


def test_suspect_opener_that_fires_only_raw_is_void():
    # Demopoulos / Jauregui: raw -850 / +596, validated +596 / -850, model 80% on Jauregui
    d = sg.classify(0.20, 6.96, 1.1176, 120, 110, o1_raw=1.1176, o2_raw=6.96, open_suspect=True)
    assert d.void and d.reason == sg.VOID_SUSPECT and d.signal == ("FLIP", 2)
    assert d.open_line == 596 and d.stake == 0.0 and not d.placed
    # the same pick firing at the validated open is a real signal, not void
    v = sg.classify(0.05, 6.96, 1.1176, 120, 110, o1_raw=1.1176, o2_raw=6.96, open_suspect=True)
    assert v.signal == ("GAP", 2) and not v.void and v.reason == ""
    # a clean opener never voids, whatever the raw ticks say
    c = sg.classify(0.20, 6.96, 1.1176, 120, 110, o1_raw=1.1176, o2_raw=6.96, open_suspect=False)
    assert c.rule is None and not c.void


def test_no_signal_is_a_projectable_decision_without_a_rule():
    d = sg.classify(0.55, 1.9, 1.9, 100, 100)
    assert d.projectable and d.rule is None and d.signal is None and d.stake == 0.0
