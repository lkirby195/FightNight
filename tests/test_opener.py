"""Opener validity (scrape_bfo.validated_open, 2026-09-30) and what rides on
it: bout_summary's fields, card_settle.paired_summary's validated opens,
betting_system's VOID rows and the live-flag debut rule."""
import pandas as pd
import pytest

import betting_system as bs
import card_settle
import scrape_bfo

H = 60 * 60 * 1000          # an hour in BFO's millisecond timestamps
M2, M8 = 2 * 60 * 1000, 8 * 60 * 1000


def test_first_paired_tick_is_the_open_when_nothing_is_wrong():
    p1, p2 = [(0, 2.6), (H, 2.44)], [(0, 1.54), (H, 1.61)]
    assert scrape_bfo.validated_open(p1, p2) == (2.6, 1.54, 0, False, 2.6, 1.54)
    assert scrape_bfo.validated_open([(5, 1.5)], [(5, 2.6)]) == (1.5, 2.6, 5, False, 1.5, 2.6)


def test_demopoulos_jauregui_opener_is_the_corrected_pair():
    # BFO 2026-09-26 mu 44971: -850 / +596 at 22:28 UTC, +596 / -850 at 22:36
    dem = [(0, 1.1176), (M8, 6.96), (10 * 60 * 1000, 5.51)]
    jau = [(0, 6.96), (M8, 1.1176), (10 * 60 * 1000, 1.1667)]
    assert scrape_bfo.validated_open(dem, jau) == (6.96, 1.1176, M8, True, 1.1176, 6.96)


def test_a_pair_is_corrected_as_a_whole_when_one_side_jumps():
    # a 50-point move on side 1 only: the open is the whole second pair, never
    # side 1's second tick next to side 2's first
    p1 = [(0, 1.5), (M2, 6.0), (4 * 60 * 1000, 5.5)]           # -200 -> +500
    p2 = [(0, 2.69), (M2, 1.125), (4 * 60 * 1000, 1.14)]       # +169 -> -800
    assert scrape_bfo.validated_open(p1, p2) == (6.0, 1.125, M2, True, 1.5, 2.69)


def test_crossing_even_money_alone_is_not_a_correction():
    # Perez / Dumont mu 44969: -200 / +169 at 22:28, -106 / -110 at 22:30:
    # 15-point moves that happen to cross even money on the Dumont side. A
    # line move, not a data-entry slip: the first pair stands, no flag.
    per = [(0, 1.5), (M2, 1.943), (4 * 60 * 1000, 1.862)]
    dum = [(0, 2.69), (M2, 1.909), (4 * 60 * 1000, 2.0)]
    assert scrape_bfo.validated_open(per, dum) == (1.5, 2.69, 0, False, 1.5, 2.69)
    # a -110/-110 placeholder replaced by -175/+150 moves 11 and 12 points: no flag
    assert scrape_bfo.validated_open([(0, 1.909), (M2, 1.571)],
                                     [(0, 1.909), (M2, 2.5)])[3] is False


def test_a_jump_over_forty_points_is_suspect():
    s = lambda v: scrape_bfo.validated_open([(0, 2.0), (H, v)], [(0, 2.0), (H, 2.0)])[3]
    assert s(1 / 0.09) is True      # 50 -> 9 pts: 41
    assert s(1 / 0.11) is False     # 50 -> 11 pts: 39


def test_a_correction_later_than_an_hour_is_a_line_move_not_a_correction():
    late = scrape_bfo.validated_open([(0, 1.1176), (H + 1, 6.96)], [(0, 6.96), (H + 1, 1.1176)])
    assert late == (1.1176, 6.96, 0, False, 1.1176, 6.96)
    on_time = scrape_bfo.validated_open([(0, 1.1176), (H, 6.96)], [(0, 6.96), (H, 1.1176)])
    assert on_time == (6.96, 1.1176, H, True, 1.1176, 6.96)


def test_pairs_start_at_the_first_instant_both_sides_have_a_price():
    # side 2 starts later: its first tick pairs with side 1's price at that time
    p1, p2 = [(0, 1.5), (M2, 1.6)], [(M8, 2.5), (3 * H, 2.4)]
    assert scrape_bfo.paired_ticks(p1, p2) == [(M8, 1.6, 2.5), (3 * H, 1.6, 2.4)]
    assert scrape_bfo.validated_open(p1, p2) == (1.6, 2.5, M8, False, 1.6, 2.5)


def test_bout_summary_carries_the_validated_open_and_the_raw_ticks():
    js1 = [{"data": [[0, 1.1176], [M8, 6.96], [3 * H, 7.2]]}]
    js2 = [{"data": [[0, 6.96], [M8, 1.1176], [3 * H, 1.1067]]}]
    assert scrape_bfo.bout_summary(js1, js2) == dict(
        f1_open=6.96, f1_close=7.2, f2_open=1.1176, f2_close=1.1067,
        f1_ticks=3, f2_ticks=3, t_open=M8, t_close=3 * H,
        open_suspect=1, f1_open_raw=1.1176, f2_open_raw=6.96)
    clean = scrape_bfo.bout_summary([{"data": [[0, 2.6], [3 * H, 2.44]]}],
                                    [{"data": [[0, 1.54], [3 * H, 1.61]]}])
    assert (clean["f1_open"], clean["open_suspect"], clean["f1_open_raw"]) == (2.6, 0, 2.6)
    assert scrape_bfo.bout_summary([{"data": []}], js2) is None
    assert scrape_bfo.series_summary(js1) == (1.1176, 7.2, 3, 0, 3 * H)     # one side, raw


def test_paired_summary_opens_are_validated_and_the_raw_ticks_follow():
    s1 = [{"data": [[0, 1.1176], [M8, 6.96], [3 * H, 7.2]]}]
    s2 = [{"data": [[0, 6.96], [M8, 1.1176], [3 * H, 1.1067]]}]
    r = card_settle.paired_summary(s1, s2)
    assert r[:4] == (6.96, 7.2, 1.1176, 1.1067)
    assert r[7:] == (1, 1.1176, 6.96)


# --------------------------------------------------- betting_system on it --
def frame(**over):
    """One bfo_joined-style row as betting_system.load() leaves it: Demopoulos
    (a) vs Jauregui (b) with the corrected opener, the model on Jauregui."""
    base = dict(fight_id="f1", event_date="2026-09-26", event_name="E", bout_order=1,
                fighter_a_id="A", fighter_a="Vanessa Demopoulos",
                fighter_b_id="B", fighter_b="Yazmin Jauregui", method="U-DEC", rnd=3,
                year=2026, p_model_a=0.20, y_a=0,
                a_open=6.96, b_open=1.1176, a_open_raw=1.1176, b_open_raw=6.96,
                open_suspect=1, ok_raw=True, ok_val=True, rd_a=120.0, rd_b=110.0,
                projectable=True)
    base.update(over)
    return bs.signal_cols(pd.DataFrame([base]))


@pytest.fixture
def no_captures(monkeypatch):
    nan = lambda B: pd.Series(float("nan"), index=B.index)
    monkeypatch.setattr(bs, "placeable_prices", lambda B: (nan(B), nan(B)))


def test_a_signal_only_the_raw_opener_fires_is_logged_void(no_captures):
    # validated: Jauregui -850 vs model 80% -> nothing; raw: Jauregui +596 -> FLIP
    B = bs.run(frame())
    assert len(B) == 1
    r = B.iloc[0]
    assert r.reason == bs.VOID_SUSPECT and r.rule == "FLIP" and not r.mod1
    assert (r.u, r.placed, r.stake_u, r.pnl, r.pnl_at_open, r.pnl_placed) == (0, 0, 0, 0, 0, 0)
    assert (r.a_open, r.b_open) == (1.1176, 6.96)       # the line it fired on
    assert len(bs.counted(B)) == 0                      # no summary counts it


def test_the_same_pick_at_the_validated_open_is_a_real_signal_not_void(no_captures):
    # model 95% on Jauregui: GAP at -850 (fair -1900) and FLIP at the raw +596
    B = bs.run(frame(p_model_a=0.05))
    assert len(B) == 1 and B.iloc[0].rule == "GAP" and B.iloc[0].reason == ""
    assert int(B.iloc[0].u) == 1 and len(bs.counted(B)) == 1


def test_a_clean_opener_never_voids(no_captures):
    B = bs.run(frame(open_suspect=0, a_open_raw=6.96, b_open_raw=1.1176, p_model_a=0.05))
    assert list(B.reason) == [""] and list(B.rule) == ["GAP"]


def test_a_pick_that_exists_only_under_the_validated_open_is_not_on_record(no_captures):
    # 2026-05-02 is a live card reported at the raw opener. Model 95% on
    # Jauregui (fair -1900); raw opener -110/-110 fires GAP there too -> on
    # record. Raw opener Jauregui -2500 instead: nothing fires there, the GAP
    # exists only at the validated -850 -> never on record, live 0.
    on = bs.run(frame(event_date="2026-05-02", p_model_a=0.05,
                      a_open_raw=1.909, b_open_raw=1.909))
    off = bs.run(frame(event_date="2026-05-02", p_model_a=0.05,
                       a_open_raw=20.0, b_open_raw=1.04))
    assert on.iloc[0].rule == "GAP" and int(on.iloc[0].live) == 1
    assert off.iloc[0].rule == "GAP" and int(off.iloc[0].live) == 0
    # from OPEN_RULE_FROM the report itself evaluates the validated open
    new = bs.run(frame(event_date="2026-10-03", p_model_a=0.05,
                       a_open_raw=20.0, b_open_raw=1.04))
    assert int(new.iloc[0].live) == 1


def test_a_raw_signal_whose_validated_pair_fails_the_sanity_filter_is_void(no_captures):
    # McMillen / Montes 2026-07-18: raw +125/-145 fired FLIP; the "corrected"
    # pair -116/+141 sums to 0.95 and fails [1.00, 1.12]. Logged VOID, not lost.
    D = frame(event_date="2026-07-18", p_model_a=0.70, a_open=1.86, b_open=2.41,
              a_open_raw=2.25, b_open_raw=1.69)
    D["ok_val"] = (1 / D.a_open + 1 / D.b_open).between(1.0, 1.12)
    D["ok_raw"] = (1 / D.a_open_raw + 1 / D.b_open_raw).between(1.0, 1.12)
    assert not bool(D.ok_val.iloc[0]) and bool(D.ok_raw.iloc[0])
    B = bs.run(D)
    assert len(B) == 1 and B.iloc[0].reason == bs.VOID_SUSPECT and B.iloc[0].rule == "FLIP"
    assert int(B.iloc[0].live) == 1 and len(bs.counted(B)) == 0


def test_debut_rule_makes_an_unprojectable_bout_live_zero(no_captures):
    live = bs.run(frame(open_suspect=0, a_open_raw=6.96, b_open_raw=1.1176, p_model_a=0.05))
    debut = bs.run(frame(open_suspect=0, a_open_raw=6.96, b_open_raw=1.1176, p_model_a=0.05,
                         projectable=False))
    assert bs.is_live("2026-09-26")                     # the card itself is live
    assert int(live.iloc[0].live) == 1 and int(debut.iloc[0].live) == 0
    # placement is untouched by the flag
    assert (live.iloc[0].placed, live.iloc[0].stake_u) == (debut.iloc[0].placed, debut.iloc[0].stake_u)
