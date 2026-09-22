"""Placement policy v1 stake: ramp endpoints and both skip conditions.
The constants are frozen (2026-09-09); these tests pin their effect."""
import pytest

from betting_system import RAMP_HI, RAMP_LO, SKIP_LINE_MAX, UNKNOWN_RD, stake_units


def test_ramp_endpoints():
    assert stake_units(RAMP_LO, 100, -110) == 1.0
    assert stake_units(RAMP_LO - 50, 100, -110) == 1.0           # below the ramp: full unit
    assert stake_units(RAMP_HI, 100, -110) == 0.5
    assert stake_units(RAMP_HI + 100, 100, -110) == 0.5          # floored at 0.5u
    assert stake_units((RAMP_LO + RAMP_HI) / 2, 100, -110) == pytest.approx(0.75)
    assert stake_units(121, 100, -138) == 1.0                    # Joshua Van, UFC 331


def test_skip_when_the_bet_side_is_longer_than_plus_250():
    assert stake_units(100, 100, SKIP_LINE_MAX) == 1.0           # +250 exactly is placed
    assert stake_units(100, 100, SKIP_LINE_MAX + 1) == 0.0
    assert stake_units(103, 210, 504) == 0.0                     # Tai Tuivasa, UFC 331
    assert stake_units(RAMP_HI, 100, 300) == 0.0                 # price skip beats the ramp


def test_skip_when_both_fighters_are_unknown():
    assert stake_units(UNKNOWN_RD + 1, UNKNOWN_RD + 1, -110) == 0.0
    assert stake_units(UNKNOWN_RD, UNKNOWN_RD, -110) == pytest.approx(
        1 - (UNKNOWN_RD - RAMP_LO) / (RAMP_HI - RAMP_LO) * 0.5)  # 160 exactly: placed, ramped
    assert stake_units(UNKNOWN_RD + 1, UNKNOWN_RD, -110) > 0     # one side known enough
    assert stake_units(350, 100, +150) == 0.5                    # debutant backed vs a known opponent
