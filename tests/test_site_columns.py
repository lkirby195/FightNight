"""The one fight table This week and Our record share (2026-11-06): Fight,
Pick, Bet, Prediction odds, Opening line, Current / Closing odds, every odds
column on the fighter in the Pick column. Reads the repo's own data for the
record checks; no network."""
import importlib.util
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_build_cols", ROOT / "site" / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_pick_cols_measures_every_odds_column_on_the_pick():
    c = build.pick_cols(0.30, "Davey Grant", "Elijah Smith", (4.20, 1.30), (4.40, 1.28))
    assert (c["pick"], c["side"], c["note"]) == ("Elijah Smith", 2, None)
    assert c["pred"] == build.sline(build.fair_line(0.70))
    assert c["open"] == build.sline(build.american(1.30))
    assert c["now"] == build.sline(build.american(1.28))


def test_pick_cols_coin_flip_names_the_lean_and_a_debut_has_no_pick():
    c = build.pick_cols(0.51, "Billy Elekana", "Lucas Fernando", (3.6, 1.3), None)
    assert (c["pick"], c["note"], c["now"]) == ("Billy Elekana", "Coin flip", "—")
    d = build.pick_cols(None, "Marcos Degli", "Mitch Raposo")
    assert d["pick"] is None and d["note"] == "UFC debut"
    assert d["pred"] == d["open"] == d["now"] == "—"


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    rec = build.build_record(build.Data(date(2026, 11, 2)), build.Model(), out)
    rec["out"] = out
    return rec


def test_record_row_shows_the_bet_and_the_result_on_the_pick(record):
    # Malkoun FLIP on UFC 333: placed, lost; open +132 on Malkoun
    c = record["cards"]["2026-2026-10-24"]
    f = next(f for f in c["fights"] if f["cols"]["pick"] == "Jacob Malkoun")
    assert f["bet"] is True and (f["pick_mark"], f["pick_mark_kind"]) == ("✗", "miss")
    assert f["cols"]["open"] == "+132"
    assert sum(x["bet"] for x in c["fights"]) == len(c["bets"]) == 1


def test_record_coin_flips_are_marked_not_counted(record):
    c = record["cards"]["2026-2026-10-24"]
    flips = [f for f in c["fights"] if f["cols"]["note"] == "Coin flip"]
    assert flips and all(f["pick_mark"] == "–" for f in flips)
    assert c["total"] == sum(f["pick_mark"] in ("✓", "✗") for f in c["fights"])


def test_no_public_ledger_file(record):
    assert not (record["out"] / "record" / "ledger.csv").exists()
