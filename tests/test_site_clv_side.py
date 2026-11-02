"""The every-fight CLV column on Our Record is card_settle's number, measured
on our side (the fighter we price higher than the opening odds), so a card's
CLV is the mean of its column: pinned on UFC Vegas 123 (2026-10-31, -0.54
over 10) and UFC 333 (2026-10-24, +0.54 over 13), the numbers card_settle
printed. Reads the repo's own data; no network."""
import importlib.util
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_build", ROOT / "site" / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

CARDS = {"2026-2026-10-31": (10, "-0.54"), "2026-2026-10-24": (13, "+0.54")}


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    data = build.Data(date(2026, 11, 2))
    return build.build_record(data, build.Model(), tmp_path_factory.mktemp("out"))


def test_our_side_clv_is_the_move_toward_the_side_we_price_higher():
    # a opens 2.60 / b 1.54 (vig-free a .372); the close moves to a
    assert build.our_side_clv(0.45, 2.60, 1.54, 2.44, 1.61) == pytest.approx(
        build.vigfree(2.44, 1.61)[0] - build.vigfree(2.60, 1.54)[0])
    # we price b higher: the same move is against us, so the sign flips
    assert build.our_side_clv(0.30, 2.60, 1.54, 2.44, 1.61) == pytest.approx(
        build.vigfree(2.60, 1.54)[0] - build.vigfree(2.44, 1.61)[0])


@pytest.mark.parametrize("key", sorted(CARDS))
def test_card_clv_is_the_mean_of_its_column(record, key):
    n, printed = CARDS[key]
    c = record["cards"][key]
    vals = [f["clv_val"] for f in c["fights"] if f["clv_val"] is not None]
    assert len(vals) == c["clv_n"] == n
    assert sum(vals) / len(vals) == pytest.approx(c["clv_val"], abs=1e-12)
    assert c["clv"] == printed                       # what card_settle printed
    for f in c["fights"]:                            # the column shows the same numbers
        assert f["num"]["clv"] == (f"{f['clv_val'] * 100:+.1f}" if f["clv_val"] is not None else "—")


def test_a_ledger_row_shows_the_same_clv_in_both_tables(record):
    # Malkoun FLIP on UFC 333: the ledger's bet-side CLV is our side at the
    # same clean close, so the bets table and the every-fight table agree
    c = record["cards"]["2026-2026-10-24"]
    bet = next(b for b in c["bets"] if b["pick"].startswith("Jacob Malkoun"))
    fight = next(f for f in c["fights"] if "Jacob Malkoun" in (f["f1"], f["f2"]))
    assert bet["num"]["clv"] == fight["num"]["clv"] != "—"
