"""cadence.py: the automated weekly loop. Offline tests of the pieces that
decide what gets captured and written: which BFO bouts are on the card (UFC
Stats is the authority), split BFO cards read as one, the --off flag, the
incremental scrape (append-only, refuses a card without results) and the
line-ending guard."""
import csv
import subprocess

import pytest

import cadence
import card_report
import scrape_v2 as sv

CARD = [  # UFC Vegas 124 as UFC Stats listed it on 2026-11-05 (three of its bouts)
    dict(fight_id="f1", fighter_a_id="a1", fighter_a="Waldo Cortes Acosta",
         fighter_b_id="b1", fighter_b="Johnny Walker"),
    dict(fight_id="f2", fighter_a_id="a2", fighter_a="Jonny Parsons",
         fighter_b_id="b2", fighter_b="Jose Souza"),
    dict(fight_id="f3", fighter_a_id="a3", fighter_a="Seokhyeon Ko",
         fighter_b_id="b3", fighter_b="Wellington Turman"),
]


def test_match_card_keeps_card_bouts_and_drops_phantoms_and_duplicates():
    bfo = [dict(mu=45564, f1="Johnny Walker", f2="Waldo Cortes-Acosta"),   # hyphen, swapped
           dict(mu=45556, f1="Jonny Parsons", f2="Jose Souza"),
           dict(mu=45561, f1="Jonny Parsons", f2="Jose Henrique"),          # BFO's phantom
           dict(mu=45599, f1="Seokhyeon Ko", f2="Wellington Turman"),
           dict(mu=45600, f1="Wellington Turman", f2="Seokhyeon Ko")]       # listed twice
    on, off, matched = cadence.match_card(bfo, CARD)
    assert [b["mu"] for b in on] == [45564, 45556, 45599]
    assert [(b["mu"], b["why"]) for b in off] == [(45561, "not on the UFC Stats card"),
                                                  (45600, "duplicate listing")]
    assert matched == {"f1", "f2", "f3"}


def test_match_card_accepts_a_spelling_near_miss_of_the_actual_opponent():
    bfo = [dict(mu=1, f1="Seokhyeon Ko", f2="Welington Turman")]
    on, off, matched = cadence.match_card(bfo, CARD)
    assert [b["mu"] for b in on] == [1] and not off and matched == {"f3"}


def test_card_bouts_reads_split_pages_as_one_card():
    pages = {"p-4439": [dict(mu=1, f1="a", f2="b"), dict(mu=2, f1="c", f2="d")],
             "p-4437": [dict(mu=2, f1="c", f2="d"), dict(mu=3, f1="e", f2="f")]}
    got = cadence.card_bouts(["p-4439", "p-4437"], pages.__getitem__)
    assert [(b["mu"], b["slug"]) for b in got] == [(1, "p-4439"), (2, "p-4439"), (3, "p-4437")]


def test_parse_off():
    assert card_report.parse_off(["s", "2026-11-07", "--off", "1,2"]) == (["s", "2026-11-07"], {1, 2})
    assert card_report.parse_off(["--off=3", "s", "d"]) == (["s", "d"], {3})
    assert card_report.parse_off(["--backfill", "2026"]) == (["--backfill", "2026"], set())


def _write(path, cols, rows, eol="\n"):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator=eol)
        w.writeheader()
        w.writerows(rows)


def _fight(fid, ev, date, a, b, outcome="A_WIN"):
    return {c: "" for c in sv.FIGHT_COLS} | dict(
        fight_id=fid, event_id=ev, event_date=date, event_name=ev, bout_order=1,
        fighter_a_id=a, fighter_a=a, fighter_b_id=b, fighter_b=b, outcome=outcome,
        a_mw="0", b_mw="0")


@pytest.fixture
def scrape_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "OUT", str(tmp_path))
    _write(tmp_path / "fights_v2.csv", sv.FIGHT_COLS,
           [_fight("x1", "e0", "2026-10-31", "a", "b") | dict(a_mw="1")])
    _write(tmp_path / "fighters_v2.csv", sv.FIGHTER_COLS,
           [dict(fighter_id="a", name="A"), dict(fighter_id="b", name="B")], eol="\r\n")
    monkeypatch.setattr(sv, "main_c", lambda: None)
    monkeypatch.setattr(sv, "scrape_fighter", lambda fid: dict(fighter_id=fid, name=fid.upper()))
    monkeypatch.setattr(sv, "scrape_events", lambda fresh=False: [
        dict(event_id="e0", event_name="e0", event_date="2026-10-31"),
        dict(event_id="e1", event_name="e1", event_date="2026-11-07"),
        dict(event_id="e2", event_name="e2", event_date="2026-11-14")])
    return tmp_path


def test_main_new_appends_only_new_cards_and_fighters(scrape_dir, monkeypatch):
    before = (scrape_dir / "fights_v2.csv").read_text(encoding="utf-8")
    calls = []

    def page(ev, cache=None):
        calls.append((ev["event_id"], cache))
        return [_fight("x2", "e1", "2026-11-07", "c", "a")]
    monkeypatch.setattr(sv, "scrape_event_fights", page)
    new = sv.main_new(today="2026-11-08")                     # e2 is not fought yet
    assert [e["event_id"] for e in new] == ["e1"] and calls == [("e1", False)]
    _, rows = sv.read_rows(scrape_dir / "fights_v2.csv")
    assert [r["fight_id"] for r in rows] == ["x1", "x2"]
    assert rows[0]["a_mw"] == "1" and rows[1]["a_mw"] == "0"  # carried / default
    after = (scrape_dir / "fights_v2.csv").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert after.startswith(before)                           # rows on file untouched
    _, fighters = sv.read_rows(scrape_dir / "fighters_v2.csv")
    assert [r["fighter_id"] for r in fighters] == ["a", "b", "c"]
    assert sv.main_new(today="2026-11-08") == []              # nothing new: no-op


def test_main_new_refuses_a_card_without_results(scrape_dir, monkeypatch):
    before = (scrape_dir / "fights_v2.csv").read_bytes()
    monkeypatch.setattr(sv, "scrape_event_fights", lambda ev, cache=None: [
        _fight("x2", "e1", "2026-11-07", "c", "a"),
        _fight("x3", "e1", "2026-11-07", "d", "b", outcome="")])
    with pytest.raises(sv.ResultsPending):
        sv.main_new(today="2026-11-08")
    assert (scrape_dir / "fights_v2.csv").read_bytes() == before


def test_restore_eol_matches_the_committed_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    git = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True)
    git("init", "-q")
    git("config", "core.autocrlf", "false")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "lf.csv").write_bytes(b"a,b\n1,2\n")
    (tmp_path / "data" / "crlf.csv").write_bytes(b"a,b\r\n1,2\r\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    (tmp_path / "data" / "lf.csv").write_bytes(b"a,b\r\n1,2\r\n3,4\r\n")    # csv module on Linux
    (tmp_path / "data" / "crlf.csv").write_bytes(b"a,b\n1,2\n3,4\n")        # pandas on Linux
    assert sorted(cadence.restore_eol(("data",))) == ["data/crlf.csv", "data/lf.csv"]
    assert (tmp_path / "data" / "lf.csv").read_bytes() == b"a,b\n1,2\n3,4\n"
    assert (tmp_path / "data" / "crlf.csv").read_bytes() == b"a,b\r\n1,2\r\n3,4\r\n"


def test_scrape_bfo_writes_only_fought_cards(tmp_path):
    import scrape_bfo
    _write(tmp_path / "f.csv", sv.FIGHT_COLS, [_fight("x1", "e0", "2026-11-07", "a", "b")])
    # a Saturday card's page dated the Sunday is in; next weekend's is out
    assert scrape_bfo.fought_through(str(tmp_path / "f.csv")) == "2026-11-08"
    assert scrape_bfo.fought_through(str(tmp_path / "missing.csv")) is None
