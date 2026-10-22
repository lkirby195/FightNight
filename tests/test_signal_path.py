"""The permanent guard: the report path (card_report.signals, what
card_report.py and card_settle.py print) and the ledger path
(betting_system.run, what --write-ledger writes) over the SAME card give the
same signal, label and stake on every bout -- and the ledger path gives it
whichever fighter UFC Stats lists first. Both paths call signals.classify;
this test pins the wiring around it (orientation, price of record, debut
exclusion, VOID, skips)."""
import pandas as pd
import pytest

import betting_system as bs
import card_report as cr

PK = 1.90909
# One card: every kind of decision. p1 = model P(f1); o = validated opens;
# raw/suspect = opener validity; cap = earliest capture (price of record).
CARD = [
    dict(mu=1, f1="Niko Price", f2="Leon Shahbazyan", p1=0.814, o1=PK, o2=PK,
         rd1=95.0, rd2=290.0),                                  # pick'em: GAP, 1.00u
    dict(mu=2, f1="Jacob Malkoun", f2="Abubakar Vagaev", p1=0.71, o1=2.32, o2=1.645,
         rd1=141.0, rd2=281.0, cap=(2.28, 1.654)),              # FLIP at +132, record +128, 0.92u
    dict(mu=3, f1="Raul Rosas Jr", f2="Raoni Barcelos", p1=0.78, o1=1.4878, o2=2.73,
         rd1=154.0, rd2=114.0),                                 # GAP -205, 0.83u
    dict(mu=4, f1="Yazmin Jauregui", f2="Vanessa Demopoulos", p1=0.75, o1=1.1176, o2=6.96,
         rd1=181.0, rd2=126.0, raw=(6.96, 1.1176), suspect=True),   # fires only raw: VOID
    dict(mu=5, f1="Steven Koslow", f2="Chad Anheliger", p1=0.66, o1=2.09, o2=1.775,
         rd1=291.0, rd2=175.0),                                 # FLIP, skip: both unknown
    dict(mu=6, f1="Tai Tuivasa", f2="Robelis Despaigne", p1=0.70, o1=6.04, o2=1.138,
         rd1=210.0, rd2=103.0),                                 # FLIP, skip: +504
    dict(mu=7, f1="Khaos Williams", f2="Roberto Soldic", p1=0.713, o1=2.74, o2=1.5,
         rd1=130.0, rd2=350.0, known2=False),                   # debut: nothing anywhere
    dict(mu=8, f1="Alexander Volkanovski", f2="Movsar Evloev", p1=0.41, o1=1.833, o2=1.909,
         rd1=100.0, rd2=100.0),                                 # no signal
]
EXPECT = {1: ("GAP", "Niko Price", 1.0, False), 2: ("FLIP", "Jacob Malkoun", 0.92, False),
          3: ("GAP", "Raul Rosas Jr", 0.83, False), 4: ("FLIP", "Yazmin Jauregui", 0.0, True),
          5: ("FLIP", "Steven Koslow", 0.0, False), 6: ("FLIP", "Tai Tuivasa", 0.0, False)}


def known(b, side):
    return b.get(f"known{side}", True)


def report_path(monkeypatch):
    """card_report.signals over CARD as card_report.main feeds it: the bouts
    as get_card returns them, preds / rds for the bouts whose fighters both
    resolve (a debutant does not), the earliest capture as the price of record."""
    bouts = [dict(mu=b["mu"], f1=b["f1"], f2=b["f2"], f1_open=b["o1"], f2_open=b["o2"],
                  f1_close=b["o1"], f2_close=b["o2"],
                  open_suspect=int(b.get("suspect", False)),
                  f1_open_raw=b.get("raw", (b["o1"], b["o2"]))[0],
                  f2_open_raw=b.get("raw", (b["o1"], b["o2"]))[1]) for b in CARD]
    preds = {b["mu"]: b["p1"] for b in CARD if known(b, 1) and known(b, 2)}
    rds = {b["mu"]: (b["rd1"], b["rd2"]) for b in CARD}
    monkeypatch.setattr(cr, "earliest_capture",
                        lambda: {b["mu"]: b["cap"] for b in CARD if "cap" in b})
    sig = cr.signals(bouts, preds, rds, future=False)
    return {mu: (s["rule"], s["who"], round(s["stake"], 2), s["void"]) for mu, s in sig.items()}


def ledger_path(monkeypatch, swap: bool):
    """betting_system.run over CARD as load() would frame it, fighter_a = f1
    (swap=False) or fighter_a = f2 (swap=True: UFC Stats lists the winner
    first, so the ledger sees either orientation)."""
    rows, caps = [], {}
    for b in CARD:
        o1, o2 = b["o1"], b["o2"]
        r1, r2 = b.get("raw", (o1, o2))
        a, bb = (b["f2"], b["f1"]) if swap else (b["f1"], b["f2"])
        row = dict(fight_id=f"f{b['mu']}", event_date="2026-10-24", event_name="E",
                   bout_order=b["mu"], fighter_a_id=a, fighter_a=a, fighter_b_id=bb,
                   fighter_b=bb, method="U-DEC", rnd=3, year=2026, y_a=1,
                   p_model_a=(1 - b["p1"]) if swap else b["p1"],
                   a_open=o2 if swap else o1, b_open=o1 if swap else o2,
                   a_open_raw=r2 if swap else r1, b_open_raw=r1 if swap else r2,
                   open_suspect=int(b.get("suspect", False)),
                   rd_a=b["rd2"] if swap else b["rd1"], rd_b=b["rd1"] if swap else b["rd2"],
                   known_a=known(b, 2) if swap else known(b, 1),
                   known_b=known(b, 1) if swap else known(b, 2))
        row["projectable"] = row["known_a"] and row["known_b"]
        rows.append(row)
        if "cap" in b:
            caps[row["fight_id"]] = tuple(reversed(b["cap"])) if swap else b["cap"]
    D = pd.DataFrame(rows)

    def capture_prices(D):
        da = pd.Series([caps.get(f, (float("nan"),))[0] for f in D.fight_id], index=D.index, dtype=float)
        db = pd.Series([caps.get(f, (0, float("nan")))[1] for f in D.fight_id], index=D.index, dtype=float)
        return da, db, pd.Series(float("nan"), index=D.index, dtype=float)
    monkeypatch.setattr(bs, "capture_prices", capture_prices)
    B = bs.run(D)
    out = {}
    for _, r in B.iterrows():
        mu = int(r.fight_id[1:])
        out[mu] = (r.rule, r.fighter_a if r.mod1 else r.fighter_b, round(float(r.stake_u), 2),
                   bool(r.void))
    return out


def test_report_and_ledger_agree_on_every_bout_either_orientation(monkeypatch):
    rep = report_path(monkeypatch)
    led = ledger_path(monkeypatch, swap=False)
    led_swapped = ledger_path(monkeypatch, swap=True)
    assert rep == EXPECT
    assert led == rep
    assert led_swapped == rep


def test_the_pick_em_label_no_longer_depends_on_who_is_listed_first(monkeypatch):
    """Price / Shahbazyan 2026-10-10 before this: the report (Price listed
    second on BFO) said FLIP, the ledger (Price first on UFC Stats) said GAP.
    One label now, GAP, on both paths and both orientations."""
    rep = report_path(monkeypatch)
    assert rep[1] == ("GAP", "Niko Price", 1.0, False)
    assert ledger_path(monkeypatch, swap=False)[1] == rep[1]
    assert ledger_path(monkeypatch, swap=True)[1] == rep[1]


def test_a_debut_bout_is_absent_from_both_paths(monkeypatch):
    assert 7 not in report_path(monkeypatch)
    assert 7 not in ledger_path(monkeypatch, swap=False)
    assert 7 not in ledger_path(monkeypatch, swap=True)


def test_the_price_of_record_sizes_the_stake_on_both_paths(monkeypatch):
    # Malkoun: open +132 (RD 141 -> 0.92u either way here), but the line both
    # paths size at is the capture +128, which the ledger writes as bet_line
    rep = report_path(monkeypatch)
    assert rep[2][2] == pytest.approx(0.92)
    led = ledger_path(monkeypatch, swap=True)
    assert led[2] == rep[2]
