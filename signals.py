"""One signal path: every decision a signal needs, decided in one place.

card_report.py (and card_settle.py through it), betting_system.py (the
ledger) and site/build.py all call classify(); none of them carries a
threshold, a tie-break, a debut test or a stake rule of its own.
tests/test_signals.py pins the rules; tests/test_signal_path.py runs the
report path and the ledger path over one card and asserts the same signal,
label and stake on every bout. That test is the permanent guard.

Constants, with their freeze dates:
  FLIP_CONF, GAP_PTS, FLIP_UNITS, GAP_UNITS   signal rules, frozen 2026-09-08
  OVERROUND_LO, OVERROUND_HI                   odds-sanity pre-filter, frozen 2026-09-08
  SKIP_LINE_MAX, UNKNOWN_RD, RAMP_LO, RAMP_HI  placement policy v1, frozen 2026-09-09
  VOID_SUSPECT                                 opener validity, 2026-09-30 (final 2026-10-01)

Rulings 2026-10-22:
  Debut rule: a bout is projectable only when BOTH fighters are on file with a
  UFC bout dated before the event. Otherwise projectable=False, reason
  "debut": no projection, no signal, no ledger row, no pick. Decided from the
  data (fights_v2), never from a per-bout list.
  Pick'em tie-break: at equal opens the market has no favourite, so the pick
  cannot flip it; the GAP test applies to the pick's side. Before this the
  label at a tie depended on which fighter was listed first (the report said
  FLIP where the ledger said GAP for Price / Shahbazyan 2026-10-10).

classify() takes a bout's pre-fight state (the model's P(fighter1), the
validated opening pair, the raw first ticks and open_suspect, both pre-fight
RDs, whether each fighter is on record) plus the price the bet would settle
at, and returns a Decision: projectable, reason, the signal (rule and side)
and the policy-v1 stake.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- signal rules, frozen 2026-09-08 (BETTING-SYSTEM.md) -----------------
FLIP_CONF = 0.65          # FLIP: the pick is the market underdog and P(pick) >= this
GAP_PTS = 100             # GAP: the pick is the market favourite and fair - open >= this
FLIP_UNITS = 1            # flat 1u; set 2 for the higher-variance variant
GAP_UNITS = 1
OVERROUND_LO, OVERROUND_HI = 1.00, 1.12     # pre-filter: 1/o1 + 1/o2 must lie here

# ---- placement policy v1, frozen 2026-09-09 ------------------------------
# A layer on top of the signal rules: every signal is still logged at flat 1u;
# the policy decides which are placed and at what stake, from the pre-fight
# rating deviations (RD, data/prefight_rd.csv) of the backed fighter and the
# opponent and from the backed side's American price (the price of record:
# placeable where captured, else the current line for a future card, else open).
SKIP_LINE_MAX = 250        # bet side +250 or longer -> not placed
UNKNOWN_RD = 160           # both fighters RD > this -> not placed
RAMP_LO, RAMP_HI = 130, 200    # 1u up to a backed-side RD of RAMP_LO, 0.5u at RAMP_HI

# ---- opener validity, 2026-09-30 (final 2026-10-01) ----------------------
VOID_SUSPECT = "void: suspect opener"   # fires only against the raw tick of an open_suspect bout

REASON_DEBUT = "debut"
REASON_NO_PROJECTION = "no projection"


def american(dec: float) -> float:
    """Decimal -> American (unrounded)."""
    return (dec - 1) * 100 if dec >= 2 else -100 / (dec - 1)


def fair_american(q: float) -> float:
    """Probability -> fair American line (unrounded)."""
    return -100 * q / (1 - q) if q >= .5 else 100 * (1 - q) / q


def stake_units(rd_self, rd_opp, line) -> float:
    """Placement policy v1: stake in units for one signal (0.0 = not placed).
    Full 1u up to a backed-side RD of RAMP_LO, falling linearly to 0.5u at
    RAMP_HI and floored there; 0 when the bet side is longer than
    SKIP_LINE_MAX or both fighters carry RD > UNKNOWN_RD."""
    if line > SKIP_LINE_MAX:
        return 0.0
    if rd_self > UNKNOWN_RD and rd_opp > UNKNOWN_RD:
        return 0.0
    return float(min(1.0, max(0.5, 1 - (rd_self - RAMP_LO) / (RAMP_HI - RAMP_LO) * 0.5)))


def skip_reason(rd_self, rd_opp, line) -> str:
    """Why policy v1 does not place a signal ("" when it does)."""
    if line > SKIP_LINE_MAX:
        return f"skip: {line:+d}"
    if rd_self > UNKNOWN_RD and rd_opp > UNKNOWN_RD:
        return "skip: both unknown"
    return ""


def market_favourite(o1: float, o2: float):
    """1 or 2: the side with the shorter decimal price; None at a pick'em
    (equal prices: the market has no favourite)."""
    if o1 == o2:
        return None
    return 1 if o1 < o2 else 2


def rule_at(p1: float, o1: float, o2: float):
    """The frozen rule at one opening pair -> (rule, side, conf, gap_pts) or
    None. side is 1 or 2 (the model's pick), conf = P(pick), gap_pts = the
    pick's opening American line minus its fair line (positive when the market
    pays more than the model's fair price). Pre-filter first; FLIP when the
    pick is the market underdog and conf >= FLIP_CONF; GAP when the pick is
    the market favourite, or the opens are equal (no favourite to flip), and
    gap_pts >= GAP_PTS."""
    if not OVERROUND_LO <= 1 / o1 + 1 / o2 <= OVERROUND_HI:
        return None
    side = 1 if p1 >= .5 else 2
    conf = p1 if side == 1 else 1 - p1
    o_pick = o1 if side == 1 else o2
    gap = american(o_pick) - fair_american(conf)
    fav = market_favourite(o1, o2)
    if fav is not None and fav != side:
        return ("FLIP", side, conf, gap) if conf >= FLIP_CONF else None
    return ("GAP", side, conf, gap) if gap >= GAP_PTS else None


@dataclass(frozen=True)
class Decision:
    """What classify() says about one bout.
    projectable  both fighters on record before the event and a projection given
    reason       "" | "debut" | "no projection" | VOID_SUSPECT
    rule         "FLIP" | "GAP" | None (no signal)
    side         1 | 2: the picked fighter (fighter1 / fighter2), None without a signal
    conf         P(pick); gap_pts the pick's open minus fair, in American points
    open_line    the pick's opening American line (the raw tick for a VOID row)
    line         the price of record the stake is sized at (American, rounded)
    stake        policy-v1 units (0.0 when not placed or VOID); placed = stake > 0
    skip         "" | "skip: +NNN" | "skip: both unknown" for an unplaced signal
    void         True for a VOID row: logged, never settled, placed or counted"""
    projectable: bool
    reason: str = ""
    rule: str | None = None
    side: int | None = None
    conf: float | None = None
    gap_pts: float | None = None
    open_line: int | None = None
    line: int | None = None
    stake: float = 0.0
    skip: str = ""
    void: bool = False

    @property
    def signal(self):
        """(rule, side) or None: what the identity test compares."""
        return (self.rule, self.side) if self.rule else None

    @property
    def placed(self) -> bool:
        return self.stake > 0


NO_SIGNAL = Decision(projectable=True)


def classify(p1, o1, o2, rd1, rd2, *, known1=True, known2=True,
             o1_raw=None, o2_raw=None, open_suspect=False,
             dec1=None, dec2=None) -> Decision:
    """The one signal decision for a bout.

    p1            model P(fighter1), None when no projection exists
    o1, o2        validated opening decimals (scrape_bfo.validated_open)
    rd1, rd2      pre-fight RDs (public scale) of fighter1 / fighter2
    known1/2      fighter on record with a UFC bout before the event (debut rule)
    o1_raw/o2_raw raw first ticks and open_suspect from the same series: a
                  signal that fires only there is a VOID row (opener validity)
    dec1, dec2    the price of record per side (decimal) the stake is sized at
                  (placeable capture / current line); the open when None
    """
    if not (known1 and known2):
        return Decision(projectable=False, reason=REASON_DEBUT)
    if p1 is None:
        return Decision(projectable=False, reason=REASON_NO_PROJECTION)
    hit = rule_at(p1, o1, o2)
    if hit:
        rule, side, conf, gap = hit
        opn = o1 if side == 1 else o2
        dec = (dec1 if side == 1 else dec2)
        dec = opn if dec is None else dec
        line = int(round(american(dec)))
        rd_self, rd_opp = (rd1, rd2) if side == 1 else (rd2, rd1)
        stake = stake_units(rd_self, rd_opp, line)
        return Decision(projectable=True, reason="", rule=rule, side=side, conf=conf,
                        gap_pts=gap, open_line=int(round(american(opn))), line=line,
                        stake=round(stake, 2), skip=skip_reason(rd_self, rd_opp, line),
                        void=False)
    if open_suspect and o1_raw is not None and o2_raw is not None:
        raw = rule_at(p1, o1_raw, o2_raw)
        if raw:
            rule, side, conf, gap = raw
            opn = int(round(american(o1_raw if side == 1 else o2_raw)))
            return Decision(projectable=True, reason=VOID_SUSPECT, rule=rule, side=side,
                            conf=conf, gap_pts=gap, open_line=opn, line=opn, stake=0.0,
                            skip="", void=True)
    return NO_SIGNAL


def first_bout_dates(fights):
    """fighter_id -> ISO date of the fighter's first UFC bout on file, from a
    fights_v2 DataFrame (event_date, fighter_a_id, fighter_b_id). The debut
    rule's data: known(fid, event_date) = fid in this map and its date is
    before event_date."""
    import pandas as pd
    fv = fights[["event_date", "fighter_a_id", "fighter_b_id"]]
    long = pd.concat([fv[["event_date", "fighter_a_id"]].rename(columns={"fighter_a_id": "fid"}),
                      fv[["event_date", "fighter_b_id"]].rename(columns={"fighter_b_id": "fid"})])
    return long.groupby("fid").event_date.min().to_dict()


def on_record(fid, event_date: str, first: dict) -> bool:
    """The debut rule for one fighter: on file with a UFC bout before event_date."""
    d = first.get(fid)
    return d is not None and d < event_date
