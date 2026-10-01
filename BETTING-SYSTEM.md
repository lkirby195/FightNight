# FightNight betting system (KISS rule)

Two rules, applied to the BestFightOdds **opening** line using model-only
walk-forward probabilities (`data/model_only_preds.csv`, no market input).
The code in `betting_system.py` is the source of truth; this page describes it.

## Betting rule (frozen 2026-09-08)

```
GAP-FAV : model agrees on the market favorite AND prices them >= 100 American
          points stronger than the opening line. 1u flat at open.
CONF-FLIP: model picks the market underdog AND model prob >= 0.65. 1u flat at open.
```

Frozen at 0.65 / 1u, which is the rule every live bet since 2026-01 was generated
under and the rule behind the "validated 2012-2026, t=2.43" claim. An earlier
draft of this spec said 0.60 / 2u; that never matched the code. Historical sim of
0.60 / 2u: 867 bets, 435-432, +35.6u, t=0.87 (the 0.60-0.65 band goes 132-211 and
is doubled in stake). Compared after the fact, so treat this freeze as the
pre-registration point: the live record from 2026-09-08 forward is the test.

Constants in `betting_system.py`: `FLIP_CONF = 0.65`, `FLIP_UNITS = 1`,
`GAP_PTS = 100`, `GAP_UNITS = 1`.

## Placement policy v1 (frozen 2026-09-09)

The signal rules above decide what is logged; the placement policy decides
which signals are placed and at what stake. Every signal is still recorded at
flat 1u (`units`, `pnl`, `pnl_at_open`); the policy adds `placed` and
`stake_u`, and `pnl_placed` = `stake_u` x `pnl` per unit. Inputs are the
pre-fight rating deviation (RD) of both fighters from `data/prefight_rd.csv`
(written by `prefight_rd.py`: the time-inflated RD on the public scale,
exactly what `card_report.build_states` sees before the bout) and the backed
side's American price (`placeable_line` where captured, else `open_line`).

```
1. Price cap : backed side +250 or longer -> not placed        SKIP_LINE_MAX = 250
2. Unknowns  : both fighters RD > 160 -> not placed            UNKNOWN_RD = 160
3. RD ramp   : 1u up to a backed-side RD of 130, falling       RAMP_LO, RAMP_HI = 130, 200
               linearly to 0.5u at RD 200, floored at 0.5u
```

Rationale: edge concentrates where the backed fighter is well-measured, and
signals where both fighters are unknown have netted to zero since 2018.
Kelly / edge-proportional sizing was tested and rejected: the model
probability is overconfident by 15-20 points on the selected bets, so sizing
on it would put the most money on the least-evidenced picks.

Signal rules (frozen 2026-09-08) and placement policy (frozen 2026-09-09)
have separate freeze dates, and both records are reported: the flat signal
record (`LIVE RECORD` / `FULL SIM`) and the placed record (the
`PLACED (policy v1)` block printed under each, settled the same way as the
block above it). The policy is a pure function of existing ledger columns, so
it is applied retroactively to every row; the placed figures before
2026-09-09 are in-sample, like the rule's own sim. `card_report.py` prints the
stake next to each rule-fired pick (`1.00u`, `0.72u`, or `skip: both unknown`
/ `skip: +275`), using the same `stake_units` function.

Placed record at the freeze (`python betting_system.py all`, data through
2026-09-05):

```
LIVE RECORD (all): 26 bets  19-7  ...  P&L (placeable) +9.3u
PLACED (policy v1): 16 placed of 26 signals  13-3  staked 12.5u  P&L +6.2u ($+624)  ROI +50.0%  t=2.32
    FLIP: 6 placed  4-2  ROI +60.9%
    GAP: 10 placed  9-1  ROI +43.2%

FULL SIM (all): 533 bets  307-226  ...  P&L +57.1u
PLACED (policy v1): 218 placed of 533 signals  147-71  staked 185.0u  P&L +56.0u ($+5,601)  ROI +30.3%  t=4.42
    FLIP: 110 placed  67-43  ROI +43.4%
    GAP: 108 placed  80-28  ROI +16.6%
```

Of the 503 signals from 2012-2024, 200 are placed: 133-67, staked 170.8u,
+49.9u (the flat signal record over the same rows is unchanged at +49.7u).
Of the 30 signals of 2026, 18 are placed: 14-4, staked 14.3u, +6.1u. Skips:
305 signals with both fighters unknown, 10 at +250 or longer.

## Inputs, pre-filter, settlement

- Inputs: `data/model_only_preds.csv` (walk-forward model, trained strictly on
  prior years), `data/bfo_joined.csv` (BFO open/close per fight), `data/fights_v2.csv`.
- Pre-filter: skip bouts whose opening implied-probability sum is outside
  [1.00, 1.12] (odds sanity). The opening pair is the validated open (see
  "Opener validity").
- One bet per qualifying bout, on the model's pick. The sim settles at that
  side's opening decimal price; live bets with a captured placeable line settle
  there instead (see "Placeable vs open"). P&L is in units; 1u = $100 in the
  printed summary.
- 2025 is a sealed holdout: there are no 2025 model predictions, so the sim
  covers 2012-2024 and 2026.

## 2025 backtest (sealed one-shot, for display only)

`data/system_ledger_2025_backtest.csv` and
`data/model_only_preds_2025_backtest.csv` were written ONCE on 2026-09-09 by
`site/backtest_2025.py`: the walk-forward Stage 2 fit on fights before
2025-01-01 (the `clv_eval.py` recipe), the frozen rule above (2026-09-08) and
placement policy v1 (2026-09-09) over the 2025 cards, settled at the opening
line (`live=0`, no placeable line, no money down). They feed the public site's
2025 tab and nothing else: a sealed read for display, not an input to model
selection, and never regenerated (the script refuses to run if the files
exist). `data/model_only_preds.csv` and `data/system_ledger.csv` are untouched.
At the seal: 34 signals, 22-12, +8.3u flat; placed 27, 18-9, staked 21.9u,
+7.1u.

## Historical sim at the frozen rule

`python betting_system.py all` (data through 2026-09-05):

```
LIVE RECORD (all): 26 bets  19-7  staked 26u  P&L +9.3u ($+930)  ROI +35.8%  t=2.05
    FLIP: 13 bets  7-6  ROI +19.6%
    GAP: 13 bets  12-1  ROI +51.9%

FULL SIM (all): 533 bets  307-226  staked 533u  P&L +57.1u ($+5,710)  ROI +10.7%  t=2.42
    FLIP: 281 bets  132-149  ROI +12.7%
    GAP: 252 bets  175-77  ROI +8.5%
```

`python betting_system.py 2026`:

```
LIVE RECORD (2026): 26 bets  19-7  staked 26u  P&L +9.3u ($+930)  ROI +35.8%  t=2.05
    FLIP: 13 bets  7-6  ROI +19.6%
    GAP: 13 bets  12-1  ROI +51.9%

FULL SIM (2026): 30 bets  20-10  staked 30u  P&L +7.4u ($+741)  ROI +24.7%  t=1.44
    FLIP: 17 bets  8-9  ROI +3.9%
    GAP: 13 bets  12-1  ROI +51.9%
```

The freeze-day figure (data through 2026-07-25) was 524 bets, 303-221, +56.7u,
t=2.43. Regenerating `model_only_preds.csv` on 2026-09-08 moved historical
probabilities by at most 0.001 and pulled in two bets sitting exactly on the
0.65 threshold (Cerrone-Stephens 2012, Jackson-Soukhamthath 2019); the other
seven additions are the Aug-Sep 2026 cards.

**Source of record for the historical figures:**
`data/model_only_preds_frozen_2026-09-08.csv`, a byte-for-byte copy of
`data/model_only_preds.csv` as of 2026-09-08 (walk-forward probabilities
2012-2024 and 2026 through 2026-09-05; 2025 is sealed and has no rows). The
2012-2025 rows of the full sim above are quoted from that snapshot only.
`clv_eval.py` regenerates the live `data/model_only_preds.csv` on every run, so
a later data refresh can move historical probabilities again; diff the live
file against the snapshot and do not re-quote the historical figures from a
refreshed file.

Post-snapshot change, 2026-09-08: the hyphenated-name fix in `norm()`
(`clv_eval.py`, `card_report.py`; BFO "Cortes-Acosta" now meets UFC Stats
"Cortes Acosta") added 27 joined fights to `data/bfo_joined.csv`, none of which
triggers a bet, so the figures above stand and the live preds file still
matches the snapshot byte for byte.

t is the one-sample t-statistic of per-bet return on stake. The rule was chosen
on this same history, so the in-sample t overstates the evidence; the live
record from 2026-09-08 forward is the out-of-sample test.

## Settling a live card

Use `card_settle.py <bfo-slug> <YYYY-MM-DD>` (not `card_report.py`) for any event
less than ~4 months old: BFO keeps in-play ticks in the line history, and the
paired-tick overround filter in `card_settle.py` recovers the last pre-fight book.

## Ledger of record

`data/system_ledger.csv` is written by `python betting_system.py <year|all>
--write-ledger` and is never hand-edited: to change it, fix the inputs or the
code and regenerate. Columns:

```
event_date, event, fight_id, fighter, rule, live, units, rd_self, rd_opp,
placed, stake_u, pnl_placed, open_line, placeable_line, clean_close, clv_pts,
result, method, round, pnl, pnl_at_open, reason, pm_p_at_capture
```

`open_line` / `placeable_line` / `clean_close` are American odds for the side
bet. `placeable_line` is the price on the board when the pick went on record
(the earliest `data/placeable_lines.csv` capture; empty when none was
captured). `clean_close` is the last pre-fight paired tick (the
`card_settle.py` overround rule), so it is safe for cards still carrying
in-play ticks. `clv_pts` is the bet side's vig-free close minus open in
probability points. `pnl` is in units, settled at `placeable_line` where
present and at `open_line` otherwise; `pnl_at_open` is the open-line
settlement for every row. `rd_self` / `rd_opp` are the pre-fight RDs of the
backed fighter and the opponent, and `placed` / `stake_u` / `pnl_placed` the
placement-policy decision and its P&L (see "Placement policy v1"); `method` /
`round` are the bout's finish from `fights_v2.csv`. `reason` is empty on
every settled row and names the void reason on a VOID row (see "VOID rows");
`pm_p_at_capture` is the Polymarket price of the backed side at the earliest
capture, on placed rows whose capture recorded one (see "Polymarket").

The ledger only covers bouts present in `data/model_only_preds.csv` and
`data/bfo_joined.csv`; cards scraped after the last `clv_eval.py` run are not in
it until those files are regenerated.

`data/legacy_favtier_ledger.csv` is the abandoned favorites-tier system
(3u/2u/1u by price band), ending 2026-03-28 plus the Paris FLIP row. Kept for
the record; not the system described here.

## Placeable vs open

`card_report.py` often runs after lines have moved, so a P&L settled at
`open_line` is not a price that could have been bet. One ledger, two prices:

- When `card_report.py` is run for an event dated today or later, it
  re-fetches the BFO line histories (cache bypassed) and appends one row per
  bout to `data/placeable_lines.csv` (`event_date, bfo_slug, mu, fighter1,
  fighter2, f1_line, f2_line, captured_at`): the current mean line at report
  time, stamped with the run's timestamp. The file is append-only: rows
  already on file are never rewritten or replaced. A bout whose lines are
  identical to a row already on file (same `mu`, `f1_line`, `f2_line`) is
  skipped, so a re-run that finds nothing moved adds nothing, and every line
  movement seen at report time becomes a further row. Past events are never
  written.
- `placeable_line` in the ledger is the bet side's line from the EARLIEST
  captured row for that bout (smallest `captured_at`), i.e. the price at the
  time the pick went on record; later captures are informational only.
  live=0 rows and rows with no captured line leave it empty.
- `pnl` for live=1 rows with a placeable line is settled at `placeable_line`;
  everything else is settled at `open_line` as before. `pnl_at_open` keeps the
  open-line figure for every row. `clv_pts` is unchanged (open to clean close).
- `LIVE RECORD` prints both figures, `P&L (placeable)` and `P&L (open)`;
  `FULL SIM` is the pure sim at open.
- Backfill: the 26 live bets on record before 2026-09-08 have no captured
  line, so their `placeable_line` is empty and their `pnl` is unchanged. The
  first capture is Noche UFC Glendale (2026-09-12).

The rule constants are untouched (`FLIP_CONF`, `FLIP_UNITS`, `GAP_PTS`,
`GAP_UNITS`).

## Opener validity (2026-09-30, final 2026-10-01)

"The opening line" is BestFightOdds' first tick of the cross-book mean line.
Occasionally that first tick is a data-entry slip the book corrects within
minutes: Demopoulos / Jauregui (UFC Vegas 121, 2026-09-26) opened -850 / +596
at 22:28 UTC and was +596 / -850 eight minutes later, sides swapped. A signal
measured against a price nobody could bet is not a signal, so every consumer
of the series (`scrape_bfo.py`, `card_report.py`, `card_settle.py`,
`clv_eval.py`, `betting_system.py`) reads one validated open from
`scrape_bfo.validated_open`:

```
open = the bout's first paired tick (both sides' price at the first instant
       both have one),
UNLESS the next paired tick lands within 60 minutes of it and moves either
       side by more than 40 implied-probability points: then THAT pair is
       the open and the bout is open_suspect = 1.
```

The check is paired, not per side, because BFO moves both sides at the same
instants and a bettor sees a book, not a side; the open is always a pair
that was on the board together. An even-money-crossing test ("the next tick
crosses even money, or moves more than 40 points") was tried first and
rejected on 2026-10-01: it flagged 687 of 7,974 bouts (8.6%, 60 in 2026),
almost all of them BFO placeholder openers (-110 / -110, -526 / -526)
replaced by the real line minutes later and small moves that happen to
cross pick'em (Perez / Dumont -200 / +169 to -106 / -110, 15 points a
side); applied historically it had voided three placed live wins and added
two live signals that were never on record. The 40-point test keeps the
data-entry slips (side swaps such as Jauregui's 75-point move, -526 / -526
placeholders) and nothing else.

`data/bfo_lines.csv` carries `open_suspect` and the raw first ticks
(`f1_open_raw`, `f2_open_raw`) next to the validated opens; `bfo_joined.csv`
carries `open_suspect`, `a_open_raw`, `b_open_raw`. The pre-filter and both
rules evaluate at the validated open. `python scrape_bfo.py 2010 2027
--rebuild` rewrites every row from the cached tick series and lists every
flagged bout. Closing lines, CLV and the frozen rule constants are untouched;
the rule did not change, the price it is measured against did.

At the switch (7,974 bouts 2010-2026): 13 bouts flagged -- seven side
swaps corrected within minutes (Tuchscherer / Hunt 2011, Belcher /
Macdonald 2011, Cope / Brown 2012, Boetsch / Okami 2012, Oliveira / Swanson
2012, Madge / Edwards 2018, Almeida / Abdurakhimov 2022), five 2026
placeholder openers (-526 / -526 or -500 / +375 replaced by the real line:
Nascimento / Raposo, Filho / Rocha, Gantt / Ogden, Janicic / Gugnon,
Johnson / Ochoa) and Demopoulos / Jauregui 2026-09-26. No bout changed
pre-filter status. Three ledger rows became VOID: Okami 2012-02-25 and
Oliveira 2012-09-22 (backfilled) and Jauregui 2026-09-26 (live). No signal
was added or changed otherwise.

## VOID rows

A ledger row with `reason` set is VOID: it is logged so the record shows what
the rule would have fired on, and it is never settled, placed or counted.
`units`, `pnl`, `pnl_at_open`, `placed`, `stake_u` and `pnl_placed` are 0 and
`result` is `VOID`; no summary block counts it (each prints a `VOID (not
counted)` line instead). Reasons:

- `void: suspect opener` -- a signal that fires only against the raw first
  tick of an `open_suspect` bout and not at the validated open. `open_line`
  is the raw price it fired on. First instance: FLIP Yazmin Jauregui at the
  raw +596 (UFC Vegas 121); at the validated -850 nothing fires. She won;
  the void stands, because the +596 was never on the board.

Off-card bouts (withdrawals BFO keeps listed; `card_report.OFF_CARD`) are the
other kind of exclusion: dropped before the table, the signals and the
capture, so they never reach the ledger at all.

## Live-flag debut rule (2026-09-30)

`live = 1` requires, in addition to the card date (below), that both
fighters were in `data/fighters_v2.csv` with at least one UFC bout dated
before the event. A debutant has no pre-event state, so `card_report.py`
prints `debut-no proj` for the bout and no pick could have been on record
before the fight; a ledger row on such a bout only exists because the
backfill projects it after the fact. Rows failing the test are `live = 0`
(`betting_system.projectable`). Nothing else about the row changes: the
placement policy is a pure function of the other columns, so `placed`,
`stake_u` and `pnl_placed` are as before, and the row still counts in `FULL
SIM`. Four 2026 rows flipped when the rule went in: Nathaniel Wood (vs
Keita, 2026-03-21, placed 1.00u, WIN +1.30u), Robert Ruchala (2026-04-04),
Felipe Franco (vs Rodrigues Jr., 2026-07-18) and Saygid Izagakhmaev
(2026-07-25); only Wood had been placed, so the live placed record loses
that one win.

## Live-flag on-record rule (2026-09-30)

A live row must have been on record before the fight. Cards reported before
`OPEN_RULE_FROM` (2026-09-30) were evaluated pre-event at the raw first tick,
so on those cards a `live = 1` signal must also fire at the raw opener for
the same pick (`betting_system.raw_fired`); a signal that exists only under
the validated open there was never on record and is `live = 0`. From
2026-09-30 `card_report.py` itself evaluates the validated open, so the test
is moot. VOID rows keep their date-based flag (they were on record; they are
just not counted). Under the 40-point rule no row is affected by this test
(the rejected crossing test would have added two); it stays as the guard.

Live record after the three 2026-09-30 rules, `python betting_system.py
2026` (data through 2026-09-26):

```
LIVE RECORD (2026): 27 bets  21-6  staked 27u  P&L (placeable) +11.4u ($+1,142)  ROI +42.3%  t=2.66
    P&L (open) +12.0u ($+1,200)  ROI +44.4%  t=2.71
    FLIP: 12 bets  7-5  ROI +24.8%
    GAP: 15 bets  14-1  ROI +56.3%
    VOID (not counted): 1 (void: suspect opener)
PLACED (policy v1): 17 placed of 27 signals  14-3  staked 13.3u  P&L +6.2u ($+618)  ROI +46.4%  t=2.43
    FLIP: 6 placed  4-2  ROI +48.8%
    GAP: 11 placed  10-1  ROI +45.0%
```

Before them (same data) it read 31 signals 23-8, placed 18 of 31, 15-3,
+7.5u: the debut rule removes Wood's placed win (+1.30u) and three unplaced
rows, the opener rule voids Jauregui (never counted). Rosas (placed, +0.51u)
and Brener (skipped) are the UFC Vegas 121 additions.

## Polymarket: a second price source, not a signal source

`polymarket.py` reads the Polymarket moneyline for a bout (public Gamma and
CLOB APIs, disk-cached in `cache_pm/`, 0.45 s spacing). Nothing in the
signal rules, the placement policy or the pre-filter reads it; BestFightOdds
remains the only price a signal is measured against.

- `card_report.py` / `card_settle.py` print a `PM` column after `CLOSE`
  (fighter1's Polymarket price as a percentage; a resolved market shows
  0% / 100%) and the capture records `pm_p_self` (fighter1's price;
  fighter2's is 1 minus it), `pm_liquidity` and `pm_slug` in
  `data/placeable_lines.csv`, empty when there is no market. The ledger's
  `pm_p_at_capture` is the backed side's price at the earliest capture, on
  placed rows whose capture has one. Captures before 2026-09-30 carry no
  Polymarket columns.
- Full-card projection: for a future card, bouts Polymarket lists that BFO
  does not yet have are projected too (their own Stage 2 frame, so the BFO
  bouts' projections are unchanged) and printed with the BFO columns blank
  and `no BFO line` in the signal column. No signal can fire without a BFO
  open.
- `python polymarket.py backfill 2026`: for every 2026 ledger bout with a
  resolved market, both outcomes' hourly price histories go to
  `data/pm_history/<slug>.json` and `data/pm_clv_2026.csv` records the pick's
  price at our capture time (the earliest capture; event date 00:00 UTC
  minus 48 h when there is none) and its last pre-fight price (the last
  hourly point before the resolution tail), with the Polymarket CLV next to
  the BFO CLV. Read-only on everything else.

## Live vs backfilled

The `live` column separates two kinds of ledger rows.

- **live = 1**: the card report was run before the event, so the pick and the
  opening price were on record before the fight. That is every card from
  2026-01 through 2026-07-25 plus the dates listed in `LIVE_CARDS` in
  `betting_system.py`, and both fighters had a UFC bout on file before the
  event (the debut rule above). Add each new card's date there when its
  report is run pre-event.
- **live = 0**: backfilled after the fact from scraped lines; the opening price
  was not obtainable at the time.

The live record is the only admissible evidence for the rule. Backfilled rows
exist to keep the sim complete and are never quoted as results.
`betting_system.py` prints both: `LIVE RECORD` (live = 1 only, P&L at the
placeable line and at open) and `FULL SIM` (all rows, at open).
