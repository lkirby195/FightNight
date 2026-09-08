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

## Inputs, pre-filter, settlement

- Inputs: `data/model_only_preds.csv` (walk-forward model, trained strictly on
  prior years), `data/bfo_joined.csv` (BFO open/close per fight), `data/fights_v2.csv`.
- Pre-filter: skip bouts whose opening implied-probability sum is outside
  [1.00, 1.12] (odds sanity).
- One bet per qualifying bout, on the model's pick, settled at that side's
  opening decimal price. P&L is in units; 1u = $100 in the printed summary.
- 2025 is a sealed holdout: there are no 2025 model predictions, so the sim
  covers 2012-2024 and 2026.

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
event_date, event, fight_id, fighter, rule, live, units, open_line, clean_close,
clv_pts, result, pnl
```

`open_line` / `clean_close` are American odds for the side bet. `clean_close`
is the last pre-fight paired tick (the `card_settle.py` overround rule), so it
is safe for cards still carrying in-play ticks. `clv_pts` is the bet side's
vig-free close minus open in probability points. `pnl` is in units.

The ledger only covers bouts present in `data/model_only_preds.csv` and
`data/bfo_joined.csv`; cards scraped after the last `clv_eval.py` run are not in
it until those files are regenerated.

`data/legacy_favtier_ledger.csv` is the abandoned favorites-tier system
(3u/2u/1u by price band), ending 2026-03-28 plus the Paris FLIP row. Kept for
the record; not the system described here.

## Live vs backfilled

The `live` column separates two kinds of ledger rows.

- **live = 1**: the card report was run before the event, so the pick and the
  opening price were on record before the fight. That is every card from
  2026-01 through 2026-07-25 plus the dates listed in `LIVE_CARDS` in
  `betting_system.py`. Add each new card's date there when its report is run
  pre-event.
- **live = 0**: backfilled after the fact from scraped lines; the opening price
  was not obtainable at the time.

The live record is the only admissible evidence for the rule. Backfilled rows
exist to keep the sim complete and are never quoted as results.
`betting_system.py` prints both: `LIVE RECORD` (live = 1 only) and `FULL SIM`
(all rows).
