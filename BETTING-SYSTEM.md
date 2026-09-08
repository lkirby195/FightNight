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

`python betting_system.py all` (data through 2026-07-25):

```
BETTING SYSTEM (all): 524 bets  303-221  staked 524u  P&L +56.7u ($+5,671)  ROI +10.8%  t=2.43
    FLIP: 273 bets  129-144  ROI +13.2%
    GAP: 251 bets  174-77  ROI +8.2%
```

`python betting_system.py 2026`:

```
BETTING SYSTEM (2026): 23 bets  17-6  staked 23u  P&L +8.5u ($+851)  ROI +37.0%  t=2.00
    FLIP: 11 bets  6-5  ROI +22.3%
    GAP: 12 bets  11-1  ROI +50.5%
```

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
event_date, event, fight_id, fighter, rule, units, open_line, clean_close,
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
