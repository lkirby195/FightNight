# Fade buckets, 2018-2022 (read-only research)

A reviewer proposed fading market favourites when the model is much less bullish
than the market (model 52% / market 73% -> bet the dog). This is the mirror image of
GAP and had never been tested. Nothing here changes the frozen rules or any file
outside this folder.

Run: `python research/fade_buckets/fade_buckets.py` from anywhere (about 30 s). It
blocks sockets, loads nothing dated 2023-01-01 or later, runs its checks first and
writes nothing if one fails. Two runs give byte-identical outputs.

## Inputs
- Model probability: `data/model_only_preds_frozen_2026-09-08.csv`. It produces the
  same GAP/FLIP set as `data/model_only_preds.csv` over the window (max |dp| 0.00095).
- Opening prices: the validated opens in `data/bfo_joined.csv`, the ones
  `signals.classify` uses. Closes: the clean close where one exists (none before 2026),
  else the raw last tick when it is a clean book (card_settle's [1.00, 1.15]). That is
  the ledger's `_clean_close` rule without the network re-read.
- RDs: `data/prefight_rd.csv`, the same values placement policy v1 uses. "Known" means
  RD <= 160 (`signals.UNKNOWN_RD`).
- GAP/FLIP tags: `betting_system._decision` -> `signals.classify`. Not reimplemented.

## Walk-forward (check 0)
`clv_eval.gen_model_preds` fits year Y on bouts dated before Y-01-01. The script refits
it the same way from `features_v2.csv` cut at 2023-01-01 and gets max |dp| 0.0036
against the frozen file. One in-sample fit through 2022 gets max 0.084. The small
remainder comes from `stage2.build_matrix`, which centres two interaction terms and
median-fills over the whole frame it is given.

The model is not perfectly blind to the window:
- the Glicko-2 parameters were tuned on 2014-2019 log-loss (spec section 11);
- the Stage 2 variant and C = 0.3 were picked on the 2023-24 tune set (RESULTS-v2.md).

The second point means 2023-24 has already been looked at for model design, though
not for this rule.

## Universe
| step | bouts |
|---|---|
| fights_v2 bouts in the window | 2,466 |
| with a two-sided BFO open | 2,352 |
| minus 19 draws and 24 NCs (no model probability exists for them) | 2,309 |
| minus 462 debut bouts (debut rule, 2026-10-22: no projection) | 1,847 |
| minus 90 exact pick'ems at open | **1,757 bucketed** |

No bout fails the odds-sanity pre-filter and none has a gap of exactly 0.

## Definitions
- **Mean opening price:** the mean decimal price of the model side, plus its American
  equivalent.
- **Dog price cap:** compared on the rounded American line, as policy v1 does.
- **fade_cell.csv extra columns:** wins expected at the vig-free open (`mkt_exp_w`),
  wins expected at the model's price (`mod_exp_w`), and the chance of that few wins if
  the open were right (`p_mkt_le_w`). They are there because the pre-registered cell
  went 0-11, which makes its SE 0 and its t undefined.

## Checks
1. The GAP and FLIP tags reproduce `betting_system.run()` on the same frame and the
   committed ledger exactly: FLIP 59 bets 30-29 +9.99u, GAP 68 bets 47-21 +6.49u.
   9 of the GAP bets are pick'ems, which sit outside the buckets. CLV on the 118
   bucketed tagged bouts matches the ledger's `clv_pts` to 0.005 pp.
2. Every bout lands in exactly one bucket. Bucket n's sum to each tier, and
   KK + KU + UU = ALL = 1,757.

## Result, in one line
In the model-agrees subset, betting the dog loses: 741 bets, 232-509, ROI -15.2%,
t -3.20. The pre-registered cell went 0-11 against 3.2 wins expected at the market's
prices (P = 0.024).
