# Totals edge test: pre-registration (2026-11-05)

Written and committed before any run of `totals_edge.py`. Nothing here is changed
after a run. If a run shows a bug, the fix and its effect get logged under
"Amendments" at the bottom, with the reason; the original text stays as written.

## Question
Does our duration model carry information beyond BFO's closing rounds-totals
price? The project's success criterion for the prediction path applies:
the optimal blend weight on the market is below 1.0.

## Data
- Prices: `data/bfo_props_joined.csv` (feat/bfo-props). The market probability
  q = vig-free P(Over) from `p1_close_clean` / `p2_close_clean` (side 1 = Over).
  Open probability q_open uses `p1_open` / `p2_open` (validated open). A row is
  used only if its overround is in [1.00, 1.15]. If a fight has two rows for the
  same line, keep the one with more ticks.
- Outcomes: `data/fights_v2.csv`. Elapsed = (round - 1) * 5 + clock minutes.
  Over 2.5 = elapsed > 12.5; Over 1.5 = elapsed > 7.5; GTD = U/S/M-DEC. Only
  fights ending in U/S/M-DEC, KO/TKO or SUB count. Fights ending exactly on
  the half-round (elapsed == 12.5 or 7.5) are dropped from evaluation.
- Scheduled rounds are inferred as in props_parlays: title bout, or main event
  from 2012 on.

## Targets
- **Primary:** Over 2.5 rounds, 3-round fights, BFO 2.5 line.
  Rows: dev 2,531, confirm 890, holdout 864.
- Secondary: Over 1.5 rounds, 3-round fights, BFO 1.5 line (1,666 / 884 / 858).
- Secondary: goes the distance, all fights (3,694 / 900 / 914).
- Five-round 3.5/4.5 lines: too few rows; not tested.

## Model (fixed; nothing tuned in this family)
M2 from `research/props_parlays/finish.py`, unchanged:
- Inputs: base (weight class, women, scheduled 5, year), moneyline mismatch
  (|vig-free ML close - 0.5| and its square), and fighter features (shrunk
  decision share, average fight minutes, finished-in-losses, KO losses, KD/15
  landed and absorbed, pace, sub attempts/15, experience, |rating diff|, RD sum).
- Fit: StandardScaler + LogisticRegression(C=1.0). The training set is valid
  fights with a clean ML close and year >= 2008. For the two 3-round targets,
  3-round fights only. Training does not need a prop price.
- One mechanical change: the ML close uses `a/b_close_clean` where present,
  else `a/b_close`. Clean values exist only for 2026, where BFO's in-play ticks
  would otherwise leak the fight's state into the mismatch feature. Rows from
  2025 and earlier are identical to finish.py.
- Walk-forward: refit every year Y on all fights before Y, predict Y.
  Dev = 2015-2022, confirm = 2023-2024. Holdout = 2025-01-01 to 2026-11-01,
  predicted by one fit on all fights through 2024 (no 2025+ outcome is ever
  used for fitting).

## Tests (per target and block)
With q = market P(Over), m = model P(Over), L = logit:
1. **Recalibrated market:** L(p) = a + b L(q).
2. **Stack:** L(p) = a + b L(q) + c L(m). Model share = c / (b + c).
3. **Gain** = mean per-fight log-loss (recalibrated market) minus log-loss
   (stack), with t = mean / (sd / sqrt(n)).
4. **Blend weight** (project criterion): p_w = sigmoid(w L(q) + (1 - w) L(m)),
   with w* = argmin log-loss over w in [0, 1] in steps of 0.01, fit inside the
   block. Bootstrap 95% CI, 2,000 resamples, seed 7.
5. Raw log-loss of the market, the model and the stack.

The parameters for tests 1-2 come from the dev block's out-of-sample
predictions and are applied unchanged to confirm. For the holdout they are refit
on dev + confirm (2015-2024). Test 4 is fit in-block in every block, as in v1.

## Secondary diagnostics (reported, not gates)
- **Line movement:** OLS of (q_close - q_open) on (m - q_open). A positive
  slope means the market moves toward the model between open and close.
- **Blend-at-open bets:** stack fit on dev (holdout: on 2015-24) with q_open as
  the market. Bet the side whose EV at the raw open decimal is above +3%,
  1 unit flat. Report n, hit rate, ROI, and mean CLV (vig-free probability move
  toward our side). The prices are BFO cross-book means, not the best
  available line; prop limits are low.

## Gate on the holdout (primary target only)
Confirm (2023-24) **passes** if both hold:
- the stack gain is > 0, and
- the in-block blend w* is < 1.0.

If confirm fails, the holdout is not read and this family closes on the
confirm result. If it passes, `totals_edge.py --holdout` runs **once**. It
writes `HOLDOUT_READ.md` (date, commit, results) and refuses to run if that file
already exists.

## Holdout reading
- Family success = holdout w* < 1.0 (the project criterion).
- Holdout gain with t >= 2 counts as **evidence**. Gain > 0 with t < 2 counts
  as **consistent but unproven**. Gain <= 0 counts as **failed**.
- Secondary targets and diagnostics are reported, never used to rescue a
  failed primary.

## Known overlap (flagged up front)
- M2's design and its 2015-24 log-loss were seen in props_parlays, which used
  duration outcomes but no prop prices. Dev and confirm are therefore not
  pristine for M2 itself. The comparison against the totals price is new. 2025+
  has never been read by anything in either study.
- Glicko parameters behind |rating diff| and RD sum were tuned on 2014-19,
  inside dev.

## Amendments
(none)
