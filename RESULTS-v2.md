# v2.0 Prediction Path — Results Memo (Aug 2026)

**Status: experiment family closed at tune level. 2025+ holdout NOT spent — still sealed.**

## Architecture (as specced)
Stage 1: frozen v1 Glicko-2 spine, reproduced bit-for-bit (max rating delta 0.05
across all 2,722 fighters after fixing chronological bout order and merging MW flags).
Stage 2: logistic regression, antisymmetric by construction (paired differences +
mirror augmentation, no intercept; max |P(A)+P(B)−1| = 2e-16). Train ≤2022,
select on 2023–24, holdout 2025+ untouched.

## Data (all re-scraped, cross-validated)
- fights_v2.csv — 8,794 bouts w/ per-fight KD/sig-str/TD/sub-att, chronological
  bout order, v1 missed-weight flags (206 incidents)
- fighters_v2.csv — 2,722 fighters: DOB, height, reach, stance
- fight_details_v2.csv — 8,773 bouts (99.8%): strike attempts, control time,
  head/body/leg + distance/clinch/ground splits. Corner-order scramble on 3,098
  fights detected and fixed by name key; 100% sig-strike agreement with event
  tables after re-orientation.

## Results (2023–24 tune, market-matched n=791)

| Model | log-loss |
|---|---|
| Glicko spine (v1 exact) | 0.6730 (full tune n=1,017) |
| Stage 2 | 0.6415 |
| Market vig-free (v1 odds file) | 0.5877 |
| Market recalibrated (scale 1.09, fit ≤2022) | 0.5855 |

Incremental gain of Stage 2 stacked on recalibrated market, three pre-declared variants:

| Variant | features | stack w | gain | t |
|---|---|---|---|---|
| Career aggregates (pre-Pass-C) | 23 | 0.24 | +0.0020 | 1.27 |
| + all Pass C | 32 | 0.31 | +0.0014 | 0.65 |
| + d_ctrled15 only (trim) | 24 | 0.25 | +0.0016 | 0.91 |

## Conclusions
1. Stage 2 closes ~45% of the v1→market gap and is the first variant to take
   positive stacking weight against closing lines (v1 took −0.10).
2. The incremental edge is consistently positive (+0.0014…+0.0020) across
   variants but never significant. Power analysis: at the observed effect size,
   establishing t≥2 needs ~2,300–4,000 matched fights ≈ 4.5–8 years of UFC.
   The backtest cannot settle this question; only accumulation can.
3. Pass C features (accuracy/defense/control) subtracted value — these are the
   most market-visible stats in the sport and are already priced. The residual
   edge lives in age, volume, and rating-conditioned win rate.
4. Market closing lines are slightly under-confident (free +0.002 from a 1.09
   logit scale) — replicated on train and tune independently.

## Walk-forward evaluation (added after review)
The fixed split wasted 2010–2022's matched fights as pure training data.
Walk-forward recovery: for each year Y (2012–2024), Stage 2 refit on fights <Y,
stacker fit only on *out-of-sample* predictions from prior years, evaluated on
year Y. Every prediction strictly pre-fight; 2025+ untouched. Pre-registered:
career-aggregate variant, C=0.3, one run.

| Era | n | gain vs recalibrated market | t |
|---|---|---|---|
| 2012–2016 | 2,021 | +0.0008 | 1.05 |
| 2017–2021 | 2,179 | +0.0001 | 0.09 |
| 2022–2024 | 1,266 | +0.0018 | 2.29 |
| **Pooled** | **5,466** | **+0.00075** | **1.68** |

Findings:
- Pooled over 13 years: positive but not significant (t=1.68 at n=5,466). The
  true long-run effect is likely ~+0.0005–0.001 — *smaller* than the tune-set
  estimate. At that size even the full walk-forward sample cannot reach t=2.
- The stacker assigns the model weight 0.12–0.17 in every single year — the
  combination never drops the model, but the realized gain is era-dependent.
- The effect is non-stationary: dead in 2017–2021, strongest in 2022–2024
  (t=2.29, though this sub-window partially overlaps the design loop via the
  2023–24 tune years; 2022 alone — never used for any selection — shows
  +0.0022, the largest single-year gain in the series).
- Hypothesis worth tracking, not claiming: model quality rises as the per-fight
  stats era (2001+) comes to cover fighters' full careers, making career
  aggregates more informative in recent years.

## Decision
Holdout stays sealed: at holdout-matched n≈450, a true +0.002 effect yields an
expected t≈0.9 — the test is mathematically incapable of settling the question,
so spending the one shot would burn the credibility artifact for no information.
It gets spent when a variant clears t≥2 on tune, or as the final report if the
prediction path is formally shelved.

Next lever if/when resumed: style-matchup interactions, or odds-timeline
features (line movement). Both are new-information plays, not re-cuts of
career aggregates. The tune set has absorbed three looks and should be treated
as nearly spent.

## Business read
Forward-going, the model produces ~500 predictions/year. Publishing them live
alongside the rankings site turns the audience product into the accumulating
significance test the backtest can't run — the public track record *is* the
experiment, and it compounds credibility either way it resolves.


## CLV evaluation (BestFightOdds open/close lines, added Aug 2026)
Acquired full line-history for 7,962 UFC bouts 2010-2026 (BFO /api/ggd, base64+
ROT47 decoded; opening tick -> closing tick of the cross-book mean line).
Joined 7,131 to fights; 5,982 carry model-only walk-forward predictions.
Metric: vig-free CLV of the pure Stage 2 pick (no market input) taken at the
opening line, graded against the close. Random-side control: +0.07 pts (t=0.6).

| Era | n | CLV | t | slope(dis->move) |
|---|---|---|---|---|
| 2012-2016 | 2,076 | -0.09 | -0.6 | 0.03 |
| 2017-2021 | 2,224 | +1.90 | 10.6 | 0.20 |
| 2022-2026 | 1,682 | +2.04 | 9.1 | 0.22 |
| Pooled | 5,982 | +1.25 | 11.7 | 0.13 |

Findings:
1. The model demonstrably moves lines: from 2018 onward, every single year is
   positive (+1.6 to +3.0 pts), slope 0.20-0.31 -- the market validates ~a fifth
   of the model's disagreement with the opener by close.
2. Sharp regime break at 2018: 2012-2017 slope ~0 (openers already efficient
   vs our information), 2018+ slope ~0.25. Coincides with PASPA repeal (May
   2018) -- plausibly recreational-money openers became softer post-
   legalization. Interpretation, not proof.
3. Raw model picks at the open still lose money (-5.2% ROI): ~78% of model
   disagreement is model error. Deployable form is blend-at-open (anchor open,
   tilt by model; weight fit 22-23): validation 2024+2026 = 374 bets/2yr,
   CLV +4.95 pts (t=9.6), ROI +5.4% (t=1.2).
4. Business: publish picks at the opener with timestamps; running CLV is the
   headline credibility metric -- provable in months, unfakeable, and now
   backtested at t>9 across seven consecutive years.
