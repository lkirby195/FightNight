# Totals edge test (2026-11-05)

Question: does our duration model (M2 from props_parlays, unchanged) carry
information beyond BFO's closing rounds-totals price?

**Answer: no. The pre-registered gate failed on 2023-24, so 2025+ was not
read.** The model does predict which way the totals line moves from open to
close, but the close already contains that information.

Pre-registration: [PREREG.md](PREREG.md), committed at 783ecba before any run.
Nothing in this study changes the frozen signal rules, placement policy,
ledger or ratings.

```
python research/totals_edge/totals_edge.py     # dev + confirm, ~60 s; no 2025+ loaded
python research/totals_edge/posthoc.py         # NOT pre-registered; reads the output above
```

Two runs from different working directories give byte-identical outputs
(checked). The gate number was recomputed with independent code (statsmodels
instead of sklearn): identical.

## Result

Log-loss (lower is better). The stack is `logit p = a + b logit(q) + c logit(m)`.
Stack and recalibration parameters are fit on dev and applied unchanged to
confirm. **Dev gain and dev bets are in-sample**; confirm is out of sample.
The blend weight w* is fit inside each block (as in v1), where w* = 1.0 means
the model adds nothing.

| Target | Block | n | Market | Model | Market recal. | Stack | Gain vs recal. (t) | Blend w* [95% CI] |
|---|---|---|---|---|---|---|---|---|
| **O2.5, 3-rd (primary)** | dev 2015-22 | 2,499 | 0.6467 | 0.6594 | 0.6455 | 0.6448 | +0.0007 (0.93) | 0.79 [0.61, 0.99] |
| **O2.5, 3-rd (primary)** | confirm 2023-24 | 870 | 0.6431 | 0.6636 | 0.6384 | 0.6399 | **-0.0016 (-1.32)** | 0.95 [0.64, 1.00] |
| O1.5, 3-rd | dev | 1,633 | 0.6339 | 0.6479 | 0.6328 | 0.6323 | +0.0005 (0.63) | 0.82 [0.59, 1.00] |
| O1.5, 3-rd | confirm | 864 | 0.5770 | 0.5921 | 0.5708 | 0.5717 | -0.0008 (-0.88) | 0.89 [0.58, 1.00] |
| Goes the distance | dev | 3,638 | 0.6519 | 0.6662 | 0.6506 | 0.6500 | +0.0006 (1.00) | 0.81 [0.66, 0.96] |
| Goes the distance | confirm | 881 | 0.6424 | 0.6578 | 0.6366 | 0.6366 | +0.0000 (0.00) | 0.81 [0.51, 1.00] |

**Gate (primary, confirm):** gain > 0 and w* < 1.0. Gain is -0.0016, so it
**FAILS**. The holdout stays unread and this family is closed.

Reading:
- The totals market beats our model outright by 0.013-0.021 log-loss in every
  row. In props_parlays the same model beat a moneyline-mismatch proxy for the
  book; the real book already prices what our features know.
- The model's stack coefficient was borderline even on dev (p = 0.06), and
  never significant anywhere. Same shape as the moneyline result: the market
  close carries everything we have.
- The in-block w* sits below 1.0, but every CI reaches 1.0. That is in-sample
  noise, not signal: the dev-fit blend loses out of sample.

### Secondary, pre-registered (not gates)

**Line movement.** OLS of (q_close - q_open) on (m - q_open): slope 0.17 dev
(t 17), 0.08 confirm (t 6.9). The same holds for the other two targets.

**Blend-at-open bets** (stack on the open price, bet when EV > 3% at the raw
open, 1u flat):

| Target | Block | Bets | Hit | ROI (t) | CLV pts (t) |
|---|---|---|---|---|---|
| O2.5 | dev (in-sample) | 581 | 56.3% | +6.2% (1.48) | +2.59 (11.1) |
| O2.5 | confirm | 201 | 41.3% | +1.4% (0.14) | +0.79 (2.85) |
| O1.5 | confirm | 109 | 45.0% | -3.9% (-0.34) | +2.44 (4.20) |
| Goes the distance | confirm | 233 | 40.8% | +15.9% (1.52) | +0.83 (2.60) |

CLV out of sample is positive, +0.8 to +2.4 points, against about 3 points of
vig per side at BFO's mean price. The secondaries cannot rescue the primary, per
the pre-registration. The goes-the-distance ROI is mostly the 2023-24
long-fight shift below.

## Post-hoc, NOT pre-registered (`posthoc.py`, 2015-24 only)

**A) The line-movement signal is real, not a statistical artifact.** A noisy
open that drifts back toward the close produces a positive slope even for a
model with no information. Controls:

| Target | Block | Pre-reg slope (t) | Null model* (t) | M2 coef given q_open (t) |
|---|---|---|---|---|
| O2.5 | dev | 0.170 (17.4) | 0.016 (1.5) | 0.176 (17.5) |
| O2.5 | confirm | 0.080 (6.9) | 0.016 (1.9) | 0.086 (6.7) |
| O1.5 | confirm | 0.114 (8.9) | 0.013 (1.3) | 0.120 (8.9) |
| Goes the distance | confirm | 0.090 (7.6) | 0.019 (2.4) | 0.097 (7.3) |

\*The null "model" is the training-period base rate. Between open and close
the market moves 8-21% of the way toward M2's disagreement with the opener,
the same pattern as the moneyline CLV study. M2 is telling us where the line
is going, not where it is wrong at the close.

**B) Books put the vig on the short side.** Vig-free, the market underprices
the long side (Over / goes the distance) most years. For goes the distance it
is low in all 10 years from 2015 to 2024, by -0.2 to -7.4 pp. Bet blind at raw
BFO mean prices:

| Target | Price | 2015-18 | 2019-22 | 2023-24 | Short side, 2015-24 |
|---|---|---|---|---|---|
| Over 2.5 | close | -0.3% | -0.6% | +5.5% (t 1.5) | -12.7% |
| Over 1.5 | close | +2.7% | -4.9% | +2.5% (t 1.0) | -15.6% |
| Goes the distance, Yes | close | -0.2% | +0.1% | +9.7% (t 2.2) | -12.1% |
| Goes the distance, Yes | open | -4.0% | -4.5% | +8.1% (t 1.9) | -10.1% |

The long side is about break-even before 2023. The short side (Under, "doesn't
go the distance") loses 7-26% at the mean price in every window, so that is
where the overround sits. The 2023-24 long-side profit coincides with realized
decision rates jumping (goes the distance 0.49-0.54 against a market at
0.43-0.46). That could be a market lagging a real shift, or just two years
of luck. The hypothesis came from looking at 2023-24, so 2023-24 cannot
confirm it.

## Decisions to flag
1. **Holdout unread.** 2025-26 is still clean for any future totals family.
2. **Eval n is below the pre-registered row counts** (for example 870 vs 890 on
   confirm). Fights drop out without a clean moneyline close (an M2 input) or
   when they end exactly on the half-round.
3. **Prices are BFO cross-book means.** The best available line is better
   than the mean, so ROI here is conservative. Prop limits are low, though.
4. **Overlap:** M2's design saw the 2015-24 duration outcomes in
   props_parlays, and the Glicko parameters were tuned on 2014-19.

## Next (needs a call)
- **Totals as a model edge: park it.** Same conclusion as moneylines: no signal
  at the close.
- **Long-side bias:** if worth pursuing, run it as its own pre-registered family
  (blind long side at the close) on 2025-26, read once. Prior is low: break-even
  for 8 years, then 2 good ones. Cheaper option: add it as a shadow track for the
  March 2027 review alongside SHADOW_D61 and leave the holdout alone.
