# Props & parlays study, 2012-2024 (read-only research, 2026-11-05)

Three questions: (1) should we parlay our picks instead of betting them straight,
(2) is there a round over/under edge, (3) are there other props worth isolating.
Nothing here changes the frozen signal rules, placement policy, ledger, or ratings.

Windows: 2012-2022 primary, 2023-2024 confirm. **2025+ is never read** — every
script cuts its inputs to `event_date < 2025-01-01` (the ledger study to < 2023 where
noted) on load.

Run from anywhere, in this order (props_angles reads finish.py's output):

```
python research/props_parlays/parlay.py
python research/props_parlays/finish.py
python research/props_parlays/props_angles.py
```

About 10 s total. Each script blocks sockets and writes LF-only CSVs. The Monte
Carlo is seeded; two runs from different working directories give byte-identical
outputs (checked). `props_angles.py` needs `statsmodels`. Outputs:
`parlay_empirical.csv`, `parlay_montecarlo.csv`, `finish_logloss.csv`, and
`bouts_gtd.csv` (per-bout walk-forward P(goes the distance), 2008-2024).

## Bottom line

| Question | Answer |
|---|---|
| Parlays | Not now. A parlay is a leverage dial on per-leg edge, not a source of edge. Same sizing problem that got Kelly rejected. |
| Round O/U | Real duration signal in our fighter data beyond the moneyline, in both blocks. Necessary, not sufficient: we have no prop prices, so no edge claim. Next step is a BFO prop scrape. |
| Method props on our side | No. Our winners don't win by decision (or finish) more than the bout predicts. The edge is *who* wins, not *how*. |
| "Judging noise makes favorites overpriced in decision-heavy fights" | Dead. Null once you control for price. |

## 1. Parlays vs straights (`parlay.py`)

Universe: placed bets under policy v1, 2012-2024, priced at the line the ledger
settles at (the open). Equal-money comparison: a k-leg parlay stakes k units, the
same as k straights.

**Empirical (one path, legs grouped chronologically):**

| Block | Structure | Tickets | Staked | P&L | ROI | Hit | Max DD |
|---|---|---|---|---|---|---|---|
| 2012-22 | straights 1u | 139 | 139u | +39.2u | +28.2% | 68.3% | 5.6u |
| 2012-22 | 2-leg, 2u | 69 | 138u | +70.7u | +51.2% | 44.9% | 10.0u |
| 2012-22 | 3-leg, 3u | 46 | 138u | +120.8u | +87.5% | 28.3% | 21.5u |
| 2023-24 | straights 1u | 30 | 30u | +17.9u | +59.8% | 80.0% | 1.0u |
| 2023-24 | 2-leg, 2u | 15 | 30u | +43.7u | +145.6% | 60.0% | 4.0u |

Same-card parlays are rare: 22 of 142 cards in 2012-24 had 2+ placed bets
(17 with two, 5 with three). 2012-22: 20 same-card parlays made +44.0u on 43u vs
+19.8u straight. 2023-24: both lost (-6u) while the straights made +1.6u.

**Why the backtest flatters parlays:** the backtest leg ROI is +28%. Parlay ROI is
`((1+r)(1-h))^n - 1`, so a big r compounds into a huge number. Everything depends on
the true r, which we do not know yet (live record since 2026-09-08 is 3 signals).
And the primary block overlaps the Glicko tuning window (2014-19), the same caveat
as rule_search.

**Monte Carlo, one season = 12u staked** (legs drawn from our empirical price mix,
win prob set so the leg's true ROI = r):

| True leg ROI | Straights mean | P(losing season) | 5th pct | 2-leg mean | P(loss) | 5th pct | 3-leg mean | P(loss) | 5th pct |
|---|---|---|---|---|---|---|---|---|---|
| +20% | +2.4u | 23% | -3.1u | +5.2u | 30% | -7.8u | +8.7u | 35% | -12u |
| +10% | +1.2u | 36% | -4.3u | +2.5u | 41% | -12u | +4.0u | 45% | -12u |
| +5% | +0.6u | 43% | -4.9u | +1.2u | 47% | -12u | +1.9u | 51% | -12u |
| 0% | 0.0u | 50% | -5.5u | 0.0u | 53% | -12u | 0.0u | 56% | -12u |
| -3% | -0.4u | 55% | -5.9u | -0.8u | 57% | -12u | -1.1u | 59% | -12u |

**Line-shopping haircut.** A parlay has to sit at one book; straights get the best
price per leg. With a 2% price haircut h, a +2% true edge is exactly zero on every
structure, and anything below that loses faster on parlays.

| r | h | straight | 2-leg | 3-leg |
|---|---|---|---|---|
| +10% | 0% | +10.0% | +21.0% | +33.1% |
| +10% | 2% | +7.8% | +16.2% | +25.3% |
| +5% | 2% | +2.9% | +5.9% | +9.0% |
| +2% | 2% | 0.0% | -0.1% | -0.1% |
| 0% | 2% | -2.0% | -4.0% | -5.9% |

**Reading.** A 2-leg parlay is "bet A, then let the whole payout ride on B." That is
stake escalation. Under a known edge, growth-optimal betting on simultaneous fights
does include a small parlay piece, so this is not dumb math. The problem is the word
*known*: Kelly was rejected because the model is overconfident on selected bets, and
parlays reintroduce aggressive sizing through the back door. Revisit once the live
record is big enough to bound r (e.g. a lower confidence bound on live ROI > +5%).
If we ever do it: same-card only, small fixed sleeve, never > 2 legs.

## 2. Round over/under (`finish.py`)

No prop prices in the repo, so this asks the prior question: can we predict fight
duration better than weight class + the moneyline? If not, totals are a vig donation.

Walk-forward logistic regression, refit every year on all prior years (from 2008),
bouts with a clean BFO close only. Models:
- M0: weight class, women, scheduled 5 rounds, year
- M1: M0 + moneyline mismatch (|vig-free close - 0.5| and its square) — a proxy for
  what the book knows
- M2: M1 + our fighter features (shrunk prior-fight decision share, average fight
  minutes, finished-in-losses, KO losses, KD/15 landed and absorbed, strike pace,
  sub attempts/15, experience, |rating diff|, RD sum). All strictly pre-fight.
- M3: M0 + fighter features, no market

Log-loss (lower is better):

| Target | Block | n | Base rate | Const | M0 | M1 | M2 | M3 |
|---|---|---|---|---|---|---|---|---|
| Goes the distance | 2015-22 | 3,678 | 49.9% | 0.6934 | 0.6845 | 0.6804 | **0.6663** | 0.6710 |
| Goes the distance | 2023-24 | 976 | 51.6% | 0.6935 | 0.6863 | 0.6825 | **0.6612** | 0.6609 |
| Over 2.5 (3-rd) | 2015-22 | 3,295 | 55.6% | 0.6879 | 0.6801 | 0.6753 | **0.6632** | 0.6682 |
| Over 2.5 (3-rd) | 2023-24 | 878 | 55.6% | 0.6870 | 0.6817 | 0.6795 | **0.6628** | 0.6607 |
| Over 1.5 (3-rd) | 2015-22 | 3,295 | 68.4% | 0.6260 | 0.6190 | 0.6142 | **0.6061** | 0.6109 |
| Over 1.5 (3-rd) | 2023-24 | 878 | 69.5% | 0.6162 | 0.6081 | 0.6076 | **0.5926** | 0.5893 |

Fighter features beat the moneyline proxy by 0.014-0.021 log-loss on goes-the-distance,
in both blocks. For contrast, on the winner question our model carries zero signal over
the market. M2 calibration (2015-22 quintiles): predicted 0.29 / 0.42 / 0.50 / 0.57 /
0.68 vs actual 0.32 / 0.43 / 0.52 / 0.57 / 0.65 — slightly overconfident at the top.

Biggest drivers (standardized, fit 2008-22): career average fight minutes (+),
moneyline mismatch (big mismatches finish), 5-round bouts (fewer go the distance),
sub attempts (-), women's divisions (+), worst finished-in-losses rate (-).

**Caveat that matters:** M1 is a weak stand-in for the book. Books set totals with
average fight time and finish rates too — exactly our strongest features. So this is
the necessary condition, not an edge. The real test is M2 vs BFO's closing O/U and
goes-the-distance prices. Props get less sharp money and lower limits than sides, so
they are plausibly softer; that is the hypothesis to test, not a finding.

**Data feasibility (checked 2026-11-05):** BFO event pages carry the full prop set —
O/U 1.5/2.5 (3-rd) and 3.5/4.5 (5-rd), goes the distance, method per fighter,
round betting, decision type. Confirmed on noche-ufc-4318 (Sep 2026) and UFC 182
(Jan 2015, O/U 4.5 on Jones-Cormier, props on all 11 bouts). `scrape_bfo.parse_event`
currently skips prop rows. Unknown: whether prop line history (open/close ticks) is
reachable the way `/api/ggd` serves moneylines.

## 3. Other props (`props_angles.py`)

### 3a. Method skew of our signals
If our winners won by decision unusually often, "our fighter by decision" would carry
our edge at a longer price. They don't (frozen signals 2012-22, placed or not):

| Rule | Wins | Won by decision | Expected (bout's P(GTD)) |
|---|---|---|---|
| all | 196 | 50.5% | 46.2% |
| FLIP | 82 | 54.9% | 48.0% |
| GAP | 114 | 47.4% | 45.0% |

+4.3 pp overall with a standard error of about 3.6 pp: noise. FLIP's +6.9 pp is n=82,
about 1.3 SE. Win rate is the same whether the bout ended by finish (60.2%) or
decision (61.5%). Splitting our edge across a method ticket adds vig twice for no gain.

### 3b. Judging-noise hypothesis
Idea: split decisions are near-random (S=0.55), so in bouts likely to go the distance
the true win probability should be compressed toward 50% and favorites overpriced.

Pre-registered: dog ROI in the top P(GTD) tercile minus the bottom tercile, cutpoints
fixed on 2012-17. 2018-22: +14.6 pp at the open (95% CI +2.4 to +26.8), +8.0 pp at the
close (CI -6.2 to +21.9). 2023-24 same direction, CIs cross zero.

**That "pass" is confounded and should not be used.** The low-GTD tercile is the
mismatch tercile (M2 uses moneyline mismatch), so its dogs are longshots, and
longshots lose to the favorite-longshot bias. Controlling for price kills it:
`logit(fav_won) ~ logit(close price) + P(GTD)` gives a P(GTD) coefficient of +0.05,
+0.06, +0.13 (p = 0.88, 0.85, 0.82) in the three blocks. Favorites in decision-heavy
fights are priced correctly.

Side note, not new: favorites above 70% implied won 5-7 pp more than their vig-free
close in 2018-24, and dogs in that band returned -26% to -31% at the close. That's the
favorite-longshot bias the existing +250 skip rule already guards.

## Decisions to flag
1. **Orientation.** `fights_v2` lists the winner as `fighter_a`; `bfo_joined` a/b
   prices follow it. Checked: mean favorite implied 0.656 vs favorite win rate 0.663.
2. **Scheduled rounds** are inferred: title bout, or main event from 2012 on. 97.7% of
   bouts reaching round 4+ are flagged; 93% of non-title main-event decisions end in
   round 5.
3. **Clean close** uses the overround filter [1.00, 1.15], same as the hedge study.
4. **Shrinkage priors** in the fighter features (decision share toward 0.45 with k=4,
   etc.) were set once by judgment and never tuned.
5. **Glicko overlap.** `rating_diff`/RD come from params tuned on 2014-19, inside the
   primary block.
6. **Haircut h** for single-book parlays is assumed (0-4%), not measured.

## Next
1. BFO prop scrape (O/U rounds, goes the distance, method), 2015 on, open and close.
2. Then one experiment family: M2 vs the closing totals price, 2015-22 tune,
   2023-24 confirm, 2025+ once. Report the blend test against the book's implied
   probability, same as the moneyline work.
