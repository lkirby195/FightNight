# Rule search, 2008-2022 (read-only research)

research/fade_buckets showed that fading the favourite fails and turned up one post-hoc
standout cell. This study asks the broader question: if we search a grid of simple
rules across long history, do we find real edges or just luck? It scores the search
procedure against a null in which no rule has an edge. Nothing here changes the frozen
rules or any file outside this folder.

Run: `python research/rule_search/rule_search.py` from anywhere (about 6 s). It blocks
sockets, cuts every input to 2008-01-01 <= event_date < 2023-01-01 on read, runs its
checks before writing, and uses the fixed seed 20261103. Two runs give byte-identical
outputs.

## Decisions to flag
1. **There is no `research/fade_buckets/bouts.csv`.** fade_buckets never wrote one, and
   it is not in git history. Sanity check 1 compares against the per-bout frame that
   fade_buckets builds itself: its `load()`, `tag()`, `per_bout()` and `universe()`,
   imported and called. All 1,757 bouts match exactly on all 48 shared columns.
2. **The refit misses the 0.005 tolerance in 2012-2014.** Max |dp| is 0.0082 in 2012,
   0.0054 in 2013 and 0.0059 in 2014. 2015-2022 stay within 0.0037. The cause is
   `stage2.build_matrix`, which centres two interaction terms and median-fills over the
   frame it is given. The frozen file's frame ran to 2026; this refit's ends at 2022.
   - Feature drift since the freeze is at most 0.001.
   - Moving the cut from 2023 to 2020 alone shifts 2012 by 0.0036.
   - Meeting 0.005 would mean centring over 2023+ rows, which the window rule forbids,
     so the refit is fade_buckets' as written.
   - The refit supplies 2008-2011 only. Of those years, only 2010-11 have prices
     (379 usable bouts).
3. **BFO prices start in 2010.** 2008-09 have model probabilities but no opening price,
   so no usable bouts. B1 is effectively 2010-2012: 619 usable bouts, which clears the
   300 minimum, so it is not merged.
4. **Benchmarks.** GAP and FLIP are the classifier's actual bet set: pick'ems included
   (as in the sim), VOID rows excluded, 2010-2022.
5. **Mechanics.**
   - Walk-forward ties go to the first rule in rules.csv order. The only ties are
     identical bet sets (D76 = D82), so the tie-break never changes a bet.
   - "Percentile" means the share of null replicates strictly below the real value.
   - The null 95th percentile is `numpy.quantile` (linear interpolation).

## Step 0: coverage
| year | bouts | 2-sided open | p_mod (source) | usable | KK | 2-sided close | usable w/ clean close |
|---|---|---|---|---|---|---|---|
| 2008 | 201 | 0 | 201 refit | 0 | 0 | 0 | 0 |
| 2009 | 215 | 0 | 213 refit | 0 | 0 | 0 | 0 |
| 2010 | 253 | 251 | 250 refit | 185 | 27 | 251 | 185 |
| 2011 | 300 | 298 | 295 refit | 194 | 37 | 298 | 194 |
| 2012 | 341 | 335 | 333 frozen | 240 | 44 | 335 | 240 |
| 2013 | 386 | 379 | 376 frozen | 252 | 42 | 379 | 252 |
| 2014 | 503 | 488 | 494 frozen | 325 | 52 | 488 | 325 |
| 2015 | 473 | 457 | 464 frozen | 366 | 77 | 457 | 366 |
| 2016 | 493 | 477 | 483 frozen | 364 | 101 | 477 | 364 |
| 2017 | 457 | 446 | 446 frozen | 325 | 88 | 446 | 325 |
| 2018 | 474 | 452 | 469 frozen | 334 | 104 | 452 | 334 |
| 2019 | 516 | 490 | 506 frozen | 358 | 99 | 490 | 357 |
| 2020 | 456 | 441 | 444 frozen | 316 | 93 | 441 | 316 |
| 2021 | 509 | 482 | 497 frozen | 377 | 106 | 482 | 376 |
| 2022 | 511 | 487 | 506 frozen | 372 | 119 | 487 | 372 |
| **total** | 6,088 | 5,483 | 5,977 | **4,008** | 989 | 5,483 | 4,006 |

From 5,483 priced bouts to 4,008 usable:
- 42 draws and 66 NCs have no model probability.
- 1,213 debut bouts are excluded by the debut rule.
- 154 exact pick'ems are excluded.
- No bout fails the pre-filter and none has a gap of exactly 0.

Blocks: B1 2008-12 (619 usable), B2 2013-17 (1,632), B3 2018-22 (1,757). The full
per-year table, including each exclusion, is in coverage.csv.

## Step 1: rules
rules.csv has 105 grid rules (15 FAV + 90 DOG), plus STANDOUT and the GAP and FLIP
benchmarks. The 105 rules are 96 distinct bet sets, because some filters imply others
(`same_bets_as` column):
- p_mod <= 0.40 forces gap <= -10, so D13-D18 equal D31-D36.
- A +250 cap together with gap <= -25 forces p_mod < 0.50, so D76-D78 equal D82-D84.

## Step 2: consistency (rule_blocks.csv)
The null max-t 95th percentile is **2.32**. Top 10 rules by pooled t (ROI in %):

| rule | n | ROI | t | B1 | B2 | B3 | blocks + | null pct |
|---|---|---|---|---|---|---|---|---|
| D64 dog, gap <= -20, p_mod < .50, <= +250 | 481 | +20.9 | 3.44 | -6.1 | +26.2 | +32.9 | 2 | 100.0 |
| D65 = D64 + fav RD <= 160 | 159 | +34.2 | 3.24 | -20.9 | +41.2 | +45.3 | 2 | 99.4 |
| **D61** dog, gap <= -20, p_mod < .50 | 538 | +18.1 | 3.06 | +3.1 | +19.3 | +27.7 | 3 | 98.9 |
| D58 dog, gap <= -20, <= +250 | 500 | +17.7 | 2.97 | -8.6 | +22.6 | +29.4 | 2 | 98.8 |
| D46 dog, gap <= -15, p_mod < .50, <= +250 | 732 | +13.8 | 2.90 | -3.4 | +20.5 | +16.3 | 2 | 98.4 |
| D66 = D64 + both RD <= 160 | 90 | +39.4 | 2.79 | -15.4 | +26.5 | +64.3 | 2 | 97.8 |
| D28 dog, gap <= -10, p_mod < .50, <= +250 | 974 | +11.1 | 2.76 | -8.0 | +17.1 | +14.0 | 2 | 97.8 |
| D59 dog, gap <= -20, <= +250, fav RD <= 160 | 166 | +28.5 | 2.76 | -24.2 | +34.6 | +39.8 | 2 | 97.8 |
| D62 dog, gap <= -20, p_mod < .50, fav RD <= 160 | 180 | +27.2 | 2.67 | -8.8 | +29.5 | +37.6 | 2 | 97.6 |
| **D76** (= D82) dog, gap <= -25, <= +250 | 240 | +23.2 | 2.65 | +3.0 | +42.3 | +14.9 | 3 | 97.6 |

**Pass bar:**
- 22 rules are positive in every block.
- 17 have a pooled t above 2.32.
- 53 have at least 30 bets in every block.

**4 of 105 rules advance (3 distinct bet sets):**
- D43: dog, gap <= -15, p_mod < .50. 789 bets, +12.4%, t 2.64; B1 +3.3%, B2 +15.7%, B3 +14.3%.
- D61 (in the table above).
- D76 and D82 (the same bets).

All 105 FAV and DOG rules have a positive mean CLV.

## STANDOUT
Rule: bet the dog when -25 < gap <= -20 and p_mod < 0.50.

Fresh test on B1+B2 (2008-17): 149 bets, 59-90, ROI +1.6%, SE 10.4%, t +0.15. **It
does not advance** (the bar is ROI > 0 with t >= 1.65). By block:
- B1: 13-30, -20.7%
- B2: 46-60, +10.6%
- B3: 52-40, +46.1% (the block where it was found)

## Step 3: walk-forward (walkforward.csv, walkforward_pnl.png)
| years | rule picked from earlier years |
|---|---|
| 2011 | D02 (dog, gap <= -5, fav RD <= 160) |
| 2012-16 | F07 (fav, gap >= 8) |
| 2017-19 | D76 |
| 2020-22 | D64 |

Out of sample: 329 bets, 185-144, +43.8u, ROI **+13.3%**, SE 6.1%, t +2.20. At the
market's prices, the vig alone would have lost 18.5u. The null 95th percentile is
+5.5% and the real ROI sits at the 99.7th percentile, so **by the brief's bar the
procedure works.** The gains come from 2017 on, after the search moved to the
big-gap dog rules. F07's five years net +6.5u.

## Step 4: null (null_dist.csv, null_summary.csv, null_max_t.png)
1,000 replicates. Each redraws every winner from the vig-free open (fav wins with
p_mkt) and keeps prices, p_mod and RD fixed.

| statistic | real | null mean | null 95th | percentile |
|---|---|---|---|---|
| (a) max pooled t | 3.44 (D64) | 1.24 | 2.32 | 100.0 |
| (b) rules positive in all 3 blocks | 22 | 4.2 | 18 | 96.9 |
| (c) walk-forward OOS ROI | +13.3% | -6.2% | +5.5% | 99.7 |
| GAP pooled ROI (203 bets) | +7.2% | -5.7% | +3.3% | 99.6 |
| FLIP pooled ROI (195 bets) | +22.3% | -6.1% | +8.0% | 100.0 |

## Sanity checks (all PASS)
1. **2018-22 vs fade_buckets.** My 2018-22 bouts equal fade_buckets' own frame exactly:
   1,757 bouts, 48 shared columns.
2. **GAP and FLIP, 2012-2022, 1u at open.** My tags equal `betting_system.run()` on the
   same frame and the committed ledger, with the same fight_ids:
   - GAP: 165 bets, 114-51, +11.39u, ROI +6.90%
   - FLIP: 157 bets, 82-75, +38.73u, ROI +24.67%
3. **Null ROI vs the vig.** Under the null, the mean pooled ROI over the 105 rules is
   -5.590%. The exact expectation (mean 1/overround - 1 over each rule's bets) is
   -5.577%. The universe's average vig is 5.50%.
4. **Byte-identical runs.** Two runs, in the working copy and in a fresh clone, give
   byte-identical outputs and console output.

An internal check also passes: the summed scoring path equals a direct per-rule
computation (max |t diff| 9e-16).

## Reading it
- **What the null tests.** It asks whether the opening price is right. It does not
  separate model skill from a model designed partly on this window:
  - The Glicko parameters were tuned on 2014-2019 log-loss.
  - B1 (2010-12) is the only block outside that tuning. There the advancing rules are
    +3.0% to +3.3% on 69-173 bets, and the top rule D64 is -6.1%.
- **Where the edge sits.** Everything that survives is "the model picks the dog against
  a big gap": the FLIP region, widened.
  - D43 contains all 195 FLIP bets.
  - Post hoc, not a test: the non-FLIP parts are also positive (D43's other 594 bets
    +9.2%, D61's other 355 bets +15.7%).
- **How fair the null is to dogs.** Favourites priced above 0.70 won 79.8% against
  74.9% expected (favourite-longshot bias). Overall, favourites won 2,576 against
  2,546.2 expected (sd 30.1). The null is, if anything, kind to dogs, so the DOG
  results are not flattered by it.
- **What 2023-24 can confirm.** It confirms D43, D61 and D76. As fade_buckets noted, the
  Stage 2 variant and C were picked on 2023-24, so that window is untouched for these
  rules but not for the model.
