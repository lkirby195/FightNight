# MMA Rating Engine — Glicko-2 Adaptation Spec (v1.1)

**Status:** Implemented, tuned, and validated (see §11)
**Scope:** UFC fights, global pool (weight-class views deferred to v2)
**Base system:** Glicko-2 (Glickman, 2001), adapted for one-fight rating periods, graded outcomes, and missed-weight asymmetry

---

## 1. State per fighter

Each fighter carries three values:

| Symbol | Name | Public scale | Initial value |
|---|---|---|---|
| r | Rating | Elo-like, centered 1500 | 1500 |
| RD | Rating deviation (std error) | Same scale as r | 350 |
| σ | Volatility | Internal | 0.06 |

Plus bookkeeping: date of last fight, fight count.

Interpretation: true skill lies within roughly r ± 2·RD with ~95% confidence. A debutant is 1500 ± 700 ("could be anyone"); a 20-fight veteran converges toward ± 100 or tighter.

---

## 2. Scale conversion

All updates run on the Glicko-2 internal scale:

```
μ = (r − 1500) / 173.7178
φ = RD / 173.7178
```

Convert back after updating:

```
r' = 173.7178 · μ' + 1500
RD' = 173.7178 · φ'
```

---

## 3. Rating periods and time

**One fight = one rating period** for the two fighters involved. No batch processing; fights are processed in chronological order (by event date, then bout order on the card).

**Standard downtime = 1 year.** Time is measured in fractional years. Define, for each fighter entering a fight:

```
n = (fight_date − last_fight_date) / 365.25        [years, fractional]
```

For a debut, n = 0 (RD is already at maximum).

**Pre-fight RD inflation.** Before computing the update, inflate each fighter's deviation for elapsed time:

```
φ_pre = min( sqrt(φ² + σ² · n),  350 / 173.7178 )
```

RD is capped at 350 — a fighter can never be *more* unknown than a debutant.

Consequences of this calibration:
- A fighter on a normal 1-fight-per-year cadence accrues exactly one standard period of uncertainty growth between fights — the Glicko-2 default behavior.
- An active fighter (3 fights/year) accrues only n ≈ 0.33 per fight: confidence compounds.
- A 2.5-year layoff accrues n = 2.5: the returning fighter's rating is intact but wide-error-barred, so their first result back moves them fast. This is the entire discontinuity/layoff/USADA-era mechanism — no special-case rules.

---

## 4. Outcome scores (graded S)

The actual score S replaces Elo's binary win/loss. Method of victory grades the evidence:

| Result | Winner S | Loser S |
|---|---|---|
| KO / TKO / Submission | 1.00 | 0.00 |
| Unanimous decision | 0.90 | 0.10 |
| Split decision | 0.75 | 0.25 |
| Majority draw | 0.55 (the fighter one judge favored) | 0.45 |
| Draw (unanimous/split) | 0.50 | 0.50 |
| Technical decision | score as the decision type it produced | — |
| DQ win | 0.75 | 0.25 |

Notes:
- S values are **tuning parameters** with these as priors; they get fit in backtesting (§9).
- **Accepted consequence (by design):** a heavy favorite whose expected score E exceeds their graded S *loses rating on a win*. A −450 favorite squeaking out a split decision (S = 0.75, E ≈ 0.82) bleeds points. This is intended signal.
- **DQ** is scored as a low-information win (split-decision equivalent) — the fouler loses, but the result says little about skill. Tunable.
- Majority decision (win) is scored as unanimous (0.90). Optional refinement later: 0.85.

---

## 5. Special-case handling

### 5.1 No contest
No rating update for either fighter — r, RD, and σ revert to their pre-fight values (the time inflation computed in Step 0 is discarded). However, the NC **does** update last_fight_date: the fight happened and both fighters were active, so it is evidence of activity even though it is not evidence of skill. Their uncertainty clock restarts from the NC date.

### 5.2 Overturned results
A result later overturned to NC is reprocessed as NC from the original date; all subsequent ratings recompute. (Full-history replay is cheap; the engine must be deterministic and replayable.)

### 5.3 Missed weight (asymmetric credit)
Let the **offender** be the fighter who missed weight and the **victim** their opponent. Adjustments:

| Scenario | Treatment |
|---|---|
| Offender wins | S_offender × **0.80** (won with a physiological edge — discounted). Victim's loss score S_victim raised by the mirrored amount so S_off + S_vic still sums to 1: S_victim = 1 − 0.80·S_offender_raw |
| Offender loses | Full penalty, no adjustment. No sympathy for losing with an advantage. |
| Victim wins | Full S, **plus a bonus multiplier β_v = 1.10 applied to the victim's rating change Δμ** (post-computation, μ-update only — never applied to φ or σ updates). Overcoming the weight edge is rewarded. |
| Victim loses | S floor of **0.10** (a KO/sub loss scores 0.10 instead of 0.00) — softened, they fought a bigger man. |

Implementation note: the victim's win bonus is applied to the rating *change*, not to S, because S is already at or near its ceiling for finishes (S ≤ 1.0 must hold inside the Glicko-2 update math). All three constants (0.80, 1.10, 0.10) are tuning parameters.

If **both** fighters miss weight (catchweight by mutual failure): no adjustments, rate normally.

Scheduled catchweight bouts (agreed in advance, no miss): rate normally.

### 5.4 Short-notice fights
No adjustment. Accepted as noise (documented decision).

### 5.5 Age
No adjustment in v1 (documented decision; revisit in v2).

---

## 6. Update algorithm (per fight)

Fighters A and B, processed simultaneously and symmetrically. Steps for fighter A (mirror for B):

**Step 0 — Convert and inflate.**
```
μ  = (r_A − 1500) / 173.7178
φ  = RD_A / 173.7178
φ  = min( sqrt(φ² + σ_A² · n_A),  350/173.7178 )      # time inflation, §3
μ_j = (r_B − 1500) / 173.7178
φ_j = min( sqrt((RD_B/173.7178)² + σ_B² · n_B),  350/173.7178 )
```
Both fighters' updates use the *pre-fight, time-inflated* values of the opponent. Compute both updates from the same pre-fight snapshot; never sequentially.

**Step 1 — Opponent discount and expected score.**
```
g(φ_j) = 1 / sqrt(1 + 3·φ_j²/π²)
E = 1 / (1 + exp(−g(φ_j) · (μ − μ_j)))
```
g(φ_j) < 1 shrinks the impact of uncertain opponents: results against high-RD debutants carry less evidence, in both directions. (This is the mechanism that keeps a veteran's loss to an unknown from cratering them, and keeps padding a record with debutants from inflating anyone.)

**Step 2 — Estimated variance and improvement.**
With a single opponent per period, the sums collapse to one term:
```
v = 1 / ( g(φ_j)² · E · (1 − E) )
Δ = v · g(φ_j) · (S − E)
```
S is the graded score from §4, after §5.3 adjustments to S (offender discount, victim loss floor).

**Step 3 — New volatility σ'.**
Solve for σ' via Glickman's iterative procedure (Illinois algorithm) with system constant **τ = 0.5**:

Define `f(x) = [ e^x (Δ² − φ² − v − e^x) ] / [ 2(φ² + v + e^x)² ]  −  (x − ln σ²)/τ²`

Iterate per the Glicko-2 paper (convergence tolerance ε = 0.000001):
- a = ln σ²
- If Δ² > φ² + v: b = ln(Δ² − φ² − v); else step k downward until f(a − kτ) < 0, b = a − kτ
- Illinois-method regula falsi on [a, b] until |b − a| ≤ ε
- σ' = e^(a/2)

**Step 4 — New deviation and rating.**
```
φ* = sqrt(φ² + σ'²)
φ' = 1 / sqrt( 1/φ*² + 1/v )
μ' = μ + φ'² · g(φ_j) · (S − E)
```

**Step 4b — Missed-weight victim win bonus (§5.3):** if A is the victim and won:
```
μ' = μ + β_v · (μ' − μ)          # β_v = 1.10, scales the rating change only
```
φ' and σ' are untouched.

**Step 5 — Convert back, persist.**
```
r' = 173.7178·μ' + 1500 ;  RD' = 173.7178·φ' ;  σ_A = σ'
last_fight_date = fight_date ; fight_count += 1
```

---

## 7. Parameter table

| Parameter | Value | Status |
|---|---|---|
| Initial rating r₀ | 1500 | Fixed |
| Initial RD₀ | 350 | Fixed (also the RD cap) |
| Initial volatility σ₀ | **0.06 (effective)** | Tuned value 0.05 was never applied: `engine.SIGMA0` is set after import, but `Fighter.sigma` defaults to the value bound at class creation (0.06). The §11 validation results were produced at 0.06. Sensitivity to σ₀ in the 0.05–0.06 range is flat (Δ log-loss 0.0001 on 2020+). |
| System constant τ | 0.5 → **0.2 (tuned)** | Tuned (flat across 0.05–0.2; volatility machinery contributes little at MMA sample sizes) |
| Standard downtime | 1 year | Fixed (design decision) |
| S: unanimous dec | 0.90 → **0.80 (tuned)** | Tuned |
| S: split dec | 0.75 → **0.55 (tuned)** | Tuned — split decisions carry almost no skill signal |
| S: majority draw | 0.55 / 0.45 | Tunable |
| S: DQ | 0.75 / 0.25 | Tunable |
| Missed-weight offender win discount | 0.80 | Tunable |
| Missed-weight victim win bonus β_v | 1.10 | Tunable |
| Missed-weight victim loss floor | 0.10 | Tunable |
| Convergence tolerance ε | 1e-6 | Fixed |

---

## 8. Engine requirements

1. **Deterministic and replayable.** Full-history recompute from UFC 1 must produce identical output given identical inputs. All special-case handling is data-driven (flags on the fight record), never manual edits to ratings.
2. **Chronological processing.** Sort by event date, then bout order. Same-card fights involving the same fighter (rare, historical one-night tournaments) process in bout order with intra-card n = 0.
3. **Snapshot semantics.** Both fighters in a bout update from the same pre-fight state snapshot.
4. **Point-in-time queries.** The store must answer "rating of fighter X on date D" — required for backtesting and for any prediction product.
5. **Fight record schema (minimum):** event_date, bout_order, fighter_A_id, fighter_B_id, winner_id (nullable), method (KO/TKO, SUB, UD, MD, SD, TD, DQ, NC, Draw-U, Draw-S, Draw-M), missed_weight_flag per fighter, overturned_flag, weight_class (stored now, used in v2).

---

## 9. Validation protocol (backtest)

The credibility artifact. Run before any tuning claims:

1. **Train/test split by time.** Fit tunable parameters on fights through 2019; evaluate on 2020–present. No peeking.
2. **Primary metric: log-loss** of predicted win probability vs. outcome (draws scored at 0.5). Secondary: Brier score, calibration curve (predicted 70% favorites should win ~70% of the time), and straight-up accuracy.
3. **Baselines to beat, in ascending difficulty:**
   - Coin flip (log-loss 0.693)
   - Vanilla Elo, K = 32
   - Betting closing line implied probability (vig-removed) — the market. Beating this is not required for v1 credibility; getting within a few points of it is.
4. **Prediction rule:** pre-fight, time-inflated ratings only. E from Step 1 is the win probability (fold draw mass proportionally or ignore; document choice).
5. **Parameter search:** grid or Bayesian optimization over the tunable set in §7, objective = train-set log-loss, report test-set only once.

---

## 10. Deferred to v2 (explicitly out of scope)

- Weight-class leaderboard views (filters over the global pool; cross-division movers carry r/RD/σ)
- Age curves
- Non-UFC data (regional records as priors for debut seeding)
- Short-notice covariate
- Round/dominance granularity (e.g., scoring rounds won)

---

## 11. Implementation notes and v1 results (July 2026)

**Data:** 8,794 UFC fights (UFC 2, Mar 1994 → Jul 2026) scraped from UFC Stats
(anti-bot SHA-256 proof-of-work solved programmatically). Missed-weight flags
mined from Wikipedia event articles (558 events 2013+, 206 fighter-incidents,
200 flagged fights). Closing odds joined from a public dataset (2,300 matched
fights 2020+, ~87% coverage).

**Deviations from spec as written:**
- Majority draws score 0.50/0.50, not 0.55/0.45 — UFC Stats does not record
  which fighter the dissenting judge favored. Requires MMA Decisions scorecard
  data; deferred.
- DQ wins score as split-decision equivalent (tracks the tuned split value).
- No-contest updates the activity clock (last_fight_date) but not ratings.
- The missed-weight victim win bonus multiplies the rating change (Δμ), not S,
  because S is capped at 1.0 for finishes.
- Volatility solver carries iteration caps (k-search ≤ 200, Illinois ≤ 100)
  as numerical safety rails.
- σ₀ ran at 0.06, not the tuned 0.05. `engine.SIGMA0 = 0.05` is assigned after
  import (features.py, card_report.py, and the v1 tuner), but `Fighter.sigma`
  defaults to the SIGMA0 value bound at class creation, so the assignment never
  took effect; the tuner's σ₀ grid was therefore a no-op too. Every frozen
  rating and every result in this section was produced at 0.06.
  `best_params.json` now records 0.06. Sensitivity in the 0.05–0.06 range is
  flat (Δ log-loss +0.0001 on 2020+, +0.0001 on 2025+, 0.06 marginally better).

**Tuning (per §9):** three grid rounds, selection on 2014–2019 train log-loss
only. Locked: τ = 0.2, σ₀ = 0.05, S(U-DEC) = 0.80, S(S-DEC) = 0.55. Round 3
attempted S(S-DEC) < 0.5 (winner scored below loser) — rejected as noise-floor
overfitting.

**Validation:**

| Model | 2020+ test (n=2,646) | 2025+ holdout (n=656) |
|---|---|---|
| Coin flip | 0.6931 | 0.6931 |
| Vanilla Elo K=32 | 0.6845 | 0.6816 |
| Glicko-2 priors | 0.6839 | 0.6791 |
| Glicko-2 tuned | 0.6809 | 0.6738 |
| Closing line (vig-free) | 0.6079 (n=2,300 matched) | — |

Ordering is identical on the near-uncontaminated 2025+ holdout; tuning
generalized. The market retains a large edge; a blend test finds optimal
weight = 100% market — the results-only model currently carries no incremental
signal over closing lines. Calibration is good but compressed (most predictions
in the 40–60% band): the model under-separates fighters relative to the market,
which prices fight-level information (injuries, camps, style matchups) that a
results-only system cannot see. Beating or supplementing the market requires
fight-level features (v2).

**Missed-weight treatment:** prediction-neutral (Δ log-loss < 0.0001 on 200
affected fights). Retained as a fairness/credibility feature.
