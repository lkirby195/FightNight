# FightNight

UFC rating engine + fight prediction system. Glicko-2 spine, two-stage
prediction model, market benchmarking, and closing-line-value (CLV) evaluation
against BestFightOdds opening lines.

## Headline results

**The model moves betting lines.** Pure model picks taken at the opening line
show +1.9 to +2.0 pts of vig-free CLV in every era since 2018 (t ≈ 9–11,
n = 5,982 out-of-sample fights). Each 10 pts of model-vs-opener disagreement
predicts ~2.2 pts of line movement toward the model by close. Regime break at
2018 coincides with PASPA repeal.

**Deployable blend** (anchor on open, tilt by model, weights fit 2022–23,
validated on 2024+2026): ~190 bets/yr, +4.95 pts CLV/bet (t = 9.6),
ROI +5.4% (t = 1.2 — positive expectation, slow to prove).

**vs the closing line**: Stage 2 closes ~45% of the Glicko-to-market log-loss
gap; incremental gain over the recalibrated close is +0.00075 pooled
(t = 1.68, n = 5,466) — real-looking, below significance, era-dependent.

Full analysis: `RESULTS-v2.md`. Rating-engine math: `mma-glicko2-spec.md`.
Betting system: `BETTING-SYSTEM.md` (KISS rule, validated 2012–2026, t=2.43).

## Pipeline

```
pipeline/session.py     UFC Stats HTTP session (SHA-256 proof-of-work solver)
pipeline/scrape_v2.py   A: event tables -> fights_v2.csv (per-fight KD/str/TD/sub)
                        B: fighter pages -> fighters_v2.csv (DOB/height/reach/stance)
                        C: fight details -> fight_details_v2.csv (attempts, control,
                           target/position splits; corner-order fixed by name key)
pipeline/scrape_bfo.py  BestFightOdds line histories -> bfo_lines.csv
                        (sitemap -> event pages -> /api/ggd, base64+ROT47 decoded;
                         first tick = open, last tick = close)
pipeline/features.py    Stage 1 Glicko replay + leakage-safe rolling features
                        -> features_v2.csv
pipeline/stage2.py      Antisymmetric feature matrix (paired diffs, mirror aug)
pipeline/walkforward.py Walk-forward eval vs recalibrated closing market
pipeline/clv_eval.py    CLV eval vs opening lines (model-only probs; see the
                        circularity warning in the docstring)
pipeline/card_report.py Per-event report: model %, fair line, open/close, CLV,
                        results. Usage: card_report.py <bfo-slug> <YYYY-MM-DD>
pipeline/betting_system.py  KISS betting rule (flip>=65% + gap>=100pts, flat 1u).
                        Usage: betting_system.py [year|all]
```

Run order: `scrape_v2.py A` -> `B` -> `C` -> `scrape_bfo.py 2010 2027` ->
`features.py` -> `walkforward.py` -> `clv_eval.py`.

All scrapers are disk-cached (`cache/`, `cache_bfo/`) and resumable; re-runs
cost zero network requests. Be polite: built-in delays, do not remove.

## Data

```
data/fights_v2.csv          8,794 bouts 1994-2026, chronological bout order,
                            missed-weight flags (206 incidents)
data/fighters_v2.csv        2,722 fighters
data/fight_details_v2.csv   8,773 bouts: attempts, control time, strike splits
data/features_v2.csv        model-ready pre-fight features (leakage-safe)
data/market.csv             vig-free closing probs (public odds dataset join)
data/bfo_lines.csv          7,962 bouts: opening + closing mean lines 2010-2026
data/model_only_preds.csv   walk-forward model-only probs (no market input)
data/clv_eval_full.csv      per-fight CLV records
data/walkforward.csv        per-fight loss diffs vs recalibrated close
```

## Validation discipline

- Stage 2: train ≤2022, tune 2023–24, holdout 2025+ — **2025 remains sealed**
  (2026 was spent as a test window; documented in RESULTS-v2.md).
- Walk-forward: every prediction from a model trained strictly on prior fights.
- CLV: model-only probabilities (stacked probs contain the close — circular).
- Engine replay reproduces the frozen v1 ratings bit-for-bit (max Δ 0.05).

## v1 engine

`engine.py` + `best_params.json` (tau=0.2, sigma0=0.05, S_udec=0.80,
S_sdec=0.55). Key finding: split decisions carry almost no skill signal.
