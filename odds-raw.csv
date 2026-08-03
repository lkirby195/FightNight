# MMA Glicko-2 Rating Engine

Elo-style (Glicko-2) ratings for the UFC. Full pipeline: scrape -> rate -> tune -> benchmark.

## Pipeline
```
python3 pipeline/scrape.py        # UFC Stats -> fights.csv (solves their PoW anti-bot check)
python3 pipeline/mw_chunked.py    # Wikipedia -> missed-weight flags (run repeatedly; disk-cached)
python3 pipeline/run.py           # full-history replay -> leaderboard + ratings.csv (uses best_params.json)
python3 pipeline/final_eval.py    # tuned replay -> ratings.csv + preds.csv
python3 pipeline/market.py        # preds.csv vs closing odds benchmark + blend test
python3 pipeline/tune.py          # parameter grid search (train 2014-2019, report test once)
```

## Files
- `mma-glicko2-spec.md` — authoritative math spec (v1.1, includes tuned params + results)
- `engine.py` — Glicko-2 core: time-based RD inflation (standard downtime 1yr),
  graded finish scores, missed-weight asymmetry, snapshot semantics
- `session.py` — UFC Stats HTTP session with SHA-256 proof-of-work solver
- `fights.csv` — 8,794 fights, 1994-2026, with missed-weight flags
- `ratings.csv` — all 2,722 fighters: rating, RD, sigma, record
- `preds.csv` — pre-fight win probabilities for every fight (backtest artifact)
- `best_params.json` — tuned parameters (tau=0.2, sigma0=0.05, S_udec=0.80, S_sdec=0.55)
- `odds-raw.csv` — public closing-odds dataset (2010-2026), for market.py

## Headline results (2025+ holdout, n=656)
Tuned Glicko-2 log-loss 0.6738 vs vanilla Elo 0.6816 vs coin flip 0.6931.
Market closing line: 0.608 (2020+ matched set). Blend test: model adds no
incremental signal over the market yet — see spec section 11 for the v2 path.

Key empirical finding: split decisions carry almost no skill signal
(tuned S = 0.55, barely above a draw); finishes carry most of the information.
