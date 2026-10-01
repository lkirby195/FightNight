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
Betting system: `BETTING-SYSTEM.md` (KISS rule, validated 2012–2026, t=2.42).

## Pipeline

```
pipeline/session.py     UFC Stats HTTP session (SHA-256 proof-of-work solver)
pipeline/scrape_v2.py   A: event tables -> fights_v2.csv (per-fight KD/str/TD/sub)
                        B: fighter pages -> fighters_v2.csv (DOB/height/reach/stance)
                        C: fight details -> fight_details_v2.csv (attempts, control,
                           target/position splits; corner-order fixed by name key)
pipeline/scrape_bfo.py  BestFightOdds line histories -> bfo_lines.csv
                        (sitemap -> event pages -> /api/ggd, base64+ROT47 decoded;
                         validated open = first paired tick unless corrected
                         within the hour (open_suspect), last tick = close;
                         --rebuild rewrites every row from the cached series)
pipeline/features.py    Stage 1 Glicko replay + leakage-safe rolling features
                        -> features_v2.csv
pipeline/stage2.py      Antisymmetric feature matrix (paired diffs, mirror aug)
pipeline/walkforward.py Walk-forward eval vs recalibrated closing market
pipeline/clv_eval.py    CLV eval vs opening lines (model-only probs; see the
                        circularity warning in the docstring)
pipeline/card_report.py Per-event report: model %, fair line, open/close, CLV,
                        results. Usage: card_report.py <bfo-slug> <YYYY-MM-DD>
                        For an event dated today or later it also records the
                        current line of every bout in placeable_lines.csv.
                        Lists the rule-fired picks with their placement stake.
                        Bouts BFO still lists that are off the live card
                        (withdrawals) go in OFF_CARD, keyed by (event date,
                        matchup id), and are dropped before the table, the
                        signals and the placeable capture.
pipeline/prefight_rd.py Pre-fight RD of both fighters per bout (time-inflated,
                        public scale) -> prefight_rd.csv; placement-policy input
pipeline/betting_system.py  KISS betting rule (flip>=65% + gap>=100pts, flat 1u)
                        + placement policy v1 (RD ramp, both-unknown and
                        +250 skips). Usage: betting_system.py [year|all] [--write-ledger]
                        Signals at the validated open; VOID rows for signals
                        only a suspect opener fires; live=1 needs both
                        fighters on file before the event (debut rule).
pipeline/polymarket.py  Polymarket moneyline per bout (second price source,
                        never a signal source): PM column + capture columns
                        in card_report; `polymarket.py backfill 2026` writes
                        pm_history/ and pm_clv_2026.csv
```

Run order: `scrape_v2.py A` -> `B` -> `C` -> `scrape_bfo.py 2010 2027` ->
`features.py` -> `walkforward.py` -> `clv_eval.py` -> `prefight_rd.py`.
`python replay_check.py  # merge gate: must pass` (after any re-scrape, before committing data)

All scrapers are disk-cached (`cache/`, `cache_bfo/`, `cache_pm/`) and
resumable; re-runs cost zero network requests. Be polite: built-in delays, do
not remove.

## Data

```
data/fights_v2.csv          8,794 bouts 1994-2026, chronological bout order,
                            missed-weight flags (206 incidents)
data/fighters_v2.csv        2,722 fighters
data/fight_details_v2.csv   8,773 bouts: attempts, control time, strike splits
data/features_v2.csv        model-ready pre-fight features (leakage-safe)
data/market.csv             vig-free closing probs (public odds dataset join)
data/bfo_lines.csv          7,974 bouts: opening + closing mean lines 2010-2026;
                            the open is the validated open, with open_suspect
                            and the raw first ticks (f1_open_raw, f2_open_raw)
data/bfo_joined.csv         bfo_lines joined to fights_v2 by fight_id (a/b
                            orientation), with open_suspect and a/b_open_raw
data/model_only_preds.csv   walk-forward model-only probs (no market input)
data/clv_eval_full.csv      per-fight CLV records
data/walkforward.csv        per-fight loss diffs vs recalibrated close
data/prefight_rd.csv        pre-fight RD of both fighters per bout (time-inflated,
                            public scale) from prefight_rd.py; placement-policy input
data/placeable_lines.csv    current BFO line per bout, appended by
                            card_report.py when a future event is reported
                            (append-only: one row per bout per run, skipped when
                            the lines are unchanged); the bet side's earliest
                            capture is the ledger's placeable_line. Since
                            2026-09-30 also pm_p_self (fighter1's Polymarket
                            price), pm_liquidity, pm_slug; empty when no market
data/pm_history/<slug>.json hourly Polymarket price history of both outcomes
                            of a resolved 2026 ledger bout (polymarket.py backfill)
data/pm_clv_2026.csv        per 2026 ledger pick: Polymarket price at our capture
                            time and last pre-fight price, PM CLV next to BFO CLV
data/system_ledger.csv      ledger of record. GENERATED by
                            `betting_system.py <year|all> --write-ledger`;
                            never hand-edit it, regenerate it. Columns:
                            event_date, event, fight_id, fighter, rule, live,
                            units, rd_self, rd_opp, placed, stake_u, pnl_placed,
                            open_line, placeable_line, clean_close, clv_pts,
                            result, method, round, pnl, pnl_at_open, reason,
                            pm_p_at_capture (reason set = VOID row, not counted)
data/legacy_favtier_ledger.csv
                            abandoned favorites-tier system (3u/2u/1u), ends
                            2026-03-28 plus the Paris FLIP row; kept for the record
```

## Validation discipline

- Stage 2: train ≤2022, tune 2023–24, holdout 2025+ — **2025 remains sealed**
  (2026 was spent as a test window; documented in RESULTS-v2.md).
- Walk-forward: every prediction from a model trained strictly on prior fights.
- CLV: model-only probabilities (stacked probs contain the close — circular).
- Engine replay reproduces the frozen v1 ratings bit-for-bit (max Δ 0.05).

## v1 engine

`engine.py` + `best_params.json` (tau=0.2, sigma0=0.06, S_udec=0.80,
S_sdec=0.55). Key finding: split decisions carry almost no skill signal.
