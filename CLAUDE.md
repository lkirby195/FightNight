# CLAUDE.md — FightNight (Split Decision)

## 1. What this is
A UFC Glicko-2 rating engine (`engine.py`, spec `mma-glicko2-spec.md`), a two-stage
prediction model (Stage 2 logistic regression on the Glicko spine + pre-fight features),
a frozen betting system with a public ledger, and a static site generator (`site/`).
Brand: Split Decision, domain splitdecision.io (not yet purchased). Logan is not anonymous;
the repo stays private and his name may appear publicly. Audience product, not revenue:
picks are published as a transparency record, never sold.

## 2. Repo map
Scripts at the root (run from the root with the repo's Python; see §8):
- `session.py` — UFC Stats HTTP session with the SHA-256 proof-of-work solver (`v1/session.py` is an identical copy).
- `scrape_v2.py` — UFC Stats: pass A event tables -> `fights_v2.csv`; B fighter pages -> `fighters_v2.csv`; C fight details -> `fight_details_v2.csv`. Cached under `cache/`.
- `scrape_bfo.py` — BestFightOdds: sitemap -> event pages (JSON-LD date) -> `/api/ggd` tick series (base64 + ROT47) -> `bfo_lines.csv` (open/close per bout). Cached under `cache_bfo/`.
- `features.py` — Stage 1 Glicko replay + leakage-safe rolling features -> `features_v2.csv`.
- `stage2.py` — antisymmetric feature matrix (paired differences, mirror augmentation).
- `walkforward.py` — Stage 2 vs recalibrated closing market, walk-forward 2012-2024 -> `walkforward.csv`.
- `clv_eval.py` — model-only probabilities vs BFO opening lines; regenerates `model_only_preds.csv`, `bfo_joined.csv`, `clv_eval_full.csv` on every run.
- `prefight_rd.py` — pre-fight time-inflated RD of both fighters per bout -> `prefight_rd.csv` (placement-policy input).
- `betting_system.py` — the frozen signal rules + placement policy v1; `--write-ledger` writes `system_ledger.csv`.
- `card_report.py` — one event: projections, open/close, CLV, results; for a future event captures the current line into `placeable_lines.csv` (append-only) and prints the rule-fired picks with stakes. `OFF_CARD` drops withdrawn bouts.
- `card_settle.py` — `card_report` with in-play-safe closes (paired two-sided overround filter); use it for any event < ~4 months old.
- `names.py` — one name resolver for BFO vs UFC Stats spellings (`data/name_aliases.csv`, suffix stripping, fuzzy fallback).
- `replay_check.py` — the merge gate: replays `fights_v2.csv` through the engine and compares to the frozen `ratings.csv`.
- `engine.py` — the Glicko-2 engine (frozen v1.0 math); `best_params.json` — tuned parameters applied at runtime by `replay_check.apply_params`.
- `site/build.py` — renders `site/out/` (rankings, this week, record, how it works) from the data files only; `site/backtest_2025.py` — the sealed one-shot (never re-run); `site/division_overrides.csv`, `site/state/ranks_prev.json`, `site/templates/`, `site/static/`.
- `tests/` — pytest (`python -m pytest`, 19 tests on main). `v1/` — legacy v1 scripts, kept for the record, not runnable as-is (`MISSING.md`). `fightnight-card-page.html` — a static design mockup (UFC 330). `README-v1.md`, `RESULTS-v2.md` — history.

Data files (`data/`) and what regenerates them:
- `fights_v2.csv`, `fighters_v2.csv`, `fight_details_v2.csv` — `scrape_v2.py A/B/C`. `features_v2.csv` — `features.py`.
- `bfo_lines.csv` — `scrape_bfo.py` (incremental by matchup id). `bfo_joined.csv`, `model_only_preds.csv`, `clv_eval_full.csv` — `clv_eval.py`. `walkforward.csv` — `walkforward.py`. `prefight_rd.csv` — `prefight_rd.py`.
- `system_ledger.csv` — `betting_system.py all --write-ledger` only. `placeable_lines.csv` — appended by `card_report.py` for future events only.
- `ratings.csv` — frozen v1 ratings (through 2026-07-25), the replay_check reference. `name_aliases.csv` — hand-maintained (one row per BFO alias).
- SEALED, never regenerated: `system_ledger_2025_backtest.csv`, `model_only_preds_2025_backtest.csv` (2025 tab, display only), `model_only_preds_frozen_2026-09-08.csv` (source of record for the historical sim figures). Legacy, kept: `fights.csv`, `market.csv`, `legacy_favtier_ledger.csv`, `sim26_bets.csv`.
- Run order for a new card: `scrape_v2.py A` -> `B` -> `C` -> `features.py` -> `scrape_bfo.py 2010 2027` -> `clv_eval.py` -> `prefight_rd.py` -> `betting_system.py all --write-ledger` -> `replay_check.py`.

## 3. State (audited 2026-10-15, main 4bef199)
- Data: `fights_v2.csv` 8,897 bouts, 789 events, 2,757 fighters, 1994-03-11 to 2026-09-19 (UFC 331). `fight_details_v2` 8,876; `features_v2` 8,805; `bfo_lines` 7,946 bouts (2010-); `bfo_joined` 7,289; `model_only_preds` 6,433; `system_ledger` 537 rows, 29 live; `placeable_lines` 41 captures (since 2026-09-08).
- Engine (spec §7/§11): tau = 0.2, S(U-DEC) = 0.80, S(S-DEC) = 0.55 (DQ scored as split, M-DEC as unanimous). sigma0: tuned 0.05 but effectively 0.06 — `engine.SIGMA0 = x` after import is a no-op because `Fighter.sigma` binds the dataclass default at class creation; every frozen rating was produced at 0.06, `best_params.json` records 0.06. Documented, not corrected.
- Stage 2: logistic regression (C = 0.3, no intercept, StandardScaler without centring, mirror augmentation) on Glicko logit, RD terms, debut/experience/layoff and paired career differences; Pass C stats (`PASSC`: accuracy/defence/control/pace/target shares) are excluded from the live model. Walk-forward: fit on fights before the year being predicted; the live card model is trained on fights before 2026-01-01.
- Betting constants (code is the truth): signal rules frozen 2026-09-08 — FLIP: model picks the market underdog with `FLIP_CONF = 0.65`, 1u flat; GAP: model agrees with the favourite and prices it `GAP_PTS = 100` American points stronger, 1u flat; pre-filter: opening implied-probability sum in [1.00, 1.12]. Placement policy v1 frozen 2026-09-09 — skip if the bet side is longer than `SKIP_LINE_MAX = 250`; skip if both fighters' RD > `UNKNOWN_RD = 160`; else 1u up to a backed-side RD of `RAMP_LO = 130`, falling linearly to 0.5u at `RAMP_HI = 200`, floored at 0.5u.
- Live record: every card 2026-01-01..2026-07-25 plus `LIVE_CARDS` = {2026-08-22, 2026-08-29, 2026-09-05, 2026-09-12, 2026-09-19}. Live bets settle at the earliest `placeable_lines.csv` capture, else the open.
- Site: built and reproducible (`site/out/` committed, second build byte-identical), not deployed.
- Pending (PR #6, `feat/oct-pass`, open): UFC Vegas 121 settled, UFC 332 captured, validated opening line (`open_suspect`, VOID rows), live-flag debut and on-record rules, `polymarket.py`. Until it merges, nothing in that list is on main.

## 4. Hard rules
- `python replay_check.py` is the merge gate: 0 missing fighters, max |rating delta| <= 0.05. Run it before every commit that touches data.
- Every change goes on a feature branch, through a PR, verified from a clean clone. Engine changes require a spec update in the same PR.
- Validation discipline: train <= 2022, tune 2023-24, 2025 is a sealed one-shot (spent only by `site/backtest_2025.py`), 2026 is live. Never select on the holdout; flag any contamination explicitly.
- Base Glicko-2 ratings never contain prediction-only machinery (Stage 2 lives on top of the frozen spine).
- Frozen rules (signal constants, placement policy) do not change without explicit review and a new documented freeze date in `BETTING-SYSTEM.md`.
- The ledger is written only by `betting_system.py --write-ledger`; never hand-edit it or `placeable_lines.csv` (append-only; captures are evidence).
- Fail loud: a BFO parse failure or a fighter-name mismatch means no picks are posted, never a silent wrong pick (`card_report` prints `debut-no proj` on a miss — grep `fighters_v2.csv` for the surname before believing it).

## 5. Scraping
- UFC Stats serves a proof-of-work interstitial; `session.py` solves it (sha256 of `nonce:n` with leading zeros, POST to `/__c`). Pass A will not see a new event until the completed-events index cache is deleted (`cache/<sha1 of "/statistics/events/completed?page=all">.html`).
- BFO: `sitemap-events.xml` -> `/events/<slug>` (date from the page's JSON-LD `startDate`, never the sitemap lastmod) -> `/api/ggd?m=<mu>&p=1|2` decoded base64 + ROT47. Alias slugs serving the same page dedupe to the sort-first slug by matchup id. There is no `/search?query=` fallback in the code; `card_report`/`card_settle` take the slug as an argument. An event page is only re-fetched when `card_report` runs for a future event (`refresh=True`).
- `card_settle.py`, not `card_report.py`, for events < ~4 months old: BFO keeps in-play ticks, and the paired two-sided overround filter [1.00, 1.15] (`OR_LO`, `OR_HI`) drops them.
- Polymarket (Gamma API) is the approved vig-free secondary price source, never a signal source (`polymarket.py`, PR #6).
- Never hammer a source: built-in 0.45 s spacing, retry backoff, disk caches (`cache/`, `cache_bfo/`); do not remove them.

## 6. Weekly cadence
- Thursday (lines up): `python card_report.py <bfo-slug> <YYYY-MM-DD> && python site/build.py`. The run appends the current lines to `placeable_lines.csv` (append-only; the earliest capture is the price of record). Add the date to `LIVE_CARDS` in `betting_system.py` whenever a card's report is run pre-event — the code's convention; no-bet cards are listed too, so the live record includes them (0 bets). Commit the capture, `LIVE_CARDS`, `site/out/`, `site/state/`.
- Sunday (results in): the §2 run order (`scrape_v2.py A/B/C`, `features.py`, `scrape_bfo.py 2010 2027`, `clv_eval.py`, `prefight_rd.py`, `betting_system.py all --write-ledger`, `replay_check.py`), then `python card_settle.py <bfo-slug> <YYYY-MM-DD> && python site/build.py`. Commit the data with `site/out/` and `site/state/`.
- Bust the BFO page caches for a card scraped pre-event before settling it, or its closes stay pre-event.

## 7. Open launch work (in order)
1. Remove the name lint from `site/build.py` (`LINT`; separate deploy PR).
2. Paste the how-it-works copy and a one-line disclaimer into `site/templates/how_it_works.html` (the current text is a draft placeholder).
3. Fill "Show the numbers" for non-bet bouts from `bfo_lines.csv`.
4. Populate `site/division_overrides.csv`.
5. Deploy `site/out/` to Vercel (static project, root `site/out`, no build step).
6. Then: `card.py next` / `card.py settle` automation (does not exist yet) and a GitHub Action (Thu/Sun 09:00 MT) with `replay_check.py` as a required CI gate.

## 8. Environment quirks
- bash associative arrays are unreliable here; use a Python dict, and `shutil.move` through `/tmp/stage` for file operations. `pip` needs `--break-system-packages`.
- On Logan's Windows machine: `python` on PATH is the Store stub; use `C:/Users/LoganKirby/anaconda3/python.exe` (3.13; pandas, scikit-learn, requests, bs4, lxml, jinja2 installed). Working copies are CRLF (`core.autocrlf=true`), the index is LF except `data/bfo_lines.csv` (CRLF): rewrite that file byte-wise on `\r\n`.
- Console output is ASCII-only. A full BFO re-scrape walks ~800 event pages and two tick series per bout; run it in the background and do not touch `bfo_lines.csv` meanwhile.
