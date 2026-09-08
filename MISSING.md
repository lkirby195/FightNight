# Missing files

The two "Add files via upload" commits (93fa378, b3d58d7) scrambled filenames and
dropped several files outright. After the rename manifest (commit "Fix scrambled
filenames from upload"), the following are still absent from the repo and are
**not recoverable from git history**.

All of them are v1-era and are **not needed for the v2 pipeline**
(`scrape_v2.py` -> `features.py` -> `stage2.py` / `betting_system.py` / `card_settle.py`).

| Expected path | What it was | Status |
|---|---|---|
| `v1/scrape.py` | v1 UFC Stats scraper (referenced by `README-v1.md`) | missing |
| `v1/session.py` | v1 HTTP session with SHA-256 proof-of-work | identical to root `session.py`; copied into `v1/` |
| `data/mw_cache.json` | v1 missed-weight cache built by `v1/mw_chunked.py` (Wikipedia crawler) | missing |
| `data/preds.csv` | v1 replay predictions (output of `v1/run.py` / `v1/final_eval.py`) | missing |
| `data/odds-raw.csv` | v1 historical closing-odds input to `v1/market.py` | missing |
| `BETTING-SYSTEM.md` | Betting-rule doc referenced by `README.md` | recreated 2026-09-08 (the uploaded file under that name was actually `.gitignore`) |

Note: the v1 scripts in `v1/` carry an `os.chdir(_ROOT/data)` + `sys.path.insert(_ROOT/src)`
preamble from the original v1 layout; they are kept for the record and are not
expected to run as-is.

`engine.py` (frozen v1.0 Glicko-2 engine) and `card_settle.py` (in-play tick filter)
were hand-added in commit 422687a and are correct as committed.
