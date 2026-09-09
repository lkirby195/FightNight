# site/ — the public site generator

`site/build.py` renders the public record (rankings, this week's card, our record,
how it works) as static HTML into `site/out/`. It is a read-only consumer of the
repo's data files and never touches the model, the ledger, or any script outside
`site/`. `site/out/` is committed on every build: the commit is the public proof
that nothing changed after a fight.

## Weekly cadence (two commands each)

Thursday, once the card's lines are up:

    python card_report.py <bfo-slug> <YYYY-MM-DD> && python site/build.py

`card_report.py` records the price on the board in `data/placeable_lines.csv`
(the timestamp is the "Picks posted" line); the build projects the same bouts
with the same code and renders This week. Add the card's date to `LIVE_CARDS` in
`betting_system.py` in the same commit.

Sunday, after the results are scraped (`scrape_v2.py`, `features.py`,
`scrape_bfo.py`, `clv_eval.py`, `prefight_rd.py`, `betting_system.py all
--write-ledger`, `replay_check.py`; see the root README):

    python card_settle.py <bfo-slug> <YYYY-MM-DD> && python site/build.py

Commit `site/out/` and `site/state/ranks_prev.json` with the data.

## Options

    python site/build.py [--as-of YYYY-MM-DD] [--out DIR]

`--as-of` sets the build date (rankings eligibility window, which event counts
as "this week"); it defaults to today.

## Files

- `templates/` Jinja2 templates; `base.html` carries the nav, the wordmark
  (`BRAND` in `build.py`, placeholder `SPLIT DECISION`) and the few lines of
  vanilla JS (year tabs, division pills, card dropdown, "Show the numbers").
- `static/style.css` copied to `out/style.css`.
- `division_overrides.csv` (`fighter_id,division`): overrides the default
  division (modal weight class of the last three fights).
- `state/ranks_prev.json`: rank snapshot behind the "Move" column. It rotates
  only when the data moves (new results), so re-building within a week keeps the
  same comparison.
- `backtest_2025.py`: the sealed one-shot that wrote
  `data/system_ledger_2025_backtest.csv` and
  `data/model_only_preds_2025_backtest.csv` (the 2025 tab). Never re-run.
- `out/record/ledger.csv`: public ledger (2026 live rows and the 2025 backtest,
  without the `live`, `units`, `pnl_at_open` columns).

## Conventions

1 unit = $100. Model probabilities are whole-number percentages; market
probabilities are the vig-removed pair normalised to 100. "The odds said" is
the latest capture for the upcoming card and the earliest capture (else the
opening line) for results. Model picks between 48% and 52% show as a coin flip
and are not counted; draws and no contests are shown, not counted, and void
any bet. No jargon on the default view; the "Show the numbers" toggle adds
model line, open, placeable, close, CLV, rule, stake and RD columns.

## Anonymity lint

The build fails if any output file contains `lkirby`, `Logan`, or a
`github.com/lkirby195` URL.

## Deploy

`site/out/` deploys to Vercel as a static project (root directory `site/out`,
no build step). Not deployed yet: deployment is gated on the anonymity
checklist (brand GitHub org, WHOIS privacy, brand email).
