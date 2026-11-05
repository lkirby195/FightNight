# site/ — the public site generator

`site/build.py` renders the public record (rankings, this week's card, our record)
as static HTML into `site/out/`; `/` is the rankings page and This week lives at
`/this-week/`. The How it works page was retired at launch (2026-11-05);
`out/vercel.json` redirects its old URL to `/`. It is a read-only consumer of the
repo's data files and never touches the model, the ledger, or any script outside
`site/`. `site/out/` is committed on every build: the commit is the public proof
that nothing changed after a fight.

## Weekly cadence (automated)

`.github/workflows/cadence.yml` runs `cadence.py` on a schedule and pushes what
it changed to `main`; Vercel deploys `main`. Nobody has to be at a keyboard.

| When (Mountain) | Command | What it does |
|---|---|---|
| Thu 09:23 | `cadence.py report` | next card from UFC Stats; its BFO page(s); bouts BFO prices that are not on the UFC Stats card kept out; gate: 80% of the card priced; `card_report.py` capture; build |
| Fri 09:23 | `cadence.py report --final` | fallback: does nothing if Thursday posted; a card still not postable is a failure |
| Sun 09:23 | `cadence.py settle` | `scrape_v2.py new`, `features.py`, `scrape_bfo.py`, `card_settle.py` (clean closes), `clv_eval.py`, `prefight_rd.py`, `betting_system.py all --write-ledger`, `replay_check.py` (gate); build |
| Mon 09:23 | `cadence.py settle --final` | fallback: does nothing if Sunday settled; results still missing is a failure |

Every run is safe to repeat: a card already captured is never re-posted
(`placeable_lines.csv` is append-only and the first capture is the price of
record), and a settle with no new card on UFC Stats changes nothing. A failure
pushes nothing (the site keeps the last good build) and opens a "Cadence
failed" issue that mentions the repo owner. The job summary of every run shows
the bout table. Run any command by hand from the Actions tab (Run workflow;
`dry_run` checks the card and the gate and writes nothing), or locally:

    python cadence.py report --dry-run
    python cadence.py report
    python cadence.py settle

What used to be done by hand and is now automatic:

- **Which card and slug.** The next card on the UFC Stats upcoming list; the BFO
  pages dated within a day of it. BFO often splits one card over two pages (UFC
  Vegas 124: main card on a page dated Sunday, the rest on one dated Saturday);
  `card_report.py` and `card_settle.py` take a comma-separated slug list and
  treat it as one card, projected in one Stage 2 frame.
- **OFF_CARD.** UFC Stats is the authority on what is on a card. A BFO bout
  that matches no UFC Stats bout (a withdrawal, a phantom listing) or repeats one
  already matched goes to `--off` for that run. `OFF_CARD` in `card_report.py`
  still works for a manual override.
- **LIVE_CARDS.** Not updated any more; it decides nothing (liveness is the
  capture).
- **Cache busting.** CI starts with empty caches; locally, `cadence.py settle`
  drops the pre-fight copies of the settled card's BFO page and series first.
  `scrape_v2.py new` reads the completed-events list and new event pages live
  and appends only what is new.
- **Line endings.** Generated files keep the line endings of their committed
  copies, so a Linux run does not rewrite every line of a CRLF file.

The manual commands still work, for a re-run or a one-off:

    python card_report.py <bfo-slug>[,<slug>] <YYYY-MM-DD> [--off MU,...] && python site/build.py
    python card_settle.py <bfo-slug>[,<slug>] <YYYY-MM-DD> [--off MU,...] && python site/build.py

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
- `out/record/ledger.csv`: public ledger (every 2026 row and the 2025 backtest,
  without the `units` and `pnl_at_open` columns; `live` = 1 marks the rows with a
  pre-event capture).

## Conventions

1 unit = $100. Model probabilities are whole-number percentages; market
probabilities are the vig-removed pair normalised to 100. "The odds said" is
the latest capture for the upcoming card and the earliest capture (else the
opening line) for results. Model picks between 48% and 52% show as a coin flip
and are not counted; draws and no contests are shown, not counted, and void
any bet. No jargon on the default view; the "Show the numbers" toggle adds
model line, open, placeable, close, CLV, rule, stake and RD columns.

## Deploy

The cadence deploys by pushing to `main`. `site/out/` is a static Vercel project: `fightnight-out` on team
`logankirby14-8806s-projects`, git-connected to `lkirby195/FightNight` with root
directory `site/out`, no build step, no framework, production at
https://fightnight-out.vercel.app. Every push to `main` deploys production and
every branch push deploys a preview (branch alias
`fightnight-out-git-<branch>-logankirby14-8806s-projects.vercel.app`, behind
Vercel Authentication: team members only). To deploy a local build that has not
been pushed, use the Vercel CLI (`npm i -g vercel`, once):

    vercel deploy site/out --prod --yes --token "$VERCEL_TOKEN" --scope logankirby14-8806s-projects

`--yes` skips every prompt and links the directory to the project on first
use; `--prod` promotes the deployment to the production URL (drop it for a
preview URL). `VERCEL_TOKEN` is a personal token from the Vercel dashboard;
never commit it. Web Analytics is Vercel's cookieless kind (the
`/_vercel/insights/script.js` tag in `base.html`), so no cookie banner is
needed; it must be switched on once for the project in the dashboard.

Custom domain: none attached. `splitdecision.io` is not purchased.
