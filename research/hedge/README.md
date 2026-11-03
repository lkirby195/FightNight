# Hedge study, 2012-2022 (read-only research)

The model's proven edge is line movement (CLV); its profit column is noisier. This study
asks: if we bet at the open and hedge at the close when the line has moved our way, do we
turn CLV into locked profit, and what does that cost in total P&L? It is an execution
study on the frozen signals. Selection does not change: the bets are the frozen GAP and
FLIP signals and placement policy v1, exactly as `betting_system` applies them. Nothing
here changes the frozen rules or any file outside this folder.

Run: `python research/hedge/hedge.py` from anywhere (about 7 s). It blocks sockets, cuts
every input to 2012-01-01 <= event_date < 2023-01-01 on read, unit-tests the lock math
first, and runs its sanity checks before writing anything. It makes no random draws, so
it needs no seed. Two runs give byte-identical outputs.

**Bottom line:** no hedge policy passes the adoption bar, at a book or on the
hypothetical exchange. Locking gives up more than it saves, because the bets where a lock
is available are the ones that tend to win.

## Decisions to flag
1. **The clean-close columns are empty for 2012-2022.** `a_close_clean` / `b_close_clean`
   in `bfo_joined.csv` are only filled for 2026 (card_settle's backfill). For older
   events BFO has already pruned the in-play ticks, so the raw close is the clean close.
   I used the ledger's own `_clean_close` rule: the clean column where it is written, else
   the raw close when its overround is in [1.00, 1.15]. That is also fade_buckets' CLV
   rule.
   - It gives a close for 4,830 of the 4,834 frame bouts.
   - All 322 U1 bets have one. It equals the ledger's `clean_close` on every bet, and
     the ledger's `clv_pts` within its 2 dp rounding.
   - 2 U2 bouts have no clean close and are never hedged.
   - Read literally, "the clean close column" would mean every bet is "no hedge", and
     every policy would equal P0.
2. **The hedge decision uses the price you can actually get.** L is computed at the
   slipped (or exchange) price, so slippage changes which bets get hedged, not just the
   price they're hedged at.
3. **Exchange exit (hypothetical).** Y's vig-free close is `1 + c_y/c_x`. Commission k
   comes off the hedge leg's winnings, so the effective price is `1 + (c_fair - 1)(1 - k)`.
   The full lock is computed at that effective price, so it pays the same either way.
4. **t is on P&L per bet at the actual stake**, which matches "SD of P&L per bet". For
   P0 on U1-placed that gives 3.68. betting_system's PLACED t uses the per-unit return
   unweighted by stake and gives 3.56. The adoption bar compares like with like, so the
   choice doesn't decide any pass or fail.
5. **Max drawdown** is the largest peak-to-trough fall in cumulative P&L. Bets are taken
   in date order and, within a card, in `bout_order` (chronological). The peak starts
   at 0.
6. **Sanity check 3 needed a definition.** See [Sanity checks](#sanity-checks-all-pass).
   The brief's simple approximation is 1.93 pp off P3. The gap is fully explained by the
   size of the hedge stake, and the exact identity holds to 1e-15.
7. **Staking.** Hedges scale with the stake (h = s·d_o/c, thresholds are a multiple of s).
   So U1-placed P&L = stake × the per-1u P&L, which is checked to 3e-16 for every policy
   and setting. `hedge_bets.csv` carries the per-1u P&L only.
8. **Universes.**
   - U1 includes pick'ems, as the sim does. VOID rows are excluded.
   - U2 is rule_search's usable set, with the same exclusions: 1,062 debut bouts and
     143 exact pick'ems; no bout fails the pre-filter and none has a gap of 0. The
     window has frozen preds every year. rule_search's `model_preds()` is called as-is,
     so its refit runs, but it supplies no rows (this is asserted).

## Lock math
Bet s on X at open decimal d_o. At the close, hedge h = s·d_o/c on Y at decimal c, which
locks L = s(d_o - 1) - h either way. L > 0 iff 1/d_o + 1/c < 1.

Unit test, run before anything else (PASS): +150 at the open and -135 at the close gives
h 1.4362 and L +0.0638. The payoff is identical under both outcomes (|diff| 0). The sign
rule holds on a 698×698 price grid. Every execution setting locks to within 2e-16.

## How often a lock is there (lock_hist.png)
U1-flat, 322 signals, full lock per 1u:

| hedge price | bets with L > 0 | median L among them | L >= 0.02 | L >= 0.05 | L >= 0.10 |
|---|---|---|---|---|---|
| BFO close (best price) | 151 (46.9%) | +0.089u | 133 | 105 | 68 |
| book +2 pp | 116 (36.0%) | +0.084u | 102 | 82 | 55 |

71% of U1 bets have positive CLV (mean +4.69 pp). Only 47% clear the closing vig by
enough to lock anything.

## U1-placed (139 bets): policy tables
P&L and max drawdown are in units at the policy-v1 stake. ROI is on the original stake
(118.5u). Worst year is 2013 for every policy.

**Hedge at the BFO close (book +0 pp)**

| policy | hedged | P&L | ROI | SD/bet | t | max DD | worst year |
|---|---|---|---|---|---|---|---|
| P0 never | 0% | +35.18 | +29.7% | 0.811 | 3.68 | 6.80 | -2.59 |
| P1 lock L >= 0 | 47% | +25.63 | +21.6% | 0.626 | 3.47 | 4.53 | -1.75 |
| P2a L >= 0.02s | 40% | +27.69 | +23.4% | 0.661 | 3.56 | 5.54 | -2.76 |
| P2b L >= 0.05s | 32% | +27.34 | +23.1% | 0.699 | 3.32 | 6.54 | -2.76 |
| P2c L >= 0.10s | 22% | +28.90 | +24.4% | 0.726 | 3.38 | 7.41 | -3.20 |
| P3 always | 100% | +0.17 | +0.1% | 0.153 | 0.10 | 2.86 | -1.70 |
| P4 half, L >= 0 | 47% | +30.41 | +25.7% | 0.674 | 3.83 | 5.64 | -2.17 |

**Hedge at a book with 2 pp slippage**

| policy | hedged | P&L | ROI | SD/bet | t | max DD | worst year |
|---|---|---|---|---|---|---|---|
| P0 never | 0% | +35.18 | +29.7% | 0.811 | 3.68 | 6.80 | -2.59 |
| P1 lock L >= 0 | 35% | +25.62 | +21.6% | 0.692 | 3.14 | 5.59 | -2.84 |
| P2a L >= 0.02s | 32% | +27.01 | +22.8% | 0.692 | 3.31 | 5.59 | -2.84 |
| P2b L >= 0.05s | 27% | +26.26 | +22.2% | 0.718 | 3.10 | 6.59 | -2.38 |
| P2c L >= 0.10s | 17% | +30.28 | +25.6% | 0.738 | 3.48 | 6.80 | -2.59 |
| P3 always | 100% | -4.41 | -3.7% | 0.155 | -2.42 | 5.53 | -2.37 |
| P4 half, L >= 0 | 35% | +30.40 | +25.7% | 0.720 | 3.58 | 6.19 | -2.72 |

**Exchange exit, k = 2% (HYPOTHETICAL)**

| policy | hedged | P&L | ROI | SD/bet | t | max DD | worst year |
|---|---|---|---|---|---|---|---|
| P0 never | 0% | +35.18 | +29.7% | 0.811 | 3.68 | 6.80 | -2.59 |
| P1 lock L >= 0 | 55% | +22.76 | +19.2% | 0.589 | 3.28 | 4.50 | -1.72 |
| P2a L >= 0.02s | 45% | +23.40 | +19.8% | 0.629 | 3.16 | 5.52 | -2.73 |
| P2b L >= 0.05s | 37% | +27.29 | +23.0% | 0.662 | 3.49 | 5.52 | -2.73 |
| P2c L >= 0.10s | 25% | +27.28 | +23.0% | 0.721 | 3.21 | 7.40 | -3.19 |
| P3 always | 100% | +2.91 | +2.5% | 0.149 | 1.66 | 1.84 | -1.27 |
| P4 half, L >= 0 | 55% | +28.97 | +24.5% | 0.648 | 3.79 | 5.63 | -2.16 |

The settings not shown (book +1 pp, exchange k = 0% and 5%) are in hedge_results.csv,
with the hedge-stake columns and the locked/ridden split.

## Adoption bar (pre-registered, U1-placed)
To pass, a policy needs all three: t above P0's (3.68), max drawdown at least 30% below
P0's (<= 4.76u), and total P&L no more than 25% below P0's (>= +26.39u).

| policy | book +2 pp: t / DD / P&L | result | exchange k=2% (hypothetical): t / DD / P&L | result |
|---|---|---|---|---|
| P1 | FAIL / FAIL / FAIL | **FAIL** | FAIL / PASS / FAIL | **FAIL** |
| P2a | FAIL / FAIL / PASS | **FAIL** | FAIL / FAIL / FAIL | **FAIL** |
| P2b | FAIL / FAIL / FAIL | **FAIL** | FAIL / FAIL / PASS | **FAIL** |
| P2c | FAIL / FAIL / PASS | **FAIL** | FAIL / FAIL / PASS | **FAIL** |
| P3 | FAIL / FAIL / FAIL | **FAIL** | FAIL / PASS / FAIL | **FAIL** |
| P4 | FAIL / FAIL / PASS | **FAIL** | PASS / FAIL / PASS | **FAIL** |

**No policy earns a shadow column.** The closest calls:
- **P4 on the hypothetical exchange:** t 3.79 beats 3.68, but its drawdown is only 17%
  lower.
- **P1:** cuts drawdown by 34% on the exchange, but costs 35% of the P&L.

## Where the cost comes from
At book +2 pp, P1 locks 49 bets for +5.89u. Ridden, the same 49 bets made +15.45u.
- **Those are the bets that tend to win.** At the BFO close, bets with a lock available
  won 64.2%, against 57.9% for the rest, because CLV predicts the result.
- **The hedge also pays the closing vig.** That is about 3.5% hold on roughly 0.76u of
  hedge per 1u bet.
- **P4 does best of the hedging policies because it only gives up half.**
- **P3 (pure line trading)** is close to breakeven at the best close: +0.17u on 139
  placed bets. It loses -4.41u at +2 pp. So the placed bets' CLV is worth about one
  closing vig, and no more, once you have to exit at a book.

**Post hoc, not part of the bar.** On U1-flat (all 322 signals at 1u), P2c at +2 pp
clears all three thresholds: +54.20u against +50.12u, t 3.34 against 2.78, drawdown 4.15
against 6.31. But:
- At +0 pp, a better hedge price, the same policy fails the drawdown bar (5.38).
- Locking helps at +2 pp (+15.07 against +10.99 ridden) and hurts at +0 pp (+18.99
  against +20.48).

Which bets happen to cross 0.10 decides it, so read it as noise. It was not the
pre-registered view.

## U2: does hedging rescue the broad CLV signal? No.
3,629 usable bouts, 1u on the model side at the open:

| hedge price | P0 never | P1 lock L >= 0 | P3 always |
|---|---|---|---|
| BFO close | -110.6u (-3.05%), t -1.56, DD 126.9 | -143.5u (-3.96%), t -2.42, DD 159.9 | -339.2u (-9.35%), t -26.9 |
| book +2 pp | same | -175.8u (-4.85%), t -2.79, DD 185.0 | -517.9u (-14.27%), t -39.9 |
| exchange k=2% (hypothetical) | same | -115.8u (-3.19%), t -2.06, DD 144.9 | -179.1u (-4.94%), t -15.2 |

The broad CLV (+1.15 pp mean) is real, but it is smaller than the vig paid at the open
plus the vig paid to get out. Even a free exchange exit (k = 0%) leaves P1 at -105.0u.

## Sanity checks (all PASS)
1. **P0 reproduces betting_system's sim, 2012-2022.** Here = `betting_system.run()` on
   the same frame = the committed ledger:
   - U1-flat: 322 bets, 196-126, +50.1203u. By rule: GAP 165 114-51 +11.39u; FLIP 157
     82-75 +38.73u, matching rule_search.
   - U1-placed: 139 bets, 95-44, +35.1824u. The ledger shows +35.1825u from its 4 dp
     rounding.
   - Same fight_ids, sides, stakes and placed set in all three. Every window bet is
     live = 0, so it settles at the open.
   - Also: the close used here equals the ledger's clean close on all 322 bets.
2. **P3 locks are outcome-independent.** Across 24,528 hedged bet-settings (3 views ×
   6 execution settings), the max |payoff if X wins - payoff if Y wins| is 8.9e-16.
3. **P3 on U2 at 0 pp is CLV net of the closing vig.**
   - P3's mean P&L per bet is **-9.35%** (-9.43% over the 3,627 hedged bets).
   - Mean CLV in price terms is -3.97%. I defined it as d_o × (vig-free close
     probability) - 1: the open bet's expected return if the close is right. It carries
     the opening vig, so it is negative even though CLV in probability points is
     +1.15 pp.
   - The average closing vig is 3.53% (hold, 1 - 1/overround).
   - The brief's approximation, CLV - vig, gives **-7.50%**. That is 1.93 pp above P3,
     because the hedge stake averages 1.558u per 1u on U2 (the model side is more often
     the dog), not 1u.
   - Exact: L = (CLV in price terms) - (hedge stake × hold) bet by bet, max |diff|
     1.1e-15. Its mean is -9.43%.
   - PASS on "negative, and equal to CLV net of the vig the hedge pays".
4. **Byte-identical runs.** Two runs in a fresh clone give byte-identical outputs and
   console output. The outputs also match the committed files byte for byte (raw bytes
   hash to the committed blobs). A fresh checkout with autocrlf holds the two CSVs as
   CRLF. The script writes LF, so `git status` can list them as modified even though
   `git diff` is empty.

## Outputs
- **hedge.py:** the script.
- **hedge_results.csv:** 126 rows, one per (U1-flat / U1-placed / U2) × policy ×
  execution setting.
  - Columns: n bets, % hedged, mean and max hedge stake per 1u, P&L, ROI, SD per bet,
    t, max drawdown, worst year, losing years, and the locked/ridden split
    (`pnl_locked_if_ridden_u` = what the locked bets made under P0).
  - Exchange rows say "(hypothetical)" in `execution` and carry `hypothetical = 1`.
  - Adoption-bar columns are filled for U1-placed at the two bar settings.
- **hedge_bets.csv:** one row per U1 bet.
  - Opens and closes for both sides (American and decimal), CLV, and the full-lock hedge
    stake and L at 0 and 2 pp.
  - P&L per 1u under each policy at 0 and 2 pp. The placed P&L is `stake_u` × these.
- **lock_hist.png:** available L per 1u across U1-flat at the BFO close. One bet at
  +1.84u (Jackson 2019-04-13, +250 at the open, -585 at the close) is noted off the
  right edge.
- **cum_pnl.png:** cumulative U1-placed P&L for P0, P1 and P3 at book +2 pp.

## Reading it
- **What hedging actually does.** It turns some of the open-to-close edge into certainty,
  at the price of the upside on exactly the bets that were most likely to win, plus a
  second vig. On these signals that trade is a loss at any realistic price.
- **The best execution doesn't change the answer.** Even at the hypothetical exchange
  exit, the policies that cut drawdown enough (P1, P3) cost far more than 25% of P&L.
- **On U1-placed the worst year is 2013 under every policy and setting.** Hedging softens
  it a little, but doesn't change where the risk sits.
- **What it says about the edge.** The CLV is real, but it is not separable profit you
  can lock. The model's profit on the placed bets is mostly in riding them to the result.
