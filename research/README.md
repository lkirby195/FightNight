# Research log
Read-only studies. None changed the frozen signal or placement rules. Each
folder has its own README, script, and outputs, reproducible at the commit listed.

Reproducibility means byte-identical outputs on a rerun **in the same
environment**. Across library versions, expect floating-point noise in the
last digits. For example, scikit-learn 1.7.2 vs 1.9.1 (pandas 2.3 vs 3.0) moved
fitted probabilities by at most about 5e-12, with no change to any count, row,
label or reported result. Compare those runs numerically (tolerance 1e-9),
not by hash.

## fade_buckets (2026-11-03, e1915a9)
Question: should we bet the underdog when the model still picks the favorite but
rates it well below the market? Answer: no. 2018-22: 741 bets, 232-509,
ROI -15.2%, t -3.20. The pre-registered cell went 0-11. The model sits below the
market on the favorite in 71% of bouts, which is compression, not information.
CLV on the fade is about zero.

## rule_search (2026-11-03, f99fe45)
Question: does searching a grid of 105 simple rules across 2010-22 find real
edges, or luck? Answer: the search rediscovered FLIP, widened. 4 rules (3
distinct bet sets) passed a pre-registered bar against a null simulation. The
walk-forward search made +13.3% out of sample (99.7th percentile of the null).
Frozen GAP and FLIP sit at the 99.6th and 100th percentiles. A post-hoc standout
cell failed its fresh test (+1.6% on 2010-17).
Caveat: Glicko parameters were tuned on 2014-19, and the only block outside
that window, 2010-12, is the weakest.

## hedge (2026-11-03, 58c482f)
Question: if we bet at open and hedge at close when the line moves our way, does
that turn CLV into locked profit? Answer: no policy passed the pre-registered
adoption bar. Bets where a lock was available won 64.2%, against 57.9% for the
rest, so hedging sells our best bets at a discount, and hedging the rest costs
even more. Riding everything wins. With no rules at all (every bout, model side
at open, 2012-22), ROI is -3.05%, and hedging makes it worse.

## props_parlays (2026-11-05)
Questions: should we parlay our picks, is there a round over/under edge, and are
other props worth isolating? Answers: parlays, not now. They multiply whatever
the true per-leg edge is (backtest 2-leg doubled P&L), and that edge isn't
pinned down live, so it's Kelly-style sizing through the back door. Round totals:
our fighter features beat weight class + moneyline on goes-the-distance by
0.014-0.021 log-loss in both 2015-22 and 2023-24. That's signal, not an edge:
there are no prop prices in the repo yet. Method props on our side: no skew
(winners by decision 50.5% vs 46.2% expected). "Favorites overpriced in
decision-heavy fights": null once price is controlled (p 0.82-0.88).
Caveat: the pre-registered tercile test "passed" but was confounded with price;
reported as a null.

## totals_edge (2026-11-05)
Question: does our duration model beat BFO's closing rounds-totals price?
Answer: no. Pre-registered gate failed on 2023-24 (O/U 2.5, n=870: stack gain
-0.0016, t -1.32), so 2025+ was not read and stays clean. The market beats
the model outright by 0.013-0.021 log-loss. The model does predict the
open-to-close move (coef 0.09, t 6.7 with the open controlled; a null model
gets 0.016), but the close already has it. Post-hoc: books load the vig on the
short side (blind Under / "inside the distance" loses 10-16% over 2015-24);
the long side is break-even before 2023.
