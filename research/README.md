# Research log
Read-only studies. None changed the frozen signal or placement rules. Each
folder has its own README, script, and outputs, reproducible at the commit listed.

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
