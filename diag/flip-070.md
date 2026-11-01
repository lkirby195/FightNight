# Diagnostic: FLIP threshold 0.70 vs 0.65

Branch `diag/flip-070` off main a757800. Read-only run of `betting_system.run()` twice on one `load()` frame, with `signals.FLIP_CONF` overridden in memory for the second run (`diag/flip_070.py`). signals.py constants, GAP rule, policy v1 skips and ramp, debut rule and opener rule unchanged. No ledger written, nothing under data/ or site/ touched.

Data through 2026-10-24 (last settled card). Run date 2026-10-31.

2025 is the sealed holdout and is absent from `data/model_only_preds.csv`; its rows here come from the sealed `data/model_only_preds_2025_backtest.csv` pushed through the same current `load()` logic and `betting_system.run()` (so the current debut rule, pick'em tie = GAP and alias table apply). The sealed site file `data/system_ledger_2025_backtest.csv` is cross-checked in section 2b; neither sealed file was rewritten.

The site's 2026 tab (since PR #13) is every 2026 ledger row; it is shown split (live=0 / live=1) and as the union.

Conventions: `net u` is the ledger's `pnl` at 1u flat (settled at the earliest pre-event capture for live=1 rows, at open otherwise, exactly as the ledger of record); `placed` columns are policy v1 (stake_u / pnl_placed). W-L counts pnl > 0 / pnl < 0; a push (draw/NC) is in bets but in neither. VOID rows are not counted. The 0.70 run was asserted to equal the 0.65 run minus FLIP rows with conf < 0.70 (same fights, same prices, same policy).

## 0. Summary

- 0.70 removes 126 of 426 all-time bets: 109 in 2012-2024 (60-49, +33.32u, ROI +30.6%) and 17 in 2025-2026 (12-5, +8.95u, ROI +52.7%). GAP rows do not move.
- All-time at 1u flat: 426 bets +85.28u (ROI +20.0%) -> 300 bets +43.01u (ROI +14.3%). Policy v1 placed: +60.11u -> +29.63u.
- The band is as profitable per bet as the kept FLIPs in 2012-2024 and more profitable than them in 2025-2026; 0.70 lowers every window's net and ROI.
- 2026 live: no row changes. Malkoun (2026-10-24) is conf 0.704, above 0.70 by 0.004; Tuivasa 0.703 (unplaced, +504) and Van 0.751 also stay. A preds regeneration moves probs by ~0.001, so Malkoun and Tuivasa sit within noise of the line.

## 1. Window table, both thresholds

| | FLIP 0.65 (current) | | | | | | | | | FLIP 0.70 | | | | | | | | |
| window | bets | W-L | net u | ROI | placed | pl W-L | stake u | pl net u | pl ROI | bets | W-L | net u | ROI | placed | pl W-L | stake u | pl net u | pl ROI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2012-2024 (train/tune) | 369 | 231-138 | +68.38 | +18.5% | 169 | 119-50 | 141.94 | +51.05 | +36.0% | 260 | 171-89 | +35.06 | +13.5% | 111 | 81-30 | 93.34 | +26.00 | +27.9% |
| 2025 (holdout) | 25 | 16-9 | +4.67 | +18.7% | 22 | 14-8 | 17.19 | +3.15 | +18.3% | 15 | 9-6 | -0.89 | -5.9% | 13 | 8-5 | 10.68 | +0.67 | +6.3% |
| 2026 not timestamped (live=0) | 29 | 23-6 | +13.50 | +46.6% | 19 | 15-4 | 15.09 | +6.11 | +40.5% | 22 | 18-4 | +10.10 | +45.9% | 14 | 11-3 | 11.06 | +3.16 | +28.5% |
| 2026 live (live=1) | 3 | 1-2 | -1.28 | -42.5% | 2 | 1-1 | 1.92 | -0.20 | -10.2% | 3 | 1-2 | -1.28 | -42.5% | 2 | 1-1 | 1.92 | -0.20 | -10.2% |
| 2026 all (site tab) | 32 | 24-8 | +12.23 | +38.2% | 21 | 16-5 | 17.01 | +5.92 | +34.8% | 25 | 19-6 | +8.83 | +35.3% | 16 | 12-4 | 12.98 | +2.96 | +22.8% |
| all-time | 426 | 271-155 | +85.28 | +20.0% | 212 | 149-63 | 176.14 | +60.11 | +34.1% | 300 | 199-101 | +43.01 | +14.3% | 140 | 101-39 | 117.00 | +29.63 | +25.3% |

Per-rule split at 0.65 (GAP is identical under 0.70):

| window | rule | bets | W-L | net u | ROI | placed | pl net u |
|---|---|---|---|---|---|---|---|
| 2012-2024 (train/tune) | FLIP | 178 | 98-80 | +54.26 | +30.5% | 84 | +39.52 |
| 2012-2024 (train/tune) | GAP | 191 | 133-58 | +14.13 | +7.4% | 85 | +11.53 |
| 2025 (holdout) | FLIP | 15 | 8-7 | +2.69 | +17.9% | 14 | +0.26 |
| 2025 (holdout) | GAP | 10 | 8-2 | +1.97 | +19.7% | 8 | +2.88 |
| 2026 not timestamped (live=0) | FLIP | 12 | 8-4 | +5.45 | +45.4% | 6 | +2.45 |
| 2026 not timestamped (live=0) | GAP | 17 | 15-2 | +8.05 | +47.4% | 13 | +3.66 |
| 2026 live (live=1) | FLIP | 3 | 1-2 | -1.28 | -42.5% | 2 | -0.20 |
| 2026 live (live=1) | GAP | 0 | 0-0 | +0.00 | n/a | 0 | +0.00 |
| 2026 all (site tab) | FLIP | 15 | 9-6 | +4.17 | +27.8% | 8 | +2.26 |
| 2026 all (site tab) | GAP | 17 | 15-2 | +8.05 | +47.4% | 13 | +3.66 |
| all-time | FLIP | 208 | 115-93 | +61.12 | +29.4% | 106 | +42.04 |
| all-time | GAP | 218 | 156-62 | +24.16 | +11.1% | 106 | +18.07 |

## 2. Band table: FLIP bets with conf in [0.65, 0.70)

This is exactly the set a 0.70 threshold removes. GAP-only line included to show what does not move.

| window | set | bets | W-L | net u | ROI | placed | pl W-L | stake u | pl net u | pl ROI |
|---|---|---|---|---|---|---|---|---|---|---|
| 2012-2024 | FLIP [0.65, 0.70) - removed by 0.70 | 109 | 60-49 | +33.32 | +30.6% | 58 | 38-20 | 48.60 | +25.05 | +51.5% |
| 2012-2024 | FLIP >= 0.70 - kept | 69 | 38-31 | +20.93 | +30.3% | 26 | 18-8 | 22.39 | +14.47 | +64.6% |
| 2012-2024 | GAP only - unchanged | 191 | 133-58 | +14.13 | +7.4% | 85 | 63-22 | 70.95 | +11.53 | +16.2% |
| 2012-2024 | all signals | 369 | 231-138 | +68.38 | +18.5% | 169 | 119-50 | 141.94 | +51.05 | +36.0% |
| 2025-2026 | FLIP [0.65, 0.70) - removed by 0.70 | 17 | 12-5 | +8.95 | +52.7% | 14 | 10-4 | 10.54 | +5.43 | +51.5% |
| 2025-2026 | FLIP >= 0.70 - kept | 13 | 5-8 | -2.09 | -16.0% | 8 | 2-6 | 6.43 | -2.91 | -45.2% |
| 2025-2026 | GAP only - unchanged | 27 | 23-4 | +10.03 | +37.1% | 21 | 18-3 | 17.23 | +6.54 | +37.9% |
| 2025-2026 | all signals | 57 | 40-17 | +16.90 | +29.6% | 43 | 30-13 | 34.20 | +9.06 | +26.5% |

## 2b. Cross-check: the sealed 2025 site file

Sealed file: 34 signals (19 FLIP / 15 GAP), placed 27, pnl_placed +7.09u. Written 2026-09-09 under the pre-PR-#9 rules (no debut rule, no alias table, pick'em tie not yet GAP).

Current rules over the same preds: 25 signals (15 FLIP / 10 GAP), placed 22, pnl_placed +3.15u. Fights in sealed only: 10; in current only: 1.

Band on the sealed file itself (conf joined from the sealed preds):

| set | bets | W-L | net u | ROI | placed | pl W-L | stake u | pl net u | pl ROI |
|---|---|---|---|---|---|---|---|---|---|
| FLIP [0.65, 0.70) - removed by 0.70 | 12 | 8-4 | +6.85 | +57.1% | 10 | 7-3 | 7.30 | +4.29 | +58.8% |
| FLIP >= 0.70 - kept | 7 | 3-4 | -0.18 | -2.6% | 6 | 3-3 | 5.01 | +1.47 | +29.3% |
| GAP only - unchanged | 15 | 11-4 | +1.67 | +11.1% | 11 | 8-3 | 9.62 | +1.33 | +13.8% |
| all signals | 34 | 22-12 | +8.34 | +24.5% | 27 | 18-9 | 21.93 | +7.09 | +32.3% |

## 3. Bets in [0.65, 0.70) from 2025 and 2026

2025 rows are from the current-rules run over the sealed preds (section 2 basis).

| date | event | bout (pick first) | rule | conf | line | live | stake u | result | net u (1u) | net u (placed) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-02-01 | UFC Fight Night: Adesanya vs. Imavov | Fares Ziam vs Mike Davis | FLIP | 0.654 | +200 | 0 | 0.90 | WIN | +2.00 | +1.80 |
| 2025-02-15 | UFC Fight Night: Cannonier vs. Rodrigues | Gregory Rodrigues vs Jared Cannonier | FLIP | 0.657 | +110 | 0 | 0.93 | LOSS | -1.00 | -0.93 |
| 2025-03-22 | UFC Fight Night: Edwards vs. Brady | Chris Padilla vs Jai Herbert | FLIP | 0.660 | +105 | 0 | 0.50 | WIN | +1.05 | +0.52 |
| 2025-03-29 | UFC Fight Night: Moreno vs. Erceg | Manuel Torres vs Drew Dober | FLIP | 0.650 | -110 | 0 | 0.64 | WIN | +0.91 | +0.58 |
| 2025-06-21 | UFC Fight Night: Hill vs. Rountree Jr. | Ismail Naurdiev vs JunYong Park | FLIP | 0.689 | +163 | 0 | 0.72 | LOSS | -1.00 | -0.72 |
| 2025-06-21 | UFC Fight Night: Hill vs. Rountree Jr. | Ignacio Bahamondes vs Rafael Fiziev | FLIP | 0.673 | +110 | 0 | 0.94 | LOSS | -1.00 | -0.94 |
| 2025-07-26 | UFC Fight Night: Whittaker vs. De Ridder | Reinier de Ridder vs Robert Whittaker | FLIP | 0.657 | +152 | 0 | 0.52 | WIN | +1.52 | +0.79 |
| 2025-10-04 | UFC 320: Ankalaev vs. Pereira 2 | Daniel Santos vs JooSang Yoo | FLIP | 0.663 | +102 | 0 | 0.00 | WIN | +1.02 | +0.00 |
| 2025-11-15 | UFC 322: Della Maddalena vs. Makhachev | Matheus Camilo vs Viacheslav Borshchev | FLIP | 0.697 | +110 | 0 | 0.50 | WIN | +1.10 | +0.55 |
| 2025-11-15 | UFC 322: Della Maddalena vs. Makhachev | Michael Morales vs Sean Brady | FLIP | 0.666 | -105 | 0 | 0.86 | WIN | +0.95 | +0.82 |
| 2026-02-07 | UFC Fight Night: Bautista vs. Oliveira | Vinicius Oliveira vs Mario Bautista | FLIP | 0.684 | +120 | 0 | 0.67 | LOSS | -1.00 | -0.67 |
| 2026-03-28 | UFC Fight Night: Adesanya vs. Pyfer | Casey O'Neill vs Gabriella Fernandes | FLIP | 0.651 | +110 | 0 | 0.86 | WIN | +1.10 | +0.95 |
| 2026-05-30 | UFC Fight Night: Song vs. Figueiredo | Cameron Smotherman vs Kai Asakura | FLIP | 0.681 | +285 | 0 | 0.00 | LOSS | -1.00 | -0.00 |
| 2026-06-20 | UFC Fight Night: Kape vs. Horiguchi | Luana Santos vs Karol Rosa | FLIP | 0.676 | +100 | 0 | 0.79 | WIN | +1.00 | +0.79 |
| 2026-08-08 | UFC Fight Night: Gamrot vs. Salkilld | Quillan Salkilld vs Mateusz Gamrot | FLIP | 0.690 | +111 | 0 | 0.77 | WIN | +1.11 | +0.85 |
| 2026-08-29 | UFC Fight Night: Nurmagomedov vs. Song | Denise Gomes vs Yan Xiaonan | FLIP | 0.656 | +110 | 0 | 0.94 | WIN | +1.10 | +1.03 |
| 2026-10-17 | UFC Fight Night: Buckley vs. Malott | Steven Koslow vs Chad Anheliger | FLIP | 0.657 | +109 | 0 | 0.00 | WIN | +1.09 | +0.00 |

## 4. 2026 live rows

- 2026-09-19 Tai Tuivasa: FLIP conf 0.703 line +504 stake 0.00 -> unchanged
- 2026-09-19 Joshua Van: FLIP conf 0.751 line -138 stake 1.00 -> unchanged
- 2026-10-24 Jacob Malkoun: FLIP conf 0.704 line +128 stake 0.92 -> unchanged

0 of 3 live=1 rows change under 0.70.

