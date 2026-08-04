# NC-TOPI Stage-0 verified report

## Verified inventory and lineage
- 1,520 total metric rows = 1,480 scenario + 40 ablation.
- 1,320 simple scalar/classification rows independently recomputed; all 1,480 scenario rows independently reconstructed by the production contract.
- Frozen B0 checkpoint SHA256: `f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b`.
- Source commit: `c94af28795d03a91e2f4c0faa74eb19a983ed82e`.
- Clean split PRN counts: train=6,074, calibration=1,628, holdout=1,306, excluded-boundary=891.
- Clean split event counts: train=586, calibration=157, holdout=118, excluded-boundary=86.
- Causal IQ: 5,283 event contexts / 55,591 linked PRN contexts; 70 raw contexts independently reread; strict causal=true.
- DS7 legacy positive control (non-primary): coverage=5465/5465, max absolute RMSE error=9.61496e-05, tolerance=0.0003, pass=true.

## Primary q99 / median results
Primary evidence is the frozen median aggregator at q99. Values below are independently reconstructed from event scores and typed thresholds.

| Scenario | Method | FPR | ROC-AUC | PR-AUC | pAUC@0.05 | stable-pre FPR | post detection | 3-consecutive sustained delay (s/status) | persistent ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cleanStatic holdout | B0 | 0.0338983 (4/118) | - | - | - | - | - | - | - |
| cleanStatic holdout | TOPI | 0.0169492 (2/118) | - | - | - | - | - | - | - |
| cleanStatic holdout | NC_TOPI | 0 (0/118) | - | - | - | - | - | - | - |
| cleanDynamic external | B0 | 0.670659 (112/167) | - | - | - | - | - | - | - |
| cleanDynamic external | TOPI | 0.670659 (112/167) | - | - | - | - | - | - | - |
| cleanDynamic external | NC_TOPI | 1 (167/167) | - | - | - | - | - | - | - |
| DS1 | B0 | - | 0.96474 | 0.995568 | 0.96682 | 0.0206186 (2/97) | 0.934813 (674/721) | 27.1481/already_alarming_stable_pre | 1 (641/641) |
| DS1 | TOPI | - | 0.9687 | 0.996052 | 0.9677 | 0 (0/97) | 0.933426 (673/721) | 26.6481/detected | 1 (641/641) |
| DS1 | NC_TOPI | - | 0.9687 | 0.996052 | 0.9677 | 0 (0/97) | 0.932039 (672/721) | 27.1481/detected | 1 (641/641) |
| DS2 | B0 | - | 0.989589 | 0.998677 | 0.987607 | 0.0103093 (1/97) | 0.976124 (695/712) | 11.7063/already_alarming_stable_pre | 1 (632/632) |
| DS2 | TOPI | - | 0.992746 | 0.999061 | 0.988877 | 0 (0/97) | 0.976124 (695/712) | 11.7063/detected | 1 (632/632) |
| DS2 | NC_TOPI | - | 0.992746 | 0.999061 | 0.988877 | 0 (0/97) | 0.974719 (694/712) | 11.7063/detected | 1 (632/632) |
| DS3 | B0 | - | 0.982673 | 0.997803 | 0.977942 | 0.0103093 (1/97) | 0.955182 (682/714) | 21.2129/already_alarming_stable_pre | 1 (634/634) |
| DS3 | TOPI | - | 0.986572 | 0.998274 | 0.977646 | 0 (0/97) | 0.94958 (678/714) | 21.2129/detected | 1 (634/634) |
| DS3 | NC_TOPI | - | 0.986572 | 0.998274 | 0.977646 | 0 (0/97) | 0.945378 (675/714) | 21.7129/detected | 1 (634/634) |
| DS7 | B0 | - | 0.912601 | 0.970818 | 0.859137 | 0 (0/117) | 0.661442 (211/319) | 18.1236/detected | 0.8159 (195/239) |
| DS7 | TOPI | - | 0.926051 | 0.974527 | 0.842209 | 0.00854701 (1/117) | 0.570533 (182/319) | 27.6236/already_alarming_stable_pre | 0.728033 (174/239) |
| DS7 | NC_TOPI | - | 0.981459 | 0.99433 | 0.972973 | 0 (0/117) | 0.887147 (283/319) | 15.6236/detected | 0.953975 (228/239) |
| DS8 | B0 | - | 0.971395 | 0.995608 | 0.946677 | 0.00854701 (1/117) | 0.864903 (621/718) | 18.1825/already_alarming_stable_pre | 0.946708 (604/638) |
| DS8 | TOPI | - | 0.971431 | 0.995601 | 0.946103 | 0 (0/117) | 0.818942 (588/718) | 77.6825/detected | 0.910658 (581/638) |
| DS8 | NC_TOPI | - | 0.990191 | 0.998563 | 0.991063 | 0 (0/117) | 0.969359 (696/718) | 14.6825/detected | 1 (638/638) |

## NC-B0 pAUC deltas and paired 10 s block bootstrap 95% CIs
Paired scores are resampled in frozen nonoverlapping 10 s recording/label/cadence blocks (2,000 repetitions where available).

| Attack | NC-B0 pAUC delta | 95% CI | valid reps |
|---|---:|---:|---:|
| DS1 | 0.000879913 | [-0.000712251, 0.0042735] | 2000 |
| DS2 | 0.00126973 | [0, 0.00565611] | 2000 |
| DS3 | -0.00029618 | [-0.000413565, 0.0138544] | 2000 |
| DS7 | 0.113836 | [0.0490751, 0.219789] | 2000 |
| DS8 | 0.0443863 | [0.0111355, 0.0791282] | 2000 |

## Frozen c1-c8 decision
- Status: **NO-GO**.
- Criteria: c1=true; c2=true; c3=true; c4=true; c5=true; c6=true; c7=false; c8=true.
- Counts: improvement_count=4; positive_ci_count=2; stable_pre_failures=0.
- Exact NO-GO trigger(s): c7_second_peak_false.
- c6 is exactly the frozen equal-RMSE direction test; nuisance controls are diagnostics only and do not rescue or fail c1-c8.

## Synthetic physics and nuisance diagnostics
- Equal-RMSE: pass=true; trials=100; max B0 relative difference=3.33067e-16; median tangent/orthogonal TOPI ratio=1.85613e-32; median orthogonal preservation=1.
- Second peak: pass=false; power pass=true; separation pass=false; rows=25.
- Second-peak power rho by separation: {"0.25":0.9999999999999999,"0.375":0.9999999999999999,"0.5":0.9999999999999999}.
- Second-peak separation rho by power: {"0.2":-0.09999999999999999,"0.4":-0.09999999999999999,"0.8":-0.09999999999999999}.
- Nuisance amplitude (prompt-normalized shape-scale): rows=6, median normalized B0=13.1816, median normalized TOPI=4.36812e-30, pass=true.
- Nuisance shift (local numerical delay tangent (amount * gradient, edge_order=2)): rows=6, median normalized B0=4.19836, median normalized TOPI=1.77307e-30, pass=true.
- Nuisance noise (frozen seeded standardized noise): rows=3, median normalized B0=11.0486, median normalized TOPI=4969.18, pass=false.

## Frozen inventory scope and limitations
- Both median/top25_mean aggregators and q99/q995 thresholds are frozen and reported in `scenario_metrics.csv`; primary is median/q99. `ablation_metrics.csv` reports all 10 ablations.
- cleanDynamic OOD failure: external q99/median FPR is B0=0.670659, TOPI=0.670659, NC-TOPI=1; this failure is reported, not hidden.
- DS1 capture failure/reporting: q99/median NC-TOPI misses 49/721 post epochs and its 3-consecutive delay is 27.1481 s; no failed capture is removed.
- Frozen B0 predicts a finite nine-tap shape, not an infinite-support physical correlation function; edge/tail behavior is a finite-tap caveat.
- Historical B0 training overlap with Stage-0 source support limits independence; B0 was hash-pinned and never retrained or selected here.

## Baselines and claim boundary
- PD-ML is a likelihood-family method, correlator LASSO is a sparse multi-component fit, and B0 is standardized frozen-predictor nine-tap RMSE. They are distinct and not interchangeable.
- Claimable contribution: a hash-bound, clean-fit, independently reconstructable Stage-0 evaluation of tangent-orthogonal TOPI and causal IQ scale conditioning on these frozen recordings/splits.
- Non-claimable: universal spoofing detection, superiority to unimplemented PD-ML/correlator LASSO baselines, causal RF-mechanism identification, or independence from historical B0 overlap.

## Verification result
- Independent verification passed: true.
- Verification error count: 0.
