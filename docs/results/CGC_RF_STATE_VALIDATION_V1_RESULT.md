# CGC RF state validation v1 result

## Decision

`ABSOLUTE-DISTANCE STATE MAP NOT REPRODUCED; PHYSICAL ORDER PARTIALLY
REPRODUCED.` All 24 preregistered 25 MHz RF cells completed on the two held-out
receiver/LOS geometry bundles. The full-grid overall gate is `false`: both
straight power groups passed every gate, while both parallel-sweep power groups
failed only the fixed 40 m and 60 m AUC-state gates.

No geometry, cell, distance, power, comparison interval, estimator, or gate was
substituted after release. The effective runner and protocol were committed at
`158d464` before outcome access. The result rejects a universal state diagram
indexed by absolute carry-off distance, but retains a narrower and more useful
result: detection onset moves with receiver/LOS geometry, whereas the
100--160 m metric regime and the 240 m finite-aperture saturation regime
replicate in all four geometry-power groups.

## Preregistered group decision

| Geometry | Power | Full map | Failed gates |
|---|---:|:---:|---|
| straight | -6 dB | pass | none |
| straight | +3 dB | pass | none |
| parallel-sweep | -6 dB | fail | 40 m unresolved AUC; 60 m onset AUC |
| parallel-sweep | +3 dB | fail | 40 m unresolved AUC; 60 m onset AUC |

Every group passed minimum support, 60 m direction and direction-improvement,
100--160 m AUC/direction/displacement-error, and 240 m AUC/edge/bias gates.
Every condition compared 12 spoof bins with 12 independent-multipath bins.
Minimum per-bin support was 10--11 spoof PRNs and 9--11 multipath PRNs.

## Full transfer results

Entries separated by `/` are -6 dB and +3 dB.

| Geometry | Distance | AUC | Direction cosine | Norm error | Edge fraction |
|---|---:|---:|---:|---:|---:|
| straight | 40 m | .424 / .653 | .441 / .899 | +69.4% / +107.8% | 0 / 0 |
| straight | 60 m | .549 / .646 | .879 / .946 | +16.3% / +30.5% | 0 / 0 |
| straight | 80 m | .819 / .819 | .928 / .950 | +8.6% / +7.2% | 0 / 0 |
| straight | 100 m | .847 / .931 | .950 / .961 | -0.8% / +2.7% | 0 / 0 |
| straight | 160 m | .965 / .965 | .973 / .965 | +0.1% / -0.7% | 2.3% / 10.6% |
| straight | 240 m | 1.000 / 1.000 | .996 / .993 | -22.4% / -18.4% | 32.6% / 33.3% |
| parallel-sweep | 40 m | .854 / .931 | .605 / .776 | +33.2% / +44.1% | 0 / 0 |
| parallel-sweep | 60 m | .993 / 1.000 | .960 / .955 | +15.9% / +22.6% | 0 / 0 |
| parallel-sweep | 80 m | 1.000 / .993 | .972 / .953 | +10.2% / +21.3% | 0 / 0 |
| parallel-sweep | 100 m | .993 / .993 | .945 / .924 | +1.6% / +3.4% | 0 / 0 |
| parallel-sweep | 160 m | 1.000 / 1.000 | .993 / .996 | -8.3% / -2.6% | 6.8% / 30.0% |
| parallel-sweep | 240 m | 1.000 / 1.000 | .996 / .997 | -30.4% / -25.7% | 31.1% / 54.3% |

## What transferred and what did not

The first tested AUC 0.8 crossing was 80 m for both straight power groups but
40 m for both sweep power groups. At 40 m the sweep AUC exceeded straight by
0.431 at -6 dB and 0.278 at +3 dB. The same physical displacement and power
therefore do not define one receiver-independent detection state.

The directional signal nevertheless improved from 40 to 60 m in all four
groups and exceeded 0.879 at 60 m. A descriptive joint metric crossing (AUC at
least 0.8, direction at least 0.85, and absolute norm error at most 15%) first
occurred at 80 m for straight -6/+3 dB and sweep -6 dB, and at 100 m for sweep
+3 dB. These first-crossing distances are post-outcome descriptions, not new
validated thresholds.

The preregistered conservative metric cells did transfer. Across all eight
100/160 m geometry-power cells, AUC was 0.847--1.000, direction cosine was
0.924--0.996, and absolute displacement-norm error was at most 8.3%.

The saturation state also transferred at 240 m. All four groups retained AUC
1.000 and direction cosine 0.993--0.997, while edge occupancy rose to
31.1--54.3% and displacement was biased low by 18.4--30.4%. Thus detection and
direction remained observable after metric displacement recovery became
aperture limited.

## Physical contribution after validation

The supported contribution is not a universal meter-valued spoofing threshold.
It is a receiver observability decomposition:

1. **Detection onset is geometry dependent.** The independent-multipath ranking
   boundary moved by at least a factor of two in tested distance (80 to 40 m)
   between held-out geometry bundles.
2. **Detection and metric observability are distinct.** AUC can be essentially
   perfect while recovered displacement remains biased by more than 20%.
3. **Finite-aperture saturation is self-diagnosing.** At 240 m, template-edge
   occupancy and negative norm bias rose together in every geometry-power
   group without destroying classification or direction recovery.

This motivates a geometry-conditioned observability index rather than a fixed
distance threshold. A follow-up should predict the onset boundary from the LOS
design matrix, receiver motion, multipath reference distribution, power, and
tap aperture, then validate the prediction on additional unseen geometries.

## Execution integrity

The successful release contains 24 condition results, 24 receiver manifests,
and 36 retention records. All 24 composed RF files and all 12 distance-shared
counterfeit components were hash-checked and removed after durable scoring;
no campaign-created intermediate IQ remains. All composed signals retained a
byte-identical normal prefix. Existing authentic, normal, and multipath inputs
were not deleted.

One GNSS-SDR execution for straight/+3 dB/240 m was retried before scoring
because concurrent console writes merged `GPS PRN 16` into `GPS16`, while the
channel-4 MAT contained the valid G16 segment. The failed receiver directory
was preserved separately, the RF IQ and all physical settings were unchanged,
and the successful retry passed the unmodified receiver-evidence check.

The immutable raw result is
`artifacts/cgc_rf_state_validation_v1/summary.json`, SHA-256
`fb16d12a5f921e988190f17c372656b5cb9b521b5808b81a5549a0fca004fe8f`.
The four-panel plot is
`artifacts/cgc_rf_state_validation_v1/state_validation_curves.png`, SHA-256
`be790d8ddf25f0a98a5eb627db79ae1057dee3abc36f3a8942ade93b76f1f432`.

## Claim boundary

This is receiver-RF simulation across two held-out motion/LOS geometry bundles,
in addition to the earlier discovery geometry. It is not field validation and
does not establish performance on real multipath, TEXBAT multipath, arbitrary
receiver front ends, or operational attacks. The result is strong enough to
retain the observability-state paper direction, but a WCL claim should be the
geometry dependence and finite-aperture mechanism, not the rejected fixed
40/60/100/240 m state map.
