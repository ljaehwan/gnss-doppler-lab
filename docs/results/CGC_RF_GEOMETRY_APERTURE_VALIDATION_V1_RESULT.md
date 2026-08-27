# CGC RF geometry and aperture validation v1 result

## Outcome

The complete campaign finished: 40/40 receiver-RF conditions and 120/120 condition-aperture summaries were reported. No geometry, distance, power, PRN, bin, estimator, aperture, or gate was substituted after release.

The strict preregistered conclusions are:

- broader-geometry nine-tap mechanism: **not fully supported** (9/10 geometry-power groups passed every gate);
- same-stream aperture mechanism: **not fully supported** (3/4 aggregate gates passed);
- 100 m metric regime: **reproduced in all 10/10 groups**;
- 240 m direction, negative-bias, and edge-contact state: **reproduced in all 10/10 groups**;
- 240 m discrimination AUC: **passed in 9/10 groups**.

The two overall `false` decisions must not be rewritten as confirmatory passes. The detailed result is nevertheless highly structured and identifies one RF discrimination blind spot plus a correction to the physical interpretation of template-edge contact.

## Five-geometry nine-tap result

| Geometry | Power | 100 m AUC | 100 m dir. | 100 m rel. error | 240 m AUC | 240 m dir. | 240 m rel. error | 240 m edge | All gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Denver static | -6 dB | 0.993 | 0.933 | -4.3% | 1.000 | 0.924 | -28.2% | 12.9% | yes |
| Denver static | +3 dB | 1.000 | 0.931 | +10.2% | 1.000 | 0.958 | -25.2% | 37.1% | yes |
| Seoul static | -6 dB | 1.000 | 0.900 | +6.6% | 1.000 | 0.985 | -24.8% | 25.8% | yes |
| Seoul static | +3 dB | 1.000 | 0.954 | +1.8% | 1.000 | 0.985 | -19.5% | 49.2% | yes |
| Tokyo straight | -6 dB | 0.882 | 0.927 | +3.8% | 0.993 | 0.996 | -23.2% | 38.0% | yes |
| Tokyo straight | +3 dB | 0.924 | 0.966 | +8.9% | **0.417** | 0.929 | -51.4% | 42.6% | **no** |
| London circle | -6 dB | 1.000 | 0.869 | -7.6% | 1.000 | 0.982 | -29.1% | 30.3% | yes |
| London circle | +3 dB | 1.000 | 0.947 | -6.2% | 1.000 | 0.985 | -11.3% | 34.4% | yes |
| Sydney sweep | -6 dB | 1.000 | 0.975 | +7.2% | 1.000 | 0.951 | -34.9% | 26.5% | yes |
| Sydney sweep | +3 dB | 0.951 | 0.927 | +11.3% | 1.000 | 0.974 | -29.5% | 42.5% | yes |

At 100 m, AUC ranged from 0.882 to 1.000, direction cosine from 0.869 to 0.975, and maximum absolute relative displacement error was 11.3%; therefore every preregistered metric-regime gate passed.

At 240 m, all ten groups retained direction cosine at or above 0.924, showed edge contact from 12.9% to 49.2%, and underestimated displacement by 11.3% to 51.4%. This supports the separation between directional observability and magnitude observability under finite aperture. The sole strict failure was Tokyo/+3 dB discrimination: the spoof median clock-centered residual was 0.601 versus multipath 0.392, reversing the desired score ordering even though direction remained 0.929. This is a state-dependent discrimination blind spot, not a loss of the fitted spoof direction.

## Same-stream 3/5/9-tap intervention

At 240 m, medians over all ten geometry-power groups were:

| Central taps | Recovered norm (chip) | Recovered / true | Absolute rel. error | Edge fraction | AUC | Direction cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.183 | 22.3% | 77.7% | 0.0% | 0.830 | 0.712 |
| 5 | 0.438 | 53.5% | 46.5% | 2.9% | 1.000 | 0.937 |
| 9 | 0.601 | 73.3% | 26.7% | 35.7% | 1.000 | 0.978 |

Three preregistered causal predictions passed: absolute error decreased, recovered norm increased, and nine-tap 100 m error was better than three-tap error. The first two 240 m trends held independently in 9/10 paired geometry-power groups; their only exception was the same Tokyo/+3 dB cell.

The preregistered edge prediction failed because the sign was physically misspecified. Edge contact increased monotonically from 3 to 5 to 9 taps in all 10/10 paired groups. With only three central samples, a far delayed replica is under-resolved and is often assigned an interior approximate delay. Wider observations reveal the far replica, increase recovered displacement, and drive the still-fixed ±0.5-chip template search to its boundary. The observed increase in edge contact is therefore consistent with, and more diagnostic of, finite-aperture saturation than the preregistered decrease. This interpretation is post-outcome and must be labeled exploratory rather than used to flip the frozen decision.

At 100 m, median absolute relative error was 42.0%, 5.0%, and 6.9% for 3, 5, and 9 taps. This suggests a useful engineering distinction: five central taps are sufficient and slightly more stable for the metric regime, while nine taps provide much better long-distance directional and magnitude observability.

## Early geometry-dependent boundary

The nine-tap AUC range was 0.514--0.882 at 40 m and 0.438--1.000 at 60 m. The ordering was not universally monotone: Seoul and London improved strongly by 60 m, whereas Tokyo degraded at 60 m before recovering at 100 m. This confirms that 40 or 60 m cannot be presented as a universal physical detection threshold; the defensible claim is a geometry-dependent transition followed by a robust 100 m metric band in this campaign.

## Paper-level interpretation

The strongest contribution supported by this experiment is a three-state observability result rather than a universal detector threshold:

1. an under-resolved early state whose discrimination boundary moves with LOS and motion geometry;
2. a metric state near 100 m where discrimination, direction, and displacement magnitude jointly generalize across all five geometries and both powers; and
3. a finite-aperture saturation state near 240 m where direction remains coherent but magnitude is negatively biased, edge contact rises with observable aperture, and an isolated high-power/sparse-geometry discrimination reversal can occur.

This is suitable as simulation receiver-RF evidence for a WCL submission, but it still requires an external real-spoof check and a real-multipath or defensible field proxy before making operational claims.

## Provenance and retention

- Release commit: `d2504f0e7b753bbfd78f688f0d8d1dccd1d17322`
- Config SHA-256: `b03093506b5a3b79ff296c640715f38cd44ad3fc50dc6454244a709a43eef2e2`
- Protocol SHA-256: `11bb2b756ccfe657e1a4f5d6292fff61fe271548467c1d61881b3bc345ef6d3b`
- Runner SHA-256: `06f2c460b72cbcc400537c9f1c5a29f9323820469e24f9c532131fffe1d5b86a`
- Result summary SHA-256: `256fab0adc9b1b6d7bb6cc545e59a006eb634676234bb4b579d704c2c2d845e2`
- Condition-aperture CSV SHA-256: `8aae91cf9070ab2bf1240e3810c20da4afb3a76e5e45082b0c26bdab55d21afe`
- Retained campaign files: 1,847 files, 3,576,550,596 bytes before the paper figure and compact result copy
- Campaign-created intermediate IQ remaining: 0
- Shared source IQ removed: no

The claim boundary remains five simulated location/time/LOS geometries, four motion classes, 25 MHz receiver RF, independent synthetic multipath, and a same-stream aperture intervention. It does not establish field-multipath, real-spoof, TEXBAT-multipath, universal-distance-threshold, or operational performance.
