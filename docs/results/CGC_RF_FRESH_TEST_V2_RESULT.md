# CGC RF fresh locked test v2 result

## Decision

`SUPPORTED`. All six preregistered gates passed in the single released execution. No threshold was fitted or applied, no selected pair was substituted, and no post-release tuning or retest was performed.

## Primary result

| Pair | Motion | Startup LOS PRNs | Multipath median residual | Spoof median residual | Separation |
|---|---:|---:|---:|---:|---:|
| fv2-static-01 | static | 13 | 0.4584 | 0.1297 | 0.3287 |
| fv2-straight-01 | straight | 13 | 0.5739 | 0.0740 | 0.4999 |
| fv2-sweep-01 | parallel-sweep | 11 | 0.8924 | 0.0358 | 0.8566 |

All three separations have the preregistered positive direction. Pair-block AUC is 1.000, median pair separation is 0.4999, and the descriptive bootstrap interval for the median is [0.3287, 0.8566]. Every scenario-pair contributed nine eligible one-second bins.

Clock-centering improved over the zero-referenced legacy residual on all three pairs. The descriptive serial-bin AUC increased from 0.7627 (legacy) to 0.9835 (clock-centered).

## Meaning

The complex-template delays produced by independent per-satellite multipath do not share one common receiver displacement pattern, so their LOS-plus-clock residual remains relatively high. Carry-off spoof delays are jointly induced by one counterfeit receiver trajectory and therefore fit the shared LOS geometry much better, producing a lower residual and higher detection score `-residual`.

This fresh test fixes the earlier v1 input-support failure: the three startup LOS counts are 13, 13, and 11, all above the frozen minimum of eight.

## Boundary

This result supports relative multipath-versus-carry-off-spoof discrimination in fresh simulated receiver-RF data. It does not establish an absolute alarm threshold, field false-alarm rate, operational performance, or real-data multipath-versus-spoof accuracy. TEXBAT evaluation is handled separately because TEXBAT DS1–DS3 contain spoof pre/post regions but no labeled independent-multipath class.

The immutable raw result is `artifacts/cgc_rf_fresh_test_v2/summary.json`, SHA-256 `9896826d939f383c2df1252e9e988731f84bb8c413450bf56fcaee4cd827662a`.
