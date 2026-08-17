# MCTD Stage-0 Static

Final verdict: `NO_GO_MCTD_PHYSICAL_HYPOTHESIS`.

Slow uses DLL/PLL 0.5/10 Hz; fast uses 2/25 Hz. All other receiver settings are identical. The identical-loop control uses 0.5/10 Hz on both sides.

Phase A passed source equality, bit-exact within-configuration replay, stable common support, and exact identical-loop collapse on TEXBAT and OAKBAT cleanStatic. Slow/fast share authenticated raw IQ, raw range, PRN assignment, and the same scenario-specific handoff state.

Clean models use chronological train/validation/calibration/holdout roles with 5 s guards, robust median centers, Ledoit-Wolf covariance, clean calibration q99 thresholds, PRN median pooling, non-overlapping 100 ms blocks, and three-consecutive-block alarms.

## Core Full results

|Scenario|pAUC<=5%|pre-onset FPR|attack detection|onset delay s|
|---|---:|---:|---:|---:|
|DS3|0.12860805936214062|0.08527131782945736|0.19887872528769548|1.0|
|DS7|0.224159402241594|0.0|0.1095890410958904|3.9000000000000057|
|OS3|0.2203968253968254|0.0|0.05277777777777778|0.5|
|OS4|1.0|0.04285714285714286|0.9997222222222222|0.09999999999999432|

A0/A1/A2/A3/A4/A5/Full common-support results are in `ablation_metrics.csv`. Exact B0 is `UNAVAILABLE`: rerunning it on native MCTD support would change the frozen B0 contract, and historical CSV values were not copied as MCTD results.

Configuration collapse is in `configuration_collapse_metrics.json`. Gain/phase/Prompt/nav-sign invariance diagnostics and honestly unavailable raw-IQ AWGN, C/N0, clock-drift, and multipath controls are in `physical_controls.json`. Because the required raw-IQ nuisance controls and Full pairing-destruction proof are unavailable, no unique physical contribution or Stage-1 promotion is claimed.

The experiment used no post-attack bandwidth, feature, threshold, pooling, or GO-criterion changes.

Exactly one recommended next action: stop MCTD Stage-1 and retain this frozen Stage-0 bundle as the negative-result record.
