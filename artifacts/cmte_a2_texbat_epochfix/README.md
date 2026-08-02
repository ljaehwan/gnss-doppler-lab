# CMTE-A2 canonical multi-PRN epochfix

## Status

**Developmental/exploratory post-exposure diagnostic — NO-GO.**

The preserved `artifacts/cmte_a2_texbat_ds78/` PRIMARY INVALID result is not modified, deleted, or replaced. DS7 and DS8 were already exposed and are not confirmatory holdouts or preregistered primary evidence.

## Why the previous experiment was invalid

The previous implementation grouped channel-specific floating `window_end_s` values by exact equality. Every stored event therefore had `tracked_prn_count=1`, so it did not implement:

```text
S_A2(t) = mean_i[-log(p_i(t))]
```

This directory uses a separately committed canonical decision-grid policy.

## Exact epoch policy

- grid origin: recording-relative `0.0 s`
- grid stride: `0.5 s`
- availability source: `window_end_s`
- mapping: first grid time at or after residual availability (`causal ceil`)
- timestamp tolerance: `1e-9 s`
- maximum residual age: `0.55 s`
- one latest eligible residual per PRN and decision epoch
- exact tie: latest timestamp, then deterministic stable row-content SHA-256
- future residuals rejected
- stale residuals rejected
- recordings processed independently
- history/segment/channel changes replace older rows when a new residual becomes available
- cadence gaps expire through the fixed maximum age
- PRN identity used only for grouping/diagnostics, never as a feature
- input-row and PRN-order permutation invariant

`window_bin_s` is not used because it can precede residual availability. `target_window_index` is not used as a cross-channel epoch ID because it is history/channel local.

Policy commit: `c57526fe6fe461075ff0889bd5ffbe676157e21d`

Execution source commit: `1fc5f710c27bd1c179df565132cdcda4df62c7f5`

## Frozen inputs reused

The runner verifies the original invalid artifact checksums and frozen state checksums before reading data. It reuses without refitting:

- chronological B0 checkpoint;
- scaler and B0 split;
- per-PRN signed 9D residuals;
- shrinkage covariance and conformal `Q_cal`.

Frozen B0 checkpoint SHA-256:

```text
44bc85320e4fdf6ffdff3b4c12941a1f90d39832d4b13707ecb4a0317f936fa0
```

The existing `gnss_doppler_lab.cmte_a2.b0_exact_scores` function is reused. Its prior historical evaluator equivalence audit reported maximum absolute error `3.55e-15` with identical strict alarms.

Recomputed on the new grid:

- cleanStatic `[300,330)` A2/A0/B0 scores;
- q99, q99.5, and target-1% thresholds;
- independent clean test scores/FPR;
- DS1–DS4/DS7/DS8 scores;
- B0-Exact/B0-Enhanced event streams;
- clean-test matched-FPR diagnostics.

No attack data was used for model fitting, primary thresholding, or hyperparameter selection.

## Actual tracked PRN count

- cleanStatic threshold: median `10`, min `10`, max `10`
- cleanStatic independent test: median `10`, min `7`, max `11`
- DS1: median `10`, min `8`, max `11`
- DS2: median `9`, min `7`, max `10`
- DS3: median `4`, min `2`, max `11`
- DS4: median `10`, min `8`, max `11`
- DS7: median `3`, min `1`, max `10`
- DS8: median `9`, min `8`, max `10`

All scenario medians exceed one; the timestamp aggregation bug is fixed. Full histograms are in `tracked_prn_count.csv`.

## Thresholds and independent clean FPR

The clean threshold interval produced only 46 usable multi-PRN epochs. With NumPy `higher`, q99, q99.5, and target-1% all equal the maximum order statistic.

- CMTE-A2 q99.5 threshold: `1.2189356312`
- B0-Exact q99.5 threshold: `1.7035537525`
- CMTE-A2 independent clean FPR: **8.9888%**
- B0-Exact independent clean FPR: **3.7453%**

Both exceed the 1.5% diagnostic target; CMTE-A2 is substantially worse.

## q99.5 comparison

Values are percentages except delay.

- DS1 — A2/B0 stable-pre: `31.63/2.04`; post detection: `94.60/95.15`; delay: `1.5/1.5 s`
- DS2 — stable-pre: `9.18/0.00`; post: `97.34/97.34`; delay: `11.0/11.0 s`
- DS3 — stable-pre: `1.02/0.00`; post: `94.68/94.40`; delay: `20.5/21.5 s`
- DS4 — stable-pre: `8.16/1.02`; post: `72.73/80.00`; delay: `3.0/4.5 s`
- DS7 — stable-pre: `24.58/8.47`; post: `85.00/63.44`; persistent: `93.75/83.75`; delay: `1.5/28.0 s`
- DS8 — stable-pre: `4.24/0.00`; post: `93.87/91.64`; persistent: `97.65/96.08`; delay: `12.5/17.5 s`

DS7's apparently faster A2 delay is not a valid benefit because A2 was already alarming before onset. DS8 shows a small detection/delay direction improvement, but clean FPR remains uncontrolled.

## Matched clean-FPR diagnostic

Comparator thresholds were selected from independent clean-test scores only to match A2's observed `8.9888%` FPR. This is diagnostic, not an independent operating point, and attack data was not used.

- DS7 — A2/B0 stable-pre: `24.58/13.56`; post: `85.31/67.81`; persistent: `93.75/86.25`; delay: `1.5/22.5 s`
- DS8 — stable-pre: `4.24/0.00`; post: `93.87/93.73`; persistent: `97.65/97.65`; delay: `12.5/16.5 s`

The detection direction is favorable on DS7/DS8, but it is obtained at an unacceptable 8.99% clean FPR and with substantial DS7 pre-onset alarms. It does not rescue the model.

## Diagnostic criteria

- independent clean FPR <=1.5%: **FAIL** (`8.9888%`)
- every scenario stable-pre FPR <5%: **FAIL** (DS1, DS2, DS4, DS7 fail)
- every scenario median tracked PRN count >1: **PASS**
- DS7 or DS8 shows some B0-Exact improvement direction: **PASS, exploratory only**
- no catastrophic degradation on the other scenario: **PASS under the recorded rule**
- improvement direction at matched FPR: **PARTIAL PASS**, but false-alarm control fails

## Tests

```text
CMTE-A2 epochfix policy/runner tests: 12 passed
Focused epochfix + related TEXBAT B0 tests: 53 passed in 4.66 s
```

Tests cover jittered timestamp coalescing, one residual per PRN, no future data, row/PRN permutation invariance, floating-jitter stability, variable cardinality, recording/history/segment/channel/cadence-gap behavior, actual DS7 median N>1, manual mean agreement, checksum schemas, and clean/scenario N histograms.

## Scientific judgment

**NO-GO as a paper detector candidate in the present form.** Multi-PRN aggregation is now implemented correctly, but normal-score calibration does not transfer to the independent clean tail and multiple scenarios have excessive stable-pre alarms. B0-Exact remains better controlled.

Remaining limitations:

- all corrected DS1–DS8 results are post-exposure exploratory;
- only one cleanStatic normal recording supports calibration/test;
- only 46 threshold epochs make q99.5 a maximum-order-statistic operating point;
- DS4 remains a mixed-producer sensitivity;
- DS7 has lower and time-varying PRN cardinality;
- matched-FPR uses the clean test and is diagnostic only;
- no new external confirmatory dataset was evaluated.

Even if future tuning improves this diagnostic, new external data is required before any SCI-level confirmatory claim.
