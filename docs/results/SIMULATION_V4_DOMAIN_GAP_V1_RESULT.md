# Simulation-v4 to TEXBAT Clean Domain-Gap Audit v1

Date fixed: 2026-08-25

## Correction recorded 2026-08-25

This document remains the frozen historical decision for the original pilot,
but its claim that both receivers actually used 0.125-chip E/L spacing was
incorrect. The pilot was processed by stock `/usr/bin/gnss-sdr`, which
ignored the locally patched `tap_spacing_chips` key. TEXBAT used the patched
Method-A receiver and did use 0.125-chip spacing. The renderer now emits the
standard `early_late_space_chips` and `early_late_space_narrow_chips` keys, and
the simulation was reprocessed with receiver parity. See
[`SIMULATION_V4_NORMAL_CALIBRATION_V4_RESULT.md`](SIMULATION_V4_NORMAL_CALIBRATION_V4_RESULT.md).

The v1 **STOP** remains valid for the old artifacts; its numerical AUC is not a
clean estimate of the residual domain gap after receiver alignment. The later
v4 candidate is **CONDITIONAL**, not permission to scale or train a detector.

## Decision

**STOP: do not scale simulation-v4 or train a detector from it yet.** The
simulation normal windows remain strongly separable from both TEXBAT clean
recordings. This is a negative domain-fidelity result, not a detector result.

The frozen audit configuration is:

```text
configs/experiments/simulation_v4_domain_gap_gate_v1.json
SHA-256 6649df1cb5bfc7ef282e4bd5f0df1decc0ab4dfb7dc5175e28f016cd5adae0cc
```

TEXBAT `ds1`-`ds8` were not accessed. Only `cleanStatic` and `cleanDynamic`
were used. Consequently, the two clean recordings are now development-domain
references for this project, not untouched final validation recordings; the
spoofing scenarios remain reserved for a frozen detector evaluation.

## Frozen data selection

The simulation input was the qualified pilot feature dataset with SHA-256
`bdf96697bc40622db1db8aaf4fdf91f9ac9a1b7c25ce406a3df3acd268067cb2`.
Of its 2,503 rows, the audit retained 1,105 normal windows:

- 867 `steady-normal / steady_normal` rows
- 238 `normal-outage-recovery / post_recovery_normal` rows

The recovery pre-event prefix and the carry-off pre-event prefix were excluded
because they share authentic/noise samples with the steady run. Outage,
recovery-ramp, carry-off-transition, and carry-off-final windows were also
excluded. This avoids counting counterfactual siblings as independent normal
observations.

The TEXBAT Method-A receiver dumps were re-extracted using only their measured
inner E/P/L correlators. The TEXBAT receiver used 0.125-chip spacing. The
pilot manifest also recorded 0.125, but the later correction above establishes
that the stock simulation receiver ignored that patched-only key; the two sides
were therefore not physically aligned in this historical audit:

- `cleanStatic`: 1,837 windows, extracted CSV SHA-256
  `fe8f30be6c08dd58e18aad21269973180c67c0d88f9b614608663270bf3537a9`
- `cleanDynamic`: 1,806 windows, extracted CSV SHA-256
  `45241d0412f27229c847da964f7a836547ca412c5a09124b43ce43ede4683491`

Every source used 1.0-second windows with a 0.5-second stride and the same 11
E/P/L morphology and tracking-dynamics features.

## Preregistered engineering gate

The thresholds are a screening rule for deciding whether more simulation data
should be generated; they are not canonical statistical or publication
acceptance thresholds.

| Level | Grouped domain AUC | Median KS | Median robust median shift |
|---|---:|---:|---:|
| pass | <= 0.70 | <= 0.20 | <= 0.75 |
| conditional | <= 0.85 | <= 0.35 | <= 1.50 |
| stop | any conditional limit exceeded | any conditional limit exceeded | any conditional limit exceeded |

The domain classifier is a robust-scaled, class-balanced logistic regression.
Five-fold `StratifiedGroupKFold` keeps every paired-campaign/PRN group wholly in
one fold. Real groups are recording/PRN pairs. Each group contributes at most
128 time-ordered, evenly sampled windows. Thus temporally adjacent rows and
paired simulation scenarios cannot leak across train/test folds.

## Result

| Real-clean target | OOF separability AUC | Mean fold AUC | Median KS | Median robust shift | Gate |
|---|---:|---:|---:|---:|---|
| cleanStatic | 0.9869 | 0.9876 | 0.6069 | 1.3647 | stop |
| cleanDynamic | 0.9775 | 0.9850 | 0.3353 | 0.4718 | stop |
| combined | 0.9906 | 0.9918 | 0.4183 | 0.8760 | stop |

All 15 held-out folds remained highly separable. The lowest fold AUC was
0.9436, so the stop decision is not caused by a single unfortunate fold.
`cleanDynamic` passes the two univariate median limits but still fails decisively
on multivariate domain separability.

For the combined clean target, the largest real-scale median shifts were:

| Feature | Robust median shift | KS statistic |
|---|---:|---:|
| `sharp_narrow_std` | 3.2581 | 0.8315 |
| `near_sym_std` | 2.6233 | 0.7441 |
| `code_err_abs_mean` | 2.1068 | 0.6282 |
| `code_err_std` | 2.0476 | 0.6180 |
| `sharp_narrow_mean` | 1.7758 | 0.5726 |
| `doppler_std` | 0.8760 | 0.4183 |

The classifier's strongest standardized coefficient in all three comparisons
was `near_sym_std`; `prompt_mag_cv` and code-error statistics were also major
domain indicators. This points to a receiver/channel realism mismatch rather
than one threshold needing adjustment.

## Receiver-state evidence

| Source/state | Epochs | PRNs | Lock median | Fraction lock > 0.5 | Median C/N0 |
|---|---:|---:|---:|---:|---:|
| simulation steady | 222,726 | 14 | 0.0978 | 40.64% | 41.43 dB-Hz |
| simulation post-recovery | 90,505 | 7 | 0.9205 | 73.43% | 42.77 dB-Hz |
| TEXBAT cleanStatic | 198,066 | 13 | 0.9594 | 73.46% | 47.86 dB-Hz |
| TEXBAT cleanDynamic | 211,780 | 11 | 0.9396 | 70.83% | 44.09 dB-Hz |

The full steady simulation contains far more weak/unlocked tracking epochs and
a much broader C/N0 distribution than either clean recording. The post-recovery
subset is closer in aggregate lock behavior, yet the combined morphology and
tracking dynamics remain separable. The RF sample rates also differ (2.6 MHz
simulation versus 25 MHz TEXBAT), although the receiver tap spacing is aligned.

## Consequence and next experiment

Generating many variants from the current v4 contract would amplify the wrong
normal distribution. The next experiment must be a small normal-only calibration
sweep, not full dataset production:

1. vary simulation SNR/front-end bandwidth and receiver acquisition/tracking
   settings to match real-clean lock and C/N0 envelopes;
2. optimize only against `cleanStatic/cleanDynamic` normal-domain distances,
   without opening `ds1`-`ds8`;
3. rerun this exact gate on independent paired simulation seeds/epochs;
4. scale generation only if the multivariate AUC and distribution limits pass;
5. freeze the detector before the first spoof-scenario evaluation.

Only one independent simulation-v4 paired campaign exists today. Therefore this
audit cannot estimate cross-campaign simulation generalization, and its high AUC
must not be described as spoof-detection accuracy or WCL evidence.

## Reproduction

```bash
.venv/bin/python scripts/audit_simulation_v4_domain_gap.py \
  --config configs/experiments/simulation_v4_domain_gap_gate_v1.json
```

Ignored detailed outputs are under
`artifacts/simulation_v4_domain_gap_audit_v1/`. The tracked compact result is
`docs/results/simulation_v4_domain_gap_v1_summary.json`.
