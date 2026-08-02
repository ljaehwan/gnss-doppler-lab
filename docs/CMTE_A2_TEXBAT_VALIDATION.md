# CMTE-A2 TEXBAT validation protocol

> Protocol skeleton only. This file contains no DS7/DS8 inference, score, alarm,
> metric, result, or interpretation.

## Frozen inputs

- Immutable preregistration: `docs/CMTE_A2_PREREGISTRATION.md`
- Machine-readable preregistration: `configs/cmte_a2_preregistration.json`
- Implementation/freeze commit: record before confirmatory execution
- State/input/checkpoint/scaler/Qcal/threshold/code checksums: record in the freeze manifest
- DS8 preparation failure policy: primary result is explicit `NA`; no historical-input fallback

## Execution order

1. Run focused and related regression tests and record the test log.
2. Commit generator code only; require a clean tree.
3. Run normal-only training/calibration/threshold generation.
4. Freeze state, code, source, and prepared-input checksums.
5. Run development tier with exactly DS1–DS4, never DS7/DS8 paths or tokens.
6. Verify the preregistration commit is an ancestor, freeze checksums are valid,
   execution source commit matches, and the tree is clean.
7. Run confirmatory tier once with exactly DS7–DS8. The confirmatory path loads
   frozen state and thresholds only and cannot call training, fitting, or threshold APIs.
8. Publish primary and diagnostic operating points separately.

## Required artifacts

- `README.md`, `preregistration.json`, `config.json`
- training, calibration, and thresholds records
- development, confirmatory, and baseline metric CSVs
- moving-block bootstrap CSV with CI or machine-readable `NA` reason
- per-epoch and per-PRN diagnostics
- provenance, checksums, test summary, and plots

## Fixed score and operating point

- Signed standardized residual: `r = x - xhat`, tap order
  `E4,E3,E2,E,P,L,L2,L3,L4`
- Nonconformity: frozen full-shrinkage squared Mahalanobis
- Conformal p-value: inclusive ties and finite-sample `+1`
- Epoch score: exactly `mean_i[-ln(p_i)]`
- Primary threshold: normal threshold-role NumPy-style higher q99.5
- Alarm: strict `score > threshold`
- q99-higher and empirical target-1% are diagnostics only
- No mixture e-value, e-CUSUM, capital/restart, sequential score, or online adaptation

## Timing and metrics

For DS7/DS8, onset is 110 s; stable pre is `[30,90)`, transition is
`[90,110)`, established is `>=110`, and phase sensitivity intervals are
`[110,130)`, `[130,150)`, and `>=150`. Availability is `window_end_s`.
Boundary-crossing windows are excluded from both adjacent phases. DS1–DS4 use
existing authoritative metadata rather than a hard-coded onset.

Report ROC-AUC, PR-AUC, independent-clean FPR, stable-pre FPR, rising-edge
false-alarm events/minute, occupancy, post and persistent detection rates,
first and persistent-3 delays/censoring, score summaries, tracked count, and
threshold for CMTE-A2 and all frozen comparators.

## Bootstrap

Use 2,000 replicates, seed 20260802, 20-epoch (10 s) whole blocks. Blocks must
not cross phase, physical recording, segment/channel-derived cadence chain, or
gap. ROC resamples stable and post chains independently and combines them.
There is no IID fallback. Every required stratum needs at least two complete
blocks; otherwise report `NA` and the insufficient-stratum reason. A one-
recording interval is explicitly a conditional temporal CI.

## Results

Intentionally blank until the one-shot confirmatory campaign is executed from
a verified freeze.
