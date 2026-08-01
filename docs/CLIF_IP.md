# CLIF-IP R3: independent clean-test cross-layer innovation prediction

## Scope and frozen evidence

R3 is an observational OAKBAT experiment over actual `cleanStatic` and `os1`–`os4` derived data. No attack labels, attack recordings, or future clean-test rows are used for fitting or hyperparameter selection.

- **B0:** **existing B0 architecture retrained/frozen on CLIF clean train**. The shared PRN-local GRU retains the existing 12-epoch, signed nine-tap architecture/loader and no PRN-ID input, but uses a new CLIF-only checkpoint. All selected B0 rows are whole-contained in cleanStatic 0–240 s: 5,258 selected rows, of which 4,302 train-PRN rows fit the model and standardizer and 956 clean-train-region PRN-holdout rows select the epoch. Maximum input `window_end_s` is 239.71602. Checkpoint SHA-256: `c9b004179c1cea486dec581939af5399284b5b3cfffa726dd7ac8dc7b75c65ac`. The previously published B0 baseline was not changed and may remain a separate comparator.
- **M1:** an M1-style surrogate, not an existing frozen M1 checkpoint. One fit over 480 whole-contained cleanStatic 0–240 s rows freezes normalization, PCA(8), AR(6), robust innovation statistics, and covariance. Frozen state hash: `e192ea2d410d836dc2764b1d9c576be6680f8c51e5d358f4223e03da03d07887`. M1 scores are nonnegative.

## Splits, resets, and calibration

Whole support containment defines train 0–240 s, validation 250–330 s, and independent clean test 340 s onward. B0, M1, and downstream predictor histories reset at every split and recording. Validation/test and os1–os4 are transform-only. Consequently, both the dedicated B0 and all downstream states are independent of clean-test 340+ s.

P0–P3 use equal lag-trimmed target support. P1 uses past B0 residual history, P2 uses M1 history through current epoch, and P3 uses both. Validation selected a two-epoch (1 s) history lag and alpha 100 independently for P1/P2/P3; no attack labels participate.

Exact validation-only component recalibrations are:

- Full: `[B0, M1, P3, concordance]`
- `-M1`: `[B0, P1]`; M1 scalar, P3, and concordance are absent.
- `-B0history`: `[M1, P2]`; B0 marginal/history and concordance are absent.
- `-concordance`: `[B0, M1, P3]`.

## Regenerated results

Independent clean-test MSE is P0 0.010272, P1 0.010457, P2 0.010487, and P3 0.010608. P3 is 1.450% worse than P1 (`incremental R² = -0.014503`), so this run does not show incremental clean prediction information from M1.

Across os1–os4:

- B0 mean ROC-AUC/PR-AUC: 0.8478/0.9171; independent-test FPR 0.003774.
- M1: 0.9227/0.9699; FPR 0.011321.
- P3: 0.8841/0.9353; FPR 0.033962.
- Full: 0.8943/0.9562; FPR 0.022642.

Full fails os1 (ROC-AUC 0.5823, attack detection 0.1171) while os2/os4 are perfect and os3 ROC-AUC is 0.9949. The verdict remains **NO-GO** for a general or causal CLIF-IP claim.

## Alignment, timing, and permutation interpretation

`lag_analysis.csv` separates `predictor_history_lag_*` from `alignment_shift_*`; a predictor history length is not a causal alignment shift. Region-local eight-epoch M1 block permutations preserve M1 marginals and recompute actual signed nine-tap P2/P3 prediction MSE and Full scores. Nineteen repetitions give a minimum p-value resolution of 1/(19+1) = 0.05. Reported 95% bounds are quantiles of the permutation-delta distribution, not sampling-confidence intervals.

Alarm timing uses score `available_s`, but receiver processing delay and an independently authenticated common sample/time anchor are unavailable. Therefore these values are processing-time proxies, not physical detection-delay measurements.

## Provenance and claim boundary

B0/M1 source CSV and checkpoint SHA-256 values were computed live. For os1–os4 raw IQ, the manifest records the current path, size, and mtime together with a cached canonical digest; those large-file digests were **not reverified this run**. cleanStatic raw-IQ identity remains provisional with no trusted digest.

Permitted claim: leakage-safe, end-to-end independent clean-test evidence for the dedicated retrained B0 plus frozen downstream transforms on these OAKBAT recordings, including the negative incremental-prediction result. Not permitted: physical M1→B0 causality, exact sample alignment, physical delay, original frozen-M1 performance, cross-corpus generalization, or universal Full superiority.
