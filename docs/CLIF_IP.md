# CLIF-IP R3: leakage-safe cross-layer innovation prediction

## Scope

R3 is an observational OAKBAT experiment over actual `cleanStatic` and `os1`–`os4` derived data. It fixes the R1 leakage: normalization, PCA, AR, residual statistics, Ridge parameters, covariance, empirical tails, fusion calibration, lag/regularization selection, and thresholds use cleanStatic only. No attack recording is fit or used for selection.

## Frozen evidence

- **B0:** existing frozen shared PRN-local GRU, 12 past epochs of nine prompt-normalized taps to a signed nine-tap target. It receives no PRN identity. Every PRN/recording/split starts with empty history; tracked cardinality may vary.
- **M1:** **M1-style surrogate**, not an existing frozen M1 checkpoint. It uses actual 10 ms pre-correlation raw-IQ texture features at 0.5 s cadence. Exactly one fit on cleanStatic 0–240 s freezes normalization, PCA(8), AR(6), robust innovation center/scale, and Ledoit-Wolf covariance. All clean validation/test and os1–os4 calls are transforms with the same state/hash.
- os1 M1 features were newly extracted from its canonical raw IQ at 5 MHz. Extraction is not fitting.

## Splits and causality

Whole support containment is used: train 0–240 s, guard 240–250 s, validation/calibration 250–330 s, guard 330–340 s, independent clean test at/after 340 s. B0 uses strictly past B0 history; M1 uses past through current M1. Histories reset at every split and recording. Ridge P0 is the clean mean, P1 uses B0 history, P2 M1 history, and P3 both; P1/P3 share validation-selected lag and alpha.

Validation residuals define shrinkage covariance and per-PRN Mahalanobis distances. Epoch invariants are median, q90, top-3 mean, and tracked K. Full combines B0 marginal, M1 marginal, P3 residual, and causal concordance under normal-only robust scaling/covariance. Baselines include B0, M1, mean/max/Fisher, P0–P3, Full, `-M1`, `-B0history`, and `-concordance`. Thresholds are clean validation q99/q99.5; FPR is reported on independent clean test.

## Alignment destruction

Only clean test, each scenario pre-onset (<110 s), and established attack (>=130 s) are permuted. Eight-epoch M1 blocks are permuted inside each region—never with a global circular shift. M1 marginals are checked exactly, and P2/P3/Full are recalculated. The JSON reports score and squared-score proxy deltas plus permutation p-values.

## Results and interpretation

Validation selected six epochs (3 s) and Ridge alpha 10. Independent clean test MSE: P0 0.020101, P1 0.021067, P2 0.022739, P3 0.021840. P3 is **3.67% worse** than P1 (`incremental R² = -0.0367`), so M1 has no incremental clean prediction information under this protocol.

Across os1–os4, mean ROC-AUC/PR-AUC are B0 0.841/0.910, M1 0.782/0.900, P3 0.889/0.943, and Full 0.888/0.933. Full independent-test FPR is 0.00730. os2/os4 are detected perfectly by Full; os3 Full ROC-AUC is 0.946 and detection rate 1.0; os1 fails (ROC-AUC 0.606, attack detection 0.0215). Thus the decision is **No-Go** for a general or causal CLIF-IP claim despite good aggregate attack ranking.

## Provenance and claims

Raw identities/rates/formats are recovered where available, but exact B0 receiver processing delay and independently authenticated common start-sample anchors are absent. Alignment is timestamp-reconstructed, with scenarios graded reconstructed and cleanStatic provisional; grades are explicitly non-equivalent.

Permitted claim: R3 provides leakage-safe, frozen-transform OAKBAT evidence and complete negative incremental-prediction result. Not permitted: physical M1→B0 causality, exact sample-level alignment, cross-recorder/dataset generalization, actual frozen M1 performance, or universal Full superiority.
