# CLIF-IP cross-layer R3 artifacts

## Verdict: **NO-GO**

1. **M1 definition.** M1-style surrogate over actual OAKBAT raw-IQ features (not an original checkpointed M1). Frozen hash `e192ea2d410d836dc2764b1d9c576be6680f8c51e5d358f4223e03da03d07887`; exactly 1 clean fit; 480 whole-contained fit rows.
2. **Leakage/reset.** Raw clean splits are sliced by whole-window containment before M1 transform and B0 residual construction. Train, validation, test, and every attack recording begin with empty M1/B0 history.
3. **Predictor protocol.** Shared signed 9-tap P0--P3 models contain no PRN IDs and support variable cardinality. All metrics use identical lag-trimmed target IDs. StandardScaler+Ridge used the same lag candidates (2, 4, 6); common lag=6, with validation-only model-specific alphas {'P1': 100.0, 'P2': 100.0, 'P3': 100.0}. Parameter counts are in `predictor_comparison.csv`.
4. **Independent prediction.** P1 test MSE=0.0209227215; P3=0.0216904497; P3-vs-P1=-3.669%. No incremental improvement; this is a failure for the cross-layer prediction claim.
5. **Calibration.** Each predictor stores validation residual mean plus Ledoit-Wolf covariance and scores `e-mu`. Full and every ablation refit robust scale and shrinkage covariance on its exact validation-only component set.
6. **Detection.** Full mean ROC-AUC=0.8898, PR-AUC=0.9481, independent clean-test FPR=0.0153. q99/q99.5 and the 1% target threshold are recorded; attack onsets and delays use score `available_s`.
7. **Alignment destruction.** Region-local 8-epoch permutations preserve M1 marginals. Actual P2/P3 9-tap MSE and Full score deltas have 19-repeat p-values and 95% permutation intervals.
8. **Provenance/artifacts.** Source CSV and checkpoint SHA-256 values were computed in-run; accessible 9.6-GB attack IQ files have live stat verification against explicitly identified cached canonical hashes. cleanStatic has no trustworthy canonical raw hash and remains honestly null/provisional. Source commit/dependencies and artifact checksums are in the manifest.
9. **Claim boundary and failures.** Can claim a leakage-safe OAKBAT os1--os4 evaluation of this surrogate and report the regenerated outcomes. Cannot claim physical causality, exact sample alignment/receiver delay, original-M1 performance, cross-corpus generalization, or universal fusion superiority. `NO-GO` follows the regenerated P3/full criteria; scenario-level failures remain visible in CSVs.

## Files
`config.json`, provenance/frozen summaries, predictor/hyperparameter/scenario/fusion/ablation/lag CSVs, per-epoch CSVs, destruction JSON, actual residual and timeline plots, and `test_summary.txt`.
