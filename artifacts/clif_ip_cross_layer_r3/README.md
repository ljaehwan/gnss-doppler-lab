# CLIF-IP cross-layer R3 artifacts

## Verdict: **NO-GO**

R3 removes the R1 attack/refit leakage. Every learned normalizer, PCA, AR, Ridge, residual covariance, empirical tail, fusion covariance, hyperparameter, lag, and threshold is derived only from chronological cleanStatic train/validation. Attack recordings are transformed once with frozen state.

- **M1 kind:** M1-style surrogate over actual OAKBAT raw-IQ features; no reusable actual frozen M1 checkpoint was found.
- **Frozen M1 SHA-256:** `b19343d63af2aa942b911e2ad9cf51e62cb33402a84cad8c9dde63120eaf0fad`
- **Fit audit:** one fit, cleanStatic 0–240 s; os1–os4 and clean partitions transform only.
- **Provenance:** timestamp reconstructed (attack scenarios) / provisional (cleanStatic); 5 MHz interleaved int16 IQ and known canonical attack raw hashes are recorded. Exact receiver delay/common sample anchor is unavailable.

## Predictor result

Independent clean test:

- P0 MSE/MAE: 0.020101 / 0.105057
- P1 MSE/MAE: 0.021067 / 0.107648
- P2 MSE/MAE: 0.022739 / 0.111944
- P3 MSE/MAE: 0.021840 / 0.109716
- P3 vs P1: **-3.670%** (worse), incremental R² **-0.03670**

Therefore M1 does **not** add clean incremental information beyond B0 history in P3.

## Detector comparison

Mean os1–os4 ROC-AUC / PR-AUC / independent clean-test FPR:

- B0: 0.8414 / 0.9100 / 0.01460
- M1 surrogate: 0.7822 / 0.9001 / 0.01095
- P3 residual: 0.8895 / 0.9428 / 0.01095
- Full: 0.8879 / 0.9329 / 0.00730
- mean/Fisher fusion: 0.8313 / 0.9208 / 0.00365
- max fusion: 0.8279 / 0.9183 / 0.00365

Scenario Full ROC-AUC / PR-AUC / attack detection / first delay:

- os1: 0.6060 / 0.7950 / 0.0215 / 0.0 s (nonpersistent low utility; **failure**)
- os2: 1.000 / 1.000 / 1.000 / 0.0 s (**success**)
- os3: 0.9457 / 0.9366 / 1.000 / 0.5 s (**success**)
- os4: 1.000 / 1.000 / 1.000 / 0.0 s (**success**)

Full improves aggregate B0/M1 ranking but is slightly below P3 and does not establish universal superiority. os1 and negative incremental R² force No-Go.

## Destruction result

`alignment_destruction_metrics.json` performs region-local eight-epoch M1 block permutation in clean test, each pre-onset region, and each established attack region. It preserves sorted M1 marginals and recalculates P2, P3, and Full; no global circular shift is used. Effects vary by region, so destruction does not establish stable cross-layer correspondence.

## Files

- `config.json`, `provenance_manifest.json`, `frozen_m1_fit_summary.json`
- `predictor_comparison.csv`, `scenario_metrics.csv`, `fusion_comparison.csv`, `ablation_metrics.csv`, `lag_analysis.csv`
- `alignment_destruction_metrics.json`
- `per_epoch_scores_*.csv` (compact epoch-level summaries, no per-PRN raw dump)
- `test_summary.txt`, `plots/`
- `input_cache/` contains the newly extracted os1 M1 features and extraction manifest

## Paper-claim boundary

Can claim: a complete leakage-safe OAKBAT R3 evaluation, frozen-state invariants, good os2–os4 ranking, os1 failure, and no clean incremental P3 information. Cannot claim: physical causality, exact raw-sample alignment, actual checkpointed M1 performance, cross-recorder/TEXBAT generalization, or universal Full superiority.
