# CLIF-IP synthetic-normal R4 — corrected complete experiment

## Scope and timing contracts

This directory is the complete reported R4 experiment over 60 receiver-backed, attack-free synthetic runs (24 train, 3 validation, 3 independent synthetic test per target domain). Existing R3 and other baselines were not modified.

Evaluation uses **score availability time**, not nominal window start:

- **OAKBAT:** nominal onset 120 s; stable pre `t < 110`; guard/transition `110 <= t < 130` excluded; established post `t >= 130`. Detection delay remains relative to 120 s, so a valid established-post alarm has at least about 10 s delay.
- **TEXBAT:** nominal onset 100 s; stable pre `30 <= t < 90`; transition `90 <= t < 110` excluded; established post `t >= 110`.

## Data and model semantics

Raw B0 taps are nine **Method-A prompt-relative magnitudes** in `E4,E3,E2,E,P,L,L2,L3,L4` order. Signed 9-D quantities are prediction innovations `x - xhat`; raw taps are not claimed to be signed complex values.

DS1–DS3 B0 nodes are reconstructed from their existing `complex_iq,time_s,prn,...` NPZ exports using the same converter as cleanStatic: `abs(complex_iq)/abs(prompt)`, then arithmetic aggregation by `(PRN, floor(time_s*2)/2)`. This is compatible for the nine magnitude/prompt taps only; auxiliary historical Method-A fields and exact producer-window aggregation cannot be reconstructed. A cleanStatic comparison found 1,767 matched nodes / 15,903 tap values, MAE 0.006369 and maximum absolute difference 0.081885, so this is explicitly **reconstructed equivalence, not numerical or byte identity**. Each conversion has source/output hashes and an aggregation caveat in its adjacent provenance JSON.

DS1 M1 was extracted format-aware from the live 25 Msps little-endian interleaved int16 IQ (`ds1.bin`). DS2–DS4 use existing live M1 feature files with recorded hashes. Power, PSD and autocorrelation are represented in M1. AGC telemetry is unavailable and is explicit NA, not inferred.

## Regimes and leakage controls

- **R0 OAK:** exact immutable historical R3 rows only, labeled `historical_r3_noncomparable`; they must not be interpreted as the R4 protocol.
- **R0 TEX:** real-only pipeline: cleanStatic train 0–240 s, validation/calibration 250–330 s, independent clean test >=340 s; evaluated on DS1–DS4.
- **S0:** synthetic train only; no real or attack rows fit any component.
- **S1 B0:** initialized from the exact recorded S0 checkpoint and fine-tuned only on real clean train 0–240 s; before/after hashes differ.
- **S1 M1:** synthetic normalization and PCA loading basis remain bit-identical; only AR coefficients and innovation center/scale/shrinkage covariance adapt on real clean train. Basis-before/basis-after hashes match and AR hashes differ.
- **S1 P1/P2/P3:** frozen S0 Ridge prediction plus a nonzero real-clean residual-correction Ridge. Checkpoints/audits retain S0 base coefficient and correction hashes. Attack fit rows are zero.

P0–P3 use the same signed 9-D target and support. `predictor_comparison.csv` contains clean validation and independent clean-test MSE, MAE, sample count, support hash, per-tap errors, P3-vs-P1 improvement and incremental R2. PRN-level target/prediction CSVs and epoch-level scores are retained for every clean split and scenario.

Residual components are centered on validation means and scored with deterministic shrinkage covariance. `Full = [B0 marginal, M1 marginal, P3 residual, concordance]`, where concordance is positive-tail co-elevation `max(B0z,0)*max(M1z,0)`, not discordance or absolute difference. Full and pair ablations use validation-only covariance fusion.

## Full-score scenario results

Values below are ROC-AUC / PR-AUC. R4 values share the corrected timing protocol; historical R0 OAK does not.

- **Historical R0 OAK Full:** os1 0.582318 / 0.833644; os2 1.000000 / 1.000000; os3 0.994881 / 0.991178; os4 1.000000 / 1.000000.
- **S0 OAK Full:** os1 0.466136 / 0.770115; os2 1.000000 / 1.000000; os3 0.989993 / 0.983907; os4 0.972743 / 0.993225. Independent real-clean FPR is 1.0, exposing failed synthetic calibration transfer.
- **S1 OAK Full:** os1 0.475593 / 0.755995; os2 1.000000 / 1.000000; os3 0.955557 / 0.953206; os4 1.000000 / 1.000000. Independent clean FPR is 0.032847.
- **R0 TEX Full:** DS1 0.976714 / 0.996400; DS2 0.999916 / 0.999986; DS3 0.986573 / 0.997920; DS4 0.926351 / 0.918509.
- **S0 TEX Full:** DS1 0.972636 / 0.995795; DS2 0.999389 / 0.999900; DS3 0.984194 / 0.997540; DS4 0.896622 / 0.879482. Independent real-clean FPR is 1.0.
- **S1 TEX Full:** DS1 0.981466 / 0.997095; DS2 0.999808 / 0.999968; DS3 0.987673 / 0.998092; DS4 0.928604 / 0.900035. Independent clean FPR is 0.036364.

The prediction result is mixed and reported without a positive-effect claim. On independent clean test, P3 vs P1 MSE is: S0 OAK 0.000484 vs 0.000349; S1 OAK 0.005354 vs 0.005406; S0 TEX 0.001217 vs 0.001162; S1 TEX 0.020292 vs 0.020283; R0 TEX 0.000368 vs 0.000366. Only the small S1 OAK improvement is positive.

## Domain gap and destruction

M1 domain-gap rows use unique `(run_id,t)` units rather than PRN-duplicated values. Standardized Wasserstein, SMD, exploratory unbiased MMD, actual/synthetic RMSE ratio, and the old 5.5 comparator are reported. Eight-epoch run/block summaries are retained separately. Because there is only one real cleanStatic recording per domain, recording-level uncertainty is explicit NA; blocks are not misrepresented as independent recordings. Notable M1 RMSE ratios are S0 OAK 3.9711, S1 OAK 0.1944, S0 TEX 14.9352, S1 TEX 1.1119, and R0 TEX 0.0234.

Alignment destruction contains 45 clean/pre/post regime-domain regions × P2/P3/Full. Every region has 199 reproducible 8-epoch block replicates (26,865 raw rows; p-value resolution 0.005). M1 score and innovation vector are shuffled jointly within `(run_id,PRN)` boundaries; support and marginals are preserved. Each replicate rebuilds actual frozen P2/P3 9-D predictions and re-scores Full with frozen validation calibration—there is no scalar proxy or first-dimension shortcut.

## Reproducibility and repository limits

`synthetic_bundle_ledger.csv` contains every run ID, row counts, B0/M1 output hashes, IQ/source/generator hashes, receiver hashes and a canonical leaf hash; `provenance_manifest.json` records the aggregate ledger/Merkle hash, all real evaluation input hashes, source tree and code-file hashes. Counts are calculated from live bundles by the finalizer, never hard-coded.

A clone contains code, ledgers, summaries, per-epoch scores/predictions, plots and provenance, but **does not contain approximately 3 GB of ignored per-run feature bundles**. The ledger is sufficient to verify those retained local files; it cannot recreate absent feature bytes by itself.

Primary artifacts: `config.json`, `synthetic_run_manifest.csv`, `synthetic_bundle_ledger.csv`, `generation_summary.json`, `impairment_distribution.json`, `training_summary.json`, `predictor_comparison.csv`, `scenario_metrics.csv`, `domain_gap_metrics.csv`, `domain_gap_group_summaries.csv`, `alignment_destruction_metrics.json`, `alignment_destruction_raw_metrics.csv`, `per_epoch/`, `predictions/`, `evaluation_provenance.json`, `provenance_manifest.json`, `checksums.json`, plots, and `test_summary.txt`.
