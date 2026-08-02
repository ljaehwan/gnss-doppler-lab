# CLIF-IP synthetic-normal R4 artifacts

## Status: final 60-run campaign, real adaptation, and evaluation complete

All 60 receiver-backed normal simulations completed atomically (24 train, 3 validation, 3 synthetic test per domain; 83,202 B0 rows and 14,400 M1 rows). B0 and M1 in every run share the same IQ SHA-256. Raw IQ/MAT transients were removed only after `_SUCCESS` publication.

## Training contracts

- **R0 OAKBAT:** exact immutable R3 result, read-only. **R0-TEX-real-only:** reported unavailable rather than fabricated because no complete chronological Method-A training-node export exists for a new real-only CLIF comparator.
- **S0:** shared PRN-local GRU trained on exactly 24 synthetic runs/domain; 3 synthetic validation runs select lag/alpha/calibration; 3 synthetic test runs remain independent. No real rows or attack rows enter fitting. M1 is fit once across run-reset synthetic histories.
- **S1:** initialized from the exact S0 checkpoint and genuinely fine-tuned on real cleanStatic whole windows in 0–240 s. M1 is synthetic-basis pretrained and its AR residual/covariance state adapted on real clean train only. Threshold/calibration uses real cleanStatic 250–330 s only; independent clean test is >=340 s. OAK adaptation used 5,258 B0 and 480 M1 rows; TEX used 5,031 B0 and 480 M1 rows. Before/after weight hashes differ in both domains and checkpoints record allowed live source hashes, row counts, roles, and zero attack rows.

Raw B0 taps are **Method-A magnitudes** in exact producer layout `E4,E3,E2,E,P,L,L2,L3,L4`. “Signed 9D innovation” always means `x-xhat`; it does not mean raw signed-complex taps. TEX cleanStatic's receiver export was converted format-aware from the complete 9-tap IQ NPZ by taking magnitude and prompt-relative morphology; it was not mixed as signed-complex input.

## Main observed results

- **S1 OAKBAT Full** (os1–os4 mean): ROC-AUC 0.916081, PR-AUC 0.983415, independent clean FPR 0.003650, established-attack detection 0.748649.
- **S0 OAKBAT Full:** ROC-AUC 0.860290 but synthetic-native calibration transferred catastrophically (independent real-clean FPR 1.0), directly demonstrating the domain gap rather than hiding it.
- **S1 TEXBAT DS4 Full:** ROC-AUC 0.139189, PR-AUC 0.192981, independent clean FPR 0.025455. DS1–DS3 S0/S1 are explicit NA because compatible Method-A node rows are absent; no residual export was silently treated as trainable node targets.
- P3 did not beat matched P1 on validation: S1 OAK MSE 9.6e-5 vs P1 9.2e-5; S1 TEX 3.69e-4 vs 3.62e-4. Thus the claimed cross-layer predictive increment is not supported.
- Actual domain-gap RMSE ratios include S0 TEX M1 14.9336x (above 5.5x) and S0 OAK M1 3.9157x; complete SMD/Wasserstein/MMD values are in `domain_gap_metrics.csv`.

## Alignment destruction and limitations

The evaluation executed 199 reproducible, region-local, 8-epoch block permutations over 24 available clean/pre/post regions (4,776 raw replicates; p-value resolution 0.005), preserving M1 marginals and support. The raw replicate CSV and summary JSON are retained.

TEX DS1–DS3 have residual score exports but not compatible full Method-A magnitude node tables. Using those residuals as B0 training targets would violate the producer/model contract, so those comparisons and R0-TEX are explicitly NA. DS4 is the only feasible paired TEX attack path. CN0 is available in producer nodes; AGC is unavailable; power/PSD/autorrelation are represented by actual M1 IQ features. Fields not exposed by the source are not inferred.

See `config.json`, `training_summary.json`, `scenario_metrics.csv`, `predictor_comparison.csv`, `domain_gap_metrics.csv`, `alignment_destruction_metrics.json`, `provenance_manifest.json`, plots, and `test_summary.txt`.
