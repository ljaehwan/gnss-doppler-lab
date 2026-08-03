# AMCF-R1 Shape-Only campaign

**Primary q99 decision: NO-GO**
Status: PRIMARY COMPLETE. Any failed mandatory criterion yields **NO-GO** and **AMCF WCL no-go**.

## Single hypothesis

**Is Prompt-normalized correlator complex shape more useful for detection than magnitude shape?** This is the only hypothesis.

## Context and leakage controls

This Shape-Only study removes the prior AMCF-R1 error/context and active-query branches. C/N0 is excluded. Prompt is used only for the low-P QA gate and complex normalization; Prompt itself is never a model feature, target, score, or context input. No attack data fits a model, scaler, conformal calibration, threshold, or hyperparameter.

## Fair model comparison

- Complex: 4 features/tap, 32 total side features, hidden 32, 19208 trainable parameters.
- Magnitude: 2 features/tap, 16 total side features, hidden 32, 19012 trainable parameters.
- Absolute difference: 196; min-denominator difference: 1.030928% (limit <=5%; PASS).
- No zero/duplicate features: True. Shared GRU/decoder trunk; representation-specific adapter/head only. B0 Exact is the frozen comparator.

## All 12 seed convergence

| Variant | Seed | Best epoch | Epochs | Updates | Early stop | Converged | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---|---|---|
| complex_EPL_seed101 | 101 | 47 | 68 | 2584 | True | True | 6fc3d118d55d093a2c4ba3bd635d8aff9db2ac542e09d9a4d3069c45d2d83658 |
| complex_EPL_seed202 | 202 | 26 | 47 | 1786 | True | True | c3ce090ff041d054d0e3b62aaf40a70c897ae0a03f28f832525fdd954378a3f7 |
| complex_EPL_seed303 | 303 | 39 | 60 | 2280 | True | True | 6f141673f9f6d7463a362c34a417f99fff2c643c9b46690b9f3fa547c75cf718 |
| complex_all9_seed101 | 101 | 17 | 38 | 5700 | True | True | dd3ac0fb67f3d4a366cb291cb91b321e1948157521bb6f9a2ed6e919ba097ef3 |
| complex_all9_seed202 | 202 | 15 | 36 | 5400 | True | True | 6b4fb5b169fbb25a75c1689d49da068108df51a8e6e61180641892d2c6df5eec |
| complex_all9_seed303 | 303 | 25 | 46 | 6900 | True | True | e4cb5d970207ff4dcb754185637a0a36a16e4a657493dd45a62f89bbc1fcac34 |
| magnitude_EPL_seed101 | 101 | 32 | 53 | 2014 | True | True | f3ad061505b2c5720d6becc4992c8a7fdeca01316f88fd600c27a4573cd06d5d |
| magnitude_EPL_seed202 | 202 | 26 | 47 | 1786 | True | True | bfaaad237f74b360762d21c58e2b9940ad12f1455cbe2432d9e7157df28e1064 |
| magnitude_EPL_seed303 | 303 | 27 | 48 | 1824 | True | True | 38b5e4085c44a5be97602aa597ed8ee84dedba71ab22a0a04f2d8313ab2cedb5 |
| magnitude_all9_seed101 | 101 | 22 | 43 | 6450 | True | True | 2eaa38ed8db5e9fa7cf319f2309db69c8860d1fd380647ecc14531937083ed42 |
| magnitude_all9_seed202 | 202 | 16 | 37 | 5550 | True | True | b931b201c6f2c26bc9e887e767340aeba703f5c1f1daf62d4d641acd199fee31 |
| magnitude_all9_seed303 | 303 | 16 | 37 | 5550 | True | True | 52a66dcf66d9428d62ce70a160e8aeb30467f3ab56f8aaa556af138ecf0335b2 |

## q99 scenario results

Full-precision values remain in scenario_metrics.csv; rounded values below are display only. Sustained3 status is shown verbatim when already alarming.

### cleanStatic (reported separately)

| Model | q99 threshold | clean-test FPR |
|---|---:|---:|
| Complex all9 | 4.00798 | 0.205607 |
| Magnitude all9 | 4.14313 | 0.00934579 |
| B0 Exact | 1.70355 | 0.00934579 |

### Attack scenarios (exploratory/developmental)

| Scenario | Model | ROC-AUC | PR-AUC | stable-pre FPR | post detection | persistent detection | sustained3 delay/status |
|---|---|---:|---:|---:|---:|---:|---|
| DS1 | Complex all9 | 0.944004 | 0.96248 | 0.0909091 | 0.932133 | 1 | 26 |
| DS1 | Magnitude all9 | 0.962093 | 0.987321 | 0.010101 | 0.930748 | 1 | 26 |
| DS1 | B0 Exact | 0.973404 | 0.996647 | 0.020202 | 0.951524 | 1 | 6 |
| DS2 | Complex all9 | 0.967834 | 0.965014 | 0.10101 | 0.978962 | 1 | N/A: already alarming in stable-pre |
| DS2 | Magnitude all9 | 0.993795 | 0.999176 | 0 | 0.976157 | 1 | 10.5 |
| DS2 | B0 Exact | 0.99014 | 0.9987 | 0 | 0.97195 | 1 | 11 |
| DS3 | Complex all9 | 0.944402 | 0.94848 | 0.131313 | 0.957983 | 1 | N/A: already alarming in stable-pre |
| DS3 | Magnitude all9 | 0.974599 | 0.988759 | 0.010101 | 0.946779 | 1 | 20 |
| DS3 | B0 Exact | 0.960643 | 0.994985 | 0 | 0.942577 | 1 | 21.5 |
| DS7 | Complex all9 | 0.949225 | 0.967308 | 0.0504202 | 0.80625 | 0.891667 | N/A: already alarming in stable-pre |
| DS7 | Magnitude all9 | 0.926956 | 0.959669 | 0.00840336 | 0.640625 | 0.795833 | 37.5 |
| DS7 | B0 Exact | 0.889601 | 0.957754 | 0.0840336 | 0.63125 | 0.833333 | N/A: already alarming in stable-pre |
| DS8 | Complex all9 | 0.986722 | 0.997883 | 0.0168067 | 0.93454 | 0.974922 | 2.5 |
| DS8 | Magnitude all9 | 0.972221 | 0.987772 | 0.00840336 | 0.855153 | 0.932602 | 21 |
| DS8 | B0 Exact | 0.981976 | 0.997224 | 0 | 0.915042 | 0.960815 | 17.5 |

## EPL auxiliary summary

EPL is auxiliary only (E/L side targets; Prompt remains normalization-only) and cannot affect GO.

| Scenario | Model | ROC-AUC | stable-pre/clean FPR | post detection |
|---|---|---:|---:|---:|
| cleanStatic | Complex EPL | NA | 0.17757 | NA |
| cleanStatic | Magnitude EPL | NA | 0.0654206 | NA |
| DS1 | Complex EPL | 0.935442 | 0.0909091 | 0.939058 |
| DS1 | Magnitude EPL | 0.970893 | 0.020202 | 0.939058 |
| DS2 | Complex EPL | 0.979557 | 0.0909091 | 0.981767 |
| DS2 | Magnitude EPL | 0.990657 | 0.0606061 | 0.976157 |
| DS3 | Complex EPL | 0.944572 | 0.111111 | 0.955182 |
| DS3 | Magnitude EPL | 0.97989 | 0.030303 | 0.952381 |
| DS7 | Complex EPL | 0.951313 | 0.0756303 | 0.8875 |
| DS7 | Magnitude EPL | 0.943724 | 0.0420168 | 0.76875 |
| DS8 | Complex EPL | 0.966275 | 0.10084 | 0.967967 |
| DS8 | Magnitude EPL | 0.976458 | 0.0420168 | 0.912256 |

## Paired comparisons

All deltas are Complex minus comparator with paired 95% CI, same timestamps, 10 s blocks, and the persisted replicate count.

| Scenario | Comparator | Metric | Delta | 95% CI | Reps | Block s |
|---|---|---|---:|---|---:|---:|
| DS1 | Magnitude all9 | roc_auc | -0.0180895 | [-0.039246, 0.00604649] | 2000 | 10.0 |
| DS1 | Magnitude all9 | post_detection | 0.00138504 | [0, 0.00415512] | 2000 | 10.0 |
| DS1 | Magnitude all9 | stable_pre_fpr | 0.0808081 | [0.0606061, 0.10101] | 2000 | 10.0 |
| DS1 | B0 Exact | roc_auc | -0.0294007 | [-0.0600283, -0.00283187] | 2000 | 10.0 |
| DS1 | B0 Exact | post_detection | -0.0193906 | [-0.0443213, 0] | 2000 | 10.0 |
| DS1 | B0 Exact | stable_pre_fpr | 0.0707071 | [0.030303, 0.10101] | 2000 | 10.0 |
| DS2 | Magnitude all9 | roc_auc | -0.0259609 | [-0.056523, 0.000318156] | 2000 | 10.0 |
| DS2 | Magnitude all9 | post_detection | 0.00280505 | [0, 0.00849858] | 2000 | 10.0 |
| DS2 | Magnitude all9 | stable_pre_fpr | 0.10101 | [0.040404, 0.161616] | 2000 | 10.0 |
| DS2 | B0 Exact | roc_auc | -0.0223058 | [-0.05, -0.00097711] | 2000 | 10.0 |
| DS2 | B0 Exact | post_detection | 0.00701262 | [0, 0.0212465] | 2000 | 10.0 |
| DS2 | B0 Exact | stable_pre_fpr | 0.10101 | [0.040404, 0.161616] | 2000 | 10.0 |
| DS3 | Magnitude all9 | roc_auc | -0.0301969 | [-0.06032, 0.00371408] | 2000 | 10.0 |
| DS3 | Magnitude all9 | post_detection | 0.0112045 | [0, 0.0294118] | 2000 | 10.0 |
| DS3 | Magnitude all9 | stable_pre_fpr | 0.121212 | [0.0721649, 0.191919] | 2000 | 10.0 |
| DS3 | B0 Exact | roc_auc | -0.0162408 | [-0.0709894, 0.0393591] | 2000 | 10.0 |
| DS3 | B0 Exact | post_detection | 0.0154062 | [0, 0.039224] | 2000 | 10.0 |
| DS3 | B0 Exact | stable_pre_fpr | 0.131313 | [0.0721649, 0.222222] | 2000 | 10.0 |
| DS7 | Magnitude all9 | roc_auc | 0.0222689 | [-0.0270958, 0.0721788] | 2000 | 10.0 |
| DS7 | Magnitude all9 | post_detection | 0.165625 | [0.0436719, 0.325] | 2000 | 10.0 |
| DS7 | Magnitude all9 | stable_pre_fpr | 0.0420168 | [0.00840336, 0.092437] | 2000 | 10.0 |
| DS7 | B0 Exact | roc_auc | 0.0596245 | [-0.0194686, 0.180609] | 2000 | 10.0 |
| DS7 | B0 Exact | post_detection | 0.175 | [0.034375, 0.3375] | 2000 | 10.0 |
| DS7 | B0 Exact | stable_pre_fpr | -0.0336134 | [-0.168067, 0.0847458] | 2000 | 10.0 |
| DS8 | Magnitude all9 | roc_auc | 0.0145011 | [-0.00157072, 0.0370733] | 2000 | 10.0 |
| DS8 | Magnitude all9 | post_detection | 0.0793872 | [0.0167131, 0.157393] | 2000 | 10.0 |
| DS8 | Magnitude all9 | stable_pre_fpr | 0.00840336 | [0, 0.0252101] | 2000 | 10.0 |
| DS8 | B0 Exact | roc_auc | 0.00474591 | [-0.0146638, 0.0352141] | 2000 | 10.0 |
| DS8 | B0 Exact | post_detection | 0.0194986 | [-0.0432961, 0.0919633] | 2000 | 10.0 |
| DS8 | B0 Exact | stable_pre_fpr | 0.0168067 | [0, 0.0336134] | 2000 | 10.0 |

## Criterion evidence

- all_required_seeds_converged: PASS — 12/12 variant/objective/seed checkpoints converged
- auc_bootstrap_ci_lower_gt_zero_3_of_5: FAIL — 0/5 AUC CIs with lower bound > 0
- beats_b0_with_fpr_guard_3_of_5: FAIL — 1/5 scenarios meet B0 gain and FPR guard
- complex_auc_gt_magnitude_4_of_5: FAIL — 2/5 scenarios Complex > Magnitude
- ds7_ds8_no_collapse: PASS — DS7/DS8 collapse audit pass=True
- same_seed_direction_each_scenario: FAIL — seed-direction counts DS1/DS2/DS3/DS7/DS8 = 0/0/0/3/3
- stable_pre_fpr_all_below_0.05: FAIL — 1/5 scenarios below 0.05

## Claims and limitations

- Claimable: this leakage-safe developmental representation comparison, deterministic artifact recomputation, model fairness audit, and recorded NO-GO/GO outcome.
- Not claimable: confirmatory attack performance, independent-clean generalization, causality, deployment benefit, or a matched-threshold superiority claim over B0.
- All DS1/DS2/DS3/DS7/DS8 findings and rendered plots are exploratory/developmental. DS4 is explicit NA and excluded from GO.
- B0 calibration limitation: AMCF uses cleanStatic [340,410), while frozen protected B0 Exact uses [300,330); threshold-dependent comparisons are not matched-operating-point evidence.
- q99 alone is primary; q995 is diagnostic and cannot rescue a failed criterion.

## Numerical verifier QA correction

Identity, order, and source intervals remain exact. Conformal p/e/ensemble recomputation alone uses rtol=0.0 and atol=1e-14 for cross-NumPy arithmetic roundoff, far below every threshold margin. q99/q995, strict alarms, metrics, and GO decisions are still recomputed and must be exactly/semantically identical. This is verifier QA, not model, threshold, attack-selection, or hyperparameter retuning.
