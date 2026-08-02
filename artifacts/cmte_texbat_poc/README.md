# CMTE TEXBAT static PoC

## Decision

**NO-GO.** This artifact was generated from commit `bf81eef66b5103d4ba19660634762efcef697dff`; exact source hashes are in `code_hashes.json`. The detector was frozen before DS1–DS4 scoring. No attack labels were used to select the model, covariance, kappa mixture, drift, or thresholds.

The main failure is false-alarm control. At the preregistered `target1` operating point, Full CMTE occupied the alarm state for **98.8764%** of the CMTE-calibration-independent clean test and for **100.0%, 43.33%, 66.67%, and 90.0%** of the stable-pre intervals of DS1–DS4. Any Full stable-pre occupancy `>=20%` is a catastrophic failure; this occurred in all four scenarios.

## Frozen B0 and input provenance

- Frozen shared PRN-local B0 GRU checkpoint SHA-256: `f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b`.
- Tap order: `E4,E3,E2,E,P,L,L2,L3,L4`.
- Raw taps are **magnitudes** reconstructed as `hypot(I,Q)` for cleanStatic/DS1–DS3. “Signed innovation” means the sign of `x - x_hat`, not signed complex raw taps.
- PRN identity is not a feature. All PRNs share predictor weights.
- Predictor `history_id` is separate from immutable physical `recording_id`. B0 history resets at split, segment, channel, and cadence-gap boundaries; multi-PRN evidence is then recombined by `(recording_id, availability epoch)`.
- Provenance grades: cleanStatic/DS1–DS3 are reconstructed-equivalence **A-** inputs; DS4 is a verified node artifact, grade **B**; historical B0 comparison is grade **C / historical_noncomparable**.
- The frozen B0 was historically trained across cleanStatic with PRN holdout. Therefore the clean test is independent of **CMTE fitting/calibration**, but not end-to-end independent of B0 training.

Exact source paths, SHA-256 values, cadence-chunk audits, and timing policy are in `provenance.json` and `provenance/`.

## Normal-only split and calibration

- Train: cleanStatic `0–240 s`
- Guard: `240–250 s`
- Validation/conformal calibration/threshold selection: `250–330 s`
- Guard: `330–340 s`
- CMTE-calibration-independent clean test: `>=340 s`
- Signed 9D residual rows: train `4,772`, validation `1,469`, test `2,714`
- Predictor and sequential histories reset at each split. No attack recording, including its pre-onset portion, enters fitting, model selection, conformal calibration, drift selection, or threshold calibration.

Full uses full shrinkage Mahalanobis nonconformity:

`q = (r-mu)^T Sigma_reg^{-1} (r-mu)`

Split-conformal p-values use inclusive ties and finite-sample correction:

`p = (1 + #{q_cal >= q}) / (|Q_cal| + 1)`

PRN evidence uses the fixed mixture `kappa in {0.25,0.50,0.75}`:

`e(p) = mean_k [kappa * p^(kappa-1)]`

Epoch evidence is the permutation-invariant mean over tracked PRNs. Two sequential detectors were implemented: a fully log-domain fixed-prior restart betting capital and resettable e-CUSUM

`G_t = max(0, G_(t-1) + log(E_epoch + eps) - drift)`.

Validation-only selection chose S2 with drift `0.0`. The Full q99, q99.5, and target1 thresholds all equal `3.606925652969479` because only five 20-second validation block maxima exist. This is severe finite-sample uncertainty, not a formal 1% guarantee. CMTE is described only as a **sequential conformal evidence detector with empirically calibrated false-alarm control**; no unconditional anytime-valid claim is made.

## TEXBAT policy

All scores are evaluated at actual availability time. The user-requested policy is nominal onset `100 s`, stable pre-attack `[30,90) s`, transition exclusion `[90,110) s`, and established attack `>=110 s`. DS4 has short established-attack coverage.

## Target1 results

Full CMTE:

- **DS1:** ROC-AUC `1.0000`, PR-AUC `1.0000`, stable-pre occupancy `100.00%`, rising-edge false-alarm events `1` (`1.0/min`), delay `10.15 s`, persistent detection `100.00%`.
- **DS2:** ROC-AUC `0.9950`, PR-AUC `0.9992`, stable-pre occupancy `43.33%`, false-alarm events `1` (`1.0/min`), delay `10.62 s`, persistent detection `99.86%`.
- **DS3:** ROC-AUC `0.9534`, PR-AUC `0.9922`, stable-pre occupancy `66.67%`, false-alarm events `1` (`1.0/min`), delay `22.12 s`, persistent detection `96.55%`.
- **DS4:** ROC-AUC `0.6732`, PR-AUC `0.6892`, stable-pre occupancy `90.00%`, false-alarm events `1` (`1.0/min`), delay `14.63 s`, persistent detection `75.68%`.

Clean and DS1 comparison:

- **Full:** clean alarm-state occupancy `98.8764%`; sequence-any `1/1`; first crossing at epoch `4`. DS1 stable-pre occupancy `100%`.
- **A0, scalar RMSE q99:** clean occupancy `13.1086%`; DS1 stable-pre `25.83%`; delay `26.15 s`; persistence `92.47%`.
- **A1, threshold-count + binomial-tail:** clean occupancy `0.7491%`; DS1 stable-pre `0%`; delay `19.15 s`; persistence `95.60%`.

A1 is the only one of these three that meets the clean target and controls DS1 pre-onset false alarms. Full can appear faster or more persistent because it is already in alarm before the attack; that is not a valid detection advantage. Full also underperforms A1 on DS3 and DS4 under matched-power conditions.

Complete q99/q99.5/target1 results for A0–A4 and Full are in `scenario_metrics.csv`, `independent_clean_operating_points.csv`, and `ablation_metrics.csv`.

## False-alarm semantics

- `epoch_fpr` and `stable_pre_fpr`: fraction of epochs for which state/score exceeds threshold (alarm-state occupancy).
- `alarm_epoch_occupancy_per_min`: above-threshold epochs per observed minute.
- `false_alarm_events`: rising-edge crossings from below to above threshold, reset per recording.
- `false_alarms_per_min`: rising-edge event count per observed minute.
- `sequence_any_alarm_fraction`, first crossing, censored run length/ARL, and reset-aware 20-second block-any are reported separately.
- No forced detector reset occurs after an alarm.

## Success-criterion audit

- CMTE-calibration-independent clean FPR near 1%, upper bound 1.5%: **FAIL**.
- Every DS stable-pre FPR <=5%: **FAIL**.
- Catastrophic stable-pre FPR >=20% in any scenario: **FAIL — observed in all scenarios**.
- Formal comparator rule records at least three nominal speed/persistence improvements: mechanically true, but invalidated by catastrophic false alarms.
- Matched-power DS3/DS4 useful improvement at controlled FPR: **FAIL in scientific interpretation**.
- PRN permutation invariance and variable-cardinality operation: implemented and tested; performance independence from PRN count is **not established**, and N-stratum heterogeneity was observed.
- Attack-based tuning: **none**.

## What may and may not be claimed

Potential SCI/WCL contribution: a reproducible normal-only framework preserving signed 9D innovations, finite-sample conformal ranks, continuous PRN e-values, permutation-invariant aggregation, sequential evidence, strict fit/calibration boundaries, and explicit history/event identity separation.

Not supportable from this PoC: superiority over B0/binomial-tail, target-1% false-alarm control on unseen normal data, robust matched-power detection, PRN-count-independent performance, exact historical B0 equivalence for reconstructed inputs, or unconditional anytime-valid guarantees.

## Artifact map

- Metrics: `scenario_metrics.csv`, `ablation_metrics.csv`, `independent_clean_operating_points.csv`, `deterministic_comparator_audit.csv`, `full_by_N_diagnostic.csv`
- Evidence: `per_epoch/`, `per_prn/`, `per_prn_evidence_summary.csv`
- Calibration: `training_summary.json`, `calibration_summary.json`, `thresholds.json`
- Provenance/checksums: `provenance.json`, `provenance/`, `checksums.json`
- Diagnostics and figures: `diagnostics/`, `plots/`
- Executed commands/tests: `test_summary.txt`
