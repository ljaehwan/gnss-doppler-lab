# B0-CS Stage-0 Static result bundle

## 1. Historical-B0 reproduction

The frozen checkpoint and canonical binomial contract were hash-verified and scored as H0. It is a historical reference only.

## 2. Paper-B0 difference

Paper-B0 retains the exact shared nine-tap GRU architecture but uses chronological cleanStatic train/validation roles, train-only scaling, and excludes calibration/holdout from predictor fitting. Historical-B0 used a PRN holdout and cleanStatic+cleanDynamic gate calibration.

## 3. cleanStatic split and leakage audit

The machine-readable 50/15/20/15 split, 6 s guards, target/sample/byte overlap checks, role-local causal resets, and content hashes are in `split_and_overlap_audit.json`.

## 4. B0-CS formulas and assumptions

Scalar B0 RMSE is conformal-ranked, mapped by `0.5*p^-0.5`, averaged over at least four PRNs, block-max calibrated, and accumulated by `C_b=max(1,C_(b-1))*e_b` with alarm 100.

## 5. PRN dependence handling

Arithmetic mean e-value aggregation is permutation invariant and does not require PRN independence when component evidence is valid. PRN identity is never a feature or stratum.

## 6. Temporal dependence handling

The clean-train IAT rule fixed the non-overlapping block length before attacks. Results are `EMPIRICALLY_BLOCK_CALIBRATED`; arbitrary-dependence distribution-free and anytime-valid claims are forbidden.

## 7. DS3/DS4/DS7-8 performance

DS3 was available; Full normalized pAUC was 1.0 with zero pre-onset alarms, but its first alarm was 62.1 s after signal onset and much later than Paper-B0/simple consecutive. DS4 is `UNAVAILABLE` because no compatible nine-tap/C/N0/lineage export was found. DS7/DS8 remain one family and are `UNAVAILABLE` because both located exports omit the frozen C/N0 nuisance input. No configuration was adapted after attack access.

## 8. Comparisons

H0, A0, A1, A2, A3, A4, Full, Linear-AR, and SimpleConsecutive use frozen thresholds and common source processing. See `ablation_metrics.csv` and `bootstrap_intervals.csv`.

## 9. External static FPR

See `external_static_fpr.csv`. Missing source-distinct compatible normal data is explicitly `UNAVAILABLE` and blocks deployment-level FPR claims.

## 10. Physical controls

See `control_metrics.json`. Feature/residual-level controls are identified as such; raw-IQ AWGN/retracking is `UNAVAILABLE` without a sealed receiver rerun receipt.

## 11. Failed or limited scenarios

Every unavailable input, insufficient support, missing duration, or invalid lineage is carried as `UNAVAILABLE`/`LIMITED` in the structured files; no result was imputed.

## 12. Validity limits

Finite-sample ranks are tie-conservative, but nuisance estimation, temporal dependence, stationarity, block exchangeability, and mixing remain empirical assumptions. TEXBAT is developmental.

## 13. Final verdict

`PIVOT_TO_PROVENANCE_EVALUATION_PAPER`

Verdict audit: `{"clean_holdout_fpr_requirement": true, "controls_no_persistent_alarm": false, "core_full_scenarios_available": ["DS3"], "external_source_distinct_static_normal_available": false, "go_blockers": ["no source-distinct compatible static normal sequence", "persistent alarms in one or more available controls", "DS4 and DS7/DS8-family Full results unavailable under the frozen input contract", "Full did not beat the simple consecutive threshold in two core families", "GRU-versus-Linear-AR significance in two core families not established"], "go_possible": false}`

## 14. WCL-claimable contribution

Claimable output is limited to a preregistered, leakage-audited evaluation framework and its honestly bounded empirical result; deployment validity is not established.

## 15. Forbidden claims

Do not claim arbitrary-dependence distribution-free validity, anytime validity, independent DS7/DS8 confirmation, deployment-level FPR, or attack-blind source lineage beyond the recorded hashes.

## 16. Exactly one recommended next action

Acquire one sealed, source-distinct 20–30 minute static receiver capture with the identical complex-nine-tap, sample-counter, and C/N0 export contract, then evaluate once without recalibration.

## Structured availability and verification addendum

`data_inventory.json`, `scenario_metrics.csv`, `ablation_metrics.csv`, `bootstrap_intervals.csv`, and `external_static_fpr.csv` carry explicit DS4 and DS7/DS8-family `UNAVAILABLE` records. The only additional compatible normal candidate was a receiver replay of the same TEXBAT cleanStatic raw IQ and was excluded as non-independent. `validity_limitations.json` also marks the A3 no-nuisance ablation `LIMITED` because tied calibration multiplicities are collapsed in that frozen reporting path; Full is unaffected. The artifact manifest is generated only after this addendum and all verification inputs are final.

Runner run IDs: 20260815T162404Z-b0-cs-clean-freeze, 20260815T162742Z-b0-cs-ds123-evaluation, 20260815T162835Z-b0-cs-ds123-evaluation-safe-load, 20260815T163122Z-b0-cs-physical-controls-bootstrap-finalize, 20260815T163542Z-b0-cs-availability-provenance-report. The failed DS1-3 attempt is retained; it produced no metrics and was retried only with a Torch safe-global loader allowance.
