# AMCF-R1 campaign summary

developmental: DS1-3; post-exposure exploratory: DS7/8; DS4 NA

## Decisions
- **Detector operating point:** NO-GO
- **Complex:** NO-GO
- **Active:** NO-GO
- **WCL:** NO-GO

## Exact criteria
- C1: PASS
- C2: FAIL
- C3: FAIL
- C4: FAIL
- C5: FAIL
- C6: FAIL
- C7: PASS
- C8: FAIL
- C9: FAIL

## 1. Temporal aggregation and clean FPR
- AMCF-R1 primary held-out chronological clean FPR: **0.935%** (C1 PASS).
- Previous AMCF-Lite primary clean FPR: **4.918%**; temporal/full-row correction improved this operating point.
- Raw valid-row utilization is recorded per scenario in `window_qa.json`; no recording discarded approximately 98% of observations.

## 2. Complex versus magnitude
- Complex all9 ROC-AUC exceeded the same-architecture magnitude all9 model in **3/5** scenarios; C3 requires at least 4/5 and is **FAIL**.
- Per-scenario values are in `ablation_metrics.csv`; no single favorable seed is selected.

## 3. Phase destruction
- Intact complex all9 ROC-AUC exceeded phase-destroyed complex all9 in **3/5** scenarios (C4 FAIL).
- Intact temporal order exceeded temporal-shuffled ROC-AUC in **3/5** scenarios (C5 FAIL).
- Complex evidence decision: **NO-GO**; a complex-phase causal contribution is not claimed unless C3-C5 all pass.

## 4. Sample-dependent active paths
- IG modal path fraction range: **0.0605–0.3836**; unique ordered path range: **6–76**.
- C7 static-collapse criterion (<95% modal path) is **PASS**. Passing C7 means paths are sample-dependent; it does not prove useful selection.

## 5. Fixed/random policy comparison
- C6 same-budget IG superiority across multiple model seeds is **FAIL**.
- Active query policy decision: **NO-GO**. Query budgets are offline replay only and are not measured SDR computation savings.

## 6. B0 comparison
- Matched-clean diagnostic found material primary improvement over B0 in **0/5** scenarios; C8 requires at least 3 and is **FAIL**.
- Primary q99/q99.5 alarms and matched-clean diagnostic alarms are stored in independent columns and never overwrite each other.

Scenario | Primary matched ROC/post/stable-pre | B0 matched ROC/post/stable-pre
--- | --- | ---
DS1 | 0.9768/96.266%/23.000% | 0.9736/94.744%/2.000%
DS2 | 0.7072/54.202%/26.000% | 0.9898/97.059%/0.000%
DS3 | 0.3981/35.056%/27.000% | 0.9592/94.134%/0.000%
DS7 | 0.1953/100.000%/100.000% | 0.8896/61.059%/5.833%
DS8 | 0.4571/100.000%/100.000% | 0.9815/90.833%/0.000%

## 7. Failure attribution
- **Front-end/domain:** C2 stable-pre control is **FAIL**. DS1 Prompt rejection was stable-pre 6.728%, takeover 37.278%, persistent 34.694%; this is phase-dependent missingness rather than a single global producer-scale explanation.
- **Calibration:** C1 is PASS, but q99/q99.5 are coarse normal-calibration order statistics; exact/binomial and 10-second block-bootstrap intervals are saved.
- **Model:** 4/6 models reached epoch 50 without early-stop convergence (complex_seed101, complex_seed303, magnitude_seed101, magnitude_seed303); all saved checkpoints were finite and best model/optimizer states were restored.
- **Selector:** C7 is PASS while C6 is FAIL; path diversity and detection utility are therefore separated.
- **Domain robustness:** C9 DS7/DS8 guard is **FAIL**.

## 8. WCL claims
- WCL primary-model decision: **NO-GO**.
- Claimable: causal full-row temporal-window implementation, alarm-column correction, and a controlled negative/positive ablation result exactly as recorded.
- Not claimable after any failed criterion: confirmatory attack performance, independent-clean generalization, active SDR savings, or a superior primary detector.
- DS1-DS3 are developmental and DS7/DS8 are post-exposure exploratory; all attack scenarios were already exposed.

All thresholds use clean calibration only with strict `score > threshold`. Clean segments are held-out chronological segments, never independent; q99.5 can be a second-maximum or maximum order statistic at small calibration N.
