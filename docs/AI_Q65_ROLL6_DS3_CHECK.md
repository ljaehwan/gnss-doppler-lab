# AI morphology-GRU q65 roll6 event calibration check (2026-07-21)

This check keeps the AI detector in the q70 morphology-quorum framing: shared PRN-local normal-only GRU RMSE over the currently tracked PRN set, event-level aggregation, quorum gating, and cleanStatic+cleanDynamic event-score quantile calibration. Thresholds are global clean-reference event-score quantiles, not PRN-ID thresholds.

Probe setting:
- `score_col`: `ai_rmse_q_tau50_gate`
- low-node quantile: cleanStatic PRN-node RMSE q0.70
- event aggregation quantile: current tracked PRN-set q0.65
- quorum roll window: 6 event bins (0.5 s bins)
- onset-aware metrics: pre `<90 s`, post `>=110 s`

Metrics:

| scenario | AUC | q99 pre FP | q99 post det | q99 first delay s | q95 pre FP | q95 post det | q95 first delay s |
|---|---|---|---|---|---|---|---|
| ds1 | 0.982845 | 0.000000 | 0.954481 | 26.000000 | 0.000000 | 0.955903 | 25.500000 |
| ds2 | 1.000000 | 0.000000 | 0.997118 | 11.000000 | 0.000000 | 0.998559 | 10.500000 |
| ds3 | 0.994459 | 0.000000 | 0.969828 | 20.500000 | 0.000000 | 0.971264 | 20.000000 |
| ds4 | 0.928975 | 0.000000 | 0.666667 | 15.500000 | 0.000000 | 0.750000 | 14.500000 |

Comparison to statistical q70 ds3 reference:
- statistical `score_mean_tau50_gate` q99: pre FP 0.000000, post detection 0.971264, first delay 20.0 s, AUC 0.989886.
- AI q65/roll6 `ai_rmse_q_tau50_gate` q99 on ds3: pre FP 0.000000, post detection 0.969828, first delay 20.5 s, AUC 0.994459.
- Gap at q99 on ds3: detection -0.001437 absolute (~-0.14 percentage points) and first delay +0.5 s, with equal 0% pre-spoof FP.

Interpretation:
The roll6 context smoothing slightly improves the AI q99 ds3 detection rate versus the prior q65/roll4 result while preserving 0% pre-spoof false positives. It is still just below the statistical detector at q99, and ds4 remains weaker, indicating a remaining model/feature generalization issue rather than a receiver-tracking issue. ds6 is not included because post-onset scoreable windows are unavailable after tracking collapse.

Artifacts generated but not tracked by default:
- `artifacts/ai_morph_gru_cleanStatic_q70_frame/q65_roll6_event_calibration/summary_all.json`
- `artifacts/ai_morph_gru_cleanStatic_q70_frame/q65_roll6_event_calibration/q65_roll6_ds1_ds2_ds3_ds4_ai_q_metrics.csv`
