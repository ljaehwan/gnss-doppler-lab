# SCI ds4 PRN-quorum offset gate experiment

## Hypothesis
A bounded additive PRN-quorum offset should help ds4 when the spoofing trace has a broad, coherent but not uniformly extreme morphology shift: the primary score remains PRN-local morphology, while the relation/geometry term only adds calibrated evidence for constellation-wide prevalence.

## Paper-style problem definition and protocol
- Normal-only GNSS spoofing detection from 1 s / 0.5 s stride tracking windows: fit PRN-local tracking-morphology normality on cleanStatic, calibrate cutoffs on cleanStatic, and evaluate held-out TEXBAT ds4 with spoof onset=100 s.
- Primary signal: PRN-local prompt-relative 9-tap tracking morphology; no Doppler/CN0/raw power in the score.
- Auxiliary relation/geometry gate: same-epoch q70 PRN quorum fraction, used only as a bounded multiplier plus additive offset on morphology evidence.
- Scaler/cutoffs: cleanStatic robust median/MAD and cleanStatic event-score quantiles q90/q95/q99/q99.5/q99.9.
- Evaluation: held-out ds4 only, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.

## ds4 metrics
- score_mean: AUC=0.907; q90 det/FP/delay=0.278/0.028/15.0; q95=0.056/0.022/15.5; q99=0.000/0.000/None; q99.5=0.000/0.000/None; q99.9=0.000/0.000/None
- score_soft_quorum_gate: AUC=0.932; q90 det/FP/delay=0.778/0.034/14.0; q95=0.750/0.011/14.5; q99=0.750/0.000/14.5; q99.5=0.750/0.000/14.5; q99.9=0.722/0.000/14.5
- score_quorum_offset_gate: AUC=0.933; q90 det/FP/delay=0.778/0.039/14.0; q95=0.778/0.011/14.0; q99=0.778/0.000/14.0; q99.5=0.750/0.000/14.5; q99.9=0.750/0.000/14.5

## Interpretation
The additive quorum offset is a small extension of the soft-quorum detector. It slightly improves q99/q99.9 ds4 detection and AUC versus the previous soft gate while preserving zero q99+ pre-FP in this buffered ds4 evaluation. The trade-off is a small q90 pre-FP increase, so the result is most useful for strict high-quantile operating points.
