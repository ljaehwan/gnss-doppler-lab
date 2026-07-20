# SCI ds4 causal PRN-quorum persistence switch

## Hypothesis
A short causal persistence check on the PRN-quorum relation gate should catch ds4's sustained constellation-wide morphology shift earlier than an instantaneous quorum offset. The primary evidence remains PRN-local prompt-relative tracking morphology; the relation/geometry term only switches on when the q70 PRN-high fraction persists for three event bins.

## Paper-style problem definition and evaluation protocol
- Normal-only GNSS spoofing detection from 1 s / 0.5 s stride tracking windows: fit PRN-local tracking-morphology normality on cleanStatic, calibrate scaler and all event cutoffs on cleanStatic, then evaluate held-out TEXBAT ds4 with spoof onset=100 s.
- Primary signal: PRN-local prompt-relative 9-tap tracking morphology; no Doppler, CN0, or raw power is used.
- Auxiliary relation/geometry gate: causal roll3 q70 PRN-high fraction, used only as a bounded gate on morphology evidence.
- Scaler/cutoffs: cleanStatic robust median/MAD and cleanStatic event-score quantiles q90/q95/q99/q99.5/q99.9.
- Evaluation: held-out ds4, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.

## ds4 metrics
- score_mean: AUC=0.907; q90 det/FP/delay=0.278/0.028/15.0; q95=0.056/0.022/15.5; q99=0.000/0.000/None; q99.5=0.000/0.000/None; q99.9=0.000/0.000/None
- score_quorum_offset_gate: AUC=0.933; q90 det/FP/delay=0.778/0.039/14.0; q95=0.778/0.011/14.0; q99=0.778/0.000/14.0; q99.5=0.750/0.000/14.5; q99.9=0.750/0.000/14.5
- score_persistence_switch: AUC=0.949; q90 det/FP/delay=0.972/0.497/10.5; q95=0.833/0.112/12.5; q99=0.750/0.000/14.5; q99.5=0.750/0.000/14.5; q99.9=0.750/0.000/14.5

## Interpretation
The causal persistence switch improves AUC and q90/q95 post-onset coverage relative to the instantaneous quorum-offset baseline, which is consistent with ds4 behaving as a sustained constellation-wide morphology change. The trade-off is higher pre-onset false positives at q90/q95/q99, so this variant is best viewed as an early-warning operating point rather than the strictest high-quantile detector.
