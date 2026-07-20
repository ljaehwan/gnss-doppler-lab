# SCI ds4 soft-quorum-gated PRN morphology experiment

## Hypothesis
For ds4, spoofing may appear as a weak-but-broad PRN-local morphology shift before every PRN becomes an extreme local outlier. A softer cleanStatic q70 PRN quorum, used only as a bounded relation/geometry gate on the same morphology mean, should improve high-quantile detection relative to the prior q90 consensus gate.

## Paper-style problem definition and protocol
- Normal-only GNSS spoofing detection from 1 s / 0.5 s stride tracking windows: fit PRN-local tracking-morphology normality on cleanStatic, calibrate cutoffs on cleanStatic, and evaluate held-out TEXBAT ds4 with spoof onset=100 s.
- Primary signal: PRN-local prompt-relative 9-tap tracking morphology; no Doppler/CN0/raw power in the score.
- Auxiliary relation/geometry gate: same-epoch soft PRN quorum fraction, used only as a bounded multiplier on the morphology mean.
- Scaler/cutoffs: cleanStatic robust median/MAD and cleanStatic event-score quantiles q90/q95/q99/q99.5/q99.9.
- Evaluation: held-out ds4 only, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.

## ds4 metrics
- score_mean: AUC=0.907; q90 det/FP/delay=0.278/0.028/15.0; q95=0.056/0.022/15.5; q99=0.000/0.000/None; q99.5=0.000/0.000/None; q99.9=0.000/0.000/None
- score_prior_consensus_gate: AUC=0.924; q90 det/FP/delay=0.750/0.011/14.5; q95=0.750/0.011/14.5; q99=0.500/0.000/14.5; q99.5=0.361/0.000/14.5; q99.9=0.306/0.000/14.5
- score_soft_quorum_gate: AUC=0.932; q90 det/FP/delay=0.778/0.034/14.0; q95=0.750/0.011/14.5; q99=0.750/0.000/14.5; q99.5=0.750/0.000/14.5; q99.9=0.722/0.000/14.5

## Interpretation
The soft quorum gate improves high-quantile ds4 detection while keeping q99+ pre-FP at zero in this evaluation. The cost is a higher q90 pre-FP than the prior hard q90 consensus gate, so this is best interpreted as a sensitivity-oriented detector for strict q99/q99.5/q99.9 operating points.
