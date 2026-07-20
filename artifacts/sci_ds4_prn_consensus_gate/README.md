# SCI ds4 PRN-consensus-gated morphology experiment

## Hypothesis
If spoofing affects the receiver constellation coherently, a cleanStatic-calibrated cross-PRN consensus gate should amplify sustained PRN-local prompt-relative tap morphology shifts while not using PRN relation/geometry as the primary anomaly signal.

## Paper-style problem definition and protocol
- Normal-only GNSS spoofing detection from 1 s / 0.5 s stride tracking windows: learn PRN-local correlation-morphology normality on cleanStatic, calibrate event cutoffs on cleanStatic, and evaluate held-out TEXBAT ds4 with spoof onset=100 s.
- Primary signal: PRN-local prompt-relative 9-tap tracking morphology; no Doppler/CN0/raw power in the score.
- Auxiliary relation/geometry proxy: same-epoch cross-PRN consensus fraction, used only as a bounded gate on the local morphology event score.
- Scaler/cutoffs: cleanStatic robust median/MAD and cleanStatic event-score quantiles q90/q95/q99/q99.5/q99.9.
- Evaluation: held-out ds4 only, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.

## ds4 metrics
- score_mean: AUC=0.907; q90 det/FP/delay=0.278/0.028/15.0; q95=0.056/0.022/15.5; q99=0.000/0.000/None; q99.5=0.000/0.000/None; q99.9=0.000/0.000/None
- score_consensus_gate: AUC=0.924; q90 det/FP/delay=0.750/0.011/14.5; q95=0.750/0.011/14.5; q99=0.500/0.000/14.5; q99.5=0.361/0.000/14.5; q99.9=0.306/0.000/14.5

## Interpretation
The consensus gate is a small, paper-defensible addition: it leaves the node anomaly definition PRN-local and uses the PRN relation only to ask whether many PRNs become morphologically abnormal at the same epoch. On ds4 this turns the high AUC morphology separation into usable high-quantile detections, especially at q95/q99, while preserving zero pre-FP at q99+ in this run.
