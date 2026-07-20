# SCI ds4 temporal-persistence morphology experiment

## Hypothesis
A causal 5-bin temporal persistence filter on PRN-local prompt-relative tap morphology suppresses isolated normal/pre-attack spikes and exposes ds4's sustained post-onset morphology shift.

## Protocol
- Scaler: robust median/MAD fitted on cleanStatic only.
- Features: PRN-local `tap_*_rel_prompt_*` morphology only; no Doppler/CN0/raw power.
- Thresholds: q90/q95/q99/q99.5/q99.9 from cleanStatic after the identical causal score filter.
- Evaluation: ds4 only, spoof onset assumed 100 s; pre-FP uses t<90 s and post detection uses t>=110 s.

## ds4 metrics
- score_mean: AUC=0.907, q90 det/FP=0.278/0.028, q95 det/FP=0.056/0.022, q99 det/FP=0.000/0.000, q99.5 det/FP=0.000/0.000, q99.9 det/FP=0.000/0.000
- score_roll5_median: AUC=0.895, q90 det/FP=0.250/0.034, q95 det/FP=0.083/0.028, q99 det/FP=0.000/0.000, q99.5 det/FP=0.000/0.000, q99.9 det/FP=0.000/0.000
- score_roll5_mean: AUC=0.899, q90 det/FP=0.111/0.017, q95 det/FP=0.028/0.011, q99 det/FP=0.000/0.000, q99.5 det/FP=0.000/0.000, q99.9 det/FP=0.000/0.000

## Interpretation
The rolling filter is causal and therefore paper-defensible as temporal evidence accumulation, not attack-data tuning. It primarily tests whether ds4 is a sustained PRN-local morphology shift rather than an isolated per-window anomaly.
