# SCI ds4 low-quorum causal persistence gate

## Hypothesis
A slightly lower cleanStatic node cutoff (q65 instead of q70), admitted only after a causal 3-bin quorum persistence check, should capture ds4's weak-but-broad PRN-local morphology shift without making the relation/geometry term the primary detector.

## Paper-style problem definition and evaluation protocol
- Normal-only GNSS spoofing detection from 1 s / 0.5 s stride tracking windows: fit PRN-local tracking-morphology normality on cleanStatic, calibrate scaler and all event cutoffs on cleanStatic, then evaluate held-out TEXBAT ds4 with spoof onset=100 s.
- Primary signal: PRN-local prompt-relative 9-tap tracking morphology; no Doppler, CN0, or raw power is used.
- Auxiliary relation/geometry gate: causal roll3 q65 PRN-high fraction, used only as a bounded prevalence gate on morphology evidence.
- Scaler/cutoffs: cleanStatic robust median/MAD and cleanStatic event-score quantiles q90/q95/q99/q99.5/q99.9.
- Evaluation: held-out ds4, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.

## ds4 metrics
- score_mean: AUC=0.907; q90 det/FP/delay=0.278/0.028/15.0; q95=0.056/0.022/15.5; q99=0.000/0.000/None; q99.5=0.000/0.000/None; q99.9=0.000/0.000/None
- score_quorum_offset_gate: AUC=0.933; q90 det/FP/delay=0.778/0.039/14.0; q95=0.778/0.011/14.0; q99=0.778/0.000/14.0; q99.5=0.750/0.000/14.5; q99.9=0.750/0.000/14.5
- score_low_quorum_persistence_gate: AUC=0.942; q90 det/FP/delay=0.750/0.034/14.5; q95=0.750/0.011/14.5; q99=0.750/0.000/14.5; q99.5=0.750/0.000/14.5; q99.9=0.750/0.000/14.5

## Interpretation
The low-quorum persistence gate is a small sensitivity-oriented change: relative to the q70 instantaneous quorum-offset baseline, it raises buffered AUC while preserving the q95/q99/q99.5/q99.9 detection rates and zero q99+ pre-FP. q90 detection drops slightly, but q90 pre-FP is also slightly lower. This supports the paper claim that ds4 is better described as a broad PRN-local tracking-morphology deformation, with PRN prevalence useful as a calibrated auxiliary gate rather than a standalone geometry detector.
