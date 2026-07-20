# SCI ds4 q70 morphology quorum

## Hypothesis
Replace the event-level cross-PRN mean of PRN-local tracking-morphology scores with a modest upper-tail aggregator (cross-PRN q70), while keeping cleanStatic robust scaling, the q65 local-node threshold, the causal roll3 quorum>=0.50 gate, and cleanStatic+cleanDynamic normal-validation cutoffs fixed. The q70 aggregator should retain the local-morphology primary signal but emphasize the subset of PRNs that carry ds4's spoofing deformation, improving very-low-FP detection at q99.5/q99.9.

## Paper-style problem definition and evaluation protocol
- Normal-only GNSS spoofing detection from 1 s / 0.5 s stride PRN tracking-morphology windows. Fit per-feature robust median/MAD scaling on cleanStatic, set q90/q95/q99/q99.5/q99.9 event cutoffs on cleanStatic plus cleanDynamic normal validation, and evaluate held-out TEXBAT ds4 with spoof onset=100 s.
- Primary signal: PRN-local prompt-relative 9-tap tracking morphology; no Doppler, CN0, or raw power is used.
- Auxiliary relation/geometry gate: causal roll3 q65 PRN-high fraction, used only as a bounded prevalence gate on morphology evidence.
- Scaler/cutoffs: cleanStatic robust median/MAD scaler; cleanStatic+cleanDynamic normal-validation event-score quantiles q90/q95/q99/q99.5/q99.9.
- Evaluation: held-out ds4, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.

## ds4 metrics
- score_mean_tau50_gate: AUC=0.952; q90 det/FP/delay=0.750/0.000/14.5; q95=0.750/0.000/14.5; q99=0.722/0.000/15.0; q99.5=0.667/0.000/15.0; q99.9=0.111/0.000/16.0
- score_q70_tau50_gate: AUC=0.949; q90 det/FP/delay=0.778/0.056/14.0; q95=0.778/0.034/14.0; q99=0.778/0.011/14.0; q99.5=0.778/0.000/14.0; q99.9=0.778/0.000/14.0

## Interpretation
The q70 morphology aggregator trades some ranking AUC and low-quantile FP control for substantially stronger high-cutoff sensitivity: at q99.5 and q99.9 it keeps zero pre-onset false positives while detecting 77.8% of post-onset windows with a 14.0 s first delay. This supports using PRN-local morphology as the main signal and reserving PRN relation/geometry for a conservative persistence gate, but suggests the event aggregation statistic should be selected for the target operating false-positive regime.
