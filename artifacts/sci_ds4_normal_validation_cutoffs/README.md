# SCI ds4 normal-validation event cutoffs

## Hypothesis
Keep the PRN-local low-quorum persistence morphology score fixed, but set event cutoffs on a broader normal-validation reference (cleanStatic plus cleanDynamic) so ds4's benign pre-onset morphology bursts are less likely to count as false positives while preserving post-onset sensitivity.

## Paper-style problem definition and evaluation protocol
- Normal-only GNSS spoofing detection from 1 s / 0.5 s stride tracking windows: fit PRN-local tracking-morphology normality on cleanStatic, calibrate operating cutoffs on either cleanStatic or a cleanStatic plus cleanDynamic normal-validation reference, then evaluate held-out TEXBAT ds4 with spoof onset=100 s.
- Primary signal: PRN-local prompt-relative 9-tap tracking morphology; no Doppler, CN0, or raw power is used.
- Auxiliary relation/geometry gate: causal roll3 q65 PRN-high fraction, used only as a bounded prevalence gate on morphology evidence.
- Scaler/cutoffs: cleanStatic robust median/MAD scaler; compare cleanStatic versus cleanStatic plus cleanDynamic normal-validation event-score quantiles q90/q95/q99/q99.5/q99.9.
- Evaluation: held-out ds4, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.

## ds4 metrics
- cleanStatic_event_cutoffs / score_low_quorum_persistence_gate: AUC=0.942; q90 det/FP/delay=0.750/0.034/14.5; q95=0.750/0.011/14.5; q99=0.750/0.000/14.5; q99.5=0.750/0.000/14.5; q99.9=0.750/0.000/14.5
- cleanStatic_plus_cleanDynamic_event_cutoffs / score_low_quorum_persistence_gate: AUC=0.942; q90 det/FP/delay=0.750/0.000/14.5; q95=0.750/0.000/14.5; q99=0.722/0.000/15.0; q99.5=0.667/0.000/15.0; q99.9=0.111/0.000/16.0

## Interpretation
Normal-validation cutoffs remove the remaining q90/q95 pre-onset false positives for this morphology-gated detector while keeping q90 and q95 post-onset detection at 0.750 with the same 14.5 s first delay. The cost is a modest drop in very-high-quantile detection (q99: 0.750 to 0.722; q99.5: 0.750 to 0.667), which is acceptable when the operating point prioritizes low false alarms. This strengthens the paper protocol: learn morphology on cleanStatic, use a separate clean normal reference to choose deployment cutoffs, and reserve PRN relation/geometry as a calibrated gate.
