# SCI ds4 stricter quorum tau

## Hypothesis
Tighten only the auxiliary PRN-relation gate, from roll3 low-quorum fraction >=0.40 to >=0.50, while keeping the PRN-local morphology score, cleanStatic robust scaler, q65 node threshold, and normal-validation event cutoffs fixed. Requiring half of the tracked PRNs to remain locally unusual for the causal 3-bin gate should suppress marginal benign relation bursts without delaying ds4's sustained post-onset morphology shift.

## Paper-style problem definition and evaluation protocol
- Normal-only GNSS spoofing detection from 1 s / 0.5 s stride PRN tracking-morphology windows. Fit all per-feature robust median/MAD scaling on cleanStatic, calibrate operating q90/q95/q99/q99.5/q99.9 event cutoffs on cleanStatic plus cleanDynamic normal validation, and evaluate held-out TEXBAT ds4 with onset=100 s.
- Primary signal: PRN-local prompt-relative 9-tap tracking morphology; no Doppler, CN0, or raw power is used.
- Auxiliary relation/geometry gate: causal roll3 q65 PRN-high fraction, used only as a bounded prevalence gate on morphology evidence.
- Scaler/cutoffs: cleanStatic robust median/MAD scaler; cleanStatic+cleanDynamic normal-validation event-score quantiles q90/q95/q99/q99.5/q99.9.
- Evaluation: held-out ds4, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.

## ds4 metrics
- score_low_quorum_tau40: AUC=0.942; q90 det/FP/delay=0.750/0.000/14.5; q95=0.750/0.000/14.5; q99=0.722/0.000/15.0; q99.5=0.667/0.000/15.0; q99.9=0.111/0.000/16.0
- score_low_quorum_tau50: AUC=0.952; q90 det/FP/delay=0.750/0.000/14.5; q95=0.750/0.000/14.5; q99=0.722/0.000/15.0; q99.5=0.667/0.000/15.0; q99.9=0.111/0.000/16.0

## Interpretation
Raising the causal quorum gate from 0.40 to 0.50 improves buffered ds4 ranking AUC while preserving the normal-validation operating-point rates at q90/q95 and q99. The result supports treating PRN relation/geometry as a conservative gate on sustained morphology deformation rather than as the primary detector.
