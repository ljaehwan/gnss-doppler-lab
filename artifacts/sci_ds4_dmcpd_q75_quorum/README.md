# SCI ds4 DMCPD q75 PRN-quorum experiment

## Hypothesis
DMCPD peak-shape morphology concentrates ds4 spoofing evidence in a broad upper quartile of PRNs; using the per-epoch PRN q75 local morphology score as the primary event signal, with a bounded q80 PRN-quorum relation gate, should improve strict cleanStatic-calibrated ds4 detection without relying on raw power, Doppler, or CN0.

## Paper-style problem definition and protocol
- Normal-only GNSS spoofing detection from 1 s / 0.5 s stride tracking windows: learn PRN-local tracking-morphology normality on cleanStatic, calibrate event cutoffs on cleanStatic, and evaluate held-out TEXBAT ds4 with spoof onset=100 s.
- Primary signal: PRN-local DMCPD/tap-derived peak-shape tracking morphology; event evidence is the same-epoch PRN q75 local score.
- Auxiliary relation/geometry gate: fraction of PRNs over the cleanStatic local q80 node cutoff, used only as a bounded gate on morphology evidence.
- Scaler/cutoffs: cleanStatic robust median/MAD and cleanStatic event-score quantiles q90/q95/q99/q99.5/q99.9.
- Evaluation: held-out ds4 only, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.

## ds4 metrics
- score_mean: AUC=0.888; q90 det/FP/delay=0.000/0.011/None; q95=0.000/0.006/None; q99=0.000/0.000/None; q99.5=0.000/0.000/None; q99.9=0.000/0.000/None
- score_q75: AUC=0.914; q90 det/FP/delay=0.806/0.095/10.5; q95=0.750/0.039/14.5; q99=0.750/0.000/14.5; q99.5=0.750/0.000/14.5; q99.9=0.722/0.000/14.5
- score_q75_quorum: AUC=0.901; q90 det/FP/delay=0.806/0.140/10.5; q95=0.778/0.022/14.0; q99=0.778/0.000/14.0; q99.5=0.778/0.000/14.0; q99.9=0.778/0.000/14.0

## Interpretation
The q75 PRN-local morphology aggregation treats spoofing as a broad constellation effect without requiring every PRN to be extreme. The q80 quorum gate improves the strict q99/q99.5/q99.9 operating points to 77.8% post-onset detection with zero buffered pre-onset false positives, while q95 still has low pre-FP. This is a cleaner strict-threshold trade-off than relying on the cross-PRN mean alone.
