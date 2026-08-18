# CHORD Stage-0A artifact

Clean-only physical-identifiability artifact. The preregistration commit fixes every source, split, statistic, control, and gate before scoring.

## Result

Verdict: `NO_GO_CHORD_CLEAN_IDENTIFIABILITY`.

The receiver-faithful raw-IQ alignment gate passed for all 15,521 scheduled
epochs and all 470 prior MOSAIC anchors. OAKBAT Full ROC-AUC was 0.503957
(10-second block-bootstrap 95% CI 0.493234–0.512390); TEXBAT Full ROC-AUC was
0.502118 (95% CI 0.494224–0.509226). The 10-second/30-second lag AUCs were
0.513939/0.503310 for OAKBAT and 0.513033/0.487317 for TEXBAT. The tested
nuisance-normal projective residual direction therefore did not identify PRN
origin on either stationary clean recording.

The negative population in `pair_metrics.csv` is jointly matched to its positive
counterpart on calibration-scaled C/N0 difference and log residual-norm
difference at the same target epoch. Thus the reported Full AUC is the frozen
C/N0- and residual-norm-matched AUC; no unmatched alternative pair population is
used. OAKBAT's strongest scalar baseline was C/N0-only at 0.499071 and Full
improved by only 0.004886. TEXBAT's C/N0-only baseline was 0.677341 and Full was
lower by 0.175223. Full did not meet the preregistered +0.10 scalar delta gate.

Holdout fingerprint availability was 94.56% for OAKBAT and 95.99% for TEXBAT.
No PRN exceeded 10.60% of pair evidence, so unavailability and row-count
dominance do not explain the no-go. Exact gain, global-phase, NAV-sign, and
common amplitude controls changed projective similarity by at most 2.23e-16.
The deterministic label-shuffle AUCs were 0.498777 and 0.500558.

No attack recording or label was accessed; no threshold, detector, injection,
or AI model was created. Under the preregistered rule, CHORD stops here and no
Stage-0B attack design is authorized.

Broader multi-PRN channel correlation and RF-fingerprinting ideas are prior art;
the only tested distinction was a stationary single-antenna, complex nine-tap,
nuisance-tangent-normal projective residual direction. No first-detector claim
is made.
