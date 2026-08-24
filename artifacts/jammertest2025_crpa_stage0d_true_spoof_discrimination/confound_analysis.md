# Confound analysis

## Identification failure

The exact Spoof and Prn classes have disjoint received-power support. The nearest class ranges are separated by 21.714379367 dB and Track-A M0 achieves AUROC 1.0. The frozen 0.10, 0.25, 0.50, and 1.00 dB calipers produce zero pairs in every fold. Consequently, the conditional spatial question is not identifiable and Track B is not executed.

## Diagnostic-only observations

Track-A AUROC is M0 1.000000, M1 1.000000, M2 0.912618, M2R 0.222364, and M3 1.000000. The perfect power and single-channel baselines show that the recording classes differ strongly before a four-channel spatial relation is needed. Train-only cubic power residualization cannot manufacture common support; its out-of-support extrapolation is not a substitute for matched observations.

M2 destruction-retrained AUROC remains 0.795830 for mismatched tuples, 0.808419 for circular shifts, and 0.720446 for Fourier phase randomization. The corresponding actual-trained cross-applied scores fall below chance. These are useful engineering diagnostics but cannot rescue a failed primary overlap gate.

## Remaining provenance limits

All 124 Spoof snapshots occupy five consecutive sample-index blocks. Official recording IDs, transmitter position, receiver orientation, and array calibration are unavailable. Therefore the artifact makes no clean-versus-spoof, general detector, recording-independent, or WCL-readiness claim.
