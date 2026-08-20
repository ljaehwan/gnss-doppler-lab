# CRID-GNSS Stage-0

CRID asks whether the outputs obtained by replaying one raw-IQ stream through
four tracking configurations remain explainable by one delay/carrier latent
state after clean-only causal dynamics compensation. It does not use received
power, C/N0, lock, PRN identity, or an existing detector score as its test
statistic.

The native receiver is GNSS-SDR 0.0.19 with the authenticated TRACE schema-v2
instrumentation. C0 uses 0.125-chip tap spacing, 1.5-Hz DLL, 20-Hz PLL, order
3, and 1-ms coherent integration. C1 changes spacing to 0.10 chip. C2 uses a
0.5-Hz DLL and 5-Hz PLL. C3 uses a 5-Hz DLL and 25-Hz PLL. Acquisition,
sampling, source range, channel count, navigation decoder, receiver binary,
and timestamp definition remain fixed.

The signed response contains native DLL, PLL and FLL errors, increments of the
code/carrier loop-filter outputs, Prompt phase increment, and complex signed
E-P/L-P coordinates. A shared PRN-agnostic causal FIR predicts the response.
Clean calibration residuals determine shrinkage covariance and whitening.
H0 fits one latent state through configuration response matrices; H1 permits
regularized configuration-specific states. The statistic is twice the
Gaussian likelihood improvement expressed as an RSS reduction, less a BIC
degrees-of-freedom penalty, and is pooled by the median over at least four
common PRNs.

Alignment uses exact absolute raw-sample coordinates. Nonnegative receiver
group delays are estimated on cleanStatic only and applied as past-only native
cadence shifts. Gaps reset the causal history. Reacquisition and invalid lock
rows are excluded. Any failure of deterministic replay, exact sample binding,
causal group-delay certification, four-configuration support, or four-common-
PRN support is an inconclusive receiver/alignment result rather than a
physical NO-GO.

## Novelty boundary

Prior work already compares tracking-loop robustness, decomposes multi-peak
correlator shapes using LASSO, models multi-correlator transient response, and
studies tracking-loop effects. CRID therefore does not claim novelty for using
multiple correlators or changing loop bandwidth. A possible contribution is
limited to their combination as same-IQ counterfactual receivers, a
dynamics-corrected shared-state H0 versus configuration-dependent H1 with an
explicit complexity penalty, and robust multi-PRN persistence. A single-PRN
two-source indication is not spoofing proof because multipath can create the
same physical structure.

Primary comparison sources:

- ION GNSS+ 2018, DOI 10.33012/2018.15912.
- Schmidt, Gatsis, and Akopian, arXiv:2004.13900.
- IEEE Xplore document 9771473 (WSCM).
- NAVIGATION 72(4), navi.724.

CRID configuration frozen before this CRID evaluation; TEXBAT/OAKBAT were
previously inspected by the broader project.
