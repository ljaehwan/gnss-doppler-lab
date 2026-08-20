# CORA-GNSS bounded novelty audit

Search date: 2026-08-20. This is a bounded literature audit, not a proof of
novelty. Searches covered IEEE/ION, GPS Solutions, Sensors, ACM WiSec, and
publisher indexes for GNSS spoofing combined with correlator distortion,
cross-satellite dependence, common source, RF fingerprinting, SVD/rank,
higher-order cumulants, and normal-only anomaly detection.

## Adjacent prior art found

- Correlator-shape and delay–Doppler distortion monitoring is established.
  Turner et al. decompose the full correlation function and apply a likelihood
  ratio test: [IEEE/ION PLANS 2020](https://www.ion.org/publications/abstract.cfm?articleID=17485).
- Cross-satellite common-source reasoning is established in the spatial domain.
  Broumandan et al. explicitly exploit the shared spatial signature of multiple
  counterfeit satellite signals and amplitude/Doppler correlation:
  [IEEE/ION PLANS 2012](https://www.ion.org/publications/abstract.cfm?articleID=10116).
- Physical-layer GPS transmitter fingerprinting from complex correlator output
  is established. Spotr builds per-satellite fingerprints and a normal model:
  [ACM WiSec 2020](https://doi.org/10.1145/3395351.3399353).
- GNSS RF fingerprinting using IQ imbalance, ADC non-linearity, phase noise,
  PA non-linearity, Gaussianity/kurtosis, energy, and learned models is an active
  line of work: [Sensors 2024 review and experiment](https://www.mdpi.com/1424-8220/24/23/7698).
- Higher-order cumulants in GNSS processing are not new; cumulant-based GNSS
  acquisition has been published: [Sensors 2024](https://www.mdpi.com/1424-8220/24/19/6234).
- Aggregating correlation residual evidence and correlator metric fusion are
  also established ideas: [Remote Sensing 2024](https://www.mdpi.com/2072-4292/16/15/2868)
  and [Computers & Electrical Engineering 2021](https://doi.org/10.1016/j.compeleceng.2021.107159).

## Exact-combination assessment

The bounded search did not identify an exact isomorph of all five elements:

`code-aided post-nominal raw residual`
`+ receiver-common nuisance quotient`
`+ fourth-order cross-PRN conditional dependence`
`+ variable-PRN shared-emitter likelihood`
`+ normal-only calibration`

That absence is only a search result, not a first-in-literature claim. The
closest risks are Spotr/RF fingerprinting, spatial cross-satellite correlation,
correlation-residual likelihood methods, and existing higher-order GNSS
statistics. A broader patent search, full-text IEEE search, and expert review
would be required before making a novelty claim.

## Claim boundary after this experiment

No positive research-contribution claim is supported by Stage-0. The frozen
CORA implementation is a tested analytic combination, but it failed its real
attack, false-positive, relation-destruction, LOPO, shortcut, and B0 evidence
gates. It must not be described as a demonstrated detector, as first, or as a
validated common-emitter attribution method. The defensible statement is only
that this precise implementation and protocol were evaluated fail-closed on
the listed OAKBAT/TEXBAT support and returned
`NO_GO_CORA_COMMON_ORIGIN_HYPOTHESIS`.
