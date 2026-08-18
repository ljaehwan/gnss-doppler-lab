# CHORD Stage-0A clean identifiability

CHORD asks one deliberately narrow question: after removing complex amplitude,
global phase/NAV sign, and a small code-delay nuisance from receiver-faithful
complex nine-tap correlations, is the remaining projective direction stable for
one PRN and distinct across PRNs in a stationary, single-antenna clean recording?

This is a physical identifiability gate. It is not a detector, does not choose an
attack threshold, does not inspect attack recordings, and does not train an AI
model. A no-go ends CHORD; a pass permits only a separately preregistered Stage-0B
design, not an attack claim.

For taps `c`, the runner fits `m = alpha a(tau)` on the frozen delay grid. A
fit-only complex-linear shrinkage covariance defines `W`. In real coordinates it
projects `W(c-m)` orthogonally away from the local complex-amplitude/phase and
delay tangent. Samples below the calibration-only residual-norm floor are marked
`DIRECTION_UNAVAILABLE`, never replaced by zero. For available residuals,
`d=r/||r||` and similarity is `|d_a^H d_b|^2`.

Each clean recording is sampled at 2 Hz over 30–430 seconds. Fit, calibration,
and final holdout occupy 190, 76, and 114 seconds respectively, separated by two
10-second guards. Holdout pairs are evaluated at 1, 5, 10, and 30-second lags;
uncertainty resamples recording-time 10-second blocks rather than treating pairs
as IID observations.

The broad ideas of correlated multi-PRN channels, SVD/channel tests, RF
fingerprinting, complex correlator distortion, and two-source decomposition are
prior art and are not claimed as new. Relevant examples include
[US7952519B1](https://patents.google.com/patent/US7952519B1/en), Gallardo et al.,
[Satellite Fingerprinting Methods for GNSS Spoofing Detection](https://www.mdpi.com/1424-8220/24/23/7698),
and Marata et al., [Meta-Learning Based Radio Frequency Fingerprinting for GNSS
Spoofing Detection](https://arxiv.org/abs/2511.00491). In particular, this work
does not claim the first common-source detector or the first RF-fingerprint
detector. Its only tested distinction is the clean-only, stationary,
single-antenna, nuisance-normal complex nine-tap projective residual direction.

## Reproduction

The two-stage command order is intentional:

```bash
python scripts/run_chord_stage0a.py preregister
python scripts/run_chord_stage0a.py evaluate --preregistration-sha <pushed-sha>
python scripts/verify_chord_stage0a.py
```

The evaluator refuses to run if its checked-out commit differs from the supplied
preregistration SHA. Source bindings and every sampled TRACE row are checked
before scientific scoring. Only the two explicit cleanStatic paths encoded in
