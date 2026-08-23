# Literature review

## Search protocol

The search was performed through 2026-08-23. Primary claim sources were IEEE
Xplore records, Institute of Navigation publication records, official
journal/conference pages, and author or institution manuscript pages. arXiv
was used only to discover or cross-check author manuscripts; it is not the
sole support for any claim in this audit.

Query families combined “GNSS/GPS spoofing” with TEXBAT, causal onset/change
detection, correlation distortion, multi-correlator/CAF, tracking-loop
innovation, code phase, carrier phase, Doppler consistency, common emitter,
clock state, normal-only/anomaly/self-supervised learning, RF fingerprint,
cross-dataset, cross-receiver, telemetry, and navigation-message consistency.
Backward/forward method matching was then done from the official records.
The structured source ledger records DOI, inputs, method, and candidate
overlap. This is a design audit, not a claim of an exhaustive bibliometric
meta-analysis.

## Findings by observable family

### TEXBAT and causal evaluation

The original TEXBAT publication defines controlled attack recordings rather
than an indefinitely reusable blind benchmark. Lemmenes et al. subsequently
documented scenario timing, power offsets, RF artifacts, and global
code/carrier range-rate divergence. Together with this repository's extensive
DS1-DS4 and DS7-DS8 exposure, these records require a development/confirmatory
family split and predeclared causal metrics; they do not support calling a
new reuse “blind.”

Published papers often report ROC or aggregate detection, but the prospective
task here requires decisions at time (t) to use no sample after (t),
official onset to be withheld from features, first-alarm delay to be measured
only after freezing decisions, and confidence intervals to resample
recording-safe temporal blocks.

### Correlation distortion and tracking response

Turner et al. use the full delay-Doppler surface, propagation-channel
decomposition, and a likelihood-ratio test on simulations, TEXBAT, and live
data. Wesson et al. join correlation distortion with power. Wang et al. use
maximum-likelihood mixture estimation. Recent multiscale-eye work again
encodes correlator shape and evaluates TEXBAT. Peng et al. directly study
tracking-loop parameter response to intermediate spoofing.

These are method-level overlaps with B0, MOSAIC, PG-SCC, MIRAGE, MCTD, CRID,
and related repository branches. Replacing their estimators with a CNN, GRU,
Transformer, autoencoder, graph network, or multiscale image does not create a
new physical hypothesis.

### Doppler, code/carrier, clock, and solution closure

Gao et al. derive a code/carrier phase consistency detector. Chu et al. compare
measured and geometry-calculated Doppler in a GLRT. Zhou et al. use
Doppler-positioning residual energy and satellite discrimination. Clock-bias
characterization and the 2026 self-consistent clock-state method explicitly
use receiver clock observables and pseudorange/Doppler consistency.

This literature supports the physical plausibility of Candidate 1, but also
creates the decisive novelty problem: the candidate combines well-established
closures rather than exposing a clearly new physical variable. Its strongest
possible contribution would be causal normal-only calibration and
availability masking, which is methodological engineering and overlaps this
repository's prior sequential work.

### Common emitter, RF fingerprint, and learned anomaly scores

Spatial-correlation work explains why common-emitter detection becomes
identifiable with antenna motion or multiple spatial observations. The
single-front-end repository setting does not provide that independent spatial
channel. GCI-GANomaly and RF/TSVD work show active contemporary interest in
learned RF fingerprints and anomaly encoders, including TEXBAT, but they do
not remove receiver/session/power confounding. Repository CORA, CINDER,
BITPROBE, M1, and NC-TOPI controls already expose those failure mechanisms.

### NAV and telemetry consistency

Message parity, TOW progression, issue-of-data, and ephemeris validity are
causal and signal-specific. They offer an observable distinct from analog
tracking shape. However, an adversary that replays or generates valid
navigation content can preserve them while changing delay and receiver time.
Thus the family is useful against malformed or inconsistent spoofers, not
physically identifying for the stated TEXBAT threat. Candidate 2 fails before
novelty becomes the deciding question.

### Normal-only learning and cross-domain generalization

Normal-only reconstruction or latent anomaly scoring does not create
identifiability: it estimates the support of whichever nuisance-laden feature
is supplied. The explicit SSRL-UAVs study likewise shows self-supervision as
a representation-learning strategy over its supplied navigation variables,
not a new source-authentication measurement. Similarly, published supervised
cross-dataset performance does
not establish cross-receiver transfer if recording, front-end, scenario, or
power cues remain available. A valid GPS-to-Galileo external test would have
to freeze the structure and physical logic while preregistering only the
signal-specific observable canonicalizer.

## Candidate-to-prior-art conclusion

Candidate 1 is closest to Gao 2013, Chu 2017, Zhou 2022, low-cost clock-bias
characterization, and the 2026 receiver-clock self-consistency method. It is
physically testable but insufficiently novel for the requested WCL
contribution.

Candidate 2 is less duplicated inside the repository, but its coherent-replay
null space is fundamental. No model architecture can recover information that
valid NAV content does not carry about RF source authenticity.

The search therefore reinforces, rather than merely fails to contradict, the
terminal decision: no candidate reaches the frozen scientific and novelty
bar.
