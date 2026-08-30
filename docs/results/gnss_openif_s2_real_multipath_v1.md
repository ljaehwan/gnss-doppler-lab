# GNSS-OpenIF S2 real-multipath validation v1

## Decision

The frozen support-normalized clock-centered geometry-coherence (CGC) rule
passes every preregistered specificity gate on the outcome-unseen GNSS-OpenIF
S2 urban-vehicle recording:

- 88 one-second bins are evaluable, exceeding the minimum of 60;
- each evaluated bin contains 8--15 healthy GPS PRNs;
- the fixed partial-F rule produces 3/88 persistent spoof alarms (3.41%),
  below the preregistered 5% maximum; and
- all 88 bins pass the frozen correlator-distortion enrichment threshold, so
  the enriched persistent-alarm rate is also 3/88 (3.41%), below 10%.

The decision is `REAL_MULTIPATH_SPECIFICITY_SUPPORTED`. S2 is a second,
independent real-field multipath-rich negative and, unlike S1, its receiver,
support, detector, analysis interval, and gates were committed before the I/Q
file was downloaded. It supports offline false-alarm transfer only. It does
not establish per-event multipath classification or real-spoof sensitivity.

## Frozen test

The per-PRN signed-delay vector \(y\) is tested under the nested models

\[
H_0:y_i=c+e_i,\qquad
H_1:y_i=-u_i^\mathsf{T}d+c+e_i,
\]

using

\[
F=\frac{(\mathrm{SSE}_0-\mathrm{SSE}_1)/3}
        {\mathrm{SSE}_1/(N-4)},\qquad
p_F=P(F_{3,N-4}\ge F_{\mathrm{observed}}).
\]

The frozen raw alarm is \(p_F\le0.06028418845288192\), and a persistent alarm
requires three raw alarms in the causal five-bin window. The S2 minimum is
eight simultaneous PRNs because the earlier support sweep supported
specificity at \(N=8\)--10 but left a documented seven-PRN boundary. No S2
threshold, persistence rule, PRN minimum, time interval, or restart window was
changed after outcome access.

## External data and receiver provenance

- Dataset: GNSS-OpenIF commit
  `09715cd679a0300782074e96bfed9f11eae4ef88`, scenario S2, a vehicle in the
  dense urban canyon of Mong Kok, Hong Kong, recorded on 2025-06-03.
- Raw I/Q: 11,593,056,256 bytes, SHA-256
  `7dfb51d0973b45b86441f1c50db95079bbf47459fe73fab3ca3d0b6bdcf3a8c5`.
- Ground-truth trajectory SHA-256:
  `c7f7e248a7db8a2d0587c51c42fbf12ab0f07701bfeb6562c0113dfa5333f96c`.
- Dataset README SHA-256:
  `4c42059b2cbdd8bffd7e0932b2e817aac44f76faad8a316145bee174b09baa76`.
- Orbit oracle: NOAA CORS daily GPS broadcast navigation file
  `brdc1540.25n.gz`, SHA-256
  `9164bf9acc8619b37715e9cc7be09cd997293f4686e0fa8536594e3503173a30`.
- Receiver executable SHA-256:
  `fc00d610fdb966358ac85486b616ff2411cb163ca8269dd0b435919c5e197e25`.

The signed-int8 complex stream is read at 58 MHz, frequency-translated at
-4.58 MHz, and decimated by ten to 5.8 MHz. GNSS-SDR records nine complex
correlator taps at 0.125-chip spacing. One full-file run and eight fixed
restart windows beginning at 10, 20, 30, 40, 50, 60, 70, and 80 s contain
1,638,526 valid receiver epochs before overlap consolidation. Overlapping
PRN/bin observations are median-merged and never counted as additional
satellites.

The official README describes ray-tracing-supported short-lived multipath in
S2, but it does not publish machine-readable per-PRN or per-event multipath
labels. The whole spoof-free urban recording is therefore the primary negative
control; the distortion-enriched subset is secondary and must not be described
as event-level ground truth.

## Frozen gates and results

| Gate or metric | Frozen criterion | Result |
|---|---:|---:|
| Evaluable bins | at least 60 | 88 |
| Healthy PRNs per bin | at least 8 | 8--15 |
| Persistent alarm rate | at most 5% | 3/88 = 3.41% |
| Enriched persistent alarm rate | at most 10% | 3/88 = 3.41% |
| Partial-F raw alarm rate | descriptive | 10/88 = 11.36% |
| Legacy raw alarm rate | descriptive | 12/88 = 13.64% |
| Legacy persistent alarm rate | comparison only | 3/88 = 3.41% |

The three partial-F persistent-positive bins are 75--78 s. They form one
causal persistence episode and are retained as false alarms. The legacy score
also yields three persistent-positive bins, at 81--84 s, so S2 independently
supports the absolute false-alarm claim but does not show a persistent-rate
improvement over the legacy score. This differs from S1, where the same
support normalization reduced persistent alarms from 11/84 (13.10%) to 2/84
(2.38%).

## Sealed artifacts

- Preregistration commit: `cf307df`.
- I/Q-hash sealing commit: `c03c57b`.
- Experiment config SHA-256:
  `7e06e7dc7dfa32f55382c2e9acd835eae1a3221541c4eb1afe8eca04b99a3d1e`.
- Support-audit summary SHA-256:
  `c07dee4ecc82e9d0ce98c98d7e8ba9b00ba6ff86400406e2cbb86821708c07d4`.
- Full S2 result summary SHA-256:
  `8730f0176f02412c6bd9d440524b8a1a89e7e83ac9356dec830e5a019eeb404c`.
- Delay-estimate CSV SHA-256:
  `7dc486a6a0e70d601273faa20ce4f9e12ded8138cc6ed29e91dc023d8c44af06`.
- Geometry-score CSV SHA-256:
  `faaca28fbc2d43acf69d3c7802b87060d6709bf73d5fbc40a5ac69b811b77abb`.
- PRN-summary CSV SHA-256:
  `1bb31af9f26eaea1e03c90ed64c654a024e32222e0fa8dc827188e2b1736b497`.

The 11.6 GB raw I/Q, receiver dumps, and full evaluation tables remain on the
SSD. The repository stores the frozen code and protocol, hashes, and compact
result.

## Paper-safe interpretation

Together, S1 and S2 show low persistent false-alarm rates on two real urban
GNSS-OpenIF recordings with different sites, motion modes, dates, and satellite
support. They are two recording-level replications, not 172 independent field
trials. The strongest supported statement is therefore that the frozen CGC
rule transferred with persistent false-alarm rates of 2.38% and 3.41% on the
two recordings. A same-pipeline, sufficiently supported real spoof recording
is still required for a real-data sensitivity claim.
