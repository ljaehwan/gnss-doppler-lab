# GNSS-OpenIF S1 real-multipath validation v1

## Decision

The support-normalized clock-centered geometry-coherence (CGC) score passes
the frozen external real-multipath specificity gates on GNSS-OpenIF S1:

- all 84 one-second bins from receiver time 3 s through 87 s are evaluable;
- every bin contains 7--11 healthy GPS PRNs and the officially identified
  whole-recording multipath satellite G22;
- the unadjusted residual produces 11/84 persistent alarms (13.10%);
- the support-normalized partial-F score produces 2/84 persistent alarms
  (2.38%), below the frozen 5% maximum; and
- all 84 bins meet the frozen correlator-distortion enrichment threshold, so
  the enriched persistent-alarm rate is also 2/84 (2.38%), below 10%.

The decision is REAL_MULTIPATH_SPECIFICITY_SUPPORTED. It supports only
offline specificity on this external urban field recording. It does not
establish real-spoof sensitivity.

## Physical/statistical correction

For measured per-PRN delay vector \(y\), the clock-only null and
direction-plus-clock alternative are

\[
H_0: y_i=c+e_i,\qquad
H_1: y_i=-u_i^\mathsf{T}d+c+e_i,
\]

where \(u_i\) is the satellite line-of-sight unit vector, \(d\) is the common
three-dimensional displacement, and \(c\) is receiver clock bias. The original
score was the residual ratio

\[
r=\mathrm{SSE}_1/\mathrm{SSE}_0.
\]

A fixed threshold on \(r\) is support dependent: \(H_1\) adds three parameters,
and with only seven satellites it leaves just \(N-4=3\) residual degrees of
freedom. It can therefore fit ordinary multipath too easily.

The corrected score tests the same nested physical models with

\[
F=\frac{(\mathrm{SSE}_0-\mathrm{SSE}_1)/3}
        {\mathrm{SSE}_1/(N-4)},\qquad
p=P(F_{3,N-4}\ge F_{\mathrm{observed}}).
\]

Small \(p\) means that a common direction-plus-clock displacement explains the
cross-satellite delays beyond the improvement expected from the three added
degrees of freedom. The alarm threshold \(p\le0.06028418845288192\) is the
fifth percentile of the original cleanStatic calibration interval
\([330,420)\) s. No S1 score or outcome was used to choose it. A persistent
alarm still requires three positives in the causal five-bin window.
Because the estimated-delay errors need not be independent Gaussian samples,
the calibrated tail probability is used as a support-normalized ranking score,
not claimed as an exact frequentist false-alarm probability.

## Support audit before S1 scoring

The already-frozen unadjusted threshold was first stress-tested with 64
deterministic exact-seven-PRN subsets of the existing cleanDynamic and DS7
records. It failed every specificity trial: negative persistent false-alarm
rates ranged from 5.56% to 35.56% with median 20.00%.

The partial-F score passed all 64 specificity trials. Its false-alarm range was
0% to 4.44% with median 0%. This repair has a sensitivity cost on synthetic
DS7: it detected in 59/64 trials (92.19%), but the median delay was 124 s
(range 40--168 s), and only 3/64 trials met the legacy 60 s delay gate.

This support adaptation was made after observing that S1 receiver support could
fall to seven PRNs, but before computing any S1 geometry score. S1 is therefore
an outcome-held-out validation of the adapted score, not a fully
pre-registered validation of the entire method.

## External data and receiver provenance

- Dataset: GNSS-OpenIF commit
  09715cd679a0300782074e96bfed9f11eae4ef88, scenario S1, pedestrian near a
  building in Tsim Sha Tsui, Hong Kong.
- Raw I/Q: 10,200,547,328 bytes, SHA-256
  79952be6e4b8c36d0dee2841d837c731499aef2c8cdcf42890c5bb066007ff91.
- Ground truth SHA-256:
  97991b3b38855cfb8c25eceefaeaf30ce73a37833a2e2cda7d0416334353d6bc.
- Dataset README SHA-256:
  4c42059b2cbdd8bffd7e0932b2e817aac44f76faad8a316145bee174b09baa76.
- Orbit oracle: NOAA CORS validated daily GPS broadcast navigation file
  brdc0440.25n.gz, SHA-256
  702bc509b15a3ccd078d9bd58c7b07047c58c175076d2ab5e6af4670e9925fac.
- Receiver executable SHA-256:
  fc00d610fdb966358ac85486b616ff2411cb163ca8269dd0b435919c5e197e25.

The signed int8 complex stream is read at 58 MHz. Its measured complex-spectrum
GPS L1 center is -4.58 MHz, so the receiver frequency-translates by -4.58 MHz
and decimates by ten to 5.8 MHz. GNSS-SDR records nine complex correlator taps
at 0.125-chip spacing. One full-file run and eight sealed restart windows at
10, 20, 30, 40, 50, 60, 70, and 75 s provide continuous seven-PRN-or-greater
support; overlapping PRN/bin observations are median-merged, never counted as
additional satellites. The nine runs contain 1,116,572 valid receiver epochs
before overlap consolidation.

The external broadcast navigation supplies the offline line-of-sight oracle.
The receiver independently decoded ephemerides for G05, G09, G12, G17, G19,
G20, and G22. At the recording start, selected broadcast ephemeris toe ages are
314--330 s.

## Frozen gates and results

| Gate or metric | Frozen criterion | Result |
|---|---:|---:|
| Evaluable bins | at least 60 | 84 |
| Healthy PRNs per bin | at least 7 | 7--11 |
| G22 tracked | required | 84/84 bins |
| Persistent alarm rate | at most 5% | 2/84 = 2.38% |
| Enriched persistent alarm rate | at most 10% | 2/84 = 2.38% |
| Legacy persistent alarm rate | comparison only | 11/84 = 13.10% |
| Partial-F raw alarm rate | descriptive | 8/84 = 9.52% |
| G22 median leave-one-out error | descriptive | 0.10664 chips |

The two support-normalized persistent positives are adjacent bins 16 and 17.
They are retained as false alarms; no post-result exclusion is applied.

## Sealed artifacts

- Experiment config SHA-256:
  50b68d52cbfff9e66b475c1396fc689e6baa087c290b728d6c2537c2233a0e12.
- Seven-PRN partial-F audit summary SHA-256:
  c07dee4ecc82e9d0ce98c98d7e8ba9b00ba6ff86400406e2cbb86821708c07d4.
- Full S1 result summary SHA-256:
  6b98b218c093544486b7dd24cef20c74157396e5de02f2f142227aabbf420ada.
- Delay-estimate CSV SHA-256:
  75cf9bf90f6b028fed4f9d027648030dbe6fcde58eb8e8eaa84986a4372a048f.
- Geometry-score CSV SHA-256:
  e1c280feed0b1f7169abfa0719147339476833ee5f253c99222bc539d497c451.
- PRN-summary CSV SHA-256:
  d8f79674f341aea510583de9137335caf152926d1d62ae847112d77ace34b8ed.

The 10.2 GB raw I/Q, receiver dumps, and full evaluation tables remain on the
SSD. The repository records the code, hashes, compact result, and support-audit
tables.

## Claim boundary and next evidence

One 84-second recording is not enough for a WCL-level general claim. The
strongest current contribution is the identified finite-support failure mode,
the physically interpretable partial-F correction, and the held-out reduction
of real-multipath false alarms. A paper still needs:

1. independent urban recordings spanning different sites, receivers, satellite
   counts, and geometries;
2. same-pipeline real spoof recordings to quantify the specificity/sensitivity
   tradeoff without relying only on synthetic DS7;
3. an acquisition strategy that maintains support without restart-window
   stitching, or an ablation showing restart invariance; and
4. recording-level uncertainty intervals and ablations by PRN count and
   geometry conditioning.
