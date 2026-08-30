# GNSS-OpenIF S2 real-multipath protocol v1

## Purpose

This is the second external real-field multipath negative for CGC. GNSS-OpenIF
S2 was recorded from a vehicle in a dense urban canyon in Mong Kok on
2025-06-03. It differs from the opened S1 pedestrian/building recording in
date, site, receiver motion, satellite geometry, blockage, and noise.

The primary question is whether the S1-frozen support-normalized CGC rule keeps
its persistent spoof-alarm rate below 5% without any S2-specific refit. This is
a false-alarm transfer test, not a real-spoof sensitivity test.

## Outcome-blind release boundary

This protocol, the generic receiver/evaluator code, the deterministic restart
schedule, and every detector parameter must be committed before the S2 IQ file
is downloaded. The repository currently records the IQ hash as
`PENDING_DOWNLOAD`, which makes the evaluator refuse to run.

After download, the only permitted amendment is to replace
`PENDING_DOWNLOAD` with the full-file SHA-256 after the exact
11,593,056,256-byte size is verified. Receiver settings, analysis interval,
support threshold, score threshold, persistence, and gates may not change
after IQ access. Failed support or performance gates remain terminal results.

## Frozen signal and receiver contract

- GPS L1 complex int8 I/Q, 58 MHz, observed spectral center -4.58 MHz.
- The same GNSS-SDR executable used for S1, SHA-256
  `fc00d610fdb966358ac85486b616ff2411cb163ca8269dd0b435919c5e197e25`.
- Nine complex correlator taps at 0.125-chip spacing.
- 31 acquisition channels, PFA 0.01, five maximum dwells.
- One full-file run plus fixed restart windows at 10, 20, ..., 80 s.
- Restart-window observations for the same PRN/bin are median-consolidated and
  never counted as extra satellites.
- The full run must hash-bind its receiver manifest to the sealed IQ file.

The restart schedule is fixed before signal access to prevent support-driven
window selection. It is a receiver-support mechanism, not data augmentation.

## Frozen detector

For each one-second bin, CGC compares

[
H_0: y_i=c+e_i,qquad
H_1: y_i=-u_i^T d+c+e_i
]

and uses the nested-model tail score

[
F=rac{(SSE_0-SSE_1)/3}{SSE_1/(N-4)},qquad
p_F=P(F_{3,N-4}geq F).
]

The fixed alarm is `p_F <= 0.06028418845288192`; a persistent alarm requires
three positive bins in the latest causal five-bin window. No S2 calibration is
allowed. To avoid the known seven-PRN boundary, the primary analysis requires
at least eight simultaneously eligible PRNs, each with at least 200 epochs.

## Frozen analysis and gates

- Analysis interval: receiver seconds [3, 96).
- Primary unit: the recording; serial one-second bins are descriptive.
- Minimum evaluable bins: 60.
- Primary persistent spoof-alarm rate: at most 5%.
- Multipath-enriched persistent alarm rate: at most 10%.
- Report raw and persistent alarms, PRN support, evaluable coverage, geometry
  conditioning, and the unadjusted-residual comparison.
- No bin, PRN, restart run, or time interval may be removed after score access.

The official S2 README describes ray-tracing-supported short-lived multipath
events but does not publish machine-readable per-event PRN labels. Therefore
the whole spoof-free urban recording is the primary negative; the existing
correlator-distortion enrichment is secondary and cannot be presented as
event-level ground truth.

## Decision language

If every gate passes, the allowed statement is: the frozen CGC rule transfers
with low persistent false alarms to a second, independent multipath-rich urban
recording. If support is insufficient, report `INSUFFICIENT_SUPPORT`. If a
false-alarm gate fails, report `SPECIFICITY_NOT_SUPPORTED`. None of these
outcomes establishes real-spoof sensitivity or operational detection.
