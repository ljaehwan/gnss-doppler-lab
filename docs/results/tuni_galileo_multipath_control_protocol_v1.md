# TUNI Galileo multipath control protocol v1

## Purpose and boundary

This is a locked external-control evaluation of the clean-only Galileo
correlator-geometry-coherence (CGC) model. It asks whether authentic PRNs in a
labelled multipath recording are falsely grouped as spoofed, and whether two-
or four-PRN true-position spoof groups retain coherent correlator deformation.

The model, thresholds, input decoder, and nine-tap receiver patch were frozen
using TUNI C-1 only at commit `12e8e34`. No SS-1/3/5/11/12/13 I/Q payload was
decoded, feature-extracted, or scored before this protocol and its conforming
runner were committed. Download-time byte counts and MD5 verification, README
inspection, and metadata hashing are not scientific outcome access.

This is controlled-laboratory Galileo E1 evidence. It is not GPS L1 evidence,
an uncontrolled urban field recording, or a carry-off displacement test. The
TUNI spoofers transmit a true-position solution, so a failure to detect them
can identify a mechanism boundary rather than invalidate the simulated
nonzero-displacement result.

## Frozen signal processing

- Decode the raw files as interleaved signed 16-bit I and Q at 50 MSps.
- Process exactly 10 complex-sample seconds per recording, represented as
  1,000,000,000 16-bit source items.
- Resample to 12.5 MHz and track Galileo E1B with twelve channels.
- Dump nine complex taps at 0.125-chip spacing from E4 through L4.
- Exclude the first 1.0 s of receiver time as a tracking transient.
- Form 0.25 s bins with at least twenty epochs per PRN.
- Divide every complex profile by its prompt tap, remove the prompt, and use
  the sixteen real/imaginary non-prompt components.
- Apply the frozen clean median and robust per-component scale without any
  refit on evaluation recordings.

A PRN has coherent evidence in a bin only if it and another PRN both have
robust residual RMS at least 1.5 and their residual cosine is at least 0.8. A
PRN-level detection requires four consecutive evidence bins. The continuous
ranking score is the 95th percentile over bins of the strongest pairwise joint
score, defined before labels are evaluated.

## Frozen roster and labels

| Scenario | Multipath | Spoofed PRNs | Primary role |
|---|:---:|---|---|
| SS-1 | no | 9 | single-spoofer no-MP reference |
| SS-3 | no | 6, 9, 23 | multi-spoofer no-MP reference |
| SS-5 | no | 4, 6, 9, 23, 31 | multi-spoofer no-MP reference |
| SS-11 | yes | 31 | same-stream authentic-MP specificity; single-spoofer boundary |
| SS-12 | yes | 9, 31 | two-spoofer MP sensitivity and specificity |
| SS-13 | yes | 5, 9, 23, 31 | four-spoofer MP sensitivity and specificity |

Every acquired PRN absent from the scenario's spoofed list is an authentic PRN
under that recording's labelled propagation condition. Labels come from the
scenario README and are not inferred from detector output.

## Support and decision gates

Primary sensitivity uses spoofed PRNs from SS-12 and SS-13 because the detector
mechanism explicitly requires a PRN pair. Primary specificity uses authentic
PRNs from all three multipath recordings.

Support is sufficient only if all six receiver runs complete, both SS-12
targets are acquired, at least three of four SS-13 targets are acquired, and
each multipath recording supplies at least two authentic PRNs.

Report `SUPPORTED` only if support is sufficient and all of these frozen gates
pass:

1. pooled PRN-level AUC in SS-12/13 is at least 0.80;
2. at least 50% of supported SS-12/13 spoof targets are persistently flagged;
3. at most 10% of supported authentic PRNs in SS-11/12/13 are persistently
   flagged.

Report `SPECIFICITY_ONLY` if support and the authentic false-positive gate pass
but the sensitivity or AUC gate fails. Report `INSUFFICIENT_SUPPORT` if the
support gate fails. Otherwise report `NOT_SUPPORTED`.

No threshold, interval, target label, receiver setting, scenario, or exclusion
may change after release. No result-driven retry is allowed. A technical resume
may reuse a receiver directory only when its executable and config hashes match
the frozen inputs and no aggregate metrics have been emitted.

