# TUNI GPS SS-29 score-free support preflight v1

## Purpose

SS-29 is an unopened TUNI2025 GPS L1 recording containing authentic signals,
multipath, and four true-position spoofers (PRNs 1, 2, 21, and 32). This
preflight may answer only whether the recording supplies enough simultaneous
tracking support for the already frozen CGC detector. It must not compute a
delay estimate, geometry residual, partial-F value, alarm, or detection rate.

## Frozen access boundary

- The official README and Spectracom scenario file may be read before release.
- Only bytes `[0, 1,999,999,999]` of the raw IQ file may be downloaded during
  this preflight. The server must return HTTP `206` with the matching
  `Content-Range`; a full-file response fails closed.
- The prefix is processed by the frozen big-endian GPS L1 C/A complex-nine-tap
  receiver for 10 seconds.
- No learned delay template is loaded. No detector module or frozen threshold
  is loaded. PRN identifiers, epoch counts, and complex-tap schema presence are
  the only permitted outputs.

## Gate

The fixed analysis interval is `[5, 10)` seconds in five one-second bins. A PRN
is eligible in a bin only with at least 200 tracked epochs. The recording passes
only if:

1. at least 8 eligible PRNs occur in at least 3 of the 5 bins; and
2. each documented spoofed PRN (1, 2, 21, 32) is eligible in at least 3 bins.

Duplicate receiver channels cannot inflate a PRN: the maximum channel epoch
count is used for each PRN/bin.

## Terminal decisions

- `SUPPORT_ELIGIBLE`: the complete raw recording may be downloaded after a
  separate one-shot detector protocol is committed.
- `INSUFFICIENT_SUPPORT`: SS-29 is not used for the CGC confirmatory test.
- `DOWNLOAD_OR_RECEIVER_FAILURE`: no scientific conclusion is drawn.

The gate is immutable after the raw prefix is accessed. A failure cannot be
rescued by changing the interval, epoch threshold, PRN threshold, or target
coverage rule.
