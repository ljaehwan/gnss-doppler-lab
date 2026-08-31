# FGI-SpoofRepo TGD cadence-corrected support preflight v2

Date frozen: 2026-08-31

## Purpose

This protocol corrects a measurement-contract error in the score-free v1
support gate.  It does not change the detector, inspect signed delays, compute a
CGC score, fit a threshold, or open an alarm result.

The v1 gate required 200 tracking rows per PRN in each one-second bin.  Direct
inspection of the pinned receiver output established that the GNSS-SDR tracking
dump changes cadence after GPS navigation-bit synchronization: the same
continuously tracked PRN is recorded near 1 kHz before synchronization and at
50 Hz afterward.  Consequently, v1 incorrectly marked stable, telemetry-decoded
authentic channels as absent while retaining newly acquired high-rate channels.
The terminal v1 result remains preserved as an invalid support measurement and
must not be reinterpreted as a detector miss.

## Pinned input

V2 reuses the exact 240 s receiver output from v1; the RF file is not replayed.
The receiver manifest, receiver configuration, 31 tracking MAT files, nine
complex taps, clean interval 40--120 s, and post-onset support interval
160--230 s remain unchanged.  Their paths and hashes are frozen in
`configs/experiments/fgi_spoofrepo_tgd_support_preflight_v2.json`.

## Corrected support rule

For each one-second bin and PRN, duplicate receiver channels are collapsed by
taking the largest row count.  A PRN is supported when it has at least 40 rows
in that bin.  This is 80% occupancy at the observed, protocol-defined 50 Hz
telemetry-synchronized dump cadence.  The 40-row rule is a cadence correction,
not a relaxation selected from a detector score.

An interval is eligible only when at least eight unique GPS PRNs satisfy that
rule in at least 60 one-second bins.  The eight-PRN geometry requirement and all
time intervals are unchanged from v1.

## Terminal outcomes

- `SUPPORT_ELIGIBLE`: both frozen intervals meet the corrected support rule.
- `INSUFFICIENT_SUPPORT`: either interval fails it.

Only a `SUPPORT_ELIGIBLE` result permits a separately frozen, one-shot detector
evaluation.  V2 itself makes no spoofing-detection claim.

