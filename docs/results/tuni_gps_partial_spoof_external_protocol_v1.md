# TUNI GPS partial-spoofer external protocol v1

## Question

Does the already frozen support-normalized correlator-geometry statistic
transfer, without refitting, from simulated common carry-off and TEXBAT replay
to independent real RF containing one, two, or four spoofed GPS PRNs?

This is deliberately a boundary test. TUNI SS-17, SS-18, and SS-20 contain
partial-constellation true-position spoofers from the beginning of each
recording. They are not delayed-onset, all-channel carry-off attacks. A
negative result therefore limits the mechanism's scope and is not to be
relabelled as receiver failure.

## Sealed data boundary

Before this protocol and its runner are committed and pushed, the SS-17,
SS-18, and SS-20 raw payloads may only have been downloaded and checksumed.
No correlator, tracking, feature, score, or outcome from those attack payloads
has been inspected. C-5 is the only TUNI GPS raw recording previously opened;
its clean ten-second compatibility preflight is pinned in the configuration.

The official scenario labels are:

| Recording | Label | Documented spoofed PRNs |
|---|---|---|
| C-5 | static, no multipath, clear-sky | none |
| SS-17 | static, no multipath, true position, one spoofer | 1 |
| SS-18 | static, no multipath, true position, two spoofers | 1, 2 |
| SS-20 | static, no multipath, true position, four spoofers | 1, 2, 21, 32 |

The SS-20 Zenodo record and payload name are authoritative. Its README body
inconsistently calls the scenario SS-19; this discrepancy is retained as
provenance rather than silently corrected.

## Frozen receiver and physical statistic

All four recordings are processed contiguously for 149.99916 s with the same
31-channel GPS L1 C/A receiver, big-endian `ishort` adapter, 50-to-5 MHz direct
resampler, and complex nine-tap tracker at 0.125-chip spacing. No restart
windows, acquisition threshold change, PRN allowlist, or attack-specific
receiver setting is permitted.

For every one-second bin and PRN, the frozen simulation-trained nine-tap
template estimates a relative code-peak delay. The nested models are

    H0: y_i = c + e_i
    H1: y_i = -u_i^T d + c + e_i.

The receiver-decoded ephemeris snapshot supplies one LOS per PRN. Because the
recordings are static, only 110 s are analysed and some authentic and simulated
PRNs have different navigation time bases, each PRN's decoded snapshot TOW is
used and its LOS is held fixed. This is an offline geometry oracle, not an
operational timing claim. The receiver position is the robust median C-5 NMEA
GGA fix in [60,140] s and is reused for every recording; attack-derived PVT is
prohibited.

The primary score is the already frozen partial-F tail probability

    F = ((SSE0 - SSE1) / 3) / (SSE1 / (N - 4)),
    p = Pr(F_(3,N-4) >= F).

The raw alarm is `p <= 0.06028418845288192`. A persistent alarm requires at
least three raw alarms in the causal five-bin window. The old unadjusted
residual threshold 0.33023636533817136 is reported only as an ablation.

## Support and time rules

- Primary bins require at least eight PRNs, because the preregistered support
  sweep passed specificity at N=8--10 and left a seven-PRN boundary.
- Exactly seven-PRN bins are reported as a secondary boundary analysis and
  cannot satisfy a primary gate.
- The fixed analysis interval is `[30,140)` s. The first 30 s are receiver
  acquisition/navigation stabilization, not a clean pre-attack interval.
- Each PRN bin requires at least 200 tracking epochs.
- Every recording requires at least 60 primary bins.
- Every documented spoofed PRN must appear in at least one eligible delay bin;
  otherwise the experiment is `INSUFFICIENT_SUPPORT`.

Since attack transmission is present from recording start, this experiment
does not estimate attack-onset delay. It reports the first persistent-alarm
time after recording start and serial-bin AUC against C-5 descriptively.

## Frozen decisions

Clean specificity passes when the C-5 primary persistent-alarm rate is at most
5%. An attack recording is detected when it contains any primary persistent
alarm. Sensitivity passes when at least two of SS-17/18/20 are detected and
SS-20 is one of them.

The terminal decision is:

1. `INSUFFICIENT_SUPPORT` if any input/receiver/ephemeris/position/support gate
   fails;
2. `REAL_PARTIAL_SPOOF_TRANSFER_SUPPORTED` if support, clean specificity, and
   attack sensitivity all pass;
3. `SPECIFICITY_ONLY_DETECTION_NOT_SUPPORTED` if support and clean specificity
   pass but sensitivity fails; or
4. `REAL_PARTIAL_SPOOF_TRANSFER_NOT_SUPPORTED` otherwise.

No threshold, persistence, interval, target list, support requirement, LOS
rule, exclusion, or decision gate may change after attack outcomes are opened.
A support failure is reported as such; it is not repaired by a receiver retune
inside v1.

## Claim boundary

A positive result would support transfer to controlled real partial-PRN,
true-position, no-multipath RF. It would not establish field multipath
specificity, all-channel carry-off latency, live operation, or a universal
distance threshold. A negative result would show that the common-displacement
CGC mechanism does not automatically generalize to sparse true-position
spoofers, while leaving the earlier simulation, TEXBAT directional, and
GNSS-OpenIF S1 specificity results intact.
