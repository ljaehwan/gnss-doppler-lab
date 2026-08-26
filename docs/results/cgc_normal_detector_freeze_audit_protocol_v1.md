# CGC train-normal detector-freeze audit protocol v1

Date frozen: 2026-08-26
Status: preregistered before complex nine-tap processing of normal train pairs
002--006

## Objective

Determine whether the frozen clock-centered CGC candidate remains finite and
non-alarming on authentic normal RF, and whether a deterministic train-only
threshold plus a two-bin persistence rule can be frozen before any test access.

This is an internal detector-freeze audit. It is not validation.

## Frozen candidate

The score remains unchanged:

    Rdir = SSE_LOS+clock / SSE_clock-only
    score = -Rdir
    alarm if score > threshold

The existing numerical contract also remains unchanged. If the clock-centered
energy is at most 1e-12, the residual is one and the score is minus one. Such a
bin is retained as a finite benign-valued observation; no epsilon, feature,
estimator, rank rule, or score direction may be changed after outcome
inspection.

## Authorized data

Only train pairs 002--006 are authorized.

For each pair, the existing normal RF manifest and IQ hash are pinned in
`configs/experiments/cgc_normal_detector_freeze_audit_v1.json`. Existing
multipath and carry-off geometry scores are reused byte-for-byte from the
preregistered train replication.

Pair 001 is accessible only through frozen records. Validation pairs 007--009,
test pairs 010--012, and TEXBAT attack recordings remain forbidden.

## Receiver and analysis controls

Each normal RF recording is processed through the same pinned complex nine-tap
GNSS-SDR used for multipath and spoofing.

- one-second bins
- at least eight LOS-matched PRNs
- required geometry rank four
- normal audit window: all complete bins starting at one second
- multipath and spoof window: the already frozen per-pair post-transition
  boundary
- at least 20 normal bins per pair
- at least five multipath and spoof bins per pair

Every config, RF manifest, IQ, LOS, receiver executable, patch, frozen result,
and analysis input is hash checked.

## Frozen threshold procedure

Threshold selection uses leave-one-pair-out train evaluation.

For each of five folds:

1. Hold out one complete pair.
2. Fit a threshold using the other four pairs.
3. Treat normal and independent multipath as benign and carry-off spoof as
   positive.
4. Evaluate threshold candidates at adjacent-score midpoints plus finite
   boundary candidates.
5. Maximize macro balanced accuracy across pair-scenario streams.
6. Break ties by lower macro benign false-positive rate and then the higher,
   more conservative threshold.
7. Apply the selected threshold to the held-out pair only.

An alarm requires two consecutive eligible one-second bins above threshold.
The final threshold proposed for the still-locked test is the median of the
five fold-training thresholds. It is written even if gates fail, but it is
declared frozen and usable only if every support gate passes. Refitting after
the audit is forbidden.

## Frozen support gates

All conditions must hold:

1. Exactly five pairs complete.
2. Every analyzed score is finite.
3. Every analyzed bin has rank four.
4. No held-out normal pair has a persistent alarm.
5. At most one held-out multipath pair has a persistent alarm.
6. At least four held-out spoof pairs have a persistent detection.
7. Leave-one-pair-out macro balanced accuracy is at least 0.80.

Failure means the current candidate cannot be frozen as an operational
detector. A failed gate must be reported; test data remain locked.

## Claim boundary

Passing supports only a train-derived detector freeze and permits writing a
still-locked test protocol. It does not estimate field false-alarm probability,
validate an operating threshold, or establish test, TEXBAT, operational, or
WCL generalization.
