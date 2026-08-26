# CGC train-normal detector-freeze audit v1: negative result

Date: 2026-08-26
Status: FAIL on the preregistered detector-freeze gate

## Question

Can the frozen clock-centered CGC score be converted into one transferable
absolute threshold that does not persistently alarm on normal or independent
multipath RF while still detecting carry-off spoofing?

This audit used only train pairs 002--006. It was an internal freeze gate, not
validation.

## Preregistration and data boundary

The audit protocol and all support gates were frozen before complex nine-tap
normal outcomes were processed.

- config:
  `configs/experiments/cgc_normal_detector_freeze_audit_v1.json`
- config SHA-256:
  `4fe2a1a97ceaa4ce7bebd679396e8ef1d3ed0fbf5fb6395a3334b6ce9b9771bf`
- protocol:
  `docs/results/cgc_normal_detector_freeze_audit_protocol_v1.md`
- protocol commit: `200bbb2`
- runner commit before normal processing: `8eb5a4a`

Only normal train RF for pairs 002--006 and the already frozen train
multipath/spoof geometry rows were authorized. Validation pairs 007--009, test
pairs 010--012, and TEXBAT attack recordings were not accessed.

The score, epsilon policy, windows, leave-one-pair-out threshold procedure,
two-consecutive-bin persistence, and gates were unchanged after outcome
inspection.

## Frozen detector candidate

    Rdir = SSE_LOS+clock / SSE_clock-only
    score = -Rdir
    alarm if score > threshold
    persistent alarm = two consecutive eligible alarm bins

For each leave-one-pair-out fold, four pairs selected the threshold and the
fifth complete pair was evaluated. Normal and independent multipath were
benign; carry-off spoofing was positive.

The final proposed threshold was the median of the five fold-training
thresholds. It is recorded but is not usable because the freeze gates failed.

## Serialization-only interruption

The first analysis invocation completed score calculations in memory but
stopped while writing a mixed-schema CSV. Normal rows and frozen replication
rows had different extra provenance columns. No summary or threshold artifact
was written and no performance values were printed.

The CSV writer was changed only to use the union of row fields. The score,
threshold search, persistence, and gates were not changed. The fix and a
regression test were committed as `5bb4527` before the identical analysis was
rerun.

## Numerical stability result

All normal scores were finite and every eligible geometry fit had rank four.

| Pair | Domain | Normal bins | Median Rdir | Minimum clock energy | Degenerate bins |
|---|---|---:|---:|---:|---:|
| 002 | static | 28 | 0.7321 | 0.04920 | 0 |
| 003 | dynamic straight | 28 | 0.8035 | 0.03295 | 0 |
| 004 | dynamic straight | 28 | 0.7471 | 0.04134 | 0 |
| 005 | dynamic circle | 28 | 0.6889 | 0.03557 | 0 |
| 006 | dynamic parallel sweep | 28 | 0.6278 | 0.01889 | 0 |

Across 140 eligible normal bins:

- finite score fraction: 1.0
- full-rank fraction: 1.0
- clock-energy degeneracy count: 0

Therefore the failure was not caused by a zero or near-zero denominator.

## Leave-one-pair-out result

| Held-out pair | Fold threshold | Normal alarm rate | Multipath alarm rate | Spoof alarm rate | Persistent N/M/S |
|---|---:|---:|---:|---:|---|
| 002 | -0.3278 | 0.0357 | 0.0000 | 0.7778 | no / no / yes |
| 003 | -0.3278 | 0.0357 | 0.0000 | 1.0000 | no / no / yes |
| 004 | -0.2452 | 0.0357 | 0.0000 | 0.4444 | no / no / yes |
| 005 | -0.4787 | 0.3214 | 0.8571 | 1.0000 | yes / yes / yes |
| 006 | -0.3278 | 0.0714 | 0.0000 | 1.0000 | no / no / yes |

Aggregate cross-validated metrics:

- macro normal false-positive rate: 0.1000
- macro multipath false-positive rate: 0.1714
- macro benign false-positive rate: 0.1357
- macro spoof true-positive rate: 0.8444
- macro balanced accuracy: 0.8544
- persistent spoof detections: 5/5
- persistent normal alarms: 1/5
- persistent multipath alarms: 1/5
- proposed final threshold: -0.327800133845091

## Frozen gate decision

| Gate | Required | Observed | Result |
|---|---:|---:|---:|
| Pair count | 5 | 5 | PASS |
| Finite score fraction | 1.0 | 1.0 | PASS |
| Full-rank fraction | 1.0 | 1.0 | PASS |
| Persistent normal alarm pairs | 0 | 1 | **FAIL** |
| Persistent multipath alarm pairs | at most 1 | 1 | PASS |
| Persistent spoof detections | at least 4 | 5 | PASS |
| LOPO macro balanced accuracy | at least 0.80 | 0.8544 | PASS |

Exactly one gate failed, but the preregistered rule requires every gate to pass.
The detector-freeze decision is therefore negative. The threshold artifact
sets `usable_for_locked_test=false`.

## Pair-005 root-cause diagnosis

When pair 005 was held out, the other four pairs selected the much more
permissive threshold -0.4787. Pair 005 then produced nine normal alarm bins,
including consecutive runs at bins 10--12 and 26--27. Its multipath stream also
alarmed in six of seven eligible bins.

This was not numerical denominator instability:

- median clock energy in pair-005 normal alarm bins: 0.1773
- median clock energy in non-alarm bins: 0.1060
- median PRN count in both groups: 11
- score-to-clock-energy correlation: -0.0419

The evidence instead indicates a geometry/domain transfer problem. The
circular-motion pair has a benign high-score tail not represented by the
other four training geometries. The relative multipath-versus-spoof separation
still exists, but one absolute threshold does not transfer cleanly.

## Independent verification

The `lopo_scored_rows.csv` artifact was re-read independently of the result
writer. All five normal, multipath, and spoof rates, aggregate balanced
accuracy, and three persistent-event counts matched the JSON result exactly.

Primary SHA-256 values:

- `summary.json`:
  `60794d951dbdeaeb5ef703a49c9bfcd6c3d714911e0e2e62215f90d86a4dec54`
- `threshold_freeze.json`:
  `fcc425fff1097ec77679c79e417041ced20ed44af7b2ddaba81025423199a651`
- `lopo_scored_rows.csv`:
  `ca3bb1b7f7c21f8f120208a08f3a9fd5e103e8fcdd0de1d41d6ce723d2caabca`
- `normal_geometry_scores.csv`:
  `32450ade0e936fce20917899d64bb7f1ab6a95c8443c6450f2f783d07be69220`

## Scientific interpretation

The earlier positive result remains valid within its claim: clock-centered
CGC consistently ranks coherent spoofing above satellite-specific multipath at
the pair level.

This audit adds an important limitation: the absolute score baseline varies
with receiver motion and satellite geometry enough to cause persistent normal
alarms in an unseen train pair. A good pair-level separation is not by itself
a deployable detector.

The current threshold must not be sent to the locked test. Test pairs remain
unopened.

## Next research gate

A new train-only candidate should target geometry-invariant calibration rather
than adjust this threshold after seeing pair 005. A defensible direction is a
causal within-run baseline or change statistic that removes the per-geometry
score level while preserving a coherent carry-off increase. It must be defined,
preregistered, and evaluated on train pairs before any test protocol is frozen.

Simply increasing persistence or hand-adjusting the threshold based on pair 005
would be post-hoc and is not acceptable evidence.

## Claim boundary

This result does not estimate field false-alarm probability, validate an
operating threshold, or establish test, TEXBAT, operational, or WCL
generalization. It is a train-only negative detector-freeze result that
identifies geometry-dependent score calibration as the next narrow problem.

## Artifacts and SSD archive

Local artifact:

    artifacts/cgc_normal_detector_freeze_audit_v1

SSD archive:

    /home/ubuntu/ssd_data/gnss-early-detection/artifacts/cgc-normal-detector-freeze-audit-v1

The SSD archive contains 157 regular files totaling 364,008,914 bytes. A full
`rsync --checksum --dry-run --itemize-changes` produced no differences, and
the SSD `summary.json` SHA-256 is
`60794d951dbdeaeb5ef703a49c9bfcd6c3d714911e0e2e62215f90d86a4dec54`.
