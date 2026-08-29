# OAKBAT Run-Normalized 9-Tap B-Tail Audit V1

## Status

Exploratory post-hoc external-dataset audit, not a prospective sealed validation.

The frozen GRU weights and a clean-only gate were separated from attack evaluation,
but the run-normalization protocol was designed after inspecting earlier OAKBAT
attack scores. A new independent dataset is required for a confirmatory claim.

## Question

Can a frozen nine-tap morphology score detect OAKBAT spoofing after removing the
receiver/run score offset, and does matched-power position-push create a sustained
cross-PRN physical signature?

## Data and receiver contract

- OAKBAT GPS L1 C/A, 5 MHz interleaved int16 IQ, 480 s per recording.
- Clean calibration recordings: `cleanStatic_gps.bin`, `cleanDynamic_gps.bin`.
- Attack evaluation recordings: OS2, OS3, OS4; attack onset is 120 s.
- GNSS-SDR Method-A, 11 channels, nine taps, 0.125-chip tap spacing.
- All five receiver runs exited successfully and covered approximately 480 s.
- Frozen GRU checkpoint SHA-256:
  `f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b`.

The original TEXBAT gate is not zero-shot transferable: its node thresholds are
below the OAKBAT clean score floor, causing essentially every clean event to alarm.
Those alarms are a domain-offset failure and are not counted as spoof detections.

## Run-normalized detector

For PRN j, use only the trusted startup interval t < 60 s:

```text
m_j = median(r_j,t for t < 60 s)
s_j = max(1.4826 * MAD(r_j,t for t < 60 s), 0.001)
z_j,t = (r_j,t - m_j) / s_j, for t >= 60 s
```

At least 40 startup score rows are required per PRN. No detector output is emitted
before 60 s. The q50/q70/q80 node thresholds and q99 event threshold are fitted
from cleanStatic and cleanDynamic only. Attack evaluation then loads the sealed
calibration without reading either clean score file. Events require at least eight
PRNs and use the existing binomial-tail maximum with previous-state EWMA weight
0.75.

Sealed clean-only calibration SHA-256:

`aeb4e6955e75d85c3aefa3111318692a5485e853597e1e5e568d2d1998b92fc3`

## Main results

The 10 s onset guard excludes 110--130 s. Therefore the reported first
operational delay is constrained to be at least approximately 11 s: 10 s guard plus
the one-second score-availability offset. It must not be interpreted as unconstrained
algorithm latency.

| Scenario | Pre-onset false alarms | Post-onset detections | 95% Wilson interval | First operational delay |
|---|---:|---:|---:|---:|
| OS2, +10 dB time-push | 0/101 (0.00%) | 691/697 (99.14%) | 98.13--99.60% | 11.35 s |
| OS3, matched-power time-push | 3/101 (2.97%) | 35/697 (5.02%) | 3.63--6.90% | 14.28 s |
| OS4, matched-power position-push | 0/100 (0.00%) | 698/698 (100.00%) | 99.45--100.00% | 11.04 s |

Across all attack-recording preambles, the independent negative-control false
alarm rate is 3/302 = 0.993%, with 95% Wilson interval 0.338--2.879%.

OS3 is not a stable detection: its 5.02% post-onset alarm rate is close to sparse
false excursions and no PRN has a sustained post-onset median above the clean q80
node threshold.

## Physical cross-PRN contrast

| Scenario | Median q80-exceeding PRNs, pre -> post | Sustained post-q80 PRNs | Median gate score, pre -> post |
|---|---:|---:|---:|
| OS2 | 2 -> 5 of 11 | 3/11 | 0.895 -> 4.215 |
| OS3 | 2 -> 3 of 11 | 0/11 | 1.020 -> 1.474 |
| OS4 | 2 -> 9 of 11 | 9/11 | 1.011 -> 11.171 |

OS4 is therefore not explained by one or two isolated satellite outliers. Nine of
11 comparable PRNs have a sustained normalized morphology score above the clean
q80 threshold, while OS3 has zero. Since both OS3 and OS4 are matched-power
attacks, the separation is consistent with a position-push-specific, cross-PRN
tracking/correlation deformation rather than a simple received-power cue.

This is a physical interpretation supported by the observed morphology statistics;
it is not yet a proof of the exact RF mechanism. Tap-level ablation and an
independent position-push recording are still required.

## Warm-up sensitivity

Warm-up endpoints 30, 45, 60, 75, and 90 s were audited without selecting a new
final setting from attack performance.

- OS4 detection remains 100% for every warm-up.
- OS2 detection ranges from 91.69% to 99.71%.
- OS3 detection ranges from 0.29% to 5.02% and remains unstable.
- OS4 and OS2 pre-onset false-alarm rate is 0% for every warm-up tested.

This supports the robustness of the OS4 cross-PRN result to the startup-window
choice.

## Reproduction

```bash
.venv/bin/python scripts/eval_oakbat_run_normalized_btail_gate.py calibrate \
  --score-root /home/ubuntu/ssd_data/gnss-early-detection/artifacts/oakbat_9tap_frozen_champion_20260829 \
  --out-dir /home/ubuntu/ssd_data/gnss-early-detection/artifacts/oakbat_9tap_frozen_champion_20260829/run_normalized_btail_v1/calibration

.venv/bin/python scripts/eval_oakbat_run_normalized_btail_gate.py evaluate \
  --score-root /home/ubuntu/ssd_data/gnss-early-detection/artifacts/oakbat_9tap_frozen_champion_20260829 \
  --out-dir /home/ubuntu/ssd_data/gnss-early-detection/artifacts/oakbat_9tap_frozen_champion_20260829/run_normalized_btail_v1/evaluation \
  --calibration-json /home/ubuntu/ssd_data/gnss-early-detection/artifacts/oakbat_9tap_frozen_champion_20260829/run_normalized_btail_v1/calibration/calibration.json \
  --scenarios os2 os3 os4 \
  --onsets-json '{"os2":120.0,"os3":120.0,"os4":120.0}'
```

Primary sealed artifacts:

- `.../run_normalized_btail_v1/calibration/calibration.json`
- `.../run_normalized_btail_v1/evaluation/summary.json`
- `.../run_normalized_btail_v1/warmup_sensitivity.json`

## Claim boundary and next confirmation

OAKBAT has no multipath control, so this experiment validates spoofing response but
not spoofing-versus-multipath discrimination. The defensible present contribution is
a run-offset correction plus a cross-PRN observability result:

1. trusted-startup median/MAD normalization repairs the OAKBAT score-offset failure;
2. matched-power position-push produces a strong sustained common morphology shift;
3. matched-power time-push remains close to the correlator-shape observability limit.

A confirmatory WCL experiment must freeze this V1 protocol before opening a new
position-push dataset and must include a real multipath negative control.
