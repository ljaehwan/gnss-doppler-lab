# TUNI Galileo authentic-window support scan v1 result

Date: 2026-08-28
Status: FAIL on the preregistered control-pair support gate

## Question

The initial first-10-second TUNI evaluation produced no trackable authentic
PRN in multipath scenarios SS-11, SS-12, or SS-13.  This preregistered scan
tested whether later parts of the approximately 300-second recordings contain
a receiver-usable same-stream multipath control pair.

This was an input-support audit only.  It did not compute or inspect detector
features, scores, thresholds, AUC, detection rate, or false-alarm rate.

## Frozen design

- Scenarios: SS-11, SS-12, and SS-13.
- Windows: 10 seconds beginning at 30, 90, 150, 210, and 270 seconds.
- Channels: every PRN not named as spoofed in the scenario README.
- Receiver: the frozen Galileo-safe nine-tap executable with unchanged
  acquisition, resampling, and tracking settings.
- Sustained support: at least 1,000 valid tracking epochs.
- Usable control window: at least two sustained authentic PRNs in the same
  scenario and window.
- Overall support: every scenario must contain at least one usable control
  window.

The protocol and runner were committed at `429a5a6` before execution.  The
clean release head was `78331a0`; the later commit only added the already
completed Tokyo result and did not change this scanner or its protocol.

## Result

| Scenario | 30 s | 90 s | 150 s | 210 s | 270 s | Any control pair |
|---|---|---|---|---|---|---|
| SS-11 | none | none | none | none | none | no |
| SS-12 | none | none | none | none | none | no |
| SS-13 | none | none | none | PRN 6: 2,333 epochs | PRN 6: 1,297 epochs | no |

All 15 preregistered scenario-window receiver runs completed.  SS-11 and SS-12
produced no nonempty authentic tracking output in any later window.  SS-13
provided one sustained authentic PRN in two windows, but never the two-PRN
pair required by the cross-PRN coherence detector.

The overall result is therefore
`all_scenarios_have_control_pair_window=false`.

Primary artifact:

- `artifacts/tuni_galileo_authentic_window_scan_v1/summary.json`
- SHA-256:
  `7258b8d16215ea435a0cc9ecd086e2852ffb861654c423775ade6cb61ed8105e`
- 15 receiver logs; artifact size approximately 12 MB.

## Decision

The planned detector AUC, spoof detection rate, and authentic false-alarm rate
must not be computed from this experiment.  The detector requires at least two
simultaneously tracked PRNs to test cross-PRN coherence, and the negative class
does not meet that support condition.

TUNI's README provides a metadata-level statement that authentic Galileo
signals are present, but this pinned receiver exposes no usable authentic pair
in any of the six tested portions of the recordings (the initial window plus
five later windows).  For the present receiver chain, TUNI SS-11--SS-13 are
therefore unsuitable for validating the claim that real multipath is not
mistaken for spoofing.

## Claim boundary and next action

The simulated independent-multipath result remains valid, as does the Tokyo
independent-seed replication.  The real-multipath specificity claim remains
unverified and must not be inferred from TUNI.

The next confirmatory route requires a different data source with at least two
sustained authentic PRNs under labelled multipath.  Appropriate options are a
new controlled RF collection with clean, multipath-only, spoof-only, and
spoof-plus-multipath conditions, or an independently sourced multipath-only
raw-IQ corpus.  Changing acquisition sensitivity on TUNI would be a separate
receiver-compatibility study, not a continuation or repair of this frozen test.
