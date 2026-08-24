# Receiver-quality score contract

## Purpose

PRN-local sequence models must not predict a target from history that crosses a
receiver tracking loss, reacquisition, or missing-window gap. The versioned
contract implemented in `receiver_quality_contract.py` makes that invariant
machine-checkable while preserving receiver state needed for later
quality-conditioned calibration.

The score schema is
`gnss-doppler-lab.prn-local-quality-score.v1`.

## Source requirements

Every node window must contain:

- Identity: `run_id`, `prn`
- Receiver state: `channel`, `segment_index`, `window_index`, `epoch_count`
- Timing: `window_bin_s`, `window_start_s`, `window_mid_s`,
  `window_end_s`
- All model feature columns

`channel`, `segment_index`, and `window_index` are nonnegative integers.
`epoch_count` is a positive integer. Source identity
`(run_id, prn, channel, segment_index, window_index)` and event identity
`(run_id, prn, window_bin_s)` must both be unique.

## Boundary algorithm

For each raw receiver segment `(r, p, c, s)`, sort windows by
`(window_index, window_bin_s)`. Start a new continuity block whenever

$$
\Delta window\_index \ne 1
\quad\text{or}\quad
|\Delta window\_bin\_s-\delta|>\epsilon .
$$

Here, `delta` is the configured feature stride and `epsilon = 1e-6 s` by
default.

For history length `L`, target `i` is eligible only when its entire history

$$
(x_{i-L},\ldots,x_{i-1})
$$

belongs to the same continuity block. The standardized next-window anomaly
score remains

$$
e_i =
\sqrt{\frac{1}{d}
\left\|
f_\theta(x_{i-L:i-1})-x_i
\right\|_2^2}.
$$

Thus the change does not introduce attack labels, change the frozen model, or
alter the residual definition. It removes temporally invalid samples.

## Reacquisition semantics

Raw `segment_index` is channel-local and is not a reacquisition flag. For each
`(run_id, prn)`, receiver segments are ordered by first observed time. Their
zero-based order is `prn_segment_ordinal`.

- `reacquisition_flag = 1[prn_segment_ordinal > 0]`
- `sequence_restart_flag = 1` for a later PRN segment or a later continuity
  block within the same receiver segment

This distinction matters: a first observed segment may have raw
`segment_index=3`, while a later reacquired segment may have raw
`segment_index=0`.

## Score metadata

Each PRN score preserves the raw receiver identity and adds:

- Segment state: `prn_segment_ordinal`, `continuity_block_index`,
  `reacquisition_flag`, `sequence_restart_flag`
- Target state: `target_window_index`, `target_sequence_position`,
  `epoch_count`
- Causal ages: `tracking_age_s`, `continuity_age_s`, `segment_start_s`
- Auditable history bounds: `history_start_window_index`,
  `history_end_window_index`, `history_start_s`, `history_end_s`,
  `history_length`, `history_same_segment_flag`

`tracking_age_s = window_index × stride_s`; it therefore remains the
receiver's absolute segment age even when a chronological partition begins in
the middle of a segment. `continuity_age_s` is age from the first row visible
in the uninterrupted block.

These fields are receiver bookkeeping available no later than the target
window. Same-window C/N0, code error, and residual statistics are deliberately
excluded from the quality contract because spoofing can alter them.

## Integration

- Generic PRN-GRU training constructs one series per continuity block.
- TEXBAT scoring writes the v1 metadata and accepts explicit `--stride-s`.
- OAKBAT's legacy `score_partition` remains unchanged so the frozen v1
  campaign can still be semantically revalidated.
- OAKBAT attack evaluation uses `score_partition_with_quality` and emits an
  evaluation v2 report. Old cached v1 scores fail the new schema check and are
  rescored.
