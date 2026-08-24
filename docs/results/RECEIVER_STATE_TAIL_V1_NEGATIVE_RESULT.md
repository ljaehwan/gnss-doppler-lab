# Receiver-state conditional tail v1 — negative result

## Decision

Do **not** promote receiver-state conditional calibration as the primary WCL
detector on the current evidence. It removed one of two held-clean false flags,
but lost post-onset detections and increased first-detection delay on both DS4
and DS8. The fixed no-degradation gate therefore failed.

This remains a useful, reproducible ablation: it separates the benefit of the
receiver-quality data contract from the unsupported claim that conditioning
the frozen GRU score on receiver state improves spoofing detection.

## Frozen protocol

- Frozen local model:
  `artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt`
- Model SHA-256:
  `f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b`
- Clean runs: TEXBAT cleanStatic and cleanDynamic
- Attack evaluations: TEXBAT DS4 (onset 100 s) and DS8 (onset 110 s)
- Clean split, independently within each run and without splitting an event:
  first 60% local reference, next 20% event-threshold calibration, final 20%
  held-clean evaluation
- Receiver age bins: `[0, 10)`, `[10, 30)`, and `[30, inf)` seconds
- Minimum reference-pool size: 100 PRN rows
- Event threshold: clean event-calibration q99
- Guard interval for pre/post rates: 10 s
- Score availability: window start plus 1 s

The clean reference pools and both event thresholds were frozen before either
attack score CSV was opened. DS4 and DS8 labels did not select bins, pools,
thresholds, or fallback behavior in the final campaign.

DS4 and DS8 had already been inspected during the earlier v0 study and an
exploratory prototype. This frozen run is therefore a reproducible confirmation
of that implementation result, not an independent external validation.

`epoch_count` was preserved in the score contract but deliberately excluded
from conditioning. In these captures it changes between receiver integration
or update-rate regimes and is not a portable lock-quality measure.

## Algorithm

Let `s_i,t` be the frozen PRN-local GRU anomaly score. The causal receiver
state is

`z_i,t = (origin_i,t, age_bin(tracking_age_i,t))`,

where `origin` is `initial`, `gap_restart`, or `reacquired`. A clean reference
pool is selected in the fixed order exact state, pooled age bin, then all clean
rows. For sorted clean pool `R`, the finite-sample right-tail conformal value is

`p_i,t = (1 + sum_{r in R} 1[r >= s_i,t]) / (|R| + 1)`.

For `alpha` in `{0.5, 0.3, 0.2}`, simultaneous PRN evidence is

`K_t(alpha) = sum_i 1[p_i,t <= alpha]`

and

`B_t(alpha) = -log P(X >= K_t(alpha)), X ~ Binomial(n_t, alpha)`.

The raw event score is `max_alpha B_t(alpha)`. A causal EWMA with previous
weight 0.75 produces the final event score. The matched global control uses
the same formula, clean rows, alphas, split, and q99 threshold, differing only
in that every PRN uses the global clean pool.

Because PRNs and adjacent windows are dependent, the binomial tail is treated
as an aggregation score, not as a calibrated hypothesis-test p-value. Its
operating threshold is estimated separately on clean event windows.

## Calibration support

| Quantity | Rows / events |
|---|---:|
| Clean local-reference rows | 2,025 |
| Clean local-reference events | 200 |
| Event-threshold calibration rows | 678 |
| Event-threshold calibration events | 66 |
| Held-clean rows | 662 |
| Held-clean events | 68 |
| Early-age reference rows | 176 |
| Middle-age reference rows | 820 |
| Mature-age reference rows | 1,029 |

Both clean runs contain only `initial` receiver origins in these score ranges.
Consequently, DS8's 908 `gap_restart` rows and 174 `reacquired` rows use the
fixed age-bin fallback. This experiment therefore does not estimate a
clean reacquisition-specific distribution.

The fitted event thresholds were 4.221579 for the matched global detector and
3.878587 for the receiver-state detector.

## Results

| Scenario | Detector | Pre/clean flags | Pre/clean FPR | Post flags | Post TPR | First delay |
|---|---|---:|---:|---:|---:|---:|
| Held clean | Global | 2/68 | 2.941% | — | — | — |
| Held clean | State | 1/68 | 1.471% | — | — | — |
| DS4 | Global | 0/169 | 0% | 28/34 | 82.353% | 13.417 s |
| DS4 | State | 0/169 | 0% | 27/34 | 79.412% | 13.917 s |
| DS8 | Global | 0/189 | 0% | 582/698 | 83.381% | 68.291 s |
| DS8 | State | 0/189 | 0% | 580/698 | 83.095% | 69.291 s |

The paired flag audit is unambiguous:

- Held clean: one event flagged by both detectors, one by global only, zero by
  state only.
- DS4 post-onset: 27 flagged by both, one by global only, zero by state only.
- DS8 post-onset: 580 flagged by both, two by global only, zero by state only.

Thus the apparent clean improvement is one event out of only 68 held-clean
events, while the same suppression also removes attack detections. It is not a
free false-alarm reduction.

The decision checks were:

| Check | Result |
|---|---|
| Fewer held-clean false flags | Pass |
| No increase in attack pre-onset flags | Pass |
| No loss of post-onset attack detections | **Fail** |
| No increase in first-detection delay | **Fail** |
| Overall gate | **Fail** |

## Reproduction

Generate receiver-quality-contract v1 scores with the frozen model:

```bash
.venv/bin/python scripts/score_texbat_prn_node_gru.py \
  --node-csv artifacts/ai_morph_gru_window_ablation_ds4_20260723/w1.0_s0.5/cleanStatic/multi_prn_method_a_9tap_normalized_dmcpd_w1.0_s0.5/normal_prn_node_windows.csv \
  --model-dir artifacts/ai_morph_gru_cleanStatic_q70_frame \
  --out-dir artifacts/receiver_state_tail_v1/quality_scores \
  --scenario cleanStatic --clean-only --stride-s 0.5

.venv/bin/python scripts/score_texbat_prn_node_gru.py \
  --node-csv artifacts/ai_morph_gru_window_ablation_ds4_20260723/w1.0_s0.5/cleanDynamic/multi_prn_method_a_9tap_normalized_dmcpd_w1.0_s0.5/normal_prn_node_windows.csv \
  --model-dir artifacts/ai_morph_gru_cleanStatic_q70_frame \
  --out-dir artifacts/receiver_state_tail_v1/quality_scores \
  --scenario cleanDynamic --clean-only --stride-s 0.5

.venv/bin/python scripts/score_texbat_prn_node_gru.py \
  --node-csv artifacts/ai_morph_gru_window_ablation_ds4_20260723/w1.0_s0.5/ds4/multi_prn_method_a_9tap_normalized_dmcpd_w1.0_s0.5/normal_prn_node_windows.csv \
  --model-dir artifacts/ai_morph_gru_cleanStatic_q70_frame \
  --out-dir artifacts/receiver_state_tail_v1/quality_scores \
  --scenario ds4 --onset-s 100 --stride-s 0.5

.venv/bin/python scripts/score_texbat_prn_node_gru.py \
  --node-csv artifacts/texbat_ds8_9tap_eval_20260724/ds8/multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv \
  --model-dir artifacts/ai_morph_gru_cleanStatic_q70_frame \
  --out-dir artifacts/receiver_state_tail_v1/quality_scores \
  --scenario ds8 --onset-s 110 --stride-s 0.5
```

Then run the frozen campaign:

```bash
.venv/bin/python scripts/eval_receiver_state_tail.py \
  --manifest configs/experiments/receiver_state_tail_v1.json \
  --out-dir artifacts/receiver_state_tail_v1/evaluation
```

Tracked machine-readable outputs are
`docs/results/receiver_state_tail_v1_summary.json` and
`docs/results/receiver_state_tail_v1_comparison.csv`.

## Paper interpretation and next gate

This result is appropriate as an ablation or negative result, not as the WCL
headline contribution. The receiver-quality contract remains necessary to
prevent histories from crossing receiver boundaries, but receiver-state tail
conditioning does not improve the present detector under a strict matched
comparison.

Do not tune the age cutoffs or fallback policy against DS4/DS8. A scientifically
meaningful retry requires independent clean captures containing real tracking
loss and reacquisition. Without that support, the next useful direction is a
different spoofing-sensitive invariant rather than another calibration sweep.
