# CMTE-A2 canonical decision epoch policy

## Status

This policy defines a **developmental/exploratory post-exposure diagnostic**. The existing `artifacts/cmte_a2_texbat_ds78/` PRIMARY INVALID result remains immutable. DS7/DS8 have already been exposed and are not confirmatory holdouts.

## Source timestamp semantics

The frozen per-PRN residual table contains:

- `window_end_s`: residual availability time after the causal 1 s input window;
- `window_bin_s`: producer-local nominal window bin that can precede availability;
- `target_window_index`: history/channel-local target index, not a global cross-channel epoch ID.

Therefore only `window_end_s` is used for decision availability. Exact floating equality is never used as an epoch key.

## Canonical grid

- recording-relative origin: `0.0 s`
- stride: `0.5 s`
- decision times: `t_k = 0.0 + 0.5*k`
- mapping: a residual first becomes eligible at `ceil((window_end_s - 1e-9)/0.5)*0.5`
- numerical timestamp tolerance: `1e-9 s`
- maximum residual age: `0.55 s`

At each decision time and for each PRN:

1. reject any residual with `window_end_s > decision_time + 1e-9`;
2. reject residuals older than `0.55 s`;
3. select the latest eligible residual;
4. on an exact availability tie, select by a deterministic SHA-256 of stable row content;
5. include each PRN at most once.

A row can be carried forward only while its age is within the fixed maximum. Recording IDs are processed independently. History/segment/channel switches replace older rows when the first new residual becomes available; cadence gaps expire through the age limit. PRN identity is used only as the grouping key and for diagnostics, never as a predictive feature.

The CMTE-A2 score is:

```text
S_A2(t) = mean_i[-log(p_i(t))]
```

Input row order and PRN ordering do not affect the score.

## Frozen/recomputed components

Reused after checksum verification:

- chronological B0 checkpoint and scaler;
- per-PRN signed residuals;
- covariance and conformal `Q_cal`.

Recomputed on this grid:

- cleanStatic `[300,330)` A2/A0/B0 event scores;
- q99, q99.5 and target-1% thresholds;
- independent clean test FPR;
- DS1–DS4/DS7/DS8 scores;
- B0-Exact and B0-Enhanced multi-PRN event streams;
- clean-test matched-FPR diagnostic thresholds, without attack data.

## Fixed tests

The policy commit must pass tests for jittered timestamp coalescing, unique PRN contribution, no future data, row and PRN permutation invariance, small floating jitter invariance, variable cardinality, boundary/gap expiry, actual DS7 `median N>1`, and exact manual multi-PRN mean agreement.
