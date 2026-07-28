# Synthetic Normal-v3 30-run Negative Result

Date fixed: 2026-07-28

## Status

This experiment completed end-to-end, but it **failed as a deployable detector**
and does not replace the frozen real-clean baseline in
[`../BINOMIAL_TAIL_GATE_BASELINE.md`](../BINOMIAL_TAIL_GATE_BASELINE.md).
The failure is retained because it establishes a concrete sim-to-real limitation.

## Frozen experiment input

The exact 30-run selection is tracked at:

```text
configs/experiments/normal_v3_synthetic_only_30.csv
```

- train: 24 runs
- validation: 3 runs
- test/calibration: 3 runs
- scenario profile: `open_sky_normal`
- duration: 300 seconds per run
- attack data used for training or calibration: no

The realistic RF impairment implementation was present at commit `cee2d35`
(`feat(rf): add realistic normal IQ impairment pipeline`). Generated worker
shards, IQ, GNSS-SDR raw dumps, score dumps, and the failed model checkpoint are
not tracked.

## Model contract

- architecture: PRN-local GRU next-window predictor
- input: exactly nine prompt-normalized Method-A taps
- feature subset: `tap_rel_prompt_mean`
- PRN identity input: no
- receiver graph or PRN-relation input: no
- receiver score label: `btail_max_507080_ewma075`
- calibration: synthetic validation + synthetic test only

Important provenance limitation: this negative experiment used a reconstructed gate
evaluator before the original frozen probe's exact `-ln`, `window_bin_s`, and
previous-state-weight EWMA contract was recovered. Its numeric event threshold must
therefore not be compared directly with the frozen real-clean threshold. The
negative conclusion remains valid because all pre-onset windows were flagged and,
independently of the gate, the raw PRN-local RMSE decreased after spoof onset in all
four TEXBAT scenarios.

Feature columns:

```text
tap_E4_rel_prompt_mean
tap_E3_rel_prompt_mean
tap_E2_rel_prompt_mean
tap_E_rel_prompt_mean
tap_P_rel_prompt_mean
tap_L_rel_prompt_mean
tap_L2_rel_prompt_mean
tap_L3_rel_prompt_mean
tap_L4_rel_prompt_mean
```

## Data and training result

- 3-tap feature rows: 153,252
- 9-tap feature/node rows: 155,980
- 9-tap graph rows exported for diagnostics: 17,948
- train node rows: 122,137
- train sequence windows: 88,989
- validation sequence windows: 23,761
- best validation loss: 0.0202450761
- final train/validation/test non-finite counts: 0
- synthetic node thresholds:
  - q50: 0.0724881403
  - q70: 0.0982629001
  - q80: 0.1171906114
- synthetic event q99: 1.8010452966

During the 3-tap worker merge, six rows from three runs contained non-finite
`cn0_std`. Those rows were removed without imputation and the original CSVs were
preserved locally as `*.raw_nonfinite.csv`. This manual recovery is a provenance
limitation; the final 9-tap train/validation/test inputs contained no non-finite
values.

## Frozen TEXBAT evaluation

All thresholds and model weights were fixed before opening TEXBAT ds1-ds4.

- ds1: pre-onset FP 100%, post detection 100%
- ds2: pre-onset FP 100%, post detection 100%
- ds3: pre-onset FP 100%, post detection 95.95%
- ds4: pre-onset FP 100%, post detection 100%

The apparent zero-second detection delays are invalid evidence because every
pre-onset window was already flagged.

The PRN-local raw RMSE also moved in the wrong direction on real spoofing data:
its median decreased after onset in ds1-ds4. In ds2 and ds4 the aggregate
binomial-tail score increased only because all PRNs were already above the
synthetic thresholds and the tracked PRN count increased. Therefore threshold
recalibration alone cannot rescue this model.

## Scientific interpretation

The model learned the synthetic normal domain, not a real-data spoof-specific
morphology score. Real TEXBAT clean prefixes were approximately 5.5 times higher
than the synthetic calibration median in PRN RMSE. A stronger or cleaner spoofed
signal can be more simulator-like than the authentic real prefix and therefore
reduce prediction error.

Valid uses of the generated normal data are limited to:

- normal-only pretraining
- augmentation after anchoring to real clean data
- RF/receiver pipeline stress tests
- feature-extractor regression tests
- negative-result and domain-gap analysis

It must not be presented as the current champion. The next external check is a
fully frozen evaluation on OAKBAT GPS cleanStatic/cleanDynamic and os1-os6, with
raw pre/post score direction reported before any threshold tuning.
