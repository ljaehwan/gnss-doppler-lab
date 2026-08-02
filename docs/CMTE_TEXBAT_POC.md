# CMTE TEXBAT proof of concept

CMTE is a **sequential conformal evidence detector with empirically calibrated false-alarm control**. It does not claim anytime-valid inference. Attack recordings are evaluation-only and never select a distribution, conformal reference, ablation, drift, detector, or threshold.

## Frozen inputs and provenance

The CLI accepts explicit paths only; it never searches. `scripts/prepare_cmte_texbat_inputs.py` converts explicit cleanStatic/DS1/DS2/DS3 complex NPZ (`complex_iq[N,9,2]`) into causal 1 s, 0.5 s-stride Method-A-compatible nodes. It computes `hypot(I,Q)`, divides each tap by that epoch's prompt magnitude, then takes window means in exact order `E4,E3,E2,E,P,L,L2,L3,L4`. It groups by PRN/segment/channel and splits timestamp gaps, so it never bridges a segment or gap. These products are labeled `reconstructed_equivalence`, not exact historical producer output. DS4 is validated and copied as `verified_node_artifact`. During the real campaign, clean reconstruction is compared with the available historical subset and MAE/max differences are recorded.

Training requires `--checkpoint --expected-sha --clean-node-csv --clean-manifest`. The hardened existing `score_tap_residual_common_drive.py::load_frozen_b0` verifies bytes, architecture, exact features/order, seq_len=12, scaler, and freezes parameters. The manifest is a positive allowlist requiring exact `scenario=cleanStatic`, `role=normal_clean`, source/node/checkpoint hashes and producer grade. Case-insensitive `ds1`-`ds4`, `attack`, `spoof`, or `external-validation` tokens anywhere in paths, metadata, manifest, or rows are rejected. A run ID alone is never trusted.

The checkpoint SHA is `f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b`. B0 was historically trained across cleanStatic with PRN holdout; therefore CMTE clean test is calibration-independent only, not B0-training-independent.

## Split-reset extraction and conformal roles

Canonical node windows are partitioned **before** B0 extraction. Only windows fully inside train `[0,240)`, validation `[250,330)`, and test `[340,∞)` are retained. Role-specific run IDs force fresh PRN histories. Within every role and evaluation recording, each `run_id/prn/segment/channel` identity is sorted deterministically and split again at every non-0.5 s cadence boundary. If `segment_index` is present it is the authoritative source segment identity, so converter gap-piece labels in `segment` cannot conceal a gap. Every scored chunk gets a chunk-specific run identity and its first eligible target has `target_window_index=12`; no predictor window crosses a role, recording, source-segment, channel, or cadence-gap boundary. Chunks with at most `seq_len=12` rows produce no prediction and are dropped without interpolation/fill, with input/scored/dropped row counts and explicit reasons persisted in training residual manifests and per-scenario evaluation provenance audits.

`fit_distribution(train)` fits only shared μ, diagonal scales and a fixed deterministic diagonal-shrinkage covariance. `attach_calibration(state, validation)` freezes all `Q_cal` arrays from validation residuals only. Test and DS rows are query-only. Full and diagonal Mahalanobis scores are squared quantities:

- full: `(r-μ)ᵀΣλ⁻¹(r-μ)`
- diagonal: `Σj((rj-μj)/sj)²`
- ablations: residual RMSE and maximum absolute standardized tap

Full shrinkage is predeclared; attacks cannot select a method. Conformal p-values are `(1 + #{Qcal >= q})/(n+1)`, including ties. Fixed κ `{.25,.5,.75}` mixture e-values use log-sum-exp; clipping constant and count are recorded.

## Baselines and sequential protocol

- A0: epoch maximum retained per-PRN scalar B0 RMSE; validation q99.
- A1: train node-RMSE q50/q70/q80 and exceedance rates; each epoch takes maximum exact-binomial-tail surprise, then causal EWMA α=.75 reset by run; validation q99.
- A2: epoch mean `-log(p_i)` multivariate conformal evidence; validation q99.
- A3: epoch mean scalar RMSE resettable CUSUM; drift/threshold validation-only.
- A4: epoch mean mixture e-value; validation q99. A2 and A4 are intentionally distinct.
- Full: selected S1/S2 sequential evidence.

S1 uses fixed prior restart weights `w_s=2^-(s+1)` plus unstarted reserve. There is no free maximum or floor. S2 is `G=max(0,G+log(E)-drift)`. Fixed drift grid is `[0,.01,.025,.05,.1]`. Normal validation selection minimizes 20 s block-any-alarm fraction then FA/min, ties by higher censored ARL, then fixed S2 and smaller drift. Thresholds are higher finite q99 of 20 s validation block maxima; limited block count is explicit. Reports define FA/min as alarm epochs per observed stable minutes, censored ARL as epochs through first alarm (or full run if none), and block-any as the fraction of reset-aware 20 s blocks with any alarm.

## Commands

```bash
python scripts/prepare_cmte_texbat_inputs.py \
 --cleanstatic-npz /explicit/clean.npz --cleanstatic-source-manifest /explicit/clean.manifest.json \
 --ds1-npz /explicit/ds1.npz --ds1-source-manifest /explicit/ds1.manifest.json \
 --ds2-npz /explicit/ds2.npz --ds2-source-manifest /explicit/ds2.manifest.json \
 --ds3-npz /explicit/ds3.npz --ds3-source-manifest /explicit/ds3.manifest.json \
 --ds4-node-csv /explicit/ds4_nodes.csv --checkpoint-sha f171...a6b \
 --out /new/cmte-input-bundle

python scripts/train_cmte_texbat.py --checkpoint /explicit/b0.pt \
 --expected-sha f171...a6b --clean-node-csv /bundle/cleanStatic_nodes.csv \
 --clean-manifest /bundle/cleanStatic_manifest.json --out /new/cmte-state

python scripts/eval_cmte_texbat.py --state-dir /cmte-state --checkpoint /explicit/b0.pt \
 --expected-sha f171...a6b \
 --scenario DS1=/bundle/DS1_nodes.csv=/bundle/DS1_manifest.json \
 --scenario DS2=/bundle/DS2_nodes.csv=/bundle/DS2_manifest.json \
 --scenario DS3=/bundle/DS3_nodes.csv=/bundle/DS3_manifest.json \
 --scenario DS4=/bundle/DS4_nodes.csv=/bundle/DS4_manifest.json --out /new/cmte-eval
```

Evaluation emits actual scenario/ablation CSVs, per-epoch and per-PRN evidence, ROC/PR/FPR/FA/min/detection/delay/persistence summaries, PNG plots, order-shuffle diagnostics, provenance, checksums, README and finalized GO criteria. Historical B0 summaries remain separately labeled `historical_noncomparable`.
