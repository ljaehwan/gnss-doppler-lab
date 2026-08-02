# CMTE TEXBAT proof of concept

## Scope and claim

CMTE is a **sequential conformal evidence detector with empirically calibrated false-alarm control**. This POC does not claim anytime-valid inference: dependence and exchangeability assumptions needed for that claim have not been proved. Attack files are evaluation-only and cannot affect fit, drift, detector choice, or thresholds.

The frozen B0 checkpoint is pinned to SHA-256 `f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b`. B0 uses causal length-12 history and history is reset at every split/run boundary. An independent clean test split is independent of CMTE calibration, but it is **not** independent of historical B0 training: B0 was trained on the entire cleanStatic recording with PRN holdout.

## Data semantics and timing

The nine raw taps are prompt-relative magnitudes in exact order `E4,E3,E2,E,P,L,L2,L3,L4`. CMTE inputs `residual_000` through `residual_008` are signed standardized `target - prediction` B0 innovations in that order. PRN identity is metadata, never a feature. `b0_prn_node_rmse` is checked against `sqrt(mean(r²))`.

Every epoch is keyed within `run_id`; its scoring availability is `max(window_end_s)` over participating PRNs. Windows and all sequential updates are causal. Empty epochs are skipped, and summaries without eligible epochs are NaN. One PRN remains valid. Epoch aggregates (`mean e`, minimum/median p, maximum/top-quartile mean e, and N) are permutation invariant.

The nominal TEXBAT onset is 100 s. Availability-time masks are stable `[30,90)`, transition `[90,110)`, and established `[110,∞)`. Boundary inclusion is fixed exactly as written.

## Immutable residual inputs

The implementation never searches for residual files: paths must be explicit. The live external inputs verified during this code stage were:

- `cleanStatic/texbat_cleanStatic_prn_local_scores.csv`: `9a6bc537bd8f1bc16a17257a5f7ae2e47f327c10e215c63d7ebd82ca0b80c36a`
- `ds1/texbat_ds1_prn_local_scores.csv`: `a8ae0c15428dbe90961272cd35163efa65853b1b00eca7864a7f16a2a287aba1`
- `ds2/texbat_ds2_prn_local_scores.csv`: `97fc00c309789e4ac88e9dfde57fc0ee9b9f16a51a08998236f69a8d6311ad4e`
- `ds3/texbat_ds3_prn_local_scores.csv`: `c62321e25fb152c8ece0fef45c3e79bb27d6679bf6ea37ef09b5db711df800d6`
- `ds4/texbat_ds4_prn_local_scores.csv`: `0669581b8a4f3ae73fe1279eb345e1651e3de16955a8ce5eec77d2c1541ae18d`

They are under `/home/ubuntu/projects/gnss-doppler-lab/artifacts/ai_morph_gru_cleanStatic_q70_frame/scored/`. These hashes document discovery only; this stage does not create real-data artifacts.

## Fit and scores

Only clean train `[0,240)` fits shared state: mean `μ`, diagonal scales, and a deterministic diagonal-shrinkage covariance

`Σλ = (1-λ) Σemp + λ diag(Σemp) + εI`, with `ε > 0`.

No PRN-specific parameter is fit. Clean validation `[250,330)` alone selects thresholds, drift, and S1/S2 stability. Clean test is `>=340`. The CLI rejects empty roles and overlap. No `ds*` row or scenario prefix may enter train/calibration.

Four methods are predeclared and all are emitted:

- RMSE: `sqrt(mean(r²))`
- diagonal Mahalanobis: `sqrt(mean(((r-μ)/s)²))`
- full shrinkage Mahalanobis (default): `sqrt((r-μ)^T Σλ^-1 (r-μ))`
- maximum standardized tap: `max |(r-μ)/s|`

The default is fixed before attacks; an attack cannot select an ablation.

For calibration scores `q₁…qₙ`, the exact upper-tail conformal p-value is

`p(q) = (1 + #{i: qᵢ >= q}) / (n + 1)`.

Ties are inclusive and the finite-sample `+1` is mandatory. Evidence uses fixed `κ ∈ {.25,.5,.75}`:

`eκ(p) = κ p^(κ-1)`, and `e(p) = meanκ eκ(p)`.

It is evaluated by log-sum-exp with a recorded lower p clipping constant so values remain finite.

## Sequential detectors and validation protocol

S1 is a causal equal mixture of parallel restart capitals (including the unit no-bet capital), reported as nonnegative log capital. S2 is resettable e-CUSUM:

`Gₜ = max(0, Gₜ₋₁ + log(Eₜ) - d)`.

Both reset per split and run. S1 is distinct from the simple drift-zero max recursion; S2 with `d=0` is that max recursion. Drift candidates are fixed at `[0,.01,.025,.05,.1]` and selected by normal validation stability only. S1/S2 validation block behavior is audited; ties choose S2. Sequence thresholds are finite-sample quantiles of 20 s validation block maxima. Reports include epoch FPR, false alarms/minute, normal ARL, and blocks with any alarm. No attack prefix is calibration data.

Baselines are protocol-frozen:

- A0: scalar B0 RMSE, validation q99.
- A1: train q50/q70/q80 node thresholds and train exceedance rates, exact binomial upper tail, causal EWMA α=.75, validation q99 (q70 default).
- A2: epoch full-Mahalanobis conformal evidence threshold.
- A3: RMSE CUSUM with validation drift/threshold.
- A4: epoch mean mixture-e.
- Full: validation-chosen sequential CMTE.

Historical frozen-B0 result files are labeled `historical_noncomparable`; they are not substituted for protocol-matched baselines.

## CLI

Training never searches for inputs and never opens DS files:

```bash
python scripts/train_cmte_texbat.py \
  --clean-prn-csv /explicit/cleanStatic_prn_local_scores.csv \
  --checkpoint /explicit/prn_local_gru_predictor.pt \
  --expected-sha f171bf0b...c96a6b --out /new/state-dir
```

It writes deterministic JSON state/config/training/calibration/threshold documents plus clean per-PRN and per-epoch scores. The state includes an internal canonical SHA-256.

Evaluation verifies the frozen state and checkpoint pin **before** opening explicit scenarios, and performs no attack tuning:

```bash
python scripts/eval_cmte_texbat.py --state-dir /state-dir \
  --scenario ds1=/explicit/ds1_prn_local_scores.csv \
  --scenario ds2=/explicit/ds2_prn_local_scores.csv \
  --scenario ds3=/explicit/ds3_prn_local_scores.csv \
  --scenario ds4=/explicit/ds4_prn_local_scores.csv --out /new/eval-dir
```

Outputs contain scenario, ablation, per-epoch, per-PRN, diagnostics, plots guidance, provenance, checksums, and a README. The fixed-seed attack epoch-order shuffle is diagnostic only: epoch-score multisets are unchanged, while a sequential trajectory may change.

## Reproducibility and smoke boundary

`tests/test_cmte.py` uses synthetic fixtures only and covers formulas, leakage gates, timing/reset contracts, permutation and edge cases, SPD covariance, clipping, sequential resets, exact masks, validation-only thresholds, shuffle diagnostics, state hashes, RMSE, and tap metadata. A synthetic CLI smoke may use a dummy file whose bytes are explicitly hashed and passed as its expected hash. Such a smoke is not represented as real frozen-checkpoint provenance and creates no final real-data artifact. Real TEXBAT artifacts belong to the next execution stage.
