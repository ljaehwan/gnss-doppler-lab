# CLIF-IP synthetic-normal R4

## Scope

This new pipeline builds **60 final, normal-only, run-level split** recordings: 30 per target domain, each fixed at 120 s (24 train, 3 validation, 3 synthetic test). Location, UTC, run ID, and impairment seed are disjoint across splits. A smoke campaign is stored under `artifacts/clif_ip_synthetic_normal_r4/smoke` and never enters the final index.

## Verified target contract

The installed `gps-sdr-sim` advertises and was probed with arbitrary `-s` and `-b 16`. R4 therefore generates target format directly—SYN-OAK 5,000,000 complex samples/s and SYN-TEX 25,000,000 complex samples/s—using native host little-endian `short`, i.e. interleaved `<i2` I,Q / GNSS-SDR `ishort`. Exact bytes are `duration * Fs * 2 * 2`. No duplication, zero padding, or resampling is used.

The bounded state machine is: native clean target IQ → streaming open-sky frontend impairments → hash/size check → format-aware M1 memmap extraction (10 ms blocks, 0.5 s stride, exact sample ranges) → the specified Method-A complex9 GNSS-SDR binary → receiver-backed 9-tap feature and node export → finite/nonzero/common-IQ-hash validation → atomic manifest and `_SUCCESS` → transient IQ and receiver MAT removal. Failures retain IQ and dumps. Resume accepts only a manifest whose `_SUCCESS` contains its current hash.

All configured normal variation is recorded: AWGN/CN0, CFO, CFO drift, phase noise, frontend bandwidth, IQ gain/phase imbalance, DC I/Q, AGC gain, quantization and clipping. Attack and spoofing are false.

## B0 wording

Method-A tracking exports **tap magnitudes**. “Signed 9-tap innovation” means the canonical `(E4,E3,E2,E1,P,L1,L2,L3,L4)` standardized residual `x - xhat` from the shared PRN-local GRU. It is not claimed to be raw signed complex correlator output. Zero placeholders fail extraction.

## Leakage-safe regimes

- **R0:** exact R3 result bundle, read-only; never refit or overwritten.
- **S0:** normalization, M1 PCA/AR, B0, P0–P3, covariance and thresholds fit on each target's synthetic train/validation only.
- **S1:** S0 pretrain followed by real cleanStatic chronological adaptation only. OAKBAT follows the R3 cleanStatic contract. TEXBAT static cleanStatic is primary. Attack pre-onset data is never used for fitting/calibration. Missing TEXBAT/DS1 M1 must be extracted from raw IQ with the same format-aware extractor. Any DS4 morphology reconstruction must be explicitly graded `reconstructed`; it cannot be silently substituted.

M1 and B0 histories reset at every run. B0 has no PRN identity input and keys history by `(run_id, prn)`. P0–P3 use identical targets/support; P1/P3 share lag/regularization candidates and standardization. P3 shuffled control is capacity matched.

## Evaluation

OAKBAT os1–os4 and TEXBAT DS1–DS4 are primary. TEXBAT uses stable pre 30–90 s, excludes transition 90–110 s, and evaluates established post at ≥110 s (nominal onset 100 s). Thresholds are per regime/domain and normal-validation-only; independent clean FPR is reported separately. Metrics include signed 9D prediction MSE/MAE/improvement/incremental R², B0/M1/fusions/P3/Full, domain-gap SMD/Wasserstein/MMD and the reported 5.5× RMSE ratio diagnostic.

Final alignment destruction uses clean test, attack pre, and established attack regions independently, shuffling M1 in region-local 8-epoch blocks for 199 repetitions. P2/P3 predictions and Full are recalculated; support/marginals are preserved. Raw replicate CSV gives p-value resolution `1/(199+1)=0.005`. Smoke uses three permutations.

A Full success claim is allowed only when P3 MSE < P1 MSE, Full AUC exceeds M1 and simple fusion, and independent normal FPR is controlled. Unavailable results are explicit NA with a reason; metrics are never fabricated.

## Commands

```bash
PY=/home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python
$PY scripts/generate_clif_synthetic_normal.py --index-only
$PY scripts/generate_clif_synthetic_normal.py --resume --domains SYN-OAK SYN-TEX --duration 120
$PY scripts/train_clif_synthetic.py --regimes R0 S0 S1
$PY scripts/eval_clif_synthetic.py --permutations 199 --scores artifacts/clif_ip_synthetic_normal_r4/all_scores.csv
$PY scripts/finalize_clif_synthetic.py
```

Do not launch the final generation command during the smoke/code stage.
