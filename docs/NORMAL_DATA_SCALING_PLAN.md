# Normal data scaling plan for conditional integrated PRN-relation model

## Current retained artifacts

Keep only the current useful dataset/model artifacts:

- `artifacts/model_datasets/normal_multi_prn_morphology_dynamics_v2/`
- `artifacts/conditional_integrated_gru_poc/`
- `artifacts/trajectories/`

Bulky raw IQ and receiver intermediate outputs should be treated as disposable unless a run is being debugged.

## Goal

Train a normal-only model that learns local PRN tracking/correlation behavior first and uses receiver-level PRN relation context conditionally when local PRN surprise is high.

The dataset must not encourage memorizing fixed PRN IDs or fixed PRN constellations. Generate many independent normal runs across different times, places, durations, and motion profiles, then split by complete run/location/time.

## Recommended extraction scale

### Stage A: fast expansion

- 30-50 normal runs
- 2-5 minutes each
- mix of static and simple motion
- delete raw IQ after receiver tracking and feature extraction succeeds
- expected final CSV size: small enough to keep in git-ignored artifacts

### Stage B: serious training

- 100-300 normal runs
- 5-10 minutes each
- multiple cities / dates / times of day
- static, straight, circle, stop-go, sweep trajectories
- split by run and location, not random rows

### Stage C: paper-grade normal corpus

- 500+ runs or many hours of clean recordings/simulated authentic data
- multiple RINEX days
- multiple receiver positions and trajectory types
- include front-end/noise/clock/channel augmentations if simulation-only
- reserve full locations/days for validation/test

## Run design

Vary these axes deliberately:

1. Time
   - multiple UTC start times
   - different days / RINEX nav files
   - avoid training only on one satellite geometry

2. Location
   - Korea/Seoul, Austin/TEXBAT-like, London, Canberra, Nairobi, Brasilia, Washington, etc.
   - include lat/lon/altitude spread

3. Motion
   - static
   - straight line at different speeds
   - circle/turning
   - mild sweep/stop-go

4. Receiver/channel conditions
   - C/N0/noise variation where available
   - channel count changes
   - tracking-validity and PRN visibility changes

5. Sampling policy
   - keep 1.0 s window / 0.5 s stride for now
   - later compare 0.5 s and 2.0 s windows as ablation

## Storage policy

At 2.6 MS/s, s8 interleaved IQ is about 5.2 MB/s:

- 5 min: ~1.45 GiB raw IQ
- 10 min: ~2.91 GiB raw IQ
- 30 min: ~8.72 GiB raw IQ
- 60 min: ~17.43 GiB raw IQ

Therefore do not keep all raw IQ by default. Use this lifecycle:

1. Generate IQ into `artifacts/rf_runs/<run_id>/`.
2. Run GNSS-SDR receiver into `artifacts/receiver_runs/<run_id>/`.
3. Export tracking feature dataset.
4. Export node/graph CSV dataset.
5. Verify row counts, PRN counts, feature finiteness, and a short training smoke test.
6. Delete raw IQ and GNSS-SDR raw channel dumps unless debugging is needed.
7. Keep only:
   - final node CSV
   - final graph CSV
   - dataset manifest
   - run metadata/provenance table
   - trained model artifacts

## Dataset structure

Preferred next dataset name:

`artifacts/model_datasets/normal_multi_prn_morphology_dynamics_v3_large/`

Required files:

- `normal_prn_node_windows.csv`
- `normal_receiver_graph_windows.csv`
- `manifest.json`
- `run_index.csv` with run_id, location, utc, duration, motion_type, split

## Splitting rule

Never split randomly by window row. Use full-run splits:

- train: many runs/locations/times
- validation: held-out runs and preferably held-out times
- test: held-out locations or days

This protects against learning one continuous run or one PRN constellation pattern.

## Model implications

The model should continue to enforce:

- no PRN ID as input
- shared PRN node encoder
- permutation-invariant PRN set aggregation
- conditional relation gate
- one final anomaly score

Use PRN-level error only for explanation/localization, not as separate final detectors.
