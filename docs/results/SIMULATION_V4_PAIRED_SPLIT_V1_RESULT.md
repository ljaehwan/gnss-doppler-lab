# Simulation-v4 Paired Train/Validation/Test Split v1

Date fixed: 2026-08-25

## Decision

**FROZEN BEFORE GENERATION: twelve new paired base runs are assigned as six
train, three validation, and three locked test pairs.**

The split unit is `paired_group_id`, not a PRN, feature window, or individual
normal/spoof scenario. Every pair contains one steady-normal scenario and one
carry-off-spoof scenario with a byte-identical pre-onset prefix contract. Both
members, all tracked PRNs, and every time window inherit the pair's partition.

No paired IQ or feature data was generated in this step, no detector was
trained, and no test feature was accessed. This document freezes the assignment
before those outcomes exist.

## Why the previous five runs are excluded

The independent-normal runs `iv-s-denver-a`, `iv-s-seoul-b`,
`iv-d-tokyo-straight`, `iv-d-london-circle`, and `iv-d-sydney-sweep` were
already inspected against TEXBAT `cleanStatic`/`cleanDynamic`. They remain
calibration evidence and are explicitly excluded from all three model
partitions. Calling any of them an untouched test run would be data reuse.

The exclusion is pinned to
`docs/results/simulation_v4_normal_independent_validation_v1_summary.json`,
SHA-256
`0f038e31179ab24f94f50ddd38c06d76b985ccef851ab5568d7226c459824e6a`.

## Frozen assignment

| Partition | Pair IDs | Normal scenarios | Spoof scenarios | Motion coverage | Access |
|---|---|---:|---:|---|---|
| train | `001`-`006` | 6 | 6 | 2 static, 2 straight, 1 circle, 1 sweep | planned |
| validation | `007`-`009` | 3 | 3 | 1 static, 1 straight, 1 circle | planned |
| test | `010`-`012` | 3 | 3 | 1 static, 1 straight, 1 sweep | locked |

The full IDs are `pv1-pair-NNN`. Each pair uses a unique location, UTC epoch,
receiver seed, and canonical base-run fingerprint.

| Pair | Split | Location | Motion | UTC | Seed | C/N0 target |
|---|---|---|---|---|---:|---:|
| 001 | train | Paris | static | 01:00 | 20260901 | 59.5 |
| 002 | train | Singapore | static | 03:00 | 20260902 | 60.0 |
| 003 | train | Ottawa | straight | 05:00 | 20260903 | 60.5 |
| 004 | train | Pretoria | straight | 07:00 | 20260904 | 59.5 |
| 005 | train | Berlin | circle | 09:00 | 20260905 | 60.0 |
| 006 | train | Santiago | parallel sweep | 11:00 | 20260906 | 60.5 |
| 007 | validation | Madrid | static | 13:00 | 20260907 | 59.5 |
| 008 | validation | Bangkok | straight | 15:00 | 20260908 | 60.0 |
| 009 | validation | Nairobi | circle | 17:00 | 20260909 | 60.5 |
| 010 | test | Wellington | static | 19:00 | 20260910 | 60.5 |
| 011 | test | Mexico City | straight | 21:00 | 20260911 | 60.0 |
| 012 | test | Oslo | parallel sweep | 23:00 | 20260912 | 59.5 |

The RF/receiver profile stays at the selected 25 MHz, 8 MHz fifth-order front
end, nine 0.125-chip Method-A taps, and inner three-tap feature extraction. C/N0
is restricted to the frozen 59.5/60.0/60.5 dB-Hz grid.

## Leakage firewall

The planner rejects a campaign if any of the following occurs:

1. a `paired_group_id`, receiver seed, UTC, or base-run fingerprint is reused;
2. normal and spoof members of a pair land in different partitions;
3. a `run_id` or IQ `source_fingerprint` appears in more than one pair;
4. any PRN/window is randomly reassigned independently of its pair;
5. a prior calibration run is added to train, validation, or test;
6. the selected RF/receiver profile or required motion coverage changes.

The generated audit reports empty intersections for train-vs-validation,
train-vs-test, and validation-vs-test.

## Partition use

- Fit the model and every normalization/preprocessing statistic on train only.
- Select architecture, early stopping, and the decision threshold on validation.
- Do not fit, normalize, calibrate, or adapt anything on test.
- Release `test.csv` only after SHA-256 pinning the model, preprocessing, decision
  threshold, and this split manifest.
- After test release, report the result once; refitting or threshold adaptation is
  forbidden for that reported experiment.

The splitter enforces the test release rule. Without a valid
`gnss-doppler-lab.simulation-v4-model-freeze` manifest, it writes train and
validation CSVs but leaves test locked.

## Artifacts

The frozen configuration is
`configs/experiments/simulation_v4_paired_split_v1.json`, SHA-256
`124bb766dcc0c52e248627e20499b55149b5477cabd4c88fa960585368f62b3e`.
The planner is `scripts/plan_simulation_v4_paired_split.py`, SHA-256
`92f990efa49b8a6fd987edbeb418f06e6e2fd3c4cd42753a159ce9fff8415e72`.

The ignored full plan is under `artifacts/simulation_v4_paired_split_v1/`:

| Artifact | SHA-256 |
|---|---|
| split manifest | `dd20c4917e3c406e716cb00f251d04f7aa900544151da03c37fa8ba29528a18e` |
| pair catalog, 12 rows | `1ab398e312e9bdb72d450e6233f9820e573a8b2c537d3daaf23b766b95faf2e2` |
| scenario catalog, 24 rows | `10c1a7ebf9ea348958743b1ead5d40fd257824f4ac4529e04cda06ca1ef61289` |
| train group list | `2d5e58b4547125d3b0a8a8a2808ec1f635ff4d343c7c9fc3fd10b146f70a2b0d` |
| validation group list | `10399d05a6adf83a72c51a84607844356f36c592e25c60e83b62157b993bd923` |
| locked test group list | `89f0eddd48a90a4a0c21f12a48c791353c8a1bd011cfa653e074d4e624dd9f2a` |

The tracked compact result is
`docs/results/simulation_v4_paired_split_v1_summary.json`. The SSD bundle is:

```text
/home/ubuntu/ssd_data/gnss-early-detection/artifacts/
simulation-v4-paired-split-v1/
```

## Limitation and next action

Three validation and three test pairs are suitable only for the exploratory
paired-spoofing pilot. They are not enough for a final WCL uncertainty or broad
generalization claim. Window counts may also become imbalanced after GNSS-SDR
tracking even though scenario counts are balanced.

The next action is to generate and receive train pairs `001`-`006` first, then
validation pairs `007`-`009`. Test pairs `010`-`012` should remain sealed until
the detector and threshold are frozen.

## Reproduction

```bash
.venv/bin/python scripts/plan_simulation_v4_paired_split.py \
  --config configs/experiments/simulation_v4_paired_split_v1.json \
  --output-root artifacts/simulation_v4_paired_split_v1
```
