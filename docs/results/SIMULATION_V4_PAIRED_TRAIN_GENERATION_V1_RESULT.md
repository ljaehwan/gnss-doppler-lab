# Simulation-v4 Paired Train Generation v1 Result

Date fixed: 2026-08-26

## Decision

**CONDITIONAL: all six preregistered train pairs were generated and received
without opening validation or test, and the resulting train-only feature set is
qualified for exploratory model fitting. This is not a detector-accuracy result
and does not establish a WCL claim.**

The completed stage contains:

- six paired base runs, each with one steady-normal and one carry-off-spoof
  member;
- 25 MHz signed 8-bit complex IQ with the frozen 8 MHz fifth-order front end;
- the patched Method-A receiver with nine measured taps at 0.125-chip spacing;
- the current 11-feature, inner E/P/L, 1-second window contract;
- 7,499 labeled train windows across 12 receiver runs.

No model, normalization transform, architecture choice, early-stopping rule, or
decision threshold was fitted in this stage.

## Frozen provenance

The generation configuration is
`configs/experiments/simulation_v4_paired_train_generation_v1.json`, SHA-256
`b44781e7e0f03c60ca18ecccdd8cbbd7f02a98c8270244658009419ad9772f78`.
The runner is
`scripts/run_simulation_v4_paired_train_generation.py`, SHA-256
`c15c23f96e5d6137e265ff0048f2cd89a51c9123635ce0f7331c0d4fe07375d0`.

The runner verified the frozen split config SHA-256
`124bb766dcc0c52e248627e20499b55149b5477cabd4c88fa960585368f62b3e`
and canonical split-manifest SHA-256
`dd20c4917e3c406e716cb00f251d04f7aa900544151da03c37fa8ba29528a18e`.
The test release status remained `locked`.

## Paired-generation contract

Within each pair, normal and spoof members share the authentic component,
location, UTC, motion, receiver seed, front-end noise, receiver impairments,
and fixed authentic-derived gain. The spoof member alone adds the counterfeit
component through its preregistered power envelope and false trajectory.

The spoof envelope is exactly zero before 10 seconds. At 25 MHz this makes the
first 250,000,000 complex samples, or 500,000,000 IQ8 bytes, an exact paired
prefix. All six pairs passed direct byte comparison and prefix SHA-256 checks.

Authentic and counterfeit simulator components were retained. Each source and
final member IQ is exactly 1,495,000,000 bytes. The complete local tree is about
35 GB after counting the 12 receiver-IQ aliases as hard links rather than
duplicate data.

## Train artifacts

| Pair | Motion | Normal rows | Spoof-member rows | Prefix identical | Normal lock fraction | Normal median C/N0 |
|---|---|---:|---:|---|---:|---:|
| `001` | static | 632 | 627 | yes | 0.8739 | 46.57 |
| `002` | static | 629 | 623 | yes | 0.7761 | 46.22 |
| `003` | straight | 637 | 637 | yes | 0.8930 | 47.30 |
| `004` | straight | 638 | 618 | yes | 0.8232 | 46.96 |
| `005` | circle | 632 | 631 | yes | 0.8613 | 46.19 |
| `006` | parallel sweep | 596 | 599 | yes | 0.8632 | 48.90 |

The combined dataset is
`artifacts/simulation_v4_paired_train_generation_v1/train_tracking_features_labeled.csv`,
SHA-256
`ce353d0cb902f15af52cffc6b45e04f3515e4023752a5c50815eb93db04850bd`.

Window midpoint determines the event label:

| Event state | Rows | Meaning |
|---|---:|---|
| steady normal | 3,764 | all windows from the normal members |
| pre-event normal | 1,219 | spoof-member windows before onset |
| carry-off transition | 1,312 | onset through false-position transition |
| carry-off final | 1,204 | post-transition spoofing |

Thus the train CSV has 4,983 normal-labeled and 2,516 spoofing-labeled windows.
Scenario counts are balanced, but event-window labels are not; model fitting
must account for this without resampling across pair boundaries.

## Leakage and partition audit

The completed dataset validator reported:

- six paired groups and 12 unique receiver run IDs;
- 12 unique tracking-source fingerprints;
- 7,499 train rows, zero validation rows, and zero test rows;
- no source-fingerprint overlap;
- normal and spoof members present in every pair;
- all PRNs and windows inheriting their pair's train assignment.

Only `pv1-pair-001` through `pv1-pair-006` were generated. Validation pairs
`007`-`009` and test pairs `010`-`012` were not accessed. TEXBAT
`ds1`-`ds8` were not accessed. TEXBAT `cleanStatic` and `cleanDynamic`
were read only as the already-declared normal-fidelity references, not as
training examples.

## Normal-fidelity gate

The normal members were compared with the matched TEXBAT clean references using
the preregistered grouped domain-classifier and distribution gates.

| Comparison | Sim rows | Real rows | Pooled domain AUC | Mean fold AUC | Median KS | Median robust shift | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| static vs cleanStatic | 1,261 | 1,837 | 0.6893 | 0.6985 | 0.2115 | 0.3058 | conditional |
| dynamic vs cleanDynamic | 2,503 | 1,806 | 0.8261 | 0.8427 | 0.1657 | 0.3155 | conditional |
| pooled vs cleanCombined | 3,764 | 3,643 | 0.7435 | 0.7492 | 0.1089 | 0.1439 | conditional |

Static is conditional because median KS is slightly above the 0.20 pass limit.
Dynamic and pooled comparisons are conditional because domain AUC is above the
0.70 pass limit. None exceeds a conditional stop boundary.

The normal receiver-state gate is also conditional. Every pair stayed inside
the conditional envelopes. Pair 002 passed both receiver-state metrics; the
other pairs were conditional primarily because their carrier-lock fraction was
above the narrow 0.68-0.80 pass envelope, not because tracking was lost. Pair
006 also had median C/N0 48.90 dB-Hz, above the 48 dB-Hz pass limit but below the
50 dB-Hz stop limit.

## Integrity and storage

The full source summary is
`artifacts/simulation_v4_paired_train_generation_v1/summary.json`, SHA-256
`bb7023ea7a8a57ae3197214e85e879524aace84a1c5d1306321274127a2449d7`.
The tracked compact result is
`docs/results/simulation_v4_paired_train_generation_v1_summary.json`.

A post-run readback checked 60 artifacts. All SHA-256 values matched, including
12 authentic/counterfeit components, 12 final member IQ files, RF manifests,
receiver manifests, and feature CSVs. All 12 local receiver-IQ aliases resolve
to their canonical final IQ hard links.

The SSD bundle is:

```text
/home/ubuntu/ssd_data/gnss-early-detection/artifacts/
simulation-v4-paired-train-generation-v1/
```

## Limitation and next action

This is a small, simulation-only train pilot. Three validation pairs and three
test pairs remain too small for a broad uncertainty or final WCL
generalization claim. The remaining dynamic domain signature must be reported,
not hidden by model complexity.

The next controlled step is to generate validation pairs `007`-`009` with
the same immutable contracts. Model and preprocessing fitting must use train
only; architecture, early stopping, and threshold selection may use validation.
Test pairs `010`-`012` must stay locked until model, preprocessing, and
threshold artifacts are SHA-256 frozen.

## Reproduction

```bash
.venv/bin/python scripts/run_simulation_v4_paired_train_generation.py \
  --config configs/experiments/simulation_v4_paired_train_generation_v1.json \
  --resume
```
