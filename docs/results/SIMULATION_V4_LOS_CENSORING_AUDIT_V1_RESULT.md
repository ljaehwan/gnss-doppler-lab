# Simulation-v4 LOS and Tracking-Censoring Audit v1

Date fixed: 2026-08-26

## Decision

**PROMISING TRAIN-GENERATED CANDIDATE, NOT CONFIRMATORY EVIDENCE.** The narrow
mechanism worth carrying forward is not merely that spoofing distorts a
correlation peak. It is that carry-off-induced deformation also removes tracking
channels, so a detector that requires many simultaneous PRNs preferentially
deletes its most abnormal observations.

On the six train pairs, a four-PRN event rule detected 5/6 pairs with 9.875 s
median availability delay. Keeping the same width statistic, q99 paired-normal
diagnostic, and three-window persistence but allowing the statistic to remain
available while one or more PRNs survived detected 6/6 with 4.1875 s median
delay. Neither policy produced a three-window persistent warning on its six
paired normal counterparts. This one-PRN policy was generated after inspecting
train and must be frozen and tested once on validation pairs 007-009.

The result is not yet a WCL result. Validation, independent normal/multipath
stress, TEXBAT attack evaluation, and a broader novelty review remain required.

## Narrow physical question

For two equal-shape normalized correlation profiles with authentic weight 1,
counterfeit voltage ratio `rho`, and code-delay separation `delta`, the
within/between-mixture variance identity is

```text
Delta V = rho / (1 + rho)^2 * delta^2.
```

For satellite `j`, the simulation carry-off displacement gives the first-order
LOS approximation

```text
delta_j(t) = u_j^T Delta r(t) / L_chip,
Delta V_j(t) = rho(t) / (1 + rho(t))^2 * delta_j(t)^2.
```

Here `u_j` is the ENU line-of-sight unit vector parsed from the preserved
gps-sdr-sim startup log, `Delta r(t)` is the smoothstep counterfeit trajectory
offset, and `L_chip = c / 1.023 MHz`. The observed quantity is the baseline
excess of the normalized nine-tap correlation-profile variance at offsets
`[-0.5, ..., +0.5]` chips.

This identity is exact for an untruncated linear mixture of equal profiles. It
is only a mechanistic proxy at the receiver output because the real observable
uses magnitude correlators, finite tap aperture, noise, carrier phase, DLL
recentering, and only channels that remain tracked.

## Why the first hypothesis was retained as a negative result

The preregistered v1 envelope audit used displacement magnitude instead of a
satellite LOS projection. Its pooled event-level Spearman correlation was
0.6038, but the frozen requirement that every train pair reach rho >= 0.5
failed; pair 004 was -0.4947. Its status remains
`not_supported_on_train`. The support rule was not weakened after seeing the
result.

The post-hoc LOS audit moved pair 004 to a positive PRN-window correlation of
0.2988. Across 1,642 spoof-transition PRN windows, the pooled association was
rho = 0.4423 (two-sided p = 1.29e-79), versus 0.4295 for the same-row ENU
magnitude envelope. All six pair-level LOS associations were positive, but the
small gain and moderate pooled value do not support an exact predictive-law
claim.

| Pair | Transition PRN windows | LOS-width rho | Four-PRN delay (s) | Censoring-aware delay (s) |
|---|---:|---:|---:|---:|
| `001` | 233 | 0.2632 | 10.250 | 8.625 |
| `002` | 289 | 0.6314 | 9.875 | 4.000 |
| `003` | 288 | 0.3319 | 8.250 | 8.250 |
| `004` | 270 | 0.2988 | not detected | 2.375 |
| `005` | 271 | 0.5540 | 10.625 | 4.375 |
| `006` | 291 | 0.5606 | 8.875 | 2.250 |

## Tracking attrition is the main finding

The baseline-normalized number of available PRNs had median 1.0 on reference
events and 0.4545 on attack events. Treating support deficit alone as a
descriptive attack score gave AUC 0.8345. More importantly, 32.74% of attack
event times were removed by the minimum-four-PRN eligibility rule.

This is a missing-not-at-random mechanism: the physical deformation being
measured also changes whether the measurement exists. Requiring a fixed number
of simultaneous tracked satellites can therefore bias a peak-shape detector
toward mild deformation and late reacquisition.

The candidate improvement does not add a learned model. For each surviving PRN
it computes a robust pre-onset width score

```text
z_W,j(t) = [W_j(t) - median_baseline(W_j)] /
           max(1.4826 * MAD_baseline(W_j), 1e-6),
```

then takes the median over currently available PRNs and requires three
contiguous 0.125 s scores above the paired-normal q99 diagnostic threshold. The
comparison changes only the event eligibility floor from four PRNs to one.

| Policy | Detected pairs | Median delay | Maximum delay | Persistent paired-normal warnings |
|---|---:|---:|---:|---:|
| minimum 4 PRNs | 5/6 | 9.875 s | 10.625 s | 0/6 |
| minimum 1 surviving PRN | 6/6 | 4.1875 s | 8.625 s | 0/6 |

Among the five pairs detected by both policies, the median paired delay
reduction was 5.875 s. This is a train-only diagnostic, not an estimate of field
false-alarm probability. In four candidate detections at least one persistence
window was supported by only one PRN, so independent validation is essential.

## Relation to prior work

Correlation distortion, multiple peaks, SQM symmetry metrics, and aggregation
across satellites are established ideas. Turner et al. decompose distorted
correlation functions and apply a likelihood-ratio test; Pini et al. apply
statistical tests to Early/Late correlator outputs; A-CoRLiAn aggregates
multi-tap correlation-residual decisions across tracked satellites.

The candidate contribution here is narrower: quantify **tracking attrition as
informative censoring of the peak-shape measurement itself**, then compare a
fixed multi-PRN eligibility rule with an attrition-tolerant persistent width
statistic. A targeted preliminary search did not establish that this exact
censoring analysis is already standard, but that is not a novelty proof.

Primary references:

- M. Turner et al., “Spoofing Detection by Distortion of the Correlation
  Function,” IEEE/ION PLANS 2020, DOI 10.1109/PLANS46316.2020.9110173.
- M. Pini et al., “Detection of Correlation Distortions Through Application of
  Statistical Methods,” ION GNSS+ 2013.
- “Advanced GNSS Spoofing Detection: Aggregated Correlation Residue Likelihood
  Analysis,” Remote Sensing, 2024, 16(15), 2868.

## Provenance and boundary

The v1 configuration SHA-256 is
`5e172a4672a027f3ee21337dc120fdc27f37606eb474d3604614ae0f0da18131`.
The v1 full summary SHA-256 is
`8fc0126826fa633e23238e8c3cefcc2e572999ddf173583f8500d7766356ab84`.

The LOS-censoring configuration SHA-256 is
`f4239274a4bd989625d21385285b49c614d91ffb70703a92a447d5ee35da400b`.
The runner SHA-256 is
`9f0aec82e5dbb4fea6631aa1f29684dd98475d56911e2d24b1041ca441887826`.
The physical module SHA-256 is
`2afd34230448c1a827292f8850943574f90eecf1a1fef50286de979bb8badcfc`.
The full summary SHA-256 is
`8df9e1068ce3a0fc36214b699840e09861f3a84a966499c52c17c1abe6ec2a9b`.

Hash-verified SSD bundles are stored at:

```text
/home/ubuntu/ssd_data/gnss-early-detection/artifacts/simulation-v4-peak-mixture-law-v1/
/home/ubuntu/ssd_data/gnss-early-detection/artifacts/simulation-v4-los-censoring-audit-v1/
```

Only `pv1-pair-001` through `pv1-pair-006` were read. Validation pairs
`007`-`009`, test pairs `010`-`012`, and all TEXBAT recordings were not
accessed. The gps-sdr-sim logs pin navigation SHA-256
`7db04513dd2d0e13c0ee20cb4eaa8f71e5a28ab58b65c9b5b789f86eeab436cd`.

## Limitations and next gate

- This mechanism and the one-PRN policy were generated after viewing train.
- The q99 thresholds use complete paired-normal train counterparts and are not
  deployable thresholds.
- Initial LOS is held fixed over each 30 s run.
- The exact variance law does not include coherent carrier phase, finite tap
  truncation, tracking-loop recentering, or channel-selection dynamics.
- Correlation distortion and channel loss can also arise from multipath,
  blockage, interference, and receiver transients.

The next gate is a single frozen run on validation pairs 007-009. No formula,
tap spacing, window, minimum-PRN policy, persistence, or diagnostic quantile may
change after validation is opened. Test and TEXBAT attack recordings stay
locked.

## Reproduction

With the pinned v1 source artifact present:

```bash
.venv/bin/python scripts/audit_simulation_v4_los_censoring.py \
  --config configs/experiments/simulation_v4_los_censoring_audit_v1.json \
  --overwrite
```

The full artifacts are under
`artifacts/simulation_v4_peak_mixture_law_v1` and
`artifacts/simulation_v4_los_censoring_audit_v1`. The tracked compact result is
`docs/results/simulation_v4_los_censoring_audit_v1_summary.json`.
