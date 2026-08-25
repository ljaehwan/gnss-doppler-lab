# Simulation-v4 Independent Normal Validation v1 Result

Date fixed: 2026-08-25

## Decision

**CONDITIONAL: the selected simulation-v4 RF profile remains inside every
predeclared conditional domain and receiver-state limit on five independent
normal runs, but it is not yet qualified for scale generation or final detector
training.**

This stage fixed the calibration-v4 selection before generating any new run:

- 25 MHz signed 8-bit complex IQ;
- 8 MHz fifth-order front-end cutoff;
- target composite C/N0 varied only from 59.5 to 60.5 dB-Hz;
- the patched Method-A receiver with nine measured taps at 0.125-chip spacing;
- inner measured E/P/L taps exported into the same 11-feature contract used by
  the TEXBAT clean references.

The frozen configuration is
`configs/experiments/simulation_v4_normal_independent_validation_v1.json`,
SHA-256
`256977cfb14f0828d6cc98af83f459e7dee0407b41cd1188b0c37cb170593a78`.
The runner SHA-256 is
`6097970b0c562d4d02a85eec3bc1b76bdcadf6c925b2dc9b4e75f604ae2c200b`.

No TEXBAT spoofing recording (`ds1`-`ds8`) was accessed, no spoofing IQ was
generated, and no detector was trained. TEXBAT `cleanStatic` and
`cleanDynamic` were used as development calibration references and therefore
are not untouched final-validation recordings.

## Independent-run protocol

The pilot generated two static and three dynamic 30-second runs. None reused
the calibration-v4 Austin location, UTC, or receiver seed. The dynamic routes
use the existing generic WGS-84 trajectory generator; they do not reproduce a
TEXBAT route.

| Run | Domain/motion | UTC | C/N0 target | Seed | Feature rows |
|---|---|---|---:|---:|---:|
| `iv-s-denver-a` | static, Denver | 02:00 UTC | 59.5 | 20260831 | 633 |
| `iv-s-seoul-b` | static, Seoul | 04:00 UTC | 60.5 | 20260832 | 608 |
| `iv-d-tokyo-straight` | 10 m/s straight, 300 m | 06:00 UTC | 59.5 | 20260833 | 547 |
| `iv-d-london-circle` | 8 m/s circle, 40 m radius | 08:00 UTC | 60.0 | 20260834 | 635 |
| `iv-d-sydney-sweep` | 10 m/s parallel sweep | 10:00 UTC | 60.5 | 20260835 | 627 |

Each dynamic trajectory contains exactly 300 rows at 10 Hz. Each final IQ has
exactly 1,495,000,000 bytes. The five final IQ files total 7,475,000,000 bytes;
the complete local tree is about 15 GB because it also retains the authentic
components and receiver outputs.

## Frozen gate result

The domain gate is unchanged from calibration v4:

| Level | Grouped domain AUC | Median KS | Median robust shift |
|---|---:|---:|---:|
| pass | <= 0.70 | <= 0.20 | <= 0.75 |
| conditional | <= 0.85 | <= 0.35 | <= 1.50 |
| stop | any conditional limit exceeded | any conditional limit exceeded | any conditional limit exceeded |

Static simulation was compared only with `cleanStatic`, dynamic simulation only
with `cleanDynamic`, and the pooled result with the combined clean references.

| Matched comparison | Simulation rows | Real rows | OOF domain AUC | Mean fold AUC | Median KS | Median robust shift | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| static vs cleanStatic | 1,241 | 1,837 | 0.5706 | 0.6473 | 0.2430 | 0.5638 | conditional |
| dynamic vs cleanDynamic | 1,809 | 1,806 | 0.8314 | 0.8412 | 0.1812 | 0.2889 | conditional |
| pooled vs cleanCombined | 3,050 | 3,643 | 0.7599 | 0.7619 | 0.1073 | 0.2231 | conditional |

No comparison reached `stop`. The static comparison is conditional only because
median KS is above 0.20. The dynamic and pooled comparisons are conditional
because domain AUC is above 0.70. The dynamic result therefore shows a remaining
multivariate domain signature even though its median univariate distribution
metrics pass.

Per-run diagnostic AUCs were 0.5480 and 0.5138 for the two static runs, and
0.8351, 0.8200, and 0.8098 for the straight, circle, and sweep dynamic runs.
Seoul passed the complete per-run domain gate; the other four were conditional.
Per-run diagnostics are secondary to the matched aggregate gates.

## Receiver-state gate

The hard receiver-state envelopes were predeclared as follows:

| Metric | Pass | Conditional |
|---|---:|---:|
| fraction with carrier lock > 0.5 | 0.68-0.80 | 0.60-0.90 |
| median C/N0, dB-Hz | 44-48 | 42-50 |

All five runs tracked 11 PRNs and all median C/N0 values passed. Tokyo, London,
and Seoul passed the complete receiver-state gate. Denver's lock fraction was
0.8125 and Sydney's was 0.8609, so both were conditional but remained below the
0.90 stop boundary. This is an envelope mismatch against the calibration target,
not a receiver lock failure.

## What still differs

The largest residual static differences are `sharp_narrow_std` (KS 0.333),
`doppler_slope` (KS 0.280), and `code_err_abs_mean` (KS 0.263). The largest
dynamic differences are `sharp_narrow_mean` (KS 0.549), `prompt_mag_cv`
(KS 0.315), and `cn0_std` (KS 0.205).

The new motion models substantially improved the pooled distribution result, but
motion alone did not eliminate the dynamic multivariate domain signature. The
next physical refinement should therefore target correlator-shape and
amplitude/quality variability rather than add arbitrary classifier complexity.

## Artifacts and integrity

The full source summary is
`artifacts/simulation_v4_normal_independent_validation_v1/summary.json`,
SHA-256
`7724d7d266f7cdb5079284e940f3f87d0ce0895acaf662cabfb12af110b2dabb`.
The tracked compact result is
`docs/results/simulation_v4_normal_independent_validation_v1_summary.json`.

All five final IQ files were read back after generation and their SHA-256 values
matched their immutable RF manifests:

| Run | IQ SHA-256 |
|---|---|
| London circle | `b7db9bb40897bbaaaf988aad3a360004effab07b3cc5b490fa3a2b5f09fc1d39` |
| Sydney sweep | `c422e833621ba088461cafcb3572b9dc87aa3c898ac23712c062942d260fbf66` |
| Tokyo straight | `2087a6137a99bea3347e9f9236274a76a56af6772a35282a613d927eeb51d92f` |
| Denver static | `2b5a626fbbaa57c0ae29e282f2ff5c6beb694a6207e92b938794cd6b5557e615` |
| Seoul static | `39de651601a88d548baaa23451c1a53f977f5e41c6669cf12678a5c61bec2090` |

The SSD bundle is:

```text
/home/ubuntu/ssd_data/gnss-early-detection/artifacts/
simulation-v4-normal-independent-validation-v1-methoda9/
```

## Consequence for the next experiment

This result supports additional independent normal coverage and, at most, a
small exploratory paired-spoofing pilot. It does not support mass data
generation, a final detector claim, or opening TEXBAT spoofing recordings.

The next pilot should freeze these five normal runs, create paired normal/spoof
IQ with identical authentic component, impairment seed, location, time, and
motion, and vary only preregistered spoof onset/ramp parameters. Training,
validation, and test partitions must split by scenario seed/run rather than by
window. The detector should remain frozen before any TEXBAT `ds1`-`ds8` final
evaluation.

This is a normal-domain fidelity result, not spoof-detection accuracy and not by
itself evidence that a WCL claim has been established.

## Reproduction

```bash
.venv/bin/python scripts/run_simulation_v4_normal_independent_validation.py \
  --config configs/experiments/simulation_v4_normal_independent_validation_v1.json \
  --resume
```
