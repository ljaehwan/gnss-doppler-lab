# Simulation-v4 Normal Calibration v4 Result

Date fixed: 2026-08-25

## Decision

**CONDITIONAL: the selected 30-second normal candidate is qualified only for
independent-seed normal calibration, not for scale generation or detector
training.**

The selected candidate is `fs25-cn060-bw8p0`:

- 25 MHz signed 8-bit complex IQ;
- 8 MHz fifth-order front-end cutoff;
- target composite C/N0 of 60 dB-Hz;
- the patched Method-A receiver with nine measured taps at 0.125-chip spacing;
- the inner measured E/P/L taps exported into the same 11-feature contract used
  for the TEXBAT clean references.

The frozen configuration is
`configs/experiments/simulation_v4_normal_calibration_v4.json`, SHA-256
`c59c2de9dc5e86cadb921ba320b121009c60d97fbb18ae69717ec8a072f0e201`.
The runner SHA-256 is
`128a360617ba34861c169d12c3ed5029a694d28f0fdead5e0eff53ee0edcc7b8`.

No TEXBAT spoofing recording (`ds1`-`ds8`) was accessed, no spoofing IQ was
generated in this sweep, and no detector was trained. TEXBAT `cleanStatic` and
`cleanDynamic` are development calibration references and are no longer
untouched final-validation recordings.

## Correction to the v1 audit

The v1 audit correctly concluded that the original simulation distribution must
not be scaled, but one receiver-alignment statement in that report was wrong.
The simulation pilot had been processed by stock `/usr/bin/gnss-sdr`. Its
configuration contained only the locally patched `tap_spacing_chips` key,
which stock GNSS-SDR ignored. TEXBAT was processed by the patched Method-A
receiver and did use 0.125-chip spacing.

The receiver renderer now writes both standard GNSS-SDR keys,
`early_late_space_chips` and `early_late_space_narrow_chips`, in addition to
the patched tap keys. All calibration results in this document were reprocessed
after that correction. The Method-A result also uses the same patched receiver
binary as TEXBAT and exports only its measured inner E/P/L taps.

Therefore the original v1 AUC range of 0.9775-0.9906 remains evidence that the
old pipeline was unusable, but it is not a clean estimate of the residual
physical domain gap after receiver alignment.

## What made the original simulation different

The controlled sweeps isolated four material differences:

1. The original simulation used 2.6 MHz IQ while TEXBAT uses 25 MHz.
2. The stock receiver silently ignored the intended 0.125-chip spacing.
3. The unfiltered simulator correlation peak was too sharp relative to the
   hardware recording.
4. The original C/N0 and lock distribution produced excessive code-error and
   correlator-window variability.

After correcting the standard spacing keys, the 25 MHz, C/N0 57, unfiltered
stock-receiver candidate reached combined AUC 0.8345 but still failed the
dynamic target. Reprocessing the unfiltered candidate with Method-A produced
combined AUC 0.8561. An 8 MHz front-end cutoff reduced the C/N0 57 Method-A
candidate to AUC 0.7843, but its cleanStatic median KS remained 0.3775. Raising
only the target C/N0 to 60 dB-Hz reduced tracking variability enough to pass the
static and pooled gates. Raising it again to 63 dB-Hz made the dynamic median KS
worse (0.4379), so the selected value is not simply the cleanest signal.

## Frozen gate result

The preregistered gate is unchanged from the v1 audit:

| Level | Grouped domain AUC | Median KS | Median robust shift |
|---|---:|---:|---:|
| pass | <= 0.70 | <= 0.20 | <= 0.75 |
| conditional | <= 0.85 | <= 0.35 | <= 1.50 |
| stop | any conditional limit exceeded | any conditional limit exceeded | any conditional limit exceeded |

The selected candidate produced 630 one-second feature windows.

| TEXBAT clean target | OOF domain AUC | Mean fold AUC | Median KS | Median robust shift | Gate |
|---|---:|---:|---:|---:|---|
| cleanStatic | 0.5228 | 0.6541 | 0.1940 | 0.3299 | pass |
| cleanDynamic | 0.8374 | 0.8659 | 0.2291 | 0.3574 | conditional |
| combined | 0.6391 | 0.7157 | 0.1822 | 0.1445 | pass |

The selected receiver tracked 11 PRNs over 144,971 epochs. Median C/N0 was
46.93 dB-Hz. Its fraction of epochs with carrier lock above 0.5 was 0.8819,
which is above the calibration target range of 0.70-0.75. Receiver-state range
distance is included in the secondary fidelity loss, but is not a hard
distribution gate. This remaining lock mismatch is one reason the result is not
permission to scale.

The strongest residual cleanDynamic differences are consistent with the current
candidate still being a static source: `sharp_narrow_mean`,
`prompt_mag_cv`, code-error statistics, and Doppler variation. A real dynamic
normal trajectory is still required; this candidate must not be described as a
physical replacement for cleanDynamic.

## Selected artifacts

| Artifact | SHA-256 |
|---|---|
| IQ, 1,495,000,000 bytes | `a844cea7d06141e0557935d6293141a94674fdb32d46e9baa3a6f99c8f48ccb7` |
| RF manifest | `82dd7bf27a911ce943d7791a0123c2a8ce8b5e7460c563c2c60049161e073d7e` |
| receiver manifest | `94bd001db6b95428b0278246ec437c80e2748d911165664af93160359d62800f` |
| E/P/L feature CSV, 630 rows | `053834cd622245b9e3e12128952a25c1eaa91b2fcf51a64e38cba9d4725e286e` |
| full ignored summary | `d49b33fe2cbb3c8fae3e63cdac8b3b7aa5d10fcf3631c7fc5f20a1d927c7edf0` |
| Method-A executable | `4c556bcd5dcb227f43f72aa7da2bd6381aee559ac31953ae4f02ad49ee511443` |

The verified SSD bundle is:

```text
/home/ubuntu/ssd_data/gnss-early-detection/artifacts/
simulation-v4-normal-calibration-v4-selected-fs25-cn060-bw8p0-methoda9/
```

The bundle contains the selected RF run, Method-A receiver dump, feature CSV,
full comparison tables, frozen config, runner, and receiver-rendering source.
The source and SSD hashes were checked after copying and matched for the IQ,
receiver manifest, feature CSV, summary, and config.

Ignored detailed outputs remain under
`artifacts/simulation_v4_normal_calibration_v4/`. The tracked compact result is
`docs/results/simulation_v4_normal_calibration_v4_summary.json`.

## Required next experiment

1. Generate independent normal runs across new seeds, UTC epochs, and locations
   with the selected 25 MHz / 8 MHz front end and a preregistered C/N0 range
   around the selected operating point.
2. Add genuine dynamic normal trajectories using the existing straight,
   circle, and parallel-sweep generator. Do not fit the exact TEXBAT route.
3. Compare static simulation only to cleanStatic and dynamic simulation only to
   cleanDynamic, while retaining the pooled gate.
4. Add an explicit receiver-lock envelope gate or justify its replacement before
   scaling.
5. Freeze the simulation contract and normal detector before opening any TEXBAT
   spoofing scenario.

This is a normal-domain fidelity result, not spoof-detection accuracy and not
evidence by itself that a WCL paper claim is established.

## Reproduction

```bash
.venv/bin/python scripts/run_simulation_v4_normal_calibration.py \
  --config configs/experiments/simulation_v4_normal_calibration_v4.json \
  --resume \
  --receiver-variant methoda9 \
  --receiver-executable .tools/gnss-sdr-method-a-9tap \
  --receiver-tap-count 9 \
  --feature-tap-count 3
```
