# Simulation-v4 Paired RF Pilot Result

Date fixed: 2026-08-25

## Status

The paired RF/IQ generation and GNSS-SDR receiver pipeline completed successfully.
This is a **pipeline-qualification pilot**, not a trained detector result and not
external validation evidence. TEXBAT scenario recordings remain reserved for the
final frozen evaluation.

The tracked frozen input is:

```text
configs/experiments/simulation_v4_pilot.yaml
SHA-256 74a94f5d6bdba8b735fd6f48b4cf89073a3dc6909c225b88e7fc503fe86061ec
```

The local generated bundle is under `artifacts/simulation_v4_pilot/` and is
ignored by Git. It occupies approximately 961 MB including three IQ recordings,
GNSS-SDR raw tracking dumps, normalized tracking CSVs, and features.

## Paired-generation contract

All three scenarios use one authentic GPS-SDR-SIM source, one receiver
impairment realization, and the same seeded AWGN samples:

- steady normal
- normal all-signal outage from 18.0 to 30.0 seconds, followed by a 1.5-second restore ramp
- all-visible carry-off spoofing from 18.0 seconds, reaching a 100 m east offset over 16 seconds and a +3 dB final power advantage

Receiver gain is calibrated once using only the authentic source and then frozen
for all scenarios. No scenario-dependent AGC or future-event power is used. The
composer superposes simulator IQ8 sources in float32 and performs one final IQ8
quantization.

The two event scenarios are byte-identical to steady normal for all 46,800,000
complex samples before the 18-second onset. Both prefix files have SHA-256:

```text
cd193efe40238f459a29af0297bf8030a4268cad1b06425e6e11aa9f7a9012e4
```

This closes the earlier leakage defect where final spoof power attenuated the
authentic pre-onset prefix.

## RF output audit

The pinned GPS-SDR-SIM contract produced 129,740,000 complex samples per run,
or 49.9 seconds at 2.6 MHz. The 0.1-second difference from the requested 50
seconds is the simulator's documented warm-up/block behavior.

| Scenario | IQ SHA-256 | RMS | Clipping | Tracked PRNs |
|---|---|---:|---:|---:|
| steady normal | `26de6cea61a186e81f32854b201f7d148c8b5e348a93359c9043715acbb453fc` | 22.004 | 0% | 14 |
| normal outage/recovery | `4078a30bc23c1d6f9c55795ac0d27ed9397c1c105d722bc3099f98c2ffc13321` | 21.743 | 0% | 10 |
| carry-off spoof | `d50e72a2db2caeb2305c76abd368aa8aa61274dc7c00cd7504aa84672ff204d1` | 22.947 | 0% | 12 |

The realized final counterfeit displacement was 100.0000000002 m east, with
numerically negligible north/up error.

## Receiver loss and reacquisition result

GNSS-SDR did continue writing tracking epochs during the outage. Those rows are
not evidence of valid tracking: the carrier-lock statistic shows the intended
loss and recovery clearly.

| Window | Carrier-lock median | Fraction `lock > 0.5` | Median C/N0 |
|---|---:|---:|---:|
| pre-event, 10-17 s | 0.9120 | 68.47% | 41.06 dB-Hz |
| outage interior, 19-29 s | 0.0265 | 0.00% | 29.00 dB-Hz |
| post-recovery, 32.5-40.5 s | 0.8789 | 60.88% | 43.07 dB-Hz |

The receiver produced six post-restore tracking segments. Ten PRNs were present
before the outage, seven after restoration, and seven were common to both
periods. All frozen recovery acceptance checks passed. This is why the pipeline
uses carrier-lock evidence rather than the incorrect requirement that an outage
must contain zero MAT rows.

The carry-off run retained at least four PRNs both before onset and after the
transition. The source trajectory and tracking behavior are validated, but the
50-second carry-off run did not produce a false PVT solution; it must not be
claimed as position-takeover evidence.

## Feature dataset

The pipeline exported 2,503 labeled three-tap E/P/L tracking-window rows:

- steady normal: 867
- paired pre-event normal: 629
- normal outage: 102
- post-recovery normal: 238
- carry-off transition: 328
- carry-off final: 339

Dataset SHA-256:

```text
bdf96697bc40622db1db8aaf4fdf91f9ac9a1b7c25ce406a3df3acd268067cb2
```

The outage and recovery rows remain class `normal` with a separate
`event_state`; only post-onset carry-off rows are class `spoofing`. The dataset
role is explicitly `simulation_only_pilot`. Every row also carries
`paired_group_id=texbat-wrw-simulation-v4-pilot`; all counterfactual siblings
that share authentic/noise samples must stay in the same data split.

## Scientific boundary and next use

This result establishes that the integrated generator can create causal paired
normal/spoof prefixes and genuine receiver loss/reacquisition. It does **not**
solve the previously measured simulation-to-real domain gap, demonstrate model
performance, or justify a WCL detection claim by itself.

The next defensible experiment is to expand this contract across independent
locations, epochs, receiver seeds, SNRs, outage lengths, carry-off offsets, and
power ramps; split atomically by paired campaign group before model fitting so
no shared prefix/noise crosses partitions; freeze the detector;
and use untouched TEXBAT scenarios only for external evaluation.

Reproduction commands:

```bash
PYTHONPATH=src python3 scripts/run_simulation_v4_pipeline.py \
  --config configs/experiments/simulation_v4_pilot.yaml --generate-only

PYTHONPATH=src python3 scripts/run_simulation_v4_pipeline.py \
  --config configs/experiments/simulation_v4_pilot.yaml \
  --reuse-generated
```

A completed receiver stage can be re-audited without rerunning GNSS-SDR only
when both the generated campaign config hash and each RF manifest hash match:

```bash
PYTHONPATH=src python3 scripts/run_simulation_v4_pipeline.py \
  --config configs/experiments/simulation_v4_pilot.yaml \
  --reuse-generated --reuse-receiver
```
