# Current Project State

Last updated: 2026-07-11

## Repository

- Project: `gnss-doppler-lab`
- Remote: `https://github.com/ljaehwan/gnss-doppler-lab.git`
- Branch: `main`
- Working directory on GNSS VM: `/home/ubuntu/projects/gnss-doppler-lab`

## Research direction

Build a GNSS-only UAV GPS spoofing-detection study with a reproducible RF-to-receiver chain:

```text
RINEX NAV + UTC + UAV trajectory
→ GPS L1 C/A IQ
→ GNSS-SDR acquisition/tracking/NAV/PVT
→ receiver observables and Doppler
→ statistical/ML/DL comparison
→ external-corpus generalization
```

Use IEEE literature first for UAV motion/methodology. The three normal dynamic trajectory families are representative test scenarios, not a claim that they exhaust all real UAV motion.

## Implemented and verified

### Static receiver

- Static `position.type: static` remains supported through GPS-SDR-SIM `-l`.
- A post-change Seoul static RF-generation smoke test succeeded.
- Static and dynamic modes are separate validated branches; dynamic support did not replace static support.

### Parameterized dynamic trajectories

The trajectory generator supports arbitrary controlled values rather than hard-coded speeds or distances:

- start latitude/longitude and WGS-84 ellipsoidal altitude;
- duration (upstream limit: 300 s);
- speed and heading;
- circle radius and optional integer laps;
- parallel-sweep leg length and lane spacing.

The exact simulator motion contract is 10 Hz, with `D * 10` rows at `0.0, 0.1, ..., D - 0.1` seconds.

Implemented families:

1. Straight-line flight
   - IEEE basis: Nelson et al., IEEE Transactions on Robotics, DOI `10.1109/TRO.2007.898976`
2. Constant-radius circular orbit
   - IEEE basis: Nelson et al., DOI `10.1109/TRO.2007.898976`
3. Parallel-sweep coverage
   - IEEE basis: Xu et al., IEEE ICRA, DOI `10.1109/ICRA.2011.5979707`

Generated trajectory CSV and JSON sidecars include effective distance/speed/laps, WGS-84 metadata, literature metadata, row/time bounds, and SHA-256 provenance. LLH and ECEF trajectory inputs are supported and validated.

### Corrected UTC/GPST semantics

A moving run exposed an exact 18-second time-scale error:

```text
5 m/s × 18 s ≈ 90 m observed trajectory phase error
```

Root cause: the pinned GPS-SDR-SIM `-t` path treats supplied calendar fields as GPST-like fields, while GNSS-SDR outputs UTC after applying broadcast leap seconds.

The fixed pipeline now:

1. parses `LEAP SECONDS` from the exact RINEX NAV header (no hard-coded 18 s);
2. defines `scenario.utc` as real UTC;
3. passes `UTC + GPS_MINUS_UTC` to GPS-SDR-SIM `-t`;
4. records requested UTC, simulator GPST calendar, GPS-minus-UTC offset/source NAV hash, GPS week/TOW, and time scale in schema-2 RF manifests;
5. rejects fractional scenario UTC seconds because this upstream CLI truncates them;
6. directly aligns corrected GNSS-SDR UTC PVT to requested UTC with zero implicit correction.

Legacy mislabeled dynamic artifacts are retained under:

```text
artifacts/legacy_timescale_runs/
```

Do not use those legacy runs for final metrics without an explicit documented legacy offset override.

### Dynamic RF-to-PVT validation

Common dynamic test settings:

- Seoul: `37.5665°, 126.9780°`
- WGS-84 ellipsoidal altitude: `120 m`
- epoch: `2022-01-01T00:00:00Z`
- duration: `90 s`
- trajectory cadence: `10 Hz`
- GPS L1 C/A IQ: `2.6 Msps`, signed 8-bit interleaved IQ
- each IQ: `467,480,000 bytes`, actual duration `89.9 s`
- GNSS-SDR channels configured with dynamic-PRN headroom

Corrected run IDs:

```text
seoul-normal-s1-straight-90s-2022_20220101T000000Z
seoul-normal-s2-circle-90s-2022_20220101T000000Z
seoul-normal-s3-sweep-90s-2022_20220101T000000Z
```

Results:

- S1 straight (`5 m/s`, east, `450 m`)
  - horizontal position median / p95: `0.840 / 1.403 m`
  - horizontal velocity median / p95: `0.093 / 0.198 m/s`
- S2 circle (`50 m` radius, one closed lap in `90 s`)
  - horizontal position median / p95: `0.919 / 1.527 m`
  - horizontal velocity median / p95: `0.094 / 0.183 m/s`
- S3 parallel sweep (`5 m/s`, `120 m` leg, `40 m` lane spacing)
  - horizontal position median / p95: `0.834 / 1.267 m`
  - horizontal velocity median / p95: `0.103 / 0.226 m/s`

Each run produced 119 precise PVT fixes from UTC 00:00:30.5 through 00:01:29.5 and decoded GPS NAV messages for 11 PRNs.

Dynamic validation:

- binds truth CSV bytes to the RF manifest SHA-256 and fails closed on mutation;
- aligns truth and PVT by corrected UTC;
- supports LLH and ECEF truth;
- separates all acquisition/tracking candidates from NAV-decoded PRNs;
- keeps acquisition-only/unstable candidates in the summary but plots normal Doppler only for NAV-decoded PRNs;
- writes `validation/summary.json`, `validation/aligned_samples.csv`, and `validation/dashboard.png` under each receiver run.

## Tests and code quality

- Automated tests: `90 passed`
- Final specification review: passed
- Final code/scientific-quality review: approved
- Latest implementation commit before this state note: `5718cda Add validated dynamic UAV GNSS scenarios`

## Important limitations / not yet completed

- The canonical Notebook still has a static-only safety guard. Dynamic execution currently uses tested CLI/module paths.
- A single persistent experiment YAML schema covering both static and parameterized dynamic flight has not yet been implemented.
- Independent PRN-wise expected-Doppler comparison from a separate RINEX/SP3 predictor is not yet complete. Current Doppler plots are GNSS-SDR measurements and RF-chain validation, not independent simulator validation.
- Spoof IQ, attack scenarios, detector datasets, statistical/ML/DL baselines, and external-corpus validation are not yet implemented.
- `altitude_m` is WGS-84 ellipsoidal height, not terrain-relative AGL. DEM/terrain conversion is future work if true AGL is required.
- The pinned upstream GPS-SDR-SIM limit for this path is 300 seconds per run.

## Next recommended task

Implement one unified experiment YAML schema plus one linear Jupyter Notebook orchestration path.

Desired config modes:

```text
static
straight
circle (speed-controlled or laps-controlled)
parallel_sweep
```

The YAML should store source flight parameters, not only a generated trajectory path. The Notebook should:

```text
select YAML
→ validate and display resolved settings/estimated IQ size
→ generate or reuse trajectory truth
→ optionally generate IQ
→ optionally run GNSS-SDR
→ validate PVT/velocity/Doppler
→ display artifact-derived progress and figures
```

Keep source notebooks output-free in Git. Default expensive flags should remain off (`RUN_IQ_GENERATION=False`, `RUN_GNSS_SDR=False`) to prevent accidental multi-hundred-MB regeneration. Tested modules under `src/` must perform validation and computation; Notebook cells should remain thin orchestration and visualization layers.

## Resume checklist

```bash
cd /home/ubuntu/projects/gnss-doppler-lab
git status --short
git log --oneline --max-count=5
.venv/bin/pytest -q
```

Then read this file and inspect the corrected RF/receiver manifests before modifying the experiment schema.
