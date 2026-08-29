# TUNI GPS C-5 clean receiver preflight v1

## Decision

The TUNI2025 GPS C-5 clear-sky recording is compatible with the project
nine-tap GNSS-SDR pipeline after explicit 16-bit byte swapping. A clean-only
10 s preflight acquired and tracked eight GPS PRNs and produced 76,886 valid
tracking epochs. No spoofing payload was opened or analyzed.

This is a receiver-compatibility result, not a spoof-detection result. Its
purpose is to remove data-format uncertainty before the attack evaluation
contract is frozen.

## Source provenance

- Dataset: TUNI2025 GPS C-5, clear-sky, static, no multipath, true position.
- Zenodo record: `10.5281/zenodo.15572976`.
- Raw file size: 29,999,832,000 bytes.
- Raw MD5: `a03dedd79ac4208f6d60b4c916484dba`.
- Official record sampling rate: 50 MSps.

The Zenodo prose labels the payload as interleaved 32-bit float. Direct byte
inspection does not support that representation: values are laid out as
big-endian signed 16-bit I/Q pairs. Reading the payload as little-endian
shorts or floating point produces implausible amplitudes; reading it as
big-endian signed 16-bit I/Q at 50 MSps yields valid GPS acquisition and
tracking. The successful receiver result is therefore the operational format
check used by this project.

An earlier 25 MSps attempt produced zero valid PRNs. That rate came from an
unrelated FGI-GSRx example whose embedded input path points to TEXBAT DS7, not
from the C-5 metadata, and is rejected for this dataset.

## Receiver contract

- Input: big-endian interleaved signed int16 I/Q at 50 MSps.
- Conditioner: `Ishort_To_Complex` with
  `DataTypeAdapter.swap_endian=true`.
- Internal sampling rate: 5 MSps via direct resampling.
- Acquisition: GPS L1 C/A, 31 channels, 1 ms coherent integration,
  `pfa=0.01`, five dwells, Doppler range +/-6 kHz in 125 Hz steps.
- Tracking: nine complex taps at 0.125-chip spacing.
- Evaluated duration: first 10 s.
- Patched executable SHA-256:
  `0f11cb72a00ccbddde1efc32330e279acd8392d359742a66175b88953cb50ecd`.
- Receiver configuration SHA-256:
  `dfd12d9c4acfff8ed936739274be40fac1950b68cfcb6c2ab4317d88a527f344`.

The pre-existing nine-tap executable was preserved byte-for-byte at SHA-256
`fc00d610fdb966358ac85486b616ff2411cb163ca8269dd0b435919c5e197e25`.
The TUNI executable adds only the optional endian-swap data adapter on top of
the recorded nine-tap patch.

## Result

| Metric | Result |
|---|---:|
| GNSS-SDR return code | 0 |
| Tracking MAT files | 31 |
| Valid PRNs | 8 |
| Valid tracking epochs | 76,886 |
| Valid PRN IDs | 5, 7, 16, 18, 23, 26, 27, 29 |
| Compatibility decision | PASS |

The receiver output occupies 28 MB at
`/home/ubuntu/ssd_data/gnss-early-detection/artifacts/tuni-gps-clean-complex9-preflight-be-50msps-v1`.
The repository stores the compact manifest, runner, unit test, and reproducible
GNSS-SDR patch; the full tracking dumps stay on the SSD.

Sealed compact artifact hashes:

- result manifest SHA-256:
  `4e2949fd796092313f4c10db630ce41c030b531b8ae14c878e7d190f81b43174`;
- endian patch SHA-256:
  `36b21572e6343e2b59c3288cde4b4b7cb14768871adade553b8f5e35737167e1`;
- preflight runner SHA-256:
  `d7ee247f566d725d7c3a36e6cfb35de6f9a9e57bb784f18f49fc64fb87966464`.

## Sealed boundary and next step

The SS-17, SS-18, and SS-20 raw payloads remain outcome-sealed. Only their
previously recorded download sizes and checksums are known. Before opening
them, the project must freeze:

1. which already-trained detector/checkpoint is evaluated;
2. feature extraction and support rules, including behavior below eight PRNs;
3. attack timing and guard intervals from scenario documentation;
4. primary specificity, sensitivity, and detection-delay metrics; and
5. failure handling, exclusions, and the no-retuning rule.

Only after that commit is pushed should the attack recordings be processed.
