# TEXBAT official dataset staging

This directory is reserved for the public TEXBAT benchmark dataset used only for external validation. Training normals should come from our own controlled receiver/scenario runs, not from TEXBAT clean files.

Official source pages:

- TEXBAT page: https://radionavlab.ae.utexas.edu/texbat/
- RNL datastore path referenced by the official page: https://rnl-data.ae.utexas.edu/datastore/texbat/
- Redirecting download URL: https://radionavlab.ae.utexas.edu/datastore/texbat/

## Project policy

- Do not train the main detector on TEXBAT spoofing labels.
- Use TEXBAT after normal-only model selection as an external validation corpus.
- Keep large raw IQ / dataset archives out of git. Store them under `raw/`.
- Keep small provenance files, checksums, and extraction manifests in git when possible.
- Preserve the original official filenames and record SHA-256 hashes before any conversion.
- Write converted GNSS-SDR outputs or feature windows under `derived/` or `artifacts/`, with a manifest linking back to the raw hash.

## Git policy

Tracked:

- `README.md`
- `manifests/manifest.template.yaml`
- `manifests/source_urls.txt`
- `manifests/scenarios.json`
- `.gitkeep` placeholders

Not tracked:

- `raw/**/*.bin`
- `derived/**`

## Suggested layout

```text
data/external/texbat/
├── README.md
├── raw/                 # official .bin IQ files, git-ignored except .gitkeep
├── checksums/           # sha256 manifests for downloaded official files
├── manifests/           # source URL, access date, file metadata, scenario notes
└── derived/             # optional local unpack/converted files, git-ignored except .gitkeep
```

Expected local raw filenames when downloaded:

```text
raw/cleanStatic.bin
raw/cleanDynamic.bin
raw/ds1.bin
raw/ds2.bin
raw/ds3.bin
raw/ds4.bin
raw/ds5.bin
raw/ds6.bin
raw/ds7.bin
raw/ds8.bin
```


## Download helper

Use the VM-local helper when an external validation subset is needed. It defaults to the ScienceDB mirror because the UT `rnl-data.ae.utexas.edu` host may timeout from this VM:

```bash
python3 scripts/download_texbat.py --files ds1.bin
```

The helper writes raw `.bin` files under `raw/`, writes per-file SHA-256 files under `checksums/`, and keeps transient `download_status.json` out of git. The full TEXBAT corpus may exceed the current VM disk, so download subsets intentionally.

Full corpus size: 373.02 GB on ScienceDB. Current VM free space is smaller than the full corpus, so download only selected external-validation files as needed. Do not download `cleanStatic.bin`/`cleanDynamic.bin` for training unless we explicitly decide to use them as auxiliary reference data.

## Scenario summary

See `manifests/scenarios.json` for machine-readable metadata.

Short grouping:

- Clean references: `cleanStatic.bin`, `cleanDynamic.bin`
- Static spoofing: `ds1`, `ds2`, `ds3`, `ds4`, `ds7`, `ds8`
- Dynamic spoofing: `ds5`, `ds6`
- Time push: `ds2`, `ds3`, `ds5`, `ds7`, `ds8`
- Position push: `ds4`, `ds6`
- Switch: `ds1`

## Minimum manifest fields for each downloaded file

```yaml
dataset: TEXBAT
source_page: https://radionavlab.ae.utexas.edu/texbat/
source_url: <official download URL>
retrieved_utc: <ISO-8601>
local_path: data/external/texbat/raw/<filename>
sha256: <sha256>
bytes: <file size>
scenario_id: <for example ds4/ds7 when known>
sampling_format: <for example 16-bit complex IQ, 25 Msps>
center_frequency_hz: <if known>
sample_rate_hz: 25000000
license_or_access_notes: <source-page notes>
notes: <conversion or receiver-processing notes>
```

## Research usage

Use TEXBAT spoofing scenarios primarily as external validation. The main detector should be trained on normal/authentic tracking observables, then evaluated on TEXBAT spoofing recordings.
