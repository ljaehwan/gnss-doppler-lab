# TEXBAT official dataset staging

This directory is reserved for the official TEXBAT spoofing benchmark dataset used only for external validation.

Official source pages:

- TEXBAT page: https://radionavlab.ae.utexas.edu/texbat/
- RNL datastore path referenced by the official page: https://rnl-data.ae.utexas.edu/datastore/texbat/

Project policy:

- Do not train the main detector on TEXBAT spoofing labels.
- Use TEXBAT after normal-only model selection as an external validation corpus.
- Keep large raw IQ / dataset archives out of git. Store them under `raw/`.
- Keep small provenance files, checksums, and extraction manifests in git when possible.
- Preserve the original official filenames and record SHA-256 hashes before any conversion.
- Write converted GNSS-SDR outputs or feature windows under `derived/` or `artifacts/`, with a manifest linking back to the raw hash.

Suggested layout:

```text
data/external/texbat/
├── README.md
├── raw/                 # official archives/files, git-ignored except .gitkeep
├── checksums/           # sha256 manifests for downloaded official files
├── manifests/           # source URL, access date, file metadata, scenario notes
└── derived/             # optional local unpack/converted files, git-ignored except .gitkeep
```

Minimum manifest fields for each downloaded file:

```yaml
dataset: TEXBAT
source_page: https://radionavlab.ae.utexas.edu/texbat/
source_url: <official download URL>
retrieved_utc: <ISO-8601>
local_path: data/external/texbat/raw/<filename>
sha256: <sha256>
bytes: <file size>
scenario_id: <for example ds4/ds7 when known>
notes: <sampling format, center frequency, documentation link, if known>
```
