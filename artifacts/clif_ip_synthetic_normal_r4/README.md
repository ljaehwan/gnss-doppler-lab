# CLIF-IP synthetic-normal R4 artifacts

## Status: code + real target smoke complete; final campaign not run

The tracked final index contains exactly 60 planned 120 s normal recordings (30 per target: 24 train, 3 validation, 3 synthetic test). It is an index/provenance artifact only: the expensive final campaign was intentionally **not launched** in this stage. Consequently final R0/S0/S1 prediction, scenario, domain-gap, and 199-permutation metrics are provisional/NA and must not be interpreted as results.

## Actually generated and receiver-backed smoke

`smoke/` is separate from the final 60-row index and contains two actual 20 s end-to-end runs:

- SYN-OAK: native 5 Msps s16le, exact 400,000,000-byte final IQ, SHA-256 `9bd0c5071ff4db9accab30046488f7c810d7bb81a1c62a9d20616af6ad4d0864`; Method-A tracked 7 PRNs / 74,868 tracking rows; 183 B0 node rows and 40 M1 rows; 31.218 s pipeline runtime.
- SYN-TEX: native 25 Msps s16le, exact 2,000,000,000-byte final IQ, SHA-256 `1f3500c2607bbe2334578964820e9a75990574ae35b739b16aa3fcdbbe5b7b5e`; Method-A tracked 4 PRNs / 50,751 tracking rows; 144 B0 node rows and 40 M1 rows; 144.133 s pipeline runtime.

Both manifests record `finite=true`, `zero_placeholder=false`, and identical B0/M1/final IQ hashes. GNSS-SDR receiver logs, normalized tracking CSVs, 9-tap features/nodes and atomic manifests remain. Final IQ and raw receiver MAT dumps were removed only after `_SUCCESS`; failure paths preserve them.

These B0 artifacts are **actual Method-A receiver-backed tap magnitudes**. Signed 9D innovations are downstream `x-xhat` residuals in canonical order, not raw signed complex taps. No reconstructed B0 adapter or synthetic placeholder was used in the smoke.

## Generator probe finding

The installed gps-sdr-sim supports arbitrary `-s` and `-b 16`, but emits `duration - 0.1 s` and its `-T` TOE/TOC overwrite produced all-zero IQ for the day-199 RINEX. R4 therefore uses matching-ephemeris `-t`, requests `duration + 0.1 s`, and requires exact output bytes and nonzero receiver-backed features. No resampling, sample duplication, or zero padding is used.

See `docs/CLIF_IP_SYNTHETIC.md`, each smoke run `manifest.json`, and `checksums.json` for contracts/provenance. Files with explicit NA reasons are provisional scaffolding, not fabricated metrics.
