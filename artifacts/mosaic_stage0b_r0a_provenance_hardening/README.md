# MOSAIC Stage-0B R0a provenance hardening

Final verdict: **RAW_MAPPING_MISMATCH**. Stage-0B injection is **not authorized**.

This audit did not decode Prompt again, change any NAV bit, alter boundary phase/polarity, access attack data, run a receiver/injection, train a model, or modify the frozen R0 artifact.

## Independent scientific recomputation

The verifier read the actual transmitted logical bits from the frozen compressed sidecar and used its own explicit IS-GPS-200 parity equations. It recomputed 6000 bits, 200/200 parity-valid words, 20/20 preambles, 10/10 TOW/subframe continuity checks, and zero D29*/D30* chain errors. The distribution is 1=2986, 0=3014; all ten 600-bit PRN sequence hashes are distinct. OAK TOW is 381636→381642 and TEX TOW is 477918→477924.

The old statement `two_separated_intervals=true` is corrected: each PRN has one contiguous approximately 12-second interval containing two consecutive subframes, with starts 6 seconds apart. `two_separated_intervals=false`; `distant_interval_validation=NOT_PERFORMED`.

## Boundary and mapping audit

TRACE endpoint transcription itself passes: 14976 start/end values were compared to native TRACE and the transcription error count is 0. The former constant `sample_boundary_error_samples=0` is not reused; the defensible field is `trace_endpoint_transcription_error_samples=0`.

However, authenticated receiver execution order proves `data_symbol_boundary` means **END_OF_CURRENT_BIT**. `save_correlation_results()` accumulates the current Prompt and increments/modulos the symbol counter before the flag is evaluated. The flagged row is the current bit's final 1-ms interval; the next row is the new bit's first interval. All 754 observed sign-transition flags agree (transition from flagged row to next; zero from previous row to flagged row). Frozen R0 used the flagged row itself as each bit start, so its raw NAV-bit mapping is one epoch early.

## Frozen common intersections

- OAKBAT: `[150270296, 210197273)`, 11.98539540 s, PRNs [10, 11, 21, 24, 27].
- TEXBAT: `[817790304, 1117492038)`, 11.98806936 s, PRNs [3, 13, 16, 19, 30].

These intersections were computed from frozen `coverage_summary.json`, not hardcoded. Candidate starts are recorded only for audit and are not authorized because the mapping semantics failed.

## Tamper-negative tests

All 7/7 mutations were rejected with their expected labels: bit flip, derived word-hex change, HOW-bit change with unchanged TOW CSV, false parity CSV PASS, preamble flip, sample endpoint change, and constant +1 replacement. Mutations used in-memory copies; committed R0 was untouched.

## Test record

R0a-focused plus existing Stage-0A/NAV-bit tests: 32 passed. Baseline test set excluding the new R0a test file: 547 passed, 6 failed. Full set including R0a: 553 passed, the same 6 failed. All six are the pre-existing missing `scripts/train_peak_floor_temporal_autoencoder.py` failures; R0a added no failure. The committed-artifact verifier passed independently and is also exercised by the fresh-clone procedure.

Next action: in a separately authorized task, create a new versioned mapping shifted to the following TRACE row and independently re-audit it. Do not overwrite R0.
