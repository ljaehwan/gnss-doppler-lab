# MOSAIC-GNSS Stage-0A/0B Foundation

This branch adds a fail-closed MOSAIC-GNSS foundation harness.  It does not modify existing TRACE/MCTD/B0/M1/GCMR/ACAF/PG-SCC code and does not run neural training or attack-scenario scoring.

## Scope

Stage-0A inventories cleanStatic raw IQ, receiver executable metadata, native 1 ms TRACE dumps, receiver state/action fields, and compact stable PRN-epoch rows.  It validates causal TRACE alignment and raw sample bounds using the existing TRACE schema.

Stage-0B code provides short-window PRN-specific raw-IQ counterfeit generation, residual projection, residual CAF, and analytic recovery helpers.  Receiver-in-loop injection is not run unless Stage-0A and navigation-bit provenance pass.

## Fail-closed outcome in this worktree

The available MCTD cleanStatic manifests bind TEXBAT and OAKBAT raw IQ and the patched GNSS-SDR receiver.  However, no decoded navigation-bit sequence or independently validated 20 ms Prompt sign provenance is available in the Stage-0A inputs.  The harness therefore refuses to assume a `+1` nav bit and records `INCONCLUSIVE_NAVIGATION_BIT_PROVENANCE`.

## Commands

```bash
python scripts/run_mosaic_stage0a.py
python scripts/run_mosaic_stage0b.py
python scripts/verify_mosaic_stage0ab.py
pytest tests/test_mosaic_alignment.py tests/test_mosaic_injector.py tests/test_mosaic_residual_caf.py
```

## Artifact root

`artifacts/mosaic_stage0ab_foundation/`

The artifact manifest records SHA-256 checksums for all compact MOSAIC outputs.  No raw IQ, receiver dumps, large NPZ/MAT/BIN files, or scenario-scale synthetic IQ are stored there.
