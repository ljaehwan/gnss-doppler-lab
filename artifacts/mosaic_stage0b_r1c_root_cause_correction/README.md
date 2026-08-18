# MOSAIC Stage-0B R1c root-cause scientific correction audit

This compact audit uses committed R1a/R1b retained evidence only. It performs no IQ injection, receiver replay, case regeneration, threshold selection, tuning, or Stage-1 work. R1a remains `NO_GO_MOSAIC_MULTI_PRN_RECOVERY` and the reproduced R1b primary result remains `MIXED_OR_UNIDENTIFIED_ROOT_CAUSE`.

The corrected causal verdict is `CAUSAL_ROOT_CAUSE_NOT_IDENTIFIED`. H3 is `PRESENT_BUT_NOT_DISCRIMINATIVE`, H4 is `PRESENT_BUT_NOT_DISCRIMINATIVE`, and H6 is `NOT_TESTABLE_FROM_RETAINED_EVIDENCE`. The computed recommendation is `TERMINATE_MOSAIC_AS_STAGE1_PATH`.

## Claimable and non-claimable language

See `paper_safe_claims.md`. In particular: Template mismatch and tracking instability were observed, but neither factor discriminated failed targets from successful comparators sufficiently to establish a causal root cause.

## Reproduction and verification commands

```bash
python scripts/run_mosaic_stage0b_r1c_root_cause_correction.py
pytest -q tests/test_mosaic_stage0b_r1.py tests/test_mosaic_stage0b_r1_execution.py tests/test_mosaic_stage0b_r1a_frozen_analysis.py tests/test_mosaic_stage0b_r1b_root_cause.py tests/test_mosaic_stage0b_r1c_root_cause_correction.py
python scripts/verify_mosaic_stage0b_r1_results.py
python scripts/verify_mosaic_stage0b_r1a_frozen_analysis.py
python scripts/verify_mosaic_stage0b_r1b_root_cause.py
python scripts/verify_mosaic_stage0b_r1c_root_cause_correction.py
```

## Recorded focused verification

- Focused pytest: `72 passed`.
- R1 result verifier: PASS.
- R1a verifier: PASS; recomputed `NO_GO_MOSAIC_MULTI_PRN_RECOVERY`.
- R1b verifier: PASS; 8 failure-case target rows and 20 comparator-case target rows.
- R1c verifier: PASS; 26 committed compact files checked.

The fresh-clone verifier validates the committed compact case-level evidence and independently recomputes corrected verdict/recommendation logic. It does not claim raw-science regeneration without the immutable external retained-evidence volume.
