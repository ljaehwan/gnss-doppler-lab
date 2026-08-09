# PG-SCC Root-Cause Audit Implementation Freeze

This document records implementation construction errors observed before the
implementation-freeze commit. They occurred while editing source code, before the
protected root-cause audit was run, and are unrelated to attack outcomes or outcome
tuning.

## Pre-freeze construction error log

1. **Invalid long patch.** An oversized patch was rejected before it could form a
   valid implementation update. Useful code already present in the untracked runner
   was preserved; the runner was continued rather than rewritten.
2. **Split signature syntax error.** A function signature was accidentally split
   into invalid Python syntax during construction. It was corrected before freeze
   and caught by compile-only checking.
3. **Append-boundary indentation error.** Appended code initially crossed a function
   boundary with invalid indentation. It was corrected before freeze and caught by
   compile-only checking.
4. **Calibration comprehension bracket error.** A list-comprehension bracket typo
   introduced during Phase 1 completion was caught by `py_compile` and corrected
   before freeze.

No protected audit was executed to diagnose or fix these errors. No attack result
content was used to choose formulas, masks, covariance settings, selector settings,
thresholds, or verdict rules.

## Runtime contract

The runner requires the full pushed implementation commit SHA. Before opening any
protected cache or result it verifies the exact branch and preregistration ancestor,
tracked implementation files, clean worktree apart from declared generated outputs,
frozen config/source hashes, frozen input checksums, local/remote SHA equality, and
ahead/behind 0/0. Any violation fails closed.

The independent verifier reads committed/generated artifacts and checks file
presence, schemas, labels, checksums, and optionally a fresh clone. It never invokes
the audit producer.
