# PG-SCC Stage-0 R2 implementation freeze notes

## Pre-freeze construction log

1. The first metadata-inventory invocation failed before emitting a summary
   because the loader projected away `time_s` while a second validation still
   required it. Event support is keyed by structural source, scenario, phase,
   and receiver-second, so the redundant `time_s` requirement was removed.
2. A mechanical edit used while the managed patch helper was unavailable
   briefly removed the unique-PRN set insertion from the pure support module
   and inserted it in the universe helper. Inspection and `py_compile` caught
   the mistake before any test or protected execution; the correct insertion
   was restored.
3. The first synthetic test run had an over-broad assertion that rejected the
   verifier for reading the producer source during a static ordering audit.
   The test was narrowed to prohibit subprocess execution of the producer.
4. The first mechanical test correction matched an earlier similar source-read
   block. The resulting `NameError` was caught by pytest and the two blocks were
   corrected by exact line scope.

No construction error opened an NPZ, score table, detector outcome, or
protected R2 outcome. They are implementation notes, not scientific results.

## Runtime boundary

The R2 producer requires the full pushed implementation-freeze SHA and verifies
the exact branch, preserved R1 ancestry and hashes, R2 preregistration hashes,
frozen input manifests, clean worktree outside declared generated outputs, and
local/remote equality before opening even the metadata sidecars. The metadata
support gate then runs before any protected NPZ is opened.

The verifier never invokes the producer. It independently reconstructs the
event universe, unique-PRN histogram, K eligibility, strata, denominators, and
paired support fingerprints, and it supports read-only fresh-clone checking.

Protected Phase 2 execution remains prohibited until a separate instruction.
