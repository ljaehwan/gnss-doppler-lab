# MOSAIC Stage-0B R1 execution

This branch executes the frozen 72-case R1 campaign from preregistration commit `3db0e12976b6ff98452096e921cf298be459d0e8` in two immutable stages.

Before any raw result is generated, the executor, metrics, tests, configuration allowlist, identity tolerances, controls, bootstrap policy, and dataset-specific gates are committed and pushed as `PRE_EXECUTION_FREEZE`. Execution commands require that pushed SHA and a clean worktree.

Every receiver replay starts at raw sample zero with the original prefix byte-identical. Injection exists only inside the R0c-authorized interval. Temporary injected IQ is processed one case at a time and deleted only after its SHA, size, quantization receipt, receiver log, and native TRACE inventory are written beneath the dedicated external result root.

The engineering identity gate runs before the 72 cases. Single-PRN cases run before the single-PRN physics gate; four-PRN cases are forbidden unless that gate passes. No result-dependent score, threshold, subset, or gate changes are permitted.
