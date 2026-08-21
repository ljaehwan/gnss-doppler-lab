# CRID Stage-0 R4 Phase A clean physical-identifiability

The pre-result freeze was prepared from base `8cf594bde9c8e48bf80b5872e4ca13e1d0d13b0d` and committed and pushed as `66602181d3e024490fd71ef606c84d04c09c0ed5`. The exact 66 clean R3 controls, truth sidecars, 45-second packages, receiver, C0-C3 definitions, clean model source, threshold, code, window, and gates are bound.

The pushed freeze SHA passed clean-checkout authorization and the allowlisted input preflight. Deterministic threshold recomputation then failed the preregistered exact-equality binding before any control replay or score was run:

- OAK q99: committed `-21.705587048010322`; recomputed `-21.705587048009935`; absolute difference `3.87e-13`.
- TEX q99: committed `-21.942672917134093`; recomputed `-21.94267291713392`; absolute difference `1.73e-13`.
- The clean holdout FPRs matched exactly, but the frozen contract requires exact q99 equality.

The run therefore stopped fail-closed with threshold substatus `INCONCLUSIVE_THRESHOLD_BINDING` and terminal verdict `INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE`. Replay count is zero, score metrics were not produced, and the frozen physical gate was not evaluated. No code, threshold, window, score, gate, or control selection was changed after observing the result.

Phase B and actual spoofing-data evaluation were not executed. Attack access is zero bytes, zero opens, zero hashes, zero stats, and zero mmaps.

Manifest integrity and a terminal-state semantic check pass. The pre-result frozen compact verifier's `final` mode only models a completed 264-replay path, so it reports FAIL for this required pre-replay threshold stop; its unmodified output is retained in `logs/frozen_final_verifier.txt` instead of changing verifier code after observing the result.
