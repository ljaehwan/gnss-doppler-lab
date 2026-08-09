# PG-SCC Stage-0 R2 pre-inventory design intent

Recorded before inspecting any R2 support-inventory counts.

R2 is a post-R1-failure diagnostic iteration. It is not independent
confirmatory evidence, and any promising result will require a new untouched
holdout for confirmation. R1 remains a preserved, valid `FAIL_CLOSED`
execution.

Analysis eligibility will be determined only by event-level common-support
cardinality across every compared method. Scores, detector outputs, attack
performance, thresholds, verdicts, and causal metrics must not influence
eligibility, stratum assignment, or method availability.

Candidate policy principles established before inventory are:

1. Construct one exact method intersection for each event and require paired
   comparisons to use that identical PRN set.
2. Prefer fixed, separately reported K strata (`K=9`, `K=5`, `K=3`) over a
   varying `K_eff`; never combine scores computed at different K as if they
   shared a scale.
3. Treat support below three as sparse-method `UNAVAILABLE`, retain the event
   in the denominator ledger, and allow only prespecified dense-only handling.
4. Never silently exclude an event. Every event contributes to aggregate
   stratum and availability counts even when no sparse comparison is feasible.
5. Set minimum-event and uncertainty gates before protected outcome execution.
   A stratum that misses a gate is `LIMITED` or `UNAVAILABLE`, not deleted or
   pooled opportunistically with another K.
6. Preserve clean-only calibration, relational ablation/permutation controls,
   AWGN controls, block bootstrap uncertainty, seed stability, leakage guards,
   frozen lineage, and fail-closed output ordering.

The support inventory may refine numeric sample gates and determine which of
these prespecified strata are feasible. It may not choose a policy using
observed performance.
