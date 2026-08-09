# PG-SCC Stage-0 R2 support-feasibility preregistration

## Scientific status and fixed scope

R1 remains a preserved valid `FAIL_CLOSED` execution at commit
`8cd78ed724e57f97498da26547a9ecbbc2a78fe1`. R2 is a post-R1-failure,
support-repaired diagnostic iteration informed by the R1 support gate. It is
not independent confirmation. Any favorable R2 result requires a new untouched
holdout and a separately preregistered confirmatory execution.

Phase 1 used only cache-side support metadata. No score, detector outcome,
AUROC, threshold, verdict, causal metric, or protected R2 plot was inspected.
Eligibility and exclusions are fixed exclusively from support cardinality and
algorithmic method availability.

## Event universe and common support

The event universe is every unique `(source_role, scenario, phase, second)` in
the two frozen metadata sidecars. `time_s` is descriptive and is not an event
identity field because its PRN-local acquisition-window mean can differ within
one receiver-second. This corrects the R1 fragmentation mechanism without
changing R1.

For each comparison family and event, the producer constructs the set of unique
PRNs available to every named method and takes their exact intersection.
Duplicates are collapsed only by PRN within the same event. A method is
algorithmically unavailable if it cannot produce exactly one finite row for a
required event/PRN; this causes an explicit unavailable cell or fail-closed
support mismatch, never a performance-driven exclusion. Paired comparisons use
the identical common PRN set and the same event set on both sides.

The committed support summary is aggregate. Event identities and PRN sets are
not committed. The verifier independently reconstructs the event universe,
intersections, cardinalities, strata, and aggregate denominators from the
metadata sidecars.

## Fixed support strata and K eligibility

Strata are mutually exclusive:

- `K9`: at least 9 common unique PRNs. K9, K5, K3, and dense families are
  eligible, but each fixed-K estimand is reported separately.
- `K5`: 5–8 common unique PRNs. K5, K3, and dense families are eligible; K9 is
  `UNAVAILABLE`.
- `K3`: 3–4 common unique PRNs. K3 and dense families are eligible; K5 and K9
  are `UNAVAILABLE`.
- `DENSE_ONLY`: 1–2 common unique PRNs. Sparse comparisons are `UNAVAILABLE`.
  Dense output is descriptive only because no paired sparse-synergy estimand is
  feasible.
- `UNSUPPORTED`: zero common PRNs. Every method is `UNAVAILABLE` and the event
  remains in denominator accounting.

The metadata-only inventory found 260 events: 241 `K9`, 19 `K5`, and zero
`K3`, `DENSE_ONLY`, or `UNSUPPORTED`. The common unique-PRN histogram is
`{6: 2, 7: 5, 8: 12, 9: 24, 10: 184, 11: 33}`. Thus 241, 260, and 260 events
are cumulatively eligible for K9, K5, and K3 respectively, and zero events are
support-infeasible. These counts, not outcome performance, determine the
policy.

`K_eff` is prohibited. A method is evaluated only at its frozen fixed K. It is
never reduced to the event's available count, and scores at different K are
never treated as a shared scale. Raw scores are not pooled across K or support
strata.

## Methods, relational controls, and paired estimands

The fixed families are:

- K9: `pg_scc_k9`, `fixed9`, and `shuffled_k9`.
- K5: `pg_scc_k5`, `uniform_k5`, and `shuffled_k5`.
- K3: `pg_scc_k3`, `epl3`, and `shuffled_k3`.
- Dense: `dense_two_source_glrt`, descriptive unless paired to an eligible
  sparse method on identical support.

The non-learned fixed/uniform member is the relational ablation. The frozen
shuffled member is the coordinate-relationship permutation control. Evidence
for PG-SCC synergy requires paired PG-SCC improvement over both controls on the
same events and exact PRN intersections, with at least two 10-second temporal
blocks. Missing permutation evidence makes causal synergy interpretation
fail closed. Attack outcomes never choose a control, mask, K, or stratum.

Within each family × support-stratum cell, events receive equal weight. Primary
estimands are the paired score effect and normalized partial AUROC at FPR 0.05,
reported by DS3, DS4 transition-only, and DS7/DS8 combined. DS7 and DS8 are one
family. When all three families are available, their family summaries receive
equal weight. Secondary overall summaries report both event-count weighting
and equal-stratum weighting, with included cell counts and total eligible cell
counts. `LIMITED`/`UNAVAILABLE` cells are named and remain in denominators; raw
scores are never pooled across strata or K.

## Calibration, minimum samples, and uncertainty

Thresholds are fit only from `cleanStatic` calibration events in the same
support stratum at the same fixed K, using q0.99 with NumPy's `higher` rule.
Each reported cell requires at least 10 calibration events, 10 clean holdout
events, and 10 positive events. A confidence interval requires at least two
10-second temporal blocks. Missing a sample gate yields `LIMITED` or
`UNAVAILABLE`; it does not authorize cross-stratum borrowing.

False-alarm reporting includes clean holdout and strict external pre-onset
rates, Clopper–Pearson 95% intervals, and a clean-calibrated maximum FPR gate of
0.05. Calibration uncertainty uses leave-one-block-out and 2,000 whole-block
bootstrap replicates. Paired effects use whole 10-second blocks with the three
fixed seeds in the config. No iid-row bootstrap is allowed.

## Preserved controls and leakage guards

R2 retains the R1 clean/synthetic-only selector inputs, 20-seed selector
stability analysis, covariance provenance, empirical-residual AWGN multipliers
0.5/1/2, frozen physics/masks/timeline, DS4 transition-only limitation, and
DS7/DS8 family binding. Selector, pooling, thresholds, covariance choices, K,
support rules, and controls cannot be chosen or retuned using attack outcomes.

The implementation verifies the full exact freeze SHA, branch, R1 ancestry,
clean worktree outside declared outputs, committed config/support-summary/source
hashes, frozen-input hashes, and local/remote equality before opening an NPZ or
score table. A mismatch fails closed.

## No-drop ledger, missingness, and fail-closed output order

Every universe event is counted once in an aggregate stratum ledger. For each
method the ledger reports available + unavailable = total events. For each
estimand it reports eligible, paired, limited, unavailable, numerator, and
denominator counts. A denominator mismatch, missing event, duplicate method
row, nonidentical paired support, outcome-dependent selection attempt, missing
relationship permutation, manifest drift, source drift, or frozen-ancestry
failure is `FAIL_CLOSED` with no causal interpretation.

Before the freeze/provenance gate, no output may be written. After metadata
support reconstruction, only `support_accounting.json` or `fail_closed.json`
may be written. Calibration/control outputs require support validation;
paired results require clean-calibration and false-alarm gates; a final
diagnostic requires relationship controls, uncertainty gates, provenance, and
the checksum manifest. Sample-limited cells may be written only with explicit
`LIMITED`/`UNAVAILABLE` reasons.

The final R2 label, if Phase 2 is separately authorized, remains
`POST_R1_SUPPORT_REPAIRED_DIAGNOSTIC`. It cannot claim independent confirmation.
