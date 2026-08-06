# ACAF-NF Stage-0 R12: cleanStatic raw-IQ / tracker alignment audit

## Scope

R12 is **cleanStatic-only**. It performs receiver provenance binding and a raw-IQ CAF alignment diagnostic; it does not evaluate attacks, thresholds, detection performance, B0, a neural field, budgets, or a second source.

## Binding discovery

The runner starts at the retained tracking-MAT directory and walks ancestor directories for the matching cleanStatic receiver `manifest.json`. It parses that parent manifest plus `receiver.conf` and `receiver.runtime.conf`. A parent manifest is valid evidence when its retained raw MAT inventory and authenticated raw input identify the run. The supplied `cleanStatic.bin` must hash exactly to the authenticated SHA-256 for A1 to pass; a missing colocated manifest under `raw/` is not a binding failure.

## Interval and sampling contract

Each raw interval is built from true adjacent tracker rows in the same channel and PRN: `[previous, current)` or `[current, next)`. No fixed 25,000-sample interval is assumed. Roles are assigned by chronological recording ranges first, and sampling is PRN-stratified without tracker-row reuse. The R12 default selects 950 epochs (19 retained PRNs × 50) and fails closed if any used PRN has fewer than 50 validation epochs. Same-epoch overlap among distinct PRNs/channels is explicitly allowed; overlap between time roles is forbidden.

## Alignment and fail-closed selection

The registered CAF is delay `-1…+1` chips in `0.125`-chip steps by Doppler `-250…+250` Hz in `50`-Hz steps. The runner records exact-center recovery separately from tolerance recovery (`|delay| <= .125`, `|Doppler| <= 50`), boundary rate, pooled prompt-magnitude Spearman, and PRN-median Spearman.

A non-null `selected_alignment` requires all gates: A1 authenticated source binding, A2 adequate/consistent temporal support, and A3 recovery. If A2 or A3 fails, `selected_alignment` is JSON `null`; `diagnostic_best_candidate` remains available but is not a selected reconstruction. Alignment failure is not a physics no-go claim.

## Output integrity

The R12 artifact directory contains the required schema artifacts, six purpose-named plots, an executed focused-test transcript, and a recursive SHA-256 manifest covering every artifact file other than the self-referential checksum manifest.
