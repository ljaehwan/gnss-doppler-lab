# CLIF-IP Phase 1 R2 — provenance-first gate

## Decision: BLOCKED

No scenario satisfies the provenance prerequisite. No evaluator/model was run. This is a new R2 provenance decision, not reuse of the R1 No-Go conclusion.

OAKBAT B0 receiver manifests authenticate raw-IQ identities and 5 MHz rates. M1 summaries for os2–os4 name the same-looking path and 5 MHz rate, but do not bind raw IQ by SHA-256 or state a recording-start sample/time origin. os1 has no paired M1 artifact. B0 tracking windows and M1 blocks cannot therefore be mapped to proven common raw sample intervals. GNSS receiver latency is also unquantified, preventing receiver-delay-aware causal alignment. Matching scenario labels and 0.5 s timestamps were explicitly rejected as proof.

No paired TEXBAT B0/M1 evidence was found, so TEXBAT was not evaluated.

## Data required to unblock

1. Independent B0 and M1 manifests binding the exact same raw-IQ SHA-256.
2. Shared raw sample zero/absolute time origin plus every seek, skip, decimation, and padding offset.
3. Sample rate/format and exact half-open sample-range formulas for B0 windows and M1 blocks.
4. Receiver processing/group delay and score-availability semantics sufficient to establish causal alignment.
5. Equivalent scenario-level paired evidence for any TEXBAT scenario.

Run `python scripts/verify_clif_ip_provenance_r2.py artifacts/clif_ip_feasibility_r2/provenance_manifest.json`; exit 2 means blocked. The gate implementation is fail-closed and tests show a blocked manifest cannot invoke an evaluator callback.
