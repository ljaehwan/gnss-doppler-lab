# CRID Stage-0 R4b Phase A physical-identifiability execution

Final verdict: `INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE`
Next state: `NOT_AUTHORIZED`

All 264 frozen clean-control replays completed successfully from pushed freeze commit `f61e1338bb43c432171a163e25b1ed547e632434`. The authoritative R2 literals were used under the R4a decision-equivalence authorization; no threshold was re-estimated or replaced.

Analysis was stopped with explicit user authorization after 38 of 66 cases completed because the frozen primary PASS became mathematically impossible: all 15 completed OAK negative controls failed the required per-case alarm-ratio limit, while the gate requires 30/30 negative PASS. The remaining 28 cases were not started. Consequently support and the 36-case positive response surface are incomplete, so this artifact records the fail-closed INCONCLUSIVE verdict rather than a full NO_GO verdict.

No post-result code, threshold, score, window, gate, truth, or control change was made. Phase B was not executed, and attack stat/hash/open/mmap/read counters are all zero.
