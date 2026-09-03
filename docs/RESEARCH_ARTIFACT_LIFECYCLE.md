# Research artifact lifecycle

This repository preserves several generations of GNSS spoofing research. A
negative result is not the same as disposable code, and a useful development
experiment is not automatically final paper evidence. Use the following states
before moving, deleting, or citing anything.

| State | Meaning | May support a final performance claim? | Retention rule |
|---|---|---:|---|
| `paper_core` | Directly implements or quantifies a current manuscript claim | Yes, within its stated boundary | Pin config, result, code, and artifact hashes |
| `paper_support` | Mechanism, ablation, public transfer, or limitation evidence | Only as separately qualified support | Keep separate from the final test pool |
| `development_only` | Used to choose a window, gate, threshold, or research direction | No | Preserve and label as reused/development data |
| `negative_or_failed` | Frozen gate failed, input support was insufficient, or the result contradicted a broad claim | No positive claim | Preserve; it defines the valid claim boundary |
| `superseded` | A later experiment replaced the endpoint or corrected the design | Not by itself | Keep provenance and point to the replacement |
| `legacy_other_direction` | B0/GRU, raw-IQ continuity, GCMR, Doppler-only, or other work outside the current CGC letter | No for the CGC paper | Maintain independently; do not mix thresholds or metrics |
| `uncommitted_review` | Present in the worktree but not accepted into a research line | No | Review ownership and provenance before committing |
| `shared_infrastructure` | Receiver, simulator, ephemeris, RF, and plotting utilities reused by several lines | Indirectly | Keep APIs stable and regression-tested |

## Rules

1. Never call a failed or unsupported experiment “meaningless.” Its protocol,
   failure mode, and hashes can prevent claim inflation and repeated work.
2. Do not physically reorganize hash-pinned experiment files in the same commit
   as a scientific result. First add an alias or manifest migration and verify
   every reference.
3. Do not pool public transfer audits with the final outcome-unseen simulated
   receiver--RF test. They use different data roles and, in some cases, an
   earlier released base-CGC configuration.
4. A development-selected parameter may appear in the method, but its reused
   development score must not be reported as final generalization performance.
5. Full IQ and receiver artifacts remain local/ignored. Commit compact summaries,
   protocols, hashes, configs, and deterministic analysis code.
6. Before citing a number, locate its `claim_id` in
   `configs/paper/wcl_cgc_v1_manifest.json` and follow the linked result record.

## Safe cleanup sequence

1. Run `python scripts/audit_wcl_cgc_manifest.py`.
2. Resolve `uncommitted_review` entries one at a time.
3. Mark an obsolete experiment `superseded` and identify its replacement.
4. Check all repository references with `rg` before moving a path.
5. Move files only in a dedicated migration commit; never delete full evidence
   until its compact summary and hashes have been verified.
