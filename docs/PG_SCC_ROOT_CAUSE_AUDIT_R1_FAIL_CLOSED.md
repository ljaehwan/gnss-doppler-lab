# PG-SCC Stage-0 R1 Fail-Closed Record

PG-SCC R1 terminated at its frozen common-support gate. This is a preserved
execution failure, not a scientific root-cause verdict.

- Status: `FAIL_CLOSED`
- Blocker: `common_support_event_has_fewer_than_4_unique_prns`
- Runtime message: `pooled event has insufficient unique PRN support`
- Attempts: 1
- Frozen implementation: `5359bfab74d44d7153a32c5cf708ccab240fe219`
- Preregistration: `5e5339282c6154630fd11f94415cba794d9fa1ec`
- Command: `python3 scripts/run_pg_scc_root_cause_audit.py --implementation-sha 5359bfab74d44d7153a32c5cf708ccab240fe219`
- Invocation: `2026-08-09T11:25:35.544Z`
- Exit evidence: `2026-08-09T11:26:47.041Z`, code 1
- Terminal marker: `FAIL_CLOSED_BLOCKER=common_support_event_has_fewer_than_4_unique_prns`

The traceback ended in `validate_common_support` at frozen runner line 151. The
failure message did not identify the particular event, so its identity remains
explicitly `UNAVAILABLE`; no value was inferred or recovered through further
protected-outcome inspection.

After the gate, the audit was not rerun, the support rule was not relaxed, the
frozen implementation was not changed, and no outcome-based retuning occurred.
No `root_cause_verdict.json`, checksum manifest, diagnostic table, or plot was
produced. The artifact directory retained only the frozen `config.json` and
`source_commit.json` before this preservation report was added.

At preservation precheck, the exact R1 branch was clean at local/remote SHA
`5359bfab74d44d7153a32c5cf708ccab240fe219` with ahead/behind 0/0. The config
SHA-256 remained `ade700892fe3e055e0154ae859ba701a36f11a6ec9245db4b39fd087426eb0e6`,
the source metadata SHA-256 remained
`13ab9e544f06903dd6fde04c42e8aefb35a66946bbaf0456e95040fbb777d428`,
and `main` remained unchanged at `461eb4dc7bb794e719295daf028f6811658ba37f`.

Provenance: Codex session `019fe536-5cac-78c0-9cba-2583e15b4362`, session log
`/home/ubuntu/.codex/sessions/2026/08/09/rollout-2026-08-09T15-29-32-019fe536-5cac-78c0-9cba-2583e15b4362.jsonl`.
