# GCSPO Stage-0 R1 Frozen Completion

- Base commit: `57271e526e9e346c8d4d7626b006c5a88166f1be`
- Branch: `research/gcspo-stage0-r1-frozen-completion`
- Invocation: `gcspo-stage0-r1-frozen-completion-0acca83b5245429b`
- Scientific config SHA-256: `0db816116b95b41db8b7af7379cd7411cc52d43b6428ae00ab02d6ccac19f4ad`
- State at bootstrap: implementation pending; clean-only reproduction PASS; protected evaluation not started.
- Clean Full q99 holdout FPR: `0.071729957805907171` (1% gate FAIL).
- Exact-support predecessor blocker: B0 phase-local GRU warm-up emits no target scores for the first 12 half-second steps (6 s), while Full emits earlier valid windows; predecessor required complete key-set equality.
- Approved repair: exact intersection for B0 comparisons, all Full standalone rows preserved, no nearest-neighbor join, no threshold/model/science change.
- Final scientific decision: pending protected DS3/DS7 completion; GO is already impossible because the clean FPR gate failed.
