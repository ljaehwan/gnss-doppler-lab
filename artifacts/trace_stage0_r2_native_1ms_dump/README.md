# TRACE Stage-0 R2 native 1 ms receiver dump

Configuration and Phase-A gates were frozen before receiver smoke replay.

This artifact is in its preregistered receiver-validation stage. Large native
receiver dumps remain under
`/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2-native-1ms-dump/`;
only their path, byte size, SHA-256, command, receiver/source identity, raw IQ
identity, sample range, and config hash are retained here.

The prior retained 20 ms products are audit evidence only. They are not read,
interpolated, restored, or scored by R2.


## Final outcome

Verdict: `INCONCLUSIVE_INPUT_OR_RECEIVER`.

Failure labels: `NATIVE_1MS_DUMP_IMPLEMENTATION_INVALID`.

Same final receiver build and authenticated TEXBAT cleanStatic raw slice produced different per-channel dump SHA-256 values and acquisition/session timing across repetitions; the preregistered deterministic-reproduction gate failed. All other Phase-A source-binding, schema, causal, cadence, finite-data, reassignment, and multi-PRN checks passed. Phase B and frozen TRACE-R1 scoring were not run. All performance,
FPR, delay, comparison, shuffle, bootstrap, and physical-control outputs are
explicitly unavailable; no placeholder performance plot was created.
