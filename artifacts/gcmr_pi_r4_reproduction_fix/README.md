# GCMR-PI r4 reproduction fix

## Scope

This directory is a new diagnostic artifact for `research/gcmr-pi-r4-reproduction-fix`. It does **not** alter the earlier, invalid `artifacts/gcmr_pi_r4_reproduction/` result.

The prior runner expanded `reconstruct_event_innovation(...)` directly into `rescore_from_innovations(...)`. Reconstruction returns `(residual, z)`, whereas re-scoring requires `(z, residual)`. This runner now performs the required explicit handoff:

```python
residual, z = reconstruct_event_innovation(pipe, event)
diagnostics, scores = rescore_from_innovations(pipe, event, z, residual)
```

## Execution contract

- CUDA is authoritative; CPU is persisted only as a reference comparison.
- Frozen training seed: `7`.
- Deterministic PyTorch algorithms and deterministic cuDNN are enabled.
- Component identity tolerance: `1e-8`.
- A reproduction pass requires every frozen component (`S_common`, `N_eff`, `S_pair`, `energy`, `Full`) to meet that tolerance in every `os1`–`os4` scenario, plus identical `Full.q99` alarm decisions whenever the frozen threshold exists.
- Only `Full.q99` exists in the frozen threshold artifact. Per-component `threshold`, `alarm_agreement_rate`, and `alarm_disagreement_events` are therefore JSON `null` rather than synthetic values or Infinity.
- JSON is written with `allow_nan=False`; strict standard-JSON parsing was verified.

## Corrected outcome

`reproduction_diagnosis.json` reports:

- `status`: `reproduction_pass`
- `cuda_reproduction_pass`: `true`
- For CUDA `os1`, `os2`, `os3`, and `os4`, all five frozen-score components have `max_abs_error: 0.0`.
- `Full.q99` alarm agreement is `1.0` with zero disagreement timestamps in every scenario.

Accordingly, the prior branch's artifact-provenance failure conclusion is not carried forward. The corrected replay passes its defined CUDA identity gate.

## Tests

```bash
PYTHONPATH=$PWD/src:$PWD/scripts \
  /home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python -m pytest \
  tests/test_gcmr_pi_r4_reproduction.py tests/test_gcmr_pi_r4_corrected.py -q
```

Result: `12 passed`.

The test coverage includes:

1. `reconstruct_event_innovation` return contract: `(residual, z)`.
2. Correct forwarding into re-scoring as `(z, residual)`.
3. A deliberately reversed call is rejected by the re-scoring contract.
4. A real frozen `os1` event produces identical direct and helper scores.
5. Missing component thresholds serialize alarm fields as JSON `null`.

## Downstream gate

The CUDA reproduction gate is now passed. This branch records the corrected reproduction result only; it has not fabricated or backfilled innovation NPZ / corrected relation-destruction artifacts. Those outputs must be produced by their dedicated downstream diagnostic execution under a new artifact contract, now that the prerequisite gate is satisfied.
