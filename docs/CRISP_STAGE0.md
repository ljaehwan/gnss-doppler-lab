# CRISP Stage-0 protocol

CRISP tests whether a receiver-native complex nine-tap vector loses its
single-source projective rigidity under overlapping authentic and counterfeit
signals. It is a statistical physics probe, not a neural detector.

The Full representation is frozen as

`P_t = c_t c_t^H / (c_t^H c_t + epsilon)` and `R_t = P_t - P_(t-1)`.

A single PRN-agnostic ridge model per dataset predicts the clean conditional
mean from causal tracking context. Ledoit-Wolf shrinkage fitted on cleanStatic
calibration residuals whitens the innovation. Native 1 ms scores are reduced to
non-overlapping 20 ms PRN blocks, and the receiver score is the second largest
valid PRN score when at least four PRNs exist. Three consecutive q99 crossings
form an alarm. All configuration, thresholds, and models are committed before
attack evaluation.

Commands:

```bash
python scripts/run_crisp_stage0.py preregister
python scripts/run_crisp_stage0.py evaluate --preregistration-sha <sha>
python scripts/verify_crisp_stage0.py
pytest -q tests/test_crisp.py
```

The input contract accepts only authenticated TRACE `TRC1MS02` schema-v2
native complex dumps. Missing lineage is fail-closed; there is no magnitude,
zero-fill, or interpolation fallback.
