# GCMR-PI r4 corrected diagnostic — fail-closed result

## Result

This run intentionally produced **no proxy relation-destruction result**, no corrected thresholds, and no warm-up metrics.

The frozen `model.pt` was loaded without retraining and the original indexed, causal 3-tap EventRecord construction (`history=4`) was replayed against OAKBAT `os1`.  The reconstructed raw GRU residual → frozen whitener → frozen diagnostics path did **not** reproduce the frozen r3 score CSV within the required `1e-8` tolerance. Exact measured component differences are in `blocker_evidence.json`.

Under the corrected contract, this is a hard blocker: relation-cache observations cannot stand in for residuals/innovations; cosine-matrix deltas cannot stand in for a frozen pair-model re-score. Therefore the run stopped before direction-pool sampling, relation destruction, threshold emission, or warm-up evaluation.

## Preserved inputs / output policy

- Frozen source left unchanged: `artifacts/frozen/gcmr-pi-oakbat-3t-r3/`
- New diagnostic directory only: `artifacts/gcmr_pi_r4_corrected/`
- `blocker_evidence.json` records the actual reproduction failure.
- No `innovations/*.npz`, thresholds, destruction statistics, or fabricated component summaries were written.

## Tests actually run

```bash
PYTHONPATH=$PWD/src /home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python \
  -m pytest tests/test_gcmr_pi_r4_corrected.py -q
# 6 passed
```

The test suite calls the real corrected module and checks warm-up boundaries, direct RelationOnly normal quantiles, normal-only pool enforcement, independent source allocation, PRN norm preservation, fixed-seed reproducibility/different-seed variation, and fail-closed unavailable innovations.

```bash
PYTHONPATH=$PWD/src:$PWD/scripts /home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python -m pytest \\
  tests/test_gcmr_experiment.py tests/test_gcmr_geometry.py tests/test_gcmr_model.py \\
  tests/test_gcmr_peak_innovation.py tests/test_gcmr_peak_innovation_adapter.py \\
  tests/test_gcmr_peak_innovation_indexed.py tests/test_gcmr_peak_innovation_pipeline.py \\
  tests/test_gcmr_relations.py tests/test_gcmr_pi_r4_corrected.py tests/test_r4_diagnostics.py -q
# 85 passed
```

## Reproduction command actually run

```bash
PYTHONPATH=$PWD/src:$PWD/scripts /home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python \
  scripts/run_gcmr_pi_r4_corrected.py \
  --frozen artifacts/frozen/gcmr-pi-oakbat-3t-r3 \
  --out artifacts/gcmr_pi_r4_corrected
# expected exit: 2 (fail-closed mismatch)
```

A future corrected r4 campaign requires the exact r3 EventRecord/predictor/whitener implementation provenance needed to reproduce `S_common`, `N_eff`, `S_pair`, `energy`, and `Full` before any destruction calculation is scientifically valid.
