# r4 reproduction diagnosis

## Scope and gate

This branch diagnoses only r3 frozen-score reproducibility. It does not retrain a model, generate innovations, calculate thresholds, run destruction, or create corrected diagnostics.

CUDA was available and required. The reproduction runner loaded the immutable r3 `model.pt` directly on the NVIDIA RTX A6000, applied frozen seed `7`, and enabled `torch.use_deterministic_algorithms(True)`, deterministic cuDNN, and disabled cuDNN benchmarking. CPU is retained only as a separately labeled comparison; it is never considered a passing result.

## Result

CUDA replay did not reproduce the frozen r3 score CSV. Consequently this remains an **artifact-provenance failure**, not a model/detection claim. No tolerance was relaxed and no innovation NPZ/destruction/warm-up output was created.

Detailed per-scenario (`os1`–`os4`) CUDA and CPU component agreement is in `reproduction_diagnosis.json`, including mean/median/q95/q99/max absolute error, max-error time, relative error, Pearson/Spearman correlation, Full q99 alarm agreement, and disagreement times.

## Provenance

- Artifact commit: `6f51f4809bbd4c012b39424003f8bcc40e22cf4a`
- Current diagnostic base: `6b5e478d5efecab54515051e25473e385b1f3f67`
- Relevant r3 extractor/predictor/whitener/pair-model source paths have no Git diff between those commits.
- Runtime versions and CUDA/cuDNN information are machine-recorded in `reproduction_diagnosis.json`.

## Tests actually run

```bash
PYTHONPATH=$PWD/src /home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python \
  -m pytest tests/test_gcmr_pi_r4_reproduction.py -q
# 1 passed
```

## Runner actually run

```bash
PYTHONPATH=$PWD/src:$PWD/scripts /home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python \
  scripts/run_gcmr_pi_r4_reproduction.py \
  --frozen artifacts/frozen/gcmr-pi-oakbat-3t-r3 \
  --out artifacts/gcmr_pi_r4_reproduction
```
