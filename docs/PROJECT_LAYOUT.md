# GNSS Doppler Lab project layout

This repository is currently organized around the **normal-only PRN graph spoofing-detection workflow**.

## Research-facing entry points

- `notebooks/experiments/normal_only_prn_graph_training.ipynb`
  - Main notebook for the current normal-only model workflow.
  - It consumes the two model-input CSVs: PRN-node windows and receiver-graph windows.

- `artifacts/current_pipeline/README.md`
  - Human-readable artifact map. Start here when you want to understand which generated files are currently used.

## Code

- `src/gnss_doppler_lab/`
  - Reusable pipeline modules.
  - Current important module: `normal_multi_prn_dataset.py`, which converts tracking/window features into the two model-input CSVs.

- `tests/`
  - Unit and notebook-structure tests.

- `configs/experiments/`
  - Versioned experiment YAMLs.
  - Current PoC config: `texbat_wrw_static_5min_normal.yaml`.

## Generated artifacts

The clean, researcher-facing view is:

```text
artifacts/current_pipeline/
├── 00_experiment_config/
├── 01_rf_iq_source_runs/
├── 02_receiver_tracking_outputs/
├── 03_whole_window_feature_csv/
├── 04_model_input_csv/
├── 05_trained_normal_models/
└── 06_external_validation_raw/
```

Meaning:

1. `00_experiment_config/` — YAML scenario settings.
2. `01_rf_iq_source_runs/` — generated RF IQ source runs, not AI input.
3. `02_receiver_tracking_outputs/` — GNSS-SDR tracking outputs.
4. `03_whole_window_feature_csv/` — per-PRN/per-window feature CSV before splitting into model inputs.
5. `04_model_input_csv/` — the two CSVs the model uses:
   - `normal_prn_node_windows.csv`
   - `normal_receiver_graph_windows.csv`
6. `05_trained_normal_models/` — current normal-only model outputs.
7. `06_external_validation_raw/` — external validation raw data, currently TEXBAT `ds4.bin`.

The older physical storage directories still exist only where needed by manifests and code paths:

- `artifacts/rf_runs/`
- `artifacts/receiver_runs/`
- `artifacts/model_datasets/`
- `artifacts/normal_only_prn_graph_training/`

The `current_pipeline/` directory links to the active files so the working flow is visible without duplicating multi-GB IQ data.

## Removed as inactive/stale

The following stale or non-current outputs were removed from the working tree:

- legacy time-scale audit runs
- old spoofing PoC runs
- old v1 morphology-only dataset
- old temporal-predictor outputs
- old paper-baseline output directory
- generated acquisition figures
- temporary root IQ file
- notebook backup copy
- pytest/cache debris

## Root cleanup rule

Do not leave generated IQ, `.tmp`, executed notebook copies, figures, or ad-hoc datasets in the repository root. New generated outputs should go under the appropriate current pipeline stage with a manifest or README.
