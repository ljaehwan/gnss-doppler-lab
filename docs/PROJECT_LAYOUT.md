# GNSS Doppler Lab project layout

This repository is organized as a notebook-led research project with tested Python modules.

## Main directories

- `notebooks/`
  - Research-facing orchestration and explanation.
  - The canonical notebook is `notebooks/gnss_spoofing_research_workflow.ipynb`.
  - Use it as the main experiment control surface: choose configuration, run/reuse stages, and display figures.

- `src/gnss_doppler_lab/`
  - Reusable implementation code called by the notebook and tests.
  - Keep signal generation, receiver parsing, feature extraction, acquisition-surface plotting, and model baselines here rather than duplicating logic in notebook cells.

- `tests/`
  - Unit and notebook-structure tests for the implementation.

- `docs/`
  - Current-state notes, methodology notes, and project layout documentation.

- `artifacts/`
  - Generated data and derived outputs only. This directory is ignored by git except `.gitkeep`.

## Artifact subdirectories

- `artifacts/rf_runs/`
  - Generated normal GPS L1 C/A IQ runs from `gps-sdr-sim`.
  - Each run keeps `manifest.json`, `gps_l1ca_s8_iq.bin`, and simulator logs.

- `artifacts/receiver_runs/`
  - GNSS-SDR outputs from RF/IQ runs.
  - Each run keeps `manifest.json`, `receiver.log`, `tracking.csv`, `tracking_summary.csv`, and raw tracking dumps under `raw/`.

- `artifacts/figures/`
  - Paper/notebook figures derived from runs.
  - Current acquisition-stage plots are under `artifacts/figures/acquisition/<run_id>/`.

- `artifacts/paper_baselines/`
  - Derived feature datasets and baseline model outputs for paper experiments.

- `artifacts/legacy_timescale_runs/`
  - Older runs kept for audit/comparison where time-scale handling differs from the current pipeline.

## Root cleanup rule

Do not leave generated IQ, `.tmp`, notebook outputs, figures, or datasets in the repository root. Generated outputs should go under `artifacts/` with a run-specific manifest or summary JSON.
