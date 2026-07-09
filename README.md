# gnss-doppler-lab

Development repository for GNSS-only Doppler-based spoofing detection research.

This repository is intentionally separate from the paper repository. It is for:
- global GNSS visibility and Doppler simulation
- observation-level spoofing injection
- dataset generation
- model training and baseline comparison
- experiment utilities and visualization

## Scope

This repo is for *development and experiments*.
The paper-writing repo remains separate.

## Research constraints

- GNSS-only
- No IMU
- Doppler-centered detection
- Observation-level spoofing injection, not RF waveform generation

## Initial layout

- `src/gnss_doppler_lab/`: core simulator and utilities
- `configs/`: scenario and experiment configs
- `scripts/`: runnable entrypoints
- `tests/`: unit tests
- `data/`: local data pointers and placeholder structure
- `artifacts/`: generated outputs kept out of git
- `docker/`: container build assets
- `.devcontainer/`: VSCode remote container support

## Quick start

```bash
git clone https://github.com/ljaehwan/gnss-doppler-lab.git
cd gnss-doppler-lab
docker compose up --build -d
python -m pytest
```

## Main server layout

Recommended Unraid host layout:

- repo: `/mnt/user/Storage_HDD/projects/gnss-doppler-lab/repo`
- data: `/mnt/user/Storage_HDD/projects/gnss-doppler-lab/data`
- outputs: `/mnt/user/Storage_HDD/projects/gnss-doppler-lab/outputs`

## Near-term milestones

1. Build satellite visibility/Doppler PoC with `gnss_lib_py`
2. Add scenario config loader for region/time/trajectory sweeps
3. Add spoofing injection module for short-window multi-PRN inconsistency
4. Generate first normal/spoofed dataset pair
5. Add baseline detectors before learned models
