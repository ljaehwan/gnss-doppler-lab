# Research lines

This directory is a navigation layer over the repository's immutable experiment
paths. It separates active paper evidence from earlier research without moving
hash-pinned scripts, configs, or result records.

## Active paper line

- [`wcl-cgc/`](wcl-cgc/): current WCL clock-centered correlator geometry
  consistency paper
  - [`figures/`](wcl-cgc/figures/): manuscript figure generators and sources
  - [`equations/`](wcl-cgc/equations/): equation-to-implementation map
  - [`simulation/`](wcl-cgc/simulation/): RF/IQ generation and final experiment
  - [`data/`](wcl-cgc/data/): local HDD/SSD and public-dataset locations
  - [`experiments/`](wcl-cgc/experiments/): core, support, development, and
    negative experiment roles

## Earlier research

- [`legacy/`](legacy/): B0/GRU, graph/learned detectors, Doppler/static-reference,
  raw-IQ, GCMR, and superseded CGC campaigns

## Why this is an index instead of a physical move

Frozen configs and result summaries pin paths and SHA-256 values. Moving the
underlying files would break reproducibility and old commit references. New
work should use this directory to find the canonical file, while the executable
source remains under `src/`, `scripts/`, `configs/experiments/`, and
`docs/results/`.

The exact machine-readable classification is
[`../configs/paper/wcl_cgc_v1_manifest.json`](../configs/paper/wcl_cgc_v1_manifest.json).
Run `python scripts/audit_wcl_cgc_manifest.py` from the repository root after
changing any indexed path.
