# WCL CGC manual reproduction

This procedure answers two different reproducibility questions separately.

1. **Analysis replay:** starting from the retained GNSS-SDR tracking outputs,
   do signed-delay extraction, temporal stabilization, Partial-F scoring, and
   the final metrics reproduce exactly?
2. **Full RF rebuild:** starting from simulator source and navigation data, do
   newly generated 25 MHz RF streams reproduce the receiver-level conclusion?

The first is repeatable now and is the recommended manual check. The second is
an expensive new campaign because the original 25 MHz intermediate IQ files
were deliberately removed after their hashes and receiver outputs were sealed.
It must use a new output directory and must not overwrite the frozen endpoint.

## Frozen inputs and scenarios

- code/config: current `main`; original final release commit
  `e578a09e8585e90a86a1d03d63da386f9826981f`;
- retained endpoint: `/home/ubuntu/hdd_data/cgc_temporal_final_static_v1`
  (approximately 1.5 GB);
- five static geometries: Fairbanks, Punta Arenas, Casablanca, Sapporo, and
  Prince George;
- four conditions per geometry: normal, independent PRN multipath,
  carrier-coupled 100 m carry-off, and authentic-Doppler-locked 100 m code
  carry-off;
- frozen full-summary SHA-256:
  `b154e411f4447fb534c0e10dee45f2f14f84b0e4e4eeedaa771cfefaf721045d`.

Exact UTCs, coordinates, ENU displacements, random seeds, thresholds, and
evaluation gates are in
[`../../../configs/experiments/cgc_temporal_final_static_v1.json`](../../../configs/experiments/cgc_temporal_final_static_v1.json).

## Clean-clone setup on this VM

Use the HDD because the Ubuntu root disk has little free space.

```bash
cd /home/ubuntu/hdd_data/reproductions
git clone git@github.com:ljaehwan/gnss-doppler-lab.git gnss-doppler-lab-wcl-cgc-v1
cd gnss-doppler-lab-wcl-cgc-v1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/wcl-cgc-v1.txt
.venv/bin/python -m pip install --no-deps -e .
```

The frozen config checks the exact simulator and receiver binaries even though
analysis replay does not execute them. On this VM, expose the already verified
tool build without copying 1.9 GB:

```bash
ln -s /home/ubuntu/projects/gnss-doppler-lab/.tools .tools
```

The following ignored, input-only artifacts are needed by the focused test
suite. They are not generated outputs of the new clone.

```bash
mkdir -p artifacts
ln -s /home/ubuntu/projects/gnss-doppler-lab/artifacts/simulation_v4_paired_train_generation_v1 artifacts/
ln -s /home/ubuntu/projects/gnss-doppler-lab/artifacts/simulation_v4_peak_mixture_law_v1 artifacts/
ln -s /home/ubuntu/projects/gnss-doppler-lab/artifacts/correlator_geometry_identifiability_train_v1 artifacts/
```

## Recompute the primary result

```bash
.venv/bin/python scripts/reproduce_wcl_cgc_final_analysis_v1.py \
  --data-root /home/ubuntu/hdd_data/cgc_temporal_final_static_v1 \
  --output-dir reproduction-output/wcl-cgc-final-analysis-v1
```

Success is reported only when all four logical objects—decision, aggregates,
gates, and per-pair results—match and the three regenerated CSV files have the
same SHA-256 as the sealed files. Output is written only below the requested
`--output-dir`.

The 2026-09-03 fresh-clone execution at commit `d93dc04` returned
`EXACT_MATCH`; its committed audit note is
[`../../../docs/results/WCL_CGC_FRESH_CLONE_REPRODUCTION_V1.md`](../../../docs/results/WCL_CGC_FRESH_CLONE_REPRODUCTION_V1.md).

## Recompute the equation audit

The equation audit currently writes its three tracked outputs in place. Run it
only in the disposable clone, then confirm that Git sees no difference:

```bash
.venv/bin/python scripts/audit_cgc_formula_accuracy_v1.py
git diff --exit-code -- \
  docs/results/cgc_formula_accuracy_audit_v1_summary.json \
  docs/results/figures/cgc_formula_accuracy_audit_v1.png \
  docs/results/figures/cgc_formula_accuracy_audit_v1.pdf
```

## Focused code tests

The lightweight WCL core and controlled-mechanism set is:

```bash
.venv/bin/pytest -q \
  tests/test_wcl_cgc_manifest.py \
  tests/test_correlator_geometry.py \
  tests/test_clock_centered_geometry.py \
  tests/test_static_reference_geometry.py \
  tests/test_temporal_cgc.py \
  tests/test_code_carrier_sim.py \
  tests/test_satellite_multipath.py \
  tests/test_peak_mixture_law.py \
  tests/test_tracking_peaks.py \
  tests/test_correlator_geometry_validation.py \
  tests/test_preflight_cgc_rf_fresh_candidates.py \
  tests/test_cgc_rf_fresh_test_v2.py \
  tests/test_cgc_rf_geometry_aperture_validation.py
```

The JammerTest, TEXBAT, and GNSS-OpenIF adapters are separate transfer audits.
Their current import chain also requires the optional `gcmr`/PyTorch extra;
they are not needed to reproduce the final-static primary endpoint.

## Full 25 MHz RF rebuild boundary

The sealed run removed eight 1,495,000,000-byte intermediate IQ files per
geometry after receiver success, retaining their hashes in each
`retention.json`. Therefore `--resume-before-metrics` is not a rebuild command.
A full rebuild requires 40 generated IQ files over the five geometries,
substantial temporary storage, patched `gps-sdr-sim`, and the patched GNSS-SDR
9-tap receiver. Create a versioned `v2` config with a new output root before
doing that experiment; never point it at
`/home/ubuntu/hdd_data/cgc_temporal_final_static_v1`.
