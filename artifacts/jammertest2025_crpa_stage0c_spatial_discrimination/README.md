# Jammertest 2025 CRPA Stage-0C spatial discrimination feasibility

Final verdict: `NO_INCREMENTAL_SPATIAL_DISCRIMINATION`.

This experiment asks only whether four-channel spatial relationships add discrimination between spoof/meacon and non-deceptive terrestrial jammer at matched nominal transmit-power strata. It is not a clean-versus-spoof experiment; the released CRPA object has no clean class.

## Frozen design and data binding

- Base: `df98657cf1a50814e169bcd4f91a3b555c9025c0`
- Label-only design commit pushed before IQ access: `67495be08486b5479fbf09ee1b03c9faddb2a077`
- Raw object: 1,398,308,992 bytes, SHA-256 `d869fa20d552288002e4d2a5b6c5d1300083a6348c01a956cd6a34ff232e0a3f`, shape `(42673, 4, 1024)`, `complex64`
- Unique selected snapshots read for features: 3,588 (117,571,584 logical selected-snapshot bytes)
- Primary: Area 1, 30/40 dBm, block 32, 3 folds, 12 unique OOF test blocks, 260 OOF test rows per model
- Sensitivity A: Area 1, 30/40 dBm, block 128, 2 folds, 8 unique OOF test blocks, 324 OOF test rows per model
- Sensitivity B was frozen infeasible before feature access; block 2048 was not executable.

The frozen design, split, feature/model definitions, and seed remain byte-identical to the design commit. The existing NPY was reused read-only with `allow_pickle=False`; it was not copied, redownloaded, or committed. TEXBAT, OAKBAT, Tuni, and Innosense access was zero.

## Primary OOF results

| Model | AUROC | AUPRC | Balanced accuracy | TPR@5% FPR |
|---|---:|---:|---:|---:|
| M0 power | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| M1 single-channel | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| M2 calibration-free spatial | 0.979172 | 0.978907 | 0.784615 | 0.800000 |
| M3 phase-aware diagnostic | 1.000000 | 1.000000 | 0.992308 | 1.000000 |

`M2 - max(M0,M1) = -0.020828` AUROC; its paired block-bootstrap 95% CI lower bound against the baselines is `-0.070779`. M2 therefore does not provide incremental discrimination.

M2 retrained on mismatched, circular-shifted, and Fourier-phase-randomized tuples gives AUROC `0.596864`, `0.500178`, and `0.627396`; actual-minus-control differences are all positive. This shows real simultaneous structure exists, but does not overcome the perfect power and single-channel baselines. The fixed channel permutation leaves M2 unchanged but changes M3 AUROC from `1.000000` to `0.388876`, exposing phase/order sensitivity. One-channel ablation reduces M2/M3 to about chance.

At 30 dBm, all four models have AUROC 1.0. At 40 dBm, M0/M1/M3 remain 1.0 while M2 is 0.924308. Sensitivity A likewise gives M0/M1/M3 1.0 and M2 0.962506, so M2 incremental direction remains negative (`-0.037494`).

## Scope limitations

The realized primary OOF support contains Meac (130) and Prn (130) only. There are no Spoof OOF rows, so Spoof recall is explicitly `NOT_ESTIMABLE_NO_PRIMARY_OOF_SUPPORT`. Official recording IDs, transmitter position, receiver orientation, and array calibration are unavailable. Sample-index blocks reduce leakage risk but do not prove recording-independent generalization.

No clean detector, general spoofing detector, `READY_FOR_WCL`, or recording-independent success is claimed.

## Reproduction and verification

Run the frozen executor with the existing read-only NPY and metadata split root, then verify:

```bash
PYTHONPATH=src python3 scripts/run_jammertest2025_crpa_stage0c.py --raw-npy /path/to/all_crpa_files.npy --split-root /path/to/splits --artifact artifacts/jammertest2025_crpa_stage0c_spatial_discrimination
PYTHONPATH=src python3 scripts/verify_jammertest2025_crpa_stage0c.py
```

All metrics use OOF predictions. Confidence intervals use 2,000 bootstrap replicates at the test-block level. `test_output.txt` and `verifier_output.txt` contain the final validation logs.
