# Jammertest 2025 CRPA Stage-0D true-Spoof discrimination

## Result

Final verdict: `SPOOF_EVALUATION_INVALID_NO_RECEIVED_POWER_OVERLAP`.

The exact Area 1, 40 dBm comparison contains 124 Spoof and 164 Prn snapshots. Their received-power supports are disjoint by 21.714379367 dB: Spoof spans -53.189203 to -44.914608 dB and Prn spans -23.200229 to -22.447872 dB. Track-A M0 therefore has AUROC 1.0. Fold-local maximum-cardinality/minimum-cost matching retained zero train and zero test pairs at every frozen caliper (0.10, 0.25, 0.50, and 1.00 dB). The primary 0.25 dB gate failed all four conditions, so Track B and all matched spatial scoring were not executed.

This is a provenance-limited invalid evaluation, not evidence for or against a general GNSS spoof detector. It does not establish clean-versus-spoof detection, recording-independent generalization, or readiness for WCL.

## Frozen lineage

- Base: `c71b225e3c07f28d685666a977dae94c4cb03214`
- Design freeze: `33c3c6924f2a7f42ca964e1bd136239cacaea04c`
- Power-match freeze: `aa1859b411b65a2599325642b7c0d4a9abf6558c`
- Branch: `research/jammertest2025-crpa-stage0d-true-spoof-discrimination`

The user supplied 11 explicit class blocks (five Spoof blocks 231-235 and six Prn blocks 95-100). This explicit inventory takes precedence over the later prose phrase describing 10 blocks. All 288 snapshots appear exactly once in OOF test: Spoof 124 and Prn 164, with all 11 enumerated blocks covered. Train, guard, and test are pairwise disjoint in every fold.

## Track A diagnostics

| Model | AUROC | AUPRC | Balanced accuracy | TPR at 5% FPR | AUROC block-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| M0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | [1.000000, 1.000000] |
| M1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | [1.000000, 1.000000] |
| M2 | 0.912618 | 0.927940 | 0.866935 | 0.822581 | [0.697843, 1.000000] |
| M2R | 0.222364 | 0.416640 | 0.389359 | 0.185484 | [0.000233, 0.662162] |
| M3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | [1.000000, 1.000000] |

These values are diagnostic only. M0 and M1 already separate the recordings perfectly. M2R collapses after train-only cubic power residualization. Although each M2 fold AUROC is 1.0, pooled OOF M2 AUROC is 0.912618 because probabilities from independently calibrated fold models are not globally rank-equivalent.

## Destruction and stress diagnostics

M2 retrained AUROC is 0.795830 after within-class/power channel mismatch, 0.808419 after independent circular shifts, and 0.720446 after independent Fourier phase randomization. Actual-trained cross-application gives 0.443450, 0.437328, and 0.431009. None can override the failed power-overlap gate. Fixed channel permutation leaves M2 and M2R AUROC exactly unchanged; global gain and phase invariance contracts pass. One-channel ablation reduces M2 to 0.426682.

## Data and access

The only raw object was opened read-only with `mmap_mode="r"` and `allow_pickle=False`. Its binding is size 1,398,308,992 bytes, SHA-256 `d869fa20d552288002e4d2a5b6c5d1300083a6348c01a956cd6a34ff232e0a3f`, shape `(42673,4,1024)`, dtype `complex64`. It was not copied, downloaded, modified, or added to Git. TEXBAT, OAKBAT, Tuni, and Innosense access is recorded as zero.

## Reproduction without raw access

Run the compact verifier with:

```bash
PYTHONPATH=src python3 scripts/verify_jammertest2025_crpa_stage0d.py
```

The scientific details are in `aggregate_metrics.json`, `matched_metrics.json`, `destruction_results.json`, `paired_bootstrap_results.json`, `power_residualization_audit.json`, `invariance_results.json`, and `final_verdict.json`.
