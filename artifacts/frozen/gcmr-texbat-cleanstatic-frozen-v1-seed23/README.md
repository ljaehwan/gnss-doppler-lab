# TEXBAT cleanStatic-only frozen GCMR result (seed 23)

This directory contains the exact frozen result bundle for the one-model experiment implemented by `scripts/run_gcmr_texbat_cleanstatic.py`.

## Contract

- Train exactly one model from standalone TEXBAT `cleanStatic` only.
- Time roles: train `[30,180)`, epoch selection `[190,260)`, clean reference `[270,330)`, threshold calibration `[340,400)`, sealed normal `[410,470)`.
- Save and reload one checkpoint before opening DS1–DS4 caches.
- Reuse the same checkpoint and strict `score > threshold` rule for DS1–DS4.
- No DS adaptation, recalibration, LOSO donor fitting, or scenario-prefix fitting.

## Frozen outcome

- Best epoch: 40
- Threshold: `134914.8453732542`
- Sealed cleanStatic alarms: `0/119`
- DS1 alarms: `0/922`
- DS2 alarms: `0/913`
- DS3 alarms: `0/914`
- DS4 alarms: `0/255`

This is **not detector success**. The threshold calibration interval contains a PRN-8 reacquisition/false-lock segment around 340–355 s. It produces roughly 9–10 kHz geometry-residual mismatch and inflates the q99 threshold above every DS score. The bundle is preserved as a provenance-valid negative/calibration-failure result.

## Contents

- `model.pt`: exact frozen checkpoint
- `cleanStatic_scores.csv`, `cleanStatic_metrics.json`: sealed clean normal result
- `DS*_scores.csv`, `DS*_metrics.json`: scenario results
- `summary.json`, `provenance.json`: complete frozen contract and source/checkpoint hashes
- `texbat_cleanstatic_frozen_dashboard.png`: result dashboard
- `plot_texbat_cleanstatic_frozen_dashboard.py`: deterministic plot generator
- `texbat_cleanstatic_frozen_dashboard.sha256`: bundle integrity manifest

## Verification

```bash
cd artifacts/frozen/gcmr-texbat-cleanstatic-frozen-v1-seed23
sha256sum -c texbat_cleanstatic_frozen_dashboard.sha256
```
