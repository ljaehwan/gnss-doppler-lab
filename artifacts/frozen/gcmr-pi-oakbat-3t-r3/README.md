# GCMR-PI OAKBAT 3T r3 — Frozen preliminary evidence

This directory preserves the completed GPU campaign `oakbat-3t-streaming-gpu-r3` without overwriting the earlier GCMR-PI artifacts.

## Contract

- **Input:** E/P/L only (3-tap)
- **Predictor:** shared PRN-local GRU; CUDA recorded in `training_summary.json`
- **Fit/calibration:** OAKBAT `cleanStatic` normal roles only
- **Attack scenarios:** OS1–OS4 are inference-only
- **Onset evaluation:** onset 120 s; guard ±10 s; pre `<110 s`, post `>=130 s`

## Important limitation

This is **preliminary negative/mixed evidence, not a successful detector claim**. The q99 Full score had high attack-recording pre-onset FPR (~13–15%) and weak OS1/OS3 post-onset detection. `relation_destruction` did not consistently decrease pair evidence. See `onset_metrics.json` and `scenario_metrics.json`; no threshold was tuned on attack data.

`SHA256SUMS` covers the frozen files. Verify with:

```bash
sha256sum -c SHA256SUMS
```
