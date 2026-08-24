# Quality-conditioned tail v0 frozen result

This is a provenance-preserving negative result.  Causal observed-score
continuity age (`<5 s`, `5-20 s`, `>=20 s`) did not materially improve the
clean-calibrated multi-PRN binomial-tail detector.

Key outcomes:

- TEXBAT pooled calibration: no TPR or FPR gain on DS1-DS4; DS2 was 0.5 s later.
- TEXBAT cleanStatic to cleanDynamic: OOD clean FPR changed only from 92.78% to
  90.03%, so the detector remains non-deployable.
- OAKBAT: no scenario improved; OS3 pre-FPR and post-TPR both worsened.

The matched global implementation recovered the frozen TEXBAT event threshold
`4.169877716047041` exactly.

Tracked evidence:

- `summary.json`: aggregate machine-readable report and input hashes.
- `comparison.csv`: matched detector metrics.
- `<campaign>/calibration.json`: frozen normal-only thresholds and input hashes.
- `<campaign>/summary.json`: per-scenario metrics and provenance.

Large reproducible per-event CSVs are generated locally but are intentionally
not tracked.  See `docs/results/QUALITY_CONDITIONED_TAIL_V0_NEGATIVE_RESULT.md`
for interpretation and reproduction commands.
