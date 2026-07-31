# GCMR v2 honest comparison

**Policy:** No matched-superiority claim is supported. B0/M1 matched re-evaluation remains pending. The frozen contract is pre availability ≤90 s, transition 90–110 s, post window start ≥110 s, and availability = window end.

| Method | Evaluation scope | Threshold | Audited result | Interpretation |
|---|---|---:|---|---|
| GCMR v1 original | cleanStatic-frozen → DS1–DS4 | 134914.845 | All DS multi/combined alarms 0 | Original threshold failure |
| GCMR v1 recalibrated | post-hoc diagnostic | 1.7693 | Diagnostic only | Not final performance; do not compare as frozen result |
| GCMR v2 single | cleanStatic-frozen → DS1–DS4 | τ 8.157904927719994 | Post DS1 49/702; DS2 149/693; DS3 122/694; DS4 11/35. Pre DS1 0/119; DS2 2/119; DS3 1/119; DS4 3/119 | Single-PRN tracking fault diagnostic, **not** final multi-PRN spoof alarm |
| GCMR v2 multi | cleanStatic-frozen → DS1–DS4 | R 0.09195067847110018 | Sealed 0/119. Post DS1 0/702; DS2 138/693 (19.91%, first 111.0 s); DS3 37/694 (5.33%, first 120.5 s); DS4 0/35. Pre all 0 | Final frozen v2 result |
| B0 frozen tap9 GRU | current B0 artifact | — | No matched result reported | Current B0 artifact is not matched to this DS1–DS4 contract; re-evaluation pending |
| M1 dense raw-IQ | per-scenario 0–90 s self-fit; DS1 unavailable | — | Descriptive: DS2 delay 8.85 s/post 49.58%; DS3 delay 19.71 s/post 34.32%; DS4 onset 110 s, delay 3.99 s/post 77.85%; pre ≈0.09% | Incomparable: not cleanStatic-frozen external; re-evaluation pending |

## Verdict

GCMR v2 final multi-PRN spoof alarms occur on DS2 and DS3, but not DS1 or DS4. Single-PRN tracking fault flags must not be relabeled as final multi-PRN spoof alarms. These data do **not** establish superiority over B0 or M1 because their evaluation contracts are unmatched.

Failed v2 artifacts remain preserved separately and are excluded from this table and all reported results.
