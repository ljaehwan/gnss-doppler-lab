# ACAF-NF Stage-0-R1.1 validity repair

**Scope:** cleanStatic only; no attacks are read, fit, scored, or claimed. All stable valid PRNs are retained from the tracker and bounded raw reconstruction uses deterministic PRN round-robin strata restored to chronological order with non-overlapping raw windows.

**Tracker/remnant formula:** `remnant_chips = aux1_samples * code_freq_chips / fs`; the signed reconstruction uses `remnant_sign * remnant_chips`. Raw samples are s16 little-endian interleaved IQ; each row records `[sample_start, sample_end_exclusive)` and raw bytes `[start*4, end*4)` with raw and tracker-MAT SHA-256 provenance.

**Actual clean validation:** alignment `600` / requested `600`; center `500` / requested `500`; PRNs `[3, 16, 19]` (`3`). True 3x3-center rate is `0.014000` and true raw-center-magnitude/MAT-Prompt Spearman correlation is `0.131249`. R0 delay error is mean absolute reconstructed peak center offset `0.090750` chips; R1 Doppler error is mean absolute reconstructed peak center offset `48.100000` Hz.

**Normal split:** train/calibration/holdout = `1000/500/500`, chronological and raw-range non-overlapping: `True`.

**Gates:** Gate A `FAIL` (n≥500, center≥0.95, Prompt Spearman≥0.90, boundary≤0.05, ≥4 PRNs, dominant≤0.50). Gate B is not evaluated until A passes. Gate C is `INCOMPLETE` because DS4/B0 provenance is unavailable. Verdict: `CENTER_RECONSTRUCTION_INVALID`. This is **not** a physics NO-GO claim and makes **no** detection/performance claim.
