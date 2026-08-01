# CLIF-IP Phase 1 feasibility — No-Go (provenance gate)

## Decision

**No-Go: the requested cross-layer model comparison was not run.**

The prerequisite is not merely that B0 and M1 have a numerically matching relative-time grid. The project must prove that the paired RF and peak evidence comes from the **same immutable raw-IQ recording** and shares one recording-start/sample-time anchor. The current OAKBAT pairing satisfies the former time-grid candidate but does not record an M1 raw-IQ hash or start-sample anchor. TEXBAT raw/derived evidence is not available in the tracked repository for an equivalent paired verification.

Implementing C0–C5, lag selection, a shuffled-time test, ROC/PR metrics, or a Go conclusion from those inputs would turn an unproven scenario-label join into a cross-layer physical claim. This branch deliberately fails closed instead.

## Actual B0 evidence

- Input: PRN-local, prompt-normalized 9 correlation taps `E4,E3,E2,E,P,L,L2,L3,L4`.
- Predictor: frozen GRU takes `[batch, 12, 9]` and predicts the next 9-tap target.
- Before scalar/binomial-tail aggregation: a signed standardized 9-tap prediction residual exists; the standard B0 scorer persists PRN RMSE/MAE, not the signed vector.
- Time: 1.0 s tracking window, 0.5 s stride; target `window_start_s`, causally available at the target window end.
- Normalization: frozen training-split feature mean/std.
- Threshold: cleanStatic+cleanDynamic PRN quantiles → exact binomial-tail surprise → causal EWMA → clean q99.

Code evidence: `scripts/train_prn_node_gru.py`, `scripts/score_texbat_prn_node_gru.py`, `scripts/eval_btail_support_gate.py`, `configs/detectors/texbat_btail_gate_v1.json`.

## Actual M1 evidence

- Input: pre-correlation raw interleaved int16 I/Q.
- Actual features: I/Q/power statistics, phase-increment/coherence, PSD entropy/flatness/band powers, amplitude statistics, and complex autocorrelation.
- Before scalar score: PCA AR innovation vector; its AR-RMSE and a robust feature-level drift vector are the scalar-score inputs.
- Time: 10 ms RF block at 0.5 s stride, recording-relative `window_start_s`.
- Normalization: fit-prefix mean/std before PCA; median/MAD residual/level scaling.
- Threshold: per-recording authentic-prefix causal-EWMA q99.

Code evidence: `scripts/iq_noise_continuity_detector.py`, `docs/RAW_IQ_NOISE_CONTINUITY_SECOND_METHOD.md`.

## Alignment result

For `cleanStatic`, `os2`, `os3`, `os4`, the existing OAKBAT manifest paths exist, file hashes match their manifest entries, and both exported timestamp grids are 0.5 s. This proves only **candidate relative-time alignment**.

It does not prove the required raw-data identity:

- B0 morphology provenance is strong, but the pair record does not expose an immutable `raw_iq_sha256` for direct cross-modal equality.
- M1 floor evidence has an output-file SHA and `floor_scenario` string only; it lacks the source raw-IQ SHA-256, recording-start sample/offset, and extractor provenance tied to B0.
- Therefore `window_start_s=120.0` cannot be asserted to mean the same physical samples in both layers.

`data_alignment_report.json` is the machine-readable gate evidence. No attack label, onset, lag, feature weight, threshold, or metric was used to make this decision.

## Leakage and generalization checks

- B0’s frozen calibration is clean-only, but OAKBAT cleanStatic training is chronological partitions of one recording, not a recorder/run holdout.
- M1 is per-recording prefix self-calibration; independent cleanStatic/cleanDynamic recorder-stability evidence is not present in this pairing.
- M1 raw-IQ features can encode AGC/power/front-end/transmitter/campaign identity. Without paired raw provenance and recorder-group controls, cross-layer gains could be recorder/capture shortcuts.

## What is not claimed

- No claim that RF innovations causally produce peak innovations.
- No claim that a cross-layer relationship exists or does not exist.
- No ROC-AUC, PR-AUC, alarm delay, FPR, or comparison of C0–C5: those are **not estimable under the failed pair-provenance gate**.
- No temporal-alignment destruction or lag-sensitivity result: a shuffle test on an unproven join would not validate a physical temporal relation.

## Required evidence before reopening Phase 1

For every B0/M1 pair, create an immutable pairing manifest containing:

1. identical `raw_iq_sha256`, byte size, sample rate, format, and recording/run ID;
2. raw sample-index or UTC/GPST recording-start mapping for both extractors;
3. B0 tracking feature/receiver-manifest hashes and M1 extractor/version/config hash;
4. whole-recording/run group splits, clean-only calibration sets, and separate OAKBAT/TEXBAT contracts;
5. M1 recorder/fingerprint ablations (power/AGC, PSD, phase-only, autocorrelation-only).

Only after this gate passes should C0–C5 be fit on normal-only data and evaluated with causal lag/degradation tests.
