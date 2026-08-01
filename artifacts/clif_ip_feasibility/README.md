# CLIF-IP Phase 1 feasibility

## Verdict — **No-Go for Phase 2**

This is an observational OAKBAT-only Phase 1 analysis, not proof of physical causality. It evaluates the frozen B0 predictor and an M1-style PCA/AR raw-IQ innovation constructed from existing paired RF-floor feature files. The result does **not** justify a GRU/cross-attention CLIF-IP implementation.

Reasons:

1. There is no TEXBAT paired derived evidence in this run, and OAKBAT has only one clean recording split chronologically rather than recorder/run holdout.
2. Clean-validation lag association is weak and inconsistent: the largest absolute correlation is diagnostic *future* lag -0.5 s (`0.133`), while causal lags are `-0.158` (0 s), `-0.147` (0.5 s), `0.006` (1 s), `0.031` (1.5 s). This is not evidence for a stable causal M1→B0 delay.
3. Alignment destruction changes the simple continuous cascade proxy, but its direction changes by scenario: aligned minus shifted is positive for os2/os4 and negative for os3. That is not a reproducible cascade signature.
4. C4 lagged score is highly discriminative in the available OAKBAT scenarios but has unstable behavior and no independent recorder/dataset confirmation. C5 does not reliably exceed the simpler alternatives.
5. Existing M1 RF-floor pair records lack a modality-pair raw-IQ SHA/start-sample anchor. Relative epoch alignment is operationally used here, but same-sample provenance remains a limitation.

## Actual evidence

### B0
- Frozen PRN-local GRU: prior 12 epochs of 9 prompt-normalized correlation taps predict the next 9-tap vector.
- `B0` evidence retained before binomial tail: signed 9-dimensional residual, per-PRN RMSE, event median/q90/q99 RMSE, residual energy, tracked PRN count, and q99 exceedance fraction.
- B0 input does **not** contain PRN identity.
- Peak window: 1.0 s, 0.5 s cadence; target score is causally available at window end.

### M1
- Existing raw-IQ floor features only: I/Q/power, phase increment/coherence, PSD entropy/flatness/bands, amplitude statistics, complex autocorrelation.
- M1 innovation: clean-trained PCA representation followed by AR(6) residual vector, AR-RMSE, and robust feature-level drift.
- RF block: 10 ms, 0.5 s cadence. No B0 feature was used to construct M1.

## Alignment and normal-only protocol

OAKBAT `cleanStatic`, `os2`, `os3`, `os4` have B0 and M1 0.5-second relative grids. B0 availability is its tracking-window end; M1 availability is raw block end; rows are joined only by matching recording-relative epoch.

Normal-only fitting:
- B0 frozen checkpoint unchanged.
- `cleanStatic [0,240] s`: PCA, AR, feature normalizers, joint covariance, ridge predictor, full-score covariance.
- `cleanStatic [250,330] s`: validation calibration (`q99`, `q99.5`, and reference tails).
- Attack rows are shifted beyond 1000 s internally before score transforms, so they cannot enter any fit/validation interval.
- No random window split, attack-label lag selection, attack-derived weight, or attack-derived threshold is used.

This is **not** a recorder-level generalization split. It cannot exclude M1 front-end/capture/AGC fingerprint shortcuts.

## Compared scores

- C0: B0 evidence distance.
- C1: M1 AR-RMSE + level drift distance.
- C2: clean-tail mean, max, Fisher combination.
- C3: Ledoit-Wolf same-time B0/M1 joint distance.
- C4: Ridge prediction of current B0 evidence from causal 0–3 s M1 history; score is residual Mahalanobis distance.
- C5: covariance score of B0, M1, cross-layer residual, and causal cascade.

See `fusion_comparison.csv` and `scenario_metrics.csv`. ROC/PR use each attack recording's pre-onset region (≤110 s) versus post-guard attack region (≥130 s); normal FPR is fixed from clean validation q99.

## What may be claimed

- It is feasible to extract frozen B0 residual evidence and M1 AR innovation on a shared 0.5-second OAKBAT relative grid.
- Simple temporal correspondence affects a cascade proxy while leaving each marginal series unchanged.
- The current evidence is insufficient to claim a stable, transferable RF-to-peak relationship.

## What may not be claimed

- Physical causation from RF innovation to tracking peak innovation.
- Cross-dataset or cross-recorder robustness.
- Superiority of Full CLIF-IP over B0/M1/simple fusion.
- Any result for TEXBAT or OAKBAT os1.

## Outputs

- `*_epoch_scores.csv`: time-resolved B0, M1, cross-layer and cascade scores.
- `plots/*_timeline.png`: B0/M1/cross/cascade timelines.
- `plots/*_aligned_vs_shuffled.png`: circular-time-shift destruction diagnostic.
- `plots/lag_sensitivity.png`: clean-only lag diagnostic.
