# R2C-GNSS Stage-0 feasibility

## Outcome

The pre-attack input gate fixes the verdict as **`DATA_INVALID`**. This worktree contains neither receiver-produced complex I/Q at all nine taps nor the corresponding TEXBAT raw IQ needed to re-extract it. Its existing nine-tap interface reads `abs_E4 … abs_L4`, which is magnitude-only; prompt I/Q does not restore the phases of the other eight taps. No time-aligned authentic PVT/LOS product is present either. Those facts make real A1/A2 evaluation invalid and geometry-dependent A3/Full evaluation impossible. No attack epoch was scored or used to tune a choice.

The required cleanStatic, cleanDynamic, DS3, DS7, DS8, and diagnostic DS1/DS2 raw files are absent. Repository manifests describe 25 Msps GPS L1 C/A complex IQ, but a manifest and checksum are not source bytes. The six required clean/primary recordings alone exceed the available local storage before receiver outputs. Magnitudes, triangular synthetic attacks, and historical attack-result tables were not substituted.

## Frozen model

For PRN `i`, Stage-0 fits the actual receiver's documented C/A correlation template on its measured tap spacing:

`H0: y_i = a0_i R(τ−τ0_i) + n_i`

`H1: y_i = a0_i R(τ−τ0_i) + a1_i R(τ−τ1_i) + n_i`.

Complex amplitudes are profiled by weighted least squares over supported positive and negative delays. Coincident sources and extrapolated delays are excluded. The statistic is `S_second,i = 2(log p(y_i|H1) − log p(y_i|H0))`, using a profiled noise scale so common complex phase and positive gain cancel. The implementation exposes delays, complex amplitudes, predictions, residuals, likelihoods, boundary status, and identifiability.

The geometry layer converts `Δτ_i` to `Δρ_i = cΔτ_i` and robustly fits

`Δρ_i ≈ −u_iᵀΔp + cΔt`, with `β=[Δx,Δy,Δz,cΔt]`.

It reports rank, singular values, condition number, robust weights, leverage, residuals, and effective support. Invalid geometry returns zero shared evidence. PRN ordering is irrelevant and the PRN count is variable. Actual LOS is mandatory; PRN number is never geometry.

## Frozen experiment and ablations

The configuration freezes chronological cleanStatic-only train/calibration/holdout support, 20 s guards, attack phases, cleanStatic-only `higher` q99/q99.5 thresholds with strict `>`, 3-at-0.5 s sustained alarms, and paired complete 10 s block bootstrap (2,000 repetitions, seed 20260803, no IID fallback). cleanDynamic is evaluation-only. DS3/DS7/DS8 are confirmatory; DS1/DS2 are diagnostic.

The registered comparisons are frozen B0, A1 GLRT, geometry-free A2 robust top-k, analytic-whitened A3 geometry, neural-whitened geometry-free A4, Full neural-plus-geometry, Power-only, and causal Noise-floor-only when available. Each requires its own cleanStatic calibration threshold. B0 retains its historical PRN-holdout/time-overlap and cleanStatic+cleanDynamic calibration limitation; its checkpoint hash is frozen, and it was not retrained or evaluated here.

The analytic nuisance covariance and compact shared MLP are implemented with hard fit-role guards accepting only `normal_train`. The neural model predicts residual mean and diagonal uncertainty from numeric causal conditions; it has no PRN/scenario/recording identity or label interface.

## Controls and unavailable results

Synthetic mechanics-only tests cover gains 0.5–2.0, global phase, AWGN, independent-delay multipath, complex second-source injection with signed delay, consistent geometry, and delay/LOS relation destruction. These validate invariance and fail-closed mechanics only. They do not replace real attacks, establish cleanDynamic FPR, or support a physics claim. Machine-readable control results and their plot source are in the artifact.

Consequently cleanDynamic FPR, DS3/DS7/DS8 ROC/PR/low-FPR pAUC and alarms, DS1/DS2 diagnostics, all real thresholds, all bootstrap CIs, Power-only/Noise-floor-only comparisons, and Full-versus-B0/A1/A2/A4 improvements are explicitly unavailable. The decision is not `NOT_SUPPORTED`: valid inputs do not exist to test the hypothesis.

This differs from existing CAF monitoring (no 2D delay–Doppler surface), LASSO/CCAF decompositions (explicit two-component profile likelihood here), and PD-ML (no power-delay classifier or attack supervision). Nine taps have no Doppler axis, so no Doppler claim or pseudo-Doppler feature is made. A later raw-IQ 2D model is **not justified** by a `DATA_INVALID` Stage-0 result.

## Reproduction and publication status

Run with the active interpreter:

```bash
python scripts/run_r2c_gnss_stage0.py
python scripts/verify_r2c_gnss_stage0.py --write-result
python -m pytest tests/test_r2c_gnss.py
```

The retry used sandbox bypass solely because the first Codex attempt failed before any command with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`; Hermes independently confirmed the frozen worktree was clean. Provenance records this fallback. Codex commits locally on `research/r2c-gnss-stage0` but does not push or merge; remote SHA and `0/0` remote-branch status therefore remain pending Hermes verification/publication. The final handoff reports the local commit, comparison with `origin/main`, tests actually run, and main-worktree observation without modifying main or protected worktrees.
