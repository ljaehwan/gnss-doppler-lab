# NC-TOPI Stage-0 preregistration

**Status:** frozen before detector results. This stage implements contracts and
mathematics only. It contains no full experiment runner and no attack scoring.

## Scientific lineage and admissible evidence

Exact legacy actual/predicted peaks are unavailable for cleanStatic and
DS1/DS2/DS3. A tangent recovered from residuals alone is **non-identifiable**
and is forbidden. The later primary runner must regenerate every scenario from
the provenance-fixed canonical complex 9-tap NPZ, using exactly the B0 prompt
normalization, windowing, and GRU below. It must **never retrain** B0. Exact
legacy DS7/DS8 residuals are positive-control diagnostics only.

Canonical immutable inputs are:

- cleanStatic: `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/exports/cleanStatic.npz`, SHA-256 `fcd1d378c28e79fe4a550b65fc1208cde3c8fb334db11406a07fed4d90fba237`
- DS1: `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds1.npz`, SHA-256 `b24d947c83890dbfa1c801bfbcb72e1fd192dd66509e927eb5afb8118902b072`
- DS2: `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds2.npz`, SHA-256 `dae0f245cbb107febd220c6de33b9a279a2bad356cb0ba772daf9418bc75d7c9`
- DS3: `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds3.npz`, SHA-256 `38eb5842dfec306d99bf0c5d61df6cffcb6faa25ed63721cafa8e3c3776f9b3e`
- DS7: `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/exports/ds7.npz`, SHA-256 `d0e6da4e27d51e3e96abf2ef7786501124072f28667671e4e40da756eb35f3c8`
- DS8: `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cmte-a2-ds8-complex-d67f813/exports/ds8.npz`, SHA-256 `d1973fa150b7b4e7359df4827f36ce60289f206e9db11c1ac2bc1fd33a0df533`

The tap order is `E4,E3,E2,E,P,L,L2,L3,L4`. The physical coordinates, verified
from GNSS-SDR source, are stored explicitly as
`[-.5,-.375,-.25,-.125,0,.125,.25,.375,.5]` chips. Runtime code must not infer
them from an assumed spacing.

## Frozen B0 and epoch identity

- Checkpoint SHA-256: `f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b`.
- Prompt-relative magnitude ratio epsilon: `1e-6`; 1 s window, 0.5 s stride,
  minimum four raw epochs; float32 standardizer; sequence length 12.
- PRN identity is forbidden. Prediction becomes available at target
  `source_end`.
- B0's historical training-overlap limitation is explicitly disclosed. It is
  frozen rather than repaired by retraining.
- B0 and every ablation use the same Stage-0 epoch identity and valid mask.
  Each detector, aggregator, and quantile combination gets its own threshold
  from clean calibration. Exact identity joins, not nearest-time joins, apply.

## Fit, split, phase, and timing rules

All fitting—standardization, covariance, regression, and thresholding—uses only
cleanStatic roles. Attack rows and normal holdout rows can never fit anything.
Split by complete source support:

- train: `source_end <= 300`;
- calibration: `source_start >= 320 AND source_end <= 400`;
- holdout: `source_start >= 420`.

The 20 s gaps guarantee no support overlap. Attack onsets are 100 s for
DS1/DS2/DS3 and 110 s for DS7/DS8. Stable-pre requires
`source_start >= 30 AND source_end <= onset-20`; transition is every otherwise
pre-post epoch; post requires `source_start >= onset`; persistent requires
`source_start >= onset+40`. Delay uses availability `source_end` and onset.
Already-alarming stable-pre status is always reported.

A sustained alarm is three contiguous 0.5 s epochs within the same physical
recording and without a time gap. A recording/time gap resets the run. The
reported alarm time is the availability time of the third epoch.

## Geometry and scores

`Sigma` is sklearn Ledoit-Wolf fit on clean-train `residual_raw` only. Its
floor is `epsilon = 1e-8*trace(Sigma)/dim`; `W` is a pseudoinverse. Projection
uses relative ridge `1e-8` and pseudoinverse `rcond=1e-10`, and reports
coefficients, fit, orthogonal residual, all W-energies, ranks and condition.
Tangents come from the regenerated predicted peak: amplitude, numerical first
derivative on the physical coordinates, and optional second-derivative width.
Prompt-relative normalization attenuates global amplitude, so that tangent is
a declared nuisance/ablation direction, never residual-derived evidence.

B0 is standardized residual RMSE. NC-TOPI is orthogonal residual energy; TOPI,
width tangent, alternate aggregator, and quantile are diagnostics/ablations.
The primary is **q99 NC-TOPI median only**. Median is the primary aggregator.
Top25 mean is the mean of the largest `ceil(.25*n)` valid PRN scores; it is
permutation invariant, supports variable `n`, and records tracked count.
Diagnostics **cannot rescue the primary**.

Threshold quantiles use NumPy's `higher` rule and alarms use strict `>`.
Low-FPR performance is sklearn standardized **McClish** partial AUC with
`max_fpr=.05`.

Uncertainty uses paired, gap-safe, nonoverlapping, complete 10 s physical
blocks, 2000 repetitions, fixed seed 20260803, and percentile95 intervals.
There is **no IID fallback**. Point estimates use every eligible common epoch,
including eligible epochs outside complete resampling blocks.

## Causal IQ conditioner

The later runner reads raw int16 IQ at 25 MHz and takes one 10 ms block every
0.5 s. Features are log power, robust noise-floor scale, spectral flatness,
and lag-1 autocorrelation magnitude. For a target, strict as-of context uses
only blocks with `block_end <= min(target source_start)` and the latest four
past blocks; no current overlap or future block is legal.

Predictors are standardized by clean-train median/IQR. Fixed Huber regression
uses `epsilon=1.35`, `alpha=1e-4`, and `max_iter=1000` on clean-train PRN rows.
PRN, scenario, and onset are forbidden features. Predicted scale has an epsilon
lower bound only and is capped at the clean-calibration predicted-scale higher
q995. The time-shuffle control uses seed 20260803 within clean train before
regression and is never constructed on attacks.

## Synthetic physics preregistration

Second-peak relative powers are `[.05,.1,.2,.4,.8]`; physical separations are
`[.0625,.125,.25,.375,.5]` chips. Interpolation is on physical coordinates and
synthetic grids cannot select a threshold. Equal-RMSE tangent versus
W-orthogonal tests use 100 deterministic trials. Nuisance controls are signed
amplitude `+/-[.02,.05,.1]`, signed shifts `+/-[.01,.025,.05]` chips, and noise
scales `[1,1.25,1.5]`.

The exact physics pass grammar is:

1. Equal RMSE: B0 relative difference `<=1e-8`; median tangent/orthogonal TOPI
   ratio `<=.05`; orthogonal preservation `>=.95`.
2. Second peak: at each separation `>=.25`, Spearman power rho `>=.8`; at each
   power `>=.2`, separation rho `>=.8`.
3. Nuisance: median normalized TOPI increase is less than B0 increase for
   amplitude and small shift. Noise is reported and requires TOPI increase
   `<=` B0 increase to pass.

## Frozen decision grammar

Attack scenario improvement passes iff standardized pAUC point delta is `>0`,
**OR** both sustained delays are finite and NC is at least 0.5 s earlier. The
CI criterion passes iff paired pAUC-delta percentile95 CI lower bound is `>0`.
The shuffle criterion passes iff actual NC mean pAUC gain over TOPI is greater
than shuffled gain and actual gain is `>0`.

- **GO:** all frozen criteria pass, evaluated on q99 NC-TOPI median only.
- **NO-GO:** any physics criterion fails, **or** clean holdout FPR is `>5%`,
  **or** stable-pre fails in at least three scenarios, **or** attack improvement
  passes in at most one scenario, **or** no scenario has a positive CI lower
  bound.
- **INCONCLUSIVE:** every outcome not classified GO or NO-GO by the exact rules
  above.

This grammar is frozen now. TOPI, top25 mean, q995, width tangent, DS7/DS8
legacy residual positive controls, and any diagnostic cannot alter or rescue
the primary GO result.
