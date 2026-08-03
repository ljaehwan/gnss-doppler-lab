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
Every public fit/calibration path requires the immutable typed
`FitProvenance(scenario, role, tuple[EpochIdentity])`. Covariance and conditioner training
require `cleanStatic normal_train` and derives a fit digest over provenance, predictors,
and target; conditioner cap and detector threshold require
`cleanStatic normal_calibration`, record an identity digest, and calibration
identities must be disjoint from conditioner-fit identities. The shuffled control
is a deterministic permutation strictly within typed clean train. Array-only
quantile/covariance math helpers are low-level and are not primary fit entry points.
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
recording and without a time gap. Recording IDs are nonempty, duplicate
`(recording,time)` rows are rejected, and input is sorted within each recording
before independent run evaluation. A recording/time gap resets the run. The
reported result is the globally earliest third-alarm availability and delay;
already-alarming stable-pre status is audited separately for every recording.

## Geometry and scores

### Coordinate and identity contract

Every score starts from a validated `PeakPredictionPair`. It requires both the
actual and predicted raw prompt-relative magnitude-ratio peaks, their full epoch
identity, explicit immutable physical coordinates, the frozen standardizer
standard deviation, and the B0-standardized residual. Residual-only construction
is forbidden. Runtime validation enforces
`residual_raw = actual_raw - predicted_raw` and
`residual_standardized ~= residual_raw / standardizer_std`; there is no default
`input_kind` that can weaken this rule.

B0 is exactly `sqrt(mean(residual_standardized^2))`. Geometry never uses that
standardized coordinate: covariance, tangent projection, and all TOPI energies
use raw prompt-relative-ratio space. The primary per-PRN formula is exactly
`S_perp = r_perp.T W r_perp`. Total, tangent, perpendicular, and cross energy
are always returned, and `total ~= tangent + perpendicular + cross` is checked.
Negative quadratic energy is not blindly clamped: only tiny floating-point
negative values are tolerated; a material negative value fails closed.

Covariance is Ledoit-Wolf fit only on `cleanStatic` `normal_train`
`residual_raw`, derived internally from a validated `PeakPredictionPair` sequence
or `ResidualBatch.from_pairs`. Public primary fit never accepts a raw ndarray, so
standardized residuals cannot be mislabeled. Its audit binds the raw-space tag,
pair identity count, provenance digest, and pair identity digest. Calibration,
holdout, mixed, missing-provenance, or attack rows are rejected. `W` must be
symmetric positive semidefinite.

The primary projection is a whitening/SVD W-metric projector. Effective rank uses
the exact same explicit singular-value cutoff as projection. It returns that rank,
cutoff, machine-epsilon-scaled orthogonality tolerance, retained-span and full-span
defects, and verifies energy decomposition. If a small direction is dropped, the
result is labeled `retained_effective_tangent_span` and never falsely claims
full-span orthogonality. A retained-span defect beyond tolerance fails closed.
Ridge exists only in the separately named diagnostic path and is marked
`ridge_diagnostic_not_orthogonal`; it cannot be represented as primary.

The frozen primary basis is amplitude plus shift, with width defaulting to
JSON boolean `false`. `TangentBasis` is a factory-only immutable provenance
object: `primary_amp_shift` binds peak identity, coordinates, and `predicted_raw`
digests plus exact names/matrix. Primary score accepts only the matching
`PeakPredictionPair + TangentBasis`; raw, arbitrary, residual-generated, width,
or mismatched bases fail closed. The separate `width_diagnostic` builder and
score result are explicitly labeled diagnostic and cannot enter primary. Because
inputs are prompt normalized,
the requested amplitude column is a **normalized-shape scale direction**, not
physical receiver global gain. It is not physical receiver global gain. Physical receiver global gain was removed by
prompt normalization and cannot be claimed as an identified nuisance. The
user-requested amplitude ablation remains reported under that caveat.

The IQ Huber target is `log(max(S_perp, energy_epsilon))`. Huber predicts log
energy; detector scale is `exp(predicted_log_energy)`, capped at the
clean-calibration higher q995 predicted scale. NC-TOPI is
`S_perp/max(capped_scale, energy_epsilon)`: scale, not scale squared, and no
square root. PRN scores are aggregated to median (primary) or top25 mean per
epoch, and every detector/aggregator has its own clean-calibration threshold.
The primary remains **q99 NC-TOPI median only** and diagnostics cannot rescue it.

Threshold quantiles use NumPy's `higher` rule and alarms use strict `>`.
Low-FPR performance is sklearn standardized **McClish** partial AUC with
`max_fpr=.05`.

### Identity, timing, IQ, and uncertainty

The immutable `EpochRecord` requires physical recording ID, scenario,
PRN-or-event ID, availability time, source start/end, valid true bool, and label.
Its fixed full key is the typed `EpochIdentity(physical_recording_id, scenario,
prn_or_event_id, target_index, availability_time_s)`, so event-level evaluation is supported without weakening
identity. The primary epoch join accepts records and exact score maps, requires
full identity set equality, and makes source interval, label, and valid equality
mandatory after alignment. It never silently intersects. A separately named
common-mask diagnostic reports all exclusions.

Sustained alarm evaluation requires recording IDs and a post-eligible mask.
Runs reset at recording changes, cadence gaps, and transition/non-post rows.
Delay is the third alarm's availability `source_end` minus onset; a stable-pre
mask reports already-alarming status separately.

IQ `target_groups` and `block_groups` are mandatory nonempty strings and their
group sets must match exactly. Duplicate block ends within a group and unsorted
group time are rejected; cadence and gaps are audited, a valid history is
contiguous, and blocks can never cross groups.
There is no cross-recording default.
The runner reads raw int16 IQ at 25 MHz and takes one 10 ms block every 0.5 s.
The four frozen features are exactly `log_power`, `log_noise_floor_scale`,
`spectral_flatness`, and `lag1_autocorr_magnitude` in that order. Predictor normalization is the
clean-train median/IQR; PRN, scenario, and onset remain forbidden features.

Primary uncertainty is paired pAUC-delta bootstrap, not a generic statistic
bootstrap. It requires labels, paired scores, recording IDs, times, and
`max_fpr`. Complete nonoverlapping 10 s blocks are formed only within a
recording, label stratum, and contiguous cadence run. Negative and positive
block pools are resampled separately while retaining paired scores and labels
at the same indices. The point estimate uses all eligible epochs. Class-deficient
input or fewer than two blocks in either stratum returns an explicit unavailable
result and reason. There is **no IID fallback**; valid replicate count is returned.

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

The machine evaluator accepts only q99 NC-TOPI median primary evidence. Before
applying any criterion it validates evidence domains: FPRs and pAUC points/means
must be finite in `[0,1]`; pAUC deltas and both CI endpoints must be finite in
`[-1,1]` with lower `<=` upper; delays must be finite and nonnegative or explicit
censored `None`; every scenario mapping must have exactly DS1/DS2/DS3/DS7/DS8;
and physics flags must be true booleans. Malformed or missing evidence always
returns **INCONCLUSIVE** with `validation_errors`, never GO or NO-GO. Its criteria
use these exact operators:

1. `c1`: clean-holdout NC FPR `<= .02`.
2. `c2`: NC clean-holdout FPR minus B0 clean-holdout FPR `<= .01`.
3. `c3`: all five scenario stable-pre FPR values are strictly `< .05`.
4. `c4`: at least three scenarios have pAUC delta strictly `> 0`, **or** both
   delays finite and `B0_delay - NC_delay >= .5`.
5. `c5`: at least two paired pAUC CI lower bounds are strictly `> 0`.
6. `c6`: equal-RMSE physics passes.
7. `c7`: second-peak physics passes.
8. `c8`: actual NC mean pAUC gain over TOPI is strictly greater than shuffled
   gain and actual gain is strictly `> 0`.

`GO = c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8`.

**NO-GO** applies iff equal-RMSE is explicitly false, second-peak is explicitly
false, clean NC FPR is strictly `> .05`, observed stable-pre failures are at
least three, all five improvement outcomes are known and the pass count is at
most one, or all five CI outcomes are known and the positive count is zero.
NO-GO triggers have precedence. Missing or censored mandatory evidence prevents
GO but is not invented as a failure; absent an explicit NO-GO trigger the result
is **INCONCLUSIVE**. Only explicit `None` represents a censored delay; NaN,
infinity, and negative delay are invalid evidence and produce `validation_errors`.
Width, top25, q995, legacy residual controls, and all other
diagnostics cannot alter or rescue this result.


## Sealed primary-state and strict schema boundary

Stage-0 primary code has no raw-`W` score entry point. Covariance fitting emits a
factory-only `CovarianceFit` whose seal covers `Sigma`, `W`, unfloored covariance,
the complete raw-space audit, and the `ResidualBatch` input digest. The batch
digest covers `residual_raw`, ordered pair content digests, and the
`prompt_relative_ratio_raw` tag. Every tangent/score consumer recomputes these
content seals, so `object.__setattr__` mutation fails closed. `TangentBasis` also
binds the exact pair content digest and covariance-fit seal. Pure TOPI uses
`produce_topi_scores`; primary NC-TOPI uses `produce_nc_topi_scores` and requires
a real fitted, q995-calibrated, seal-valid `RobustConditioner`—`None` and duck
types are rejected.

Every fit row and peak pair uses
`EpochIdentity(recording_id, scenario, prn, target_index, availability_time_s)`.
The first three fields are nonempty unpadded strings, target index is an integer
excluding bool, and availability is a finite Python/NumPy real scalar excluding
bool. Identity and basis digests are canonical and type-tagged. Primary pairs
have exactly nine float64 taps and their coordinates must equal the frozen
explicit GNSS-SDR vector with zero tolerance; merely increasing or nearly equal
coordinates are invalid.

The conditioner's only schema is exactly
`('log_power','log_noise_floor_scale','spectral_flatness','lag1_autocorr_magnitude')`.
Fit and transform width are exactly four. Reordered, missing, padded, case-aliased,
and case/strip-normalized forbidden PRN/scenario/onset names fail closed. The
shuffle helper validates and preserves the exact schema and typed clean-train
provenance while permuting only the target.

`ThresholdCalibration` is factory-only and sealed over value, score digest,
detector, aggregator, quantile, and clean-calibration identity provenance. The
public primary `strict_alarms` accepts only that typed object and requires q99
NC-TOPI median; raw scalar comparison is private/diagnostic. Decision evidence
is type-checked before conversion: only Python/NumPy real scalars are accepted
for numeric evidence (never bool, string, list, ndarray, or arbitrary object),
and physics flags accept only `bool`/`np.bool_`. Malformed evidence always yields
**INCONCLUSIVE** with `validation_errors`.
