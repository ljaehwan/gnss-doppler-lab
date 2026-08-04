# NC-TOPI Stage-0B IQ-shortcut audit preregistration

**Status:** contract frozen before reading attack outcomes. This document and
`configs/nc_topi_stage0b_audit.json` are the complete Stage-0B contract. No
production implementation, tests, result artifacts, or metric calculations are
part of this commit.

## 1. Purpose, lineage, and isolation

Stage-0B asks one narrow question: are the frozen NC-TOPI gains supported by the
tangent residual, or can the IQ-conditioned scale alone explain them? It is a
separate audit and can never revise the frozen Stage-0 decision.

The authoritative lineage is:

- parent branch: `origin/research/nc-topi-stage0`;
- parent artifact commit: `6fe5315ca0d71689609895cd3b1366bcfa1b93c1`;
- parent generation source commit: `c94af28795d03a91e2f4c0faa74eb19a983ed82e`;
- immutable parent artifact: `artifacts/nc_topi_stage0`;
- new append-only output: `artifacts/nc_topi_stage0b_audit`.

The audit must refuse to overwrite an existing Stage-0B directory. It must not
modify, delete, reinterpret, or republish any file below
`artifacts/nc_topi_stage0`. Attack files are score-only and remain inaccessible
to all fits, calibration, clamp selection, and model/decision selection.

The following are expressly out of scope: full raw-IQ processing, complex
second-peak synthesis, B0 retraining, any fit using attacks, and any tuning after
results. Stage-0B reuses the frozen Stage-0 event/PRN identity, labels, phases,
source support, and valid masks exactly. It does not create a favorable
intersection for any method.

## 2. Profile S: exact Stage-0 cleanStatic roles

Profile S is primary and retains the existing source-support split exactly:

- `normal_train`: cleanStatic event `source_end <= 300`;
- `normal_calibration`: cleanStatic event `source_start >= 320` and
  `source_end <= 400`;
- `normal_holdout`: cleanStatic event `source_start >= 420`.

Support crossing a boundary is excluded. Role assignment is made on the frozen
event support and broadcast to its linked PRN rows. Roles are disjoint and no
normal holdout row fits or calibrates anything.

All methods use the full frozen identity key
`(physical_recording_id, scenario, prn_or_event_id, target_index,
availability_time_s)` and require exact equality of source start, source end,
label, and valid metadata. PRN and event inventories must be identical across
methods. Attacks preserve the same frozen identity, phase, label, and masks and
are scored only after all clean fits, scale bounds, and thresholds are sealed.
Frozen onsets are 100 s for DS1/DS2/DS3 and 110 s for DS7/DS8. Stable-pre is
`event source_start >= 30` and `event source_end <= onset-20`; post starts when
event `source_start >= onset`; persistent starts at `onset+40`; crossing events
remain transition/non-post. Event support is the union of linked PRN support,
with event availability at maximum linked PRN source end.

`cleanDynamic` under Profile S is an **external cleanDynamic diagnostic**. It is
not a training domain, does not calibrate a threshold or clamp, and must never be
described as Profile D evidence.

## 3. Three target-matched IQ conditioners

Fit three separate `sklearn.linear_model.HuberRegressor` models, one for each
positive target:

1. `TOPI`: frozen `S_perp`;
2. `B0`: frozen B0 score;
3. `total`: frozen total quadratic energy.

A shared or multi-output fit is forbidden. Each event's frozen causal IQ vector
is broadcast to every linked valid PRN row, so events are naturally PRN
weighted exactly as in Stage-0. The only predictors, in this exact order, are:

1. `log_power`;
2. `log_noise_floor_scale`;
3. `spectral_flatness`;
4. `lag1_autocorr_magnitude`.

These are the existing causal Stage-0 contexts; Stage-0B does not process the
full raw IQ again. On Profile-S clean-train PRN rows only, standardize each
feature by its median and IQR. If an IQR is `<= 1e-8`, replace it with `1`. The
response for every target is `log(max(target, 1e-12))`. Huber parameters are
`epsilon=1.35`, `alpha=1e-4`, and `max_iter=1000`. At inference, clip predicted
log scale to `[-745, 709]` before exponentiation.

Attack labels, attack rows, scenario IDs, PRN IDs, attack onsets, normal
calibration targets, normal holdout targets, and cleanDynamic rows cannot enter
a fit. Fit manifests must bind ordered row identities, feature/target digests,
feature medians/IQRs, fallback flags, coefficients, intercept, and all Huber
parameters.

## 4. Mandatory reconstruction of original NC-TOPI

Before accepting any audit output, refit the original TOPI conditioner from the
frozen Profile-S clean-train rows. Reproduce the original upper cap from the
**actual clean-calibration predicted TOPI scales** at q99.5 using
`numpy.quantile(..., method='higher')`. There is no original lower cap. The
original denominator is

```text
max(min(predicted_TOPI_scale, clean_calibration_q995), 1e-12)
```

and `NC_TOPI_original = TOPI / denominator`.

Every reconstructed original score must match the frozen original NC-TOPI with
both relative and absolute tolerance `1e-12`; any mismatch fails closed. A
second diagnostic computes effective scale as `TOPI / NC_TOPI_original` and
requires it to match the refit's materialized denominator at the same tolerance.
Ordinary division by zero must fail closed. In the mathematically defined
both-zero boundary, no scale may be inferred from `0/0`; compare the
implementation's directly materialized denominator to the refit denominator.
All frozen rows are reported nonzero, but tests must cover TOPI zero, NC zero,
and both-zero boundaries rather than relying on that observation.

This reconstruction is a provenance check, not a chance to repair or tune the
parent detector.

## 5. Frozen scale clamps and output methods

For each of TOPI, B0, and total separately, evaluate its predicted scales on
Profile-S clean-calibration **PRN rows**. The primary target-specific clamp is
q1 to q99, with both bounds computed by NumPy `method='higher'`. Attacks cannot
select or alter a bound.

Store these clamp diagnostics separately and do not let them replace the
primary:

- two-sided q0.5 to q99.5;
- lower-only q1;
- upper-only q99;
- no clamp.

The required primary method names and order are exactly:

1. `B0`;
2. `TOPI`;
3. `NC_TOPI_original`;
4. `IQ_LOW_ONLY`;
5. `IQ_OOD_ONLY`;
6. `NC_TOPI_clamped`;
7. `NC_B0_clamped`;
8. `NC_total_clamped`.

The normalized primary scores are target matched:

```text
NC_TOPI_clamped  = TOPI  / max(clamp(pred_TOPI, TOPI_q1, TOPI_q99), 1e-12)
NC_B0_clamped    = B0    / max(clamp(pred_B0,   B0_q1,   B0_q99),   1e-12)
NC_total_clamped = total / max(clamp(pred_total,total_q1,total_q99),1e-12)
```

Diagnostic clamp variants belong under a separate namespace and cannot be
chosen from attack performance.

## 6. IQ-only controls

`IQ_LOW_ONLY` is exactly:

```text
-log(max(predicted TOPI scale, 1e-12))
```

`IQ_OOD_ONLY` uses only the multiset of Profile-S clean-calibration predicted
TOPI scales. For evaluated scale `s`, calibration size `n`, and ties included on
both sides:

```text
p_lower = (1 + # {calibration scale <= s}) / (n + 1)
p_upper = (1 + # {calibration scale >= s}) / (n + 1)
p       = min(1, 2 * min(p_lower, p_upper))
score   = -log(max(p, 1e-12))
```

The `<=` and `>=` tie operators are frozen. Calibration self-evaluation uses the
same full reference multiset; no leave-one-out variant is allowed.

IQ-only scores aggregate by the median over linked valid PRN rows. This
preserves the exact common event mask and equals the event-level score whenever
the broadcast predicted scale is common within the event. The implementation
and verifier must audit that common-scale equality rather than assume it.

## 7. Aggregation, thresholds, metrics, and uncertainty

The primary event aggregation is median. Each method receives its own q99
threshold from Profile-S clean-calibration event scores using NumPy
`method='higher'`; alarm semantics are strictly `score > threshold`. If the
parent data already expose top25/q99.5 diagnostics, retain them only as a
separate diagnostic. They cannot alter the primary or decision.

On the exact common eligibility mask, report:

- ROC-AUC;
- PR-AUC;
- sklearn standardized pAUC with `max_fpr=.05`;
- q99 normal FPR;
- attack stable-pre FPR;
- post detection rate;
- three-consecutive-alarm delay;
- persistent alarm ratio.

The sustained delay requires three strict alarms at 0.5 s cadence. Sort within
physical recording; reject duplicate recording/time rows; reset on a time gap,
recording change, or any non-post row. Delay uses the third alarm's event
availability/source end minus onset and is explicitly censored if no run
completes. Persistent ratio is the strict-alarm fraction on the frozen Stage-0
persistent phase.

For DS7 and DS8, run exactly six paired comparisons with
`NC_TOPI_clamped` as the first score and these comparators:

1. `B0`;
2. `TOPI`;
3. `IQ_LOW_ONLY`;
4. `IQ_OOD_ONLY`;
5. `NC_B0_clamped`;
6. `NC_total_clamped`.

`NC_TOPI_original` still receives point metrics, q99 alarms, and the overlap and
gain diagnostics required by the decision, but it is a reconstruction/control
method rather than one of the six paired-bootstrap comparators.

Every paired pAUC delta uses 2,000 gap-safe complete nonoverlapping 10 s block
bootstrap repetitions, stratified by label and formed only within a physical
recording and contiguous cadence run. Scores remain paired at the same event
indices. Report percentile 95% intervals; the point estimate uses every eligible
event. Class-deficient support or fewer than two complete blocks in either
stratum produces an explicit unavailable result. There is **no IID fallback**.
The bootstrap seed remains the frozen Stage-0 value `20260803`.

## 8. Frozen time-shuffle diagnostic

Stage-0 source code's more exact safe rule is retained: typed clean-train row
identity and the ordered feature matrix remain fixed, and only the response is
permuted. Stage-0B changes the requested deterministic seed to `0` and uses

```text
permutation = numpy.random.default_rng(0).permutation(n_train_rows)
```

strictly within Profile-S cleanStatic normal-train PRN rows. Apply the same
permutation index to TOPI, B0, and total response vectors. No calibration,
holdout, cleanDynamic, or attack row is permuted, and no attack timing or label
participates. Fit three shuffled target models and apply the same target-specific
clamps, event aggregation, thresholds, and metrics under a separate
`time_shuffle` diagnostic namespace. It cannot select, rescue, or fail the
primary decision.

This exact choice is frozen here, before metrics: **features stay in temporal
identity order; targets alone are permuted within clean normal training rows**.

## 9. Profile D sufficiency pre-rule

Profile D uses cleanDynamic but is not another name for the Profile-S external
cleanDynamic diagnostic. Its availability is decided before any Profile-D fit.

For every candidate event, effective source support includes target support,
the full frozen B0 12-window history, and the frozen causal IQ history. A valid
split must consist of three chronological, nonoverlapping contiguous roles in
train/calibration/holdout order, with at least a 10 s effective-source gap
between adjacent roles. Minimum event counts are:

- train: 50;
- calibration: 101, so a `method='higher'` q99 is not the sample maximum;
- holdout: 50.

No random split is allowed. If no such assignment exists, status is exactly
`INSUFFICIENT_NORMAL_SUPPORT`; do not fit, clamp, threshold, or report Profile-D
performance. The observed best chronological gap-safe split of `33/33/33` is
recorded only as support evidence, never as performance. If Profile D is
available in a future conforming inventory, all three target-matched models must
be newly fit on its train role, all clamps and thresholds must come only from
its calibration role, and only its holdout is evaluated.

## 10. Frozen decision grammar

The Stage-0B status is separate from Stage-0 and is one of:

- `IQ_SHORTCUT_DOMINATED`;
- `TANGENT_SUPPORTED`;
- `INCONCLUSIVE`.

Validate evidence first. Then evaluate **all shortcut triggers before tangent
support**, so a shortcut cannot coexist with a supported label. Invalid or
missing evidence cannot produce `TANGENT_SUPPORTED`; absent a fully evaluable
true shortcut, it produces `INCONCLUSIVE` with validation errors.

### 10.1 Shortcut triggers

Any one trigger yields `IQ_SHORTCUT_DOMINATED`:

**A — IQ point or CI dominance.** Trigger if either IQ-only method is pointwise
`>= NC_TOPI_clamped` on both DS7 and DS8, or if any DS7/DS8 paired CI for
`NC_TOPI_clamped - IQ_LOW_ONLY` or `NC_TOPI_clamped - IQ_OOD_ONLY` has upper
endpoint `<= 0`.

**B — statistically indistinguishable from normalized B0.** Trigger if the
paired `NC_TOPI_clamped - NC_B0_clamped` CI includes zero on **both** DS7 and
DS8. “Includes” is inclusive: `lower <= 0 <= upper`.

**C — clamp reverses an original positive gain.** Trigger on either DS7 or DS8
when clamped `NC_TOPI_clamped - B0` point pAUC delta is `<= 0` while the same
scenario's original `NC_TOPI_original - B0` delta was `> 0`.

**D — scale-only gain fraction plus alarm overlap.** For each DS7 and DS8 only
when `NC_TOPI_original - B0 > 0`, calculate

```text
(IQ_LOW_ONLY - B0) / (NC_TOPI_original - B0)
```

from point pAUC values. Trigger only if this fraction is `>= .5` on both
scenarios **and** original-NC q99 alarm overlap with IQ_LOW q99 alarms is `>= .5`
on both, where overlap is
`count(NC_original alarm AND IQ_LOW alarm) / count(NC_original alarm)`. A
nonpositive gain denominator excludes that scenario and makes trigger D false
unless both scenarios are eligible. Zero original-NC alarms make D false with an
explicit not-evaluable reason; no arbitrary zero division is permitted.

**E — external cleanDynamic collapse.** Trigger if Profile-S external
cleanDynamic q99 FPR for `NC_TOPI_clamped` is `>= .50`. This remains a Profile-S
external diagnostic and is not Profile D.

### 10.2 The seven tangent conditions

If no shortcut trigger is true, tangent support requires all seven conditions:

1. On both DS7 and DS8, clamped NC point pAUC is strictly higher than
   `NC_B0_clamped`.
2. On both DS7 and DS8, it is strictly higher than `IQ_LOW_ONLY`.
3. On both DS7 and DS8, it is strictly higher than `IQ_OOD_ONLY`.
4. At least one DS7/DS8 paired `NC_TOPI_clamped - NC_B0_clamped` CI has lower
   endpoint strictly `> 0`.
5. On both DS7 and DS8, point pAUC
   `NC_TOPI_clamped - B0` is strictly `> 0`.
6. Profile-S cleanStatic holdout q99 FPR for `NC_TOPI_clamped` is `<= .02`.
7. Every Profile-S stable-pre q99 FPR on DS1/DS2/DS3/DS7/DS8 is strictly
   `< .05`.

In addition, if and only if Profile D is available, its holdout q99 FPR must be
strictly `< .05`. `INSUFFICIENT_NORMAL_SUPPORT` makes this conditional gate not
applicable; it is not a fabricated pass or performance result.

With no shortcut and every required tangent condition true, return
`TANGENT_SUPPORTED`. Otherwise return `INCONCLUSIVE`.

## 11. Frozen second-peak limitation

Stage-0 criterion c7 remains a failure. Stage-0B must not delete it, relabel it as
a pass, or use this audit to change the Stage-0 decision. The exact limitations
are:

1. prompt-normalized magnitude direct sum;
2. missing complex carrier phase;
3. no Prompt renormalization after synthesis;
4. finite 9-tap zero padding;
5. strict separation monotonicity;
6. single reference.

No complex second-peak synthesis is performed in Stage-0B. A complex-domain
correction may be proposed **only** if the final Stage-0B shortcut decision is
`TANGENT_SUPPORTED`; even then it is a new proposal, not a retroactive c7 pass.

## 12. Artifact and independent-verification contract

The future runner publishes atomically and append-only to
`artifacts/nc_topi_stage0b_audit`. Required root files are:

- `README.md`, `config.json`, `provenance.json`, `data_manifest.json`,
  `parent_inventory.json`;
- `profile_support.json`, `fit_audit.json`, `refit_equivalence.json`,
  `scale_bounds.json`, `thresholds.json`;
- `per_prn_scores.csv`, `event_scores.csv`, `scenario_metrics.csv`,
  `bootstrap_results.json`, `decision.json`;
- `verification.json`, `hashes.json`.

Required diagnostic files are:

- `diagnostics/clamp_variant_metrics.csv`;
- `diagnostics/time_shuffle_metrics.csv`;
- `diagnostics/time_shuffle_fit_audit.json`;
- `diagnostics/iq_scale_checks.json`;
- `diagnostics/second_peak_limitations.json`.

`diagnostics/top25_q995_metrics.csv` is allowed only when the corresponding
parent diagnostic data are present. No other optional result may silently enter
the decision. Artifact `config.json` must be byte-identical to the committed
machine-readable config. Provenance must include the contract commit, both
parent commits, clean worktree status, invocation, library versions, the fact
that attack loading opened only after freeze, and `post_result_tuning=false`.

An independent verifier must recompute rather than trust summary scalars. It
checks parent commit/inventory binding; absence of Stage-0 modifications; exact
method/scenario/role/PRN/event inventories and masks; clean-only fit and clamp
provenance; all three Huber fits; original reconstruction and effective scales;
quantile methods; IQ tie handling; event aggregation; thresholds; point metrics;
2,000 paired gap-safe bootstrap requests with no IID fallback; Profile-D
sufficiency before fit; shortcut-first decision regeneration; c7 preservation;
and all forbidden-operation attestations. Any mismatch exits nonzero and blocks
a verified publication.

`hashes.json` is a sorted, complete, self-excluding SHA-256 manifest over every
regular file below the Stage-0B artifact root. Each entry stores relative path,
byte size, and digest. The verifier independently recomputes it; missing,
unexpected, duplicate, or mismatched files fail closed.

## 13. Freeze boundary

This commit freezes only the config and documentation. Production code, tests,
and artifacts must come later in separate commits. No raw-IQ campaign, attack
metric, bootstrap result, clamp comparison, or decision has been observed or
computed to create this contract.

## Contract correction: reconstruction roles, Profile D support, and exact inventory

This correction is frozen before any Stage-0B attack result is viewed. The phrase *existing split / same conditioner structure* means the exact preserved Stage-0 PRN roles: 6,074 valid `cleanStatic/normal_train` PRN rows fit every original or target-matched conditioner, and 1,628 valid `cleanStatic/normal_calibration` PRN rows select original/target clamp bounds. The preserved PRN inventory is 6,074 train, 1,628 calibration, 1,306 holdout, and 891 boundary-crossing exclusions. No event-derived split may replace these fit rows.

Thresholds and metrics are event-level and use the preserved aggregate event roles on the common valid event mask: 586 train, 157 calibration, 118 holdout, and 86 boundary-crossing exclusions. Thus reconstruction/clamp fitting uses PRN roles while thresholding/evaluation uses event roles; this boundary is intentional and audited byte-for-byte against the parent.

Profile D effective support is the union minimum/maximum of (1) parent event and linked-PRN target support, (2) B0 12-window history at 0.5 s cadence, whose earliest support is exactly `target source_start_s - 6.0 s` in this frozen lineage after cadence/sequence validation, and (3) every IQ block interval parsed from `iq_context.block_start_s` and `block_end_s`. The exhaustive chronological rule is `max(previous role end)+10 <= min(next role start)` with minima 50/101/50. Frozen actual support remains insufficient, with best balanced split 33/33/33; therefore no Profile-D fit or performance metric is permitted. A synthetic available path must invoke its callback and return fit, clamp, threshold, and holdout evidence.

The JSON contract now lists exact regular-file inventories independently for the artifact root, `diagnostics/`, and `plots/`. Compatibility names required by the user are canonical names, not aliases. Unknown, missing, or duplicate semantic files fail closed. Verification is prepared in staging by the standalone verifier, hashed excluding only `hashes.json`, confirmed in a final read-only verifier pass, and only then atomically published with no-replace semantics.
