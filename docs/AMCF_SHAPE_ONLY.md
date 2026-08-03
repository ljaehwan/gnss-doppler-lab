# AMCF Shape-Only preregistration and implementation contract

## Status and hypothesis

This document freezes the analysis before any attack scenario is executed. The single hypothesis is: **is Prompt-normalized correlator complex shape more useful for detection than magnitude shape?** This is a context-free *representation* ablation, not removal of temporal history. No active querying, information gain, attention, or other new architecture is permitted. DS1, DS2, DS3, DS7, and DS8 are all exploratory/developmental in this campaign. There is no confirmatory claim.

No attack data may be used to fit a Prompt gate, scaler, model, checkpoint-selection objective, conformal calibration, threshold, clipping rule, or hyperparameter. There is no matched-clean operating point and no post hoc attack retuning.

## Frozen signal and feature contract

Tap order is `E4,E3,E2,E,P,L,L2,L3,L4`; Prompt is index 4 and side indices are `0,1,2,3,5,6,7,8`. For each raw row:

`z = C * conj(P) / (abs(P)^2 + eps)`

The low-P threshold is the cleanStatic-train q0.005 quantile using method `higher`; admission is the literal comparison `abs(P) >= minimum` (no tolerance easing). The primary epsilon is frozen at `1e-12`. Prompt is used only for this gate, normalization, and QA. Prompt index 4 is never present in a feature, target, query, score, or model tensor. The model API cannot receive C/N0, log absolute Prompt, Prompt magnitude, valid fraction, valid/rejected/raw counts, recording/scenario ID, PRN identity, or any ID/time. IDs and times are indexing metadata only.

Each one-second source window is exactly `(T-1.0,T]`, with decisions every 0.5 s. Source rows must be wholly in one role. Complex side-tap features are exactly `median(real), median(imag), literal MAD(real), literal MAD(imag)`. Magnitude features are exactly `median(abs), literal MAD(abs)`. “Literal MAD” means median absolute deviation with no 1.4826 multiplier. There is no zero padding, duplicate feature, Prompt column, or QA column. Every target tap/dimension must have clean-train IQR strictly above tolerance or the run hard-fails.

The robust scaler is fitted independently for each representation, tap, and dimension from cleanStatic train only, using median and IQR. Its arrays are immutable. There is no attack-derived clipping.

## Causal history and model fairness

A current example is admitted only with the strictly previous 12 full side-feature windows from the same recording, segment, channel, PRN, and role. Endpoints must have exact 0.5 s cadence within declared tolerance. A gap resets history. There is no padding, current/future inclusion, or split-boundary crossing.

Each representation has its necessary `Linear(D,H)` adapter and representation-specific output head. `H=32` is primary. Both use the same permutation-invariant mean tap pooling, same one-layer GRU over the 12 pooled historical windows, and same decoder family. Only the adapter and output head vary with representation dimension. Trainable parameter-count difference must be at most 5%, and all parameters must receive gradients. The model has no tap-position or PRN embedding; tap and PRN ordering cannot change aggregate scores.

## Objectives and scoring

Training and primary scoring use all-side LOO (“all9” retains the nine-tap source name, but Prompt is excluded). For each current side target, the other seven current sides are observed and the previous 12 complete side-feature windows are history. The predicted distribution is diagonal Student-t. The target NLL is averaged over actual target dimensions; the largest two of eight target NLLs are averaged to produce the PRN raw score; the epoch score is the median over tracked PRNs. Tap order and PRN order must not alter the result.

The auxiliary diagnostic is labelled EPL for continuity, but P is normalization reference only. It is an E (source index 3) and L (source index 5) two-target LOO diagnostic: current L alone is observed for the E target, and current E alone is observed for the L target, with the same full history. It is never the GO-primary result.

## Training freeze and convergence

Only cleanStatic train is optimization data. Primary seeds are exactly `101,202,303`; optimizer AdamW, learning rate `1e-3`, real minibatch size `256` **flattened `(sample,target_side_index)` tuples**, maximum `200` epochs, and patience `20`. The all-side bank has eight tuples per sample; EPL has exactly the E/L two. An epoch has `ceil(train_target_tuples/256)` optimizer updates (the final remainder batch may be smaller), and the realized tuple batch sizes and update counts are audited. The validation sample/target bank is generated once, covers fixed target tuples, is hashed, and is reused unchanged every epoch so that the validation objective does not drift. The exact optimizer-update count is audited and must exceed one. Sampling 256 examples and expanding each to eight losses is forbidden. Primary fit accepts only immutable `AuditedCleanSplit` objects emitted by the causal history factory. Its manifest binds actual source intervals, same recording/segment/channel/PRN/role, strict previous-only order, 12-history length, 0.5 s cadence, and split boundary checks to the tensors; arbitrary tensor containers cannot enter fit. Losses and gradients must remain finite.

Best model and best optimizer state are restored together. A seed is converged only when finite patience early stopping occurs at or before the cap. Reaching the cap without a patience stop is nonconverged. Any nonconverged seed is excluded and makes the primary result incomplete; it is never treated as a successful run. The entire primary configuration is hard-frozen: Prompt q0.005, epsilon `1e-12`, H=32, tuple batch 256, lr `1e-3`, cap 200, patience 20, history 12, stride 0.5 s, source width 1.0 s, and seeds exactly `{101,202,303}`. Overrides require `primary=False`; a mismatch on a primary path hard-fails.

## Calibration, alarms, and phases

For each seed, upper-tail conformal p-values are

`p(x) = (1 + count(cal >= x)) / (n + 1)`, with evidence `e = -log(p)`.

For clean calibration sample `i`, leave-one-out calibration is

`p_i = (1 + count over j != i of cal_j >= cal_i) / n`.

A primary threshold is generated only when the seed set is exactly `{101,202,303}`, all three convergence audits pass, and every seed has identical calibration timestamps and row indices. Missing/nonconverged/misaligned input hard-fails rather than producing a normal primary threshold. Evidence is averaged across the three seeds. q99 and q995 are `method="higher"` quantiles of the cleanStatic calibration LOO ensemble evidence. Alarms are always recomputed as strict `score > threshold`; saved alarm columns are not trusted. q99 is GO-primary and q995 is diagnostic.

Phases use actual `source_start` and `source_end`, not nominal endpoint assumptions:

- stable-pre: source interval wholly contained in `[30,onset-20]`;
- post: `source_start >= onset`;
- persistent: `source_start >= onset+40`.

Sustained-three delay starts at the first wholly-post window and requires three consecutive 0.5 s alarms. If a sustained alarm already occurs in stable-pre, report exactly `N/A: already alarming in stable-pre`, never zero.

## Paired inference and collapse audit

Every Complex/comparator contrast is joined on common timestamps before analysis. Paired bootstrap resamples labels, timestamps, Complex scores, and comparator scores with the **same row indices** from the same 10 s timestamp blocks. Fewer than 2000 replicates hard-fails. ROC-AUC resamples the aligned labelled rows; post detection and stable-pre FPR first select the declared phase using actual source-interval masks and then construct/resample blocks within that phase (phase-local resampling). Representation-specific thresholds are applied after each paired draw. Delta sign is **Complex - comparator**. Frozen metrics are ROC-AUC, post detection, and stable-pre FPR. A positive delta always favors Complex for ROC-AUC/post detection; stable-pre FPR interpretation retains the signed definition.

DS7 and DS8 must both be present and pass a schema-collapse audit: the actual schema must exactly equal the declared Complex/Magnitude schemas and contain no C/N0 or context field; every tap/dimension in clean and in each scenario must be finite with IQR strictly above tolerance; median tracked PRN count must exceed one; stable-pre alarm rate must be below 100%. `no_cn0_branch=True` assertions without inspecting the actual schema are not evidence. Feature extraction must not branch by DS7/DS8 or by C/N0.

## Exact GO / NO-GO decision

All clauses are mandatory at q99:

1. stable-pre FPR is below 5% in all five scenarios;
2. Complex ROC-AUC is greater than Magnitude in at least 4 of 5 scenarios;
3. the paired bootstrap CI lower bound for the Complex-minus-Magnitude AUC delta is above zero in at least 3 scenarios;
4. the AUC direction is positive in at least 2 of 3 seeds for each scenario;
5. versus frozen B0, Complex improves AUC by at least 0.02 **or** post detection by at least 0.05 in at least 3 of 5 scenarios, while FPR degradation is at most 0.01;
6. all three seeds converge under the frozen definition;
7. neither DS7 nor DS8 collapses under the audit above.

The machine decision emits criterion-level PASS/FAIL plus full-precision evidence. Exact scenarios `DS1,DS2,DS3,DS7,DS8` are mandatory; missing/duplicate scenarios hard-fail. Strict boundaries are used (`FPR < .05`, AUC/CI direction `> 0`, B0 gains `>=`, FPR degradation `<=`). Any criterion failure means final **NO-GO** and **AMCF WCL no-go**. q995 is diagnostic, is rejected by the primary decision API, and cannot rescue q99. No attack retune is allowed after seeing any DS result.

## Campaign runner and deterministic replay

`scripts/run_amcf_shape_only.py` is the sole end-to-end entry point. Primary mode requires a clean committed tree, verifies all six canonical SHA-256 pins, performs a real CUDA tensor probe, reads only the six allowlisted NPZ fields (never C/N0), writes through an atomic staging directory, and refuses overwrite. Its only primary destination is exactly `artifacts/amcf_r1_shape_only/`. A synthetic `--smoke` mode must use another destination and cannot create the final result. The too-small-P gate is `abs(P) >= max(clean-train q0.005 higher, float64 positive floor)` together with the explicit `abs(P) > 0` quality rule, so an exact-zero Prompt is rejected even when the empirical quantile is zero.

Feature provenance binds the input digest, Prompt gate, literal representation schema, causal/role QA, and actual feature tensor digest. Scaler, validation-bank, convergence, checkpoint, calibration, metric, paired-comparison, schema-collapse, and alarm evidence are persisted. Fit and GO paths fail closed when these bindings differ. A primary ensemble requires exactly three converged seeds; nonconverged models are marked excluded and cannot be silently averaged.

`scripts/summarize_amcf_shape_only.py` accepts finalized artifacts only. It verifies `hashes.json` before parsing evidence, independently recomputes strict alarms, metrics, and the q99-only GO criteria from saved score tables, checks checkpoint/convergence digests, and requires a byte-identical deterministic README. q995 remains diagnostic. Every attack result is exploratory/developmental.

The required inventory is `README.md`, `config.json`, `feature_schema.json`, `provenance.json`, `input_hashes.json`, `training_history.csv`, `convergence_audit.json`, `thresholds.json`, `scenario_metrics.csv`, `seed_metrics.csv`, `paired_comparisons.csv`, `per_epoch/`, `plots/`, `models/`, and self-excluding `hashes.json`; additional QA and feature-cache evidence is allowed.

## Scope of this source commit

This source commit adds the leakage-safe campaign runner, deterministic summarizer, and synthetic runner tests. It deliberately does **not** execute the full attack campaign or create `artifacts/amcf_r1_shape_only/`.
