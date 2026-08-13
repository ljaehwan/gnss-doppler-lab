# GCSPO Stage-0: design and preregistration

## Scope

GCSPO (Geometry-Constrained Shared Pull-Off Observer) is a linear/statistical Stage-0 experiment. Its question is narrow: after a normal-only one-step dynamics model, do the *signed* innovations across satellites admit a materially better common position/clock and velocity/clock-drift explanation than H0, even after penalizing the shared state's effective degrees of freedom?

Phase 1 contains no detector implementation and no attack result. The authoritative machine-readable contract is under `artifacts/gcspo_stage0_static_rerun/`.

## What the receiver actually exposes

The selected TEXBAT cleanStatic complex-9 receiver cache contains 11 channel MAT files, 553,639 valid rows, and 46 aligned datasets per file. The audit found no missing/non-finite values in the candidate fields. The patched receiver source proves signed I/Q for nine taps, Prompt I/Q, the E-minus-L DLL discriminator, PLL discriminator output, filtered loop commands, carrier Doppler, code frequency, C/N0, carrier-lock statistic, remnant code phase in samples, raw sample counter, and PRN.

Several names need care. `carr_error_hz` is computed as atan divided by 2π and is therefore a phase discriminator in cycles, not Hz. On cleanStatic, `carr_error_filt_hz` is bit-identical to `carrier_doppler_hz`, so it is excluded from `q` and cannot masquerade as a separate innovation. `code_error_chips` has unresolved unit/scale ambiguity and its physical loading fails closed until a source-derived unit test passes; `code_error_filt_chips` is source-commented as chips/s. Carrier Doppler and code-frequency offset are state/NCO levels: only their frozen one-step VAR residuals are innovations. The internal FLL frequency error is not dumped. Prompt phase increment is not a direct field. Elevation, azimuth, and LOS are not tracking columns; they are conditional derived quantities and are legal only after time-anchor, ephemeris-health, and static-position preflights pass. C/N0 is metadata-only and excluded from the primary score; lock quality is nuisance metadata, and neither conditions the frozen primary covariance. PRN is grouping/diagnostic metadata.

## Frozen normal split

Only cleanStatic establishes the static normal model. At 20 ms bins, the audited intervals below each retain at least ten tracked PRNs:

- train `[30,210)` s;
- guard `[210,220)` s;
- calibration `[220,340)` s;
- guard `[340,350)` s;
- sealed holdout `[350,470)` s.

The first 30 s and tail after 470 s are excluded. Raw samples and epochs cannot overlap roles. Histories reset at every role/scenario/segment/gap boundary, the first 0.20 s after reset is ineligible, and score windows must be wholly contained in one role. Only provenance-valid DS3/DS4 pre-onset is external FPR evidence. The byte-identical DS7/DS8 pre-110 replay is descriptive training replay and is excluded from G1, worst external FPR, intervals, and resampling.

## Frozen model

For each PRN, `q_i(t)` contains gain-normalized signed complex E/P/L components, raw DLL error, PLL phase error, carrier Doppler, and code-frequency offset. A PRN-invariant shared-coefficient VAR(10) is fitted only on the cleanStatic train role. All robust residual location, per-PRN Ledoit-Wolf covariance, common-mode Gamma, eigenvalue floors, ridge, lambda, and A2 loading are fitted from cleanStatic train only. Calibration emits frozen scores and derives thresholds only. Whitening is

`z_i(t) = Sigma^(-1/2) (r_i(t) - mu)`.

H0 adds no coordinated input. H1 uses

`s = [delta_p(3), delta_b, delta_v(3), delta_bdot]`,

`delta_rho_i = -u_i^T delta_p + delta_b`,

`delta_rhodot_i = -u_i^T delta_v + delta_bdot`.

Only channels with source-verified units receive a direct physical loading. Signed complex EPL channels remain in the joint residual likelihood but receive zero direct state loading, preventing an invented correlator-unit-to-meter mapping. The frozen VAR operator is applied to the H1 measurement shift as well as H0 prediction. A 1 s causal linear-Gaussian constant-velocity/smooth-random-walk window is evaluated every 0.5 s with at least four PRNs; this cadence also permits exact common-window support with B0.

The state is estimated by a fixed-prior penalized MAP smoother, not by marginal/integrated Kalman evidence. On common-nuisance-whitened observed coordinates, `xhat=argmin ||y-Gx||²+xᵀRx`; `ell_obs(y;x)=-0.5||y-Gx||²+constant`; `A=G(GᵀG+R)^-1Gᵀ`; and `S=2[ell_obs(y;xhat)-ell_obs(y;0)]-trace(A)log(len(y))`. The prior quadratic/determinant never enters `ell_obs`, so complexity is counted exactly once. Stable solves use frozen pseudoinverse `rcond=1e-10` when necessary and must satisfy `0<=trace(A)<=rank(G)<=n_obs`. Raw RSS improvement alone, RMSE, power, innovation norm alone, and arbitrary score fusion are forbidden.

## Evaluation freeze

Timelines are scenario-specific: DS3 onset 118.9 s and pull-off 195 s; DS4 onset 113.8 s and pull-off 225 s; DS7/DS8 injection 110 s and pull-off 150 s. DS1/DS2 are diagnostic. cleanDynamic and DS5/DS6 are OOD only and cannot influence static GO/NO-GO. DS4 is transition-only unless authenticated coverage past 225 s is established.

Thresholds are deterministic chronological cleanStatic-calibration nearest-rank empirical q99/q99.5; primary q99 uses zero-based index `ceil(0.99*n)-1` and strict `score>threshold`. Temporal dependence prevents finite-sample conformal coverage, so false-alarm validity is empirical on holdout and provenance-valid external pre-onset evidence. Within each frozen contrast, compared methods share exact event/epoch/PRN support; optional-method unavailability affects only that contrast, not unrelated core metrics. The required ablations are B0 exact, robust pooled signed norm, no-geometry rank-1 input, code geometry, carrier geometry, independent per-PRN states, and Full. B0 is pinned to checkpoint `f171bf0b...` and its exact nine-tap/sequence/binomial-tail score definition, while its alarm thresholds are recalibrated from the frozen cleanStatic calibration role. Historic metrics may not be copied. Fixed complex-9 remains unavailable. M1 requires a support adapter. GCMR natively uses the same 1.0 s window/0.5 s stride as Full and is adapter-required, conditionally available from checkpoint `4eccfdba…`: the identical epoch+PRN mask must be applied to raw TrackingRows before common-clock removal, geometry, and complete-pair construction. An event needs at least four common PRNs, twenty common 20-ms bins, valid geometry, continuity, and a complete pair set; otherwise that event is removed from every compared method. Historic score CSVs may never be joined or copied.

Relation destruction uses same-epoch LOS shuffle and label-phase-bounded independent per-PRN temporal shifts. Controls cover gain, Prompt amplitude, C/N0 metadata-exclusion invariance, empirical clean noise, PRN drop-only, one-PRN disturbance, receiver-clock path, and independent multipath-like disturbances. IQ power is never scored and no synthetic PRN identity is added.

Metrics include ROC-AUC, normalized low-FPR pAUC through 5%, PR-AUC, clean holdout/external pre-onset FPR, phase-specific detection, onset and pull-off persistent-alarm delays, persistent ratio, shared-state magnitudes, likelihood improvement, penalty, and tracked-N. Uncertainty uses paired 10 s block bootstrap; DS7/DS8 form one family.

## Decision and provenance boundary

`GO_FOR_NEURAL_STAGE1` requires every integrity, false-alarm, incremental-value, geometry-destruction, persistence, control, and shared-state gate in `preregistration.json`. Any failure yields `NO_GO_PHYSICAL_HYPOTHESIS`; neural networks or added scores cannot rescue it. A future GCSPO-N is GO-only documentation, TEXBAT is developmental, and final confirmation must use OAKBAT or a sealed new static receiver session.

Current provenance gaps are recorded explicitly rather than inferred away. Authoritative raw TEXBAT IQ exists on MainServer `192.168.0.77` under `/mnt/user/gnss-datasets/texbat/raw/` but is not mounted on this VM. Preferred DS7/DS8 receiver products and available file hashes are pinned, and their first 110 s are byte-identical to cleanStatic, making all three one evidence family. The patched receiver tree remains dirty without a sealed full patch/build recipe, DS4 lacks authenticated output hashes/end coverage, DS5/DS6 receiver outputs were not located, and protected timing/geometry joins remain unopened. Protected evaluation must fail closed until required identities and joins are authenticated.


## Verification and access boundary

Implementation must pass the machine-readable verifier suite in `config.json` and `preregistration.json`: causality; normal-only fitting/calibration; raw-sample and epoch role-overlap leakage; timestamps/onsets; geometry units/signs; PRN permutation invariance; variable PRN counts with fewer than four unavailable; relation-shuffle preservation; effective-DoF correction; common support; deterministic reproduction; artifact checksums; and a fresh-clone verifier. A protected-access ledger, canonical-path allowlist preflight, and before/after file-access trace are mandatory before any protected read.

The repository and team have prior TEXBAT exposure. An isolated independent Phase-1 audit used one overly broad source search and saw pre-existing result-path output, but no numeric prior result was transmitted to this implementation worker or used to choose any frozen feature, score, threshold, control, gate, or hyperparameter. Old results must not be reopened to elaborate. TEXBAT remains developmental; final confirmation requires OAKBAT or a sealed new static session.

Future outputs must use the exact filenames and ordering in `future_artifact_contract`. Temporary files are preflight failures and are never artifacts. `artifact_manifest_sha256.json` is written after scientific/non-verifier provenance and excludes exactly itself and the two verifier reports; those attestations are then written with its hash. Only a valid protected run may write `final_verdict.json`, with exactly `GO_FOR_NEURAL_STAGE1` or `NO_GO_PHYSICAL_HYPOTHESIS`.

## Semantic BLOCK resolution

H0 and H1 share the same zero-mean epochwise common q-channel nuisance. The final coordinate, estimator, floor, independence, block-diagonal covariance, and lower-Cholesky rules are defined in “Final computational uniqueness” below. It is never an H1-only smooth mean. Every range/rate epoch requires the exact `M_t` rank-four and `kappa<=1e4` rule.

Direct physical loading is conditional on a mandatory source/synthetic `closed_loop_transfer_jacobian` preflight. It proves source hashes, cadence, receiver equations, units/signs, wrapping/linear range, loop/NCO paths, and deterministic impulse/ramp gain through the frozen VAR. Unproved rows receive zero loading; insufficient A3/A4/Full geometry makes the experiment invalid before protected access. Complex EPL remains covariance-only.

Epochs and 1-s score windows are anchored to recording start on exact 20-ms/0.5-s grids, timestamped at right endpoints, and cannot straddle roles or phases. Persistence is causal trailing 3-of-5; a missing slot breaks the run. DS7/DS8 pre-110 scores are `TRAINING_REPLAY_DESCRIPTIVE` only and excluded from G1, worst external FPR, intervals, and resampling. Primary pooling is scenario-phase balanced; DS7/DS8 post-110 is one family. The aligned paired 10-s bootstrap and all contrast-specific support rules are machine-readable in `preregistration.json`.

All mandatory controls run on non-overlapping cleanStatic-holdout blocks with resets and frozen thresholds. C/N0 is a metadata-exclusion invariance test, PRN manipulation is drop-only, and the pure-clock specificity criterion is exactly `Epos/(Eclock+1e-12)<=0.25`. A control that cannot be generated with verified semantics is invalidity, not scientific NO-GO.

Validity and scientific verdict are separate. Any failed pre-attack integrity/semantic check writes `invalid_run.json`, sets `INVALID_EXPERIMENT_NO_ATTACK_ACCESS`, keeps protected rows sealed, and emits no `final_verdict.json`. Only `VALID_SCIENCE` may emit exactly `GO_FOR_NEURAL_STAGE1` or `NO_GO_PHYSICAL_HYPOTHESIS`. Canonical test, clean-only, full-verifier, and fresh-clone commands plus exact `verifier_report.json` schemas are frozen in the machine-readable package.


## Final computational uniqueness

Pooled train-fit whitening is first: `z_i,t=Sigma_train^-1/2(r_i,t-mu_train)` using a symmetric eigendecomposition. At each train-fit epoch with at least four PRNs, `m_t` is the coordinatewise PRN median in `z`; columns are arithmetic-mean centered and Ledoit-Wolf gives Gamma. Common effects are independent by epoch. `C_t=I+(11^T) kron Gamma` on observed coordinates, the 50-epoch covariance is block diagonal, and deterministic lower-Cholesky solves whiten both observations and designs.

For physical state `[p,b,v,bdot]`, `rho=-u^Tp+b` and `rhodot=-u^Tv+bdot`. Conditional direct shifts are exactly `-rho/code_chip_m`, `-rho/lambda_L1_m`, `-rhodot/code_chip_m`, and `-rhodot/lambda_L1_m`; all EPL/CN0/quality rows load zero. The source column `carr_error_hz` means cycles. An opposite source sign fails preflight rather than being flipped. With `q_shift_t=B_tx_t`, the only residual design is `g_t=B_tx_t-sum_l A_lB_(t-l)x_(t-l)`, with zero pre-window state history, followed once by common-covariance whitening.

Inner roles are train-fit `[30,140)`, guard `[140,150)`, and lambda-validation `[150,210)`. Method lambda minimizes mean common-window GCV with at least 100 windows and the exact relative tie rule. A1 averages Huber loss over all 50 epochs/channels per continuously present PRN then takes the PRN median. A2 uses arithmetic-centered train-fit median-`z` PCA with the frozen eigengap/sign rules. A5 uses independent normalized range/range-rate states with exact `F2` and block-diagonal segment prior.

Weighted low-FPR pAUC groups exact float64 ties, interpolates at FPR 0.05, trapezoid-integrates, and normalizes by 0.05. Bootstrap blocks use the availability-endpoint `nextafter` rule, full-window containment, post-resample equal weights, DS7/DS8 family clustering, and nearest-rank percentiles. Every random object uses a SHA-256-derived 128-bit big-endian PCG64 seed from its full semantic identity; NumPy version and seeds are recorded. Relations and controls have exact input stages, source-block mapping, PRN ordering, formulas, and no fallback.

Pre-access validity tests control transform generation and semantics, not alarm performance. Poor clean-control alarm behavior is scientific G5 and yields NO-GO only in a valid run. Scientific input/source/access/integrity failure is `INVALID_EXPERIMENT_NO_ATTACK_ACCESS`. Manifest ordering, unchanged-data PNG bytes, or report packaging is `VERIFICATION_ADMIN_WARNING`; it may be repaired in an evidence-only descendant, but final delivery requires verifier PASS. The manifest excludes itself and the two reports, while reports bind the manifest hash plus `target_commit` and `evidence_commit`.
