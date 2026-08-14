# GCSPO Stage-0 static rerun

Round 6 packages three clean-static full-workload reproductions created and
signed by the external controller against source commit
`9774fe1048e467808b53769f94a717507fac5a38`. The bounded public package is
indexed by `round6_a5_provenance.json`, hashed by
`round6_evidence_manifest.json`, compared in `round6_a5_parity.json`, and
summarized in `round6_audit_report.json`. The package is ready for one
independent read-only review; it does not authorize protected access.

The two CUDA runs are byte-identical. The CPU run is within the unchanged
Round-4 preregistered absolute/relative tolerances. `cleanStatic` alone is gate
evidence; `cleanDynamic` remains OOD diagnosis. Round-5's unsigned failed root
remains rejected, immutable, and excluded.

## Historical preregistration checkpoint

This directory is the isolated rerun root. The legacy
`artifacts/gcspo_stage0_static/` package remains byte-identical and is not a
runtime input to this rerun.

Status: preregistration and provenance inventory only. No detector, score, metric, plot, or scientific verdict has been produced.

The frozen experiment asks whether signed multi-PRN tracking innovations are better explained by a shared geometry-constrained position/clock and velocity/clock-drift state than by normal dynamics after an explicit likelihood and effective-DoF penalty. It is deliberately not a power, norm, residual-RMSE, or score-fusion detector.

Files in this checkpoint:

- `preregistration.json`: hypotheses, timelines, ablations, controls, metrics, gates, and future artifact/checksum contract.
- `config.json`: machine-readable frozen Stage-0 method skeleton.
- `data_inventory.json`: cleanStatic field/coverage audit plus metadata-only external scenario inventory and blockers.
- `source_commit.json`: authoritative repository and receiver/source provenance.

MainServer `192.168.0.77` holds the authoritative TEXBAT raw IQ under `/mnt/user/gnss-datasets/texbat/raw/`; it is host-accessible but the corresponding `/home/ubuntu/unraid` mount is absent on this VM. Preferred DS7/DS8 receiver roots and their available manifest, configuration, tracking, ephemeris, and NMEA hashes are now pinned; DS7 observables are pinned and DS8 has no observables file. The exact first 110 s of cleanStatic, DS7, and DS8 have identical SHA-256 and form one evidence family—never independent calibration, FPR, or bootstrap evidence.

Fail-closed blockers remain: DS4 end coverage/output provenance is incomplete; DS5/DS6 receiver outputs were not located; protected timing/geometry joins remain unopened; and the patched receiver source/build is not preserved as a clean immutable snapshot. Raw IQ availability does not cure the unsealed receiver build provenance. No past metric artifact and no protected attack feature/post-onset row was read by this implementation worker during the correction.

Phase 2 is intentionally out of scope. This directory must remain at the preregistration checkpoint until an external relay authorizes implementation.

External-review corrections also freeze GCMR as adapter-required on exact common support, the exact future artifact names and two-value verdict enum, mandatory implementation verifiers, protected-access preflight/trace requirements, and a transparent TEXBAT pre-exposure ledger. TEXBAT evidence remains developmental and requires confirmation on OAKBAT or a sealed new static session.

Semantic BLOCK resolution freezes the computational meaning before implementation: fixed-prior MAP state estimation with observation-only likelihood and one explicit EDF-BIC penalty; train-only residual/Gamma fitting; conditional closed-loop physical transfer; geometry rank/conditioning; chronological empirical thresholds; exact causal scheduling/persistence; replay exclusion; balanced pooling/bootstrap; fully computable ablations, destructions, and holdout controls; and validity separated from scientific verdict.

Any failed source, checksum, timeline, geometry, closed-loop transfer, test, or control-generation preflight yields `INVALID_EXPERIMENT_NO_ATTACK_ACCESS`, requires `invalid_run.json`, keeps protected rows sealed, and forbids `final_verdict.json`. Only a valid protected run can emit one of the two scientific verdicts. Exact verifier commands and report schemas are frozen in the JSON contract.

The final uniqueness correction removes remaining implementation degrees of freedom: Gamma is median/Ledoit-Wolf in pooled-whitened `z` coordinates with epoch-independent block-diagonal covariance; physical shifts use the exact preregistered `B`, one VAR transfer, and lower-Cholesky whitening; method-specific lambda uses common-window GCV; and A1/A2/A5, weighted pAUC, endpoint block assignment, nearest-rank bootstrap percentiles, and every seeded control are computationally exact.

Scientific invalidity, scientific NO-GO, and administrative verification are distinct. Invalidity seals attacks and emits the exact `invalid_run.json` set. Poor control alarm behavior is scientific G5 and can yield NO-GO only in a valid run. Packaging-only defects are `VERIFICATION_ADMIN_WARNING`, repairable without changing scientific evidence, but delivery still requires verifier PASS.
