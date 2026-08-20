# CORA-GNSS Stage-0: cross-PRN common origin

Configuration frozen before this CORA evaluation.

This first artifact commit freezes the alpha=0 complex fourth-cumulant experiment before attack TRACE or raw-IQ payloads are read. The freeze is a prospective evaluation freeze, not a claim of a fully blind preregistration; earlier TEXBAT work was developmental.

The primary score tests whether clean-conditioned residual complex tokens from at least four simultaneously tracked PRNs contain a rank-1 off-diagonal fourth-cumulant component. The complete equations, data split, grids, controls, ablations, thresholds, gates, seeds, and fail-closed rules are in `config.json`.

At this freeze stage no attack result, score, threshold outcome, or verdict exists. Running `python scripts/run_cora_stage0.py evaluate` is forbidden until the freeze commit is pushed and its local/remote SHA is recorded and verified.
