# CORA-GNSS Stage-0: cross-PRN common origin

Configuration frozen before this CORA evaluation.

This first artifact commit freezes the alpha=0 complex fourth-cumulant experiment before attack TRACE or raw-IQ payloads are read. The freeze is a prospective evaluation freeze, not a claim of a fully blind preregistration; earlier TEXBAT work was developmental.

The primary score tests whether clean-conditioned residual complex tokens from at least four simultaneously tracked PRNs contain a rank-1 off-diagonal fourth-cumulant component. The complete equations, data split, grids, controls, ablations, thresholds, gates, seeds, and fail-closed rules are in `config.json`.

The freeze was pushed and independently resolved at `c226b942a82dbd63c6682e76e44b2aefe1c60156` before attack payload access. The completed raw-IQ evaluation returned `NO_GO_CORA_COMMON_ORIGIN_HYPOTHESIS`. Neural Stage-1, score fusion, and threshold retuning are not permitted. The main failures were clean-holdout and external pre-onset FPR, inadequate OAK/TEX family reproduction, lack of required ablation superiority, incomplete relation-destruction behavior, minimum-PRN LOPO scope, nuisance baseline shortcuts, and unavailable same-support B0 evidence. Raw IQ is not stored in Git. Reproduction commands and full source/cache binding are recorded in `raw_source_binding.json`.
