# PG-SCC R2 Terminal Fail-Closed Delivery

Implementation freeze: `05cb5a5421461bb90bafc6cd3683a119ddf09555`

The protected production audit was invoked exactly once. It exited with code 1 during plot generation when `_save_plots` attempted to read `mean_improvement_per_k` and raised `KeyError`. The producer was not rerun, and no model, mask, threshold, formula, support filter, score, output, or scientific definition was altered to rescue the run.

## Delivery status

- Protected attempt count: 1
- Terminal state: `TERMINAL_FAIL_CLOSED`
- Final scientific verdict: `UNAVAILABLE`
- Reproduction artifact status: `PASS`
- Common support: `PASS` with 260 eligible and 0 excluded events
- Committed verifier: `FAIL` because nine required deliverables are missing
- Committed tests: `PASS` (25 passed, 0 failed, 1 warning)

`root_cause_verdict.json` exists and passes a limited structural check. It contains the candidate `REPAIRABLE_BUT_REQUIRES_NEW_CONFIRMATION`, but the committed verifier did not validate it because verification stopped on missing deliverables. The candidate is preserved as partial evidence and is not promoted to a final scientific verdict.

## Missing required artifacts

- `README.md`
- `artifact_manifest_sha256.json`
- `plots/calibration_threshold_uncertainty.png`
- `plots/detector_performance.png`
- `plots/empirical_noise_awgn_response.png`
- `plots/learned_random_percentile.png`
- `plots/rss_score_dilution.png`
- `plots/seed_mask_stability.png`
- `plots/synthetic_real_delay_doppler.png`

## Exact protected command and failure

Command: `python3 scripts/run_pg_scc_root_cause_audit.py --implementation-sha 05cb5a5421461bb90bafc6cd3683a119ddf09555`

Exit code: `1`

Failure: `_save_plots` raised `KeyError: 'mean_improvement_per_k'`.

All pre-existing and partial producer artifacts were retained byte-for-byte. `terminal_fail_closed_delivery.json` contains their SHA-256 inventory and the complete machine-readable reconciliation.
