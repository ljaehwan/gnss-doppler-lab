# MOSAIC Stage-0B R1a frozen scientific finalization

This is `FROZEN_POLICY_COMPLETION` over the existing 72-case R1 result set. No IQ injection, receiver replay, new case, feature, score, threshold, CAF grid, or subset was introduced. Final verdict: **NO_GO_MOSAIC_MULTI_PRN_RECOVERY**.

## Why the prior R1 bundle was not a science verdict

The prior compact bundle had an empty `control_metrics.csv`, an empty `bootstrap_intervals.csv`, and placeholder plots. Its single/four `status=PASS` meant receiver replay success rather than physics recovery. Its finalizer returned fixed `INCONCLUSIVE_PREREG_GATE_UNDERSPECIFIED` after all 72 cases, and the external `single_gate.json` was not preserved in the compact artifact. Those defects are documented here rather than hidden.

## Interpretation limits

The 72 cases repeat parameter variations over roughly a 12-second recording region; they are not 72 independent receiver campaigns. Every interval is labeled `DESIGN_CASE_BOOTSTRAP`: it describes stability over the frozen parameter grid only and does not establish generalization to new days, receivers, or environments. OAKBAT and TEXBAT are receiver/data-domain checks, not confirmation using attack data.

Controls are tap-domain diagnostics using retained clean/case complex taps, identical epoch/PRN/tap support, frozen CAF grid, and frozen BIC formula. C1 uses one scalar to RMS-match clean taps; C2 uses deterministic complex AWGN at the target H0 residual RMS with seed `20260818 + PRN`; C0 uses actual collapsed frozen-design cases; C3 uses the same-case median non-target ΔBIC.

The same-rho collapsed comparison has three eligible pairs per dataset. The frozen rho_db >= -6 strong rule makes the rho_db=-10 collapsed case non-estimable against a same-rho strong case; it remains reported in collapsed_source_metrics.csv but is excluded by the pre-frozen finite-value bootstrap behavior. This is an explicit design limitation, not a post-result substitution.
