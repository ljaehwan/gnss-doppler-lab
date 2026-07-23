# Baseline Algorithm: Clean-Calibrated Binomial Tail Gate

Date fixed: 2026-07-23

This repository treats `btail_max_507080_ewma075` as the current baseline detection algorithm for the TEXBAT 9-tap PRN-node GRU experiments.

## Detector identity

Base model:
- 9-tap morphology PRN-node GRU.
- The GRU remains the primary anomaly source; the gate summarizes cross-PRN simultaneous tail surprise.

Final baseline score:

```text
btail_max_507080_ewma075
```

Interpretation:
1. Estimate per-PRN node RMSE from the tap9-only GRU.
2. From cleanStatic + cleanDynamic, calibrate node-RMSE quantiles q50, q70, q80.
3. At each event/window with N tracked PRNs, count how many PRNs exceed each clean node threshold: K50, K70, K80.
4. Convert K-of-N exceedance counts into binomial-tail surprise: `-log10(P_clean[X >= K])`.
5. Combine the q50/q70/q80 tail surprises by taking the maximum: `btail_max_507080`.
6. Apply causal EWMA persistence with alpha 0.75: `btail_max_507080_ewma075`.
7. Calibrate detector thresholds only from cleanStatic + cleanDynamic event-score distributions.

## Why this is the baseline

The earlier hard support/quorum gate performed well but was more heuristic: it boosted when multiple PRNs crossed a fixed support threshold. The binomial-tail gate keeps the same physical intuition while replacing the heuristic with clean-calibrated rarity:

```text
How unlikely is it under clean data that this many currently tracked PRNs
would simultaneously sit in the clean node-RMSE tail?
```

This keeps the detector anchored to the tap9 morphology GRU while making the multi-PRN support term more generalizable and easier to defend in the paper.

## Fixed q99 baseline results

Clean threshold source: cleanStatic + cleanDynamic.
Pre-onset false positive rate is measured before each scenario onset.

- ds1: pre FP 0.0%, post detection 95.73%, first delay 25.04s, threshold 4.169878
- ds2: pre FP 0.0%, post detection 100.00%, first delay 10.37s, threshold 4.169878
- ds3: pre FP 0.0%, post detection 97.41%, first delay 19.45s, threshold 4.169878
- ds4: pre FP 0.0%, post detection 77.14%, first delay 14.04s, threshold 4.169878

## Artifact snapshot

Primary artifact directory:

```text
artifacts/ai_morph_gru_cleanStatic_q70_frame/binomial_tail_gate_probe_20260723/
```

Tracked summary files:
- `btail_persistence_metrics.csv`
- `all_scenarios_binomial_tail_metrics.csv`
- `btail_max_507080_ewma075_q99_baseline.csv`

Large per-event score dumps are intentionally left under the ignored artifact directory unless explicitly force-added for a paper freeze.

## Guardrails against overfitting

- Do not tune thresholds on ds1-ds4 attack labels.
- Keep threshold calibration clean-only.
- Keep the tap9-only GRU as the primary morphology detector.
- Treat the binomial-tail term as PRN-set support evidence, not a PRN-ID-specific rule.
- Preserve the earlier `ai_rmse_q_tau50_gate` result as an ablation/reference, not the new primary baseline.
