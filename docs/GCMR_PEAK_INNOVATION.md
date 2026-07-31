# GCMR-PI: peak innovation relations

GCMR-PI consumes **only real early/prompt/late (E/P/L) tracking peaks**.  It
prompt-normalizes these three amplitudes safely, uses a shared PRN-identity-free
GRU to predict each PRN's next E/P/L vector, conditionally whitens residual
innovations by normal C/N0 context, and scores pairwise residual cosine errors
against a geometry-conditioned expected relation.  The pair condition is exactly
`[LOS dot product, min(elevation), max(elevation)]`; geometry is a condition,
not a learned PRN feature.  Variable visible-PRN cardinality is retained through
`N`, `N_eff`, `S_common`, pair score `S_pair`, and PRN loading count.

This is **not TRCD**. TRCD uses nine-tap B0 signed innovations and a leave-one-out
common direction. GCMR-PI uses E/P/L, conditional whitening, direct
geometry-conditioned expected pair relations, and variable-PRN `N_eff` diagnostics.
`Full` excludes the binomial-tail (`btail`) diagnostic; A0--A4 are emitted
separately.

## Reproducibility and gates

`train_gcmr_peak_innovation.py` fits only OAKBAT `cleanStatic` temporal normal
roles. Selection, whitening/pair calibration, and q99/q995/FPR1 thresholds are
normal-only. OAKBAT os1--os4 require `--open-attacks` and are inference-only.
It fails closed and writes `blocker_evidence.json` rather than fabricating
metrics if the existing relation events cannot be joined to genuine 3-tap data.

`eval_gcmr_peak_innovation.py` accepts only a frozen model, frozen thresholds,
and an authenticated normal EventRecord payload. It permits TEXBAT cleanStatic
or cleanDynamic only and labels the result a **contained normal-only protocol**;
no DS attack data are accepted and no fitting operation exists in the evaluator.

Artifacts are placed under `artifacts/gcmr_peak_innovation/`: configuration,
training/threshold/scenario/ablation JSON, score CSV, relation-destruction
column, plots for score/S_common/N_eff/S_pair/N/PRN loading, README, and
`SHA256SUMS`. Verify with `sha256sum -c SHA256SUMS` and recompute metrics from
the score CSV before making numerical claims.

Allowed claim after an actual completed campaign: performance of this frozen,
normal-only, specific-data implementation. Not allowed: universal spoofing
detection, external TEXBAT validation from a contained protocol, comparison to
B0/TRCD/GCMR without matched frozen runs and identical scenario splits, or
claims based on blocked/synthetic artifacts.
