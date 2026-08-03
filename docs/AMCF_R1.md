# AMCF-R1 correction campaign

## Scope and evidence status

AMCF-R1 consumes only the pinned nine-tap complex NPZ exports for cleanStatic,
DS1, DS2, DS3, DS7, and DS8. It does not replay raw IQ or a receiver. DS4 is
`NA` because its producer is incompatible. DS1--3 are developmental and DS7/8
are explicitly post-exposure exploratory; none is confirmatory evidence.

The clean roles are chronological: train `[0,240)`, validation `[250,330)`,
calibration `[340,410)`, and clean test `>=420`. Boundary gaps are excluded.
These are held-out chronological segments, **not independent datasets**.
Quality gating, model fitting, early stopping, selector modeling, and q99/q99.5
threshold calibration are clean-only. Attacks are never used for tuning.

## Signal and causal contract

Tap order is `E4,E3,E2,E,P,L,L2,L3,L4`. Prompt index 4 is reference/current
quality context only: it is never masked, predicted, queried, or scored. Each
raw row is normalized as `C*conj(P)/(|P|^2+epsilon guard)`. A q0.005 Prompt
magnitude gate is fitted on clean train. Only a low-P raw row is rejected; its
PRN or whole window is not dropped. Outputs audit raw/valid/rejected counts,
PRNs, phases, windows, tracked N, and unique accepted-raw utilization.

For each PRN, a decision window contains every valid row in `(T-1,T]`, with a
0.5 s stride. A clean window is emitted only when all source rows have the same
role. History is exactly 12 strictly previous windows for the same recording,
PRN, and role; warm-up windows with fewer than 12 are excluded from every split. Current/future values cannot enter history. Summaries use fixed
median and `1.4826*MAD`, phase concentration, valid fraction/count, E/L temporal
signals, and current Prompt `log|P|`/valid fraction (plus C/N0 if available).
PRN identifiers are grouping keys only.

Complex and magnitude representations have the same seven-feature tensor and
same model/output shape. Complex retains median Re/Im, component MADs,
magnitude, phase concentration, and fraction. Magnitude maps magnitude and MAD
to fixed slots while imaginary/phase-bearing slots are deterministic zero.

## Model, masks, and query policies

The sole model class is end-to-end PyTorch: one-layer GRU, coordinate-conditioned
tap tokens, one two-head masked self-attention layer, and a diagonal Student-t
(df=4) decoder. Default hidden size is 32. AdamW runs at most 50 epochs with
patience 8; the finite best validation checkpoint is restored. Weights,
optimizer state, history, finite-gradient status, and seeds 101/202/303 are
saved. Complex and magnitude use identical architecture, splits, dimensions,
seeds, and training contract. Training/inference uses CUDA when available;
windowing/bootstrap remain CPU work.

K3 means the budget E/P/L, but Prompt is context, leaving E and L as scored side
taps. E and L are each scored leave-one-side-out before reveal. K5 adds E2/L2;
K7 then adds E3/L3. Random extras use policy seeds 11/23/37 and deterministic SHA-256 of
(recording,time,PRN,policy seed), epoch-wise and value-blind for every model seed. IG receives only an
observed dictionary, candidates, model/history/context. With 8 clean-model Monte
Carlo samples it maximizes expected global entropy reduction, never scale
argmax and never a candidate's true value. A selected tap is scored under the
pre-reveal distribution and only then revealed. All9 predicts each of eight
side taps leave-one-out and aggregates order-invariantly.

Ablations include magnitude K3, complex K3, magnitude/complex all9, phase
destruction, temporal-order shuffle, complex IG K7, fixed/random/IG K5/K7, and
B0 exact. Phase destruction permutes raw normalized side-tap relative phases
independently by tap, preserving row magnitudes and each tap phase marginal, then
rebuilds both current windows and their histories. Temporal shuffle uses a
(recording,time,PRN,model-seed) hash permutation of the 12 positions (not reverse). B0 scores may be reused, but every alarm is recomputed. All seeds,
mean, and standard deviation are reported; there is no best-seed selection.

## Decisions and output contract

The primary is fixed before attacks as the three-seed mean complex IG K7. q99
is primary and q99.5 diagnostic; both use clean calibration only and strict
`>`. Matched-clean is a separate diagnostic. Every per-epoch file contains
exactly `alarm_primary_q99`, `alarm_primary_q995`, and
`alarm_matched_clean_diagnostic`; regeneration verifies every score/threshold.
Metrics include exact binomial intervals, 10 s clean-FPR block bootstrap, and
scenario detection bootstrap. Query diagnostics report unique ordered paths,
modal fraction, tap frequencies, phase/scenario entropy, collapse at >=95%,
time, budget, and correlation fraction. Budget claims mean offline replay only,
not measured SDR savings.

`scripts/summarize_amcf_r1.py` alone regenerates C1--C9 and independent
Complex/Active/WCL GO/NO-GO decisions. The runner atomically emits config,
provenance, input hashes, environment, QA/rejections, histories/model audit,
thresholds, metrics/seed/ablation/query/per-epoch/bootstrap data, decision,
README, plots, checkpoints, and SHA-256 manifest.

Tiny real-data smoke example:

```bash
python scripts/run_amcf_r1_texbat.py --epochs 1 --seeds 101 \
  --scenarios DS1 --max-train-samples 32 --max-val-samples 16 \
  --max-eval-samples 200 --bootstrap-reps 5 --batch-size 200 \
  --out /tmp/amcf-r1-batched-correction
```

The corrected 400-record (200 clean + 200 DS1) RTX A6000 smoke measured 27.2 s
for 6,400 policy-record rows and estimates the full 3-seed/6-dataset campaign
at about 3.2 hours (2.93 h scoring + 0.19 h training + 0.08 h window/artifacts).

Full defaults intentionally require no `--smoke` flag and must be scheduled as
a campaign, not confused with the tiny smoke.
