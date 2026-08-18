# CRISP Stage-0 static result

Verdict: `NO_GO_CRISP_PHYSICAL_HYPOTHESIS`.

## 1. Source and lineage

The input is authenticated TRACE-R2e phase-B receiver-native 1 ms
`TRC1MS02/v2` (416-byte) complex nine-tap output. The taps
`[E4,E3,E2,E,P,L,L2,L3,L4]` cover -0.5 to +0.5 chip at 0.125-chip spacing.
TEXBAT uses 25 MHz raw IQ and OAKBAT 5 MHz; both are signed little-endian
interleaved int16 I/Q. `data_inventory.json` binds every dump to raw-IQ,
receiver executable, config, patch, and manifest hashes. The authenticated
raw-recorrelation reference is MOSAIC commit
`b90d0f72bc6686e66dce06c617f1ca6895c0886b`. All six available inventories
and lineage checks passed. No replay or fallback was used. DS8, diagnostic/OOD
scenarios, Fixed9, and B0 exact were unavailable in this exact lineage.

## 2. Frozen representation

Full uses `P_i(t)=c_i(t)c_i(t)^H/(c_i(t)^Hc_i(t)+epsilon)` and the causal
`R_i(t)=P_i(t)-P_i(t-1)`. A shared PRN-agnostic ridge model predicts normal
motion and clean-calibration Ledoit-Wolf shrinkage whitens its innovation.
Scores use native 1 ms features, non-overlapping 20 ms blocks, the
second-largest valid PRN, at least four PRNs, and three consecutive q99
crossings.

## 3. Split and leakage

Separate TEXBAT/OAKBAT cleanStatic models use train `[30,170)` s, 10 s guard,
calibration `[180,310)` s, 10 s guard, and holdout `[320,end)` s. Train and
calibration each contain 150,000 sampled rows per dataset; each method's
threshold uses 6,500 calibration blocks. The chronological audit passed,
raw-sample overlap is false, and attack rows used for fitting/thresholds are 0.

## 4. Invariance

Gain 0.5/1/2, random common phase, NAV sign, common Doppler ramp, Prompt
amplitude-only common scaling, and arbitrary nonzero complex-scalar tests all
passed at `1e-10`. Maximum representation error was `2.58e-13`; Hermitian,
idempotence, and rank-one properties passed (idempotence error `1.46e-14`).

## 5. Core performance

| Scenario | coverage | pre-FPR | detection | delay (s) | Full pAUC |
|---|---:|---:|---:|---:|---:|
| DS3 | 0.9814 | 0.0000 | 0.0075 | 67.88 | 0.3209 |
| DS7 family | 1.0000 | 0.0000 | 0.4056 | 71.46 | 0.4734 |
| OS3 | 0.9999 | 0.0129 | 0.7508 | 0.46 | 0.8119 |
| OS4 | 1.0000 | 0.0200 | 1.0000 | 0.00 | 0.9792 |

Persistent alarm ratios after onset are DS3 0.00325, DS7 0.3659, OS3 0.4287,
and OS4 1.0. DS7 pull-off first-alarm delay is 31.46 s; DS3 has no post-pull-off
alarm. OAK OS3/OS4 pooled Full pAUC is 0.8737. Exact values are in
`supplemental_metrics.json` and `oak_os3_os4_pooled_metrics.json`.

Only OS4 met both the frozen 80% detection and 5 s delay gates; the required
three-of-four gate failed. The preregistration did not freeze separate
transition/established/persistent or pull-off windows, so none were defined
post-result. Per-block/per-PRN exports preserve the underlying scores.

## 6. Ablations

Normalized pAUC at FPR <=5%, using identical support and cadence:

| Scenario | A0 | A1 | A2 | A3 | A4 | A5 | A6 | Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DS3 | .0188 | .0759 | .0499 | .0903 | .3176 | .4327 | .0894 | .3209 |
| DS7 | .0333 | .1051 | .0536 | .2160 | .4721 | .1011 | .2160 | .4734 |
| OS3 | .2278 | .2978 | .2049 | .4920 | .7960 | .4554 | .4871 | .8119 |
| OS4 | .0844 | .3262 | .1837 | .4552 | .9790 | .9382 | .4554 | .9792 |

Full beat A1/A2 in all four scenarios. Paired 10 s bootstrap CIs show a Full
contribution over both A3/A4 in DS7 and OS3. Full was nearly A4 in DS3/OS4.
Fixed9/B0 are `UNAVAILABLE`; no historical number was copied.

## 7. False-positive rates

Clean holdout q99 FPR is 1.174% (TEXBAT) and 1.063% (OAKBAT). Scenario
pre-onset rates are DS3 0%, DS7 0%, OS3 1.286%, OS4 2.0%; worst external
static FPR is 2.0%, below the 5% gate.

## 8. Controls

Gain/phase/NAV/Doppler algebraic controls passed. Timestamp gaps and
reacquisition reset state; second-largest aggregation rejects one PRN glitch.
C/N0 below the frozen 28 dB-Hz floor and observed lock-loss examples were
`UNAVAILABLE`, not synthesized. Empirical noise scaling is diagnostic only.
The synthetic two-source test is formula sanity only and not verdict evidence.

## 9. Failure interpretation

DS3/DS7 retained adequate support and low pre-FPR but did not produce a
persistent q99 projective innovation. OS3 was strong but below the 80% rate;
OS4 was decisive. Projective rigidity loss therefore appears in some overlap
conditions but is not consistent across these four developmental attacks.
This does not reconstruct or identify two peaks.

## 10. Verdict

`NO_GO_CRISP_PHYSICAL_HYPOTHESIS`: only `core_detection` failed among encoded
GO checks. No neural model was built and the detector was not retuned.

## 11. WCL scope

Claimable: a preregistered clean-only test on authenticated developmental
TEXBAT/OAKBAT static replays, exact nuisance invariance, and a negative result
for robustness across the available attacks. Not claimable: independent or
dynamic confirmation, DS8 replication, Fixed9/B0 superiority, source
decomposition/localization, or deployment readiness. Confirmation needs a new
sealed static receiver session.

## 12. One next action

Terminate CRISP as a neural Stage-1 path and do not retune this result.

## Reproduction

Two full evaluations from preregistration SHA
`9a3119fc77ad830f438d723a225274f19c43be90` produced byte-identical summary
metrics, bootstrap, and verdict. Detailed exports matched row/schema; the final
report renderer produced ten byte-identical PNGs on two consecutive runs.
maximum floating difference was `6.93e-14`, below `1e-12`. Runtime/heartbeat
metadata is excluded. See `reproduction_validation.json`.

## Git/provenance closure

The scientific verdict remains `NO_GO_CRISP_PHYSICAL_HYPOTHESIS`. Its result
commit is `cd36f1d2a3ab3ca64c85747db1e1463198a966ee`; this administrative closure is
performed in a descendant commit. The closure artifact intentionally does not
record its own commit SHA because a Git commit cannot stably contain its own
hash. No experiment was rerun and no result, threshold, score, metric, or plot
was recalculated or modified.
