# CGC receiver/RF unused-train replication v1

Date: 2026-08-26
Status: PASS on the preregistered unused-train replication gate

## Question

Does the post-hoc clock-centered correlator-geometry consistency residual from
train pair 001 preserve its physical direction across five previously unused
train geometries after independent multipath and carry-off spoofing pass
through the same RF front end and complex nine-tap GNSS-SDR receiver?

This experiment is a replication gate, not detector validation.

## Preregistration and data boundary

The protocol was frozen before signal processing of pairs 002--006:

- protocol:
  `docs/results/cgc_rf_train_replication_protocol_v1.md`
- config: `configs/experiments/cgc_rf_train_replication_v1.json`
- config SHA-256:
  `ee6a5fe713dfeaec05c9ec8688fb68f4443e575768a58af963eeb502a396f596`
- protocol commit: `bfa25c2`
- deterministic runner commit: `c2ce2fa`

Only train pairs 002--006 were authorized. Pair 001 was available only through
its frozen pilot records. Validation pairs 007--009, test pairs 010--012, and
TEXBAT attack recordings were not accessed.

The score, estimator, one-second aggregation, comparison boundary, seeds,
bootstrap, and support gates were not changed after outcome inspection. No
threshold was fitted.

## Execution note

The first simulator invocations stopped before generating IQ because the C
extension rejected the long absolute `-m` filename. A later defensive path
check also stopped before simulator execution. The final fix passes the
runner-created `multipath.csv` basename under the simulator's already pinned
working-directory contract.

- simulator-safe staging commit: `d639207`
- final working-directory contract commit: `9237c9d`
- regression tests after the fix: 8 passed
- pre-signal failure and diagnostic directories were retained separately and
  are not part of the final artifact

These changes affect only file staging. Frozen multipath values, RF settings,
receiver settings, estimator, and evaluation criteria were unchanged.

## Primary result

For each pair, the primary separation is:

    median Rdir(multipath) - median Rdir(spoof)

where:

    Rdir = SSE_LOS+clock / SSE_clock-only
    spoof score = -Rdir

Positive separation is the preregistered physical direction.

| Pair | Domain | Eligible bins per scenario | Multipath Rdir | Spoof Rdir | Separation | Legacy separation |
|---|---|---:|---:|---:|---:|---:|
| 002 | static | 9 | 0.7769 | 0.0855 | +0.6915 | +0.0554 |
| 003 | dynamic straight | 9 | 0.8244 | 0.0127 | +0.8117 | +0.1673 |
| 004 | dynamic straight | 9 | 0.6589 | 0.2766 | +0.3822 | -0.1785 |
| 005 | dynamic circle | 7 | 0.3837 | 0.0548 | +0.3289 | +0.0446 |
| 006 | dynamic parallel sweep | 9 | 0.7143 | 0.0496 | +0.6647 | +0.1292 |

All five independent pair-level separations are positive.

## Frozen support gates

| Gate | Required | Observed | Result |
|---|---:|---:|---:|
| Completed authorized pairs | 5 | 5 | PASS |
| Positive clock-centered separations | 5 | 5 | PASS |
| Pair-block AUC | >= 0.80 | 1.00 | PASS |
| Pairs improved over legacy | >= 4 | 5 | PASS |
| Minimum eligible bins per scenario and pair | >= 5 | 7 | PASS |

Additional preregistered summaries:

- median pair separation: 0.6647
- 10,000-resample pair-bootstrap 95% percentile interval: 0.3289--0.8117
- pair-block AUC bootstrap interval: 1.00--1.00
- secondary serial-bin AUC, clock-centered: 0.9811
- secondary serial-bin AUC, legacy zero-referenced: 0.6739

The bin-level AUC is descriptive because one-second bins within a scenario are
serially dependent. The primary independent unit is the pair.

## Independent verification

The five-row `pair_summary.csv` was read independently of the result writer.
Pair IDs, all five separations, positive count, improvement count, median
separation, pair-block AUC, and minimum bin count matched the JSON result
exactly.

Primary result SHA-256:

- `summary.json`:
  `5d9e82dcb3576ac1509870648901abc9d16048f163d85da74fc98a0f31b7d3fe`
- `pair_summary.csv`:
  `ed6cbf1bcc87ae8aa274d06a2cb012170def892aa6e2570d0392996e68842165`
- `pair_scenario_medians.csv`:
  `4eeff09f6b45033da3f03230b7a66b34fd67847e062a236926063d5c1b55d61b`
- `geometry_scores.csv`:
  `49e29fcb943f272f1ca7ead9c13a59e8af2e045add8f13aa21bf6eb607af457f`
- `delay_estimates.csv`:
  `55bdc988e856d855bfb1ea10ae89e26eb346d7a0ff0020ae9e173c62b7f94dd2`

## Physical interpretation

A coherent carry-off spoof induces a cross-satellite delay pattern that the
LOS geometry plus a shared receiver-clock term can explain, so its directional
residual is small. Satellite-specific multipath delays do not share that LOS
pattern, so their residual remains larger.

Centering the null model on the fitted clock removes the common-delay energy
that confounded the legacy zero-referenced normalization. The fact that the
direction holds in all five static and dynamic geometries supports this
mechanism beyond the single exploratory pilot.

## Claim boundary

The result supports the clock-centered physical mechanism on five unused
train simulations. It does not establish:

- a validated detector or operational threshold
- false-alarm or detection probability
- generalization to still-locked test pairs
- generalization to real receiver data or TEXBAT attacks
- a completed WCL contribution

The sample contains only five independent simulated pairs from one generation
and receiver toolchain. The bootstrap interval is therefore descriptive, and
the perfect AUC must not be presented without the pair count and simulation
boundary.

## Next gate

Before any test access, freeze a final protocol for still-locked test pairs
010--012. The preprocessing, score direction, aggregation, and any threshold
rule must be fixed without looking at those outcomes. The previously inspected
validation partition cannot validate this post-hoc candidate.

## Artifacts and SSD archive

Local primary artifact:

    artifacts/cgc_rf_train_replication_v1

It contains 345 regular files totaling 15,831,022,225 bytes.

SSD archive:

    /home/ubuntu/ssd_data/gnss-early-detection/artifacts/cgc-rf-train-replication-v1

A full `rsync --checksum --dry-run --itemize-changes` produced no differences.
The SSD tree byte count is 15,831,022,225 and its `summary.json` SHA-256 is
`5d9e82dcb3576ac1509870648901abc9d16048f163d85da74fc98a0f31b7d3fe`.
