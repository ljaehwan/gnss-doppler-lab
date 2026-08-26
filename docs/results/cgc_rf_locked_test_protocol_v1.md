# CGC RF locked test v1 protocol

Date frozen: 2026-08-26
Status at freeze: SEALED; test signals 010--012 unopened

## Scientific question

On three held-out receiver-RF simulations, does the frozen clock-centered
correlator-geometry consistency (CGC) score rank coherent carry-off spoofing
above satellite-specific multipath?

This is deliberately narrower than an operational spoof detector. The earlier
train-normal audit showed that one absolute score threshold does not transfer
reliably across receiver motion and satellite geometry. That failed threshold
is not being repaired or carried into this test. The final test concerns the
relative physical signature only.

## Why this is the final held-out set

The clock-centered candidate was proposed after a train pilot and then
replicated without modification on unused train pairs 002--006. Validation
pairs 007--009 were inspected in earlier development, so they cannot be
presented as independent evidence for this post-hoc candidate.

The only still-locked partition is test pairs 010--012. Before this protocol
was committed, their split metadata were known but their signal artifacts,
tracking outputs, CGC scores, and outcomes were not generated or opened.

| Pair | Motion | Site metadata | Spoof transition | Comparison starts |
|---|---|---|---:|---:|
| 010 | Static | Wellington | 10 s | 21 s |
| 011 | Straight, 12 m/s | Mexico City | 12 s | 23 s |
| 012 | Parallel sweep, 9 m/s | Oslo | 11 s | 22 s |

The locations are simulation parameters, not field recordings.

## Frozen physical idea and score

The nine complex tracking taps are used to estimate a delay displacement for
each tracked satellite. A coherent carry-off spoofer imposes a delay pattern
that is largely shared across satellites after accounting for satellite
geometry and a receiver-clock-like common mode. Ordinary multipath is
satellite-specific and therefore should not fit that shared low-dimensional
pattern as well.

For each eligible one-second bin, the frozen residual is

    R_c = SSE_LOS+clock / sum_i w_i (delay_i - weighted_mean(delay))^2

and the detection score is `-R_c`. Smaller residual therefore means a more
spoof-like coherent geometry. No classifier is fitted in the test.

For each pair, the primary separation is

    Delta_pair = median(R_c,multipath) - median(R_c,spoof)

so a positive value supports the predicted direction. The primary statistical
unit is the whole pair, not the serially dependent one-second bins.

## Frozen signal-generation contract

The exact split rows, normal RF profile, paired signal-generation law,
multipath simulator, GNSS-SDR receiver, nine-tap configuration, seeds, and
analysis rules are hash-pinned in
`configs/experiments/cgc_rf_locked_test_v1.json`.

Each test pair produces source components under the same pinned law as the
train campaign. Its spoof member shares the authentic component, time,
position or trajectory, receiver seed, front-end, impairments, noise law, and
pre-onset samples with its normal reference. The independent multipath member
reuses the exact authentic trajectory and applies a deterministic
satellite-specific reflection to every valid startup LOS PRN:

- delay: uniform deterministic draw in 0.12--0.45 chips;
- amplitude: uniform deterministic draw in 0.20--0.70;
- phase: SHA-256-derived uniform draw in [-180, 180) degrees;
- seeds: 2026091310, 2026091311, and 2026091312 for pairs 010--012.

Test manifests and their hashes will be created only after this protocol and a
conforming runner implementation are committed. They are provenance records,
not permission to modify this protocol.

## Frozen processing

- GNSS-SDR uses 11 channels and nine complex taps at 0.125-chip spacing.
- Scores use one-second bins with at least eight PRNs.
- Comparison begins at spoof start plus the greater of transition duration and
  power-ramp duration, plus one second.
- Each pair contributes one multipath median and one spoof median.
- One-second-bin AUC and plots are secondary descriptive results only.
- The normal stream may be retained for generation provenance, but its scores
  cannot calibrate a threshold, alter the candidate, exclude a pair, or enter
  the primary endpoint.

## Decision fixed before release

The claim is `SUPPORTED` only if every gate passes:

| Gate | Frozen requirement |
|---|---:|
| Complete held-out pairs | 3 |
| Pairs with positive clock-centered separation | 3/3 |
| Pair-block AUC | at least 0.80 |
| Pairs where clock-centered separation exceeds legacy separation | at least 2/3 |
| Eligible comparison bins in each scenario and pair | at least 5 |
| Startup LOS PRNs in each pair | at least 8 |

The pair-block bootstrap uses 10,000 repetitions and seed 2026091399. With only
three independent pairs, its interval is descriptive and will not replace the
fixed gates.

No absolute threshold, threshold selection, model refit, normalization fit,
pair exclusion, seed replacement, endpoint change, or post-release retest is
allowed. Failure of any gate yields `NOT SUPPORTED` for this frozen claim.

## Release and failure handling

The test remains sealed until a runner that exactly implements the pinned
config is reviewed and committed. After that commit, test generation and
analysis may run once.

An infrastructure failure before any test metric is emitted may be corrected
only by a documented code-only repair that leaves every scientific input and
rule unchanged; the repair must be committed before resuming. Once any test
metric is emitted, no rerun or repair may replace the outcome. Partial or
missing results count against the completeness gate unless an independently
verifiable infrastructure failure occurred before metric emission.

## Claim boundary

A passing result would establish held-out simulated receiver-RF evidence that
the clock-centered physical consistency score separates the selected
satellite-specific multipath family from the selected coherent carry-off
spoofing family. It would not establish a transferable alarm threshold, field
false-alarm rate, TEXBAT performance, operational deployment readiness, or a
general WCL claim by itself.

The train-normal threshold audit remains a negative result and must be reported
alongside this test regardless of its outcome.
