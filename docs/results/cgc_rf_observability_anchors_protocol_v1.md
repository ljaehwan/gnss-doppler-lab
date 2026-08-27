# CGC RF observability anchors v1 protocol

## Question

Does the profile-level ordering `40 m below boundary, 60 m at boundary, 80 m
above boundary` reproduce after 25 MHz signal generation, fixed RF impairments,
and the frozen complex-nine-tap GNSS-SDR receiver?

This is a receiver-RF pilot on one previously used static satellite geometry.
It is not held-out geometry validation. The six conditions were selected before
this RF run by the profile screen and cannot be substituted after outcome
access.

## Matched design

All six conditions reuse the exact authentic GPS component, UTC, static
position, startup LOS, receiver seed, front-end reference, AWGN law, and
GNSS-SDR executable. Only two variables change:

- target spatial displacement: `40`, `60`, or `80 m` along ENU direction
  `[0.8, 0.6, 0]`;
- final counterfeit advantage: `-6 dB` or `+3 dB`.

Every spoof begins at 10 s, reaches its target position in 5 s, and reaches its
final power in 5 s from an initial `-30 dB`. The causal comparison interval
begins at 16 s. The pre-onset composed IQ must be byte-identical to the pinned
normal RF prefix.

The common independent-multipath receiver recording from the earlier fresh RF
test is reused byte-for-byte for every comparison. It was generated from the
same authentic geometry and frozen receiver reference.

## Score and evaluation

The signed-delay estimator and clock-centered residual are unchanged. For each
condition, calculate serial-bin AUC between the common multipath bins and the
condition's post-transition spoof bins using negative residual as the spoof
score.

For each power regime, the profile-screen ordering is reproduced only if:

1. both streams have at least eight eligible one-second bins;
2. 40 m AUC is below `0.80`;
3. 60 m and 80 m AUC are at least `0.80`; and
4. AUC strictly increases from 40 to 60 to 80 m.

All per-condition medians, separations, AUCs, PRN support, generated-IQ hashes,
receiver manifests, and gates are reported regardless of the outcome. No
threshold or anchor may change after release.

## Storage policy

Available root and SSD space is insufficient to retain twelve additional
1.495 GB intermediate IQ files. The existing authentic and multipath artifacts
are never deleted. For each condition, the runner generates the counterfeit
component and composed spoof RF, verifies hashes and prefix equality, runs
GNSS-SDR, writes the complete receiver output and condition score, and only
then deletes those two campaign-created intermediate IQ files. Their hashes,
sizes, trajectories, settings, manifests, and deterministic regeneration inputs
remain. A failed condition retains its intermediates for diagnosis.

## Claim boundary

Passing would reproduce the predicted ordering on one simulated static
receiver geometry. It would not yet establish a universal RF boundary,
real-multipath performance, a TEXBAT displacement mapping, live-field false
alarm rate, or an operational detector.
