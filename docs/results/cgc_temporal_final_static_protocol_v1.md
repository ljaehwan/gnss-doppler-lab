# Temporal observable CGC untouched final-static protocol v1

## Question and release boundary

Does the development-selected detector remain useful on five untouched static
satellite geometries when counterfeit PRNs have independent initial carrier
phases? The four conditions are clean normal, independent per-PRN multipath,
carrier-coupled carry-off, and authentic-Doppler-locked code carry-off.

Fairbanks, Punta Arenas, Casablanca, Sapporo, and Prince George were selected
only by their preregistered one-second startup LOS counts (13, 12, 11, 13, and
13). No selected 25 MHz stream, multi-tap output, signed delay, Partial-F score,
or label was available before release. Pair substitution and outcome-driven
reruns are forbidden.

## Matched receiver-RF conditions

Every pair uses a 30 s static authentic component and a 100 m carry-off from 5
to 10 s. Counterfeit power ramps from -30 to +3 dB over 5 s. The counterfeit
PRNs receive deterministic independent initial carrier phases from the frozen
pair seed; coupled and Doppler-locked versions share that seed. In the locked
condition code follows the false path but carrier range/rate follows the true
static path. Truth invariants must prove these relationships before scoring.

The multipath control assigns every startup LOS PRN an independent delay in
[0.12, 0.45] chip, amplitude ratio in [0.20, 0.70], and phase in [-180, 180]
degrees. All four conditions use 25 MHz signed IQ, the same frozen frontend
parameters and gain reference, and the same seeded output-noise realization.
Normal/coupled/locked IQ must be byte-identical before 5 s.

## Frozen detector

Use the nine complex correlator taps at 0.125-chip spacing and the existing
deterministic two-path template bank to estimate one signed delay per supported
PRN and second. Apply a causal five-bin median independently to each PRN. A bin
is observable only when the cross-PRN centered delay RMS is at least 0.10 chip.
Only observable bins with `p_F <= 0.06028418845288192` are raw spoof alarms. A
persistent alarm requires at least three raw alarms among the latest five wall-
clock bins. Use every startup-LOS PRN that passes the existing estimator rules;
do not search PRN subsets.

Spoof hold bins begin at 12 s. Benign evaluation begins at bin 4 so every
continuous PRN can fill the causal window. At least eight PRNs are required per
scored bin. Abstention is reported separately and is not counted as a benign
classification.

## Frozen gates

The release is `SUPPORTED` only if all five truth audits pass, all 20
pair-condition streams contain supported scored bins, at least four of five
coupled and four of five Doppler-locked attacks produce a persistent alarm, no
normal pair ever produces a persistent alarm, at most one multipath pair does,
no spoof stream alarms persist before onset, median locked hold raw-alarm rate
is at least 0.50, median locked latency is at most 12 s, and locked-hold versus
multipath observable-gated Partial-F AUC is at least 0.90.

These gates are preregistered engineering criteria. A pass supports a static
simulated receiver-RF mechanism claim, not field detection probability,
moving-receiver performance, or universal superiority to Doppler detectors.
