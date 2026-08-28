# CGC RF Tokyo replication v1 protocol

## Question

The five-geometry campaign produced one isolated reversal at Tokyo straight,
240 m, and +3 dB: nine-tap serial-bin AUC was 0.417 while direction cosine
remained 0.929. This follow-up asks whether that result is a repeatable
high-power discrimination blind spot or a single RF/multipath realization.

The already observed campaign motivates the question but is excluded from the
primary replication counts.

## Frozen design

- Geometry, UTC, motion, authentic component, LOS table, 240 m target vector,
  pull-off timing, power ramp, receiver binary, nine-tap offsets, template bank,
  binning, comparison interval, and score direction are inherited byte-for-byte
  from `cgc-rf-geometry-aperture-validation-v1`.
- Five new receiver-noise seeds and five new PRN-specific multipath seeds are
  fixed in the JSON configuration before any new outcome is read.
- Each receiver seed creates a new normal RF reference. Its authentic prefix is
  required to match the two paired spoof branches byte-for-byte before attack
  onset.
- Each replica has one independent multipath RF negative control and two spoof
  branches: -6 dB and +3 dB. All ten spoof conditions must be reported.
- Three- and five-tap central-subset scores may be emitted as secondary
  diagnostics. The primary decision uses only the frozen nine-tap score.

## Primary support

Every replica-power cell must contain at least eight comparison bins in both
the spoof and multipath streams and at least eight PRNs in every retained
comparison bin. A support failure remains a failed primary release; it cannot
be repaired by replacing a seed.

## Frozen decision

`SYSTEMATIC_HIGH_POWER_BLIND_SPOT` requires all of the following over the five
new replicas:

1. all five replicas pass support;
2. -6 dB AUC is at least 0.8 in at least four replicas;
3. +3 dB AUC is below 0.8 in at least four replicas;
4. +3 dB absolute direction cosine is at least 0.85 in at least four replicas;
5. the median within-replica AUC difference, AUC(-6 dB) minus AUC(+3 dB), is at
   least 0.20.

`SINGLE_REALIZATION_EXCEPTION` requires full support, both powers to pass AUC
in at least four replicas, +3 dB direction to pass in at least four replicas,
and no more than one +3 dB AUC failure. Every other supported pattern is
`MIXED_OR_UNRESOLVED`.

No threshold, seed, power, interval, estimator, tap subset, or decision rule may
be changed after release. The original Tokyo result cannot be added to the five
new primary replicas.

## Claim boundary

This experiment can establish repeatability only for the fixed Tokyo straight
LOS/motion geometry at 240 m under the specified synthetic receiver-RF chain.
It cannot establish a universal high-power law, field multipath performance, or
an operational detector.
