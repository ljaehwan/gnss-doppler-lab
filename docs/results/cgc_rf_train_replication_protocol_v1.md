# CGC receiver/RF unused-train replication protocol v1

Date frozen: 2026-08-26
Status: preregistered before signal-level access to train pairs 002--006

## Objective

Test whether the clock-centered directional residual discovered post-hoc on
train pair 001 keeps the same multipath-versus-carry-off direction across five
unused train geometries. This is a replication gate, not validation.

## Frozen candidate

The primary residual remains:

    Rdir = SSE_LOS+clock / SSE_clock-only
    detection score = -Rdir

No feature, delay-template, clock treatment, or score-law change is permitted
after inspecting pair 002--006 outcomes. The earlier zero-referenced residual
is retained only as a preregistered comparator.

## Authorized data

Only train pairs 002--006 are authorized:

- pair 002: static Singapore
- pair 003: straight dynamic Ottawa
- pair 004: straight dynamic Pretoria
- pair 005: circular dynamic Berlin
- pair 006: parallel-sweep dynamic Santiago

Pair 001 may be read only through its already frozen pilot records. Validation
pairs 007--009, test pairs 010--012, and TEXBAT attack recordings remain out of
scope.

## RF and receiver controls

For each authorized pair:

1. Reuse its exact startup time, position, authentic trajectory, front-end
   configuration, fixed normal reference, AWGN seed, and existing carry-off RF.
2. Generate one new satellite-specific multipath component for every PRN in the
   frozen authentic startup LOS table.
3. Draw delay, amplitude, and phase deterministically from the pair-specific
   preregistered seed.
4. Pass multipath and carry-off RF through the same complex nine-tap GNSS-SDR
   receiver configuration.
5. Fail closed on every config, executable, patch, trajectory, RF, receiver,
   and analysis hash mismatch.

Dynamic authentic trajectories must be reused byte-for-byte from the original
paired-train component record. A static substitute is forbidden.

## Aggregation and primary endpoint

Per-PRN delay estimates are aggregated into one-second bins with at least eight
PRNs. The comparison begins one second after both spoof transition and power
ramp are complete.

For each pair and scenario, the primary value is the median Rdir across eligible
bins. Bins are serially dependent and are not primary independent samples.

The primary separation is:

    median_Rdir(multipath) - median_Rdir(spoof)

Positive values support the physical hypothesis.

## Frozen support gates

All conditions must hold:

1. Exactly five authorized pairs complete.
2. All five pair-level clock-centered separations are positive.
3. Pair-block AUC from the ten scenario medians is at least 0.80.
4. Clock-centered separation exceeds legacy separation in at least four pairs.
5. Every scenario contributes at least five eligible comparison bins.

Pair-block bootstrap intervals are descriptive and use 10,000 pair resamples
with seed 2026091299. No detector threshold is fitted.

## Claim boundary

Passing this gate supports replication across unused train geometries only.
It does not validate a detector. Because the previous validation partition was
already inspected before this post-hoc candidate existed, a later final test
protocol must be frozen before accessing still-locked test data.
