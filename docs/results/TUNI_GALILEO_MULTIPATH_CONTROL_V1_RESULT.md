# TUNI Galileo multipath-control result v1

## Decision

The preregistered detector evaluation is **INSUFFICIENT_SUPPORT**.  The TUNI
README files state that SS-11, SS-12, and SS-13 contain authentic Galileo E1
signals together with the listed spoofing PRNs under multipath.  However, no
authentic PRN was trackable in the frozen first-10-second receiver interval.
Consequently, these recordings do not yet provide an analyzable multipath
negative control for the detector claim.

This is not evidence that the detector confuses multipath with spoofing.  It is
also not evidence of multipath specificity.  The negative class was absent
from the receiver output, so sensitivity, specificity, and AUC gates cannot be
interpreted.

## Preregistered evaluation

The release was executed from commit `4b15ba8` with the frozen clean model,
nine-tap receiver, thresholds, labels, and first 10.0 seconds of each scenario.

| Scenario | README spoof PRNs | Tracked PRNs | Authentic tracked PRNs |
|---|---:|---:|---:|
| SS-11, multipath | 31 | 31 | none |
| SS-12, multipath | 9, 31 | 31 | none |
| SS-13, multipath | 5, 9, 23, 31 | 31 | none |

Only two target observations were available across the primary SS-12/SS-13
set, whereas the support gate required both SS-12 targets, at least three SS-13
targets, and at least two authentic PRNs in every multipath scenario.  The
reported authentic flag rate of `1.0` is the runner's conservative empty-class
guard value; it is **not** an observed 100% false-alarm rate.

Primary artifact SHA-256:
`5937971031aa74123935bc1aec0be84b6f28c7a173115940d40b9eeb3a4026a3`.
The release state records no post-release tuning or retest.

## Exhaustive PRN-support audit

To determine whether automatic channel scheduling caused the missing control
class, a separate input-support diagnostic explicitly assigned every Galileo
PRN a receiver channel while keeping the same data interval, acquisition
threshold, tracking settings, and receiver binary.

The initial v1 diagnostic, frozen at commit `ab442fb`, requested 36 simultaneous
Galileo channels.  GNSS-SDR v0.0.19 rejected that flowgraph before processing
samples because of its channel-count guard.  Its empty MAT sentinels are not
scientific observations.  The failed artifact is retained for auditability.

The v1.1 correction was frozen at commit `0812c4a` before any successful
fixed-PRN run.  It divided the unchanged roster into PRNs 1--18 and 19--36.

| Scenario | Any tracked PRNs (epochs) | Sustained spoof PRNs | Sustained authentic PRNs |
|---|---|---|---|
| SS-11 | 31 (1,643) | 31 | none |
| SS-12 | 9 (847), 31 (2,355) | 31 | none |
| SS-13 | 9 (1,501), 31 (2,428) | 9, 31 | none |

Sustained support was frozen at at least 1,000 valid epochs.  Even before that
cutoff, every nonempty tracking output belonged to a README-labelled spoof PRN.
The v1.1 artifact SHA-256 is
`6e2d174464419193aed03ef4027526d090743f07d2417224cb86d1a160a05439`.

## Claim boundary and next test

The data package contains a *metadata-level* same-stream control, but the
current receiver and interval do not expose a *usable signal-level* control.
The paper must therefore retain the claim boundary: separation of simulated
independent multipath from coherent simulated spoofing is supported, while
real multipath specificity remains unverified.

The next defensible experiment is a newly preregistered receiver-support study,
not detector threshold tuning.  It should scan later non-overlapping intervals
from the approximately 300-second recordings and, only if necessary, test a
fixed acquisition-sensitivity ladder.  Detector evaluation may resume only
after at least two authentic PRNs per multipath scenario are recovered with
sustained tracking under a frozen receiver setting.  Otherwise an external
multipath-only corpus is required.
