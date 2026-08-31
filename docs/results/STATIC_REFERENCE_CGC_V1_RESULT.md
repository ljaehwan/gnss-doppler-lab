# Static reference-CGC v1 terminal result

## Decision

**`STATIC_REFERENCE_CGC_NOT_SUPPORTED`**.  The run was executed from the
pre-score release commit `79800239cd4293a45732811912e24b755f4860f3`; all
input hashes, splits, thresholds, persistence, and gates were committed and
pushed before score access.  The complete local result is
`artifacts/static_reference_cgc_v1/summary.json` (SHA-256
`c37b66c1abc659f6e1105ea756ec965b3e1a13e100e4c1284bf23477ea3543d8`).

Six of seven gates passed.  The one failed gate is retained rather than
relabeling or retuning the experiment.

| Frozen check | Result | Gate |
|---|---:|:---:|
| Clean calibration support | 21 bins | pass |
| OAKBAT cleanStatic held persistent FAR | 0/61 (0%) | pass |
| OAKBAT OS2, 10-dB time push | 0/319 (0%) | pass |
| OAKBAT OS3, 1.3-dB time push | 319/319 (100%) | **fail** |
| OAKBAT OS4, 600-m ECEF-Z position push | first alarm 163 s, delay 43 s | pass |
| Static multipath pair 001 | 0/5 persistent alarms | pass |
| Static multipath pair 002 | 0/5 persistent alarms | pass |

The clean-only joint rule was

- support-normalized partial-F tail at most `0.7077163431`; and
- fitted displacement norm at least `89.1146 m`;
- followed by three positive wall-clock seconds in the latest five.

OS4 produced 79 persistent positive bins among 288 available stable-post
bins.  The two matched receiver-RF multipath controls instead had median
clock-centered residuals 0.9883 and 0.7323, with median apparent displacement
1.97 m and 10.15 m, respectively.

## Why OS3 failed the clock-only interpretation

The follow-up audit did not change or rerun the frozen decision.  It inspected
the per-PRN inputs and found that OS3 was already incompatible with a single
clock-only reference before attack evaluation.  At 100 s, its detrended
pseudoranges split into groups separated by about `918 km`; at 160 s the split
was about `5,744 km`, and it grew thereafter.  The common component itself is
absorbed by the clock nuisance, but the PRN-dependent group split projects
onto the LOS columns and creates a median fitted displacement of
`27,076 km`.  Thus the receiver output did not present the nominal
two-microsecond stimulus as a common observed code shift.

This is not evidence that OS3 is a position carry-off.  It means the v1
protocol wrongly assumed that every recording supplied a trustworthy static
pre-attack code reference.  A subsequent version must determine reference
eligibility from pre-onset data and **abstain** on ambiguity-group or reset
failures; it may not count such an abstention as correct clock-push rejection.

## Implication for the paper

The useful result is narrower than a completed detector claim:

1. the same CGC geometry law works with an absolute static code reference for
   a large coordinated position push;
2. clock centering correctly rejects OS2 and both independent multipath RF
   controls; and
3. reference integrity is a necessary observable-state gate.

The clean displacement threshold is also too high for the already opened FGI
70-m position change: all 70 stable-post bins remain below `89.1146 m` and
would be missed by this frozen joint rule.  Therefore v1 must not replace the
paper's prompt-local nine-tap result or be described as solving its FGI
observability loss.  The defensible next test is a pre-onset reference-quality
gate frozen on development recordings, followed by one held common-time and
one held position-push recording (OAKBAT OS5/OS6) without threshold adaptation.

## Claim boundary

This is a retrospective static-receiver audit.  OAKBAT had been opened by
earlier detectors, the multipath controls are synthetic, and the differential
orbit model omits precise transmit-time, satellite-clock, Sagnac,
relativistic, ionospheric, and tropospheric corrections.  No operational or
general field spoof-versus-multipath claim follows from v1.
