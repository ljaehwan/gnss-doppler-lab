# Static reference-CGC v2 OAKBAT held protocol

## Purpose

V1 passed normal, OS2, OS4, and both static multipath checks but failed OS3.
The terminal v1 result is unchanged.  Its root-cause audit showed that OS3's
pre-onset receiver pseudoranges already formed ambiguity groups incompatible
with a trusted single static code reference.  V2 therefore adds one
**pre-onset eligibility gate**, not a new detector feature or a post-onset
repair.

OS5 and OS6 are the only held outcomes.  OS5 is the OAKBAT 2-microsecond
common-time push and OS6 is the 600-m ECEF-Z position push.  Their raw IQ
hashes, receiver binary, adapter, v1 thresholds, split, persistence, and gates
are frozen before either receiver is run for this experiment.

## Frozen reference eligibility

The candidate recording is scored only on `[40,110)` s first.  Its eligibility
statistic is the 99th percentile of baseline fitted displacement norm.  The
threshold is the same statistic from cleanStatic only:

\[
 Q_{0.99}\{\lVert\widehat{\mathbf d}(t)\rVert_2:t\in[40,110)\}
 \leq 75.6821493\ \mathrm{m}.
\]

If this fails, the runner records an abstention and must not inspect or score
the post-onset interval.  Abstention fails the held success gate; it is not
counted as correct rejection.

## Unchanged decision

- minimum eight PRNs and 30 observations per PRN-second;
- per-PRN linear nuisance trends fitted only in `[40,110)` s;
- stable post `[160,479)` s after the documented 120-s onset;
- raw alarm iff partial-F tail `<= 0.7077163431` and displacement norm
  `>= 89.1145618 m`;
- persistent alarm iff at least three of the latest five wall-clock seconds
  are positive, with missing seconds treated as negative.

## Held gates

1. OS5 and OS6 both pass reference eligibility;
2. OS5 stable-post persistent alarm rate is at most 5%;
3. OS6 first persistent alarm is no later than 60 s after onset; and
4. OS6 stable-post median displacement has ECEF-Z direction cosine at least
   0.8.

Every failure is terminal for v2.  No threshold, PRN, interval, correction,
or eligibility rule may change after post-onset access.  Even if all gates
pass, the claim remains static, retrospective OAKBAT transfer plus controlled
synthetic multipath; it is not field-wide spoof detection.
