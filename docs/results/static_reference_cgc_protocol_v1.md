# Static reference-CGC protocol v1

## One question

For a stationary single-antenna receiver, does applying the existing
clock-centered CGC nested model to **pre-attack-referenced code residuals**
reject normal, common time-push, and satellite-specific multipath while
detecting a coordinated position push?

This is not a new learned model.  It changes only the delay observable.  The
prompt-local nine-tap observable remains appropriate while two correlation
lobes overlap, but it loses an absolute displacement after the tracking loop
recenters.  The static reference observable retains that displacement:

\[
 z_i(t)=\{\rho_i(t)-R_i(\mathbf x_0,t)\}
        -\widehat b_i(t),
 \qquad
 z_i(t)\simeq-\mathbf u_i^T\mathbf d+c,
\]

where \(\widehat b_i(t)\) is a per-PRN linear nuisance trend fitted only on
the frozen clean interval `[40,110)` s.  NMEA is used only to establish the
static reference position before onset; it is never a detector score or
decision input.

## Frozen split and decision

- Minimum support: eight PRNs, at least 30 pseudoranges per PRN per second.
- OAKBAT clean baseline: `[40,110)` s.
- Clean-only threshold calibration: `[134,344)` s.
- Disjoint held-clean test: `[361,479)` s.
- OAKBAT attack onset: 120 s; `[110,160)` is excluded; stable post is
  `[160,479)` s.
- The lower 5% clean quantile of the support-normalized partial-F tail and
  upper 95% clean quantile of displacement norm are frozen jointly.
- A raw alarm requires both threshold crossings.  A persistent alarm requires
  at least three positive wall-clock seconds in the latest five seconds;
  unavailable seconds count as negative rather than compressing time.

OS2 and OS3 are common 2-microsecond time pushes and therefore test the
clock-only null.  OS4 is the primary 600 m ECEF-Z position-push positive.
Two previously generated static matched receiver-RF pairs provide
satellite-specific multipath controls.  Their comparison is limited to the
five supported seconds `[24,29)` and is a controlled mechanism result, not a
field multipath rate.

## Success gates fixed before static score access

1. at least ten clean calibration bins;
2. held-clean persistent alarm rate at most 5%;
3. OS2 and OS3 stable-post persistent alarm rate at most 5% each;
4. OS4 produces a persistent alarm no later than 60 s after the documented
   120 s onset; and
5. neither matched multipath pair produces a persistent alarm.

Every failed gate is retained.  The outcome is retrospective with respect to
the already opened OAKBAT recordings and cannot be called an untouched field
test.  It also does not replace the nine-tap aperture analysis: it tests the
static post-recentering regime identified by the FGI observability audit.
