# CGC real-IQ detection protocol v1

## Question

Can the frozen clock-centered correlator geometry candidate produce an actual
spoof alarm on real receiver-IQ replays while avoiding alarms on separate clean
and high-correlation-distortion negative data?

## Freeze and data roles

- `cleanStatic [330,420)` is calibration only. Its lower 5% clock-centered
  residual quantile is the only alarm threshold.
- `cleanDynamic` is a locked normal recording and is never used to fit the
  threshold.
- TEXBAT DS7 is the primary attack. Its CGC outcome was not accessed before
  this protocol and configuration were committed. DS7 is a carrier-aligned,
  matched-power time-push case.
- DS1--DS3 are secondary replication cases because their earlier threshold-free
  CGC direction results are already known.
- Stable attack regions are `[30,90)` pre, `[90,110)` excluded transition, and
  `[110,end)` post.

The candidate, complex signed-delay template estimator, LOS-plus-clock model,
and clock-centered residual are unchanged. All source and implementation inputs
are SHA-256 pinned in the configuration.

## Alarm

A one-second bin is a raw spoof alarm when

```text
clock-centered residual <= cleanStatic calibration q0.05
```

An operational persistent alarm requires at least three raw alarms among the
current and previous four consecutive one-second bins. The score is available
at the end of the current bin.

## Multipath-enriched stress subset

For each complex nine-tap epoch, compute early/late magnitude asymmetry from
the four symmetric tap pairs. Take the median per PRN and then q75 across PRNs
in each bin. A negative bin above the calibration q0.80 of this statistic is
called `multipath-enriched`. This label means correlation-distortion enriched;
TEXBAT provides no reflector/path ground truth, so it must not be reported as a
labeled multipath class.

## Primary gates

The primary decision is `REAL_SPOOF_DETECTION_SUPPORTED` only if all of these
hold in the single released run:

1. at least 80 calibration bins, 25 locked-normal bins, 50 DS7 stable-pre bins,
   and 150 DS7 stable-post bins are scoreable with at least eight healthy PRNs;
2. pooled `cleanDynamic + DS7 stable-pre` persistent alarm rate is at most 5%;
3. the multipath-enriched subset of that pool has persistent alarm rate at most
   10%;
4. DS7 has a persistent post-onset alarm no later than 60 seconds after the
   nominal 100-second onset.

Raw alarm rates, serial-bin AUC, DS1--DS3 results, and all calibration metrics
are reported descriptively. No threshold, interval, persistence rule, source
selection, or gate may change after release.

### Input-support repair

The first release stopped before computing the DS7 CGC score because the DS7
NPZ omits the optional `cn0_db_hz` diagnostic array. Its required complex I/Q,
time, PRN, channel, sample-count, and segment arrays are present with the frozen
shapes. The adapter was changed only to accept the diagnostic array as optional;
the estimator, score, calibration interval, threshold quantile, persistence,
regions, gates, and source roster are unchanged. The failed release state is
preserved separately and declares `metrics_emitted=false`.

## Claim boundary

This is real replay-IQ detector evidence, not live-field false-alarm validation.
The normal data contain natural receiver and propagation effects but no
ground-truth multipath annotations. A future controlled-reflector or labeled
urban dataset remains necessary for a strict real multipath-versus-spoof
accuracy claim.
