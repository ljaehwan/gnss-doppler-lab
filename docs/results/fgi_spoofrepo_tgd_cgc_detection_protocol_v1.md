# Frozen FGI-SpoofRepo TGD CGC detection protocol v1.1

Date frozen: 2026-08-31

## Score-free adapter restart

The committed v1 release stopped before loading the delay estimator or reading tap values; its pinned state is `released_before_score_access` with `score_accessed=false`. The reusable OAKBAT preflight expects the minimum positive observables `RX_time`, whereas FGI stores its first positive value only when pseudorange becomes valid at TOW 480024.02 s. V1.1 uses that value solely for observables/ephemeris validation and keeps RF-start TOW 480006.0 s for NMEA position, LOS, and every detector bin. The corresponding attack onset in the observables reference is 119.98 s. No signal, PRN, interval, template, threshold, persistence, endpoint, or success gate changes from v1.

## Question and boundary

Apply the existing complex-nine-tap, support-normalized CGC detector once to
the unopened tap values of the FGI-SpoofRepo targeted DFMC recording.  The
primary question is whether the same frozen rule simultaneously keeps the
authentic pre-attack false-alarm rate low and persistently detects the
post-onset targeted synchronous spoofing signal.

This is one real RF-front-end, live-sky-plus-simulator composite recording.  It
does not contain a separately labelled multipath class and cannot establish
universal spoof detection or real multipath-versus-spoof classification.

## Frozen input and geometry

The RF file, 240 s GNSS-SDR receiver output, receiver configuration, complex
nine-tap schema, observables, NMEA, and end-of-run GPS ephemeris are pinned by
SHA-256.  The score-free v3 preflight passed with twelve healthy decoded GPS
PRNs in both analysis regions.  Only those twelve PRNs may enter the detector.

The recording start is GPS TOW 480006.0 s, verified from the receiver
observables.  Broadcast ephemerides must pass the existing week, age, and
`SV_health == 0` checks.  Receiver ECEF is the median of checksum-valid GGA
positions in receiver seconds 40--120 and is held fixed.  Post-onset PVT is
never used for LOS.

## Frozen transformation and detector

For each supported PRN, apply the unchanged nearest-template signed-delay
estimator to every complex nine-tap epoch.  Aggregate by the median within each
one-second PRN bin.  A PRN requires at least 40 epochs, matching 80% occupancy
at the 50 Hz telemetry-synchronized dump cadence.  Duplicate receiver channels
for the same PRN/bin are median-consolidated and never counted as extra
satellites.

For bins with at least eight healthy PRNs, fit

`H0: y_i = c + e_i`

and

`H1: y_i = -u_i^T d + c + e_i`.

The support-normalized statistic is

`F = ((SSE0-SSE1)/3) / (SSE1/(N-4))`,

with upper-tail probability `p_F`.  A raw alarm is
`p_F <= 0.06028418845288192`; no FGI calibration is allowed.  A persistent
alarm requires at least three raw alarms in the latest five consecutive bins.

## Frozen intervals and gates

- Clean primary region: 40--120 s.
- Transition excluded from rate endpoints: 120--160 s.
- Stable post-onset primary region: 160--230 s.
- Documented attack onset: approximately 138 s; first persistent-alarm latency
  from this onset is descriptive.

The terminal result is `SUPPORTED` only if all conditions hold:

1. at least 60 geometry bins exist in each primary region;
2. the clean persistent-alarm rate is at most 5%;
3. the stable-post persistent-detection rate is at least 80%; and
4. the stable-post median `p_F` is below the clean median `p_F`.

Serial-bin AUC, raw-alarm rates, fitted displacement, and onset latency are
secondary descriptive quantities.  No PRN, bin, interval, threshold, template,
persistence rule, or gate may change after release.

