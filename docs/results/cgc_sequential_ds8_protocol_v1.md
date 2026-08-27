# CGC sequential DS8 v1 frozen protocol

## Question

Can a normal-calibrated sequential accumulator turn the fixed clock-centered
correlator-geometry residual into a timely alarm on an outcome-unseen TEXBAT
attack without alarming on normal replay-IQ or the attack recording before its
official onset?

DS1--DS3 and DS7 CGC outcomes have already been inspected and are development
evidence only. The DS8 complex-nine-tap input keys, shape, time extent, PRN
roster, and hashes were inventoried before this release, but its CGC residual or
alarm outcome was not calculated or inspected.

## Frozen score and sequential rule

The per-second physical residual remains
`SSE_LOS_plus_clock / SSE_clock_only`. No delay estimator, satellite geometry,
clock centering, bin width, or minimum-PRN rule changes.

Use only the 90 cleanStatic calibration bins in `[330,420)` to estimate the
normal median `m` and robust scale `s = 1.4826 median(|r-m|)`. For each causal
one-second residual, calculate

`z_t = clip((m-r_t)/s, -3, 3)`

and the one-sided Page statistic

`C_t = max(0, C_(t-1) + z_t - 0.5)`.

Alarm when `C_t >= 5`. Reset only at the beginning of a recording or across a
missing one-second bin. Do not reset at the known attack onset. The constants
`0.5`, `5`, and `3` are fixed before DS8 outcome access and will not be tuned
after release. This detector layer changes no physical score.

## Time and data roles

- cleanStatic `[330,420)`: normal baseline estimation and calibration trace.
- cleanDynamic: locked normal replay-IQ check, never used to estimate `m` or
  `s`.
- DS8: primary outcome-unseen attack. Monitoring begins at 30 s and continues
  through the whole recording. Official onset is 110 s. Bins available before
  110 s are pre-onset negatives; bins available at or after 110 s are post-onset.
- The interval from 120 s onward is additionally reported as stable-post for
  descriptive median and AUC calculations.

All score timestamps are one-second bin end times. A threshold crossing at time
`t` can be acted on at `t`, not at bin start.

## Frozen primary gates

The result is supported only if all gates pass:

1. At least 80 cleanStatic baseline bins and 25 cleanDynamic locked-normal bins
   are available.
2. There are no sequential alarms in either normal trace.
3. At least 70 monitored DS8 pre-onset bins and 300 DS8 post-onset bins are
   available.
4. There is no DS8 alarm before the official 110 s onset.
5. A DS8 alarm occurs at or after onset and no later than 60 s after onset.

The entire alarm trace, first crossing, pre/post counts, score medians, serial
bin AUC, input hashes, release commit, and output hashes will be reported even
if a gate fails. No gate, interval, score, threshold, or data role may be
changed after release.

## Claim boundary

This is a replay-IQ test of an outcome-unseen spoofing scenario. TEXBAT normal
recordings contain uncontrolled propagation effects but do not provide labeled
reflector/path ground truth. Passing does not establish a complete real-world
multipath benchmark or live-field false-alarm rate.
