# Candidate 2: Causal NAV-Content Coherence Monitor

## Core hypothesis

An imperfect spoofer may reveal itself through causal parity, time-of-week,
issue-of-data, ephemeris, and cross-PRN frame-coherence violations even when
analog tracking observables appear plausible.

## Feature and statistic

For available PRN `i` and decoded word `w` received by time `t`, define

```text
q[i,w,t] = (
  1 - parity_ok[i,w,t],
  abs(delta_TOW[i,w,t] - delta_t_word[w]),
  IOD_mismatch[i,w,t],
  ephemeris_age[i,w,t],
  frame_phase_residual[i,w,t])
```

The epoch feature is an availability-masked robust median and upper tail of
each component plus pairwise TOW/IOD disagreement across PRNs. PRN identity,
scenario, filename, and official onset are excluded. A trailing 30 s causal
window uses a finite-state validity model learned from clean navigation
messages and a one-sided e-value/CUSUM threshold calibrated as the q99 of
source-distinct clean block maxima. Two successive decoded-word exceedances
provide persistence.

Receiver-specific parity conventions, normal bit errors, ephemeris uploads,
week rollover, time-scale conversion, data/pilot differences, and
reacquisition are nuisances. They require a versioned signal-specific
decoder, validity masks, and explicit state transitions rather than score
features based on C/N0 or lock.

## Falsification controls

Positive controls mutate parity, TOW increments, or issue-of-data in one and
four synthetic PRN streams without changing receiver code. Negative controls
include authentic ephemeris changes, week rollover, bit errors, channel
dropout, and reacquisition.

A coherent replay that preserves authentic message content is an essential
boundary control and is expected to pass as normal. That boundary directly
limits the hypothesis for the TEXBAT spoofing task.

## TEXBAT and Tuni conditions

A future TEXBAT success condition would require source-distinct clean FPR at
most 0.01, at least four available PRNs with decoded content, every synthetic
content-mutation positive detected, all authentic-update negative controls
within their clean bounds, at least 0.80 attack detection probability with a
recording-block-bootstrap 95% lower confidence limit above 0.50, median
first-alarm delay at most 30 s, and no dependence on message-availability
collapse. Tuni transfer would need separate GPS LNAV and Galileo I/NAV
canonicalizers whose hashes are frozen before SS-3/SS-5 access; the statistic
and gate would remain unchanged.

Expected failures are a content-faithful meaconer, a sophisticated simulator
with valid NAV data, sparse telemetry decode, normal navigation-data updates,
and receiver-specific decoding errors. The compute cost is linear in decoded
words and available PRNs.

The closest literature is broad telemetry/authentication and
receiver-consistency work rather than an exact repository duplicate.
Nevertheless, novelty is irrelevant if the core signal is absent under
coherent replay.

## Frozen audit score

Physical identifiability 2/5; independence 4/5; clean-only falsifiability
5/5; TEXBAT onset fit 2/5; Tuni transfer 3/5; novelty 2/5; feasibility 4/5.
Weighted total: **61/100**. It fails physical identifiability and the 75/100
total threshold and is not selected.
