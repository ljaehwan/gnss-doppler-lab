# TUNI Galileo authentic-window support scan v1

## Question

The preregistered first-10-second evaluation and exhaustive fixed-PRN audit
produced no trackable authentic PRN in SS-11, SS-12, or SS-13, although each
README says authentic Galileo E1 signals are included.  This input-only audit
asks whether later parts of the recordings expose a usable same-stream
multipath control pair.

It does not compute detector features, scores, thresholds, or performance, and
cannot change the released `INSUFFICIENT_SUPPORT` outcome.

## Frozen scan

- Scenarios: SS-11, SS-12, and SS-13.
- Windows: 10.0 seconds beginning at 30, 90, 150, 210, and 270 seconds.
- Receiver executable and all acquisition, resampling, tracking, and nine-tap
  settings: identical to the released detector evaluation.
- Channels: one fixed channel for every PRN not named as spoofed in that
  scenario's README.  This gives 35, 34, and 32 channels respectively and stays
  within the GNSS-SDR v0.0.19 35-channel guard.
- Sustained PRN: at least 1,000 valid epochs in a window.
- Usable control window: at least two sustained authentic PRNs in the same
  scenario and window.
- Overall support: every multipath scenario has at least one usable control
  window.

The PRN roster, five offsets, duration, support threshold, executable hash, and
receiver settings must be committed before execution.  No acquisition
threshold may be changed after observing this scan.
