# TUNI Galileo exhaustive PRN-support audit v1.1

## Technical amendment

The preregistered detector result remains `INSUFFICIENT_SUPPORT` and is not
recomputed here.  A first diagnostic invocation attempted 36 fixed Galileo E1
channels, but GNSS-SDR v0.0.19 rejected that flowgraph before processing any
sample because its channel-count guard permits at most 35.  Its MAT files were
empty `[1, 0]` sentinels; they contain no scientific observation.

Before any fixed-PRN acquisition succeeded, this v1.1 technical correction was
frozen.  It covers the same complete PRN roster with two sequential,
non-overlapping banks: PRNs 1--18 and PRNs 19--36.  No acquisition threshold,
input interval, tracking parameter, label, or support rule changes.

## Frozen diagnostic

- Scenarios: SS-11, SS-12, and SS-13.
- Input interval: the first 10.0 complex-sample seconds used by the released
  evaluation.
- Receiver executable, acquisition, resampling, tracking, and nine-tap
  parameters: identical to the released evaluation.
- Channel assignment: one fixed channel per PRN, in two 18-channel banks.
- Sustained support: at least 1,000 valid tracking epochs for a PRN.
- Same-stream control available: at least two sustained PRNs absent from the
  README spoof list in each scenario.

This is a receiver-input availability audit only.  It cannot be reported as a
second detector evaluation or used to replace the released primary outcome.
