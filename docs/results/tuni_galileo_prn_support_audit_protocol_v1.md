# TUNI Galileo exhaustive PRN-support audit v1

## Purpose

The preregistered same-stream multipath-control evaluation returned
`INSUFFICIENT_SUPPORT`: its automatic 12-channel receiver tracked only PRN 31
in SS-11, SS-12, and SS-13.  This audit asks only whether authentic PRNs can be
sustained by the receiver when every Galileo PRN is explicitly assigned a
channel.  It does not recompute detector scores and cannot replace or alter the
released result.

The dataset README states that each recording contains authentic Galileo E1
signals plus spoofing on the listed PRNs.  Thus the diagnostic separates an
automatic-channel-scheduling limitation from absence of same-stream controls.

## Frozen diagnostic

- Scenarios: SS-11, SS-12, and SS-13.
- Input interval: the same first 10.0 complex-sample seconds as the released
  evaluation.
- Receiver binary and all acquisition, resampling, tracking, and nine-tap
  parameters: identical to the released evaluation.
- Channel bank: 36 channels, fixed one-to-one to Galileo PRNs 1 through 36.
- A PRN has sustained support when its tracking dump contains at least 1,000
  valid epochs.
- A scenario has usable same-stream control support when at least two sustained
  PRNs are absent from its README spoof list.

No acquisition threshold may be changed in v1.  A later receiver-sensitivity
study, if needed, must be labelled separately and must not be presented as the
preregistered detector result.
