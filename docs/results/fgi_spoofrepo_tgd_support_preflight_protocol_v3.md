# FGI-SpoofRepo TGD identity-aware support preflight v3

Date frozen: 2026-08-31

## Purpose

V3 is the terminal score-free input gate before a one-shot FGI detector run.
It adds satellite-identity validity to the cadence-corrected v2 rule.  It does
not inspect complex tap values, estimate delay, load the delay template,
compute CGC or partial-F scores, apply a threshold, or issue an alarm.

V2 correctly accounts for GNSS-SDR's 50 Hz post-telemetry tracking-dump
cadence, but its support count includes short acquisition locks for PRNs that
never decoded a usable navigation message.  Those PRNs cannot supply the
broadcast orbit required by the LOS geometry model and must not count toward
the eight-satellite gate.

## Pinned inputs and identity rule

V3 reads the exact v2 summary and the exact end-of-run GPS ephemeris XML pinned
by SHA-256 in the config.  A PRN is identity-valid only when the XML contains
its ephemeris and the broadcast `SV_health` field equals zero.  The frozen
identity-valid roster is G05, G07, G08, G09, G13, G14, G15, G18, G20, G22,
G27, and G30.  Acquisition-only PRNs without an ephemeris are excluded.

For each v2 one-second bin, v3 intersects the cadence-eligible PRN set with
that healthy roster.  Both unchanged intervals must contain at least eight
identity-valid PRNs in at least 60 bins:

- clean: 40--120 s;
- post-onset support: 160--230 s.

## Terminal outcomes

- `SUPPORT_ELIGIBLE`: both intervals pass the identity-aware eight-PRN rule.
- `INSUFFICIENT_SUPPORT`: either interval fails.

Only `SUPPORT_ELIGIBLE` permits a separately committed one-shot detector
protocol.  This support audit makes no spoof-detection claim.

