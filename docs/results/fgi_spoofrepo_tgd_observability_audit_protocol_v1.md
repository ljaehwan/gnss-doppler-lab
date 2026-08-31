# FGI-SpoofRepo TGD observability-loss audit v1

This is a post-hoc exploratory root-cause audit.  The frozen TGD detector has
already returned `NOT_SUPPORTED`; this audit cannot revise that decision and
must not be cited as a new independent detection test.

## Question

Did the approximately 70 m spoof displacement remain observable in absolute
pseudorange while becoming unstable in prompt-referenced local nine-tap
correlator shape after receiver tracking recentered on the spoof signal?

## Fixed comparison

All streams are reduced to one-second bins and use the twelve healthy decoded
GPS PRNs.  The trusted receiver position is the checksum-valid NMEA GGA median
over the settled clean interval `[43,120)` s.

For pseudorange, each per-PRN bin median is reduced by the receive-time
broadcast-orbit range.  A per-PRN linear nuisance trend is fitted only over
`[43,120)` s and extrapolated without post-onset fitting.  The first three
analysis bins are excluded because GNSS-SDR changes its common pseudorange
clock offset inside bins 40--42; bin 43 is the first complete settled bin.
The remaining PRN residuals are fitted to

`delta rho_i = -u_i^T d + c`.

The frozen nine-tap signed-delay CSV is converted from chips to metres and fit
to the same LOS-plus-clock model.  Both fitted ECEF displacement series are
referenced to their componentwise settled-clean medians.  Checksum-valid NMEA
GGA positions are independently parsed from the file but are not independent
measurements: the PVT solution is produced from the receiver observables.

## Intervals and interpretation

- Settled clean baseline: `[43,120)` s.
- Receiver/attack transition excluded: `[120,160)` s.
- Stable post-onset comparison: `[160,230)` s.
- Baseline-start sensitivity: 43, 50, 60, 70, and 80 s, with a common 120 s
  endpoint.

The configured mechanistic signature requires adequate matched-bin support,
at least 50 m stable-post NMEA displacement, close pseudorange-to-NMEA vector
agreement, poor local-tap-to-NMEA direction agreement, an error ratio of at
least three, and zero persistent stable-post alarms in the already frozen
detector output.  Passing this signature supports only the tracking-recentering
observability diagnosis.  It neither validates a new spoof detector nor
establishes multipath-versus-spoof classification on FGI.

The orbit calculation is a receive-time approximation and omits transmit-time,
Sagnac, satellite-clock, ionospheric, and tropospheric corrections.  The
clean-only PRN trend removes their short-record nuisance evolution; therefore
the pseudorange result is a differential mechanism audit, not precise PVT.
