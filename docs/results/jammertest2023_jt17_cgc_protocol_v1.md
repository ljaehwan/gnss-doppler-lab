# Frozen JammerTest 2023 JT23-17.1.6 CGC protocol v2

Date frozen: 2026-08-31, before any tap value, signed-delay estimate, or CGC
score from this recording was accessed.

## Score-free amendment from v1

The v1 support program stopped before score access. Two receiver-interface
assumptions were wrong: the tracking dump is exactly 50 Hz rather than the
assumed 1 kHz, and the receiver makes one 80 ms time correction during initial
clock settling. The raw tracking log independently confirms eight authentic
GPS tracks from startup. Protocol v2 therefore requires 40 of 50 epochs per
PRN/bin (80% coverage), fits GPS time on the longest cadence-consistent
observables segment, and excludes any one-second bin not fully contained in
that stable segment. The nine taps, delay template, PRN count, evaluation
regions, partial-F threshold, persistence rule, and terminal gates are
unchanged. This is an input-adapter amendment, not detector tuning.

## Question

Can the already frozen complex-nine-tap, support-normalized CGC rule transfer
without recalibration to an independent outdoor coherent-spoof recording and
raise a causal alarm when the planned false route begins to carry the receiver
solution away?

This is a single external field recording. A positive result supplies missing
real-spoof sensitivity evidence, but it is not a universal field-accuracy or
real multipath-versus-spoof confusion-matrix claim.

## Input and labels fixed before signal access

The only primary input is the open CC BY 4.0 FGI Jammer Test 2023 recording
`JT23_17.1.6_L1_E1.iq`. Fairdata publishes an exact size of 72,110,000,000
bytes and SHA-256
`4bd6e3963f0b3d6806670db5d1a653de05fd9fe602ec42b005eb6cc4d45931e3`.
The LabSat recording contains signed interleaved complex samples at 30.69
Msample/s, zero IF, centered at 1575.42 MHz. The source file is stored on the
HDD, never committed to Git.

Independent Table 3 of Bhuiyan et al., *GPS Solutions*, 2026 labels the
coherent high-power spoof signal onset at receiver second 226 and the
transmit power at +25 dBm. The official A7 test plan says that every dynamic
test in group 17 first holds the spoofed solution at the true start position
for five minutes before route motion. Therefore the motion-onset label is
fixed at `226 + 300 = 526 s`. It is not inferred from CGC output.

Only receiver seconds `[0,560)` are processed. The primary regions are:

- authentic clean: `[40,200)` s;
- aligned coherent spoof, before route motion: `[246,500)` s, descriptive;
- first prompt-local carry-off interval: `[526,556)` s.

The 26 s guard after RF onset excludes the power transition. The last region
is deliberately short: a nine-tap aperture of plus or minus 0.5 chip observes
only the early overlap before a moving secondary peak can leave the aperture.
Later disappearance is not counted as a miss.

## Score-free support gate

The frozen receiver decimates 30.69 MHz complex I/Q by five to 6.138 MHz and
dumps nine complex taps at 0.125-chip spacing for 31 GPS L1 C/A channels. The
support preflight may read only PRN identities, sample counters, schema names,
observables timing, and broadcast ephemerides. It must not read tap values,
load the signed-delay template, or compute a CGC score.

A PRN/bin requires at least 40 of the 50 tracking epochs in a one-second bin.
Duplicate
channels for the same PRN/bin are consolidated and never counted as extra
satellites. At least eight healthy GPS PRNs are required in at least 60 clean
bins, 60 aligned-spoof bins, and 20 of the 30 carry-off-onset bins. Failure is
reported as `INSUFFICIENT_SUPPORT`; detector scoring is forbidden.

GPS time zero is fitted without detector access from the 50 Hz observables by
`RX_time = TOW_0 + 0.02 * row_index`, after week-unwrapping. Cadence-consistent
segments are split at clock steps, and the longest segment with at least
10,000 rows is selected. Its maximum absolute residual is 0.021 s; only full
one-second bins inside that segment are eligible. A checksum-valid NMEA
position from the clean
interval is held fixed for every LOS calculation; attacked post-onset PVT is
not used for detector geometry.

## Frozen detector

For each supported healthy PRN, the unchanged matched-template estimator maps
the complex nine-tap profile to a signed secondary-code delay. Per-epoch delays
are median aggregated within each one-second PRN bin. For every bin with at
least eight PRNs, fit the nested models

`H0: y_i = c + e_i`

and

`H1: y_i = -u_i^T d + c + e_i`.

The support-normalized statistic is

`F = ((SSE0-SSE1)/3) / (SSE1/(N-4))`,

with upper-tail probability `p_F`. A raw alarm is fixed at
`p_F <= 0.06028418845288192`. A causal persistent alarm requires three raw
alarms in the latest five consecutive one-second bins. No JammerTest threshold,
template, PRN, interval, or persistence fitting is allowed.

## Terminal decision

`REAL_CARRYOFF_TRANSFER_SUPPORTED` requires all of the following:

1. all score-free support gates pass;
2. at least 60 clean, 60 aligned-spoof, and 20 carry-off geometry bins exist;
3. clean persistent-alarm rate is at most 5%;
4. after resetting the causal 3-of-5 history at motion onset, a persistent
   alarm occurs no later than 30 s after 526 s; and
5. median `p_F` in `[526,556)` is below the clean median `p_F`.

The aligned-spoof alarm rate, serial-bin AUC, fitted displacement, C/N0, and
later aperture loss are descriptive. If input support passes but a terminal
gate fails, the result is `REAL_CARRYOFF_TRANSFER_NOT_SUPPORTED`; no tuning or
second primary release is permitted.
