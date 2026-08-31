# Frozen FGI-SpoofRepo targeted-DFMC support preflight v1

## Objective

Determine whether the unopened FGI-SpoofRepo `TG_DFMC/TGD_L1_E1.dat` raw-IF
recording can supply the complex nine-tap, multi-PRN support required by the
already frozen clock-centered geometry detector.  This preflight is not a
detection experiment.

## Frozen signal adapter

- Input: signed 8-bit real IF, 26 MHz sampling rate.
- GPS L1 offset: +6.39 MHz relative to the recorded center frequency.  Because
  the input is real, the spectrum is conjugate symmetric; the positive image is
  selected.
- Conditioner: `Byte_To_Short`, followed by a frequency-translating low-pass
  FIR with 2.1 MHz passband, 0.5 MHz transition width, and 4:1 decimation.
- Internal rate: 6.5 MHz.
- Receiver: pinned patched GNSS-SDR binary with 31 GPS L1 C/A channels.
- Correlator: complex nine taps at 0.125-chip spacing from -0.5 to +0.5 chip.
- Receiver prefix: the first 240 seconds only.

## Score-free support rule

The normal interval is 40--120 s.  The stable post-onset support interval is
160--230 s; it starts well after the documented approximately 138 s attack
onset.  For each interval independently:

1. form one-second bins;
2. retain a PRN in a bin only when at least 200 complex-nine-tap tracking epochs
   are present;
3. require at least eight distinct GPS PRNs in at least 60 bins; and
4. count duplicate channels for the same PRN by their maximum epoch count, not
   their sum.

The terminal outcomes are `SUPPORT_ELIGIBLE` and `INSUFFICIENT_SUPPORT`.

## Prohibited access

This stage may read receiver acquisition/tracking outputs, PRN identities,
epoch times, and complex-tap dataset names.  It must not import or compute a
signed-delay template, geometry residual, CGC score, detector threshold,
persistence alarm, detection probability, false-alarm rate, or latency.  The
support gate may not be changed after receiver output is inspected.

## Next gate

Only `SUPPORT_ELIGIBLE` permits a separately committed one-shot application of
the unchanged detector.  A support failure is reported as such and must not be
described as a missed spoofing detection.

