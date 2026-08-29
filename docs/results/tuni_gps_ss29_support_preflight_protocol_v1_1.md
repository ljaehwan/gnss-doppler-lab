# TUNI GPS SS-29 score-free support preflight v1.1

## Why v1 is invalid

The released v1 runner downloaded exactly the permitted 2,000,000,000-byte IQ
prefix and produced 31 receiver MAT files without loading a detector or delay
template. GNSS-SDR reached physical EOF at
`2,000,000,000 / 2,000,000,000` bytes but repeated the already documented TUNI
source-stop defect; a graceful SIGINT after `/proc` EOF verification flushed the
MAT files and returned code zero.

The v1 postprocessor then wrote `source.sample_rate_hz=50,000,000` in its
receiver manifest. That is the RF input rate. `PRN_start_sample_count` belongs
to the direct-resampler output and must be divided by the receiver's frozen
internal rate, 5,000,000 Hz. Consequently v1 compressed all tracking times by a
factor of ten, found no data in the frozen `[5,10)` second interval, and emitted
an invalid `INSUFFICIENT_SUPPORT` result. That output is retained and
hash-pinned; it is not a scientific terminal decision.

## Permitted correction

Version 1.1 does not download or reprocess raw IQ. It hash-verifies the v1
prefix, invalid summary, invalid receiver manifest, and canonical 31-file MAT
tree. It then links the immutable MAT directory into a new output and changes
only the timebase metadata from 50 MHz to 5 MHz before rerunning the released
score-free support counter.

No PRN, epoch count, interval, bin, support threshold, target coverage rule, or
complex-tap requirement changes. No correlator delay, learned template,
geometry residual, partial-F statistic, alarm, or detector threshold may be
loaded or computed.

## Frozen gate and terminal decision

The interval remains `[5,10)` seconds. Each PRN/bin needs at least 200 epochs.
At least eight eligible PRNs must occur in at least three bins, and every
documented spoofed PRN (1, 2, 21, 32) must be eligible in at least three bins.
Only the corrected v1.1 result can emit `SUPPORT_ELIGIBLE` or
`INSUFFICIENT_SUPPORT` for SS-29.
