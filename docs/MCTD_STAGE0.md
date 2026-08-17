# MCTD Stage-0 static experiment

MCTD compares two deterministic GPS L1 C/A tracking replays starting from the
same authenticated scenario-specific state and reading the same raw-IQ range.
Only DLL/PLL bandwidth differs: slow is 0.5/10 Hz and fast is 2/25 Hz.  The
identical-loop negative control uses 0.5/10 Hz on both sides.

The detector is non-neural.  Per PRN and nominal raw-IQ millisecond it builds
signed differences in code phase, unwrapped carrier phase, Doppler, code NCO,
discriminators, filter outputs, and Prompt-normalized complex nine taps.  A
clean-only robust center and Ledoit-Wolf covariance define Mahalanobis scores.
PRN scores are median pooled, then median pooled into non-overlapping 100 ms
blocks.  The primary threshold is clean calibration q99; an alarm requires
three truly consecutive blocks and resets across gaps.

No attack replay or scoring is authorized until source equality,
configuration-local deterministic replay, stable common support, and the
identical-loop collapse gate all pass.  Exact Paper-B0 is reported only when it
can be rerun on identical support; historical metrics are prior evidence only.

