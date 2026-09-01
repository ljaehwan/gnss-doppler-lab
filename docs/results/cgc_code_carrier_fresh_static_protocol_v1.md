# CGC Doppler-locked fresh-static final protocol v1

## Question

Does the frozen code-domain CGC alarm persist across five previously unused
static satellite geometries when the counterfeit code follows a 100-m carry-off
but its carrier Doppler is locked to the authentic static receiver trajectory?

## Release boundary

Tokyo, London, São Paulo, Sydney, and Nairobi were selected only by the
preregistered one-second startup LOS count. Their support counts are 11, 14, 14,
12, and 13. No 25-MHz source, multi-tap receiver output, delay estimate, or CGC
score from these IDs was available before this final configuration was frozen.

The five displacements are east, north, northwest, southeast, and an
east/upward direction; every vector has exactly 100-m norm. Pair substitution
or a new geometry after release is forbidden.

## Matched carrier contrast

Each geometry produces the same authentic component and false code trajectory
under two conditions:

- `carrier-coupled`: counterfeit code and carrier both follow the false path;
- `doppler-locked`: counterfeit code follows the false path while counterfeit
  carrier range/rate follows the authentic path.

Within a geometry, both conditions reuse the same authentic-only gain
calibration, frontend parameters, oscillator impairment, receiver seed, and
AWGN realization. Their IQ prefixes must be byte-identical through 5 s.

The simulator truth must show identical coupled/locked code range and rate,
locked/authentic carrier range and rate, nonzero code carry-off, and zero locked
carrier separation. Failure of an invariant stops scoring for that pair and is
reported.

## Frozen detector

Use 25-MHz signed IQ, the patched GNSS-SDR 9-tap output at 0.125-chip spacing,
the existing deterministic complex signed-delay template bank, startup LOS,
one-second medians, at least eight PRNs, and the frozen Partial-F threshold
`p_F <= 0.06028418845288192`. A persistent alarm requires at least three raw
alarms among the latest five bins. No threshold, template, tap subset, PRN
subset, hold interval, or persistence rule may be fitted after release.

The primary hold interval is `bin_start_s >= 12`. Pre-attack bins satisfy
`bin_start_s < 5`. Report every pair, including failed tracking or support.

## Frozen gates

The release is `SUPPORTED` only if all five truth invariants pass, all five
carrier-coupled controls produce a persistent alarm, at least four of five
Doppler-locked conditions produce a persistent alarm, the total pre-attack
persistent-alarm count is zero, median hold raw-alarm rate is at least 0.90 for
the coupled controls and 0.50 for Doppler lock, and median locked detection
latency is at most 10 s. Gates are descriptive engineering requirements, not
an estimate of field detection probability.

## Interpretation boundary

Success supports robustness of the code-domain geometry mechanism to removal
of a distinct counterfeit Doppler trajectory in fresh static simulated
receiver-RF data. It does not establish field performance, moving-receiver
performance, universal spoof detection, or failure of every Doppler-based or
code-carrier-consistency detector.
