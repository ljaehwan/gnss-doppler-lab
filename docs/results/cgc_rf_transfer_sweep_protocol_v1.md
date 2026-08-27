# CGC RF transfer sweep v1 protocol

## Motivation and question

The preregistered receiver-RF anchors did not reproduce the ideal profile-level
60 m boundary. At -6 dB the AUC still increased over 40/60/80 m, while at +3
dB it became nonmonotone. This experiment asks a narrower physical question:
with the receiver completely fixed, what transfer curve maps final carry-off
distance to recoverable CGC geometry?

The experiment is an exploratory mechanism map. It must report the entire grid
and will be used to formulate a later held-out multi-geometry validation. It is
not itself confirmatory evidence for a newly selected boundary.

## Distance, speed, and power control

Use the same static 30-second authentic signal, receiver seed, front-end
reference, independent-multipath control, and patched complex-nine-tap
GNSS-SDR receiver as the failed anchor experiment.

Run every combination of:

- distance: 20, 40, 60, 80, 100, 120, 160, 200, and 240 m;
- final counterfeit advantage: -6 and +3 dB.

The ENU direction remains `[0.8, 0.6, 0]`. Carry-off begins at 5 s and uses a
constant 20 m/s rate, so transition duration is exactly `distance / 20` s.
The largest distance settles at 17 s. Every condition is compared over the
same causal interval beginning at 18 s, after both the spatial transition and
five-second power ramp have completed. This removes the earlier confounding in
which larger target distances implied higher pull-off speed.

The tested range is 0.0682 to 0.8189 C/A chip. The normal prefix before 5 s
must be byte-identical for every composed RF signal.

## Fixed receiver aperture

The receiver remains nine taps at 0.125-chip spacing, spanning -0.5 to +0.5
chip. It is not widened in this experiment. Code audit showed that the current
GNSS-SDR patch couples tap spacing to the DLL discriminator spacing; changing
it would alter both observation aperture and tracking dynamics and would not be
a clean aperture-only intervention.

For every condition report serial-bin AUC against the same independent
multipath RF, residual medians, recovered displacement norm, absolute direction
cosine, template fit distance, and the fraction of estimated per-PRN delays at
the template-bank edge. Also report the peak AUC/distance, first tested 0.8
crossing, all contiguous above-0.8 grid intervals, strict monotonicity, and
whether performance falls after its peak.

No distance, power, interval, receiver configuration, or estimator axis may be
changed or omitted after release.

## Storage and recovery

Generate one unscaled counterfeit component per distance and reuse it for both
power conditions. Delete each composed spoof IQ only after the receiver output,
score tables, hashes, and condition result are durable. Delete the shared
distance component only after both powers complete. The existing authentic,
normal, and multipath inputs are never deleted. All campaign-created IQ files
remain deterministically regenerable from retained trajectories, manifests,
tools, seeds, and hashes.

## Claim boundary and next stage

This is one previously used static satellite geometry. A physically useful
shape discovered here must be frozen and tested on new LOS geometries before it
can support a WCL claim. If a finite observability window or edge saturation is
found, the subsequent algorithm experiment will hold the DLL discriminator
fixed while adding nonuniform multi-scale auxiliary taps, then test recovery on
held-out geometries.
