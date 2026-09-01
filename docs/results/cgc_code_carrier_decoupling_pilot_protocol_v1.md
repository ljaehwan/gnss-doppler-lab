# CGC code/carrier decoupling development pilot v1

## Purpose

This pilot asks whether the paper's code-domain cross-satellite geometry test remains usable when a carry-off replica does not expose a distinct carrier-Doppler trajectory. It is a development experiment, not fresh confirmatory evidence: the Seoul geometry and conventional carry-off behavior have already been inspected in earlier campaigns.

## Fixed physical contrast

Both spoof conditions use the same 100 m false code trajectory, power envelope, authentic component, frontend realization, nine complex correlator taps, template bank, LOS table, and CGC rule. Only the counterfeit carrier reference changes.

- `carrier-coupled`: code phase and carrier Doppler both follow the false trajectory.
- `doppler-locked`: code phase follows the false trajectory while carrier Doppler follows the authentic static receiver trajectory.

The simulator must first pass exact truth invariants: locked and coupled code range/rate must agree, and locked and authentic carrier range/rate must agree. The unmodified CLI path must remain byte-identical to upstream when `-X` is omitted.

The carrier lock is also audited directly in the generated RF. At 8.3--8.7 s,
the three PRNs with the largest truth-predicted coupled carrier separation
(G12/G24/G26) are inspected with 20-ms PRN correlation profiles on a 10-Hz
Doppler grid. A dominant peak is defined before reading the profile as having
normalized height at least 0.5, prominence at least 0.1, and at least 60-Hz
separation from another reported peak. The positive control should contain two
dominant carrier peaks, whereas the Doppler-locked condition should contain
one. The primary mechanism check probes the truth-predicted authentic and
coupled-spoof frequencies; blind peak count is retained as an ancillary check
because one broadened main lobe can split into nearby local maxima. This is a
mechanism diagnostic, not an independent generalization test.

## Receiver-RF evaluation

Each counterfeit component is superposed with the same pinned authentic component. The existing frozen Seoul frontend reference, seeded noise, 25 MHz s8 IQ format, patched GNSS-SDR receiver, nine taps at 0.125 chip, deterministic signed-delay template bank, minimum eight PRNs, and one-second bins are reused without retuning.

The primary development interval starts at 12 s, after the 5--10 s pull-off. Each bin's clock-centered residual is converted to the three-parameter nested-model Partial-F tail probability. The previously frozen alarm threshold is `p_F <= 0.06028418845288192`; persistence is three alarms among the latest five available one-second bins.

## Interpretation

A successful pilot means that code carry-off is present and geometrically classifiable even when the counterfeit carrier is locked to the authentic trajectory. It does not mean that every Doppler-based detector fails, that code-carrier consistency is unavailable, or that the one known Seoul geometry proves generalization. Those claims require a separate support-only-selected, fresh, static multi-geometry release.
