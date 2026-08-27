# CGC RF geometry and aperture validation protocol v1

Status: frozen before any outcome from this campaign is generated.

## Question

Does the previously observed CGC state pattern survive broader satellite and receiver-motion geometry, and is the long-distance negative bias causally associated with finite correlator observation aperture?

## Frozen data boundary

The campaign reuses five previously generated, unused-for-CGC-state-boundary authentic/normal bundles: Denver static, Seoul static, Tokyo straight motion, London circular motion, and Sydney parallel sweep. They span five UTCs, five locations, four motion classes, and startup LOS support of 9--14 GPS satellites. Every source IQ, log, and manifest is pinned by SHA-256 in the config.

For each geometry, a new satellite-independent multipath control is generated with the existing fractional-delay simulator. Spoof pull-off is generated at 40, 60, 100, and 240 m, at final relative powers of -6 and +3 dB. This gives 40 receiver-RF conditions. The onset, direction, rate, frontend, noise realization law, receiver, estimator axes, binning, and comparison start are fixed in the config.

## Primary nine-tap geometry test

The 40 and 60 m outcomes estimate the geometry-dependent early detection boundary and are descriptive. They have no universal pass/fail gate because the previous two-geometry validation already showed that this boundary moves with LOS geometry.

For every one of the ten geometry-power groups, the nine-tap result must satisfy all of the following:

- At 100 m: AUC at least 0.8, median absolute direction cosine at least 0.85, and absolute relative displacement error at most 15%.
- At 240 m: AUC at least 0.8, median absolute direction cosine at least 0.85, template-edge fraction at least 0.1, and relative displacement error at most -5%.
- Every stream supplies at least eight comparison bins and every comparison bin uses at least eight PRNs.

The broader-geometry mechanism is supported only if all gates pass in all ten groups. Individual failures remain reported.

## Same-stream aperture intervention

GNSS-SDR runs once with the fixed nine-tap tracking receiver. Analysis then uses the central 3, 5, or all 9 complex taps from the exact same raw epochs. Thus RF waveform, noise, tracking loop, PRNs, epochs, estimator latent axes, geometry fit, and comparison bins are paired; only observed correlation-profile columns are removed.

At 100 and 240 m, values are aggregated by the median over all ten geometry-power groups. The finite-aperture mechanism is supported only if:

- 240 m absolute relative error is non-increasing from 3 to 5 to 9 taps;
- 240 m recovered displacement norm is non-decreasing from 3 to 5 to 9 taps;
- 240 m template-edge fraction is non-increasing from 3 to 5 to 9 taps; and
- 100 m nine-tap absolute error is no worse than three-tap error.

No tap subset may be chosen after seeing outcomes. Failure of an aperture gate does not erase the geometry result; the two claims are reported separately.

## Leakage, retention, and claim boundary

No threshold or estimator is fitted on these outcomes. No failed geometry, power, distance, PRN, bin, or aperture may be substituted or excluded. Intermediate 25 MHz component and composed IQ files are removed only after the corresponding receiver output and result provenance are retained. The five shared source bundles are never removed.

Claims are limited to simulated 25 MHz receiver RF with independent synthetic multipath. This is not field-multipath, real-spoof, universal-distance-threshold, TEXBAT-multipath, or operational validation.

Release requires the exact token `RELEASE-CGC-RF-GEOMETRY-APERTURE-VALIDATION-V1` and committed, clean config, protocol, and runner files.
