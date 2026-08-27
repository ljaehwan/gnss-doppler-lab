# CGC observability boundary v1 result

## Main result

The exploratory profile-level screen produced a clean and monotone physical
observability boundary. Twenty-two of 64 separation-power cells passed every
frozen cell rule.

For moderate counterfeit power differences (`-9` through `+9 dB`), the first
observable grid point was `60 m`, or `0.2047 chip`. When one component was much
weaker (`-15` or `+15 dB`), the first observable point moved to `80 m`, or
`0.2730 chip`. Every larger tested separation remained observable after the
first crossing for all eight power levels.

| Counterfeit advantage | Secondary/primary amplitude | Prompt owner | Minimum observable separation |
|---:|---:|---|---:|
| -15 dB | 0.178 | authentic | 80 m (0.273 chip) |
| -9 dB | 0.355 | authentic | 60 m (0.205 chip) |
| -6 dB | 0.501 | authentic | 60 m (0.205 chip) |
| -3 dB | 0.708 | authentic | 60 m (0.205 chip) |
| +3 dB | 0.708 | counterfeit | 60 m (0.205 chip) |
| +6 dB | 0.501 | counterfeit | 60 m (0.205 chip) |
| +9 dB | 0.355 | counterfeit | 60 m (0.205 chip) |
| +15 dB | 0.178 | counterfeit | 80 m (0.273 chip) |

## Boundary contrast

The transition was not a marginal threshold artifact. At the two RF-anchor
power regimes:

| Power | Separation | Pooled AUC (95% paired bootstrap) | Minimum geometry AUC | Observable |
|---:|---:|---:|---:|---|
| -6 dB | 40 m | 0.7027 [0.6769, 0.7264] | 0.6375 | no |
| -6 dB | 60 m | 0.8741 [0.8550, 0.8938] | 0.7782 | yes |
| -6 dB | 80 m | 0.9569 [0.9461, 0.9668] | 0.8931 | yes |
| +3 dB | 40 m | 0.7210 [0.6912, 0.7474] | 0.6598 | no |
| +3 dB | 60 m | 0.8874 [0.8684, 0.9063] | 0.7939 | yes |
| +3 dB | 80 m | 0.9696 [0.9617, 0.9773] | 0.9248 | yes |

At `-15 dB`, 60 m remained below the pooled AUC rule at `0.6898`, while 80 m
reached `0.8270`. At `+15 dB`, the corresponding values were `0.6964` and
`0.8140`.

## Physical interpretation

The result supports a narrower mechanism claim: CGC observability depends on
both spatial code separation and visibility of the weaker correlation
component. Moderate power balance exposes the signed secondary-delay pattern
at roughly `0.2 chip`; when the weaker component is only about 0.18 of the
stronger component, roughly `0.27 chip` is required.

The near symmetry for positive and negative power advantage is physically
consistent with prompt takeover. Before takeover the authentic component owns
the prompt and the counterfeit is the secondary path. After takeover the
counterfeit owns the prompt, the authentic signal becomes the secondary path,
and the fitted delay sign reverses. Clock-centered geometry is insensitive to
that global sign reversal but remains sensitive to whether the signed delays
follow satellite LOS.

This provides a concrete explanation for delayed TEXBAT alarms: attack
transmission may begin before the two correlation components have enough
separation and relative visibility for the secondary-delay geometry to be
identifiable. The current screen does not estimate the TEXBAT displacement
trajectory, so that connection remains a hypothesis rather than a measured
mapping.

## Selected receiver-RF anchors

Six 25 MHz paired-RF conditions were selected automatically:

- `-6 dB`: 40 m below boundary, 60 m at boundary, 80 m above boundary.
- `+3 dB`: 40 m below boundary, 60 m at boundary, 80 m above boundary.

Each proposed offset uses ENU direction `[0.8, 0.6, 0]`, a normal acquisition
prefix, a five-second spatial transition, and a five-second power ramp. The
next experiment must run these conditions through the existing paired RF
generator, PRN-specific RF multipath control, and frozen complex-nine-tap
GNSS-SDR receiver without changing the selected cells.

## Claim boundary

This is an ideal triangular-autocorrelation, profile-level mechanism screen
using six train LOS geometries and a marginally exact matched multipath control.
It is not yet a receiver-RF, field, TEXBAT-multipath, false-alarm, or operational
boundary. The 60/80 m values become publishable physical evidence only if the
selected RF anchors reproduce the predicted ordering.

The raw summary is `artifacts/cgc_observability_boundary_v1/summary.json`,
SHA-256 `6c6579cbfe1c5326cedfce6ca375c9290fc2f4eb9241b349d04cfb14a99a19df`.
The generated heatmap is
`artifacts/cgc_observability_boundary_v1/observability_boundary.png`, SHA-256
`5fd9830498197fba61c1c78a6beab3354d10f85b001af6192da2edf9dc440057`.
