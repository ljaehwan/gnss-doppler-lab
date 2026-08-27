# CGC observability boundary v1 protocol

## Purpose

This exploratory screen replaces scenario-start time with two physical axes:
spatial counterfeit displacement and counterfeit-to-authentic power advantage.
It asks when the frozen complex-nine-tap clock-centered geometry score becomes
identifiable relative to an independent multipath control.

The screen is not a new held-out validation. It uses the six train LOS
geometries already pinned by the original correlator-geometry mechanism audit.
Its purpose is to choose a small number of below-, near-, and above-boundary
conditions for a later receiver-RF experiment.

## Physical grid

- Spatial displacement: `5, 10, 20, 30, 40, 60, 80, 120 m`.
- Counterfeit power advantage: `-15, -9, -6, -3, +3, +6, +9, +15 dB`.
- GPS L1 C/A chip length: `293.0522561094819 m`.
- 100 paired events per geometry and cell, six geometries, for 600 paired
  spoof/multipath comparisons per cell.

Zero dB is omitted because the prompt owner is ambiguous at exactly equal
power. Below zero dB, the authentic component owns the prompt and the
counterfeit is the secondary path. Above zero dB, the counterfeit owns the
prompt and the authentic component becomes the secondary path with the delay
sign reversed. In both regimes the secondary-to-primary amplitude ratio is at
most one and remains inside the frozen template support.

For a displacement vector `Delta r`, the satellite-specific secondary delay is

`d_i = -u_i^T Delta r / L_chip`.

The frozen template estimator recovers a signed secondary delay from each
complex nine-tap profile. The final score remains the clock-centered residual
of the LOS-plus-clock fit.

## Matched multipath control

Within each event, the multipath control is an exact derangement of the complete
complex profiles and estimated delays across PRN identities. Thus spoof and
multipath have the same single-PRN profile multiset, power, separation, phase,
noise, and profile-quality distribution. Only the association between a
profile and satellite LOS differs. This is deliberately a geometry-only stress
test, not a ray-traced multipath environment.

Direction, authentic-center error, relative phase, complex noise, and the PRN
derangement are reused for the same geometry/event index in every grid cell.
This paired-common-random-number design makes changes across separation and
power less sensitive to Monte Carlo noise.

## Cell metrics and exploratory boundary

Report pooled complex-score AUC, its paired bootstrap 95% interval, the minimum
per-geometry AUC, positive median-residual-separation geometry count, signed
delay accuracy, oracle AUC, and realized satellite delay spread.

A cell is called observable only if all exploratory conditions hold:

1. pooled complex AUC is at least `0.80`;
2. every geometry AUC is at least `0.70`;
3. at least five of six geometries have positive multipath-minus-spoof median
   residual separation; and
4. complex-delay sign accuracy is at least `0.75` for truth magnitude of at
   least `0.05 chip`.

For each power advantage, the boundary is the smallest tested spatial
separation satisfying all four conditions. Nonmonotonic cells remain visible
and are never repaired or relabeled.

## RF anchor plan

For `-6 dB` and `+3 dB`, select up to three RF conditions: the largest tested
nonobservable separation below the boundary, the boundary cell, and the next
larger observable cell. The later RF campaign must regenerate authentic and
counterfeit IQ through the existing 25 MHz paired simulator, add the existing
PRN-specific RF multipath control, and process every condition with the frozen
complex-nine-tap GNSS-SDR receiver.

## Claim boundary

This ideal triangular-autocorrelation screen can locate a mechanism-level
identifiability region and plan expensive RF runs. It cannot establish an RF,
field, TEXBAT-multipath, false-alarm, or operational detection boundary.
