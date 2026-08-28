# CGC RF Tokyo 240 m replication v1

Date: 2026-08-28
Status: `SINGLE_REALIZATION_EXCEPTION` on the preregistered gate

## Question

The five-geometry RF campaign had one anomalous cell: Tokyo straight at 240 m
and +3 dB produced nine-tap AUC 0.417 even though physical direction cosine
remained 0.929.  This experiment tested whether that was a repeatable
high-power blind spot or one receiver-noise/multipath realization.

The original anomalous cell motivated the experiment but was excluded from
the five new primary replications.

## Frozen design

The protocol, configuration, runner, five new receiver-noise seeds, five new
PRN-specific multipath seeds, two powers, nine-tap estimator, support gates,
and decision rule were committed at `47edb4a` before any new outcome was read.
No seed, power, estimator, interval, or gate was substituted after release.

Each replication used the same Tokyo straight geometry, 240 m carry-off, -6 dB
and +3 dB final spoof advantage, a 25 MHz receiver-RF chain, and an independent
multipath negative realization.

## Primary result

| Replica | AUC at -6 dB | AUC at +3 dB | +3 dB direction cosine | Support |
|---|---:|---:|---:|---:|
| r01 | 1.000 | 1.000 | 0.9877 | pass |
| r02 | 1.000 | 1.000 | 0.9907 | pass |
| r03 | 1.000 | 1.000 | 0.9886 | pass |
| r04 | 1.000 | 1.000 | 0.9907 | pass |
| r05 | 1.000 | 1.000 | 0.9913 | pass |

All ten power-by-replica conditions were reported.  All five replications met
the support requirement; both power levels passed the frozen AUC threshold in
5/5 replications; and +3 dB direction passed in 5/5.  The median paired AUC
drop from -6 to +3 dB was 0.0.

The preregistered decision is therefore `SINGLE_REALIZATION_EXCEPTION`.  This
name means the earlier AUC 0.417 reversal did not reproduce in five independent
seeds.  It is a favorable robustness result, not an additional failure.

## Integrity and artifacts

The original Tokyo observation is not counted in the primary 5/5.  The release
state records no post-release seed, power, estimator, or gate substitution.

During concurrent Galileo receiver compatibility work, the shared GNSS-SDR
build path was briefly rebuilt while r03 source/RF composition was still in
progress.  A process and stage audit confirmed that no r03 receiver process
had begun, and the pinned executable was restored before receiver execution.
The final pinned binary SHA-256 is
`fc00d610fdb966358ac85486b616ff2411cb163ca8269dd0b435919c5e197e25`.

Primary hashes:

- `summary.json`:
  `2f7751280415675c72c33c344c8cc73663911653a01b5422670373dfc3fb0a5d`
- `condition_summary_9tap.csv`:
  `d34b638a37cee3d1d194c2b593cd59e82c6c186534c350d26d051b4307c4bda0`
- release-state JSON:
  `253684c8f4a220a943bcb869044b86f9ad4870c7b7ccc5479209d3a09989408d`

All campaign-created intermediate IQ was removed only after the relevant
receiver succeeded; receiver outputs were retained.

## Interpretation and claim boundary

The result rejects a repeatable high-power blind spot for this fixed synthetic
Tokyo geometry and RF chain.  Together with the original direction value, it
suggests that the isolated AUC reversal came from a particular RF/multipath
realization rather than loss of the correlator-geometry mechanism.

The experiment does not establish a universal high-power law, operational
detection probability, or field-multipath specificity.  It contains five
independent synthetic receiver/multipath realizations at one geometry and one
distance.  Real multipath controls remain a separate external-validity gate.
