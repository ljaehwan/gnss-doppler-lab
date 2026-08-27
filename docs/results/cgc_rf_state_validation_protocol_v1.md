# CGC RF state validation v1 protocol

## Confirmatory question

Does the exploratory receiver state sequence reproduce on two RF geometries
that were not used to construct the 20--240 m transfer map?

The held-out geometries are `fv2-straight-01` and `fv2-sweep-01`. They have
different UTC satellite layouts and receiver motion. They were used in an
earlier broad detector test but not in selection or interpretation of the
distance-state map being tested here.

## Frozen 24-cell design

For each geometry run both -6 and +3 dB at 40, 60, 80, 100, 160, and 240 m.
The 80 m cell is a fully reported transition diagnostic but has no pass gate.
All other controls match the discovery sweep: ENU direction `[0.8, 0.6, 0]`,
20 m/s carry-off beginning at 5 s, five-second power ramp, fixed 0.125-chip
nine-tap receiver, and a common comparison interval starting at 18 s.

Receiver-internal run IDs are compact identifiers so that every generated
`Tracking_1C.dump_filename` line stays below GNSS-SDR's fixed 200-character
INI parser limit; condition IDs and all physical parameters remain unchanged.
Each geometry uses its own byte-pinned authentic component, normal RF prefix,
LOS log, and independent-multipath receiver recording. Every composed signal
must have a byte-identical normal prefix.

## Preregistered state gates

Evaluate each of the four geometry-power groups independently. A group passes
only if all of the following hold:

1. 40 m is unresolved: AUC is below 0.8.
2. 60 m is onset: AUC is below 0.8, median absolute direction cosine is at
   least 0.7, and direction cosine improves over 40 m.
3. 100 and 160 m are metrically discriminable: each has AUC at least 0.8,
   direction cosine at least 0.85, and absolute displacement-norm relative
   error no greater than 15%.
4. 240 m is discriminable but saturated: AUC is at least 0.8, at least 10% of
   per-PRN delay estimates occupy the template edge, and displacement relative
   error is at most -5%.
5. Every evaluated stream supplies at least eight eligible bins and eight
   satellites per bin.

The overall state map is reproduced only if every gate passes in all four
groups. No majority rule, threshold fitting, geometry replacement, or cell
omission is allowed after release.

## Interpretation boundary

Passing would establish simulated receiver-RF replication across three total
motion/LOS geometries when the exploratory static map is included. It would
support the distinction between detection observability and metric
observability, but it would not establish real-multipath accuracy or field
performance.

Failure is also informative: every failed state and metric will be retained,
and the claim must be narrowed to the states that replicate.

## Storage

One unscaled counterfeit component is shared by the two power conditions at
each geometry-distance cell. A composed IQ is deleted only after receiver and
score artifacts are durable. The shared component is deleted only after both
powers finish. All 36 campaign IQ files remain deterministically regenerable;
no existing authentic, normal, or multipath input is deleted.
