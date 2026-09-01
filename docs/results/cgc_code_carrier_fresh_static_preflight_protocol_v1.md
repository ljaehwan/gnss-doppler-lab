# CGC Doppler-locked fresh-static support preflight v1

## Purpose

Select five previously unused static receiver geometries without observing a
CGC score, correlation-delay estimate, 25-MHz RF stream, or receiver outcome.
This preflight is selection by satellite support only.

## Frozen candidate pool

The ten candidates are divided into five ordered slots. Each slot fixes one
100-m displacement direction, while its two candidates provide a geographic
fallback. Exact position, UTC, receiver seed, order, carry-off envelope, alarm
threshold, and persistence rule are fixed in
`configs/experiments/cgc_code_carrier_fresh_static_pool_v1.json`.

None of the ten candidate IDs has appeared in a previous full-RF experiment.
The combination of candidate ID, position, and UTC is treated as the geometry
identity; no earlier score is imported into this campaign.

## Allowed preflight observation

For each candidate, generate only one second of clean 1-MHz simulator output.
Read the startup azimuth/elevation table and count GPS L1 C/A PRNs. Preserve the
text log and delete the IQ immediately. Do not generate false trajectories,
counterfeit signals, multi-tap receiver outputs, or CGC scores.

For slots S1 through S5, select the first candidate in file order having at
least ten startup LOS PRNs. If either candidate in a slot fails, the declared
fallback may be tried. If both fail, stop the campaign; do not invent a new
candidate after seeing support.

## Boundary before the main release

The selected five complete rows and observed support counts must be copied to a
final-test configuration. That configuration, its runner, and its evaluation
protocol must be committed before generating any selected 25-MHz source or
opening any CGC outcome. Main-test failure is retained and reported; geometry,
threshold, tap count, interval, or persistence cannot be retuned.
