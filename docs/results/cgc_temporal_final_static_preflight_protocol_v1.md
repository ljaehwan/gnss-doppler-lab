# CGC temporal final-static support preflight v1

## Purpose

Select five untouched static receiver geometries before any 25 MHz receiver-RF
generation or CGC outcome access. Selection is based only on startup satellite
support. The pool, order, UTC, location, displacement, receiver seed, and
counterfeit carrier-phase seed are fixed in
`configs/experiments/cgc_temporal_final_static_pool_v1.json`.

## Allowed observation

Generate one second of clean 1 MHz IQ for each declared candidate, retain only
the simulator text log, count its startup GPS L1 C/A LOS entries, and delete the
IQ immediately. No false trajectory, counterfeit component, multi-tap receiver
output, signed-delay estimate, Partial-F score, or class label may be generated
or inspected during this step.

For slots S1--S5, select the first listed candidate with at least ten startup
LOS PRNs. If neither candidate in a slot qualifies, stop. Do not add or replace
a candidate after observing support.

## Boundary before release

Copy the five selected rows and counts into the final-test configuration. The
configuration, evaluation protocol, and runner must be committed before the
first selected 25 MHz source is generated. After release, failed pairs and
failed gates are retained; thresholds, temporal window, observability gate,
tap count, intervals, or pair roster cannot be tuned or rerun.
