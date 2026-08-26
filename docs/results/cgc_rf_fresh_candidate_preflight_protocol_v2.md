# CGC RF fresh candidate support preflight v2

## Purpose

This preflight prevents a second final-test failure caused only by insufficient visible satellites. It is not a model experiment and it cannot support the scientific claim by itself.

The nine candidate rows, their order, and all later RF, motion, multipath, and spoof parameters are fixed in `configs/experiments/cgc_rf_fresh_candidate_pool_v2.json` before any candidate signal is generated.

## Permitted operation

For each candidate, generate a one-second, 1 MHz clean gps-sdr-sim probe at the candidate's fixed UTC and initial position. Parse only the simulator's startup azimuth/elevation table. Retain the text log and discard the probe IQ.

No correlator tracking, delay estimation, CGC score, multipath generation, spoof generation, model inference, AUC, or endpoint calculation is permitted.

## Selection rule

For each motion kind in this exact order—static, straight, parallel-sweep—select the first candidate in configuration-file order with at least 10 startup LOS PRNs. Ten is a predeclared margin above the final analysis requirement of eight. If a motion kind has no eligible candidate, the fresh test cannot be frozen from this pool.

Selection uses no retry, score, or outcome-dependent substitution. After selection, the three full candidate rows are copied without parameter changes into a separate final-test configuration and committed together with its runner and final protocol before 25 MHz source or receiver generation.

## Interpretation boundary

The output proves input support only. It is excluded from CGC performance evidence and cannot be cited as detection accuracy.
