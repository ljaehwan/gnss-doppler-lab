# ACAF-NF Stage-0-R1 foundation repair

## Scope

Stage-0-R1 is a **static-receiver physical validity repair**, not the full ACAF-NF neural field or active policy. It rereads TEXBAT raw signed-int16 complex IQ and computes a complex local-replica delay--Doppler CAF. Tracker products contribute only the center lineage: `PRN`, `PRN_start_sample_count`, `carrier_doppler_hz`, `code_freq_chips`, and `aux1`.

## Corrections over the rejected R0 implementation

- G2 selector stages use the ICD's one-based positive indices (`g2[tap - 1]`); Python negative indexing is forbidden.
- GPS L1 C/A code-rate is `1.023e6` chips/s, not `1023` chips/s. At 25 Msps, a 1 ms coherent replica traverses all 1023 chips.
- PRN3 / zero-code-phase / zero-Doppler CAF is prohibited as physical evidence.
- Each actual CAF center uses tracker sample count, PRN, carrier Doppler, code frequency, and `aux1` code phase, and the tracker MAT plus raw IQ have full SHA-256 entries in the data manifest.
- H0 requires at least 20 chronological clean observations. R1 schedules 0.5 s-spaced cleanStatic train, calibration, and held-clean ranges rather than R0's 2/1/1 sample structure.
- K counts complex CAF cells; real and imaginary components cannot be counted as two queries.
- Two-source diagnostics shift a template belonging to the same PRN rather than mixing unrelated PRN templates.

## Fail-closed validity contract

R1 can issue `GO_FOR_ACAF_NF`, `PHYSICS_GO_ACTIVE_QUERY_NO_GO`, `POWER_SHORTCUT`, or `PHYSICS_NO_GO` only after all of the following pass:

1. independently computed full SHA-256 raw source binding;
2. actual tracker MAT availability and SHA binding for every required scenario;
3. tracker center recovery that does not frequently peak on the CAF grid boundary;
4. time-origin evidence that maps recording-relative tracker time to TEXBAT onset/pull-off definitions;
5. exact DS7/DS8--cleanStatic raw overlap provenance and exclusion;
6. 10 s common-support bootstrap;
7. native B0 re-score on identical `window_bin_s` support.

Any failure produces `EXPERIMENT_BLOCKED_<reason>`, never a physical NO-GO. Descriptive raw CAF outputs are marked not-for-decision.

## Comparators and scope exclusions

B0 is a frozen PRN-local 9-tap GRU plus binomial-tail/EWMA gate. It can only be compared after its native 1.0 s / 0.5 s 9-tap input and history contract is recreated on the identical R1 epochs. M1 and R2C are not modified. `cleanDynamic`, DS5, and DS6 are OOD and excluded from the static core gate. DS7 and DS8 are one dependent family, never two independent successes.

R1 does not claim a new neural CAF model, sequential policy, a replacement for prior CAF/LASSO/CCAF methods, or spoof detection until this validity contract passes.
