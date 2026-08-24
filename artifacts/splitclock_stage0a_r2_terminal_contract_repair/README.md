# SPLITCLOCK-GNSS Stage-0A R2 terminal contract repair

This versioned terminal repair changed only the three preregistered R1
contracts: dynamic-modality centering, persistent per-PRN latent likelihood,
and matched-horizon persistence evaluation. The repair-scope freeze is
`a645980a81e94499efaebbc3287f0a263302e7e9`; the implementation freeze is
`470ffa33a92b6ff41469b6cf375d58d952797409`.

The frozen C-1/C-3 execution produced
`NO_GO_SPLITCLOCK_R2_CLEAN_FALSE_ALARMS` with terminal recommendation
`TERMINATE_SPLITCLOCK_NO_FURTHER_GATE_RELAXATION`. Holdout epoch FPR and
persistent false alarms were both zero, but the score/PRN-count correlation
was -0.4815206478200818 and violated the preregistered absolute 0.3 gate.
The primary 3/5-PRN mild, moderate, and accelerating detection rates were all
zero, so the synthetic identifiability gate independently failed. Matched
horizon A0/A6 AUROC was 0.4305555555555556/0.5694444444444444, while temporal
and modal destruction did not produce the required advantage reduction.

All negative and non-identifiable all-PRN boundary controls had zero
persistent alarms. No attack or Jammertest raw input was statted, hashed,
opened, memory-mapped, or read. This terminal clean-only result does not
authorize an attack pilot and no further threshold or gate relaxation is
permitted.
