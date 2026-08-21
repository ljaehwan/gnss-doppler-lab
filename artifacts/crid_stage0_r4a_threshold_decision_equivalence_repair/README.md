# CRID R4a threshold decision-equivalence repair

Final verdict: `THRESHOLD_DECISION_EQUIVALENCE_REPAIR_PASS`

R4 remains permanently `INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE` with threshold status `INCONCLUSIVE_THRESHOLD_BINDING`. This versioned method repair does not rewrite R4 and does not replace its thresholds. The committed R2 literals remain authoritative: OAK `-21.705587048010322`, TEX `-21.942672917134093`.

The clean-only float64 recomputation differences are OAK `3.872457909892546e-13` and TEX `1.7408297026122455e-13`. On the identical recomputed clean holdout vectors, committed and recomputed thresholds produced byte-identical alarms. OAK reproduced 32/4383 = `0.00730093543235227`; TEX reproduced 58/4564 = `0.012708150744960562`.

No control replay or control scoring was performed. No attack path was stated, hashed, opened, memory-mapped, or read. Phase A and Phase B were not executed. A PASS authorizes only a future repeat of CRID Phase A.
