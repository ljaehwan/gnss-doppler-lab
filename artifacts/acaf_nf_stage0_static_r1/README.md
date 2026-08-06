# ACAF-NF Stage-0-R1

This is a static-only raw-IQ CAF foundation repair. It corrects the prior 1023/fs code-rate and negative G2 indexing errors: GPS L1 C/A uses **1.023e6 chips/s**, and ICD G2 taps are one-based positive stages. Every stored CAF is reread from s16 complex raw IQ and centered with tracker PRN, PRN_start_sample_count, carrier_doppler_hz, code_freq_chips, and aux1. Query budgets count complex CAF coordinates, not real/imag scalars. Two-source diagnostics use shifts of one same-PRN CAF template.

**Execution validity:** BLOCKED. **Verdict:** `EXPERIMENT_BLOCKED_MISSING_TRACKER_ds4_NATIVE_B0_SAME_EPOCH_PANDAS_DEPENDENCY_MISSING`. A blocked R1 is neither PHYSICS_NO_GO nor a detection claim. Missing full raw hashes, tracker coverage, center recovery, common-support bootstrap, or same-epoch native B0 validity must be repaired before GO/NO-GO. Existing B0 is not reported as comparable here because this R1 run has not completed its same-epoch native B0 input/score gate.
