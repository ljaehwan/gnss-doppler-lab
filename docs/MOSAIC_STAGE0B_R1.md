# MOSAIC Stage-0B R1 receiver-in-loop preregistration

R1 is frozen from R0c commit `e0993bd6b16628681b52c1abd52cf177af67e10a`. This commit implements and tests the execution path but deliberately stops at `READY_FOR_R1_EXECUTION`; it does not generate injected IQ, run the receiver, or inspect outcomes.

Production I/Q is little-endian interleaved signed int16 `(I,Q)`, four bytes per complex sample. The R1 path preserves bytes for zero amplitude, preserves sample count for all amplitudes, saturates to int16, and records clipping. The older foundation int8 quantizer is not used.

Counterfeit amplitude is relative to the target PRN amplitude estimated by complex least squares on the clean baseline. Replica code phase, carrier phase, Doppler accumulation, NAV sign, and absolute sample index persist across streaming chunks and receiver-NCO epochs. NAV signs come from the frozen R0c corrected mapping.

The 72-case design has canonical SHA-256 `b1a06556f7cd67738274c132f80b0581b20914d971f72f4e4ab0b5efc9a7facf`. Target assignment, timing envelope, CAF grid, H0/H1 definitions, BIC correction, collapsed controls, strong subset, and all GO/NO-GO criteria are frozen in the preregistration artifact.

Run `python3 scripts/verify_mosaic_stage0b_r1.py` to verify the frozen preregistration. `scripts/run_mosaic_stage0b_r1.py --execute` intentionally refuses execution in this commit; a separately authorized task must execute the campaign without changing the preregistered rules.
