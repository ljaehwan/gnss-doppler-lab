# MOSAIC Stage-0B R1 receiver-in-loop preregistration

Status: **READY_FOR_R1_EXECUTION**. This commit freezes the production int16 I/Q path, stateful R0c NAV/NCO replica contract, 72-case target assignment, receiver binding, analysis grid, controls, and GO/NO-GO rules. It does not execute injection, receiver replay, or inspect results.

Both clean sources are little-endian interleaved signed int16 I/Q with four bytes per complex sample. Counterfeit amplitude is referenced to the target PRN's complex least-squares `alpha_auth`, never total clean-IQ RMS. The zero-amplitude path must preserve bytes exactly and nonzero paths preserve sample count while reporting saturation.

The R0c intervals are frozen to OAK `[150275296, 210202273)` and TEX `[817815304, 1117517038)`. Each run uses 0–2 s identity, 2–4 s raised-cosine ramp, 4–10 s hold, and a smooth release to interval end. Code/carrier/NAV/absolute-sample state may not restart at chunk or epoch boundaries.

No scientific verdict is emitted at preregistration. A future authorized execution must apply the frozen BIC-corrected H0/H1 analysis and all GO criteria. Receiver/input failures map to `INCONCLUSIVE_RECEIVER_IN_LOOP`; physical recovery failures map to `NO_GO_MOSAIC_INJECTOR_PHYSICS`.
