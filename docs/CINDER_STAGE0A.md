# CINDER Stage-0A clean emitter identifiability

This experiment asks whether clean GPS transmitters remain distinguishable in raw IQ after removing PRN code, NAV sign, received scale, global carrier phase, absolute Doppler, time, and receiver-common state. It is a physical identifiability gate, not an attack detector and not a neural-model experiment.

The primary statistic is the conjugate-balanced fourth-order cyclic cumulant

`cum[x(t), x*(t-tau1), x(t-tau2), x*(t-tau3)]_alpha`,

with the three Gaussian pair products explicitly removed. Raw samples are carrier-wiped and code-aligned using authenticated receiver state, interpolated at four fixed fractional-chip positions, and reduced to edge and inner pulse-shape contrasts. The fixed cyclic frequencies are 0, 1/8, and 1/4 cycles/chip; the lag tuples are `(0,0,0)`, `(1,0,1)`, `(1,1,2)`, `(2,1,3)`, and `(4,2,4)`. The primary 500 ms window and 100/1000 ms sensitivities are nested in one non-overlapping window per ten-second parent block.

The compact nuisance quotient contains the diagonal and first upper diagonal of `vv^H/(||v||^2+epsilon)`. A clean-only diagonal shrinkage Mahalanobis metric is trained after feature-train-only robust scaling. Calibration selects the threshold; final chronological holdout never changes the feature or metric contract.

All results, controls, split boundaries, source hashes, and the final preregistered verdict are stored in `artifacts/cinder_stage0a_clean_emitter_identifiability/`.
