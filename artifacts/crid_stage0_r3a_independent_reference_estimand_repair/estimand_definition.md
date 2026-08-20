# CRID R3a independent-reference estimand definition

Status: `POST_R3_RESULT_METHOD_REPAIR_PRE_IMPLEMENTATION`

This is a versioned post-R3 method repair. It does not revise R3: the R3 verdict remains permanently `INCONCLUSIVE_CONTROL_PROVENANCE`. The disclosed R3 legacy result is 171/180 PASS with nine failures, all and only OAK PRN 21. All recovered target delays were exact; the nine realized powers were approximately 1.02–1.11 dB below their requested values. The registered cause hypothesis is an estimand mismatch between the generator's five-PRN joint complex-LS denominator and the validator's single-PRN projection denominator.

## Frozen estimand

For the five validated PRNs in a domain, let `S0` be the matrix whose columns are independently reconstructed zero-delay authentic replicas on the first 1000 complete 1 ms epochs of the frozen common interval. The authentic coefficient vector is

`alpha = solve(S0^H S0, S0^H x)`.

The implementation streams complex128 Gram and RHS contributions; it does not construct a full TEX matrix. The source samples are decoded from little-endian interleaved int16 to float64/complex128 before accumulation.

For a positive-control output, the residual is `r = complex128(output) - complex128(source)`. Its support is fixed to the legacy terminal hold: ten complete 1 ms epochs at offsets 5.0 through 5.9 seconds in 0.1-second steps. At every common candidate delay `d` on the inclusive `[-0.4, 0.4]` grid with 0.01-chip steps, `Sd` contains all five independently shifted PRN templates and

`beta(d) = solve(Sd^H Sd, Sd^H r)`.

For each PRN, the recovered delay is the grid value maximizing that PRN's `|beta_prn(d)|`; its counterfeit coefficient is the corresponding component. Relative power is exactly

`20*log10(|beta_prn|/|alpha_prn|)`.

The legacy single-PRN projection remains a diagnostic and is never used for the R3a gate.

## Independent construction

The new reference module must not import the generator module or its render, injection, envelope, delay, phase, or amplitude functions. It independently implements GPS L1 C/A shift registers and reconstructs absolute sample coordinates, R0c NAV signs, receiver code phase, and receiver carrier phase from the authenticated TRACE/NAV inputs. Generator-fitted or truth coefficients are prohibited as solver inputs; they may appear only in post-solve comparison diagnostics.

Every system must contain exactly five distinct PRN columns, contain only finite values, have rank exactly five, and have condition number at most `1e6`. Any provenance, binding, independence, rank, finite-value, or condition failure closes the gate as inconclusive.

## Frozen evaluation and gate

All 180 rows are retained: OAK and TEX, all 36 positive cases, all five domain PRNs, delays 0.05/0.15/0.30 chip, powers 0/-3/-6 dB, and single/four target modes. Target delay error must be at most 0.025 chip, target power error at most 0.75 dB, and non-target relative energy at most 0.01. There is no PRN 21 exception, tolerance expansion, data-specific correction, offset subtraction, or row averaging/exclusion.

The legacy validator must independently reproduce 171 PASS and nine identical OAK PRN 21 failures. A PASS additionally requires 180/180 joint-reference PASS, rank five and condition at most `1e6` everywhere, deterministic rerun equality, unchanged R3 artifact/control hashes, zero attack bytes, and all registered tests. Only then may R3a set `next_state` to `READY_TO_REPEAT_CRID_PHASE_A`; it still must not execute Phase A.

Attack payload access, CRID score computation, threshold/alarm evaluation, Phase A, C1/C2/C3 replay, control-IQ regeneration, generator changes, and any mutation of existing R3 artifacts are forbidden.
