# ACAF-NF Stage-0-R1.1 validity repair

R1.1 is a new fail-closed, cleanStatic-only raw-IQ/tracker center-reconstruction campaign. It does not alter R1 or baseline methods, and it does not read or claim attack data. The tracker remnant formula is `aux1_samples * code_freq_chips / fs`; the signed reconstruction is `remnant_sign * remnant_chips`.

All stable valid PRNs are parsed. Bounded raw selections use deterministic PRN round-robin strata and are restored to chronological order; their actual raw windows are non-overlapping. Every center-validation row carries scenario, channel/MAT path, tracker index, raw sample and byte bounds, raw SHA-256, tracker MAT SHA-256, aux1 samples, converted remnant chips, tracker Doppler, raw/MAT prompt magnitudes, reconstructed peak delay/Doppler offsets, center magnitude, and grid-boundary status.

The campaign reports R0 as mean absolute reconstructed peak delay offset (chips) and R1 as mean absolute reconstructed peak Doppler offset (Hz), along with true center-peak rate and raw-center/MAT-Prompt Spearman correlation. Gate A can fail without modification of the data or thresholds. In that case Gate B/two-source inference is not evaluated, Gate C remains incomplete if DS4/B0 provenance is missing, and the result is not a physics NO-GO or a performance claim.
