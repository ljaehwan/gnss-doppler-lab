# MIRAGE Stage-0A complex-minor feasibility

Verdict: **INCONCLUSIVE_INPUT_OR_SUPPORT**.

The complex-minor algebra and source SHA lineage passed, but controlled
injection was not executed. Validated five-PRN NAV/NCO common intervals are
shorter than the preregistered 29 s minimum needed for 3 s train + 10 s guard +
3 s calibration + 10 s guard + 3 s holdout. Extrapolating NAV bits or weakening
the two 10-second guards is prohibited, so the experiment failed closed before
clean threshold calibration or injection. Placeholder plots explicitly record
this non-execution and are not scientific results.

## Sources and grid

OAKBAT `cleanStatic_gps.bin` is little-endian interleaved int16 I/Q at 5 MHz,
SHA-256 `8e3428abb1b94211118c1ec9f505322ef89fdb176f0cf33961eca3cf3da80dfe`.
TEXBAT `cleanStatic.bin` uses the same format at 25 MHz, SHA-256
`dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9`.
The receiver SHA is `2f6e8e969e525bb48b4d94f016af8fd24f433b0be26b51837f316f60a6b911e0`.
Tracker/Prompt provenance is the 470/470 passing MOSAIC Stage-0A R1 raw
re-correlation. NAV signs and endpoints bind to corrected R0c mapping SHA
`db7e23761e2bb4314d6f22e990e23c0fb19387ee7476a55a665def7a10af522f`.

The frozen CAF uses 20/100/500 ms, nine delays from -0.5 to +0.5 chips, and
normalized Doppler `xi=-2..2`, corresponding respectively to offsets
`[-100,-50,0,50,100]`, `[-20,-10,0,10,20]`, and `[-4,-2,0,2,4]` Hz.

Each adjacent determinant is:

`|C[a,p]C[b,q]-C[a,q]C[b,p]| / sqrt(|C[a,p]C[b,q]|²+|C[a,q]C[b,p]|²+epsilon)`.

It is invariant to a common complex gain/global phase and distinguishes the
independent literal rank-2 known vector. The literal rank-1 maximum minor was
exactly 0; the rank-2 maximum was 0.669534. The SVD second-energy diagnostic was
0.0272805, but SVD is not the MIRAGE score.

## Support verdict and unexecuted metrics

OAK authorization is `[150275296,210202273)`, 11.9853954 s, leaving a
17.0146046 s deficit. TEX authorization is `[817815304,1117517038)`,
11.98806936 s, leaving a 17.01193064 s deficit. Both contain five PRNs and can
hold an individual 500-ms window, but neither can construct the required
chronological clean roles.

Consequently all of the following are `NOT_RUN_PREREQUISITE_FAILED`:

- clean q99/q99.5 calibration, holdout FPR, and threshold stability;
- single-PRN and simultaneous four-PRN receiver-in-loop injection;
- collapsed/gain/AWGN/drift/drop-add controls;
- multi-scale ablation and injection ROC/PR/pAUC/detection rates;
- temporal desynchronization, RMS/C/N0 shortcut, and PRN dominance metrics.

The frozen 72-case table remains available for a future support-complete rerun:
30 single-PRN and six four-PRN cases per dataset. Per dataset, power occurs 12
times per level, delay six times per level, phase nine times per level, and
Doppler seven or eight times per level.

## Claim boundary and next action

No TEXBAT DS or OAKBAT OS recording, attack label, neural model, threshold
rescue, or injection result was accessed. The algebraic normalized-minor
implementation and its invariances are supported. Whether real single-source
CAF is sufficiently rank-1, or controlled two-source injection breaks that
structure detectably, is **not** supported by this run. MIRAGE therefore makes
no physical-detection or novelty claim.

The only next action is to acquire at least 29 seconds of authenticated common
five-PRN NAV/NCO support for each clean dataset and rerun the unchanged frozen
experiment.
