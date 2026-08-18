# MIRAGE Stage-0A

MIRAGE tests whether a receiver-centred, raw-IQ complex delay–Doppler CAF is
locally separable for one source and becomes nonseparable for two sources. It
does not fit an analytic triangular ACF template and does not use CAF energy,
RMS, recovered delay/Doppler, PRN identity, or an existing detector score as its
primary feature.

For every adjacent delay/Doppler cell the primary field is

```
|C[a,p]C[b,q] - C[a,q]C[b,p]|
---------------------------------------------------------
sqrt(|C[a,p]C[b,q]|² + |C[a,q]C[b,p]|² + epsilon)
```

There are 32 values at each of 20, 100, and 500 ms. The delay grid is
`[-.5, -.375, -.25, -.125, 0, .125, .25, .375, .5]` chips. Doppler offsets keep
`xi=delta_f*T` fixed at `[-2,-1,0,1,2]`. Train-only robust minor references and
empirical CDFs produce three scale surprises; a PRN node is their maximum and a
Full score is the median over at least four available PRNs. Dataset-specific q99
thresholds may come only from clean calibration.

The source contract is deliberately fail-closed. A 500-ms CAF may use only raw
samples covered by authenticated receiver NCO state and the corrected R0c NAV
mapping. No NAV-bit extrapolation is allowed. Clean train, calibration, and
holdout require distinct raw ranges and 10-second guards. The preregistered
minimum is therefore 29 seconds: three seconds for each role plus two guards.

The inherited R0c authorization provides only 11.9853954 seconds for OAKBAT and
11.98806936 seconds for TEXBAT with all five PRNs in common. Consequently this
run must produce `INCONCLUSIVE_INPUT_OR_SUPPORT` before threshold calibration,
IQ injection, or receiver replay. Weakening the guard or extrapolating NAV bits
would change the experiment and is prohibited.

The algebraic implementation remains useful and is tested with literal,
independently specified rank-1 and rank-2 matrices. Tests also cover common
complex gain/global phase invariance, equal-magnitude phase structure,
low-energy stability, NAV wipeoff, chunk continuity, deterministic case design,
PRN permutation, temporal desynchronization preservation, and clean-path
enforcement.

Commands:

```bash
python3 scripts/run_mirage_stage0a.py preregister
python3 scripts/run_mirage_stage0a.py evaluate --preregistration-sha <pushed-sha>
python3 scripts/verify_mirage_stage0a.py
```

The sole next action after this support verdict is to obtain at least 29 seconds
of common five-PRN authenticated NAV/NCO support per clean dataset. No real
attack evaluation or score rescue is authorized.
