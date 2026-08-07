# ACAF-NF Stage-0 static R1.4 Doppler validation

R1.4 is a `cleanStatic` scientific reconstruction diagnostic. It freezes the
R1.3 candidate and checks R1.3 source/artifact hashes and reference metrics
before downstream validity. It does not select an alignment, calibrate a
detector, read an attack recording, compare B0, or make an ACAF-NF claim.

The physical configuration is canonical GPS L1 C/A, interleaved signed-int16
IQ at 25 MHz, raw offset zero, previous-row NCO and auxiliary state, remnant and
carrier signs -1, forward replica, current Prompt, and an authenticated fixed
25,000-sample support. Raw data are memory-mapped one support at a time; the
48 GB recording is never loaded as a whole.

Prompt magnitude reproduction and code-delay recovery are reconstruction
gates. Delay passes at 95% within 0.125 chip, at most 1% boundary, every role at
95%, and at least seven of eight PRNs at 95%. Doppler is reported separately.
A 1 ms coherent observation has approximately `1/T = 1 kHz` main-lobe width,
so a 50 Hz Doppler argmax requirement is a resolution diagnostic rather than a
reconstruction-failure gate.

Noncoherent L=1,5,10,20 results use the same L=20-capable anchors. A block is
consecutive within one channel, PRN, and role, meets C/N0 and lock limits, and
has valid raw support. Every constituent uses its own tracker state. The
predeclared surface is
`mean_k(|C_k|^2 / (sum_grid |C_k|^2 + eps))`; raw power sum, magnitude mean,
and robust median are diagnostics. Fixed-seed PRN-block bootstrap confidence
intervals compare L=5/10/20 with paired L=1 anchors.

Residual Doppler is `NOT_APPLICABLE` unless MAT complex Prompt continuity is
authenticated. If later authenticated, the diagnostic may use
`q=(P/(|P|+eps))^2`, phase unwrap, and slope; squaring assumes navigation-bit
sign changes are 180 degrees. It can never pass a gate.

Run the later campaign only with:

```bash
python scripts/run_acaf_nf_stage0_static_r14_doppler_validation.py \
  --raw /path/to/cleanStatic.bin --tracker-dir /path/to/tracker-mats \
  --manifest /path/to/manifest.json --execute-production
python scripts/verify_acaf_nf_stage0_static_r14_doppler_validation.py \
  artifacts/acaf_nf_stage0_static_r14_doppler_validation --write-report
```

If A3a and A3b pass while A3c fails, raw alignment work stops: delay-domain
active CAF is the priority and Doppler remains diagnostic until longer coherent
integration is available.
