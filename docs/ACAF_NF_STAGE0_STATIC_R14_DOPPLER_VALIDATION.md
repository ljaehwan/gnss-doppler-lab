# ACAF-NF Stage-0 static R1.4 Doppler validation

R1.4 is a `cleanStatic` scientific reconstruction diagnostic. It freezes the
R1.3 candidate and checks the approved R1.3 source SHA-256
`9889a5e5007c92d6016e5ef0d38a03cea96cdd40eded3cea91df1e4276d16e42`
and checksums-manifest SHA-256
`04b5395b311641b4ab3f3a58a1a5cbb54d4249068f8252659049ea4386a95abb`.
It rehashes every manifest entry, enforces the exact R1.3 inventory and PASS
verification, and freezes the checksum-bound 969-row center-validation identity
order before checking independently recomputed reference metrics. It does not select an alignment, calibrate a
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
has valid raw support. A source-authenticated one-sample overlap (24,999 start
sample delta for fixed 25,000-sample supports) is retained and audited rather
than deleting an anchor; larger overlaps, duplicate constituents, and role
crossing fail closed. Every constituent uses its own tracker state and digest.
`per_block_scores.csv` records all epoch complex surfaces plus each exact
anchor/L constituent list, interval, overlap audit, common-anchor ID, primary
surface and three independently recomputable diagnostic surfaces. Prompt
reproduction is also emitted per PRN, per role/time block, and per actual
tracker channel in `prompt_reproduction_by_channel.csv`; the verifier rebuilds
all grouped metrics from raw epoch evidence. The artifact has exactly 26
allowed top-level files plus the `plots/` directory, whose exact seven SVG
relative paths are enforced recursively with no extra directory or symlink.
Each SVG has labeled axes and legends and embeds a canonical evidence payload
and SHA-256 that the verifier independently reconstructs from CSV evidence.
The predeclared surface is
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
