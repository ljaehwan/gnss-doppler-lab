# ACAF-NF Stage-0 static R1.3 reconstruction

R1.3 repairs the invalid R1.2 raw-IQ reconstruction. It is a reconstruction
validity experiment on `cleanStatic`, not a detector or model evaluation. The
source phase adds implementation, synthetic tests, documentation, and an
independent verifier; it does not execute the production raw-IQ campaign.

## Receiver-source binding

The receiver build is bound to
`/home/ubuntu/build-gnss-sdr-complex9`, GNSS-SDR Git base
`1ddd4562723040fd66cb334b578a5b69455625f4`. The locally modified tracking file
is `src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc`, SHA-256
`6d2db43fea2728acc35fb29b4cb5027b62be442b8966108bf6b955d5b95f486c`.

The decisive source excerpts (line numbers in that modified file) are:

- 1061–1092: `do_correlation_step` passes remnant carrier phase, carrier phase
  step, remnant code phase, and forward code phase step to the multicorrelator.
- 1137–1159: the tracking loop updates carrier Doppler and code frequency.
- 1222–1231 and 1268–1291: the next interval length is computed from actual
  code frequency plus remnant samples, the code step is forward
  `d_code_freq_chips / fs_in`, and
  `d_rem_code_phase_chips = d_code_freq_chips * d_rem_code_phase_samples / fs_in`.
- 1435–1443: dumped Prompt I/Q is the complex Prompt correlator output.
- 1482–1518: the dump stores
  `PRN_start_sample_count = nitems_read(0) + d_current_prn_length_samples`, then
  carrier Doppler, code frequency, lock/CN0 fields, and `aux1` as
  `d_rem_code_phase_samples`.
- 1927–1934: correlation is performed over the current input interval and its
  Prompt value is saved.
- 2199: exactly `d_current_prn_length_samples` are consumed.

Therefore MAT row *k* is compared to raw interval `[stamp(k-1), stamp(k))`.
Prompt row *k* describes that interval. The dumped NCO/remnant values are audit
fields at its end; source-constrained candidates explicitly identify every row
used, reject cross-PRN references, and permit only forward replica progression.
R1.3 never substitutes a fixed 25,000 samples: adjacent stamps determine the
physical interval, including 24,999 and 25,001 sample epochs.

## Physical reconstruction

For every delay bin, R1.3 regenerates chip indices from

`floor(base + n*code_freq/fs + remnant_sign*aux1*code_freq/fs + delay) mod 1023`.

It imports the canonical `gps_l1ca_code`; it has no local C/A generator. Carrier
wipeoff is multiplied into IQ as
`exp(j*carrier_sign*2*pi*(tracker_doppler+doppler_offset)*n/fs)`. Fingerprints
cover the selected raw interval, code indices, wipeoff, applied rows, and CAF
result. Global offsets change actual sample and byte bounds and invoke CAF again.

Stable triples require same channel and PRN, strictly increasing in-bounds
stamps, finite physical fields, C/N0 at least 28 dB-Hz, and carrier lock at least
0.85. Cross-PRN same-time overlap is allowed. Train/calibration/holdout time
intervals cannot overlap.

## Fail-closed interpretation

The independent verifier recomputes counts, balance, center recovery, Prompt
rank correlations, gates, fingerprints, offset calls/ranges, and checksums.
Any A1 failure is `SOURCE_BINDING_INVALID`; A2 failure is
`RECONSTRUCTION_IMPLEMENTATION_INVALID`; A3 failure is
`TRACKER_RAW_ALIGNMENT_UNRESOLVED`. Only all-pass yields
`PHYSICAL_CENTER_VALID`. A failed reconstruction clears selection and never
becomes a physics or model-failure claim.
