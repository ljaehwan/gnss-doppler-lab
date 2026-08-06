# ACAF-NF Stage-0 static R1.3 reconstruction

R1.3 repairs the invalid R1.2 raw-IQ reconstruction. It is a reconstruction
validity experiment on `cleanStatic`, not a detector or model evaluation. The
source phase adds implementation, synthetic tests, documentation, and an
independent verifier; it does not execute the production raw-IQ campaign.

## Receiver-source binding

The receiver build is bound to
`/home/ubuntu/build-gnss-sdr-complex9`, GNSS-SDR Git base
`1ddd4562723040fd66cb334b578a5b69455625f4`. Preflight records SHA-256 for all
four modified tracked files (`dll_pll_veml_tracking.cc/.h` and
`dll_pll_conf.cc/.h`), the full Git diff, CMake/compiler evidence, and receiver
executable. Values are collected from the actual source/build tree.

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

This ordering disproves the earlier mapping. Correlation uses pre-update state
and exactly `d_trk_parameters.vector_length` samples. For this GPS L1 C/A
configuration the adapter computes `vector_length = 25,000`. Logging occurs
after the loop update and records a boundary made from `nitems_read(0)` plus the
new dynamic consume length. Adjacent stamp differences (including
24,999/25,001) are consumed-boundary evidence, not correlator support.

The MAT dump does not persist the correlator call's `nitems_read(0)` start or
pre-update remnant/NCO state. R1.3 records the authenticated support length
separately, leaves its raw start/end null, and forces A2 false. It never guesses
a 25,000-sample window at an adjacent boundary.

## Physical reconstruction

For every delay bin, R1.3 regenerates chip indices from

`floor(base + n*code_freq/fs + remnant_sign*aux1*code_freq/fs + delay) mod 1023`.

It imports the canonical `gps_l1ca_code`; it has no local C/A generator. Carrier
wipeoff is multiplied into IQ as
`exp(j*carrier_sign*2*pi*(tracker_doppler+doppler_offset)*n/fs)`. Application
evidence stores separate hashes for raw content and range, replica indices,
wipeoff, aux/NCO/Prompt indices, and result field. Candidate labels or combined
metadata hashes are not application evidence. Only physically applied aux/NCO
rows, signs, and global offset remain variable; Prompt is fixed to current row.

Stable triples require exact integer stamps, same channel and PRN, strictly
increasing in-bounds stamps, consumed differences no larger than 25,001, no
reacquisition/PRN transition, finite fields, C/N0 at least 28 dB-Hz, and lock at
least 0.85. Cross-PRN same-time overlap is allowed. Cross-role consumed
intervals cannot overlap.

The bound configuration is `ishort`, 25 MHz, no skip, pass-through resampling,
GPS L1 C/A, nine taps at 0.125 chip, one-symbol integration, and no pilot.
`Prompt_I/Q` are single-step complex `d_Prompt`; nine-tap `P` is separately
accumulated in `d_tap_accu[4]`. Their equivalence is not authenticated. E/P/L
exist but are not substituted for the Prompt target.

## Fail-closed interpretation

Before IQ or MAT evaluation, the runner authenticates recording identity, full
raw SHA/size/path, configs, executable, and MAT inventory/hashes against the
receiver manifest. Any mismatch rejects the run.

The independent verifier recomputes counts, balance, center recovery, Prompt
rank correlations, gates, fingerprints, offset calls/ranges, and checksums.
Any A1 failure is `SOURCE_BINDING_INVALID`; A2 failure is
`RECONSTRUCTION_IMPLEMENTATION_INVALID`; A3 failure is
`TRACKER_RAW_ALIGNMENT_UNRESOLVED`. Only all-pass yields
`PHYSICAL_CENTER_VALID`. A failed reconstruction clears selection and never
becomes a physics or model-failure claim. With current evidence A2 necessarily
fails, so production stops before creating artifacts or reading IQ/MAT content.
