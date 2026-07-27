# Realistic normal-channel RF impairments

`gnss_doppler_lab.rf_impairments` post-processes gps-sdr-sim interleaved signed
8-bit GPS L1 C/A composite IQ in bounded memory. It diversifies **benign normal**
training data; explicit composite-channel effects can also support stress and
hard-negative simulation.

## Profiles and strict configuration

Omitting `impairments` preserves the legacy disabled path and performs no
post-processing. An empty mapping is rejected rather than launching an
expensive identity pass.

The realistic pipeline default is:

```yaml
impairments:
  enabled: true
  profile: open_sky_normal
  seed: 1548829001
```

`open_sky_normal` accepts only `enabled`, `profile`, and `seed`. Its complete
realization is derived from the seed and sample rate. It uses a composite sample
SNR of -14 to -8 dB, a low-pass cutoff at 0.76 to 0.92 of Nyquist, oscillator
error, mild IQ/DC error, and a receiver AGC target of 22 to 28 signed-8-bit LSB
complex RMS. It deliberately has **no common composite multipath, fading, or
ripple**, because those layers impose artificial cross-PRN correlation.

The clean profile must be disabled:

```yaml
impairments:
  enabled: false
  profile: clean
```

Use `profile: explicit` with `enabled: true` and at least one effect for custom
processing. Preset overrides and contradictory enabled/profile combinations are
rejected. An explicit `multipath` list and `gain`/`fading_*`/`ripple_*` remain
available, but they are common **composite-channel approximations**, not
satellite-specific propagation. `gain` is fixed signal-channel gain; it is not
AGC. `agc_target_rms` controls the actual receiver AGC.

The normal-v3 runner defaults to `open_sky_normal`, deriving an exact stable
uint64 seed from each run ID. Disable post-processing with:

```bash
python scripts/run_normal_v3_large_pipeline.py --limit 1 --impairment-profile clean
```

## Layer model v2

The ordered model is:

1. optional common composite fractional-delay multipath;
2. common oscillator carrier offset, linear drift, and phase random walk;
3. stateful Butterworth frontend low-pass;
4. optional common composite fixed signal gain/fading/ripple;
5. complex AWGN, explicitly frontend-output/ADC-input-referred;
6. receiver IQ gain/phase imbalance and DC, applied after noise is present;
7. receiver AGC applied to signal plus noise;
8. symmetric clipping, rounding, and signed-8-bit IQ quantization.

A tap with delay `d=k+α` and complex gain `a exp(jφ)` uses linear fractional
interpolation:

`x_d[n] = (1-α)x[n-k] + αx[n-k-1]`.

The oscillator phase is:

`θ[n] = 2π(f0 n/Fs + 0.5 fdot (n/Fs)^2) + Σ ε[m]`,

where `ε ~ N(0, σ_phase²)`. Delay, filter, phase-walk, and absolute-time state
continue across chunks.

For deterministic frontend-output composite signal power `P`, requested sample
SNR `S` sets complex noise variance:

`σ_n² = P / 10^(S/10)`.

At 2.6 Msps, -14 to -8 dB composite sample SNR corresponds to about 50 to 56
dB-Hz composite C/N0. Under the simplifying equal-power assumption, 4 to 10
visible PRNs imply roughly 36 to 50 dB-Hz per PRN across the profile range.
**The composite is not per-PRN**, powers are not generally equal, and GNSS-SDR
calibration is required before interpreting receiver C/N0.

## Two-pass bounded-memory processing

The implementation never writes a complex channel intermediate. Pass one reads
the clean recording, hashes it, regenerates the deterministic signal channel,
and accumulates powers/statistics. It then resets all channel state and RNGs.
Pass two regenerates the same chunks, adds noise and receiver effects, applies
AGC, and writes one atomic final temporary s8 IQ file while computing final hash,
power, clipping, and count statistics. The final output is not reread.

For a five-minute 2.6 Msps recording this removes the former 12.48 GB complex128
channel temporary. Extra channel-intermediate disk usage is zero; only the
atomic final s8 temporary (the same size as the final recording) is needed.

The manifest separates requested and realized settings and records clean,
deterministic-channel, and output hashes; exact byte/sample counts; pre/post AGC
and quantized powers; applied receiver gain; clipping count/fraction; requested,
analog, and quantized SNR where meaningful; NumPy/SciPy versions; PCG64; normalized
filter cutoff and SOS; layer model/version/order; and model caveats. Output is
chunk-size invariant for the recorded runtime. Cross-version byte identity is
not claimed because numerical-library implementations can change.

Before manifest publication, the pipeline requires simulator-reported bytes,
filesystem bytes, impairment counts, and final output bytes to agree with the
active runner's exact output contract. For the pinned gps-sdr-sim build, the
contract accounts for its documented/verified 0.1-second initialization block:
it writes `(duration_seconds * 10 - 1)` blocks of `floor(Fs/10)` complex
samples. The manifest records both nominal requested-duration bytes and the
runner-contract expected bytes, so the resulting 0.1-second-shorter actual
recording cannot be mistaken for truncation.
