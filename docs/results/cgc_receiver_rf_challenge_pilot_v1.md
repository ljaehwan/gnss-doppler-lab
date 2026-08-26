# CGC receiver/RF challenge pilot v1

Date: 2026-08-26
Role: exploratory train-only mechanism development

## Question

Can a cross-satellite correlator-geometry consistency (CGC) score distinguish a
coherent carry-off spoof from physically generated, satellite-specific
multipath after both signals pass through the same RF front end and GNSS-SDR
tracking loops?

This run uses only `pv1-pair-001` from the train partition. Validation pairs,
test pairs, and TEXBAT attack recordings were not accessed.

## Receiver and RF upgrades

- GNSS-SDR upstream commit `1ddd4562723040fd66cb334b578a5b69455625f4`
  was patched to retain complex I/Q for all nine tracking correlator taps.
- The MAT dump keeps the legacy magnitudes and adds
  `tap_I_{E4..L4}` / `tap_Q_{E4..L4}`.
- gps-sdr-sim upstream commit
  `28ca29a6719475195e3aabd5930c4ed02d67190f` was patched with a
  per-PRN fractional-delay echo before satellite summation.
- Each echo has an independent PRN, delay, amplitude ratio, and carrier phase.
  This avoids the invalid common-composite-delay control.

Reproducible patches:

- `patches/gnss-sdr-complex9-tracking-v1.patch`
- `patches/gps-sdr-sim-prn-multipath-v1.patch`

## Frozen provenance

- Original gps-sdr-sim executable SHA-256: `27cd471d2023b8a26262584b14f71ebed602159da47b54172b22da9299011e28`
- Artifact-generating PRN-multipath executable SHA-256: `c18c9615cff053b7fc7ba310c0717fdcf56abdcd26a8a5408c682c4a09b3ef87`
- Corrected PRN-multipath executable SHA-256: `012ca5d2eef7bdec91e07732bbd7e1c0dfd817f3263ec0a94aecf1584f0adc46`
- Corrected PRN-multipath patch SHA-256: `7d523547e8c3f43c3631606a0dbbc5417d1173f21d8c031f14a9b6a5f7b9ee79`
- Patched GNSS-SDR executable SHA-256: `fc00d610fdb966358ac85486b616ff2411cb163ca8269dd0b435919c5e197e25`
- Legacy frozen geometry module SHA-256: `74732c182e799c2e82b4719d6572ebbd3dc4f6e16eada5bfe0f8a3d8c236b2b2`
- New clock-centered module SHA-256: `b25a00efdb654ceb96540a2597bdc542526a0c6027e393018c739394858a203a`

The original simulator path and legacy geometry module remain byte-identical
to their earlier frozen-validation hashes. The multipath executable and
post-hoc score live at separate paths, so this pilot cannot silently alter the
previous validation lineage.

Post-run review found and corrected a `-i` / `-m` switch-order regression
that was outside the executed `-m` signal path. Short `-i` and `-m`
regressions were byte-identical, and a full 1,495,000,000-byte corrected
regeneration matched the artifact component at SHA-256
`5e7f671c5019b704cee3ae89359f4f389ed0f97b8bb880ab040895ec18fdc5fd`.
The component manifest preserves both provenance records.

## End-to-end complex-tap smoke result

The existing 29.9 s pair-001 normal RF was processed by the patched receiver.

| Check | Result |
|---|---:|
| Complex-tap PRNs | 11 |
| Total tracking epochs | 143,690 |
| Complex I/Q to stored magnitude error at float32 precision | 0 |
| Complex prompt tap to legacy Prompt_I/Q error | 0 |
| Overall smoke gate | PASS |

Artifact: `artifacts/cgc_complex9_receiver_smoke_v1`

## PRN-multipath RF verification

A short deterministic RF test established:

- Patched simulator without `-m` was byte-identical to the legacy pair-001
  component for 5,000,000 bytes.
- Two identical PRN-8 echo runs produced the same SHA-256.
- For a 12.5-sample delay, the first changed byte was byte 25, i.e. the I
  component of zero-based complex sample 12, as required by the fractional
  interpolation law.

The full pilot generated 12 PRN-specific echoes. Realized delay range was
0.1381--0.4386 chips and amplitude-ratio range was 0.2067--0.6403. The
multipath component was then processed using the exact pair-001 frozen gain,
AWGN variance, front-end filter, oscillator impairment, and receiver seed.

## Initial result: legacy score failed

The pre-existing score normalized the full geometry residual against delay
energy around zero:

    R0 = SSE_full / SSE_zero
    SSE_full = sum_j w_j (delay_j - fitted_full_j)^2
    SSE_zero = sum_j w_j delay_j^2

On 18 post-carry-off one-second bins:

| Legacy metric | Multipath | Spoof |
|---|---:|---:|
| Median residual (R_0) | 0.1920 | 0.3310 |

Using (-R_0) as the spoof score gave exploratory AUC 0.2593. The direction
was wrong.

## Root cause and corrected physical score

Physical multipath delays are nonnegative and have a substantial common mean.
The fitted clock-bias intercept absorbs that mean, while the legacy denominator
still includes its energy. Independent multipath can therefore receive an
artificially small normalized residual.

The corrected score compares the full LOS-plus-clock model with the
intercept-only clock null:

    Rdir = SSE_full / SSE_clock
    SSE_clock = sum_j w_j (delay_j - weighted_mean(delay))^2
    Cdir = 1 - Rdir

Directional coherence Cdir is the partial R-squared contributed by the three
LOS columns beyond the clock nuisance. If
the clock-only denominator is zero, coherence is defined as zero.

| Clock-centered metric | Multipath | Spoof |
|---|---:|---:|
| Median residual (R_{dir}) | 0.8141 | 0.4026 |

Using (-R_{dir}) as the spoof score gave exploratory AUC 1.000 and a
multipath-minus-spoof median residual separation of +0.4115.

## Interpretation boundary

The corrected score was derived after observing the legacy failure on this
train RF pilot. Therefore AUC 1.000 is not validation evidence, has no
confidence interval, and must not be reported as detector performance. The
18 bins are serially related observations from one satellite geometry.

What this pilot does establish is narrower and useful:

1. Complex nine-tap data survive the full RF/receiver pipeline exactly.
2. Satellite-specific multipath can be generated without artificial
   cross-PRN common delay.
3. Zero-referenced residual normalization has a general physical confound.
4. Removing the clock-only nuisance produces a concrete new candidate.

## Next gate

Freeze the clock-centered candidate and repeat the receiver/RF challenge on
unused train pairs 002--006, including static and dynamic motion. Only after
that replication should preprocessing and any decision threshold be frozen.
The already inspected validation partition cannot validate this post-hoc
candidate; the still-locked test protocol must be preregistered before access.

Primary artifact: `artifacts/cgc_rf_challenge_pilot_v1`

## SSD archive

- Complex-tap smoke:
  `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cgc-complex9-receiver-smoke-v1`
  (73,459,085 bytes)
- Receiver/RF challenge:
  `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cgc-rf-challenge-pilot-v1`
  (3,162,824,008 bytes)

Both archives passed a content-checksum rsync dry run with no differences.
