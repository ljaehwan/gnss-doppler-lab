# Candidate Method M1: Raw-IQ Noise/Fingerprint Continuity Detector

Date recorded: 2026-07-29
Status: candidate second method after B0, not a replacement for B0 yet
Script: `scripts/iq_noise_continuity_detector.py`

## Position relative to B0

B0 remains the frozen operational baseline:

```text
btail_max_507080_ewma075
= tap9-only PRN-node GRU residuals
+ clean-calibrated binomial-tail PRN-set support gate
```

This document records a second, independent candidate direction, M1. M1 is not a
tracking-residual detector and does not use the B0 GRU checkpoint. It tests a
different physical-layer hypothesis:

```text
Authentic GNSS raw IQ has a locally continuous receiver/RF/noise fingerprint.
When a spoofer source enters, that fine-grained raw-IQ fingerprint trajectory may
break even if the navigation/code/carrier structure is successfully imitated.
```

The word "micro-noise" here means raw pre-correlation IQ noise/fingerprint texture,
not the B0 GRU tracking residual and not PRN-node morphology residuals.

## Core protocol

For each TEXBAT scenario, fit only on the same recording's early authentic prefix:

```text
fit:  window_start_s <= 90 s
score: whole recording, causally forward
labels: not used for fitting or threshold calibration
```

This is a self-calibrated online anomaly detector. It intentionally avoids using
attack windows from the same scenario for fitting, model selection, or threshold
calibration.

## Feature and model sketch

Raw TEXBAT interleaved IQ is read before GNSS tracking/despreading. The current
PoC uses 10 ms raw-IQ blocks with 0.5 s stride and extracts a compact fingerprint:

- I/Q mean, standard deviation, and I/Q correlation
- amplitude and power moments
- phase-increment statistics and phase coherence
- short-lag complex autocorrelation statistics
- normalized PSD band powers, spectral entropy, and flatness
- high-frequency amplitude-difference moments

Then:

1. Robustly standardize features using the 0--90 s fit prefix.
2. Project to a low-dimensional PCA fingerprint space.
3. Fit a linear autoregressive continuity predictor in that PCA space using only
   the fit prefix.
4. Score future windows by AR prediction error and companion feature-level drift.
5. Smooth causally with EWMA and threshold by the fit-prefix quantile.
6. Report onset-oriented metrics: pre-onset alarms, 90 s to onset gap alarms,
   first post-onset alarm, first 3-consecutive post-onset alarm, and post-onset
   alarm rate.

Default command pattern:

```bash
python scripts/iq_noise_continuity_detector.py \
  --scenario ds3 \
  --fit-end-s 90 \
  --block-ms 10 \
  --stride-s 0.5 \
  --pca-dim 8 \
  --lag 6 \
  --q 0.99
```

The script writes event scores, a timeline plot, and `summary.json` under:

```text
/home/ubuntu/ssd_data/gnss-early-detection/artifacts/<scenario>-raw-iq-noise-continuity-20260729-v0/
```

## Initial TEXBAT snapshot

These are first-pass PoC results from the v0 script. They should be treated as a
research note, not as a frozen benchmark.

- ds2, onset 100 s
  - fit: `window_start_s <= 90.0`
  - gap 90--100 s alarm rate/count: `0.0% / 0`
  - first post alarm: `100.0 s`, delay `0.0 s`
  - first 3-consecutive post alarm: `100.0 s`, delay `0.0 s`
  - post-onset alarm rate: `97.76%`
  - pre-onset alarm rate/count: `1.03% / 2`

- ds3, onset 100 s
  - fit: `window_start_s <= 90.0`
  - gap 90--100 s alarm rate/count: `73.68% / 14`
  - first post alarm: `100.5 s`, delay `0.5 s`
  - first 3-consecutive post alarm: `104.5 s`, delay `4.5 s`
  - post-onset alarm rate: `98.46%`
  - pre-onset alarm rate/count: `8.25% / 16`
  - caution: this scenario shows substantial pre/onset-transition warning, so it
    needs cleaner interpretation before claiming onset precision.

- ds4, onset 110 s
  - fit: `window_start_s <= 90.0`
  - gap 90--110 s alarm rate/count: `35.90% / 14`
  - first post alarm: `110.0 s`, delay `0.0 s`
  - first 3-consecutive post alarm: `111.0 s`, delay `1.0 s`
  - post-onset alarm rate: `97.30%`
  - pre-onset alarm rate/count: `7.48% / 16`
  - caution: transition/gap alarms indicate that the detector may be responding
    before the nominal onset label or to scenario-specific ramp/artifact effects.

- ds7, onset 110 s
  - fit: `window_start_s <= 90.0`
  - gap 90--110 s alarm rate/count: `0.0% / 0`
  - first post alarm: `117.5 s`, delay `7.5 s`
  - first 3-consecutive post alarm: `117.5 s`, delay `7.5 s`
  - post-onset alarm rate: `95.34%`
  - pre-onset alarm rate/count: `0.93% / 2`

DS2 and DS7 are the cleanest early evidence because the 90 s--onset gap stays
quiet and alarms persist after onset. DS3 and DS4 are still useful but must be
explained with care because they produce warning alarms in the transition/gap
region.

## Scientific guardrails

Do not overclaim this as "satellite-unique micro-noise" yet. The current v0
features are raw-IQ texture features and may include several shortcuts:

- received-power or AGC distribution shift
- spectrum-shape or bandpower shift
- quantization/front-end artifacts
- TEXBAT scenario construction artifacts
- spoofing transmitter RF fingerprint rather than authentic satellite fingerprint

Safe current claim:

```text
The raw-IQ noise/fingerprint trajectory learned from the local authentic prefix
shows continuity breaks around spoofing onset in several TEXBAT scenarios.
```

Unsafe current claim:

```text
We have isolated GPS satellite-specific microscopic noise fingerprints.
```

## Required ablations before promoting M1

Before M1 can be compared seriously against B0 or written as a paper method, run
at least these ablations:

1. Block-wise power normalization and/or AGC-like amplitude normalization.
2. Remove or separately report PSD bandpower features.
3. Phase-only and autocorrelation-only variants.
4. PSD-whitened raw-IQ variant.
5. CleanStatic and cleanDynamic negative controls, using the same 0--90 s
   self-calibration protocol.
6. If possible, PRN-despread residual/noise-floor continuity to separate
   receiver-level raw-IQ shortcuts from PRN-specific post-correlation evidence.
7. Compare explicitly with GNSS RFF/noise-like-feature prior work, especially
   convolutional-autoencoder noise-like features and pre-correlation RFF surveys.

## Relation to likely prior work

The broad idea of GNSS RF fingerprinting / noise-like features for spoofing is
not new. M1's potential contribution is narrower:

```text
online, per-recording, normal-only self-calibration of raw-IQ noise/fingerprint
trajectory continuity, evaluated as whole-recording onset detection
```

Closest prior-work keywords to track:

- GNSS RF fingerprinting / RFF
- GNSS pre-correlation sampled data
- noise-like feature assisted GNSS spoofing detection
- satellite fingerprinting for GNSS spoofing detection
- one-class novelty detection / convolutional autoencoder / SVDD / VAE-GAN

## Current implementation status

- Implementation: `scripts/iq_noise_continuity_detector.py`
- Default sampling assumption: TEXBAT 25 Msps, signed 16-bit interleaved IQ
- Output schema: `gnss-doppler-lab.raw-iq-noise-continuity.v0`
- Baseline status: candidate M1 only; B0 remains the frozen baseline.
