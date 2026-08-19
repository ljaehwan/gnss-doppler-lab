# CINDER Stage-0A Clean Emitter Identifiability

Final verdict: `NO_GO_CINDER_CLEAN_IDENTIFIABILITY`.

## Scope and source binding

Only clean OAKBAT and clean TEXBAT were read. No attack recording, attack label, or neural model was used. OAKBAT is the 5 Msps little-endian interleaved int16-I/Q source with SHA-256 `8e3428abb1b94211118c1ec9f505322ef89fdb176f0cf33961eca3cf3da80dfe`; TEXBAT is the 25 Msps source with SHA-256 `dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9`. Both entire files were streamed and hashed in this run.

OAKBAT used PRNs 10, 11, 21, 24, and 27 over receiver time 35.5000666–356.5005134 s and raw support `[177500333,1782502567)`. TEXBAT used PRNs 3, 13, 16, 19, and 30 over 35.50026632–356.50084444 s and `[887506658,8912521111)`. The direct R0c NAV intervals remained a separate alignment gate. The longer scientific support is authorized only by tested even-order NAV-sign invariance and is not reported as directly decoded NAV support.

Each dataset has 30 independent ten-second scientific parent blocks: 8 feature/statistics-train, 8 metric-train, 6 calibration, and 8 final-holdout blocks. Ten-second guards are blocks 8, 17, and 24. One nested 100/500/1000 ms target family is used per parent block, with no split overlap.

## Frozen physical feature

The primary feature explicitly estimates the conjugate-balanced fourth-order cyclic cumulant `cum[x(t),x*(t-tau1),x(t-tau2),x*(t-tau3)]_alpha` and removes all three Gaussian pair products. Authenticated receiver state supplies carrier wipeoff, code phase/frequency, and absolute raw endpoints. Four common fractional-chip coordinates preserve the chip waveform; two fixed edge/inner contrasts remove the nominal constant chip component. The cyclic grid is 0, 1/8, and 1/4 cycles/chip (0, 127875, and 255750 Hz), with lag tuples `(0,0,0)`, `(1,0,1)`, `(1,1,2)`, `(2,1,3)`, and `(4,2,4)`. The 30-complex-value C4 vector becomes an 88-real-value compact Hermitian quotient. The metric is feature-train-only robust scaling plus frozen diagonal shrinkage Mahalanobis; calibration alone selects the threshold.

All 300 resampling audits pass: total out-of-bounds queries 0 and maximum fractional-source error 0. At most two consecutive raw samples beyond the nominal last 1-ms endpoint are read to interpolate a supported absolute fractional-chip coordinate; these remain inside the same target and parent block.

## Primary result

- OAKBAT 500 ms Full-C4: median AUC 0.537117, worst seed 0.513724, seed-1 parent-block 95% CI `[0.483427,0.639182]`.
- TEXBAT 500 ms Full-C4: median AUC 0.532041, worst seed 0.514592, seed-1 parent-block 95% CI `[0.427233,0.605486]`.
- Sensitivity AUCs (100/500/1000 ms): OAKBAT 0.515434/0.537117/0.480230; TEXBAT 0.560306/0.532041/0.399158.
- Leave-one-PRN-out AUC ranges: OAKBAT 0.498512–0.522693; TEXBAT 0.493383–0.530161. All ten PRN-pair AUCs per dataset are reported; their ranges are 0.451531–0.544962 and 0.465242–0.578763. There is no isolated strong transmitter subgroup.

## Same-support baselines and controls

At 500 ms, OAKBAT B0–B7 median AUCs are 0.459184, 0.507602, 1.000000, 0.494439, 0.601505, 0.599643, 0.500000, and 0.788342. TEXBAT values are 0.731607, 0.503087, 1.000000, 0.487883, 0.584617, 0.593495, 0.500000, and 0.405179. B7 CHORD was recomputed from native nine-tap projective directions on this support; no prior number was copied. PN-SCAT is 0.627372/0.500204. Diagnostic-only C4+PN-SCAT is 0.610000/0.435255 and is not primary.

The ideal/code-only B6 is exactly 0.5 and its physical feature is zero under ideal replica, circular-shift, and chip-permutation controls. Permutation medians over 200 repetitions are all inside 0.45–0.55. Receiver-common removal changes seed-1 matched C4 AUC from 0.512194 to 0.561531 on OAKBAT and 0.477347 to 0.519337 on TEXBAT; it does not reveal preregistered identifiability. Exact-gap nuisance-matched/unmatched results, nuisance balance, time-separation, LOPO, and all PRN pairs have separate tables.

Gain 0.5/0.8/1.2/2.0, carrier phase 0/pi/4/pi/2/pi, and NAV ±1 controls pass on mathematical and representative real-raw inputs (maximum float32 feature error `1.06e-8`, tolerance `1e-7`; NAV error zero). The centered `[0.02,0.96,0.02]` filter perturbation retains feature cosine 0.999979/0.999909. Empirical clean-residual AWGN at 0.5 sigma retains 0.988563/0.993572 cosine; 1–2 sigma robustness degrades and is preserved as a limitation rather than called a pass.

## Decision

Failed decisive gates include both datasets' AUC <0.70, CI lower bound <=0.60, baseline margin <0.10, worst seed <0.60, and both sensitivity alternatives <0.65. Doppler and, on TEXBAT, C/N0 are strong shortcuts, further showing why they cannot be emitter features. The permutation and exact nuisance invariance controls pass, but they cannot rescue the absent clean C4 signal.

The defensible contribution is a preregistered, source-bound negative clean identifiability result for this feature/support. It does not establish or refute all possible GNSS RF fingerprints, does not demonstrate spoofing detection, and makes no cross-dataset identity claim. The single recommended action is to preserve this negative artifact and not implement the adaptive emitter-slot Stage-0B model.
