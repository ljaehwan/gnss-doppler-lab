# PG-SCC Stage-0 static K<=9

PG-SCC asks whether the physical evidence in a dense complex two-source CAF fit can be retained by a receiver-compatible, globally fixed set of at most nine complex correlations. It is not a B0/M1 fusion and it does not train an attack-label classifier. The learned object is a single input-independent query mask shared by every epoch and PRN.

## Data and freeze boundary

The base is `442ed57e25a84eab0ddcecfc61f073c9d01cf837` from `research/acaf-nf-stage1-r3-static-detection`. The inherited cleanStatic L20 cache binds raw 25 MHz interleaved int16 IQ to GNSS-SDR tracker MAT state and passed the independent R2a foundation audit: 1,200 L20 windows, 24,000 raw-recomputed 1 ms CAFs, 0.99833 within +/-50 Hz, and no attack access during that audit. No zero-center fallback is allowed.

cleanStatic roles are chronological, raw-range non-overlapping intervals: train 10--45 s, selection 47--62 s, calibration 64--82 s, and holdout 84--100 s, with guard gaps. The design runner accepts only the inherited `clean_features.npz/json` cache. It serializes `frozen_design.json` and `freeze_manifest.json`; the evaluator verifies every protected SHA-256 before it opens attack cache or IQ.

## Complex normalization and covariance

For complex CAF vector (y\in\mathbb{C}^{187}), prompt-phase normalization is

\[
  \tilde y = y\,e^{-j\arg y_p}/(|y_p|+10^{-9}).
\]

The comparison local-energy normalization divides by √(mean |y|^2). Both are tested under global gain and carrier-phase transforms. Prompt-phase normalization is preregistered because it explicitly fixes both nuisance variables while retaining relative complex phase.

The authentic template (a) is the coordinatewise complex median of normalized clean train CAFs. Clean H0 residuals are obtained after the complex least-squares amplitude fit. Because 180 train windows are fewer than 187 complex coordinates, the covariance is fully shrunk to its clean-only diagonal with a positive floor. This is a declared shrinkage covariance, not a C/N0 feature.

## Dense physical teacher

H0 and H1 use identical coordinate support:

\[
 H_0: y=\beta_0 a+n,
 \qquad
 H_1: y=\beta_a a+\beta_s s(\delta\tau,\delta f)+n.
\]

The same-PRN second-source template is evaluated directly, never by shifting an array:

\[
 s_{d,f}(\delta\tau,\delta f)=
 \max(1-|d-\delta\tau|,0)\,\mathrm{sinc}((f-\delta f)T)
 e^{j\pi(f-\delta f)T},\quad T=1\,\mathrm{ms}.
\]

The nuisance complex coefficient βs contains relative amplitude and carrier phase. The search covers signed delays {-0.75,-0.375,-0.125,0,0.125,0.375,0.75} chips and Dopplers {-150,-75,0,75,150} Hz, excluding (0,0). Each model uses ridge 1e-3. With clean covariance Σ and support S, the detector score is

\[
 G_S(y)=\max(\mathrm{RSS}_{H0,S}-\min_{\delta\tau,\delta f}\mathrm{RSS}_{H1,S},0)/|S|,
 \quad
 \mathrm{RSS}=r^H\Sigma_S^{-1}r.
\]

The dense K=187 score is the physics teacher and upper-bound reference. Sparse detectors recompute the same fit on their K coordinates.

## Synthetic bank and selectors

Synthetic H1 is constructed at complex-correlator level by adding the direct same-PRN template to real cleanStatic CAFs. The sweep includes signed near-zero to resolvable delays, zero and signed Doppler, 12 phases over 2π, weak/matched/strong amplitudes, and three complex-noise levels. SHA-256 of the parameter tuple assigns disjoint train/validation combinations. Different PRNs, magnitude-peak composition, shifted zero padding, and real attack labels are prohibited.

S0 greedily adds coordinates that reduce ridge regression error to the dense teacher score. S1 optimizes global soft-top-K weights with an explicit NumPy Adam implementation. Its loss contains teacher MSE, clean H0 positive-evidence penalty, synthetic H1 pairwise ranking, gain/phase/noise robustness through normalization and the bank, concentration regularization, and an exact-K soft weight sum. A deterministic top-K projection freezes one global mask. Unconstrained and center-plus-inversion-symmetric masks are compared only on synthetic validation.

PRN-local scores use median, robust mean, or top-third mean pooling; synthetic validation freezes the choice. q99 and q99.5 thresholds use only 14 cleanStatic calibration event scores. This is scientifically tail-limited: the independent audit must report that 14 events cannot precisely estimate q99/q99.5.

## Evaluation and claims

Core attacks are DS3, truncated DS4, DS7, and DS8. DS4 permits transition-only claims. DS7/DS8 count as one non-independent family. Attack pre-onset is external FPR evidence and never recalibrates thresholds. B0 is exact only if its native checkpoint/scorer can run on identical epoch/PRN support; otherwise it is `UNAVAILABLE` and no historic score CSV is copied.

Controls cover gain, carrier phase, AWGN, single-source power, same-PRN H1, random/uniform/shuffled masks, boundary concentration, raw-power alarm overlap, dense/sparse rank correlation, and raw-IQ compute. Compute timing includes the IQ read, carrier wipeoff, C/A replica, K correlations, normalization, and scoring inputs; it is not a cached-surface-only Python timing.

Claimable scope is physical two-source evidence-preserving sparse placement, K<=9 receiver compatibility, real-attack-free physics-guided query design, normal-only threshold calibration, and an actual dense/sparse compute tradeoff. It does not claim first CAF, first GLRT, first nine-tap, first normal-only, or cross-site generalization.

## Reproduction

```bash
PYTHONPATH=src pytest -q tests/test_pg_scc.py tests/test_pg_scc_physics.py
PYTHONPATH=src python3 scripts/train_pg_scc_selector.py
PYTHONPATH=src python3 scripts/verify_pg_scc_artifacts.py --freeze-only
git commit  # frozen design checkpoint, before attack access
PYTHONPATH=src python3 scripts/eval_pg_scc_static.py
PYTHONPATH=src python3 scripts/verify_pg_scc_artifacts.py
```
