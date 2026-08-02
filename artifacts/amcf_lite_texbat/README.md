# AMCF-Lite TEXBAT feasibility

## Final status

**NO-GO — developmental/post-exposure exploratory evidence only.**  DS1-DS3, DS7 and DS8 were already exposed before this study; no result is confirmatory. DS4 is **NA** because its complex9 producer differs.

- exact source commit: `6138894e6f35b1408db25958b001ce08ea99203c`
- base commit: `1537c958f7e82a83a32cd40a7d5c22a1328b3cf8`
- no raw-IQ or GNSS-SDR rerun
- no attack row used for Prompt gate, model fit, validation, threshold calibration, or tap-policy tuning
- primary model: `complex adaptive K7`; primary threshold: clean calibration q99, strict `score > threshold`
- masked-set model: frozen seeded ELM-style nonlinear token encoder plus learned coordinate-conditioned Student-t location/scale decoder; **not a fully end-to-end Set Transformer**

## Primary result

- independent clean FPR: **4.92%** (criterion <=1.5%: **FAIL**)
- all scenario stable-pre FPR <5%: **PASS**
- adaptive K consistently beats fixed/random same-budget policies: **FAIL**
- K5 or K7 approaches all9 on every scenario: **FAIL**
- material matched-FPR improvement over B0-Exact: **FAIL**

```text
scn  A7-pre  A7-ROC  A7-PR   A7-post sustained  B0-ROC B0-post  C9-ROC Mag9-ROC
DS1   1.00%  0.9601  0.9948   88.40%    26.0  0.9736   93.65%  0.9495  0.4838
DS2   1.00%  0.9828  0.9977   88.81%    12.0  0.9898   97.06%  0.9718  0.2396
DS3   0.00%  0.9750  0.9967   90.08%    22.5  0.9592   93.99%  0.9732  0.6482
DS7   0.83%  0.7550  0.9116   36.96%   107.5  0.8899   58.70%  0.8130  0.8191
DS8   0.83%  0.8515  0.9747   54.44%   100.5  0.9815   89.58%  0.9373  0.9174
```

A7 = complex adaptive K7; C9 = complex all9. Sustained delay requires three consecutive 0.5 s alarms. Full q99/q99.5 rows are in `metrics.csv`; clean-test-matched diagnostic rows are in `matched_metrics.csv`.

## Questions answered

1. **Does complex beat magnitude-only?** On all-nine taps, yes strongly on DS1-DS3 and modestly on DS8; not on DS7 ROC-AUC. This supports the narrow claim that complex phase-conditioned structure can carry information missed by magnitude-only features, but not universal dominance.
2. **Does adaptive K beat fixed/random K?** No. K5 wins 8/20 scenario/competitor ROC comparisons; K7 wins 10/20. It degrades sharply on DS7/DS8. Moreover, the learned uncertainty ranking degenerates to a static path on every clean and attack epoch: K5 always adds `E4,E3`; K7 always adds `E4,E3,E2,L2`. Thus this run provides no evidence that the selector made sample-dependent informative queries.
3. **Does K5/K7 approach all9?** On DS1-DS3 often yes, but not on DS7/DS8; the all-scenario criterion fails.
4. **Does matched-FPR adaptive K7 beat B0-Exact?** No. Adaptive K7 has lower post-onset detection than B0-Exact in every scenario under the recorded matched-clean-FPR diagnostic.

## Input/QA and policy

Canonical tensor: `complex_iq[N,9,2]`, component order I/Q, tap order `E4,E3,E2,E,P,L,L2,L3,L4`. Prompt referencing is `C*conj(P)/|P|^2` after a clean-train-only q0.005 quality gate. Actual global-phase and common navigation-sign invariance errors, near-zero Prompt rates, and per-tap magnitude/phase/wrapped phase-curvature distributions are in `qa.json`.

Each raw row maps to its first causal 0.5 s grid. The latest available row per PRN is retained with deterministic tie breaking; no future value is allowed and PRN identity is not a model feature. Actual median tracked N is cleanStatic 11, DS1 10, DS2 11, DS3 11, DS7 10, DS8 11.

Seed taps are E/P/L. Fixed, seeded random, adaptive uncertainty, and all-nine paths reveal only selected taps. Added taps are scored before reveal. Per-epoch selection frequencies are in `tap_selection_histogram.csv`; per-epoch JSON histograms are in `per_epoch/`.

## Scientific interpretation

The useful signal is **complex-field availability**, not the current active policy. The learned heteroscedastic uncertainty is not a reliable information-gain proxy across producer/domain shifts. The optimizer also reached its predeclared 25-iteration cap for all three representations (`optimizer_success=false`), so this lightweight decoder should not be treated as a converged final model.

This feasibility result does not support an SCI/WCL detector-performance claim. A defensible follow-up would require a fully trained masked-set model, normal recordings from additional environments/receivers, policy selection on normal-only domains, and a genuinely external attack holdout.

## Relation to prior work

AMCF-Lite does not claim novelty for complex correlator features, CAF-DNNs, sparse LASSO correlator decomposition, or complex CCAF decomposition. CAF-DNN work typically uses a dense delay/Doppler field; LASSO methods solve an explicit sparse dictionary decomposition; recent complex CCAF decomposition retains a richer complex ambiguity field and component interpretation. AMCF-Lite uses only nine tracking delay taps, no Doppler axis, diagonal predictive scales, no physical component identifiability, and a clean-only masked-prediction objective. Exact bibliographic overlap must be rechecked before publication.

## Files

- `provenance.json`, `qa.json`, `model_audit.json`, `config.json`, `thresholds.json`
- `metrics.csv`, `matched_fpr.csv`, `matched_metrics.csv`
- `adaptive_gain.csv`, `budget_summary.csv`, `tap_selection_histogram.csv`
- `feasibility_summary.json`, `per_epoch/`, `plots/`, `test_summary.txt`, `hashes.json`
