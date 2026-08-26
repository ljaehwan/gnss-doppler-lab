# Correlator–geometry identifiability audit (train v1)

## Outcome

The train-only matched-control audit supports **Correlator Geometry Consistency (CGC)** as an observability mechanism that requires complex multi-correlator taps. It does not yet validate an RF detector.

All 1,800 matched event pairs preserved the complete single-PRN complex-profile multiset exactly. Consequently, a single-PRN width feature had AUC 0.500 by construction. The only class difference was whether each observed profile remained associated with its original satellite LOS vector.

| Observation/score | Pooled AUC | 95% paired-event bootstrap CI |
|---|---:|---:|
| Single-PRN magnitude width | 0.500 | exact matched marginal |
| Oracle signed delay + geometry | 1.000 | 1.000–1.000 |
| Stored-style magnitude 9 taps + geometry | 0.671 | 0.654–0.687 |
| Proposed complex 9 taps + geometry | 0.864 | 0.853–0.875 |

The complex-tap gain over magnitude-only was **+0.194 AUC** (paired-bootstrap 95% CI 0.178–0.211). Per-geometry complex AUC ranged from 0.823 to 0.891 across the six train geometries. Complex signed-delay estimation achieved 0.0859-chip MAE and 0.849 sign accuracy for truth delays with magnitude at least 0.05 chips; magnitude-only achieved 0.1639-chip MAE and 0.720 sign accuracy. All frozen exploratory support gates passed.

## Mechanism and algorithm

For satellite `j`, the controlled complex profile is

```text
z_j(x) = R(x - mu_j) + rho_j exp(i phi_j) R(x - mu_j - delta_j) + n_j,
```

where `R` is the ideal triangular GPS L1 C/A correlation. A 19,680-profile deterministic physical dictionary estimates the signed secondary delay from nine taps after prompt-phase alignment.

A coherent false receiver displacement imposes the cross-satellite law

```text
delta_j = -u_j^T (Delta r / L_chip) + b,
H_j = [-u_j^T, 1].
```

CGC fits the four parameters `theta = [Delta r / L_chip, b]` and scores

```text
r_G = min_theta ||delta_hat - H theta||_2^2 / (||delta_hat||_2^2 + epsilon),
spoof score = -r_G.
```

The matched multipath control deranges complete profiles across PRN identities. This preserves every single-satellite empirical observation while breaking the shared four-parameter LOS relation. The result therefore isolates cross-satellite association rather than allowing a marginal peak-shape cue to leak into classification.

## Interpretation

This experiment narrows the earlier failure to an observability issue: the current receiver export stores magnitudes for the extra taps, which loses much of the signed-delay information needed by the geometry test. Preserving complex I/Q at all nine taps is the concrete receiver change suggested by this audit.

The oracle AUC of 1.0 shows that the low-dimensional law is sufficient under the controlled assumptions. The gap from oracle to complex AUC shows that local delay inference, not the geometry fit itself, is now the dominant problem.

## Claim boundary

- Only train pairs `pv1-pair-001` through `pv1-pair-006` were used. Validation, test, and TEXBAT data remained untouched.
- The autocorrelation was ideal and triangular; frontend bandwidth, sampled C/A sidelobes, tracking dynamics, oscillator effects, and AGC were not modeled here.
- “Multipath” is a deliberately hard identity-deranged matched control, not a ray-traced or RF-generated environment. It establishes marginal identifiability, not field false-alarm performance.
- The current GNSS-SDR receiver output does not contain complex I/Q for all nine taps, so the proposed observation has not passed the actual receiver chain.
- This is an exploratory train result and cannot support a deployable threshold or a WCL generalization claim without frozen validation and RF multipath experiments.

## Reproduction and next gate

Run:

```bash
.venv/bin/python scripts/audit_correlator_geometry_identifiability.py
```

The canonical output is `artifacts/correlator_geometry_identifiability_train_v1/summary.json` (SHA-256 `ae29fb8d5a52675bbbb3fcf2afde6f32f1ac17b2aced9ab99de03ec8ea3acbc9`). The output directory is 17 MiB.
A byte-verified SSD copy is stored at `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/correlator-geometry-identifiability-train-v1/`.

Next, export complex I/Q for every correlator tap, freeze the signed-delay estimator and CGC score, generate actual RF multipath controls, and then make one untouched run on validation pairs 007–009.
