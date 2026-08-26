# Correlator–geometry identifiability validation v1

## Decision

The frozen Correlator Geometry Consistency (CGC) candidate was **confirmed on the three held-out satellite geometries** `pv1-pair-007` through `pv1-pair-009`. This decision applies only to the controlled correlator-domain identifiability experiment. It is not an RF multipath or receiver-chain validation.

The protocol was committed and pushed as `756e248482e1121995b42bf9993e5f44d463846c` before any validation pair was accessed. The dictionary, generator ranges, 300 events per geometry, 1,000 bootstrap repetitions, and all support thresholds were inherited unchanged from train. Only the validation generator/bootstrap seeds and the preassigned LOS geometries changed. The runner refuses to overwrite an existing validation output.

## Results

| Observation/score | Train AUC | Validation AUC | Validation 95% bootstrap CI |
|---|---:|---:|---:|
| Single-PRN magnitude width | 0.500 | 0.500 | exact matched marginal |
| Oracle signed delay + geometry | 1.000 | 1.000 | 1.000–1.000 |
| Magnitude 9 taps + geometry | 0.671 | 0.672 | 0.650–0.693 |
| Complex 9 taps + geometry | 0.864 | **0.859** | 0.843–0.876 |

The validation-minus-train complex AUC difference was −0.00484. The complex-tap improvement over magnitude-only remained **+0.188 AUC** with paired-bootstrap 95% CI 0.165–0.209.

| Held-out geometry | PRNs | Magnitude AUC | Complex AUC |
|---|---:|---:|---:|
| `pv1-pair-007` | 11 | 0.616 | 0.826 |
| `pv1-pair-008` | 15 | 0.719 | 0.888 |
| `pv1-pair-009` | 13 | 0.689 | 0.875 |

Complex signed-delay estimation achieved 0.0877-chip MAE and 0.845 sign accuracy for truth delays of magnitude at least 0.05 chips. Every frozen gate passed:

- exact single-PRN complex-profile multiset matching;
- width AUC equal to 0.5;
- oracle AUC at least 0.95;
- pooled complex AUC at least 0.80;
- every held-out geometry complex AUC at least 0.75;
- complex delay MAE at most 0.10 chips;
- complex delay sign accuracy at least 0.80.

The nested `diagnostic.exploratory_status` retains the train helper's label `supported_on_train_requires_validation`; the authoritative post-validation decision is the top-level `validation_status = confirmed_on_heldout_geometry`.

## LOS validation input

Only the startup LOS table is needed by the controlled experiment. Each validation pair used its exact frozen initial LLH, UTC, the pinned RINEX NAV, and the pinned gps-sdr-sim executable. A one-second 1 MHz startup run produced the table and a retained 1.8 MB IQ provenance file.

Before validation access, the same shortcut was checked on dynamic train pairs 003–006 against their original 30-second, 25 MHz trajectory runs. The PRN rosters matched exactly and the maximum absolute LOS-vector component difference was 0.0. Validation startup constellations contained 11, 15, and 13 PRNs and all geometry matrices had rank four.

## Data and claim boundary

- Validation pairs 007–009 were accessed once.
- Test pairs 010–012 remain locked and TEXBAT was not accessed.
- The matched “multipath” class remains an exact PRN-identity derangement of the complete observed profiles. It is not an RF-generated or ray-traced multipath channel.
- The complex nine-tap observation has not yet passed through GNSS-SDR.
- This result confirms generalization across unseen constellations under the controlled model; it does not measure field false-alarm rate or establish a deployable threshold.

## Reproduction and artifacts

The one-shot command was:

```bash
.venv/bin/python scripts/validate_correlator_geometry_identifiability.py
```

The canonical output is `artifacts/correlator_geometry_identifiability_validation_v1/summary.json` with SHA-256 `add092c0d42e7e8cc703fa8c3543f9dec3f69331159a77e451b06b45d5f0eb2f`. The 15 MiB directory includes event scores, per-PRN delay estimates, exact profile assignments, template manifest, startup logs, and retained startup IQ.

A byte-identical SSD bundle is stored at `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/correlator-geometry-identifiability-validation-v1/`.

The next gate is to export complex I/Q for every GNSS-SDR correlator tap and construct satellite-specific RF multipath controls. Test remains locked until that receiver/RF challenge is frozen and completed.
