# ACAF-NF Stage-0 R1.3 reconstruction

R1.2 was scientifically invalid because it used a non-canonical C/A replica, ignored candidate dimensions, copied global-offset metrics, mislabeled A2, and omitted stable-lock filtering. R1.3 imports the canonical generator, applies aux/remnant/carrier inputs, separates variable consumed intervals from source-authenticated correlator support, and recomputes every offset on cleanStatic only.

- Canonical C/A: 32/32
- PRNs / epochs: 8 / 969
- Blocks: train=323, calibration=323, holdout=323
- Best diagnostic candidate: nco_row=previous_aux_row=previous_remnant_sign=-1_carrier_sign=-1_global_offset=0
- Within tolerance: 0.856553148
- Pooled Prompt Spearman: 0.999996505
- Median PRN Prompt Spearman: 0.999965275
- Wide-grid boundary: 0.006191950
- A1/A2/A3: PASS / PASS / FAIL
- Selected alignment: None

Failure of A3 is tracker/raw reconstruction or alignment unresolved, not an ACAF physical-model NO-GO. `physics_no_go_claim` is always false in this audit.
