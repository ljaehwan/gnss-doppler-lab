# Equation-to-code map

| Paper quantity | Mathematical role | Canonical implementation | Regression test |
|---|---|---|---|
| `z_{i,k}` | Authentic plus delayed complex replica at tap `k` | [`two_path_complex_profile`](../../../src/gnss_doppler_lab/correlator_geometry.py) | [`test_correlator_geometry.py`](../../../tests/test_correlator_geometry.py) |
| prompt-phase-aligned taps | Preserve I/Q and remove arbitrary common prompt phase | [`complex_profile_features`](../../../src/gnss_doppler_lab/correlator_geometry.py) | `test_correlator_geometry.py` |
| `hat(delta_i)` | Nearest deterministic two-path template signed delay | [`build_complex_template_bank`, `TemplateDelayEstimator`](../../../src/gnss_doppler_lab/correlator_geometry.py) | `test_correlator_geometry.py` |
| `hat(delta_i)=-u_i^T d+c+e_i` | Common displacement-plus-clock fit | [`fit_common_geometry`](../../../src/gnss_doppler_lab/correlator_geometry.py) | `test_correlator_geometry.py` |
| `SSE_0`, `SSE_1`, `r_G` | Clock-only null versus LOS-plus-clock alternative | [`fit_clock_centered_geometry`](../../../src/gnss_doppler_lab/clock_centered_geometry.py) | [`test_clock_centered_geometry.py`](../../../tests/test_clock_centered_geometry.py) |
| `F`, `p_F` | Correct for the three added displacement parameters and satellite support | [`partial_f_score`](../../../src/gnss_doppler_lab/static_reference_geometry.py) | [`test_static_reference_geometry.py`](../../../tests/test_static_reference_geometry.py) |
| `tilde(delta_{i,t})` | Causal five-bin per-PRN median | [`causal_prn_median`](../../../src/gnss_doppler_lab/temporal_cgc.py) | [`test_temporal_cgc.py`](../../../tests/test_temporal_cgc.py) |
| `A_delta`, `G_t`, `P_t` | Observability, raw decision, and three-of-five persistence | [`run_cgc_temporal_final_static_v1.py`](../../../scripts/run_cgc_temporal_final_static_v1.py) | final protocol and truth audit |

## Parameter source

The final rule is fixed in
[`cgc_temporal_final_static_v1.json`](../../../configs/experiments/cgc_temporal_final_static_v1.json):

- nine complex taps at `0.125` chip spacing;
- causal median window `W=5`;
- observability threshold `gamma=0.10` chip;
- minimum support `N=8` PRNs;
- Partial-F ranking threshold `p_F <= 0.06028418845288192`;
- persistence: at least three alarms in the latest five one-second bins.

The median window and observability threshold were selected on reused
development data. The final five-geometry experiment did not refit them.

## Accuracy audit

[`audit_cgc_formula_accuracy_v1.py`](../../../scripts/audit_cgc_formula_accuracy_v1.py)
compares the implementation with simulator truth without changing the detector:

- exact simulator range versus fixed-startup LOS linearization;
- raw and five-bin stabilized signed-delay error;
- sign accuracy for observable delays;
- recovered three-dimensional direction and displacement norm.

This establishes internal equation-to-simulator consistency. It does not prove
navigation-grade ranging accuracy, Gaussian correlator errors, or universal
field performance. The Partial-F tail remains a support-normalized ranking
score rather than an exact false-alarm probability.

## Focused validation

```bash
.venv/bin/pytest -q \
  tests/test_correlator_geometry.py \
  tests/test_clock_centered_geometry.py \
  tests/test_static_reference_geometry.py \
  tests/test_temporal_cgc.py
```
