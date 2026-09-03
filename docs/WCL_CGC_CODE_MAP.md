# WCL CGC claim-to-code and experiment map

## Purpose and scope

This document is the human-readable source of truth for the current WCL CGC
manuscript. The machine-readable companion is
`configs/paper/wcl_cgc_v1_manifest.json`.

The paper does **not** claim a universal standalone spoofing detector. CGC is a
second-stage cause classifier for a locally observable two-component pull-off
interval:

```text
complex 9-tap profile per PRN
  -> signed secondary-delay estimate
  -> causal five-bin per-PRN median
  -> delay-spread observability gate
  -> clock-only versus LOS-plus-clock nested fit
  -> support-normalized Partial-F ranking score
  -> three-of-five persistent carry-off-consistent decision
```

The final detector configuration is the frozen temporal final-static release.
Public JammerTest, TEXBAT, and GNSS-OpenIF results are separate transfer audits;
they must not be pooled into the final simulated receiver--RF endpoint.

## Direct manuscript claims

| Claim ID | What the paper uses | Entrypoint and frozen config | Compact result | Role |
|---|---|---|---|---|
| `C1_COMPLEX_SIGNED_DELAY` | Complex taps retain signed-delay/PRN--LOS information; matched-profile AUCs | `scripts/audit_correlator_geometry_identifiability.py`, `scripts/validate_correlator_geometry_identifiability.py`; the two `correlator_geometry_identifiability_*_v1.json` configs | `correlator_geometry_identifiability_*_v1.md` and summaries | Controlled mechanism evidence |
| `C2_CLOCK_CENTERED_CGC` | One displacement plus clock explains coordinated delays; common delay alone is rejected as position evidence | `src/gnss_doppler_lab/correlator_geometry.py`, `clock_centered_geometry.py`, `static_reference_geometry.py` | Unit tests plus the fresh receiver--RF result | Core method |
| `C3_FRESH_RF_MECHANISM` | Three held-out receiver--RF pairs; pair-block AUC 1.000 and clock-centering comparison | `scripts/run_cgc_rf_fresh_test_v2.py`; `cgc_rf_fresh_test_v2.json` | `CGC_RF_FRESH_TEST_V2_RESULT.md` | Controlled receiver--RF mechanism evidence; also supplies Fig. 1 data |
| `C4_FINAL_STATIC` | Five unseen static geometries, four conditions, frozen temporal rule, Table I, AUC 0.9801 | `scripts/preflight_cgc_temporal_final_static.py`, `scripts/run_cgc_temporal_final_static_v1.py`; the final-static pool and release configs | `CGC_TEMPORAL_FINAL_STATIC_V1.md` and summary | Primary outcome-unseen performance evidence |
| `C5_FORMULA_ACCURACY` | Simulator-truth accuracy of range linearization, signed delay, sign, and fitted direction | `scripts/audit_cgc_formula_accuracy_v1.py`; consumes the unchanged C4 streams | `CGC_FORMULA_ACCURACY_AUDIT_V1.md` and summary | Same-stream validation, not a new independent test |
| `C6_TAP_APERTURE` | 3/5/7/9-tap aperture at 100 m and 240 m, including saturation and one ranking reversal | `scripts/run_cgc_rf_geometry_aperture_validation.py`, `scripts/audit_cgc_rf_7tap_aperture.py`; geometry-aperture config | `CGC_RF_GEOMETRY_APERTURE_VALIDATION_V1_RESULT.md` and 7-tap summary | Supporting ablation with explicit failed preregistered gates |
| `C7_PUBLIC_TRANSFER` | JammerTest carry-off-consistent latency, TEXBAT direction/latency, GNSS-OpenIF S1/S2 multipath false alarms | Dataset-specific runners and configs listed in the JSON manifest | Dataset result reports and compact summaries | Separate public transfer/negative-control evidence |
| `C8_FINAL_FIGURES` | Principle and evidence figures reproduced from frozen/archived artifacts | `scripts/plot_wcl_cgc_figures.py` | `docs/results/figures/wcl/wcl_cgc_figure_manifest.json` | Manuscript vector figures |

## Core implementation ownership

| File | Responsibility |
|---|---|
| `src/gnss_doppler_lab/correlator_geometry.py` | Two-path complex profile, prompt-phase alignment, deterministic template banks, signed-delay estimation, full geometry fit |
| `src/gnss_doppler_lab/clock_centered_geometry.py` | Clock-only nuisance centering and LOS directional coherence |
| `src/gnss_doppler_lab/static_reference_geometry.py` | Finite-support Partial-F conversion and generic persistence utilities |
| `src/gnss_doppler_lab/temporal_cgc.py` | Causal, wall-clock five-bin per-PRN median |
| `src/gnss_doppler_lab/code_carrier_sim.py` | Coupled and authentic-Doppler-locked code/carrier experiment construction |
| `src/gnss_doppler_lab/satellite_multipath.py` | Independent PRN-specific multipath generation |
| `src/gnss_doppler_lab/peak_mixture_law.py` | Simulator LOS/range truth parsing |
| `src/gnss_doppler_lab/tracking_peaks.py` | Receiver multi-tap dump extraction |
| `src/gnss_doppler_lab/gcmr_geometry.py`, `rinex_nav.py` | Public-record ephemeris/LOS adapters |

## Evidence boundaries that must remain visible

- `CGC_TEMPORAL_FINAL_STATIC_V1` is the final paper endpoint. Its five
  geometries were selected by startup LOS support before outcome access.
- The exact-lock phase sweep and temporal stabilization audits selected the
  five-bin median and `0.10`-chip observability gate. Their scores are
  development evidence, not independent performance estimates.
- The geometry/aperture campaign did **not** pass every preregistered aggregate
  gate. It remains useful because it demonstrates the metric band, finite
  aperture saturation, and a Tokyo/+3 dB blind spot.
- One Casablanca independent-multipath pair persistently alarms in the final
  test. It must not be tuned away.
- JammerTest is one public coherent-spoof transfer record. TEXBAT has spoof
  pre/post labels but no independent-multipath class. GNSS-OpenIF S1/S2 are
  multipath-rich negative controls, not paired spoof-versus-multipath truth.
- FGI-SpoofRepo TGD, normal-detector freeze, and input-support failures are
  retained as negative results. They narrow the claim and are not evidence for
  a positive detector rate.

## Development and superseded lines

The following families influenced the research but are not the final endpoint:

- code/carrier decoupling pilot, phase sweep, root-cause audit, temporal
  stabilization development, and Doppler-silent hold audits;
- earlier locked/fresh/challenge/transfer/observability CGC campaigns;
- static-reference/OAKBAT CGC, GCMR, and other alternative observables;
- B0 normal-only GRU/binomial-tail detector, raw-IQ continuity, and earlier
  Doppler-only work.

The exact classification and replacement links are in the JSON manifest. Do
not copy thresholds, AUCs, or data splits from these families into the final
CGC result without explicitly changing the paper claim.

## Formula-audit figure policy

`docs/results/figures/cgc_formula_accuracy_audit_v1.{pdf,png}` is deliberately
kept as a **developer diagnostic**. The numerical audit is summarized in the
paper, but this figure is not copied into the paper repository and is not one
of the two WCL manuscript figures.

## Validation

Fast provenance and path check:

```bash
python scripts/audit_wcl_cgc_manifest.py
```

Focused core regression suite:

```bash
pytest -q \
  tests/test_correlator_geometry.py \
  tests/test_clock_centered_geometry.py \
  tests/test_static_reference_geometry.py \
  tests/test_temporal_cgc.py \
  tests/test_code_carrier_sim.py \
  tests/test_satellite_multipath.py \
  tests/test_tracking_peaks.py
```

The full final-static RF campaign is intentionally not a routine unit-test
command. Its retained artifact root and SHA-256 are recorded in
`CGC_TEMPORAL_FINAL_STATIC_V1.md` and its compact JSON summary.
