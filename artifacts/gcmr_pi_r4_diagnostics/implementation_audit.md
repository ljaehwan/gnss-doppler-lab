# Implementation audit

- E/P/L normalization uses center P (`safe_center_tap_normalize`); the primary E/P/L is not separately protected from the 9-tap runner contract.
- A 3D Gram matrix has rank <=3, so high S_common can be geometric/low-rank rather than spoof-specific.
- Full adds standardized components and ignores covariance; S_pair models conditional expected cosine but not conditional variance.
- `clean_reference` is built but not consumed by `fit_normal`; `selection_val` is consumed for predictor/whitener/pair calibration, not model selection or early stopping.
- A 9-tap runner exists, but this frozen artifact is 3T; no frozen 9T result exists.

Next fixes are documentation only: use covariance-aware calibration, conditional pair variance, explicit clean-reference role, held-out early stopping/model selection, and freeze/run 9T before claims.
