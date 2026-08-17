"""Preregistered MOSAIC Stage-0B R1 physical recovery metrics."""
from __future__ import annotations

import math
import numpy as np


def caf_grids() -> tuple[np.ndarray, np.ndarray]:
    return np.round(np.arange(-0.35, 0.350001, 0.025), 12), np.arange(-75.0, 75.0001, 5.0)


def bic(rss: float, observations: int, parameters: int) -> float:
    if rss <= 0 or observations <= parameters or parameters < 0:
        raise ValueError("invalid BIC inputs")
    return float(observations * math.log(rss / observations) + parameters * math.log(observations))


def delta_bic_h1_support(rss_h0: float, rss_h1: float, observations: int, *, h0_parameters: int, h1_parameters: int) -> float:
    """Positive values support H1 after its larger complexity penalty."""
    return bic(rss_h0, observations, h0_parameters) - bic(rss_h1, observations, h1_parameters)


def classify_case(delta_tau_chips: float, delta_f_hz: float) -> str:
    return "COLLAPSED_SINGLE_SOURCE_CONTROL" if delta_tau_chips == 0 and delta_f_hz == 0 else "IDENTIFIABLE_SECOND_SOURCE"


def strong_resolvable(rho_db: float, delta_tau_chips: float, delta_f_hz: float) -> bool:
    return bool(rho_db >= -6 and (abs(delta_tau_chips) >= 0.10 or abs(delta_f_hz) >= 25))


def sign_accuracy(truth: np.ndarray, estimate: np.ndarray) -> float | None:
    a, b = np.asarray(truth, float), np.asarray(estimate, float)
    mask = (a != 0) & np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.sign(a[mask]) == np.sign(b[mask]))) if mask.any() else None


def median_absolute_error(truth: np.ndarray, estimate: np.ndarray) -> float | None:
    a, b = np.asarray(truth, float), np.asarray(estimate, float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.median(np.abs(a[mask] - b[mask]))) if mask.any() else None


GO_THRESHOLDS = {
    "realized_scer_median_error_db_max": 1.0,
    "observability_each_dataset_min": 0.75,
    "delay_sign_accuracy_min": 0.80,
    "delay_median_absolute_error_chips_max": 0.05,
    "doppler_sign_accuracy_min": 0.80,
    "doppler_median_absolute_error_hz_max": 10.0,
    "four_prn_three_of_four_recovery_fraction_min": 0.75,
}


def preregistered_verdict(metrics: dict[str, object]) -> str:
    receiver_keys = ("identity_receiver_replay", "zero_amplitude_byte_identity", "actual_int16_format")
    if not all(bool(metrics.get(key)) for key in receiver_keys):
        return "INCONCLUSIVE_RECEIVER_IN_LOOP"
    numeric = (
        float(metrics["realized_scer_median_error_db"]) <= 1.0,
        float(metrics["oak_observability"]) >= .75,
        float(metrics["tex_observability"]) >= .75,
        float(metrics["delay_sign_accuracy"]) >= .80,
        float(metrics["delay_median_absolute_error_chips"]) <= .05,
        float(metrics["doppler_sign_accuracy"]) >= .80,
        float(metrics["doppler_median_absolute_error_hz"]) <= 10,
        float(metrics["four_prn_three_of_four_fraction"]) >= .75,
    )
    qualitative = ("bic_control_separation", "target_over_nontarget", "not_total_iq_rms_shortcut", "prn_permutation_invariance")
    return "GO_FOR_MOSAIC_NEURAL_STAGE1" if all(numeric) and all(bool(metrics.get(k)) for k in qualitative) else "NO_GO_MOSAIC_INJECTOR_PHYSICS"
