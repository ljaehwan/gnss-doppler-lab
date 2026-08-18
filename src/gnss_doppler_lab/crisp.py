"""CRISP Stage-0 projective complex-correlator detector primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc

from .crisp_data import TAPS, complex_taps


def projector_vector(taps: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Vectorize the Hermitian rank-one normalized projector in R^(m^2)."""
    taps = np.asarray(taps, dtype=np.complex128)
    if taps.ndim == 1:
        taps = taps[None, :]
    energy = np.sum(np.abs(taps) ** 2, axis=1)
    unit = taps / np.sqrt(energy[:, None] + epsilon)
    columns: list[np.ndarray] = [np.abs(unit[:, k]) ** 2 for k in range(unit.shape[1])]
    for k in range(unit.shape[1]):
        for ell in range(k + 1, unit.shape[1]):
            value = unit[:, k] * np.conj(unit[:, ell])
            columns.extend((value.real, value.imag))
    return np.column_stack(columns)


def projector_matrix(taps: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    taps = np.asarray(taps, dtype=np.complex128)
    energy = float(np.vdot(taps, taps).real)
    return np.outer(taps, np.conj(taps)) / (energy + epsilon)


def wedge_vector(current: np.ndarray, previous: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    current = np.asarray(current, dtype=np.complex128)
    previous = np.asarray(previous, dtype=np.complex128)
    current = current / np.sqrt(np.sum(np.abs(current) ** 2, axis=1)[:, None] + epsilon)
    previous = previous / np.sqrt(np.sum(np.abs(previous) ** 2, axis=1)[:, None] + epsilon)
    values: list[np.ndarray] = []
    for k in range(current.shape[1]):
        for ell in range(k + 1, current.shape[1]):
            value = current[:, k] * previous[:, ell] - current[:, ell] * previous[:, k]
            values.extend((value.real, value.imag))
    return np.column_stack(values)


def wrapped_phase(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class ChannelFeatures:
    epoch_ms: np.ndarray
    timestamp_s: np.ndarray
    prn: np.ndarray
    response: np.ndarray
    context: np.ndarray
    valid: np.ndarray
    reset: np.ndarray
    baselines: dict[str, np.ndarray]


def extract_channel_features(
    records: np.ndarray,
    *,
    epsilon: float,
    energy_floor: float,
    cn0_min_db_hz: float,
    lock_min: float,
    tap_indices: tuple[int, ...] | None = None,
) -> ChannelFeatures:
    """Construct strictly causal t-1 -> t features with explicit reset masks."""
    taps = complex_taps(records, tap_indices)
    energy = np.sum(np.abs(taps) ** 2, axis=1)
    pvec = projector_vector(taps, epsilon)
    response = pvec[1:] - pvec[:-1]
    same = (
        (records["tracking_session_id"][1:] == records["tracking_session_id"][:-1])
        & (records["prn"][1:] == records["prn"][:-1])
        & (records["loop_sequence"][1:] == records["loop_sequence"][:-1] + 1)
    )
    dt = records["receiver_timestamp_s"][1:] - records["receiver_timestamp_s"][:-1]
    native_dt = (dt >= 0.0009) & (dt <= 0.0011)
    reset = ~(same & native_dt)
    quality = (
        same
        & native_dt
        & (records["valid_tracking"][1:] == 1)
        & (records["valid_tracking"][:-1] == 1)
        & (records["valid_lock"][1:] == 1)
        & (records["valid_lock"][:-1] == 1)
        & (records["pull_in_transitory"][1:] == 0)
        & (records["cn0_db_hz"][1:] >= cn0_min_db_hz)
        & (records["cn0_db_hz"][:-1] >= cn0_min_db_hz)
        & (records["carrier_lock_test"][1:] >= lock_min)
        & (records["carrier_lock_test"][:-1] >= lock_min)
        & (energy[1:] >= energy_floor)
        & (energy[:-1] >= energy_floor)
    )
    previous_velocity = np.zeros(len(response), dtype=np.float64)
    if len(response) > 1:
        previous_velocity[1:] = np.linalg.norm(response[:-1], axis=1)
    context = np.column_stack(
        [
            records["dll_discriminator_chips"][1:] - records["dll_discriminator_chips"][:-1],
            records["code_filter_output_chips_s"][1:] - records["code_filter_output_chips_s"][:-1],
            records["carrier_filter_output_hz"][1:] - records["carrier_filter_output_hz"][:-1],
            records["cn0_db_hz"][:-1],
            records["carrier_lock_test"][1:],
            previous_velocity,
        ]
    )
    prompt_index = (tap_indices.index(4) if tap_indices is not None and 4 in tap_indices else 1)
    if tap_indices is None:
        prompt_index = 4
    magnitude = np.abs(taps) / np.sqrt(energy[:, None] + epsilon)
    wedge = wedge_vector(taps[1:], taps[:-1], epsilon)
    baselines = {
        "A0": np.abs(np.log(energy[1:] + epsilon) - np.log(energy[:-1] + epsilon)),
        "A1": np.linalg.norm(magnitude[1:] - magnitude[:-1], axis=1),
        "A2": np.abs(wrapped_phase(np.angle(taps[1:, prompt_index] * np.conj(taps[:-1, prompt_index])))),
        "A3": np.linalg.norm(response, axis=1),
        "A6": np.linalg.norm(wedge, axis=1),
    }
    timestamp = records["receiver_timestamp_s"][1:].astype(np.float64)
    return ChannelFeatures(
        epoch_ms=np.rint(timestamp * 1000.0).astype(np.int64),
        timestamp_s=timestamp,
        prn=records["prn"][1:].astype(np.int64),
        response=response,
        context=context,
        valid=quality & np.isfinite(context).all(axis=1) & np.isfinite(response).all(axis=1),
        reset=reset,
        baselines=baselines,
    )


@dataclass
class LinearWhiteningModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficient: np.ndarray
    residual_mean: np.ndarray
    covariance: np.ndarray
    inverse_sqrt: np.ndarray
    shrinkage: float
    ridge_alpha: float

    @classmethod
    def fit(
        cls,
        context: np.ndarray,
        response: np.ndarray,
        calibration_context: np.ndarray,
        calibration_response: np.ndarray,
        *,
        ridge_alpha: float,
    ) -> "LinearWhiteningModel":
        feature_mean = np.mean(context, axis=0)
        feature_scale = np.std(context, axis=0)
        feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
        x = (context - feature_mean) / feature_scale
        xa = np.column_stack((np.ones(len(x)), x))
        penalty = np.eye(xa.shape[1]) * ridge_alpha
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(xa.T @ xa + penalty, xa.T @ response)
        cx = (calibration_context - feature_mean) / feature_scale
        cxa = np.column_stack((np.ones(len(cx)), cx))
        residual = calibration_response - cxa @ coefficient
        estimator = LedoitWolf(assume_centered=False).fit(residual)
        covariance = estimator.covariance_
        residual_mean = estimator.location_
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        floor = max(float(np.max(eigenvalue)) * 1e-10, 1e-12)
        eigenvalue = np.maximum(eigenvalue, floor)
        inverse_sqrt = (eigenvector * (1.0 / np.sqrt(eigenvalue))) @ eigenvector.T
        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            coefficient=coefficient,
            residual_mean=residual_mean,
            covariance=covariance,
            inverse_sqrt=inverse_sqrt,
            shrinkage=float(estimator.shrinkage_),
            ridge_alpha=ridge_alpha,
        )

    def score(self, context: np.ndarray, response: np.ndarray) -> np.ndarray:
        x = (context - self.feature_mean) / self.feature_scale
        predicted = np.column_stack((np.ones(len(x)), x)) @ self.coefficient
        innovation = response - predicted - self.residual_mean
        whitened = innovation @ self.inverse_sqrt.T
        return np.mean(whitened * whitened, axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "residual_mean": self.residual_mean.tolist(),
            "covariance": self.covariance.tolist(),
            "inverse_sqrt": self.inverse_sqrt.tolist(),
            "shrinkage": self.shrinkage,
            "ridge_alpha": self.ridge_alpha,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LinearWhiteningModel":
        return cls(
            feature_mean=np.asarray(value["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(value["feature_scale"], dtype=np.float64),
            coefficient=np.asarray(value["coefficient"], dtype=np.float64),
            residual_mean=np.asarray(value["residual_mean"], dtype=np.float64),
            covariance=np.asarray(value["covariance"], dtype=np.float64),
            inverse_sqrt=np.asarray(value["inverse_sqrt"], dtype=np.float64),
            shrinkage=float(value["shrinkage"]),
            ridge_alpha=float(value["ridge_alpha"]),
        )


def fit_unconditioned(response: np.ndarray) -> LinearWhiteningModel:
    context = np.empty((len(response), 0), dtype=np.float64)
    return LinearWhiteningModel.fit(context, response, context, response, ridge_alpha=0.0)


def normalized_low_fpr_pauc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.05) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    if fpr[-1] < max_fpr:
        return float("nan")
    index = np.searchsorted(fpr, max_fpr, side="right")
    xf = np.r_[fpr[:index], max_fpr]
    yf = np.r_[tpr[:index], np.interp(max_fpr, fpr, tpr)]
    return float(auc(xf, yf) / max_fpr)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"roc_auc": float("nan"), "pauc_fpr_le_0_05": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pauc_fpr_le_0_05": normalized_low_fpr_pauc(labels, scores),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def invariance_audit(seed: int = 20260818, tolerance: float = 1e-10) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    taps = rng.normal(size=(128, 9)) + 1j * rng.normal(size=(128, 9))
    reference = projector_vector(taps)
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, len(taps)))
    sign = rng.choice(np.array([-1.0, 1.0]), len(taps))
    ramp = np.exp(1j * 2.0 * np.pi * 37.0 * np.arange(len(taps)) * 0.001)
    arbitrary = rng.uniform(0.25, 3.0, len(taps)) * np.exp(1j * rng.uniform(-np.pi, np.pi, len(taps)))
    transforms = {
        "gain_0_5": taps * 0.5,
        "gain_1": taps.copy(),
        "gain_2": taps * 2.0,
        "random_global_phase": taps * phase[:, None],
        "nav_bit_sign": taps * sign[:, None],
        "common_doppler_phase_ramp": taps * ramp[:, None],
        "prompt_amplitude_only_scaling": taps * 1.7,
        "arbitrary_nonzero_complex_scalar": taps * arbitrary[:, None],
    }
    rows = {}
    for name, transformed in transforms.items():
        error = float(np.max(np.abs(projector_vector(transformed) - reference)))
        rows[name] = {"max_abs_error": error, "tolerance": tolerance, "pass": error <= tolerance}
    return {"seed": seed, "tests": rows, "all_pass": all(row["pass"] for row in rows.values())}


def projector_property_audit(seed: int = 20260818) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    taps = rng.normal(size=9) + 1j * rng.normal(size=9)
    matrix = projector_matrix(taps)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return {
        "hermitian_max_error": float(np.max(np.abs(matrix - matrix.conj().T))),
        "idempotence_max_error": float(np.max(np.abs(matrix @ matrix - matrix))),
        "largest_eigenvalue": float(eigenvalues[-1]),
        "second_largest_abs_eigenvalue": float(np.sort(np.abs(eigenvalues))[-2]),
        "pass": bool(
            np.max(np.abs(matrix - matrix.conj().T)) < 1e-12
            and np.max(np.abs(matrix @ matrix - matrix)) < 1e-10
            and np.sort(np.abs(eigenvalues))[-2] < 1e-10
        ),
    }
