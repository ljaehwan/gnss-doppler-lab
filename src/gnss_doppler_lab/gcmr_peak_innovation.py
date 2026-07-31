"""Pure-NumPy mathematical primitives for GCMR peak innovation detection.

All ``fit`` methods accept normal validation data only; this module deliberately
contains no receiver/data-adapter code and no attack-label optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np

_EPS = 1e-12


def safe_prompt_normalize(EPL: np.ndarray, eps: float = 1e-12,
                          min_prompt: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Normalize E/P/L vectors by prompt safely, returning (values, validity)."""
    x = np.asarray(EPL, dtype=float)
    if x.shape[-1:] != (3,):
        raise ValueError("EPL must have final dimension 3")
    if eps <= 0 or min_prompt < 0:
        raise ValueError("eps must be positive and min_prompt non-negative")
    valid = np.isfinite(x).all(axis=-1) & (x[..., 1] > min_prompt)
    out = np.zeros_like(x, dtype=float)
    denominator = np.maximum(x[..., 1], eps)
    np.divide(x, denominator[..., None], out=out, where=valid[..., None])
    return out, valid


class SharedLocalPredictor:
    """Shared PRN-local next-window predictor adapter.

    ``callback`` receives only ``(window, prn, 3)`` history, so it cannot rely
    on PRN identifiers and naturally supports any number of visible PRNs.
    """
    def __init__(self, callback: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.callback = callback or (lambda history: history[-1])

    def predict(self, history: np.ndarray) -> np.ndarray:
        h = np.asarray(history, dtype=float)
        if h.ndim != 3 or h.shape[-1] != 3 or h.shape[0] < 1:
            raise ValueError("history must have shape (window, N, 3)")
        result = np.asarray(self.callback(h), dtype=float)
        if result.shape != h.shape[1:]:
            raise ValueError("predictor callback must return shape (N, 3)")
        if not np.isfinite(result).all():
            raise ValueError("predictor output must be finite")
        return result


def _validate_innovations(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim not in (2, 3) or x.shape[-1:] != (3,) or x.size == 0:
        raise ValueError("innovations must be non-empty (..., 3) arrays")
    if not np.isfinite(x).all():
        raise ValueError("innovations must be finite")
    return x


class ConditionalInnovationWhitener:
    """Deterministic context-binned, shrinkage covariance whitener."""
    def __init__(self, context_edges=(35.0, 45.0), min_bin_samples: int = 20,
                 regularization: float = 1e-5, eps: float = _EPS):
        self.context_edges = np.asarray(context_edges, dtype=float)
        self.min_bin_samples, self.regularization, self.eps = min_bin_samples, regularization, eps
        if min_bin_samples < 1 or regularization <= 0 or eps <= 0:
            raise ValueError("invalid whitener configuration")
        self.dimension = 3

    def _parameters(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        centered = values - mean
        covariance = centered.T @ centered / max(len(values) - 1, 1)
        diagonal = np.diag(np.diag(covariance))
        covariance = (1.0 - self.regularization) * covariance + self.regularization * diagonal
        covariance += np.eye(3) * self.eps
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, self.eps))) @ eigenvectors.T
        return mean, inv_sqrt

    def fit(self, normal_innovations: np.ndarray, context: Optional[np.ndarray] = None):
        x = _validate_innovations(normal_innovations)
        flat = x.reshape(-1, 3)
        self.global_mean_, self.global_inv_sqrt_ = self._parameters(flat)
        self.bin_parameters_ = {}
        if context is not None:
            c = np.asarray(context, dtype=float)
            if c.shape != x.shape[:-1] or not np.isfinite(c).all():
                raise ValueError("context must be finite and match innovation leading dimensions")
            for bin_number in range(len(self.context_edges) + 1):
                values = flat[np.digitize(c.reshape(-1), self.context_edges) == bin_number]
                if len(values) >= self.min_bin_samples:
                    self.bin_parameters_[bin_number] = self._parameters(values)
        self.fitted_ = True
        return self

    def transform(self, innovations: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("fit on normal validation innovations first")
        x = _validate_innovations(innovations)
        if context is not None:
            c = np.asarray(context, dtype=float)
            if c.shape != x.shape[:-1] or not np.isfinite(c).all():
                raise ValueError("context must be finite and match innovation leading dimensions")
            bins = np.digitize(c.reshape(-1), self.context_edges)
        else:
            bins = np.full(x.size // 3, -1)
        flat, output = x.reshape(-1, 3), np.empty_like(x.reshape(-1, 3))
        for i, value in enumerate(flat):
            mean, matrix = self.bin_parameters_.get(int(bins[i]), (self.global_mean_, self.global_inv_sqrt_))
            output[i] = matrix @ (value - mean)
        return output.reshape(x.shape)


def _validated_geometry(los: np.ndarray, elevation: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate and unit-normalize LOS geometry for an N-node event."""
    vectors = np.asarray(los, float)
    elevations = np.asarray(elevation, float)
    if vectors.shape != (n, 3) or elevations.shape != (n,):
        raise ValueError("LOS must have shape (N, 3) and elevation shape (N,)")
    norms = np.linalg.norm(vectors, axis=1)
    if not (np.isfinite(vectors).all() and np.isfinite(elevations).all() and np.all(norms > _EPS)):
        raise ValueError("LOS/elevation must be finite and LOS vectors non-zero")
    return vectors / norms[:, None], elevations


def geometry_features(los_i: np.ndarray, elevation_i: float,
                      los_j: np.ndarray, elevation_j: float) -> np.ndarray:
    """Symmetric pair geometry: normalized LOS dot product and sorted elevations."""
    a, b = np.asarray(los_i, float), np.asarray(los_j, float)
    if a.shape != (3,) or b.shape != (3,) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("LOS vectors must be finite 3-vectors")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= _EPS or nb <= _EPS or not np.isfinite([elevation_i, elevation_j]).all():
        raise ValueError("LOS must be non-zero and elevations finite")
    return np.array([np.dot(a / na, b / nb), min(elevation_i, elevation_j), max(elevation_i, elevation_j)], float)


def _cosine_matrix(z: np.ndarray, eps: float = _EPS) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(z, axis=1), eps)
    return (z @ z.T) / np.outer(norms, norms)


class PairRelationModel:
    """Symmetric ridge regression of normal-data pair cosines on geometry."""
    def __init__(self, ridge: float = 1e-3):
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        self.ridge = ridge

    def fit_events(self, normal_events) -> "PairRelationModel":
        """Fit from variable-cardinality normal `(z, los, elevation)` events only."""
        features, targets = [], []
        for z_event, los_event, elevation_event in normal_events:
            z = _validate_innovations(z_event)
            if z.ndim != 2 or len(z) < 2:
                raise ValueError("each normal event must have shape (N, 3), N >= 2")
            los, elevation = _validated_geometry(los_event, elevation_event, len(z))
            cosine = _cosine_matrix(z)
            for i in range(len(z)):
                for j in range(i + 1, len(z)):
                    features.append(geometry_features(los[i], elevation[i], los[j], elevation[j]))
                    targets.append(cosine[i, j])
        if not features:
            raise ValueError("normal pair relation fit requires at least one pair")
        X = np.column_stack([np.ones(len(features)), np.asarray(features)])
        self.coef_ = np.linalg.solve(X.T @ X + self.ridge * np.eye(X.shape[1]), X.T @ np.asarray(targets))
        return self

    def fit(self, normal_z: np.ndarray, *, los: np.ndarray, elevation: np.ndarray):
        """Fixed-N compatibility wrapper; prefer :meth:`fit_events` for real data."""
        z = _validate_innovations(normal_z)
        if z.ndim != 3:
            raise ValueError("normal_z must have shape (samples, N, 3)")
        vectors, elevations = _validated_geometry(los, elevation, z.shape[1])
        return self.fit_events([(z[index], vectors, elevations) for index in range(z.shape[0])])

    def expected_matrix(self, los: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit on normal data first")
        vectors, elevations = _validated_geometry(los, elevation, len(elevation))
        n, output = len(elevations), np.eye(len(elevations))
        for i in range(n):
            for j in range(i + 1, n):
                feature = np.r_[1., geometry_features(vectors[i], elevations[i], vectors[j], elevations[j])]
                output[i, j] = output[j, i] = np.clip(feature @ self.coef_, -1., 1.)
        return output


def pair_anomaly_score(z: np.ndarray, relation_model: PairRelationModel,
                       los: np.ndarray, elevation: np.ndarray) -> float:
    z = _validate_innovations(z)
    if z.ndim != 2 or len(z) < 2:
        raise ValueError("z must have shape (N, 3), N >= 2")
    delta = np.abs(_cosine_matrix(z) - relation_model.expected_matrix(los, elevation))
    return float(delta[np.triu_indices(len(z), 1)].mean())


@dataclass(frozen=True)
class CommonDriveStatistics:
    n: int
    n_eff: float
    loading_count: int
    at_least_four: bool
    s_common: float


def normal_loading_threshold(normal_z_events, quantile: float = .99) -> float:
    """Fit an absolute loading diagnostic threshold from normal events only."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie in (0, 1)")
    loadings = []
    for z_event in normal_z_events:
        z = _validate_innovations(z_event)
        if z.ndim != 2 or len(z) < 2:
            raise ValueError("each normal event must have shape (N, 3), N >= 2")
        u, _, _ = np.linalg.svd(z, full_matrices=False)
        loadings.extend(np.abs(u[:, 0]).tolist())
    if not loadings:
        raise ValueError("normal loading calibration requires events")
    return float(np.quantile(np.asarray(loadings), quantile))


def common_drive_statistics(Z: np.ndarray, loading_threshold: float | None = None,
                            eps: float = _EPS) -> CommonDriveStatistics:
    z = _validate_innovations(Z)
    if z.ndim != 2 or len(z) < 2:
        raise ValueError("Z must have shape (N, 3), N >= 2")
    gram = z @ z.T
    eigenvalues = np.linalg.eigvalsh(gram)
    s_common = float(max(eigenvalues[-1], 0.) / (np.trace(gram) + eps))
    u, _, _ = np.linalg.svd(z, full_matrices=False)
    loading = np.abs(u[:, 0])
    participation = loading / max(loading.sum(), eps)
    n_eff = float(1. / np.sum(participation ** 2))
    threshold = (0.25 * loading.max()) if loading_threshold is None else float(loading_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("loading_threshold must be a finite non-negative normal-calibrated value")
    loading_count = int(np.count_nonzero(loading >= threshold))
    return CommonDriveStatistics(len(z), n_eff, loading_count, loading_count >= 4, s_common)


def relation_destruction(Z: np.ndarray, seed: int = 0, eps: float = _EPS) -> np.ndarray:
    """Shuffle innovation directions across rows while retaining each row norm."""
    z = _validate_innovations(Z)
    if z.ndim != 2:
        raise ValueError("Z must have shape (N, 3)")
    norms = np.linalg.norm(z, axis=1)
    directions = z / np.maximum(norms[:, None], eps)
    perm = np.random.default_rng(seed).permutation(len(z))
    if len(z) > 1 and np.array_equal(perm, np.arange(len(z))):
        perm = np.roll(perm, 1)
    return directions[perm] * norms[:, None]


def binomial_tail_from_exceedances(exceedances: int, node_count: int, normal_rate: float) -> float:
    """Numerically stable upper binomial tail for the A1 comparison-only baseline.

    The tail can be below the smallest IEEE-754 float for many PRNs at a
    calibration-only low false-positive rate.  Calculate in log space and
    floor only at the smallest positive float, rather than returning zero and
    invalidating the A1 diagnostic.
    """
    if node_count < 1 or not 0 <= exceedances <= node_count or not 0 < normal_rate < 1:
        raise ValueError("invalid binomial-tail inputs")
    from math import lgamma, log, log1p
    terms = np.asarray([
        lgamma(node_count + 1) - lgamma(k + 1) - lgamma(node_count - k + 1)
        + k * log(normal_rate) + (node_count - k) * log1p(-normal_rate)
        for k in range(exceedances, node_count + 1)
    ], dtype=float)
    maximum = float(np.max(terms))
    log_tail = maximum + float(np.log(np.exp(terms - maximum).sum()))
    return float(max(np.exp(log_tail), np.nextafter(0.0, 1.0)))


def build_event_diagnostics(z: np.ndarray, relation_model: PairRelationModel, los: np.ndarray,
                            elevation: np.ndarray, scalar_residuals: np.ndarray,
                            loading_threshold: float | None = None, normal_scalar_threshold: float | None = None,
                            normal_exceedance_rate: float = .01) -> EventDiagnostics:
    """Compute one event’s PI statistics without duplicating relation formulas in runners."""
    values = _validate_innovations(z)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("z must have shape (N, 3), N >= 2")
    residuals = np.asarray(scalar_residuals, float).reshape(-1)
    if residuals.shape != (len(values),) or not np.isfinite(residuals).all():
        raise ValueError("scalar_residuals must be finite shape (N,)")
    stats = common_drive_statistics(values, loading_threshold=loading_threshold)
    if normal_scalar_threshold is None:
        btail = 1.0
    else:
        btail = binomial_tail_from_exceedances(int(np.count_nonzero(residuals > normal_scalar_threshold)),
                                               len(values), normal_exceedance_rate)
    return EventDiagnostics(n=stats.n, n_eff=stats.n_eff, loading_count=stats.loading_count,
                            at_least_four=stats.at_least_four, s_common=stats.s_common,
                            s_pair=pair_anomaly_score(values, relation_model, los, elevation),
                            energy=float(np.mean(np.sum(values * values, axis=1))),
                            scalar_rmse=float(np.sqrt(np.mean(residuals * residuals))), binomial_tail=btail)


@dataclass(frozen=True)
class EventDiagnostics:
    n: int
    n_eff: float
    loading_count: int
    at_least_four: bool
    s_common: float
    s_pair: float
    energy: float
    scalar_rmse: float
    binomial_tail: float = 1.0


class NormalOnlyCalibrator:
    """Stores normal-validation locations, scales, and empirical FPR quantiles."""
    def fit(self, normal_components: Mapping[str, object]):
        if not normal_components:
            raise ValueError("normal components may not be empty")
        self.values_ = {}
        self.location_, self.scale_ = {}, {}
        for name, values in normal_components.items():
            x = np.asarray(values, float).reshape(-1)
            if not len(x) or not np.isfinite(x).all():
                raise ValueError("normal calibration values must be non-empty and finite")
            self.values_[name] = x
            self.location_[name] = float(x.mean())
            self.scale_[name] = float(max(x.std(), _EPS))
        return self

    def _get(self, name: str) -> np.ndarray:
        if not hasattr(self, "values_") or name not in self.values_:
            raise ValueError("component was not calibrated from normal data")
        return self.values_[name]

    def q99(self, name: str) -> float: return float(np.quantile(self._get(name), .99))
    def q995(self, name: str) -> float: return float(np.quantile(self._get(name), .995))
    def target_fpr_threshold(self, name: str, fpr: float = .01) -> float:
        if not 0 < fpr < 1: raise ValueError("fpr must lie in (0, 1)")
        return float(np.quantile(self._get(name), 1. - fpr))
    def standardize(self, name: str, value: float) -> float:
        self._get(name)
        return float((value - self.location_[name]) / self.scale_[name])


def aggregate_event_score(diagnostics: EventDiagnostics, calibrator: NormalOnlyCalibrator,
                          ablation: str = "Full") -> float:
    """Normal-calibrated ablation scores; Full intentionally never uses A1."""
    if ablation == "A0": return calibrator.standardize("scalar_rmse", diagnostics.scalar_rmse)
    if ablation == "A1":
        if not 0 < diagnostics.binomial_tail <= 1: raise ValueError("binomial tail must lie in (0, 1]")
        return float(-np.log(diagnostics.binomial_tail))
    if ablation == "A2": return calibrator.standardize("energy", diagnostics.energy)
    if ablation == "A3":
        return calibrator.standardize("s_common", diagnostics.s_common) + calibrator.standardize("n_eff", diagnostics.n_eff)
    if ablation == "A4": return calibrator.standardize("s_pair", diagnostics.s_pair)
    if ablation == "Full":
        return sum((calibrator.standardize("s_common", diagnostics.s_common),
                    calibrator.standardize("n_eff", diagnostics.n_eff),
                    calibrator.standardize("s_pair", diagnostics.s_pair),
                    calibrator.standardize("energy", diagnostics.energy)))
    raise ValueError("ablation must be A0, A1, A2, A3, A4, or Full")
