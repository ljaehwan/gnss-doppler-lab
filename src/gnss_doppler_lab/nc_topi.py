"""NC-TOPI Stage-0 contract and mathematical primitives.

This module deliberately contains no experiment runner and no attack-data fit
path.  APIs fail closed when coordinate, provenance, identity, or timing
contracts are incomplete.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import roc_auc_score

CANONICAL_TAP_COORDS = np.array([-.5, -.375, -.25, -.125, 0., .125, .25, .375, .5])
SECOND_PEAK_POWERS = (.05, .1, .2, .4, .8)
SECOND_PEAK_SEPARATIONS = (.0625, .125, .25, .375, .5)
ATTACK_SCENARIOS = ("DS1", "DS2", "DS3", "DS7", "DS8")
NONIDENTIFIABILITY_MARKER = (
    "legacy residual-only tangent is non-identifiable without exact actual and predicted peaks"
)
DEFAULT_SEED = 20260803
RAW_SPACE = "prompt_relative_ratio_raw"
STANDARDIZED_SPACE = "b0_standardized"


def _array(value, name, ndim=None):
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and finite") from exc
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _identity_list(values, n, *, name="identities"):
    if values is None:
        raise ValueError(f"{name} provenance is mandatory")
    result = [tuple(x) if isinstance(x, (tuple, list, np.ndarray)) else (x,) for x in values]
    if len(result) != n:
        raise ValueError(f"{name} length must match rows")
    if any(len(x) == 0 or any(str(v).strip() == "" for v in x) for x in result):
        raise ValueError(f"{name} must be complete")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate identities; identities must be unique")
    return result



def _digest_json(value):
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=True,
                         default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity_digest(identities):
    return _digest_json([list(identity) for identity in identities])


def _array_digest(value):
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    payload = str(array.shape).encode() + b"|float64|" + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _readonly_array(value, name, ndim=None):
    source = np.ascontiguousarray(_array(value, name, ndim), dtype=np.float64)
    # A bytes-backed view cannot have WRITEABLE re-enabled by a caller.
    array = np.frombuffer(source.tobytes(), dtype=np.float64).reshape(source.shape)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FitProvenance:
    """Exact immutable row provenance required by every fit/calibration API."""

    scenario: str
    role: str
    identities: tuple[tuple[object, ...], ...]

    def __post_init__(self):
        if not isinstance(self.scenario, str) or not self.scenario.strip():
            raise ValueError("fit provenance scenario must be a nonempty string")
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("fit provenance role must be a nonempty string")
        identities = tuple(_identity_list(self.identities, len(self.identities),
                                          name="fit provenance identities"))
        if not identities:
            raise ValueError("fit provenance identities cannot be empty")
        object.__setattr__(self, "identities", identities)

    @property
    def identity_digest_sha256(self):
        return _identity_digest(self.identities)


def _require_provenance(provenance, rows, *, role):
    if not isinstance(provenance, FitProvenance):
        raise TypeError("typed fit provenance (FitProvenance) is mandatory")
    if provenance.scenario != "cleanStatic" or provenance.role != role:
        raise ValueError(f"fit requires cleanStatic {role} provenance")
    if len(provenance.identities) != rows:
        raise ValueError("fit provenance identities length must match rows")
    return provenance


def _default_config_path():
    return Path(__file__).resolve().parents[2] / "configs" / "nc_topi_stage0.json"


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("schema") != "gnss-doppler-lab.nc-topi-stage0.v1":
        raise ValueError("unexpected NC-TOPI Stage-0 schema")
    try:
        taps = config["taps"]
        coords = taps["coordinates_chips"]
        b0 = config["b0"]
        split = config["split"]
        decision = config["decision"]
        geometry = config["geometry"]
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete NC-TOPI Stage-0 config") from exc
    if list(coords) != CANONICAL_TAP_COORDS.tolist():
        raise ValueError("tap coordinates must be the explicit canonical tap coordinates")
    if "GNSS-SDR" not in str(taps.get("coordinate_provenance", "")):
        raise ValueError("tap coordinate provenance must name GNSS-SDR")
    if b0.get("checkpoint_sha256") != "f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b":
        raise ValueError("frozen B0 checkpoint hash changed")
    if b0.get("feature_order") != ["E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4"]:
        raise ValueError("frozen B0 feature order changed")
    expected = {
        "train": {"source_end_lte": 300},
        "calibration": {"source_start_gte": 320, "source_end_lte": 400},
        "holdout": {"source_start_gte": 420},
    }
    if split != expected:
        raise ValueError("source-support split contract changed")
    if decision.get("go_primary") != "q99 NC-TOPI median only":
        raise ValueError("primary GO detector changed")
    if geometry.get("primary_include_width") is not False:
        raise ValueError("primary width flag must be the JSON boolean false")
    if geometry.get("primary_tangents") != ["amplitude", "shift"]:
        raise ValueError("primary basis must be amplitude+shift only")
    grammar = decision.get("boolean_grammar", {})
    if grammar.get("GO") != "c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8":
        raise ValueError("machine decision GO grammar changed")
    machine = decision.get("machine_grammar", {})
    if machine.get("go") != {"all": [f"c{i}" for i in range(1, 9)]}:
        raise ValueError("structured GO grammar changed")
    criteria = machine.get("criteria", {})
    if criteria.get("c6", {}).get("rhs") is not True or criteria.get("c7", {}).get("rhs") is not True:
        raise ValueError("physics criteria must use JSON boolean true")
    provenance = config.get("fit_policy", {}).get("typed_provenance")
    if provenance != "FitProvenance(scenario, role, identities)":
        raise ValueError("typed fit provenance contract changed")
    basis = geometry.get("basis_provenance", {})
    if (basis.get("primary_kind") != "primary_amp_shift"
            or basis.get("width_kind") != "width_diagnostic"
            or basis.get("raw_matrix_primary_input") is not False):
        raise ValueError("sealed tangent basis provenance contract changed")
    epoch = config.get("epoch_schema", {})
    if epoch.get("primary_join") != "exact full identity and metadata equality":
        raise ValueError("EpochRecord primary join contract changed")
    domains = decision.get("evidence_domains", {})
    if (domains.get("fpr") != "finite [0,1]"
            or "validation_errors" not in str(domains.get("invalid_evidence", ""))):
        raise ValueError("decision evidence validation contract changed")


def load_config(path=None):
    with Path(path or _default_config_path()).open(encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


@dataclass(frozen=True)
class PeakPredictionPair:
    """One exact B0 target/prediction pair bound to physical coordinates."""

    actual_raw: np.ndarray
    predicted_raw: np.ndarray
    residual_standardized: np.ndarray
    standardizer_std: np.ndarray
    identity: tuple[object, ...]
    actual_space: str
    predicted_space: str
    residual_space: str
    coordinates: np.ndarray

    def __post_init__(self):
        actual = _array(self.actual_raw, "actual_raw", 1)
        predicted = _array(self.predicted_raw, "predicted_raw", 1)
        standardized = _array(self.residual_standardized, "residual_standardized", 1)
        std = _array(self.standardizer_std, "standardizer_std")
        coordinates = _array(self.coordinates, "coordinates", 1)
        if actual.shape != predicted.shape or actual.shape != standardized.shape:
            raise ValueError("actual, predicted, and standardized residual shapes must match")
        if coordinates.shape != actual.shape or np.any(np.diff(coordinates) <= 0):
            raise ValueError("coordinates must match peaks and be strictly increasing")
        if std.ndim == 0:
            std = np.full(actual.shape, float(std))
        if std.ndim != 1 or std.shape != actual.shape or np.any(std <= 0):
            raise ValueError("standardizer_std must be positive scalar or matching vector")
        if self.actual_space != RAW_SPACE or self.predicted_space != RAW_SPACE:
            raise ValueError("actual/predicted space must be prompt_relative_ratio_raw")
        if self.residual_space != STANDARDIZED_SPACE:
            raise ValueError("residual space must be b0_standardized")
        identity = tuple(self.identity)
        if not identity or any(not str(x).strip() for x in identity):
            raise ValueError("identity must be complete")
        expected = (actual - predicted) / std
        if not np.allclose(standardized, expected, rtol=1e-6, atol=1e-9):
            raise ValueError("residual_standardized must equal (actual_raw-predicted_raw)/standardizer_std")
        for name, value in (("actual_raw", actual), ("predicted_raw", predicted),
                            ("residual_standardized", standardized), ("standardizer_std", std),
                            ("coordinates", coordinates), ("identity", identity)):
            if isinstance(value, np.ndarray):
                value = _readonly_array(value, name, value.ndim)
            object.__setattr__(self, name, value)

    @property
    def residual_raw(self):
        result = self.actual_raw - self.predicted_raw
        result.setflags(write=False)
        return result


@dataclass(frozen=True, init=False)
class TangentBasis:
    """Factory-only immutable tangent object sealed to one prediction pair."""

    basis_kind: str
    peak_identity_digest: str
    coordinates_digest: str
    predicted_raw_digest: str
    matrix: np.ndarray
    raw: np.ndarray
    names: tuple[str, ...]
    metadata: Mapping[str, object]
    _construction_seal: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("TangentBasis is factory-only; use a pair-bound basis builder")

    @classmethod
    def _create(cls, pair, coords, matrix, raw, names, basis_kind, metadata):
        instance = object.__new__(cls)
        identity_digest = _identity_digest((pair.identity,))
        coords_digest = _array_digest(coords)
        predicted_digest = _array_digest(pair.predicted_raw)
        names = tuple(names)
        seal = _digest_json([basis_kind, identity_digest, coords_digest,
                             predicted_digest, names, _array_digest(matrix)])
        values = (
            ("basis_kind", basis_kind), ("peak_identity_digest", identity_digest),
            ("coordinates_digest", coords_digest), ("predicted_raw_digest", predicted_digest),
            ("matrix", _readonly_array(matrix, "matrix", 2)),
            ("raw", _readonly_array(raw, "raw", 2)), ("names", names),
            ("metadata", MappingProxyType(dict(metadata))), ("_construction_seal", seal))
        for name, value in values:
            object.__setattr__(instance, name, value)
        return instance

    def validate_for_pair(self, pair, W, *, primary):
        if not isinstance(pair, PeakPredictionPair):
            raise TypeError("TangentBasis validation requires PeakPredictionPair")
        expected_kind = "primary_amp_shift" if primary else "width_diagnostic"
        if self.basis_kind != expected_kind:
            raise ValueError(f"{self.basis_kind} basis cannot be used as {expected_kind}")
        expected_names = ("amplitude", "shift") if primary else ("amplitude", "shift", "width")
        if self.names != expected_names:
            raise ValueError("basis names do not match its sealed basis kind")
        expected = (_identity_digest((pair.identity,)), _array_digest(pair.coordinates),
                    _array_digest(pair.predicted_raw))
        actual = (self.peak_identity_digest, self.coordinates_digest, self.predicted_raw_digest)
        for label, left, right in zip(("identity", "coordinate", "predicted"), actual, expected):
            if left != right:
                raise ValueError(f"basis {label} provenance does not match PeakPredictionPair")
        seal = _digest_json([self.basis_kind, *actual, self.names, _array_digest(self.matrix)])
        if seal != self._construction_seal:
            raise ValueError("TangentBasis construction seal is invalid; arbitrary basis rejected")
        weight = _validate_weight(W, len(pair.predicted_raw))
        first = np.gradient(pair.predicted_raw, pair.coordinates, edge_order=2)
        columns = [pair.predicted_raw, first]
        if not primary:
            columns.append(np.gradient(first, pair.coordinates, edge_order=2))
        expected_matrix = np.column_stack(columns)
        for column in range(expected_matrix.shape[1]):
            norm2 = float(expected_matrix[:, column] @ weight @ expected_matrix[:, column])
            if norm2 <= 0:
                raise ValueError("basis tangent has zero W norm")
            expected_matrix[:, column] /= math.sqrt(norm2)
        if not np.allclose(self.matrix, expected_matrix, rtol=1e-12, atol=1e-14):
            raise ValueError("arbitrary basis matrix rejected; tangents must derive from predicted_raw")


def _validate_weight(W, dimension, *, symmetry_atol=1e-12, psd_atol=1e-12):
    weight = _array(W, "W", 2)
    if weight.shape != (dimension, dimension):
        raise ValueError("W dimensions must match vectors")
    scale = max(1.0, float(np.linalg.norm(weight, ord=2)))
    if not np.allclose(weight, weight.T, rtol=0, atol=symmetry_atol * scale):
        raise ValueError("W must be symmetric")
    weight = (weight + weight.T) / 2
    minimum = float(np.linalg.eigvalsh(weight).min())
    if minimum < -psd_atol * scale:
        raise ValueError("W must be positive semidefinite")
    return weight


def _build_tangents(pair, coords, W, *, include_width, low_signal_epsilon):
    if not isinstance(pair, PeakPredictionPair):
        raise TypeError("tangent construction requires PeakPredictionPair")
    p = pair.predicted_raw
    x = _array(coords, "coords", 1)
    if p.shape != x.shape or p.size < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("predicted peak and strictly increasing physical coords must match")
    if not np.array_equal(x, pair.coordinates):
        raise ValueError("explicit coordinates do not match PeakPredictionPair coordinates")
    if np.linalg.norm(p) <= low_signal_epsilon:
        raise ValueError("low-signal predicted peak cannot define tangents")
    first = np.gradient(p, x, edge_order=2)
    cols = [p, first]
    names = ["amplitude", "shift"]
    if include_width:
        cols.append(np.gradient(first, x, edge_order=2))
        names.append("width")
    raw = np.column_stack(cols)
    weight = np.eye(p.size) if W is None else _validate_weight(W, p.size)
    normalized = raw.copy()
    for column, name in enumerate(names):
        norm2 = float(raw[:, column] @ weight @ raw[:, column])
        if not np.isfinite(norm2) or norm2 <= low_signal_epsilon**2:
            raise ValueError(f"low-signal {name} tangent")
        normalized[:, column] /= math.sqrt(norm2)
    metadata = {
        "derivative_coordinates": "explicit physical chip coordinates",
        "normalization": "each column has unit W norm",
        "primary": not include_width,
        "width": "diagnostic ablation only" if include_width else "excluded from primary",
        "amplitude_semantics": (
            "normalized-shape scale direction in prompt-relative-ratio space; "
            "not physical receiver global gain"),
        "residual_only_allowed": False,
        "nonidentifiability_marker": NONIDENTIFIABILITY_MARKER,
    }
    kind = "width_diagnostic" if include_width else "primary_amp_shift"
    return TangentBasis._create(pair, x, normalized, raw, tuple(names), kind, metadata)


def normalize_tangents(pair, coords, W=None, *, include_width=False,
                       low_signal_epsilon=1e-12):
    """Build the primary pair-bound basis; width requests fail closed."""
    if include_width:
        raise ValueError("width is diagnostic-only; use build_width_ablation_basis")
    return _build_tangents(pair, coords, W, include_width=False,
                           low_signal_epsilon=low_signal_epsilon)


def primary_tangent_basis(pair, coords, W=None, *, low_signal_epsilon=1e-12):
    return normalize_tangents(pair, coords, W=W, include_width=False,
                              low_signal_epsilon=low_signal_epsilon)


def build_width_ablation_basis(pair, coords, W=None, *, low_signal_epsilon=1e-12):
    return _build_tangents(pair, coords, W, include_width=True,
                           low_signal_epsilon=low_signal_epsilon)


@dataclass(frozen=True, init=False)
class ResidualBatch:
    residual_raw: np.ndarray
    identities: tuple[tuple[object, ...], ...]
    pair_identity_digest_sha256: str
    residual_space: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("ResidualBatch is factory-only; use ResidualBatch.from_pairs")

    @classmethod
    def from_pairs(cls, pairs):
        values = tuple(pairs)
        if not values or any(not isinstance(pair, PeakPredictionPair) for pair in values):
            raise TypeError("ResidualBatch can be created only from PeakPredictionPair values")
        identities = tuple(pair.identity for pair in values)
        _identity_list(identities, len(values), name="pair identities")
        if len({pair.residual_raw.shape for pair in values}) != 1:
            raise ValueError("PeakPredictionPair dimensions must match")
        instance = object.__new__(cls)
        data = _readonly_array(np.stack([pair.residual_raw for pair in values]),
                               "residual_raw", 2)
        for name, value in (("residual_raw", data), ("identities", identities),
                            ("pair_identity_digest_sha256", _identity_digest(identities)),
                            ("residual_space", RAW_SPACE)):
            object.__setattr__(instance, name, value)
        return instance


@dataclass(frozen=True)
class CovarianceFit:
    Sigma: np.ndarray
    W: np.ndarray
    Sigma_unfloored: np.ndarray
    audit: dict[str, object]


def _fit_shrinkage_covariance_array(r, *, floor_relative, pinv_rcond):
    """Private array-only numerical helper; never a primary fit entry point."""
    r = _array(r, "residual_raw", 2)
    if r.shape[0] < 2 or r.shape[1] < 1:
        raise ValueError("covariance requires at least two rows and one dimension")
    if floor_relative <= 0 or pinv_rcond <= 0:
        raise ValueError("covariance numerical parameters must be positive")
    unfloored = np.asarray(LedoitWolf().fit(r).covariance_)
    unfloored = (unfloored + unfloored.T) / 2
    nominal = float(floor_relative * np.trace(unfloored) / r.shape[1])
    floor = max(nominal, np.finfo(float).eps)
    values, vectors = np.linalg.eigh(unfloored)
    sigma = (vectors * np.maximum(values, floor)) @ vectors.T
    sigma = (sigma + sigma.T) / 2
    weight = np.linalg.pinv(sigma, rcond=pinv_rcond, hermitian=True)
    return sigma, weight, unfloored, floor, nominal


def fit_shrinkage_covariance(pairs_or_batch, *, provenance,
                             floor_relative=1e-8, pinv_rcond=1e-10):
    if isinstance(pairs_or_batch, ResidualBatch):
        batch = pairs_or_batch
    else:
        if isinstance(pairs_or_batch, np.ndarray):
            raise TypeError("covariance requires PeakPredictionPair sequence or ResidualBatch, not ndarray")
        batch = ResidualBatch.from_pairs(pairs_or_batch)
    fit = _require_provenance(provenance, len(batch.residual_raw), role="normal_train")
    if fit.identities != batch.identities:
        raise ValueError("fit provenance identities must exactly match pair identities and order")
    r = batch.residual_raw
    sigma, weight, unfloored, floor, nominal = _fit_shrinkage_covariance_array(
        r, floor_relative=floor_relative, pinv_rcond=pinv_rcond)
    audit = {
        "estimator": "sklearn.covariance.LedoitWolf",
        "fit_role": "normal_train residual_raw only",
        "scenario": "cleanStatic",
        "residual_space": RAW_SPACE,
        "rows": len(r),
        "dimension": r.shape[1],
        "identity_count": len(batch.identities),
        "identity_digest_sha256": fit.identity_digest_sha256,
        "pair_identity_digest_sha256": batch.pair_identity_digest_sha256,
        "floor_relative": floor_relative,
        "floor_epsilon": floor,
        "nominal_floor_epsilon": nominal,
        "pinv_rcond": pinv_rcond,
    }
    return CovarianceFit(sigma, weight, unfloored, audit)


@dataclass(frozen=True)
class ProjectionResult:
    coefficients: np.ndarray
    fitted: np.ndarray
    r_perp: np.ndarray
    total_energy: float
    tangent_energy: float
    perp_energy: float
    cross_energy: float
    orthogonality_defect: float
    rank: int
    normal_rank: int
    condition: float
    ridge: float
    projection_kind: str
    effective_rank: int
    rank_tolerance: float
    orthogonality_tolerance: float
    full_span_orthogonality_defect: float
    orthogonality_verified_full_span: bool
    orthogonality_scope: str


def _energy(vector, weight, name):
    value = float(vector @ weight @ vector)
    tolerance = 1e-11 * max(1.0, float(np.linalg.norm(vector))**2 * float(np.linalg.norm(weight, 2)))
    if value < -tolerance:
        raise ValueError(f"{name} W-energy is materially negative; W/inputs violate PSD geometry")
    return value


def _energy_audit(r, fitted, perp, weight):
    total = _energy(r, weight, "total")
    tangent = _energy(fitted, weight, "tangent")
    perpendicular = _energy(perp, weight, "perpendicular")
    cross = float(2 * fitted @ weight @ perp)
    tolerance = 64 * np.finfo(float).eps * max(
        np.finfo(float).tiny, abs(total), abs(tangent), abs(perpendicular), abs(cross))
    if abs(total - tangent - perpendicular - cross) > tolerance:
        raise ArithmeticError("W-energy decomposition failed")
    return total, tangent, perpendicular, cross


def _project(residual, J, W, *, ridge_relative, pinv_rcond, projection_kind):
    r = _array(residual, "residual", 1)
    basis = _array(J, "J", 2)
    if basis.shape[0] != r.size:
        raise ValueError("residual and J dimensions do not match")
    weight = _validate_weight(W, r.size)
    if basis.shape[1] < 1 or not 0 < pinv_rcond < 1 or ridge_relative < 0:
        raise ValueError("projection parameters invalid")

    eigenvalues, eigenvectors = np.linalg.eigh(weight)
    eigenvalues = np.maximum(eigenvalues, 0.)
    sqrt_weight = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    whitened_basis = sqrt_weight @ basis
    whitened_residual = sqrt_weight @ r
    U, singular, Vt = np.linalg.svd(whitened_basis, full_matrices=False)
    largest = float(singular[0]) if singular.size else 0.
    rank_tolerance = float(pinv_rcond * largest)
    effective_rank = int(np.count_nonzero(singular > rank_tolerance))
    algebraic_tolerance = np.finfo(float).eps * max(whitened_basis.shape) * largest
    algebraic_rank = int(np.count_nonzero(singular > algebraic_tolerance))

    if ridge_relative == 0:
        if effective_rank:
            coefficients = Vt[:effective_rank].T @ (
                (U[:, :effective_rank].T @ whitened_residual) / singular[:effective_rank])
        else:
            coefficients = np.zeros(basis.shape[1])
        ridge = 0.
    else:
        normal_scale = float(np.sum(singular**2) / max(1, basis.shape[1]))
        ridge = float(ridge_relative * normal_scale) if normal_scale > 0 else float(ridge_relative)
        coefficients = Vt.T @ ((singular / (singular**2 + ridge)) * (U.T @ whitened_residual))

    fitted = basis @ coefficients
    perp = r - fitted
    total, tangent, perpendicular, cross = _energy_audit(r, fitted, perp, weight)
    gradient = whitened_basis.T @ (sqrt_weight @ perp)
    retained_gradient = Vt[:effective_rank] @ gradient if effective_rank else np.empty(0)
    defect = float(np.linalg.norm(retained_gradient))
    full_defect = float(np.linalg.norm(gradient))
    rhs_norm = float(np.linalg.norm(whitened_basis.T @ whitened_residual))
    scale = max(np.finfo(float).tiny, rhs_norm,
                float(np.linalg.norm(whitened_basis, 2) * np.linalg.norm(whitened_residual)))
    orthogonality_tolerance = float(64 * np.finfo(float).eps
                                    * max(1, sum(whitened_basis.shape)) * scale)
    full_verified = effective_rank == algebraic_rank
    if ridge_relative == 0:
        if defect > orthogonality_tolerance:
            raise ArithmeticError(
                "primary W-orthogonality defect exceeds machine-epsilon-scaled tolerance")
        if abs(cross) > 2 * orthogonality_tolerance * max(1., np.linalg.norm(coefficients)):
            raise ArithmeticError("primary W-energy orthogonality check failed")
    retained = singular[:effective_rank]
    condition = float(np.inf if not len(retained) or retained[-1] <= 0
                      else retained[0] / retained[-1])
    scope = "full_tangent_span" if full_verified else "retained_effective_tangent_span"
    return ProjectionResult(
        coefficients, fitted, perp, total, tangent, perpendicular, cross, defect,
        effective_rank, effective_rank, condition, ridge, projection_kind,
        effective_rank, rank_tolerance, orthogonality_tolerance, full_defect,
        full_verified, scope)


def weighted_project(residual, J, W, *, pinv_rcond=1e-10):
    """Whitened-SVD Moore-Penrose W projector with explicit rank cutoff."""
    return _project(residual, J, W, ridge_relative=0.0, pinv_rcond=pinv_rcond,
                    projection_kind="orthogonal_whitened_svd")


def weighted_project_ridge_diagnostic(residual, J, W, *, lambda_relative=1e-8,
                                       pinv_rcond=1e-10):
    if lambda_relative <= 0:
        raise ValueError("ridge diagnostic lambda_relative must be positive")
    return _project(residual, J, W, ridge_relative=lambda_relative, pinv_rcond=pinv_rcond,
                    projection_kind="ridge_diagnostic_not_orthogonal")


def b0_rmse(standardized_residual):
    value = _array(standardized_residual, "standardized_residual")
    if value.ndim == 1:
        return float(np.sqrt(np.mean(value**2)))
    if value.ndim == 2:
        return np.sqrt(np.mean(value**2, axis=1))
    raise ValueError("standardized_residual must be a vector or matrix")


@dataclass(frozen=True)
class ScoreBundle:
    identity: tuple[object, ...]
    b0: float
    total: float
    tangent: float
    perp: float
    cross: float
    topi: float
    predicted_scale: float
    nc_topi: float
    projection: ProjectionResult
    conditioner_transform: np.ndarray | None
    spaces: dict[str, str]


def _produce_scores(pair, basis, W, *, primary, conditioner=None,
                    iq_features=None, energy_epsilon=1e-12):
    if not isinstance(pair, PeakPredictionPair):
        raise TypeError("score construction requires PeakPredictionPair; residual-only is forbidden")
    if not isinstance(basis, TangentBasis):
        raise TypeError("score construction requires a sealed TangentBasis, not a raw basis matrix")
    basis.validate_for_pair(pair, W, primary=primary)
    if energy_epsilon <= 0:
        raise ValueError("energy_epsilon must be positive")
    projection = weighted_project(pair.residual_raw, basis.matrix, W)
    topi = float(projection.perp_energy)
    transformed = None
    scale = 1.0
    if conditioner is not None:
        if iq_features is None:
            raise ValueError("IQ features are required when conditioning")
        features = _array(iq_features, "iq_features", 2)
        if len(features) != 1:
            raise ValueError("one PeakPredictionPair requires one IQ feature row")
        transformed = np.asarray(conditioner.conditioner_transform(features), dtype=float)
        predicted = _array(conditioner.predict_scale(features), "predicted scale", 1)
        if len(predicted) != 1 or predicted[0] <= 0:
            raise ValueError("conditioner must return one positive scale")
        scale = float(predicted[0])
    nc = topi / max(scale, energy_epsilon)
    return ScoreBundle(
        pair.identity, b0_rmse(pair.residual_standardized), projection.total_energy,
        projection.tangent_energy, projection.perp_energy, projection.cross_energy,
        topi, scale, nc, projection, transformed,
        {"geometry": RAW_SPACE, "b0": STANDARDIZED_SPACE,
         "conditioner_target": "log_raw_perp_energy"})


def produce_nc_topi_scores(pair: PeakPredictionPair, basis: TangentBasis, W, *,
                           conditioner=None, iq_features=None, energy_epsilon=1e-12):
    """Primary amplitude+shift score; raw/arbitrary/diagnostic bases fail closed."""
    return _produce_scores(pair, basis, W, primary=True, conditioner=conditioner,
                           iq_features=iq_features, energy_epsilon=energy_epsilon)


@dataclass(frozen=True)
class WidthDiagnosticScore:
    label: str
    primary: bool
    score: ScoreBundle


def produce_width_ablation_scores(pair: PeakPredictionPair, basis: TangentBasis, W, *,
                                   conditioner=None, iq_features=None,
                                   energy_epsilon=1e-12):
    score = _produce_scores(pair, basis, W, primary=False, conditioner=conditioner,
                            iq_features=iq_features, energy_epsilon=energy_epsilon)
    return WidthDiagnosticScore("width_diagnostic", False, score)


@dataclass(frozen=True)
class AggregateResult:
    score: float
    count: int
    selected_count: int
    ids: tuple[str, ...]
    method: str


def aggregate_prn_scores(prn_ids, scores, method="median", *, valid_mask=None):
    ids = np.asarray(prn_ids, dtype=str)
    values = np.asarray(scores, dtype=float)
    if ids.ndim != 1 or values.ndim != 1 or len(ids) != len(values) or len(ids) == 0:
        raise ValueError("PRN IDs and scores must be nonempty matching vectors")
    mask = np.ones(len(values), bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if mask.shape != values.shape or not mask.any():
        raise ValueError("valid mask must match and retain at least one PRN")
    if not np.isfinite(values[mask]).all():
        raise ValueError("active PRN scores must be finite")
    if len(set(ids[mask])) != int(mask.sum()) or any(not x.strip() for x in ids[mask]):
        raise ValueError("active PRN IDs must be unique and nonempty")
    active = values[mask]
    if method == "median":
        score, selected = float(np.median(active)), len(active)
    elif method == "top25_mean":
        selected = int(math.ceil(.25 * len(active)))
        score = float(np.mean(np.sort(active)[-selected:]))
    else:
        raise ValueError("aggregator must be median or top25_mean")
    return AggregateResult(score, len(active), selected, tuple(sorted(ids[mask].tolist())), method)


def higher_quantile(scores, q):
    """Low-level higher-quantile math helper; not a provenance-bearing fit API."""
    values = _array(scores, "scores", 1)
    if len(values) == 0 or not 0 < q < 1:
        raise ValueError("quantile inputs invalid")
    return float(np.quantile(values, q, method="higher"))


@dataclass(frozen=True)
class ThresholdCalibration:
    value: float
    quantile: float
    scenario: str
    role: str
    identity_digest_sha256: str


def calibrate_threshold(scores, q, *, provenance):
    values = _array(scores, "calibration scores", 1)
    fit = _require_provenance(provenance, len(values), role="normal_calibration")
    return ThresholdCalibration(higher_quantile(values, q), float(q), fit.scenario,
                                fit.role, fit.identity_digest_sha256)


def strict_alarms(scores, threshold):
    values = _array(scores, "scores", 1)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return values > threshold


@dataclass(frozen=True)
class SplitMasks:
    train: np.ndarray
    calibration: np.ndarray
    holdout: np.ndarray
    unassigned: np.ndarray


def source_support_split(source_start, source_end, *, scenario):
    start = _array(source_start, "source_start", 1)
    end = _array(source_end, "source_end", 1)
    if start.shape != end.shape or np.any(end < start):
        raise ValueError("source support is invalid")
    if scenario != "cleanStatic":
        raise ValueError("attack scenario can never fit source-support splits")
    train = end <= 300
    calibration = (start >= 320) & (end <= 400)
    holdout = start >= 420
    stack = np.stack([train, calibration, holdout])
    if np.any(stack.sum(axis=0) > 1):
        raise AssertionError("source-support split overlap")
    return SplitMasks(train, calibration, holdout, ~stack.any(axis=0))


@dataclass(frozen=True)
class PhaseMasks:
    stable_pre: np.ndarray
    transition: np.ndarray
    post: np.ndarray
    persistent: np.ndarray


def phase_masks(source_start, source_end, onset):
    start = _array(source_start, "source_start", 1)
    end = _array(source_end, "source_end", 1)
    if start.shape != end.shape or np.any(end < start) or not np.isfinite(onset):
        raise ValueError("phase support is invalid")
    post = start >= onset
    stable = (start >= 30) & (end <= onset - 20)
    transition = (~post) & (~stable)
    persistent = start >= onset + 40
    return PhaseMasks(stable, transition, post, persistent)


@dataclass(frozen=True)
class IQContext:
    contexts: np.ndarray
    valid: np.ndarray
    block_indices: tuple[np.ndarray, ...]
    audit: dict[str, object]


def _validated_group_vector(values, rows, name):
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != rows:
        raise ValueError(f"{name} must match rows")
    if any(not isinstance(value, str) or not value.strip() for value in array):
        raise ValueError(f"{name} group IDs must be nonempty strings")
    return np.asarray(array.tolist(), dtype=str)


def build_causal_iq_context(target_source_start, block_end, block_features, *, history=4,
                            target_groups, block_groups, cadence=.5):
    targets = _array(target_source_start, "target_source_start", 1)
    ends = _array(block_end, "block_end", 1)
    features = _array(block_features, "block_features", 2)
    if len(ends) != len(features) or history < 1 or cadence <= 0:
        raise ValueError("IQ blocks/history dimensions invalid")
    target_group = _validated_group_vector(target_groups, len(targets), "target_groups")
    block_group = _validated_group_vector(block_groups, len(ends), "block_groups")
    if set(target_group) != set(block_group):
        raise ValueError("target_groups and block_groups must exactly match")
    gap_count = 0
    for group in sorted(set(block_group)):
        ix = np.flatnonzero(block_group == group)
        group_ends = ends[ix]
        if len(np.unique(group_ends)) != len(group_ends):
            raise ValueError(f"duplicate block end within group {group}")
        if np.any(np.diff(group_ends) <= 0):
            raise ValueError(f"block ends must be sorted within group {group}")
        gap_count += int(np.count_nonzero(~np.isclose(np.diff(group_ends), cadence, rtol=0, atol=1e-8)))
        target_ix = np.flatnonzero(target_group == group)
        if np.any(np.diff(targets[target_ix]) < 0):
            raise ValueError(f"targets must be sorted within group {group}")
    contexts = np.full((len(targets), history, features.shape[1]), np.nan)
    valid = np.zeros(len(targets), bool)
    selected = []
    for row, (target, group) in enumerate(zip(targets, target_group)):
        eligible = np.flatnonzero((block_group == group) & (ends <= target))
        chosen = eligible[-history:]
        selected.append(chosen)
        contiguous = len(chosen) == history and (
            history == 1 or np.allclose(np.diff(ends[chosen]), cadence, rtol=0, atol=1e-8))
        if contiguous:
            contexts[row] = features[chosen]
            valid[row] = True
    audit = {
        "group_set": tuple(sorted(set(target_group))),
        "groups_exact_match": True,
        "sorted": True,
        "cadence_seconds": cadence,
        "cadence_gap_count": gap_count,
        "cadence_ok": gap_count == 0,
        "cross_recording_default": False,
    }
    return IQContext(contexts, valid, tuple(selected), audit)


class RobustConditioner:
    """Clean-only robust-standardized Huber model of log perpendicular energy."""

    def __init__(self, *, energy_epsilon=1e-12, lower_epsilon=1e-8):
        if energy_epsilon <= 0 or lower_epsilon <= 0:
            raise ValueError("conditioner epsilons must be positive")
        self.energy_epsilon = float(energy_epsilon)
        self.lower_epsilon = float(lower_epsilon)

    def fit(self, X, y, *, provenance, feature_names=None):
        predictors = _array(X, "IQ predictors", 2)
        energy = _array(y, "S_perp target", 1)
        if len(predictors) != len(energy) or np.any(energy < 0):
            raise ValueError("conditioner fit dimensions/nonnegative energy invalid")
        fit = _require_provenance(provenance, len(energy), role="normal_train")
        names = [str(x).lower() for x in (feature_names or [f"x{i}" for i in range(predictors.shape[1])])]
        forbidden = {"prn", "prn_id", "scenario", "scenario_id", "onset", "onset_s"}
        if forbidden.intersection(names):
            raise ValueError("forbidden PRN/scenario/onset identity feature")
        self.median_ = np.median(predictors, axis=0)
        q75, q25 = np.percentile(predictors, [75, 25], axis=0)
        self.iqr_ = q75 - q25
        self.iqr_[self.iqr_ <= self.lower_epsilon] = 1.0
        target = np.log(np.maximum(energy, self.energy_epsilon))
        self.model_ = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000).fit(
            self.conditioner_transform(predictors), target)
        self.feature_names_ = tuple(feature_names or [f"x{i}" for i in range(predictors.shape[1])])
        self._fit_identities_ = fit.identities
        fit_digest = _digest_json({
            "scenario": fit.scenario, "role": fit.role,
            "identity_digest_sha256": fit.identity_digest_sha256,
            "predictors_digest_sha256": _array_digest(predictors),
            "target_energy_digest_sha256": _array_digest(energy)})
        self.fit_manifest_ = {
            "scenario": fit.scenario,
            "roles": [fit.role],
            "rows": len(target),
            "identity_digest_sha256": fit.identity_digest_sha256,
            "fit_digest_sha256": fit_digest,
            "target": "log(max(S_perp, energy_epsilon))",
            "prediction": "exp(predicted_log_energy)",
            "PRN_feature": False,
            "scenario_feature": False,
            "onset_feature": False,
            "epsilon": 1.35,
            "alpha": 1e-4,
            "max_iter": 1000,
        }
        self.cap_ = None
        self.cap_manifest_ = None
        return self

    def conditioner_transform(self, X):
        if not hasattr(self, "median_"):
            raise RuntimeError("conditioner is not fit")
        predictors = _array(X, "IQ predictors", 2)
        if predictors.shape[1] != len(self.median_):
            raise ValueError("IQ predictor dimension changed")
        return (predictors - self.median_) / self.iqr_

    def predict_log_energy(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("conditioner is not fit")
        return np.asarray(self.model_.predict(self.conditioner_transform(X)), dtype=float)

    def _uncapped_scale(self, X):
        # The model output is log energy, so scale is exactly exp(log energy).
        # Clipping only prevents IEEE overflow/underflow; the NC denominator,
        # not this transform, applies energy_epsilon.
        return np.exp(np.clip(self.predict_log_energy(X), -745, 709))

    def calibrate_cap(self, X_calibration, *, provenance, q=.995):
        predictors = _array(X_calibration, "calibration IQ predictors", 2)
        fit = _require_provenance(provenance, len(predictors), role="normal_calibration")
        if set(fit.identities).intersection(self._fit_identities_):
            raise ValueError("calibration identities must be disjoint from conditioner fit identities")
        if not np.isclose(q, .995, rtol=0, atol=0):
            raise ValueError("calibration cap is frozen at q995")
        values = self._uncapped_scale(predictors)
        self.cap_ = higher_quantile(values, q)
        self.cap_manifest_ = {"scenario": fit.scenario, "role": fit.role,
                              "identity_digest_sha256": fit.identity_digest_sha256,
                              "rows": len(values), "quantile": q}
        return self.cap_

    def predict_scale(self, X):
        if self.cap_ is None:
            raise RuntimeError("clean calibration cap is not set")
        return np.minimum(self._uncapped_scale(X), self.cap_)


def shuffled_control_target(target, *, provenance, seed=DEFAULT_SEED):
    values = _array(target, "shuffle target", 1)
    _require_provenance(provenance, len(values), role="normal_train")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("shuffle seed must be an integer")
    return values[np.random.default_rng(int(seed)).permutation(len(values))]


def standardized_pauc(labels, scores, *, max_fpr=.05):
    y = np.asarray(labels)
    score = _array(scores, "scores", 1)
    if y.ndim != 1 or len(y) != len(score) or set(np.unique(y)) != {0, 1}:
        raise ValueError("partial AUC requires binary labels with both classes")
    if not 0 < max_fpr <= 1:
        raise ValueError("max_fpr must be in (0,1]")
    return float(roc_auc_score(y, score, max_fpr=max_fpr))


@dataclass(frozen=True)
class SustainedAlarm:
    delay: float
    alarm_time: float
    already_alarming_stable_pre: bool
    stable_pre_alarm_by_recording: Mapping[str, bool]


def sustained_alarm_delay(availability_source_end, alarms, *, recording_ids,
                           post_eligible_mask, onset, required=3, cadence=.5,
                           stable_pre_mask=None):
    times = _array(availability_source_end, "availability source_end", 1)
    raw_flags = np.asarray(alarms)
    raw_post = np.asarray(post_eligible_mask)
    if raw_flags.dtype.kind != "b" and not np.isin(raw_flags, [0, 1]).all():
        raise ValueError("alarms must be boolean")
    if raw_post.dtype.kind != "b" and not np.isin(raw_post, [0, 1]).all():
        raise ValueError("post_eligible_mask must be boolean")
    flags = raw_flags.astype(bool)
    post = raw_post.astype(bool)
    recordings = _validated_group_vector(recording_ids, len(times), "recording_ids")
    if (flags.shape != times.shape or post.shape != times.shape or required < 1
            or cadence <= 0 or not np.isfinite(onset)):
        raise ValueError("sustained alarm inputs invalid")
    if stable_pre_mask is None:
        pre = np.zeros(len(flags), bool)
    else:
        raw_pre = np.asarray(stable_pre_mask)
        if raw_pre.dtype.kind != "b" and not np.isin(raw_pre, [0, 1]).all():
            raise ValueError("stable_pre_mask must be boolean")
        pre = raw_pre.astype(bool)
    if pre.shape != flags.shape:
        raise ValueError("stable_pre_mask length mismatch")
    if np.any(post & (times < onset)):
        raise ValueError("post-eligible availability cannot precede onset")

    earliest = math.inf
    stable_audit = {}
    for recording in sorted(set(recordings)):
        indices = np.flatnonzero(recordings == recording)
        order = indices[np.argsort(times[indices], kind="mergesort")]
        ordered_times = times[order]
        if len(np.unique(ordered_times)) != len(ordered_times):
            raise ValueError(f"duplicate (recording,time) rows for {recording}")
        stable_audit[recording] = bool(np.any(flags[order] & pre[order]))
        run = 0
        previous_time = None
        for index, time in zip(order, ordered_times):
            contiguous = (previous_time is not None and
                          math.isclose(float(time - previous_time), cadence,
                                       rel_tol=0, abs_tol=1e-8))
            if not post[index]:
                run = 0
            elif flags[index]:
                run = run + 1 if contiguous else 1
            else:
                run = 0
            previous_time = float(time)
            if post[index] and run >= required:
                earliest = min(earliest, float(time))
                break
    delay = float(earliest - onset) if np.isfinite(earliest) else math.inf
    audit = MappingProxyType(stable_audit)
    return SustainedAlarm(delay, earliest, any(stable_audit.values()), audit)


@dataclass(frozen=True)
class IntersectionDiagnostic:
    identities: tuple[tuple[object, ...], ...]
    scores_a: np.ndarray
    scores_b: np.ndarray
    audit: dict[str, int]


@dataclass(frozen=True)
class EpochRecord:
    physical_recording_id: str
    scenario: str
    prn_or_event_id: str
    availability_time_s: float
    source_start_s: float
    source_end_s: float
    valid: bool
    label: object

    def __post_init__(self):
        for name in ("physical_recording_id", "scenario", "prn_or_event_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        times = (self.availability_time_s, self.source_start_s, self.source_end_s)
        numeric = (int, float, np.integer, np.floating)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, numeric)
               or not np.isfinite(value) for value in times):
            raise ValueError("epoch times must be finite numbers")
        availability, source_start, source_end = (float(value) for value in times)
        if source_end < source_start:
            raise ValueError("epoch source interval is reversed")
        if availability < source_end:
            raise ValueError("epoch availability cannot precede source_end")
        object.__setattr__(self, "availability_time_s", availability)
        object.__setattr__(self, "source_start_s", source_start)
        object.__setattr__(self, "source_end_s", source_end)
        if type(self.valid) is not bool:
            raise ValueError("epoch valid must be a true bool")
        if self.label is None or (isinstance(self.label, str) and not self.label.strip()):
            raise ValueError("epoch label is mandatory")
        try:
            hash(self.label)
        except TypeError as exc:
            raise ValueError("epoch label must be immutable/hashable") from exc

    @property
    def identity_key(self):
        return (self.physical_recording_id, self.scenario, self.prn_or_event_id,
                float(self.availability_time_s))


def _record_map(records, name):
    values = tuple(records)
    if not values or any(not isinstance(record, EpochRecord) for record in values):
        raise TypeError(f"{name} must be a nonempty EpochRecord sequence")
    result = {}
    for record in values:
        if record.identity_key in result:
            raise ValueError(f"{name} contains duplicate full identity keys")
        result[record.identity_key] = record
    return values, result


def _score_map(scores, identities, name):
    if not isinstance(scores, Mapping):
        raise TypeError(f"{name} must be a score map keyed by full identity")
    if set(scores) != set(identities):
        raise ValueError(f"{name} score map keys must exactly equal EpochRecord identities")
    result = {}
    for identity, value in scores.items():
        number = _array([value], name, 1)[0]
        result[identity] = float(number)
    return result


def exact_primary_epoch_join(records_a, score_map_a, records_b, score_map_b):
    """Exact primary join with fixed identity and mandatory record metadata."""
    left, left_records = _record_map(records_a, "records_a")
    _, right_records = _record_map(records_b, "records_b")
    if set(left_records) != set(right_records):
        raise ValueError("primary join requires full identity set equality")
    left_scores = _score_map(score_map_a, left_records, "score_map_a")
    right_scores = _score_map(score_map_b, right_records, "score_map_b")
    for identity, left_record in left_records.items():
        right_record = right_records[identity]
        for field, label in (("source_start_s", "source interval"),
                             ("source_end_s", "source interval"),
                             ("label", "label"), ("valid", "valid")):
            if getattr(left_record, field) != getattr(right_record, field):
                raise ValueError(f"{label} differs at identity {identity}")
    ordered = tuple(record.identity_key for record in left)
    return ordered, np.asarray([left_scores[key] for key in ordered]), np.asarray(
        [right_scores[key] for key in ordered])


def common_epoch_intersection_diagnostic(identity_a, scores_a, identity_b, scores_b):
    a = _array(scores_a, "scores_a", 1)
    b = _array(scores_b, "scores_b", 1)
    ids_a = _identity_list(identity_a, len(a), name="identity_a")
    ids_b = _identity_list(identity_b, len(b), name="identity_b")
    map_a, map_b = dict(zip(ids_a, a)), dict(zip(ids_b, b))
    common = tuple(identity for identity in ids_a if identity in map_b)
    return IntersectionDiagnostic(
        common, np.asarray([map_a[x] for x in common]), np.asarray([map_b[x] for x in common]),
        {"input_a": len(ids_a), "input_b": len(ids_b), "common": len(common),
         "excluded_from_a": len(ids_a) - len(common), "excluded_from_b": len(ids_b) - len(common)})


@dataclass(frozen=True)
class BootstrapResult:
    available: bool
    reason: str | None
    point_estimate: float
    ci: tuple[float, float]
    replicates: np.ndarray
    valid_reps: int
    complete_block_count: int
    block_epoch_count: int
    audit: dict[str, object]


def _unavailable_bootstrap(reason, *, point=math.nan, block_count=0, block_epochs=0, audit=None):
    details = {"iid_fallback": False, "reason": reason}
    if audit:
        details.update(audit)
    return BootstrapResult(False, reason, float(point), (math.nan, math.nan), np.empty(0), 0,
                           block_count, block_epochs, details)


def paired_pauc_delta_block_bootstrap(labels, score_a, score_b, recording_ids, times, *,
                                      max_fpr=.05, block_seconds=10., cadence=.5,
                                      reps=2000, seed=DEFAULT_SEED):
    y = np.asarray(labels)
    a = _array(score_a, "score_a", 1)
    b = _array(score_b, "score_b", 1)
    t = _array(times, "times", 1)
    recordings = np.asarray(recording_ids, dtype=str)
    if not (y.ndim == 1 and len(y) == len(a) == len(b) == len(t) == len(recordings)) or len(y) == 0:
        raise ValueError("paired pAUC bootstrap inputs must be nonempty matching vectors")
    if reps < 1 or cadence <= 0 or block_seconds <= 0 or not 0 < max_fpr <= 1:
        raise ValueError("paired pAUC bootstrap parameters invalid")
    if not set(np.unique(y)).issubset({0, 1}) or len(set(np.unique(y))) < 2:
        return _unavailable_bootstrap("class-deficient eligible epochs")
    y = y.astype(int)
    point = standardized_pauc(y, a, max_fpr=max_fpr) - standardized_pauc(y, b, max_fpr=max_fpr)
    epochs = int(round(block_seconds / cadence))
    if epochs < 1 or not np.isclose(epochs * cadence, block_seconds):
        raise ValueError("block duration must be an integer cadence count")
    pools = {0: [], 1: []}
    for recording in sorted(set(recordings)):
        if not recording.strip():
            raise ValueError("recording_ids must be nonempty")
        ix = np.flatnonzero(recordings == recording)
        order = ix[np.argsort(t[ix], kind="mergesort")]
        if len(np.unique(t[order])) != len(order):
            raise ValueError("times must be unique within recording")
        boundaries = [0]
        for pos in range(1, len(order)):
            if (y[order[pos]] != y[order[pos - 1]] or
                    not np.isclose(t[order[pos]] - t[order[pos - 1]], cadence, rtol=0, atol=1e-8)):
                boundaries.append(pos)
        boundaries.append(len(order))
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            for start in range(left, right - epochs + 1, epochs):
                block = order[start:start + epochs]
                label = int(y[block[0]])
                if (np.all(y[block] == label) and
                        np.allclose(np.diff(t[block]), cadence, rtol=0, atol=1e-8)):
                    pools[label].append(block)
    total_blocks = len(pools[0]) + len(pools[1])
    audit = {
        "resampling": "paired pAUC delta; recording/gap-safe complete nonoverlapping 10s blocks stratified by label",
        "iid_fallback": False,
        "point_estimate_rows": len(y),
        "negative_blocks": len(pools[0]),
        "positive_blocks": len(pools[1]),
        "paired_indices": True,
        "max_fpr": max_fpr,
        "reps_requested": reps,
        "seed": seed,
    }
    if len(pools[0]) < 2 or len(pools[1]) < 2:
        return _unavailable_bootstrap("too few complete blocks in one or both class strata",
                                      point=point, block_count=total_blocks,
                                      block_epochs=epochs, audit=audit)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps):
        chosen = []
        for label in (0, 1):
            draws = rng.integers(0, len(pools[label]), len(pools[label]))
            chosen.extend(pools[label][index] for index in draws)
        selected = np.concatenate(chosen)
        value = (standardized_pauc(y[selected], a[selected], max_fpr=max_fpr)
                 - standardized_pauc(y[selected], b[selected], max_fpr=max_fpr))
        if np.isfinite(value):
            samples.append(value)
    replicates = np.asarray(samples, dtype=float)
    if len(replicates) == 0:
        return _unavailable_bootstrap("no valid bootstrap replicates", point=point,
                                      block_count=total_blocks, block_epochs=epochs, audit=audit)
    ci = tuple(float(x) for x in np.percentile(replicates, [2.5, 97.5]))
    audit["valid_reps"] = len(replicates)
    return BootstrapResult(True, None, float(point), ci, replicates, len(replicates),
                           total_blocks, epochs, audit)


def shift_peak(peak, coords, shift_chips):
    values = _array(peak, "peak", 1)
    x = _array(coords, "coords", 1)
    if values.shape != x.shape or np.any(np.diff(x) <= 0) or not np.isfinite(shift_chips):
        raise ValueError("physical peak/coordinate interpolation inputs invalid")
    return np.interp(x - shift_chips, x, values, left=0., right=0.)


def second_peak_perturbation(peak, coords, relative_power, separation_chips, *,
                             enforce_stage0_grid=True):
    if enforce_stage0_grid and (relative_power not in SECOND_PEAK_POWERS
                                or separation_chips not in SECOND_PEAK_SEPARATIONS):
        raise ValueError("second peak must use the frozen Stage-0 physical grid")
    if relative_power < 0:
        raise ValueError("relative power must be nonnegative")
    values = _array(peak, "peak", 1)
    return values + math.sqrt(relative_power) * shift_peak(values, coords, separation_chips)


def equal_w_norm(vector, reference, W):
    value = _array(vector, "vector", 1)
    ref = _array(reference, "reference", 1)
    weight = _validate_weight(W, len(value))
    if value.shape != ref.shape:
        raise ValueError("equal norm dimensions invalid")
    value_norm = _energy(value, weight, "vector")
    ref_norm = _energy(ref, weight, "reference")
    if value_norm <= 0:
        raise ValueError("equal norm requires positive vector norm")
    return value * math.sqrt(ref_norm / value_norm)


def w_orthogonal_vector(J, W, *, seed=DEFAULT_SEED):
    basis = _array(J, "J", 2)
    weight = _validate_weight(W, basis.shape[0])
    rng = np.random.default_rng(seed)
    for _ in range(100):
        candidate = rng.normal(size=basis.shape[0])
        perpendicular = weighted_project(candidate, basis, weight).r_perp
        if _energy(perpendicular, weight, "perpendicular") > 1e-18:
            return perpendicular
    raise ValueError("tangent span has no stable W-orthogonal complement")


@dataclass(frozen=True)
class DecisionResult:
    status: str
    criteria: dict[str, bool]
    counts: dict[str, int]
    missing_evidence: tuple[str, ...]
    no_go_triggers: tuple[str, ...]
    validation_errors: tuple[str, ...]


def _domain_value(value, name, lower, upper, errors):
    if isinstance(value, (bool, np.bool_)):
        errors.append(f"{name}: expected finite number in [{lower},{upper}]")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name}: expected finite number in [{lower},{upper}]")
        return None
    if not np.isfinite(number) or not lower <= number <= upper:
        errors.append(f"{name}: outside finite domain [{lower},{upper}]")
        return None
    return number


def _scenario_evidence(value, name, errors):
    if not isinstance(value, Mapping):
        errors.append(f"{name}: mapping is mandatory")
        return {scenario: None for scenario in ATTACK_SCENARIOS}
    keys = set(value)
    expected = set(ATTACK_SCENARIOS)
    if keys != expected:
        errors.append(f"{name}: scenario set must be exactly {ATTACK_SCENARIOS}; "
                      f"missing={sorted(expected-keys)}, "
                      f"unknown={sorted(map(str, keys-expected))}")
    return {scenario: value.get(scenario) for scenario in ATTACK_SCENARIOS}


def evaluate_stage0_decision(*, clean_nc_fpr=None, clean_b0_fpr=None,
                             stable_pre_fpr=None, nc_pauc=None, b0_pauc=None,
                             pauc_delta=None, nc_delay=None, b0_delay=None,
                             pauc_ci_lower=None, pauc_ci_upper=None,
                             equal_rmse_pass=None, second_peak_pass=None,
                             actual_nc_mean_pauc=None, topi_mean_pauc=None,
                             shuffled_nc_mean_pauc=None,
                             actual_nc_mean_pauc_gain=None,
                             shuffled_nc_mean_pauc_gain=None):
    errors = []
    if actual_nc_mean_pauc_gain is not None or shuffled_nc_mean_pauc_gain is not None:
        errors.append("legacy mean-gain evidence is unsupported; supply bounded pAUC means")
    nc_fpr = _domain_value(clean_nc_fpr, "clean_nc_fpr", 0., 1., errors)
    b0_fpr = _domain_value(clean_b0_fpr, "clean_b0_fpr", 0., 1., errors)
    mappings = {name: _scenario_evidence(value, name, errors) for name, value in (
        ("stable_pre_fpr", stable_pre_fpr), ("nc_pauc", nc_pauc), ("b0_pauc", b0_pauc),
        ("pauc_delta", pauc_delta), ("nc_delay", nc_delay), ("b0_delay", b0_delay),
        ("pauc_ci_lower", pauc_ci_lower), ("pauc_ci_upper", pauc_ci_upper))}

    stable = {}; nc_points = {}; b0_points = {}; deltas = {}; nc_delays = {}; b0_delays = {}
    lowers = {}; uppers = {}
    for scenario in ATTACK_SCENARIOS:
        stable[scenario] = _domain_value(mappings["stable_pre_fpr"][scenario],
                                         f"stable_pre_fpr:{scenario}", 0., 1., errors)
        nc_points[scenario] = _domain_value(mappings["nc_pauc"][scenario],
                                             f"nc_pauc:{scenario}", 0., 1., errors)
        b0_points[scenario] = _domain_value(mappings["b0_pauc"][scenario],
                                             f"b0_pauc:{scenario}", 0., 1., errors)
        deltas[scenario] = _domain_value(mappings["pauc_delta"][scenario],
                                          f"pauc_delta:{scenario}", -1., 1., errors)
        lowers[scenario] = _domain_value(mappings["pauc_ci_lower"][scenario],
                                          f"pauc_ci_lower:{scenario}", -1., 1., errors)
        uppers[scenario] = _domain_value(mappings["pauc_ci_upper"][scenario],
                                          f"pauc_ci_upper:{scenario}", -1., 1., errors)
        for name, destination in (("nc_delay", nc_delays), ("b0_delay", b0_delays)):
            raw = mappings[name][scenario]
            if raw is None:
                destination[scenario] = None
            else:
                destination[scenario] = _domain_value(raw, f"{name}:{scenario}", 0., math.inf, errors)
        if lowers[scenario] is not None and uppers[scenario] is not None:
            if lowers[scenario] > uppers[scenario]:
                errors.append(f"pauc_ci:{scenario}: lower exceeds upper")
        if (nc_points[scenario] is not None and b0_points[scenario] is not None
                and deltas[scenario] is not None
                and not math.isclose(deltas[scenario], nc_points[scenario] - b0_points[scenario],
                                     rel_tol=1e-9, abs_tol=1e-12)):
            errors.append(f"pauc_delta:{scenario}: inconsistent with NC-B0 pAUC points")

    actual_mean = _domain_value(actual_nc_mean_pauc, "actual_nc_mean_pauc", 0., 1., errors)
    topi_mean = _domain_value(topi_mean_pauc, "topi_mean_pauc", 0., 1., errors)
    shuffled_mean = _domain_value(shuffled_nc_mean_pauc, "shuffled_nc_mean_pauc", 0., 1., errors)
    equal_known = type(equal_rmse_pass) is bool
    second_known = type(second_peak_pass) is bool
    if not equal_known:
        errors.append("equal_rmse_pass: true bool is mandatory")
    if not second_known:
        errors.append("second_peak_pass: true bool is mandatory")

    if errors:
        return DecisionResult("INCONCLUSIVE", {f"c{i}": False for i in range(1, 9)},
                              {"stable_pre_failures": 0, "improvement_count": 0,
                               "positive_ci_count": 0}, (), (), tuple(sorted(set(errors))))

    c1 = nc_fpr <= .02
    c2 = nc_fpr - b0_fpr <= .01
    stable_failures = sum(value >= .05 for value in stable.values())
    c3 = stable_failures == 0
    missing = []
    improvement_count = 0
    improvement_known = 0
    for scenario in ATTACK_SCENARIOS:
        if deltas[scenario] > 0:
            improvement_count += 1
            improvement_known += 1
        elif nc_delays[scenario] is not None and b0_delays[scenario] is not None:
            improvement_count += int(b0_delays[scenario] - nc_delays[scenario] >= .5)
            improvement_known += 1
        else:
            missing.append(f"scenario_improvement:{scenario}:censored_delay")
    c4 = improvement_known == len(ATTACK_SCENARIOS) and improvement_count >= 3
    positive_ci_count = sum(value > 0 for value in lowers.values())
    c5 = positive_ci_count >= 2
    c6 = equal_rmse_pass
    c7 = second_peak_pass
    actual_gain = actual_mean - topi_mean
    shuffled_gain = shuffled_mean - topi_mean
    c8 = actual_gain > shuffled_gain and actual_gain > 0
    criteria = {f"c{index}": value for index, value in enumerate(
        (c1, c2, c3, c4, c5, c6, c7, c8), start=1)}
    triggers = []
    if not equal_rmse_pass: triggers.append("c6_equal_rmse_false")
    if not second_peak_pass: triggers.append("c7_second_peak_false")
    if nc_fpr > .05: triggers.append("clean_nc_fpr_gt_0.05")
    if stable_failures >= 3: triggers.append("stable_pre_failures_gte_3")
    if improvement_known == len(ATTACK_SCENARIOS) and improvement_count <= 1:
        triggers.append("improvement_count_lte_1")
    if positive_ci_count == 0:
        triggers.append("positive_ci_count_eq_0")
    if triggers:
        status = "NO-GO"
    elif all(criteria.values()) and not missing:
        status = "GO"
    else:
        status = "INCONCLUSIVE"
    return DecisionResult(status, criteria,
                          {"stable_pre_failures": stable_failures,
                           "improvement_count": improvement_count,
                           "positive_ci_count": positive_ci_count},
                          tuple(sorted(set(missing))), tuple(triggers), ())
