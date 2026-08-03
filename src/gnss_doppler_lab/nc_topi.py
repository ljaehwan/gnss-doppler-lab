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
from typing import Callable, Mapping, Sequence

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


def load_config(path=None):
    with Path(path or _default_config_path()).open(encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


@dataclass(frozen=True)
class PeakPredictionPair:
    """One exact B0 target/prediction pair with explicit coordinate spaces."""

    actual_raw: np.ndarray
    predicted_raw: np.ndarray
    residual_standardized: np.ndarray
    standardizer_std: np.ndarray
    identity: tuple[object, ...]
    actual_space: str
    predicted_space: str
    residual_space: str

    def __post_init__(self):
        actual = _array(self.actual_raw, "actual_raw", 1)
        predicted = _array(self.predicted_raw, "predicted_raw", 1)
        standardized = _array(self.residual_standardized, "residual_standardized", 1)
        std = _array(self.standardizer_std, "standardizer_std")
        if actual.shape != predicted.shape or actual.shape != standardized.shape:
            raise ValueError("actual, predicted, and standardized residual shapes must match")
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
                            ("identity", identity)):
            object.__setattr__(self, name, value)

    @property
    def residual_raw(self):
        return self.actual_raw - self.predicted_raw


@dataclass(frozen=True)
class TangentBasis:
    matrix: np.ndarray
    raw: np.ndarray
    names: tuple[str, ...]
    metadata: dict[str, object]


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


def _build_tangents(predicted_peak, coords, W, *, include_width, low_signal_epsilon):
    p = _array(predicted_peak, "predicted_peak", 1)
    x = _array(coords, "coords", 1)
    if p.shape != x.shape or p.size < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("predicted_peak and strictly increasing physical coords must match")
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
            "not physical receiver global gain"
        ),
        "residual_only_allowed": False,
        "nonidentifiability_marker": NONIDENTIFIABILITY_MARKER,
    }
    return TangentBasis(normalized, raw, tuple(names), metadata)


def normalize_tangents(predicted_peak, coords, W=None, *, include_width=False,
                       low_signal_epsilon=1e-12):
    """Build the primary basis; width requests fail closed.

    There is intentionally no ``input_kind`` default or residual-only mode.
    """
    if include_width:
        raise ValueError("width is diagnostic-only; use build_width_ablation_basis")
    return _build_tangents(predicted_peak, coords, W, include_width=False,
                           low_signal_epsilon=low_signal_epsilon)


def primary_tangent_basis(predicted_peak, coords, W=None, *, low_signal_epsilon=1e-12):
    return normalize_tangents(predicted_peak, coords, W=W, include_width=False,
                              low_signal_epsilon=low_signal_epsilon)


def build_width_ablation_basis(predicted_peak, coords, W=None, *, low_signal_epsilon=1e-12):
    return _build_tangents(predicted_peak, coords, W, include_width=True,
                           low_signal_epsilon=low_signal_epsilon)


@dataclass(frozen=True)
class CovarianceFit:
    Sigma: np.ndarray
    W: np.ndarray
    Sigma_unfloored: np.ndarray
    audit: dict[str, object]


def assert_fit_is_clean_only(roles, *, allowed=("normal_train", "normal_calibration")):
    role_list = [str(x) for x in roles]
    if not role_list:
        raise ValueError("normal-only fit roles cannot be empty")
    bad = sorted(set(role_list) - set(allowed))
    if bad:
        raise ValueError(f"attack/holdout data cannot fit; allowed normal roles are {allowed}, got {bad}")


def fit_shrinkage_covariance(residuals, *, fit_roles=None, scenarios=None, identities=None,
                             floor_relative=1e-8, pinv_rcond=1e-10):
    r = _array(residuals, "residuals", 2)
    if r.shape[0] < 2 or r.shape[1] < 1:
        raise ValueError("covariance requires at least two rows and one dimension")
    if fit_roles is None or scenarios is None or identities is None:
        raise ValueError("covariance role, scenario, and identity provenance are mandatory")
    roles = [str(x) for x in fit_roles]
    scenario_values = [str(x) for x in scenarios]
    identity_values = _identity_list(identities, len(r))
    if len(roles) != len(r) or len(scenario_values) != len(r):
        raise ValueError("covariance roles/scenarios must match residual rows")
    if set(roles) != {"normal_train"} or set(scenario_values) != {"cleanStatic"}:
        raise ValueError("covariance fit allows only cleanStatic + normal_train")
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
    encoded = json.dumps([[str(v) for v in identity] for identity in identity_values],
                         separators=(",", ":"), ensure_ascii=True).encode()
    audit = {
        "estimator": "sklearn.covariance.LedoitWolf",
        "fit_role": "normal_train residual_raw only",
        "scenario": "cleanStatic",
        "rows": len(r),
        "dimension": r.shape[1],
        "identity_count": len(identity_values),
        "identity_digest_sha256": hashlib.sha256(encoded).hexdigest(),
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


def _energy(vector, weight, name):
    value = float(vector @ weight @ vector)
    tolerance = 1e-11 * max(1.0, float(np.linalg.norm(vector))**2 * float(np.linalg.norm(weight, 2)))
    if value < -tolerance:
        raise ValueError(f"{name} W-energy is materially negative; W/inputs violate PSD geometry")
    return value


def _project(residual, J, W, *, ridge_relative, pinv_rcond, projection_kind):
    r = _array(residual, "residual", 1)
    basis = _array(J, "J", 2)
    if basis.shape[0] != r.size:
        raise ValueError("residual and J dimensions do not match")
    weight = _validate_weight(W, r.size)
    if basis.shape[1] < 1 or pinv_rcond <= 0 or ridge_relative < 0:
        raise ValueError("projection parameters invalid")
    normal = basis.T @ weight @ basis
    normal = (normal + normal.T) / 2
    scale = float(np.trace(normal) / max(1, normal.shape[0]))
    ridge = float(ridge_relative * scale) if scale > 0 else float(ridge_relative)
    operator = normal if ridge == 0 else normal + ridge * np.eye(normal.shape[0])
    coefficients = np.linalg.pinv(operator, rcond=pinv_rcond, hermitian=True) @ basis.T @ weight @ r
    fitted = basis @ coefficients
    perp = r - fitted
    total = _energy(r, weight, "total")
    tangent = _energy(fitted, weight, "tangent")
    perpendicular = _energy(perp, weight, "perpendicular")
    cross = float(2 * fitted @ weight @ perp)
    defect = float(np.linalg.norm(basis.T @ weight @ perp))
    tolerance = 1e-9 * max(1.0, abs(total), abs(tangent), abs(perpendicular), abs(cross))
    if not math.isclose(total, tangent + perpendicular + cross, rel_tol=1e-9, abs_tol=tolerance):
        raise ArithmeticError("W-energy decomposition failed")
    singular = np.linalg.svd(normal, compute_uv=False)
    condition = float(np.inf if singular.size == 0 or singular[-1] <= 0 else singular[0] / singular[-1])
    return ProjectionResult(
        coefficients, fitted, perp, total, tangent, perpendicular, cross, defect,
        int(np.linalg.matrix_rank(basis)), int(np.linalg.matrix_rank(normal)), condition,
        ridge, projection_kind)


def weighted_project(residual, J, W, *, pinv_rcond=1e-10):
    """Primary unregularized Moore-Penrose W-orthogonal projection."""
    return _project(residual, J, W, ridge_relative=0.0, pinv_rcond=pinv_rcond,
                    projection_kind="orthogonal_pinv")


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


def produce_nc_topi_scores(pair: PeakPredictionPair, J, W, *, conditioner=None,
                            iq_features=None, energy_epsilon=1e-12):
    if not isinstance(pair, PeakPredictionPair):
        raise TypeError("score construction requires PeakPredictionPair; residual-only is forbidden")
    if energy_epsilon <= 0:
        raise ValueError("energy_epsilon must be positive")
    projection = weighted_project(pair.residual_raw, J, W)
    topi = float(projection.r_perp.T @ _validate_weight(W, len(pair.residual_raw)) @ projection.r_perp)
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
        {"geometry": RAW_SPACE, "b0": STANDARDIZED_SPACE, "conditioner_target": "log_raw_perp_energy"})


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


def higher_quantile(scores, q, *, fit_roles):
    values = _array(scores, "calibration scores", 1)
    if len(values) != len(fit_roles) or not 0 < q < 1:
        raise ValueError("quantile inputs invalid")
    assert_fit_is_clean_only(fit_roles, allowed=("normal_calibration",))
    return float(np.quantile(values, q, method="higher"))


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


def build_causal_iq_context(target_source_start, block_end, block_features, *, history=4,
                            target_groups, block_groups, cadence=.5):
    targets = _array(target_source_start, "target_source_start", 1)
    ends = _array(block_end, "block_end", 1)
    features = _array(block_features, "block_features", 2)
    if len(ends) != len(features) or history < 1 or cadence <= 0:
        raise ValueError("IQ blocks/history dimensions invalid")
    target_group = np.asarray(target_groups, dtype=str)
    block_group = np.asarray(block_groups, dtype=str)
    if len(target_group) != len(targets) or len(block_group) != len(ends):
        raise ValueError("IQ group vectors must match rows")
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

    def fit(self, X, y, *, roles, feature_names=None):
        predictors = _array(X, "IQ predictors", 2)
        energy = _array(y, "S_perp target", 1)
        if len(predictors) != len(energy) or len(roles) != len(energy) or np.any(energy < 0):
            raise ValueError("conditioner fit dimensions/nonnegative energy invalid")
        assert_fit_is_clean_only(roles, allowed=("normal_train",))
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
        self.fit_manifest_ = {
            "roles": ["normal_train"],
            "rows": len(target),
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

    def calibrate_cap(self, X_calibration, *, roles, q=.995):
        values = self._uncapped_scale(X_calibration)
        if len(values) != len(roles) or not np.isclose(q, .995, rtol=0, atol=0):
            raise ValueError("calibration cap is frozen at q995")
        assert_fit_is_clean_only(roles, allowed=("normal_calibration",))
        self.cap_ = float(np.quantile(values, q, method="higher"))
        return self.cap_

    def predict_scale(self, X):
        if self.cap_ is None:
            raise RuntimeError("clean calibration cap is not set")
        return np.minimum(self._uncapped_scale(X), self.cap_)


def shuffled_control_target(target, *, roles, seed=DEFAULT_SEED):
    values = _array(target, "shuffle target", 1)
    if len(values) != len(roles):
        raise ValueError("shuffle roles length mismatch")
    assert_fit_is_clean_only(roles, allowed=("normal_train",))
    return values[np.random.default_rng(seed).permutation(len(values))]


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


def sustained_alarm_delay(availability_source_end, alarms, *, recording_ids,
                           post_eligible_mask, onset, required=3, cadence=.5,
                           stable_pre_mask=None):
    times = _array(availability_source_end, "availability source_end", 1)
    flags = np.asarray(alarms, bool)
    recordings = np.asarray(recording_ids, dtype=str)
    post = np.asarray(post_eligible_mask, bool)
    if (flags.shape != times.shape or recordings.shape != times.shape or post.shape != times.shape
            or required < 1 or cadence <= 0 or not np.isfinite(onset)):
        raise ValueError("sustained alarm inputs invalid")
    if any(not value.strip() for value in recordings):
        raise ValueError("recording_ids must be nonempty")
    pre = np.zeros(len(flags), bool) if stable_pre_mask is None else np.asarray(stable_pre_mask, bool)
    if pre.shape != flags.shape:
        raise ValueError("stable_pre_mask length mismatch")
    already = bool(np.any(flags & pre))
    run = 0
    alarm_time = math.inf
    previous_time = None
    previous_recording = None
    for time, flag, recording, eligible in zip(times, flags, recordings, post):
        contiguous = (previous_time is not None and recording == previous_recording
                      and math.isclose(time - previous_time, cadence, rel_tol=0, abs_tol=1e-8))
        if not eligible:
            run = 0
        elif flag:
            run = run + 1 if contiguous else 1
        else:
            run = 0
        previous_time, previous_recording = float(time), recording
        if eligible and run >= required:
            alarm_time = float(time)
            break
    delay = float(alarm_time - onset) if np.isfinite(alarm_time) else math.inf
    return SustainedAlarm(delay, alarm_time, already)


@dataclass(frozen=True)
class IntersectionDiagnostic:
    identities: tuple[tuple[object, ...], ...]
    scores_a: np.ndarray
    scores_b: np.ndarray
    audit: dict[str, int]


def _metadata_map(name, values, identities):
    if values is None:
        return None
    array = np.asarray(values, dtype=object)
    if len(array) != len(identities):
        raise ValueError(f"{name} must match epoch rows")
    return dict(zip(identities, array.tolist()))


def exact_primary_epoch_join(identity_a, scores_a, identity_b, scores_b, *,
                             source_intervals_a=None, source_intervals_b=None,
                             labels_a=None, labels_b=None, valid_mask_a=None, valid_mask_b=None):
    a = _array(scores_a, "scores_a", 1)
    b = _array(scores_b, "scores_b", 1)
    ids_a = _identity_list(identity_a, len(a), name="identity_a")
    ids_b = _identity_list(identity_b, len(b), name="identity_b")
    if set(ids_a) != set(ids_b):
        raise ValueError("primary join requires full identity set equality")
    map_a = dict(zip(ids_a, a))
    map_b = dict(zip(ids_b, b))
    metadata = (
        ("source interval", source_intervals_a, source_intervals_b),
        ("label", labels_a, labels_b),
        ("valid mask", valid_mask_a, valid_mask_b),
    )
    for name, left, right in metadata:
        if (left is None) != (right is None):
            raise ValueError(f"{name} metadata must be supplied for both detectors")
        left_map = _metadata_map(name, left, ids_a)
        right_map = _metadata_map(name, right, ids_b)
        if left_map is not None:
            for identity in ids_a:
                left_value, right_value = left_map[identity], right_map[identity]
                if isinstance(left_value, (list, tuple, np.ndarray)) or isinstance(right_value, (list, tuple, np.ndarray)):
                    equal = np.array_equal(np.asarray(left_value), np.asarray(right_value))
                else:
                    equal = left_value == right_value
                if not equal:
                    raise ValueError(f"{name} differs at identity {identity}")
    ordered = ids_a
    return ordered, np.asarray([map_a[x] for x in ordered]), np.asarray([map_b[x] for x in ordered])


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


def _optional_finite(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _mapping_value(mapping, key):
    return mapping.get(key) if isinstance(mapping, Mapping) else None


def evaluate_stage0_decision(*, clean_nc_fpr, clean_b0_fpr, stable_pre_fpr,
                             pauc_delta, nc_delay, b0_delay, pauc_ci_lower,
                             equal_rmse_pass, second_peak_pass,
                             actual_nc_mean_pauc_gain, shuffled_nc_mean_pauc_gain):
    missing = []
    nc_fpr = _optional_finite(clean_nc_fpr)
    b0_fpr = _optional_finite(clean_b0_fpr)
    if nc_fpr is None: missing.append("clean_nc_fpr")
    if b0_fpr is None: missing.append("clean_b0_fpr")
    c1 = nc_fpr is not None and nc_fpr <= .02
    c2 = nc_fpr is not None and b0_fpr is not None and nc_fpr - b0_fpr <= .01

    stable_known = 0
    stable_failures = 0
    for scenario in ATTACK_SCENARIOS:
        value = _optional_finite(_mapping_value(stable_pre_fpr, scenario))
        if value is None:
            missing.append(f"stable_pre_fpr:{scenario}")
        else:
            stable_known += 1
            stable_failures += int(value >= .05)
    c3 = stable_known == len(ATTACK_SCENARIOS) and stable_failures == 0

    improvement_count = 0
    improvement_known = 0
    for scenario in ATTACK_SCENARIOS:
        delta = _optional_finite(_mapping_value(pauc_delta, scenario))
        nc_raw = _mapping_value(nc_delay, scenario)
        b0_raw = _mapping_value(b0_delay, scenario)
        delay_supplied = nc_raw is not None and b0_raw is not None
        delay_pass = False
        if delay_supplied:
            try:
                nc_number, b0_number = float(nc_raw), float(b0_raw)
                delay_pass = np.isfinite(nc_number) and np.isfinite(b0_number) and b0_number - nc_number >= .5
            except (TypeError, ValueError):
                delay_supplied = False
        if delta is not None and delta > 0:
            improvement_count += 1
            improvement_known += 1
        elif delta is not None and delay_supplied:
            improvement_count += int(delay_pass)
            improvement_known += 1
        elif delta is None and delay_supplied and delay_pass:
            improvement_count += 1
            improvement_known += 1
        else:
            missing.append(f"scenario_improvement:{scenario}")
    c4 = improvement_known == len(ATTACK_SCENARIOS) and improvement_count >= 3

    positive_ci_count = 0
    ci_known = 0
    for scenario in ATTACK_SCENARIOS:
        lower = _optional_finite(_mapping_value(pauc_ci_lower, scenario))
        if lower is None:
            missing.append(f"pauc_ci_lower:{scenario}")
        else:
            ci_known += 1
            positive_ci_count += int(lower > 0)
    c5 = ci_known == len(ATTACK_SCENARIOS) and positive_ci_count >= 2

    equal_known = isinstance(equal_rmse_pass, (bool, np.bool_))
    second_known = isinstance(second_peak_pass, (bool, np.bool_))
    c6 = equal_known and bool(equal_rmse_pass)
    c7 = second_known and bool(second_peak_pass)
    if not equal_known: missing.append("equal_rmse_pass")
    if not second_known: missing.append("second_peak_pass")
    actual = _optional_finite(actual_nc_mean_pauc_gain)
    shuffled = _optional_finite(shuffled_nc_mean_pauc_gain)
    if actual is None: missing.append("actual_nc_mean_pauc_gain")
    if shuffled is None: missing.append("shuffled_nc_mean_pauc_gain")
    c8 = actual is not None and shuffled is not None and actual > shuffled and actual > 0

    criteria = {f"c{index}": value for index, value in enumerate(
        (c1, c2, c3, c4, c5, c6, c7, c8), start=1)}
    triggers = []
    if equal_known and not bool(equal_rmse_pass): triggers.append("c6_equal_rmse_false")
    if second_known and not bool(second_peak_pass): triggers.append("c7_second_peak_false")
    if nc_fpr is not None and nc_fpr > .05: triggers.append("clean_nc_fpr_gt_0.05")
    if stable_failures >= 3: triggers.append("stable_pre_failures_gte_3")
    if improvement_known == len(ATTACK_SCENARIOS) and improvement_count <= 1:
        triggers.append("improvement_count_lte_1")
    if ci_known == len(ATTACK_SCENARIOS) and positive_ci_count == 0:
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
                          tuple(sorted(set(missing))), tuple(triggers))
