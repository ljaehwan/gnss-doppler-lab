"""R2C-GNSS Stage-0 code-delay mechanics.

This module deliberately contains no TEXBAT labels or scenario-specific tuning.
Synthetic inputs are suitable only for unit/physical controls, never evidence of
real attack performance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

C_M_S = 299_792_458.0
GPS_CA_CHIP_RATE_HZ = 1_023_000.0


@dataclass(frozen=True)
class ComplexTapProvenance:
    source_path: str
    source_sha256: str
    receiver: str
    extractor: str
    sample_rate_hz: float
    tap_spacing_chips: float
    recording_id: str
    genuinely_complex: bool
    preprocessing: str


@dataclass(frozen=True)
class SourceSupport:
    source_start_s: float
    source_end_s: float
    recording_id: str


@dataclass(frozen=True)
class HypothesisFit:
    delays_chips: tuple[float, ...]
    amplitudes: tuple[complex, ...]
    prediction: np.ndarray
    residual: np.ndarray
    weighted_rss: float
    log_likelihood_profile: float
    boundary: bool
    identifiable: bool


@dataclass(frozen=True)
class SecondSourceFit:
    h0: HypothesisFit
    h1: HypothesisFit
    score: float


@dataclass(frozen=True)
class GeometryFit:
    beta_m: np.ndarray
    prediction_m: np.ndarray
    residual_m: np.ndarray
    robust_weights: np.ndarray
    leverage: np.ndarray
    rank: int
    residual_dof: int
    singular_values: np.ndarray
    condition_number: float
    valid: bool
    reason: str | None
    shared_improvement: float


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_complex_taps(values: np.ndarray, provenance: ComplexTapProvenance) -> np.ndarray:
    """Reject magnitude/phase-stripped or untraceable primary evidence."""
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != 9:
        raise ValueError("R2C primary input must have exactly nine taps")
    if not np.iscomplexobj(array) or not provenance.genuinely_complex:
        raise ValueError("R2C primary input must contain genuine complex I/Q at every tap")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError("complex taps contain non-finite values")
    required = (provenance.source_path, provenance.source_sha256, provenance.receiver,
                provenance.extractor, provenance.recording_id, provenance.preprocessing)
    if any(not str(value).strip() for value in required):
        raise ValueError("complex-tap provenance is incomplete")
    if len(provenance.source_sha256) != 64:
        raise ValueError("source_sha256 must be a SHA-256 hex digest")
    try:
        int(provenance.source_sha256, 16)
    except ValueError as exc:
        raise ValueError("source_sha256 must be hexadecimal") from exc
    if provenance.sample_rate_hz <= 0 or provenance.tap_spacing_chips <= 0:
        raise ValueError("sample rate and tap spacing must be positive")
    return array.astype(np.complex128, copy=False)


def assign_normal_split(support: SourceSupport) -> str:
    """Frozen source-support-aware chronological cleanStatic split."""
    start, end = support.source_start_s, support.source_end_s
    if not np.isfinite([start, end]).all() or end < start:
        raise ValueError("invalid source support")
    if end <= 300.0:
        return "normal_train"
    if start >= 320.0 and end <= 400.0:
        return "normal_calibration"
    if start >= 420.0:
        return "normal_holdout"
    return "excluded_guard_or_boundary"


def assign_attack_phase(support: SourceSupport, onset_s: float) -> str:
    start, end = support.source_start_s, support.source_end_s
    if end < start:
        raise ValueError("invalid source support")
    if start >= 30.0 and end <= onset_s - 20.0:
        return "stable_pre"
    if start >= onset_s + 40.0:
        return "persistent"
    if start >= onset_s:
        return "post"
    return "transition_excluded"


def availability_time(supports: Iterable[SourceSupport]) -> float:
    values = [item.source_end_s for item in supports]
    if not values:
        raise ValueError("at least one support is required")
    return float(max(values))


def gps_ca_correlation(offset_chips: np.ndarray | float) -> np.ndarray:
    """Ideal GPS L1 C/A coherent code autocorrelation main lobe.

    Stage-0 confines searches to the supported nine-tap main-lobe span. A real
    run must document that its receiver uses this conventional one-chip C/A
    autocorrelation template and the configured tap spacing.
    """
    offset = np.asarray(offset_chips, dtype=np.float64)
    return np.maximum(1.0 - np.abs(offset), 0.0)


def correlation_template(tap_offsets_chips: Sequence[float], delay_chips: float) -> np.ndarray:
    return gps_ca_correlation(np.asarray(tap_offsets_chips, dtype=np.float64) - float(delay_chips))


def empirical_template_hash(template: Sequence[complex]) -> str:
    """Canonical hash for a frozen complex receiver template."""
    value = np.asarray(template, dtype=np.complex128)
    if value.shape != (9,) or not np.all(np.isfinite(value)):
        raise ValueError("empirical template must be a finite complex nine-tap vector")
    packed = np.column_stack((value.real, value.imag)).astype("<f8", copy=False)
    return hashlib.sha256(packed.tobytes(order="C")).hexdigest()


def build_empirical_template(values: np.ndarray, roles: Sequence[str]) -> tuple[np.ndarray, dict]:
    """Robustly freeze a phase/amplitude-aligned template from normal train only.

    Tracking centers the receiver correlation at prompt (tap four).  Each row is
    rotated by prompt phase and divided by prompt amplitude before componentwise
    medians are taken.  This consumes neither recording identity nor attack data.
    """
    if not roles or set(roles) != {"normal_train"}:
        raise ValueError("empirical template fitting is restricted to cleanStatic normal_train")
    rows = np.asarray(values, dtype=np.complex128)
    if rows.ndim != 2 or rows.shape[1] != 9 or rows.shape[0] != len(roles):
        raise ValueError("nine-tap rows and roles must align")
    prompt = rows[:, 4]
    floor = max(float(np.median(np.abs(prompt))) * 1e-6, np.finfo(float).tiny)
    keep = np.isfinite(rows).all(axis=1) & (np.abs(prompt) > floor)
    if int(keep.sum()) < 10:
        raise ValueError("insufficient finite prompt support for empirical template")
    aligned = rows[keep] * np.exp(-1j * np.angle(prompt[keep]))[:, None] / np.abs(prompt[keep])[:, None]
    template = np.median(aligned.real, axis=0) + 1j * np.median(aligned.imag, axis=0)
    template /= template[4]
    metadata = {
        "method": "prompt-phase rotation; prompt-amplitude normalization; componentwise real/imag median",
        "fit_role": "cleanStatic normal_train",
        "input_rows": int(rows.shape[0]),
        "accepted_rows": int(keep.sum()),
        "rejected_rows": int((~keep).sum()),
        "hash_algorithm": "sha256(canonical little-endian float64 [real,imag])",
        "template_sha256": empirical_template_hash(template),
    }
    return template, metadata


def shifted_empirical_template(tap_offsets_chips: Sequence[float], delay_chips: float,
                               template: Sequence[complex]) -> np.ndarray:
    taps = np.asarray(tap_offsets_chips, dtype=np.float64)
    base = np.asarray(template, dtype=np.complex128)
    if taps.shape != (9,) or base.shape != (9,):
        raise ValueError("empirical template and offsets must have nine taps")
    at = taps - float(delay_chips)
    return (np.interp(at, taps, base.real, left=0.0, right=0.0) +
            1j * np.interp(at, taps, base.imag, left=0.0, right=0.0))


def _whitener(covariance: np.ndarray | None, size: int, eigen_floor: float = 1e-8) -> np.ndarray:
    covariance = np.eye(size) if covariance is None else np.asarray(covariance, dtype=np.complex128)
    if covariance.shape != (size, size) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be a finite square tap covariance")
    values, vectors = np.linalg.eigh((covariance + covariance.conj().T) / 2)
    floor = max(float(np.max(values)) * eigen_floor, eigen_floor)
    return (vectors * (1.0 / np.sqrt(np.maximum(values, floor)))) @ vectors.conj().T


def _linear_fit(y: np.ndarray, columns: Sequence[np.ndarray], whitener: np.ndarray,
                delays: tuple[float, ...], boundary: bool, min_separation: float) -> HypothesisFit:
    design = np.column_stack(columns).astype(np.complex128)
    wd, wy = whitener @ design, whitener @ y
    amplitudes, _, rank, singular = np.linalg.lstsq(wd, wy, rcond=None)
    prediction = design @ amplitudes
    residual = y - prediction
    rss = float(np.vdot(whitener @ residual, whitener @ residual).real)
    identifiable = bool(rank == len(columns) and
                        (len(delays) == 1 or abs(delays[0] - delays[1]) >= min_separation) and
                        (len(singular) < 2 or singular[-1] > singular[0] * 1e-8))
    # Profile likelihood removes the unknown common noise scale. It is invariant
    # to global complex phase and positive scalar gain.
    observed_power = float(np.vdot(wy, wy).real)
    ll = -float(y.size) * np.log(max(rss / max(observed_power, np.finfo(float).tiny),
                                         np.finfo(float).tiny))
    return HypothesisFit(delays, tuple(complex(x) for x in amplitudes), prediction,
                         residual, rss, ll, boundary, identifiable)


def fit_second_source(y: Sequence[complex], tap_offsets_chips: Sequence[float],
                      delay_grid_chips: Sequence[float], *, covariance: np.ndarray | None = None,
                      minimum_separation_chips: float = 0.10,
                      template_values: Sequence[complex] | None = None,
                      template_kind: str | None = None) -> SecondSourceFit:
    """Grid/profile-ML H0/H1 fit with signed delays and no extrapolation."""
    observation = np.asarray(y, dtype=np.complex128)
    taps = np.asarray(tap_offsets_chips, dtype=np.float64)
    grid = np.unique(np.asarray(delay_grid_chips, dtype=np.float64))
    if observation.shape != taps.shape or observation.ndim != 1 or observation.size != 9:
        raise ValueError("y and tap offsets must be matching nine-element vectors")
    if grid.size < 2 or grid[0] < taps[0] or grid[-1] > taps[-1] or not np.any(grid < 0) or not np.any(grid > 0):
        raise ValueError("delay grid must search signed offsets inside measured tap support")
    whitener = _whitener(covariance, observation.size)
    if template_kind == "empirical_receiver":
        if template_values is None:
            raise ValueError("empirical_receiver scoring requires frozen template values")
        empirical_template_hash(template_values)
        templates = {float(delay): shifted_empirical_template(taps, float(delay), template_values)
                     for delay in grid}
    elif template_kind == "synthetic_ideal" and template_values is None:
        templates = {float(delay): correlation_template(taps, float(delay)) for delay in grid}
    else:
        raise ValueError("template_kind must explicitly be empirical_receiver or synthetic_ideal")
    h0_candidates = [_linear_fit(observation, [templates[float(d)]], whitener,
                                 (float(d),), bool(d in (grid[0], grid[-1])), minimum_separation_chips)
                     for d in grid]
    h0 = min(h0_candidates, key=lambda fit: fit.weighted_rss)
    pairs: list[HypothesisFit] = []
    for index, first in enumerate(grid):
        for second in grid[index + 1:]:
            if abs(float(second - first)) < minimum_separation_chips:
                continue
            pairs.append(_linear_fit(observation, [templates[float(first)], templates[float(second)]],
                                     whitener, (float(first), float(second)),
                                     bool(first == grid[0] or second == grid[-1]), minimum_separation_chips))
    if not pairs:
        raise ValueError("delay grid contains no identifiable two-source pair")
    h1 = min(pairs, key=lambda fit: fit.weighted_rss if fit.identifiable else np.inf)
    # A scale-relative numerical floor avoids phase-dependent roundoff when a
    # synthetic vector lies exactly in H1's column space, without breaking gain
    # invariance.
    weighted_power = float(np.vdot(whitener @ observation, whitener @ observation).real)
    floor = max(weighted_power * 1e-14, np.finfo(float).tiny)
    score = max(0.0, 2.0 * observation.size * np.log((h0.weighted_rss + floor) /
                                                     (h1.weighted_rss + floor)))
    return SecondSourceFit(h0, h1, float(score))


class AnalyticResidualWhitener:
    def __init__(self, shrinkage: float = 0.2, epsilon: float = 1e-8):
        self.shrinkage, self.epsilon = float(shrinkage), float(epsilon)
        self.mean_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.fit_roles_: tuple[str, ...] = ()

    def fit(self, residuals: np.ndarray, roles: Sequence[str]) -> "AnalyticResidualWhitener":
        if not roles or set(roles) != {"normal_train"}:
            raise ValueError("analytic nuisance fitting is restricted to cleanStatic normal_train")
        data = np.asarray(residuals, dtype=np.complex128)
        if data.ndim != 2 or data.shape[1] != 9 or data.shape[0] < 2:
            raise ValueError("at least two nine-tap residuals are required")
        self.mean_ = data.mean(axis=0)
        centered = data - self.mean_
        # Rows are observations and columns are complex variables.  The proper
        # complex covariance is E[(z-mu)(z-mu)^H].  Using z^H z conjugates the
        # imaginary cross-covariance and whitens the wrong orientation.
        empirical = centered.T @ centered.conj() / max(data.shape[0] - 1, 1)
        diagonal = np.diag(np.diag(empirical))
        self.covariance_ = ((1 - self.shrinkage) * empirical + self.shrinkage * diagonal +
                            self.epsilon * np.eye(9))
        self.fit_roles_ = tuple(roles)
        return self


class SmallNeuralNuisanceModel:
    """Deterministic compact MLP predicting normal residual mean only.

    Predictive uncertainty is the diagonal variance of normal-train errors.
    Conditions are numeric causal receiver-quality features; callers cannot pass
    identities or labels through this numeric-only interface.
    """
    def __init__(self, hidden: int = 8, seed: int = 20260803):
        self.hidden, self.seed = int(hidden), int(seed)
        self.parameters_: tuple[np.ndarray, ...] | None = None
        self.variance_: np.ndarray | None = None
        self.fit_roles_: tuple[str, ...] = ()

    def fit(self, conditions: np.ndarray, residuals: np.ndarray, roles: Sequence[str], *, epochs: int = 100,
            learning_rate: float = 0.01) -> "SmallNeuralNuisanceModel":
        if not roles or set(roles) != {"normal_train"}:
            raise ValueError("neural nuisance fitting is restricted to cleanStatic normal_train")
        x = np.asarray(conditions, dtype=np.float64)
        y_complex = np.asarray(residuals, dtype=np.complex128)
        if x.ndim != 2 or y_complex.ndim != 2 or y_complex.shape != (x.shape[0], 9):
            raise ValueError("conditions and nine-tap residual rows must align")
        y = np.concatenate([y_complex.real, y_complex.imag], axis=1)
        rng = np.random.default_rng(self.seed)
        w1 = rng.normal(scale=0.1, size=(x.shape[1], self.hidden)); b1 = np.zeros(self.hidden)
        w2 = rng.normal(scale=0.1, size=(self.hidden, 18)); b2 = np.zeros(18)
        for _ in range(int(epochs)):
            hidden = np.tanh(x @ w1 + b1); prediction = hidden @ w2 + b2
            error = (prediction - y) / x.shape[0]
            gw2 = hidden.T @ error; gb2 = error.sum(axis=0)
            dh = (error @ w2.T) * (1 - hidden * hidden)
            gw1 = x.T @ dh; gb1 = dh.sum(axis=0)
            w1 -= learning_rate * gw1; b1 -= learning_rate * gb1
            w2 -= learning_rate * gw2; b2 -= learning_rate * gb2
        error = y - (np.tanh(x @ w1 + b1) @ w2 + b2)
        self.parameters_ = (w1, b1, w2, b2)
        self.variance_ = np.maximum(error.var(axis=0), 1e-8)
        self.fit_roles_ = tuple(roles)
        return self

    def predict(self, conditions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.parameters_ is None or self.variance_ is None:
            raise RuntimeError("model is not fitted")
        w1, b1, w2, b2 = self.parameters_
        value = np.tanh(np.asarray(conditions) @ w1 + b1) @ w2 + b2
        mean = value[:, :9] + 1j * value[:, 9:]
        variance = self.variance_[:9] + self.variance_[9:]
        return mean, np.broadcast_to(variance, mean.shape)


def fit_shared_constellation(delay_offsets_s: Sequence[float], los_vectors: np.ndarray,
                             evidence_weights: Sequence[float] | None = None, *, minimum_prns: int = 5,
                             maximum_condition_number: float = 1e6, huber_delta_m: float = 10.0) -> GeometryFit:
    delays = np.asarray(delay_offsets_s, dtype=np.float64)
    los = np.asarray(los_vectors, dtype=np.float64)
    if los.shape != (delays.size, 3) or not np.all(np.isfinite(los)):
        raise ValueError("actual finite N-by-3 LOS vectors must align with delays")
    ranges = C_M_S * delays
    design = np.column_stack([-los, np.ones(delays.size)])
    base = np.ones(delays.size) if evidence_weights is None else np.asarray(evidence_weights, dtype=np.float64)
    if base.shape != delays.shape or np.any(base <= 0):
        raise ValueError("evidence weights must be positive and aligned")
    rank = int(np.linalg.matrix_rank(design))
    residual_dof = int(delays.size - rank)
    singular = np.linalg.svd(design, compute_uv=False)
    condition = float(np.inf if singular[-1] == 0 else singular[0] / singular[-1])
    valid = (delays.size >= minimum_prns and rank == 4 and residual_dof >= 1 and
             condition <= maximum_condition_number)
    reason = None
    if delays.size < minimum_prns: reason = "insufficient_prns"
    elif rank < 4: reason = "rank_deficient"
    elif residual_dof < 1: reason = "zero_residual_degrees_of_freedom"
    elif condition > maximum_condition_number: reason = "ill_conditioned"
    weights = base.copy(); beta = np.zeros(4)
    if valid:
        for _ in range(8):
            root = np.sqrt(weights)
            beta = np.linalg.lstsq(design * root[:, None], ranges * root, rcond=None)[0]
            residual = ranges - design @ beta
            robust = np.minimum(1.0, huber_delta_m / np.maximum(np.abs(residual), 1e-12))
            weights = base * robust
    prediction = design @ beta
    residual = ranges - prediction
    if valid:
        normal = design.T @ (weights[:, None] * design)
        hat = (design @ np.linalg.pinv(normal) @ design.T) * weights[None, :]
        leverage = np.clip(np.diag(hat), 0, 1)
        null_mean = np.average(ranges, weights=base)
        null_rss = float(np.sum(base * (ranges - null_mean) ** 2))
        shared_rss = float(np.sum(weights * residual ** 2))
        improvement = max(0.0, null_rss - shared_rss)
    else:
        leverage = np.zeros(delays.size); improvement = 0.0
    return GeometryFit(beta, prediction, residual, weights, leverage, rank, residual_dof, singular,
                       condition, valid, reason, improvement)


def aggregate_a2(scores: Sequence[float], top_k: int = 4) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("finite per-PRN scores are required")
    chosen = np.sort(values)[-min(int(top_k), values.size):]
    return float(np.mean(chosen))


def full_score(per_prn_scores: Sequence[float], geometry: GeometryFit) -> float:
    """Noise-normalized evidence plus geometry fit; invalid geometry is never evidence."""
    if not geometry.valid:
        return 0.0
    evidence = aggregate_a2(per_prn_scores)
    residual_scale = float(np.mean(geometry.residual_m ** 2))
    support = float(np.sum(geometry.robust_weights))
    conditioning = 1.0 / max(np.log10(max(geometry.condition_number, 1.0)) + 1.0, 1.0)
    return float(evidence * support * conditioning / (1.0 + residual_scale))


def quantile_threshold(scores: Sequence[float], q: float, roles: Sequence[str]) -> float:
    if not roles or set(roles) != {"normal_calibration"}:
        raise ValueError("thresholds require cleanStatic normal_calibration only")
    values = np.asarray(scores, dtype=np.float64)
    if values.size != len(roles) or not np.all(np.isfinite(values)):
        raise ValueError("finite scores and roles must align")
    return float(np.quantile(values, q, method="higher"))


def strict_alarm(score: float, threshold: float) -> bool:
    return bool(score > threshold)


def derive_stage0_verdict(gates: Mapping[str, Mapping[str, object]]) -> dict:
    """Derive the frozen task taxonomy from explicit machine-readable gates."""
    required_inputs = ("complex_provenance", "time_alignment", "los_geometry", "b0_interface")
    evidence = ("clean_dynamic_fpr", "gain_invariance", "phase_invariance",
                "full_exceeds_b0", "full_b0_ci", "geometry_improvement",
                "relation_destruction", "shortcut_controls")
    ordered = required_inputs + evidence
    unknown = sorted(set(gates).difference(ordered))
    if unknown:
        raise ValueError(f"unknown Stage-0 gates: {unknown}")
    statuses = {}
    for name in ordered:
        status = gates.get(name, {}).get("status", "NOT_EVALUATED")
        if status not in {"PASS", "FAIL", "NOT_EVALUATED"}:
            raise ValueError(f"invalid Stage-0 gate status for {name}: {status}")
        statuses[name] = status
    missing = [name for name in required_inputs if statuses[name] != "PASS"]
    if missing:
        verdict = "DATA_INVALID"
    else:
        failed = [name for name in evidence if statuses[name] != "PASS"]
        verdict = "PHYSICS_SUPPORTED" if not failed else "NOT_SUPPORTED"
    reason = ", ".join(f"{name}={statuses[name]}" for name in ordered)
    return {"verdict": verdict, "reason": reason, "gates": dict(gates),
            "physics_supported": verdict == "PHYSICS_SUPPORTED"}


def sustained_alarms(alarms: Sequence[bool], recording_ids: Sequence[str], times_s: Sequence[float],
                     phases: Sequence[str], cadence_s: float = 0.5) -> np.ndarray:
    n = len(alarms)
    if not (len(recording_ids) == len(times_s) == len(phases) == n):
        raise ValueError("alarm metadata must align")
    output = np.zeros(n, dtype=bool); run = 0
    for i in range(n):
        boundary = (i == 0 or recording_ids[i] != recording_ids[i-1] or
                    times_s[i] - times_s[i-1] > cadence_s * 1.01 or
                    phases[i] != phases[i-1] or "transition" in phases[i])
        run = 0 if boundary else run
        run = run + 1 if alarms[i] else 0
        if run >= 3: output[i] = True
    return output


def inject_second_source(y: Sequence[complex], tap_offsets_chips: Sequence[float], delay_chips: float,
                         power_ratio: float, phase_rad: float) -> np.ndarray:
    """Add an ideal component with energy relative to the authentic tap vector.

    ``power_ratio`` is ||injected||^2 / ||authentic||^2 over the nine measured
    taps.  The template is energy-normalized after applying the requested delay,
    so the convention remains truthful for every supported delay.
    """
    if power_ratio < 0:
        raise ValueError("power ratio must be non-negative")
    authentic = np.asarray(y, dtype=np.complex128)
    template = correlation_template(tap_offsets_chips, delay_chips).astype(np.complex128)
    authentic_energy = float(np.vdot(authentic, authentic).real)
    template_energy = float(np.vdot(template, template).real)
    if authentic_energy <= 0 or template_energy <= 0:
        raise ValueError("authentic vector and delayed template must have positive energy")
    amplitude = np.sqrt(power_ratio * authentic_energy / template_energy)
    return authentic + amplitude * np.exp(1j * phase_rad) * template


def artifact_hashes(root: str | Path, *, exclude: Sequence[str] = ("hashes.json", "verification.json")) -> dict[str, str]:
    base = Path(root)
    return {str(path.relative_to(base)): sha256_file(path) for path in sorted(base.rglob("*"))
            if path.is_file() and path.name not in set(exclude)}


def write_json(path: str | Path, value: Mapping | Sequence) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
