"""CHORD Stage-0A clean-only complex-profile identifiability primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TAP_OFFSETS_CHIPS = np.arange(-4, 5, dtype=np.float64) * 0.125


def ca_single_source_template(tau_chips: float, offsets: np.ndarray = TAP_OFFSETS_CHIPS) -> np.ndarray:
    """Ideal one-source GPS L1 C/A triangular correlation profile."""
    return np.maximum(1.0 - np.abs(np.asarray(offsets, dtype=np.float64) - float(tau_chips)), 0.0)


def template_derivative(tau_chips: float, step: float = 1e-5) -> np.ndarray:
    return (
        ca_single_source_template(tau_chips + step)
        - ca_single_source_template(tau_chips - step)
    ) / (2.0 * step)


def complex_shrinkage_whitener(residuals: np.ndarray, shrinkage: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return a complex-linear Hermitian shrinkage covariance and inverse sqrt."""
    values = np.asarray(residuals, dtype=np.complex128)
    if values.ndim != 2 or values.shape[1] != 9 or len(values) < 10:
        raise ValueError("complex covariance requires at least ten nine-tap residuals")
    # Rows are residual observations. E[r r^H], rather than E[r* r^T], is
    # the covariance that transforms equivariantly under a common phase.
    covariance = values.T @ values.conj() / len(values)
    target = float(np.trace(covariance).real / covariance.shape[0])
    covariance = (1.0 - shrinkage) * covariance + shrinkage * target * np.eye(9)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalue)) * 1e-10, 1e-12)
    eigenvalue = np.maximum(eigenvalue, floor)
    inverse_sqrt = (eigenvector * (1.0 / np.sqrt(eigenvalue))) @ eigenvector.conj().T
    return covariance, inverse_sqrt


def weighted_complex_amplitude(
    taps: np.ndarray, template: np.ndarray, inverse_sqrt: np.ndarray
) -> complex:
    c = np.asarray(taps, dtype=np.complex128)
    a = np.asarray(template, dtype=np.complex128)
    wa = inverse_sqrt @ a
    wc = inverse_sqrt @ c
    denominator = np.vdot(wa, wa).real
    if denominator <= 1e-18:
        raise ValueError("degenerate nuisance template")
    return complex(np.vdot(wa, wc) / denominator)


@dataclass(frozen=True)
class ProfileFit:
    alpha: complex
    tau_chips: float
    model: np.ndarray
    whitened_residual: np.ndarray
    tangent_residual: np.ndarray
    residual_norm: float


def fit_tangent_residual(
    taps: np.ndarray,
    inverse_sqrt: np.ndarray,
    tau_grid: np.ndarray,
) -> ProfileFit:
    """Fit amplitude/delay and remove the local amplitude/phase/delay tangent."""
    c = np.asarray(taps, dtype=np.complex128)
    best: tuple[float, complex, float, np.ndarray] | None = None
    for tau in np.asarray(tau_grid, dtype=np.float64):
        template = ca_single_source_template(float(tau))
        alpha = weighted_complex_amplitude(c, template, inverse_sqrt)
        model = alpha * template
        residual = inverse_sqrt @ (c - model)
        objective = float(np.vdot(residual, residual).real)
        candidate = (objective, alpha, float(tau), model)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("empty delay grid")
    _, alpha, tau, model = best
    whitened = inverse_sqrt @ (c - model)
    template = ca_single_source_template(tau)
    derivative = template_derivative(tau)
    amplitude = inverse_sqrt @ template
    delay = inverse_sqrt @ (alpha * derivative)
    columns = np.column_stack(
        (
            np.r_[amplitude.real, amplitude.imag],
            np.r_[(-1j * amplitude).real, (-1j * amplitude).imag],
            np.r_[delay.real, delay.imag],
        )
    )
    q, _ = np.linalg.qr(columns)
    real_residual = np.r_[whitened.real, whitened.imag]
    projected = real_residual - q @ (q.T @ real_residual)
    tangent = projected[:9] + 1j * projected[9:]
    return ProfileFit(
        alpha=alpha,
        tau_chips=tau,
        model=model,
        whitened_residual=whitened,
        tangent_residual=tangent,
        residual_norm=float(np.linalg.norm(tangent)),
    )


def initial_template_residual(taps: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    identity = np.eye(9, dtype=np.complex128)
    return fit_tangent_residual(taps, identity, tau_grid).whitened_residual


def fingerprint(residual: np.ndarray, floor: float, epsilon: float = 1e-12) -> np.ndarray | None:
    value = np.asarray(residual, dtype=np.complex128)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < floor:
        return None
    return value / (norm + epsilon)


def projective_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.complex128)
    b = np.asarray(right, dtype=np.complex128)
    denominator = float(np.vdot(a, a).real * np.vdot(b, b).real)
    if denominator <= 1e-24:
        raise ValueError("projective similarity requires nonzero vectors")
    value = float(abs(np.vdot(a, b)) ** 2 / denominator)
    return float(np.clip(value, 0.0, 1.0))


def raw_projective_fingerprint(taps: np.ndarray, indices: Iterable[int] | None = None) -> np.ndarray:
    value = np.asarray(taps, dtype=np.complex128)
    if indices is not None:
        value = value[np.asarray(tuple(indices), dtype=int)]
    norm = float(np.linalg.norm(value))
    if norm <= 1e-18:
        raise ValueError("zero complex profile")
    return value / norm


def auc_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) != 2:
        raise ValueError("AUC requires both pair classes")
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "effect_size": float(
            (np.mean(s[y == 1]) - np.mean(s[y == 0]))
            / max(np.sqrt(0.5 * (np.var(s[y == 1], ddof=1) + np.var(s[y == 0], ddof=1))), 1e-12)
        ),
        "same_mean": float(np.mean(s[y == 1])),
        "different_mean": float(np.mean(s[y == 0])),
    }


def assign_split(timestamp_s: float) -> str:
    t = float(timestamp_s)
    if 30.0 <= t < 220.0:
        return "fit"
    if 220.0 <= t < 230.0:
        return "guard_1"
    if 230.0 <= t < 306.0:
        return "calibration"
    if 306.0 <= t < 316.0:
        return "guard_2"
    if 316.0 <= t < 430.0:
        return "holdout"
    return "outside"


def split_nonoverlap_audit(rows: list[dict[str, object]]) -> dict[str, object]:
    scientific = ("fit", "calibration", "holdout")
    ranges = {}
    blocks = {}
    for split in scientific:
        selected = [row for row in rows if row["split"] == split]
        ranges[split] = {
            "count": len(selected),
            "raw_start_min": min((int(row["raw_sample_start"]) for row in selected), default=None),
            "raw_end_max": max((int(row["raw_sample_end"]) for row in selected), default=None),
        }
        blocks[split] = {int(float(row["timestamp_s"]) // 10) for row in selected}
    raw_overlap = not (
        ranges["fit"]["raw_end_max"] < ranges["calibration"]["raw_start_min"]
        and ranges["calibration"]["raw_end_max"] < ranges["holdout"]["raw_start_min"]
    )
    block_overlap = bool(
        blocks["fit"] & blocks["calibration"]
        or blocks["fit"] & blocks["holdout"]
        or blocks["calibration"] & blocks["holdout"]
    )
    return {
        "chronological": True,
        "raw_sample_overlap": raw_overlap,
        "ten_second_block_overlap": block_overlap,
        "ranges": ranges,
    }


def select_matched_negative(
    anchor: dict[str, object],
    target_candidates: list[dict[str, object]],
    positive_target: dict[str, object],
    cn0_scale: float,
    norm_scale: float,
    use_counts: dict[int, int],
) -> dict[str, object]:
    candidates = [row for row in target_candidates if int(row["prn"]) != int(anchor["prn"])]
    if not candidates:
        raise ValueError("no different-PRN candidate")
    positive_cn0 = abs(float(anchor["cn0_db_hz"]) - float(positive_target["cn0_db_hz"]))
    positive_norm = abs(np.log(float(anchor["residual_norm"])) - np.log(float(positive_target["residual_norm"])))
    ranked = []
    for row in candidates:
        cn0 = abs(float(anchor["cn0_db_hz"]) - float(row["cn0_db_hz"]))
        norm = abs(np.log(float(anchor["residual_norm"])) - np.log(float(row["residual_norm"])))
        objective = (
            abs(cn0 - positive_cn0) / max(cn0_scale, 1e-6)
            + abs(norm - positive_norm) / max(norm_scale, 1e-6)
            + 0.01 * use_counts.get(int(row["prn"]), 0)
        )
        ranked.append((objective, use_counts.get(int(row["prn"]), 0), int(row["prn"]), row))
    selected = min(ranked, key=lambda item: item[:3])[3]
    use_counts[int(selected["prn"])] = use_counts.get(int(selected["prn"]), 0) + 1
    return selected


def block_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    block_ids: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    blocks = np.asarray(block_ids, dtype=np.int64)
    unique = np.unique(blocks)
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(resamples):
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(blocks == block) for block in sampled])
        if len(np.unique(y[index])) == 2:
            output.append(roc_auc_score(y[index], s[index]))
    return np.asarray(output, dtype=np.float64)


def paired_block_bootstrap_difference(
    labels: np.ndarray,
    full_scores: np.ndarray,
    baseline_scores: np.ndarray,
    block_ids: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int8)
    full = np.asarray(full_scores, dtype=np.float64)
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    blocks = np.asarray(block_ids, dtype=np.int64)
    unique = np.unique(blocks)
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(resamples):
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(blocks == block) for block in sampled])
        if len(np.unique(y[index])) == 2:
            output.append(roc_auc_score(y[index], full[index]) - roc_auc_score(y[index], baseline[index]))
    return np.asarray(output, dtype=np.float64)
