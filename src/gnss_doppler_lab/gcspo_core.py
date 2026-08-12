"""Frozen GCSPO Stage-0 scientific and access-control primitives."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf

EPOCH_S = 0.02
MINIMUM_PRNS = 4
PSEUDOINVERSE_RCOND = 1e-10
CODE_CHIP_M = 299_792_458.0 / 1_023_000.0
L1_WAVELENGTH_M = 299_792_458.0 / 1_575_420_000.0


def _array(value, name, ndim=None):
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class Role:
    name: str
    start_s: float
    end_s: float

    def __post_init__(self):
        if not self.name or not math.isfinite(self.start_s) or not math.isfinite(self.end_s) or self.start_s >= self.end_s:
            raise ValueError("invalid role")


def validate_role_disjointness(roles: Iterable[Role]):
    ordered = tuple(sorted(roles, key=lambda r: (r.start_s, r.end_s, r.name)))
    if len({r.name for r in ordered}) != len(ordered):
        raise ValueError("role names must be unique")
    if any(a.end_s > b.start_s for a, b in zip(ordered, ordered[1:])):
        raise ValueError("roles overlap")
    return ordered


def role_for_interval(start_s, end_s, roles):
    start, end = float(start_s), float(end_s)
    if not math.isfinite(start) or not math.isfinite(end) or start >= end:
        raise ValueError("invalid half-open interval")
    found = [r for r in validate_role_disjointness(roles) if start >= r.start_s and end <= r.end_s]
    if len(found) > 1:
        raise ValueError("roles overlap")
    return found[0] if found else None


def aggregate_20ms(rows, *, epoch_s=EPOCH_S):
    """Per-PRN componentwise medians in anchored [start,end) bins."""
    if not math.isfinite(epoch_s) or epoch_s <= 0:
        raise ValueError("epoch_s must be positive")
    seen, groups, samples = set(), {}, {}
    for row in rows:
        t, sample = float(row["time_s"]), int(row["sample_count"])
        prn = int(str(row["prn"]).lstrip("Gg")); channel = int(row.get("channel", -1)); segment = int(row.get("segment_index", 0))
        q = _array(row["q"], "q", 1)
        if t < 0 or not math.isfinite(t):
            raise ValueError("invalid time_s")
        identity = (t, sample, prn, channel, segment, q.tobytes())
        if identity in seen:
            raise ValueError("exact duplicate scientific row")
        seen.add(identity)
        key = (int(math.floor(t / epoch_s)), prn, channel, segment)
        groups.setdefault(key, []).append(q)
        samples.setdefault(key, []).append(sample)
    result = []
    for (epoch, prn, channel, segment), values in sorted(groups.items()):
        if len({len(v) for v in values}) != 1:
            raise ValueError("inconsistent q width")
        result.append({"epoch": epoch, "prn": prn, "channel": channel, "segment_index": segment, "start_s": epoch * epoch_s,
                       "availability_s": (epoch + 1) * epoch_s,
                       "sample_count_min": min(samples[(epoch, prn, channel, segment)]),
                       "sample_count_max": max(samples[(epoch, prn, channel, segment)]),
                       "q": np.median(np.vstack(values), axis=0)})
    return result


@dataclass(frozen=True)
class SharedVAR:
    intercept: np.ndarray
    coefficients: np.ndarray

    @property
    def lags(self):
        return int(self.coefficients.shape[0])

    @classmethod
    def fit(cls, histories, targets, *, ridge):
        history, target = _array(histories, "histories", 3), _array(targets, "targets", 2)
        if history.shape[0] != target.shape[0] or history.shape[2] != target.shape[1] or not len(history):
            raise ValueError("history/target shape mismatch")
        if not math.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be nonnegative")
        design = np.column_stack([np.ones(len(history)), history.reshape(len(history), -1)])
        penalty = np.eye(design.shape[1]) * ridge
        penalty[0, 0] = 0
        beta = np.linalg.pinv(design.T @ design + penalty, rcond=PSEUDOINVERSE_RCOND) @ design.T @ target
        q = target.shape[1]
        coefficients = beta[1:].reshape(history.shape[1], q, q).transpose(0, 2, 1)
        return cls(beta[0], coefficients)

    def predict(self, history):
        values = _array(history, "history", 2)
        if values.shape != (self.lags, len(self.intercept)):
            raise ValueError("history must contain exactly the frozen lag count")
        return self.intercept + np.einsum("lij,lj->i", self.coefficients, values)

    def residuals(self, histories, targets):
        history, target = _array(histories, "histories", 3), _array(targets, "targets", 2)
        return target - self.intercept - np.einsum("lij,nlj->ni", self.coefficients, history)


def _huber_location(values, tuning=1.345):
    location = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - location), axis=0)
    scale = np.where(scale > 0, scale, 1)
    for _ in range(100):
        u = (values - location) / scale
        weights = np.minimum(1, tuning / np.maximum(np.abs(u), np.finfo(float).tiny))
        updated = np.sum(weights * values, axis=0) / np.sum(weights, axis=0)
        if np.allclose(updated, location, rtol=0, atol=1e-12):
            break
        location = updated
    return updated


def _psd_inverse_sqrt(covariance):
    symmetric = (covariance + covariance.T) / 2
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    largest = float(eigenvalues[-1])
    if not math.isfinite(largest) or largest < 0:
        raise ValueError("invalid covariance largest eigenvalue")
    floor = max(1e-8 * largest, 1e-8 if largest == 0 else 0)
    eigenvalues = np.maximum(eigenvalues, floor)
    psd = (vectors * eigenvalues) @ vectors.T
    inv = (vectors * eigenvalues ** -0.5) @ vectors.T
    return (psd + psd.T) / 2, (inv + inv.T) / 2


@dataclass(frozen=True)
class Whitener:
    location: np.ndarray
    covariance: np.ndarray
    inverse_sqrt: np.ndarray

    def transform(self, residuals):
        return np.einsum("ij,...j->...i", self.inverse_sqrt, _array(residuals, "residuals") - self.location)


def fit_whitener(train_residuals):
    values = _array(train_residuals, "train residuals", 2)
    if len(values) < 2:
        raise ValueError("at least two train residuals are required")
    location = _huber_location(values)
    covariance = LedoitWolf(assume_centered=True).fit(values - location).covariance_
    covariance, inv = _psd_inverse_sqrt(covariance)
    return Whitener(location, covariance, inv)


def fit_common_gamma(train_z_by_epoch):
    medians = []
    for epoch in train_z_by_epoch:
        z = _array(epoch, "epoch z", 2)
        if len(z) >= MINIMUM_PRNS:
            medians.append(np.median(z, axis=0))
    if len(medians) < 2:
        raise ValueError("at least two eligible train epochs are required")
    matrix = np.vstack(medians)
    centered = matrix - matrix.mean(axis=0)
    gamma = LedoitWolf(assume_centered=True).fit(centered).covariance_
    return _psd_inverse_sqrt(gamma)[0]


def common_epoch_covariance(gamma, *, prn_count):
    value = _array(gamma, "Gamma", 2)
    if value.shape[0] != value.shape[1] or prn_count < MINIMUM_PRNS:
        raise ValueError("invalid Gamma or PRN count")
    return np.eye(prn_count * len(value)) + np.kron(np.ones((prn_count, prn_count)), value)


def geometry_observability(los_ecef, *, maximum_condition=10_000.0):
    los = _array(los_ecef, "LOS", 2)
    if los.shape[1:] != (3,) or len(los) < MINIMUM_PRNS:
        return {"available": False, "rank": min(len(los), 3), "condition_number": math.inf}
    if np.any(np.linalg.norm(los, axis=1) <= 0) or not np.allclose(np.linalg.norm(los, axis=1), 1, atol=1e-8):
        raise ValueError("LOS rows must be unit vectors")
    singular = np.linalg.svd(np.column_stack([-los, np.ones(len(los))]), full_matrices=False, compute_uv=False)
    largest = float(singular[0])
    rank = int(np.sum(singular > 1e-10 * largest)) if largest > 0 and math.isfinite(largest) else 0
    fourth = float(singular[3]) if len(singular) >= 4 else 0
    condition = largest / fourth if fourth > 0 and math.isfinite(fourth) else math.inf
    return {"available": rank == 4 and condition <= maximum_condition, "rank": rank, "condition_number": condition}


def build_physical_loading(los_ecef, *, validated_rows):
    los = _array(los_ecef, "LOS", 1)
    if los.shape != (3,) or not np.isclose(np.linalg.norm(los), 1, atol=1e-8):
        raise ValueError("LOS must be a unit 3-vector")
    loading = np.zeros((10, 8))
    range_row = np.r_[-los, 1, np.zeros(4)]
    rate_row = np.r_[np.zeros(4), -los, 1]
    if "code_error_chips" in validated_rows: loading[6] = -range_row / CODE_CHIP_M
    if "pll_phase_error_cycles" in validated_rows: loading[7] = -range_row / L1_WAVELENGTH_M
    if "carrier_doppler_hz" in validated_rows: loading[8] = -rate_row / L1_WAVELENGTH_M
    if "code_frequency_offset_chips_s" in validated_rows: loading[9] = -rate_row / CODE_CHIP_M
    return loading


def apply_var_transfer(direct_design, var_coefficients):
    direct, coefficients = _array(direct_design, "direct design", 3), _array(var_coefficients, "VAR coefficients", 3)
    if coefficients.shape[1:] != (direct.shape[1], direct.shape[1]):
        raise ValueError("VAR/direct design shape mismatch")
    output = direct.copy()
    for t in range(len(direct)):
        for lag, coefficient in enumerate(coefficients, start=1):
            if t >= lag:
                output[t] -= coefficient @ direct[t - lag]
    return output


def build_state_prior_precision(*, epoch_count, smoothness, dt_s=EPOCH_S):
    if isinstance(epoch_count, bool) or epoch_count < 1 or not math.isfinite(smoothness) or smoothness <= 0:
        raise ValueError("invalid prior parameters")
    scales = np.diag([10, 10, 10, 10, 1, 1, 1, 1.0])
    f_phys = np.eye(8)
    for position, velocity in ((0, 4), (1, 5), (2, 6), (3, 7)):
        f_phys[position, velocity] = dt_s
    f_norm = np.linalg.solve(scales, f_phys @ scales)
    operator = np.zeros((epoch_count * 8, epoch_count * 8))
    operator[:8, :8] = np.eye(8)
    for epoch in range(1, epoch_count):
        current, previous = slice(epoch * 8, (epoch + 1) * 8), slice((epoch - 1) * 8, epoch * 8)
        operator[current, current], operator[current, previous] = np.eye(8), -f_norm
    scale_block = np.kron(np.eye(epoch_count), np.linalg.inv(scales))
    return scale_block.T @ (smoothness * operator.T @ operator) @ scale_block


def map_edf_score(y, design, precision):
    observations, g, r = _array(y, "y", 1), _array(design, "G", 2), _array(precision, "R", 2)
    if g.shape[0] != len(observations) or r.shape != (g.shape[1], g.shape[1]) or not len(observations):
        raise ValueError("MAP shape mismatch")
    inverse = np.linalg.pinv(g.T @ g + r, rcond=PSEUDOINVERSE_RCOND)
    state = inverse @ g.T @ observations
    residual = observations - g @ state
    improvement = float(observations @ observations - residual @ residual)
    edf = float(np.trace(g @ inverse @ g.T))
    rank = int(np.linalg.matrix_rank(g, tol=PSEUDOINVERSE_RCOND))
    if edf < -1e-8 or edf > rank + 1e-8 or rank > len(observations):
        raise ValueError("influence trace bounds failed")
    edf = min(max(edf, 0), float(rank))
    penalty = edf * math.log(len(observations))
    return {"state": state, "likelihood_improvement_twice": improvement,
            "effective_dof": edf, "penalty": penalty, "score": improvement - penalty,
            "n_obs": len(observations), "rank": rank}


def pooled_signed_innovation_score(z_by_prn, *, tuning=1.345):
    z = _array(z_by_prn, "z", 3)
    if z.shape[1] != 50:
        raise ValueError("A1 requires 50 continuously present epochs")
    absolute = np.abs(z)
    loss = np.where(absolute <= tuning, .5 * z * z, tuning * (absolute - .5 * tuning))
    return float(np.median(np.mean(loss, axis=(1, 2))))


def empirical_threshold(scores, quantile):
    values = _array(scores, "scores", 1)
    if not len(values) or quantile not in (0.99, 0.995):
        raise ValueError("invalid frozen empirical quantile")
    return float(np.sort(values)[math.ceil(quantile * len(values)) - 1])


def persistent_three_of_five(alarms: Sequence[bool | None]):
    result = []
    for index in range(len(alarms)):
        trailing = alarms[max(0, index - 4): index + 1]
        result.append(len(trailing) == 5 and None not in trailing and sum(bool(x) for x in trailing) >= 3)
    return result


def common_support(method_scores: Mapping[str, Sequence[float]]):
    if not method_scores: raise ValueError("at least one method is required")
    arrays = {name: np.asarray(value, dtype=float) for name, value in method_scores.items()}
    if len({len(value) for value in arrays.values()}) != 1: raise ValueError("method lengths differ")
    mask = np.logical_and.reduce([np.isfinite(value) for value in arrays.values()])
    return {"mask": mask, "scores": {name: value[mask] for name, value in arrays.items()}}


def _level(level):
    if level is None or level == "NA": return "NA"
    return format(float(level), ".17g") if isinstance(level, (float, np.floating)) else str(level)


def content_seed(control_id, scenario, phase, block_id, level, object_id):
    material = f"23|{control_id}|{scenario}|{phase}|{block_id}|{_level(level)}|{object_id}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:16], "big")


def los_derangement(los_ecef, *, seed):
    los = _array(los_ecef, "LOS", 2)
    if len(los) < MINIMUM_PRNS: raise ValueError("LOS shuffle requires at least four PRNs")
    rng, identity = np.random.Generator(np.random.PCG64(seed)), np.arange(len(los))
    for draw in range(1, 1001):
        permutation = rng.permutation(identity)
        if not np.any(permutation == identity):
            return los[permutation], permutation, {"control_id": "LOS_SHUFFLE", "seed": seed, "draw": draw}
    raise ValueError("no LOS derangement in 1000 draws")


def temporal_desynchronization(residuals_by_prn, *, seed):
    residuals = _array(residuals_by_prn, "residuals", 3)
    count, length, _ = residuals.shape
    allowed, selected = list(range(10, length - 9)), []
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(count):
        candidates = [s for s in allowed if min(s, length - s) >= 10 and all(min((s-p) % length, (p-s) % length) >= 10 for p in selected)]
        if not candidates: raise ValueError("insufficient temporal shifts; no fallback")
        shift = candidates[int(rng.integers(0, len(candidates)))]
        selected.append(shift); allowed.remove(shift)
    return np.stack([np.roll(x, s, axis=0) for x, s in zip(residuals, selected)]), selected


def weighted_low_fpr_pauc(scores, labels, cells, *, alpha=.05, row_weights=None):
    score, label, cell = _array(scores, "scores", 1).astype(np.float64), np.asarray(labels, bool), np.asarray(cells)
    if score.shape != label.shape or score.shape != cell.shape or not 0 < alpha <= 1: raise ValueError("weighted pAUC shape/alpha mismatch")
    pos, neg = sorted({str(x) for x in cell[label]}), sorted({str(x) for x in cell[~label]})
    if not pos or not neg: raise ValueError("weighted pAUC requires both classes")
    text_cells = cell.astype(str)
    if row_weights is None:
        weights = np.zeros(len(score))
        for positive, names in ((True, pos), (False, neg)):
            for name in names:
                indices = np.flatnonzero((label == positive) & (text_cells == name))
                weights[indices] = 1 / (len(names) * len(indices))
    else:
        weights = np.asarray(row_weights, dtype=np.float64)
        if weights.shape != score.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weighted pAUC row weights are invalid")
        if not np.isclose(weights[label].sum(), 1.) or not np.isclose(weights[~label].sum(), 1.):
            raise ValueError("weighted pAUC class weights are not normalized")
    order, points, tp, fp, index = np.argsort(-score, kind="stable"), [(0., 0.)], 0., 0., 0
    while index < len(order):
        end = index + 1
        while end < len(order) and score[order[end]] == score[order[index]]: end += 1
        group = order[index:end]
        tp += float(np.sum(weights[group][label[group]])); fp += float(np.sum(weights[group][~label[group]]))
        points.append((fp, tp)); index = end
    clipped = [points[0]]
    for left, right in zip(points, points[1:]):
        if right[0] <= alpha: clipped.append(right); continue
        if left[0] < alpha:
            fraction = (alpha - left[0]) / (right[0] - left[0])
            clipped.append((alpha, left[1] + fraction * (right[1] - left[1])))
        break
    if clipped[-1][0] < alpha: clipped.append((alpha, clipped[-1][1]))
    x_values = np.asarray([x[0] for x in clipped], dtype=np.float64)
    y_values = np.asarray([x[1] for x in clipped], dtype=np.float64)
    area = np.sum(np.diff(x_values) * (y_values[:-1] + y_values[1:]) * .5, dtype=np.float64)
    return float(area) / alpha


def block_index(availability_endpoint_s, *, phase_start):
    return int(math.floor((np.nextafter(float(availability_endpoint_s), -np.inf) - float(phase_start)) / 10))


def nearest_rank_percentile(values, quantile):
    array = _array(values, "bootstrap values", 1)
    if not len(array) or not 0 < quantile <= 1: raise ValueError("invalid percentile")
    index = min(max(math.ceil(quantile * len(array)) - 1, 0), len(array) - 1)
    return float(np.sort(array)[index])


class AccessGate:
    """Fail-closed protected-file gate tied to an exact delivered commit."""
    def __init__(self, ledger_path):
        self.ledger_path, self.state = Path(ledger_path), "PREREGISTERED_UNVALIDATED"
        self._preflight = self._remote = None

    def set_preflight(self, *, clean_only_pass, reviews_pass, freeze_sha, frozen_hashes):
        if not clean_only_pass or not reviews_pass or len(freeze_sha) != 40 or not frozen_hashes or any(len(x) != 64 for x in frozen_hashes.values()):
            raise ValueError("incomplete implementation freeze preflight")
        self._preflight = {"freeze_sha": freeze_sha, "frozen_hashes": dict(frozen_hashes)}; self._refresh()

    def set_remote_sync(self, *, local_sha, remote_sha, ahead, behind, clean):
        self._remote = {"local_sha": local_sha, "remote_sha": remote_sha, "ahead": ahead, "behind": behind, "clean": clean}; self._refresh()

    def _refresh(self):
        if self._preflight and self._remote:
            freeze = self._preflight["freeze_sha"]
            if self._remote == {"local_sha": freeze, "remote_sha": freeze, "ahead": 0, "behind": 0, "clean": True}:
                self.state = "VALID_FOR_PROTECTED_ACCESS"

    def authorize(self, path, *, scenario, phase, expected_sha256, expected_size):
        if self.state != "VALID_FOR_PROTECTED_ACCESS":
            raise PermissionError("remote implementation freeze is not exactly synchronized" if self._preflight else "VALID_FOR_PROTECTED_ACCESS preflight not reached")
        candidate, text = Path(path), str(path)
        if any(token in text for token in ("*", "?", "[")): raise ValueError("protected path globs are forbidden")
        if candidate.is_dir(): raise ValueError("protected directories are forbidden")
        if candidate.name in {"scenario_metrics.csv", "ablation_metrics.csv", "per_epoch_scores.csv", "final_verdict.json"}: raise PermissionError("prior-result paths are forbidden")
        if scenario not in {"DS1", "DS2", "DS3", "DS4", "DS7", "DS8", "cleanDynamic", "DS5", "DS6"} or not phase: raise ValueError("unclassified protected access")
        if len(expected_sha256) != 64 or expected_size <= 0: raise ValueError("expected protected identity is incomplete")
        record = {"utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "actor": "gnss_doppler_lab.gcspo.AccessGate",
                  "canonical_path": str(candidate.resolve(strict=False)), "operation": "PREFLIGHT", "byte_or_row_range": "CLASSIFIED_BY_CALLER",
                  "scenario": scenario, "phase": phase, "purpose": "frozen GCSPO protected evaluation", "expected_sha256": expected_sha256,
                  "expected_size": expected_size, "authorization": self._preflight["freeze_sha"], "outcome": "AUTHORIZED_PENDING_IDENTITY_CHECK"}
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"); handle.flush()
        return record

# Capability implementation supersedes the preregistration-only prototype above.
from .gcspo_access import AccessGate as AccessGate
