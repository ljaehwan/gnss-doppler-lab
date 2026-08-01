"""Fail-closed primitives for the corrected frozen GCMR-PI r4 diagnostic.

These functions never substitute GCMR relation-cache observations for GRU residuals
or whitened innovations.  Callers must reconstruct innovations using the frozen
pipeline and save provenance for every sampled normal direction.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .gcmr_peak_innovation import aggregate_event_score


class FailClosedError(RuntimeError):
    """Raised when a requested diagnostic cannot be computed from real inputs."""


def validate_reconstructable_innovations(z: np.ndarray) -> np.ndarray:
    values = np.asarray(z, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 3:
        raise FailClosedError("innovation must have finite shape (N>=2, 3)")
    if not np.isfinite(values).all():
        raise FailClosedError("innovation contains non-finite values")
    if np.any(np.linalg.norm(values, axis=1) <= 0.0):
        raise FailClosedError("innovation direction is unavailable for a zero-norm row")
    return values


def warmup_period_masks(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return startup, pre, transition, post masks for the fixed r4 contract."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all():
        raise ValueError("times must be a finite one-dimensional array")
    return t < 30.0, (t >= 30.0) & (t < 110.0), (t >= 110.0) & (t < 130.0), t >= 130.0


def _normal_row_value(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def validate_direction_pool(pool: Sequence[Any]) -> None:
    if not pool:
        raise FailClosedError("normal event_calibration direction pool is empty")
    seen: set[tuple[int, int]] = set()
    for row in pool:
        role = str(_normal_row_value(row, "role"))
        if role not in {"event_calibration", "normal_reference"}:
            raise FailClosedError("direction pool contains a row not from normal event_calibration/reference")
        key = (int(_normal_row_value(row, "event_index")), int(_normal_row_value(row, "row_index")))
        if key in seen:
            raise FailClosedError("direction pool has duplicate normal event/row provenance")
        seen.add(key)
        direction = np.asarray(_normal_row_value(row, "direction"), dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) <= 0.0:
            raise FailClosedError("normal direction pool contains an unavailable direction")
        cn0 = float(_normal_row_value(row, "cn0"))
        if not np.isfinite(cn0):
            raise FailClosedError("normal direction pool contains non-finite C/N0")


def _cn0_bin(value: float, width: float = 5.0) -> int:
    return int(np.floor(float(value) / width))


def destroy_innovation_directions(original_z: np.ndarray, cn0: np.ndarray, pool: Sequence[Any], *, seed: int, cn0_bin_width: float = 5.0) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Replace every row direction by a unique normal direction while preserving norm.

    Sources are sampled without replacement within an event, preferentially from
    the same C/N0 bin.  The input pool must be exclusively normal data.
    """
    z = validate_reconstructable_innovations(original_z)
    contexts = np.asarray(cn0, dtype=float)
    if contexts.shape != (len(z),) or not np.isfinite(contexts).all():
        raise FailClosedError("C/N0 must be finite shape (N,)")
    validate_direction_pool(pool)
    if len(pool) < len(z):
        raise FailClosedError("normal direction pool has fewer independent rows than target PRNs")
    rng = np.random.default_rng(seed)
    unused = list(range(len(pool)))
    destroyed = np.empty_like(z)
    provenance: list[dict[str, Any]] = []
    for target_index, (vector, target_cn0) in enumerate(zip(z, contexts)):
        bin_candidates = [i for i in unused if _cn0_bin(_normal_row_value(pool[i], "cn0"), cn0_bin_width) == _cn0_bin(target_cn0, cn0_bin_width)]
        candidates = bin_candidates if bin_candidates else unused
        if not candidates:
            raise FailClosedError("could not allocate independent normal direction sources")
        selected = int(rng.choice(candidates))
        unused.remove(selected)
        source = pool[selected]
        direction = np.asarray(_normal_row_value(source, "direction"), dtype=float)
        direction = direction / np.linalg.norm(direction)
        norm = float(np.linalg.norm(vector))
        destroyed[target_index] = norm * direction
        provenance.append({
            "target_row_index": target_index,
            "source_event_index": int(_normal_row_value(source, "event_index")),
            "source_row_index": int(_normal_row_value(source, "row_index")),
            "source_role": str(_normal_row_value(source, "role")),
            "source_cn0": float(_normal_row_value(source, "cn0")),
            "target_cn0": float(target_cn0),
            "used_global_pool": not bool(bin_candidates),
        })
    if not np.allclose(np.linalg.norm(destroyed, axis=1), np.linalg.norm(z, axis=1), rtol=1e-12, atol=1e-12):
        raise AssertionError("relation destruction violated PRN norm preservation")
    keys = [(x["source_event_index"], x["source_row_index"]) for x in provenance]
    if len(keys) != len(set(keys)):
        raise AssertionError("relation destruction reused a normal source row")
    return destroyed, provenance


def component_scores(components: Mapping[str, np.ndarray], location: Mapping[str, float], scale: Mapping[str, float]) -> dict[str, np.ndarray]:
    """Compute exact standardised r4 component scores with frozen calibrator state."""
    required = ("s_common", "n_eff", "s_pair", "energy")
    if any(k not in components or k not in location or k not in scale for k in required):
        raise FailClosedError("missing frozen normal calibrator component")
    standardized: dict[str, np.ndarray] = {}
    for key in required:
        values = np.asarray(components[key], dtype=float)
        denom = float(scale[key])
        if not np.isfinite(values).all() or not np.isfinite(denom) or denom <= 0.0:
            raise FailClosedError("invalid frozen calibrator values")
        standardized[key] = (values - float(location[key])) / denom
    return {
        "R1": standardized["s_common"],
        "R2": standardized["s_common"] + standardized["n_eff"],
        "R3": standardized["s_pair"],
        "RelationOnly": standardized["s_common"] + standardized["n_eff"] + standardized["s_pair"],
        "EnergyOnly": standardized["energy"],
    }


def direct_normal_thresholds(scores: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Threshold each complete score distribution directly on normal calibration rows."""
    output: dict[str, dict[str, float]] = {}
    for name, values in scores.items():
        x = np.asarray(values, dtype=float)
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise FailClosedError(f"normal score {name} is unavailable")
        output[name] = {"q99": float(np.quantile(x, .99)), "q995": float(np.quantile(x, .995)), "FPR1": float(np.quantile(x, .99))}
    return output


def reconstruct_event_innovation(pipeline: Any, event: Any) -> tuple[np.ndarray, np.ndarray]:
    """Obtain raw GRU residual and whitener output from the immutable frozen pipeline."""
    residual = np.asarray(pipeline._predict_residual(event), dtype=float)
    z = np.asarray(pipeline.whitener.transform(residual, event.cn0), dtype=float)
    validate_reconstructable_innovations(z)
    if residual.shape != z.shape or not np.isfinite(residual).all():
        raise FailClosedError("frozen GRU residual cannot be reconstructed")
    return residual, z


def rescore_from_innovations(pipeline: Any, event: Any, z: np.ndarray, residual: np.ndarray) -> tuple[Any, dict[str, float]]:
    """Use frozen diagnostics/pair model and frozen calibrator; no proxy statistic."""
    values = validate_reconstructable_innovations(z)
    raw = np.asarray(residual, dtype=float)
    if raw.shape != values.shape or not np.isfinite(raw).all():
        raise FailClosedError("scalar residual is unavailable for actual re-scoring")
    diagnostics = pipeline._diagnostics(values, event)
    scores = {name: float(aggregate_event_score(diagnostics, pipeline.calibrator, name)) for name in ("A0", "A1", "A2", "A3", "A4", "Full")}
    components = {"s_common": np.asarray([diagnostics.s_common]), "n_eff": np.asarray([diagnostics.n_eff]), "s_pair": np.asarray([diagnostics.s_pair]), "energy": np.asarray([diagnostics.energy])}
    r4 = component_scores(components, pipeline.calibrator.location_, pipeline.calibrator.scale_)
    scores.update({key: float(value[0]) for key, value in r4.items()})
    return diagnostics, scores
