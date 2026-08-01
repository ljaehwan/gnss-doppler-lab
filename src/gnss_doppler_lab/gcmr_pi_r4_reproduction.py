"""Statistics and correct-order helpers for immutable r3 reproduction diagnostics."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from gnss_doppler_lab.gcmr_pi_r4_corrected import (
    reconstruct_event_innovation,
    rescore_from_innovations,
)


def reconstruct_and_rescore(
    pipeline: Any,
    event: Any,
    *,
    reconstruct: Callable[[Any, Any], tuple[np.ndarray, np.ndarray]] = reconstruct_event_innovation,
    rescore: Callable[[Any, Any, np.ndarray, np.ndarray], tuple[Any, dict[str, float]]] = rescore_from_innovations,
) -> tuple[Any, dict[str, float]]:
    """Reconstruct `(residual, z)` then rescore using the required `(z, residual)` order."""
    residual, z = reconstruct(pipeline, event)
    return rescore(pipeline, event, z, residual)


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def component_agreement(reference, actual, *, threshold: float | None, times):
    ref = np.asarray(reference, float)
    got = np.asarray(actual, float)
    t = np.asarray(times, float)
    if ref.ndim != 1 or got.shape != ref.shape or t.shape != ref.shape:
        raise ValueError("reference, actual, and times must be equal-shape vectors")
    if not len(ref) or not np.isfinite(ref).all() or not np.isfinite(got).all() or not np.isfinite(t).all():
        raise ValueError("reference, actual, and times must be non-empty finite vectors")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite or None")

    absolute = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), np.finfo(float).eps)
    max_index = int(np.argmax(absolute))
    pearson = None
    if len(ref) >= 2 and np.std(ref) != 0 and np.std(got) != 0:
        pearson = _finite_or_none(np.corrcoef(ref, got)[0, 1])
    spearman = None
    if len(ref) >= 2:
        spearman = _finite_or_none(spearmanr(ref, got).statistic)

    result = {
        "count": int(len(ref)),
        "mean_abs_error": float(np.mean(absolute)),
        "median_abs_error": float(np.median(absolute)),
        "q95_abs_error": float(np.quantile(absolute, 0.95)),
        "q99_abs_error": float(np.quantile(absolute, 0.99)),
        "max_abs_error": float(absolute[max_index]),
        "max_abs_error_time": float(t[max_index]),
        "mean_relative_error": float(np.mean(absolute / denom)),
        "max_relative_error": float(np.max(absolute / denom)),
        "pearson": pearson,
        "spearman": spearman,
        "threshold": None if threshold is None else float(threshold),
        "alarm_agreement_rate": None,
        "alarm_disagreement_events": None,
    }
    if threshold is not None:
        alarm_ref = ref > threshold
        alarm_got = got > threshold
        result["alarm_agreement_rate"] = float(np.mean(alarm_ref == alarm_got))
        result["alarm_disagreement_events"] = [float(x) for x in t[alarm_ref != alarm_got]]
    return result
