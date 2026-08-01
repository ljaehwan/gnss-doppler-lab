"""Statistics for immutable r3 CUDA/CPU reproduction diagnostics."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def component_agreement(reference, actual, *, threshold: float, times):
    ref = np.asarray(reference, float); got = np.asarray(actual, float); t = np.asarray(times, float)
    if ref.ndim != 1 or got.shape != ref.shape or t.shape != ref.shape:
        raise ValueError("reference, actual, and times must be equal-shape vectors")
    absolute = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), np.finfo(float).eps)
    alarm_ref = ref > threshold; alarm_got = got > threshold
    disagreement = t[alarm_ref != alarm_got]
    max_index = int(np.argmax(absolute))
    return {
        "count": int(len(ref)),
        "mean_abs_error": float(np.mean(absolute)),
        "median_abs_error": float(np.median(absolute)),
        "q95_abs_error": float(np.quantile(absolute, .95)),
        "q99_abs_error": float(np.quantile(absolute, .99)),
        "max_abs_error": float(absolute[max_index]),
        "max_abs_error_time": float(t[max_index]),
        "mean_relative_error": float(np.mean(absolute / denom)),
        "max_relative_error": float(np.max(absolute / denom)),
        "pearson": None if len(ref) < 2 or np.std(ref) == 0 or np.std(got) == 0 else float(np.corrcoef(ref, got)[0, 1]),
        "spearman": None if len(ref) < 2 else float(spearmanr(ref, got).statistic),
        "threshold": float(threshold),
        "alarm_agreement_rate": float(np.mean(alarm_ref == alarm_got)),
        "alarm_disagreement_events": [float(x) for x in disagreement],
    }
