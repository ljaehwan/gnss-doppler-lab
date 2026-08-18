"""Inference helpers for the MOSAIC Stage-0B R1c correction audit.

The module contains no data acquisition, injection, replay, case generation, or
detector tuning.  It operates only on committed scalar evidence.
"""
from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np

H34_VERDICTS = {
    "SUPPORTED_AND_DISCRIMINATIVE", "PRESENT_BUT_NOT_DISCRIMINATIVE",
    "INCONCLUSIVE", "UNSUPPORTED",
}
H6_VERDICTS = H34_VERDICTS | {"NOT_TESTABLE_FROM_RETAINED_EVIDENCE"}
RECOMMENDATIONS = {
    "TERMINATE_MOSAIC_AS_STAGE1_PATH", "ROOT_CAUSE_REMAINS_UNRESOLVED",
    "NEGATIVE_RESULT_ONLY", "CORRECTED_OBSERVER_REQUIRES_INDEPENDENT_DATA",
}


def cliff_delta(failures: Iterable[float], successes: Iterable[float]) -> float:
    """Cliff's delta, positive when failure-target values are larger."""
    a = np.asarray(list(failures), dtype=float)
    b = np.asarray(list(successes), dtype=float)
    if not len(a) or not len(b):
        raise ValueError("both comparison groups are required")
    return float(np.mean(np.sign(a[:, None] - b[None, :])))


def roc_auc(failures: Iterable[float], successes: Iterable[float]) -> float:
    """Probability that a random failure has the larger value (ties are half)."""
    return (cliff_delta(failures, successes) + 1.0) / 2.0


def exact_permutation_mean_test(failures: Iterable[float], successes: Iterable[float]) -> dict[str, float | int]:
    """Exact two-sided label permutation test for a difference in means."""
    a = np.asarray(list(failures), dtype=float)
    b = np.asarray(list(successes), dtype=float)
    if not len(a) or not len(b):
        raise ValueError("both comparison groups are required")
    values = np.r_[a, b]
    observed = float(np.mean(a) - np.mean(b))
    center = float(np.mean(values))
    # Difference from the permutation center is equivalent to the absolute
    # failure-group mean departure; enumerate every target-label allocation.
    threshold = abs(float(np.mean(a)) - center) - 1e-15
    extreme = 0
    total = 0
    for indices in itertools.combinations(range(len(values)), len(a)):
        total += 1
        if abs(float(np.mean(values[list(indices)])) - center) >= threshold:
            extreme += 1
    return {"statistic_mean_difference": observed, "p_value_two_sided": extreme / total,
            "permutations": total, "exact": True}


def deterministic_bootstrap_median_difference(
    failures: Iterable[float], successes: Iterable[float], *, seed: int = 20260818,
    replicates: int = 20000, confidence: float = 0.95,
) -> dict[str, float | int]:
    """Deterministic independent target-level bootstrap for median difference."""
    a = np.asarray(list(failures), dtype=float)
    b = np.asarray(list(successes), dtype=float)
    if not len(a) or not len(b) or replicates < 1:
        raise ValueError("nonempty groups and positive replicates are required")
    rng = np.random.default_rng(seed)
    sample_a = a[rng.integers(0, len(a), size=(replicates, len(a)))]
    sample_b = b[rng.integers(0, len(b), size=(replicates, len(b)))]
    values = np.median(sample_a, axis=1) - np.median(sample_b, axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return {"estimate": float(np.median(a) - np.median(b)), "lower": float(lo),
            "upper": float(hi), "confidence": confidence, "seed": seed,
            "replicates": replicates}


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    a = np.asarray(list(values), dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": len(a), "min": float(np.min(a)), "q25": float(np.quantile(a, .25)),
            "median": float(np.median(a)), "mean": float(np.mean(a)),
            "q75": float(np.quantile(a, .75)), "max": float(np.max(a))}


def discriminative_verdict(*, failure_presence: bool, comparator_presence: bool,
                           complete_separation: bool) -> str:
    """Comparator-aware rule without adding a statistical decision threshold."""
    if not failure_presence:
        return "UNSUPPORTED"
    if complete_separation and not comparator_presence:
        return "SUPPORTED_AND_DISCRIMINATIVE"
    return "PRESENT_BUT_NOT_DISCRIMINATIVE"


def decide_recommendation(*, reproduction_pass: bool, r1a_no_go: bool,
                          stage1_same_cases_prohibited: bool,
                          independent_implementation_defect: bool,
                          oracle_consistently_recovers: bool,
                          successful_comparators_not_degraded: bool,
                          negative_result_preserved: bool) -> str:
    """Recorded-condition truth table for the corrected recommendation."""
    if not reproduction_pass:
        return "ROOT_CAUSE_REMAINS_UNRESOLVED"
    if (independent_implementation_defect and oracle_consistently_recovers
            and successful_comparators_not_degraded):
        return "CORRECTED_OBSERVER_REQUIRES_INDEPENDENT_DATA"
    if r1a_no_go and stage1_same_cases_prohibited:
        return "TERMINATE_MOSAIC_AS_STAGE1_PATH"
    if negative_result_preserved:
        return "NEGATIVE_RESULT_ONLY"
    return "ROOT_CAUSE_REMAINS_UNRESOLVED"
