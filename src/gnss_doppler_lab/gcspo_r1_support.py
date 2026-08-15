"""R1 execution-only exact-support adapter for the frozen GCSPO evaluator.

This module changes no score, threshold, model, feature, or timeline.  It only
permits the frozen phase-local B0 warm-up to yield a strict subset of Full
windows, while preserving Full standalone output and requiring byte-exact
native support for every B0/Full comparison.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import numpy as np

from .gcspo_statistics import PROTECTED_REQUIRED_PHASES, exact_contrast_support

_EXPECTED = ("A0", "A1", "A2", "A3", "A4", "A5", "Full")


def _native_support(row):
    try:
        epochs = tuple(map(int, row["epoch_ids"]))
        prns = tuple(map(int, row["prns"]))
        support = tuple((int(epoch), tuple(map(int, values)))
                        for epoch, values in row["epoch_prn_support"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("protected method native support is malformed") from None
    if (not epochs or tuple(epoch for epoch, _ in support) != epochs or
            epochs != tuple(sorted(set(epochs))) or
            any(not values or values != tuple(sorted(set(values))) for _, values in support)):
        raise ValueError("protected method native support is malformed")
    union = tuple(sorted(set().union(*(set(values) for _, values in support))))
    if prns != union:
        raise ValueError("protected method native support is malformed")
    return epochs, prns, support


def integrate_protected_b0_r1(methods, b0_rows, *, score_column):
    """Preserve Full rows and attach A0 only on exact phase/time/native support."""
    full_rows = list(methods.get("Full", ()))
    if not full_rows:
        raise ValueError("protected Full exact support is empty")
    full_by = {}
    for row in full_rows:
        key = (row.get("phase"), float(row["window_start_s"]), float(row["availability_s"]))
        if key in full_by:
            raise ValueError("duplicate Full window on protected exact support")
        _native_support(row)
        full_by[key] = row
    grouped = {}
    seen_prn = set()
    for row in list(b0_rows):
        if score_column not in row:
            raise ValueError("protected B0 score column is absent")
        try:
            score = float(row[score_column])
            key = (row.get("phase"), float(row["window_start_s"]), float(row["availability_s"]))
        except (TypeError, ValueError):
            raise ValueError("protected B0 join/score is malformed") from None
        if not np.isfinite(score):
            raise ValueError("protected B0 score is nonfinite")
        if "prn" in row:
            try:
                prn = int(str(row["prn"]).lstrip("Gg"))
                epochs = tuple(map(int, row["epoch_ids"]))
                support = tuple((int(epoch), tuple(map(int, values)))
                                for epoch, values in row["epoch_prn_support"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("protected B0 native support is malformed") from None
            identity = (*key, prn)
            if identity in seen_prn:
                raise ValueError("duplicate B0 window/PRN scientific row")
            seen_prn.add(identity)
            if (not epochs or tuple(epoch for epoch, _ in support) != epochs or
                    epochs != tuple(sorted(set(epochs))) or
                    any(values != (prn,) for _, values in support)):
                raise ValueError("protected B0 native support is malformed")
        grouped.setdefault(key, []).append(row)
    a0 = []
    for key in sorted(set(full_by) & set(grouped), key=lambda value: (str(value[0]), value[1], value[2])):
        full, group = full_by[key], grouped[key]
        if any(("prn" in row) != ("prn" in group[0]) for row in group):
            raise ValueError("protected B0 support representation is mixed")
        if "prn" in group[0]:
            combined = {}
            for row in group:
                for epoch, values in row["epoch_prn_support"]:
                    combined.setdefault(int(epoch), set()).update(map(int, values))
            support = tuple((epoch, tuple(sorted(values))) for epoch, values in sorted(combined.items()))
            scores = [float(row[score_column]) for row in group]
            if any(score != scores[0] for score in scores[1:]):
                raise ValueError("protected B0 event score differs across PRN rows")
            score = scores[0]
        else:
            if len(group) != 1:
                raise ValueError("duplicate aggregated B0 scientific row")
            _, _, support = _native_support(group[0])
            score = float(group[0][score_column])
        _, _, full_support = _native_support(full)
        if support != full_support:
            continue
        a0.append({"window_start_s": float(full["window_start_s"]),
                   "availability_s": float(full["availability_s"]), "score": score,
                   "prns": list(map(int, full["prns"])),
                   "epoch_ids": tuple(map(int, full["epoch_ids"])),
                   "epoch_prn_support": full_support,
                   **({"phase": full["phase"]} if "phase" in full else {})})
    return {**methods, "A0": a0}


def validate_protected_method_support_r1(methods, *, required_phases):
    """Require Full support for frozen methods and permit exact-subset A0/B0."""
    if set(methods) != set(_EXPECTED):
        raise ValueError("protected mandatory method set is incomplete")
    phases = tuple(required_phases)
    if not phases or len(phases) != len(set(phases)):
        raise ValueError("protected mandatory phase set is invalid")
    supports, counts = {}, {}
    for method in _EXPECTED:
        by_phase = {phase: set() for phase in phases}
        for row in list(methods[method]):
            phase = row.get("phase")
            if phase not in by_phase:
                raise ValueError(f"{method} emitted an unsupported protected phase")
            epochs, prns, support = _native_support(row)
            identity = (float(row["window_start_s"]), float(row["availability_s"]), epochs, prns, support)
            if identity in by_phase[phase]:
                raise ValueError(f"duplicate {method} protected support row")
            by_phase[phase].add(identity)
        if method != "A0" and any(not by_phase[phase] for phase in phases):
            raise ValueError(f"{method} silently empty on a mandatory protected phase")
        supports[method] = by_phase
        counts[method] = {phase: len(by_phase[phase]) for phase in phases}
    for method in ("A1", "A2", "A3", "A4", "A5"):
        if supports[method] != supports["Full"]:
            raise ValueError(f"{method}/Full exact native support mismatch")
    for phase in phases:
        if not supports["A0"][phase] <= supports["Full"][phase]:
            raise ValueError("A0/Full contains non-exact native support")
    status = ("AVAILABLE" if all(supports["A0"][phase] for phase in phases)
              else "UNAVAILABLE_ON_COMMON_SUPPORT")
    return {"methods": list(_EXPECTED), "phase_counts": counts,
            "b0_common_support_status": status,
            "full_standalone_preserved": True, "timestamp_tolerance_s": 0.0}


def exact_b0_full_contrast_r1(rows: Iterable[dict], *, required_scenarios):
    """Contrast exact intersections; report unavailable phases without copied scores."""
    rows = list(rows); result = {}; all_available = True
    for scenario in tuple(required_scenarios):
        phases = PROTECTED_REQUIRED_PHASES.get(scenario)
        if phases is None:
            raise ValueError(f"unsupported mandatory scenario for A0/B0: {scenario}")
        selected = [row for row in rows if row.get("scenario") == scenario and row.get("method") in {"A0", "Full"}]
        paired = exact_contrast_support(selected, ("Full", "A0"))
        phase_results = {}
        for phase in phases:
            full = [row for row in selected if row.get("phase") == phase and row.get("method") == "Full"]
            a0 = [row for row in selected if row.get("phase") == phase and row.get("method") == "A0"]
            common = [row for row in paired if row["phase"] == phase]
            if not common:
                all_available = False
                phase_results[phase] = {"status": "UNAVAILABLE_ON_COMMON_SUPPORT",
                                        "full_standalone_windows": len(full),
                                        "b0_windows": len(a0), "common_support_windows": 0}
                continue
            differences = np.asarray([row["scores"]["Full"] - row["scores"]["A0"] for row in common], dtype=float)
            if not np.isfinite(differences).all():
                raise ValueError("A0/B0 exact-support contrast is nonfinite")
            phase_results[phase] = {"status": "AVAILABLE", "full_standalone_windows": len(full),
                                    "b0_windows": len(a0), "common_support_windows": len(common),
                                    "mean_full_minus_a0": float(np.mean(differences)),
                                    "median_full_minus_a0": float(np.median(differences))}
        result[scenario] = phase_results
    return {"status": "AVAILABLE" if all_available else "UNAVAILABLE_ON_COMMON_SUPPORT",
            "contrast": "PAIRED_FULL_MINUS_A0_ON_EXACT_NATIVE_SUPPORT",
            "full_standalone_preserved": True, "timestamp_tolerance_s": 0.0,
            "scenario_phase_results": result}


@contextmanager
def r1_support_adapter_scope():
    """Temporarily bind all R1 adapters and restore every binding in ``finally``."""
    from . import gcspo_evaluate as evaluator
    from . import gcspo_verify_artifacts as verifier

    originals = (
        evaluator.integrate_protected_b0,
        evaluator.validate_protected_method_support,
        evaluator.exact_b0_full_contrast,
        verifier.exact_b0_full_contrast,
    )
    try:
        evaluator.integrate_protected_b0 = integrate_protected_b0_r1
        evaluator.validate_protected_method_support = validate_protected_method_support_r1
        evaluator.exact_b0_full_contrast = exact_b0_full_contrast_r1
        verifier.exact_b0_full_contrast = exact_b0_full_contrast_r1
        yield evaluator
    finally:
        (
            evaluator.integrate_protected_b0,
            evaluator.validate_protected_method_support,
            evaluator.exact_b0_full_contrast,
            verifier.exact_b0_full_contrast,
        ) = originals
