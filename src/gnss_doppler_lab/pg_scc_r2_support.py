"""Pure support-accounting policy for PG-SCC Stage-0 R2."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


EVENT_FIELDS = ("source_role", "scenario", "phase", "second")
STRATA = ("K9", "K5", "K3", "DENSE_ONLY", "UNSUPPORTED")
FAMILY_MINIMUM = {"K9": 9, "K5": 5, "K3": 3, "DENSE": 1}
FORBIDDEN_SELECTION_FIELDS = {
    "score", "label", "outcome", "alarm", "threshold", "auroc", "auc",
    "metric", "verdict", "performance", "effect",
}


def event_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    missing = set(EVENT_FIELDS) - set(row)
    if missing:
        raise RuntimeError(f"event metadata missing fields: {sorted(missing)}")
    return (
        str(row["source_role"]), str(row["scenario"]), str(row["phase"]),
        int(row["second"]),
    )


def support_stratum(count: int) -> str:
    if count < 0:
        raise ValueError("support count must be nonnegative")
    if count >= 9:
        return "K9"
    if count >= 5:
        return "K5"
    if count >= 3:
        return "K3"
    if count >= 1:
        return "DENSE_ONLY"
    return "UNSUPPORTED"


def family_is_eligible(family: str, common_unique_prns: int) -> bool:
    if family not in FAMILY_MINIMUM:
        raise RuntimeError(f"unknown comparison family: {family}")
    return common_unique_prns >= FAMILY_MINIMUM[family]


def assert_support_only_selection(selection_fields: Sequence[str]) -> None:
    lowered = {str(field).lower() for field in selection_fields}
    leaked = lowered & FORBIDDEN_SELECTION_FIELDS
    if leaked:
        raise RuntimeError(f"outcome-dependent selection rejected: {sorted(leaked)}")
    allowed = set(EVENT_FIELDS) | {"prn", "method", "method_availability"}
    extra = lowered - allowed
    if extra:
        raise RuntimeError(f"non-preregistered support selection fields: {sorted(extra)}")


def universe_support(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, int], set[int]]:
    events: dict[tuple[str, str, str, int], set[int]] = defaultdict(set)
    for row in rows:
        key = event_key(row)
        prn = int(row["prn"])
        if prn < 1 or prn > 32:
            raise RuntimeError("PRN outside GPS support domain")
        events[key].add(prn)
    return dict(events)


def method_support(
    rows: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> dict[str, dict[tuple[str, str, str, int], set[int]]]:
    wanted = tuple(methods)
    if not wanted or len(set(wanted)) != len(wanted):
        raise RuntimeError("comparison methods must be unique and nonempty")
    result: dict[str, dict[tuple[str, str, str, int], set[int]]] = {
        method: defaultdict(set) for method in wanted
    }
    seen: set[tuple[str, tuple[str, str, str, int], int]] = set()
    for row in rows:
        method = str(row.get("method", ""))
        if method not in result:
            continue
        key = event_key(row)
        prn = int(row["prn"])
        exact = (method, key, prn)
        if exact in seen:
            raise RuntimeError(f"duplicate method/event/PRN row: {method}")
        seen.add(exact)
        result[method][key].add(prn)
    return {method: dict(events) for method, events in result.items()}


def common_support_by_event(
    universe: Mapping[tuple[str, str, str, int], set[int]],
    supports: Mapping[str, Mapping[tuple[str, str, str, int], set[int]]],
    methods: Sequence[str],
) -> dict[tuple[str, str, str, int], set[int]]:
    if not methods:
        raise RuntimeError("empty comparison family")
    missing_methods = set(methods) - set(supports)
    if missing_methods:
        raise RuntimeError(f"missing methods: {sorted(missing_methods)}")
    output = {}
    for event, available in universe.items():
        sets = [set(supports[method].get(event, set())) for method in methods]
        if any(not values.issubset(available) for values in sets):
            raise RuntimeError("method support contains PRN outside event universe")
        output[event] = set.intersection(*sets) if sets else set()
    extras = set().union(*(set(supports[method]) for method in methods)) - set(universe)
    if extras:
        raise RuntimeError("method support contains event outside universe")
    if set(output) != set(universe):
        raise RuntimeError("event disappeared during common-support construction")
    return output


def require_identical_paired_support(
    left: Mapping[tuple[str, str, str, int], set[int]],
    right: Mapping[tuple[str, str, str, int], set[int]],
) -> None:
    if set(left) != set(right):
        raise RuntimeError("paired event denominator mismatch")
    if any(left[event] != right[event] for event in left):
        raise RuntimeError("paired comparison does not use identical common PRN support")


def _count_summary(counts: Sequence[int]) -> dict[str, Any]:
    histogram = Counter(counts)
    strata = Counter(support_stratum(count) for count in counts)
    return {
        "total_events": len(counts),
        "common_unique_prn_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "exclusive_support_strata": {name: strata[name] for name in STRATA},
        "eligible_event_counts": {
            family: sum(family_is_eligible(family, count) for count in counts)
            for family in ("K9", "K5", "K3", "DENSE")
        },
    }


def aggregate_accounting(
    universe_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    comparison_families: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Aggregate every event; raw event keys and PRN sets are deliberately omitted."""
    assert_support_only_selection((*EVENT_FIELDS, "prn", "method_availability"))
    universe = universe_support(universe_rows)
    methods = sorted({method for values in comparison_families.values() for method in values})
    supports = method_support(method_rows, methods)
    method_counts = {}
    for method in methods:
        available = sum(bool(supports[method].get(event)) for event in universe)
        method_counts[method] = {
            "available_events": available,
            "unavailable_events": len(universe) - available,
            "denominator_events": len(universe),
        }
    family_counts = {}
    for family, family_methods in comparison_families.items():
        common = common_support_by_event(universe, supports, family_methods)
        family_counts[family] = _count_summary([len(common[event]) for event in universe])
    universe_summary = _count_summary([len(prns) for prns in universe.values()])
    if any(item["total_events"] != len(universe) for item in family_counts.values()):
        raise RuntimeError("family denominator accounting mismatch")
    return {
        "schema": "pg_scc_stage0_r2_support_accounting.v1",
        "event_identities_emitted": False,
        "prn_sets_emitted": False,
        "no_event_drop": True,
        "universe": universe_summary,
        "method_availability": method_counts,
        "comparison_families": family_counts,
    }
