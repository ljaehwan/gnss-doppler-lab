#!/usr/bin/env python3
"""Aggregate PG-SCC R2 support metadata without opening outcome-bearing files."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/acaf_nf_stage1_r3_static_detection"
DEFAULT_OUTPUT = ROOT / "artifacts/pg_scc_stage0_r2_support_feasibility/support_inventory_summary.json"
ALLOWED_INPUT_NAMES = {"clean_features.json", "attack_features.json"}
SUPPORT_FIELDS = {"scenario", "phase", "second", "prn"}
FORBIDDEN_TOP_LEVEL_FIELDS = {
    "score", "scores", "label", "labels", "outcome", "outcomes", "alarm",
    "alarms", "threshold", "thresholds", "auroc", "auc", "metric", "metrics",
    "verdict", "verdicts", "detector_outcome", "attack_score",
}
METHOD_MINIMUM_SUPPORT = {
    "dense_two_source_glrt": 1,
    "pg_scc_k3": 3,
    "epl3": 3,
    "shuffled_k3": 3,
    "pg_scc_k5": 5,
    "uniform_k5": 5,
    "shuffled_k5": 5,
    "pg_scc_k9": 9,
    "fixed9": 9,
    "shuffled_k9": 9,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _support_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if FORBIDDEN_TOP_LEVEL_FIELDS & set(row):
        raise RuntimeError("outcome-bearing field present in metadata-only input")
    missing = SUPPORT_FIELDS - set(row)
    if missing:
        raise RuntimeError(f"support metadata missing fields: {sorted(missing)}")
    scenario = row["scenario"]
    phase = row["phase"]
    if not isinstance(scenario, str) or not scenario or not isinstance(phase, str) or not phase:
        raise RuntimeError("scenario and phase must be non-empty structural strings")
    second = int(row["second"])
    prn = int(row["prn"])
    if prn < 1 or prn > 32:
        raise RuntimeError("PRN outside the GPS support domain")
    return {"scenario": scenario, "phase": phase, "second": second, "prn": prn}


def load_support_metadata(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open only a named JSON sidecar and immediately project its support fields."""
    resolved = path.resolve()
    if resolved.suffix != ".json" or resolved.name not in ALLOWED_INPUT_NAMES:
        raise RuntimeError("metadata inventory accepts only clean_features.json/attack_features.json")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError("support metadata must be a JSON list")
    projected = [_support_row(row) for row in raw]
    provenance = {
        "logical_name": resolved.name,
        "sha256": sha256(resolved),
        "rows": len(projected),
    }
    return projected, provenance


def support_histogram(
    sources: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    *,
    method_minimum_support: Mapping[str, int] = METHOD_MINIMUM_SUPPORT,
) -> dict[str, Any]:
    """Return aggregate cardinalities only; event keys and PRN sets never leave this function."""
    event_prns: dict[tuple[str, str, str, int], set[int]] = defaultdict(set)
    row_count = 0
    for source_role, rows in sources:
        if source_role not in {"clean", "attack", "synthetic"}:
            raise RuntimeError("invalid structural source role")
        for raw in rows:
            row = _support_row(raw)
            event = (source_role, row["scenario"], row["phase"], row["second"])
            event_prns[event].add(row["prn"])
            row_count += 1
    counts = [len(prns) for prns in event_prns.values()]
    histogram = Counter(counts)
    total = len(counts)
    strata = {
        "k9": sum(value >= 9 for value in counts),
        "k5": sum(5 <= value <= 8 for value in counts),
        "k3": sum(3 <= value <= 4 for value in counts),
        "dense_only": sum(1 <= value <= 2 for value in counts),
        "unsupported": sum(value == 0 for value in counts),
    }
    if sum(strata.values()) != total:
        raise RuntimeError("support stratum accounting mismatch")
    eligible = {str(k): sum(value >= k for value in counts) for k in (9, 5, 3)}
    methods = {
        method: {
            "minimum_common_unique_prns": int(minimum),
            "available_events": sum(value >= minimum for value in counts),
            "unavailable_events": sum(value < minimum for value in counts),
        }
        for method, minimum in sorted(method_minimum_support.items())
    }
    if any(item["available_events"] + item["unavailable_events"] != total for item in methods.values()):
        raise RuntimeError("method denominator accounting mismatch")
    return {
        "schema": "pg_scc_stage0_r2_support_inventory.v1",
        "inventory_mode": "METADATA_ONLY",
        "outcome_values_accessible": False,
        "event_key_used_locally_only": ["source_role", "scenario", "phase", "second"],
        "committed_event_identities": False,
        "committed_prn_sets": False,
        "total_metadata_rows": row_count,
        "total_event_count": total,
        "common_unique_prn_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "eligible_event_counts": eligible,
        "exclusive_support_strata": strata,
        "support_infeasible_event_count": strata["dense_only"] + strata["unsupported"],
        "method_availability": methods,
    }


def build_inventory(clean_path: Path, attack_path: Path) -> dict[str, Any]:
    clean, clean_provenance = load_support_metadata(clean_path)
    attack, attack_provenance = load_support_metadata(attack_path)
    summary = support_histogram((("clean", clean), ("attack", attack)))
    summary["source_provenance"] = [clean_provenance, attack_provenance]
    summary["isolation_guards"] = {
        "accepted_input_suffix": ".json",
        "accepted_input_basenames": sorted(ALLOWED_INPUT_NAMES),
        "npz_opened": False,
        "csv_opened": False,
        "outcome_fields_rejected": sorted(FORBIDDEN_TOP_LEVEL_FIELDS),
        "output_contains_only_aggregates": True,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-metadata", type=Path, default=CACHE / "clean_features.json")
    parser.add_argument("--attack-metadata", type=Path, default=CACHE / "attack_features.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_inventory(args.clean_metadata, args.attack_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
