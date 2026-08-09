#!/usr/bin/env python3
"""Independent verifier for PG-SCC Stage-0 R2 support-feasibility artifacts."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/pg_scc_stage0_r2_support_feasibility"
CACHE = ROOT / "artifacts/acaf_nf_stage1_r3_static_detection"
CONFIG = ROOT / "configs/pg_scc_stage0_r2_support_feasibility.json"
BRANCH = "research/pg-scc-stage0-r2-support-feasibility"
BASE_SHA = "8cd78ed724e57f97498da26547a9ecbbc2a78fe1"
CONFIG_SHA256 = "013d83dc2245e6cb896607aeec124e7d5cd9fb0a107832affc9aec4d9b12e904"
SUPPORT_SUMMARY_SHA256 = "e55190563d9b4711984a94e79b0d0e6e35c604c3a58e81af39b08d12ad85b513"
SOURCE_SHA256 = "9529f04e66b85861b187f20b124fd003d8ad49de0657ba78fd773d3ae0e9f57a"
R1_HASHES = {
    "config.json": "ade700892fe3e055e0154ae859ba701a36f11a6ec9245db4b39fd087426eb0e6",
    "source_commit.json": "13ab9e544f06903dd6fde04c42e8aefb35a66946bbaf0456e95040fbb777d428",
    "r1_fail_closed_report.json": "041cc432cdc893e9dba867d6d3dc005e3ee7f2c8d25d542d54d4f374ca68e3f5",
}
FREEZE_REQUIRED = {
    "README.md", "source_commit.json", "support_inventory_summary.json",
}
FINAL_REQUIRED = {
    "support_accounting.json", "calibration.json", "paired_results.json",
    "control_results.json", "final_diagnostic.json", "artifact_manifest_sha256.json",
}
COMPARISONS = {
    "K9": ("pg_scc_k9", "fixed9", "shuffled_k9"),
    "K5": ("pg_scc_k5", "uniform_k5", "shuffled_k5"),
    "K3": ("pg_scc_k3", "epl3", "shuffled_k3"),
    "DENSE": ("dense_two_source_glrt",),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stratum(count: int) -> str:
    return "K9" if count >= 9 else "K5" if count >= 5 else "K3" if count >= 3 else "DENSE_ONLY" if count else "UNSUPPORTED"


def _eligible(family: str, count: int) -> bool:
    return count >= {"K9": 9, "K5": 5, "K3": 3, "DENSE": 1}[family]


def reconstruct_metadata(cache: Path = CACHE) -> tuple[dict[tuple[str, str, str, int], set[int]], list[dict[str, Any]]]:
    events: dict[tuple[str, str, str, int], set[int]] = defaultdict(set)
    projected = []
    forbidden = {"score", "label", "outcome", "alarm", "threshold", "auroc", "verdict"}
    for role, name in (("clean", "clean_features.json"), ("attack", "attack_features.json")):
        for row in load(cache / name):
            if forbidden & set(row):
                raise RuntimeError("outcome-bearing metadata input")
            key = (role, str(row["scenario"]), str(row["phase"]), int(row["second"]))
            prn = int(row["prn"])
            events[key].add(prn)
            projected.append({"source_role": role, "scenario": key[1], "phase": key[2],
                              "second": key[3], "prn": prn})
    return dict(events), projected


def summarize(events: Mapping[tuple[str, str, str, int], set[int]]) -> dict[str, Any]:
    counts = [len(value) for value in events.values()]
    histogram = Counter(counts)
    strata = Counter(_stratum(value) for value in counts)
    return {
        "total_event_count": len(counts),
        "common_unique_prn_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "eligible_event_counts": {str(k): sum(value >= k for value in counts) for k in (9, 5, 3)},
        "exclusive_support_strata": {
            "k9": strata["K9"], "k5": strata["K5"], "k3": strata["K3"],
            "dense_only": strata["DENSE_ONLY"], "unsupported": strata["UNSUPPORTED"],
        },
    }


def fingerprint(events: Mapping[tuple[str, str, str, int], set[int]]) -> str:
    canonical = [[*event, sorted(events[event])] for event in sorted(events)]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def source_static_audit(root: Path = ROOT) -> list[str]:
    errors = []
    config = load(root / "configs/pg_scc_stage0_r2_support_feasibility.json")
    guards = config.get("leakage_guards", {})
    if any(guards.get(key) is not False for key in ("attack_fit", "attack_based_selection", "post_attack_retuning")):
        errors.append("outcome_dependent_selection_config")
    selection = {str(value).lower() for value in guards.get("support_selection_fields", [])}
    if selection & {"score", "label", "outcome", "alarm", "threshold", "auroc", "metric", "verdict"}:
        errors.append("outcome_field_in_support_selection")
    if config.get("event_support", {}).get("k_eff_allowed") is not False:
        errors.append("k_eff_not_prohibited")
    producer_path = root / "scripts/run_pg_scc_r2_support_feasibility.py"
    tree = ast.parse(producer_path.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [ast.unparse(node) for node in ast.walk(main) if isinstance(node, ast.Call)]
    freeze = min((index for index, call in enumerate(calls) if "verify_implementation_freeze" in call), default=10**9)
    support_load = min((index for index, call in enumerate(calls) if "load_support_metadata" in call), default=-1)
    protected_load = min((index for index, call in enumerate(calls) if "load_protected_records" in call), default=-1)
    if freeze >= support_load or freeze >= protected_load or support_load >= protected_load:
        errors.append("freeze_support_protected_open_order")
    inventory_source = (root / "scripts/inventory_pg_scc_r2_support.py").read_text(encoding="utf-8")
    inventory_tree = ast.parse(inventory_source)
    imports = {node.names[0].name for node in ast.walk(inventory_tree) if isinstance(node, ast.Import)}
    if imports & {"numpy", "pandas", "csv"} or ".npz" in inventory_source:
        errors.append("inventory_can_open_outcome_container")
    masks = load(root / "artifacts/pg_scc_stage0_static_k9/masks.json")
    expected_budgets = {"pg_scc_k9": 9, "fixed9": 9, "shuffled_k9": 9, "pg_scc_k5": 5, "uniform_k5": 5, "shuffled_k5": 5, "pg_scc_k3": 3, "epl3": 3, "shuffled_k3": 3}
    for methods in config.get("comparison_families", {}).values():
        for method in methods:
            if method == "dense_two_source_glrt":
                continue
            values = masks.get(method, [])
            if len(values) != expected_budgets.get(method) or len(values) != len(set(values)):
                errors.append(f"mask_cardinality:{method}")
    return errors


def verify_freeze(root: Path = ROOT, require_git: bool = True) -> dict[str, Any]:
    errors = []
    artifact = root / "artifacts/pg_scc_stage0_r2_support_feasibility"
    for name in FREEZE_REQUIRED:
        if not (artifact / name).is_file():
            errors.append(f"missing_freeze:{name}")
    bindings = {
        root / "configs/pg_scc_stage0_r2_support_feasibility.json": CONFIG_SHA256,
        artifact / "support_inventory_summary.json": SUPPORT_SUMMARY_SHA256,
        artifact / "source_commit.json": SOURCE_SHA256,
    }
    for path, expected in bindings.items():
        if not path.is_file() or digest(path) != expected:
            errors.append(f"freeze_hash:{path.name}")
    r1 = root / "artifacts/pg_scc_stage0_r1_root_cause_audit"
    for name, expected in R1_HASHES.items():
        if not (r1 / name).is_file() or digest(r1 / name) != expected:
            errors.append(f"r1_immutability:{name}")
    errors.extend(source_static_audit(root))
    try:
        events, _ = reconstruct_metadata(root / "artifacts/acaf_nf_stage1_r3_static_detection")
        actual = summarize(events)
        inventory = load(artifact / "support_inventory_summary.json")
        for key in ("total_event_count", "common_unique_prn_histogram", "eligible_event_counts", "exclusive_support_strata"):
            if actual[key] != inventory.get(key):
                errors.append(f"inventory_reconstruction:{key}")
        if inventory.get("support_infeasible_event_count") != actual["exclusive_support_strata"]["dense_only"] + actual["exclusive_support_strata"]["unsupported"]:
            errors.append("support_infeasible_denominator")
        for method, item in inventory.get("method_availability", {}).items():
            if item.get("available_events", 0) + item.get("unavailable_events", 0) != len(events):
                errors.append(f"method_denominator:{method}")
    except Exception as exc:
        errors.append(f"metadata_reconstruction:{exc}")
    if require_git:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
        if branch and branch != BRANCH:
            errors.append("branch_mismatch")
        if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=root).returncode:
            errors.append("base_ancestry")
        for relative in (
            "scripts/run_pg_scc_r2_support_feasibility.py",
            "scripts/verify_pg_scc_r2_support_feasibility.py",
            "tests/test_pg_scc_r2_support_feasibility.py",
            "configs/pg_scc_stage0_r2_support_feasibility.json",
        ):
            if subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=root,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                errors.append(f"untracked_implementation:{relative}")
    return {
        "schema": "pg_scc_stage0_r2_verification.v1", "mode": "freeze_only",
        "status": "PASS" if not errors else "FAIL", "errors": sorted(set(errors)),
        "analysis_executed": False,
    }


def verify_final(root: Path = ROOT, require_git: bool = True) -> dict[str, Any]:
    report = verify_freeze(root, require_git=require_git)
    errors = list(report["errors"])
    artifact = root / "artifacts/pg_scc_stage0_r2_support_feasibility"
    for name in FINAL_REQUIRED:
        if not (artifact / name).is_file():
            errors.append(f"missing_final:{name}")
    if any(error.startswith("missing_final") for error in errors):
        return {**report, "mode": "final", "status": "FAIL", "errors": sorted(set(errors))}
    manifest = load(artifact / "artifact_manifest_sha256.json")
    actual = {str(path.relative_to(artifact)): digest(path) for path in sorted(artifact.rglob("*"))
              if path.is_file() and path.name != "artifact_manifest_sha256.json"}
    if set(manifest) != set(actual):
        errors.append("manifest_file_set")
    errors.extend(f"manifest_checksum:{name}" for name in set(manifest) & set(actual) if manifest[name] != actual[name])
    events, _ = reconstruct_metadata(root / "artifacts/acaf_nf_stage1_r3_static_detection")
    accounting = load(artifact / "support_accounting.json")
    summary = summarize(events)
    if accounting.get("universe", {}).get("total_events") != len(events) or accounting.get("no_event_drop") is not True:
        errors.append("event_drop_accounting")
    if accounting.get("universe_fingerprint") != fingerprint(events):
        errors.append("universe_support_fingerprint")
    for family, item in accounting.get("comparison_families", {}).items():
        if item.get("total_events") != len(events):
            errors.append(f"family_denominator:{family}")
        expected_eligible = sum(_eligible(family, len(prns)) for prns in events.values())
        if item.get("eligible_event_counts", {}).get(family) != expected_eligible:
            errors.append(f"family_eligibility:{family}")
    paired = load(artifact / "paired_results.json")
    cells = paired.get("cells", [])
    permutation = [cell for cell in cells if cell.get("control_role") == "RELATIONSHIP_PERMUTATION"]
    if paired.get("relationship_permutation_required") is not True or not permutation:
        errors.append("relationship_permutation_missing")
    for cell in cells:
        if cell.get("support_fingerprint_left") != cell.get("support_fingerprint_right"):
            errors.append("paired_support_not_identical")
        family = cell.get("outcome_family")
        scenarios = {"ds3": {"ds3"}, "ds4": {"ds4"}, "ds7_ds8": {"ds7", "ds8"}}.get(family, set())
        stratum = cell.get("support_stratum")
        selected = {event: prns for event, prns in events.items()
                    if event[0] == "attack" and event[1] in scenarios and event[2] != "strict_pre"
                    and _stratum(len(prns)) == stratum and _eligible(cell["k_family"], len(prns))}
        if cell.get("support_fingerprint_left") != fingerprint(selected):
            errors.append("paired_support_fingerprint_reconstruction")
        if cell.get("paired_events") != len(selected):
            errors.append("paired_event_denominator")
    ablation_keys = {(c["k_family"], c["support_stratum"], c["outcome_family"])
                     for c in cells if c.get("control_role") == "RELATIONAL_ABLATION"}
    permutation_keys = {(c["k_family"], c["support_stratum"], c["outcome_family"])
                        for c in permutation}
    if ablation_keys != permutation_keys:
        errors.append("relationship_control_cell_mismatch")
    calibration = load(artifact / "calibration.json")
    for cell in calibration.get("cells", []):
        if cell.get("eligible_event_denominator", -1) < 0:
            errors.append("calibration_denominator")
        if cell.get("status") == "AVAILABLE" and (cell.get("threshold_q99") is None or cell.get("threshold_block_bootstrap_ci95") is None):
            errors.append("calibration_uncertainty_missing")
        if cell.get("clean_holdout_events", 0) and cell.get("clean_holdout_clopper_pearson_95") is None:
            errors.append("clean_false_alarm_interval_missing")
        if cell.get("strict_external_pre_events", 0) and cell.get("strict_external_pre_clopper_pearson_95") is None:
            errors.append("external_false_alarm_interval_missing")
    controls = load(artifact / "control_results.json")
    selector = controls.get("selector_seed_stability", {})
    if selector.get("seed_count") != 20 or selector.get("seed_rows") != 60 or selector.get("attack_based_selection") is not False:
        errors.append("selector_seed_stability")
    if controls.get("leakage_guards") != {"attack_fit": False, "attack_based_selection": False, "post_attack_retuning": False}:
        errors.append("control_leakage_guard")
    if set(controls.get("relationship_controls", [])) != {"RELATIONAL_ABLATION", "RELATIONSHIP_PERMUTATION"}:
        errors.append("relationship_control_schema")
    aggregate = paired.get("aggregate_estimands", {})
    if aggregate.get("raw_scores_pooled_across_k_or_strata") is not False or aggregate.get("total_preregistered_cells") != len(cells):
        errors.append("aggregate_estimand_denominator")
    final = load(artifact / "final_diagnostic.json")
    if final.get("status") != "POST_R1_SUPPORT_REPAIRED_DIAGNOSTIC" or final.get("independent_confirmatory_evidence") is not False:
        errors.append("diagnostic_label")
    if final.get("k_eff_used") is not False or final.get("all_denominators_explicit") is not True:
        errors.append("final_policy_semantics")
    return {
        "schema": "pg_scc_stage0_r2_verification.v1", "mode": "final",
        "status": "PASS" if not errors else "FAIL", "errors": sorted(set(errors)),
        "analysis_executed": False, "events_reconstructed": len(events),
        "relationship_permutation_cells": len(permutation),
    }


def verify_fresh_clone(revision: str, final: bool) -> dict[str, Any]:
    temp = Path(tempfile.mkdtemp(prefix="pg-scc-r2-verify."))
    try:
        clone = temp / "clone"
        subprocess.run(["git", "clone", "--no-local", "--quiet", str(ROOT), str(clone)], check=True)
        subprocess.run(["git", "checkout", "--quiet", revision], cwd=clone, check=True)
        report = verify_final(clone, require_git=True) if final else verify_freeze(clone, require_git=True)
        report["fresh_clone"] = True
        report["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip()
        return report
    finally:
        shutil.rmtree(temp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--fresh-clone", action="store_true")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    if args.fresh_clone:
        report = verify_fresh_clone(args.revision, final=not args.freeze_only)
    else:
        report = verify_freeze() if args.freeze_only else verify_final()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
