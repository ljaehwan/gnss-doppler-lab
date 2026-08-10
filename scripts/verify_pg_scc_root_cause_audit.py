#!/usr/bin/env python3
"""Independent read-only verifier for PG-SCC root-cause audit artifacts."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/pg_scc_stage0_r2_root_cause_audit"
R1_ARTIFACT = ROOT / "artifacts/pg_scc_stage0_r1_root_cause_audit"
CONFIG_SHA256 = "336802a95e82df1da82822520fe8bd838bf18ce17da6ae29aa5695449f3b67f5"
SOURCE_SHA256 = "571b80ec11a9f860317f84c5d1808fddda270e988dfb8d25948df0999de0f8a4"
R1_FAILURE_BINDING_SHA256 = "550f3fde25742b571fa0a5206a96d0454300d1fe1732671b9bf655ccbb3f379f"
PREREGISTRATION_SHA = "162b67f921a719ddb8ad68ef99b50ad2c263773d"
R1_FAIL_CLOSED_SHA = "8cd78ed724e57f97498da26547a9ecbbc2a78fe1"
R1_ARTIFACT_SHA256 = {
    "config.json": "ade700892fe3e055e0154ae859ba701a36f11a6ec9245db4b39fd087426eb0e6",
    "source_commit.json": "13ab9e544f06903dd6fde04c42e8aefb35a66946bbaf0456e95040fbb777d428",
    "r1_fail_closed_report.json": "041cc432cdc893e9dba867d6d3dc005e3ee7f2c8d25d542d54d4f374ca68e3f5",
}
FROZEN = ROOT / "artifacts/pg_scc_stage0_static_k9"
CACHE = ROOT / "artifacts/acaf_nf_stage1_r3_static_detection"
FROZEN_EXPECTED_SUPPORT_PATH = FROZEN / "per_epoch_scores.csv"
FROZEN_EXPECTED_SUPPORT_SHA256 = "d3fd253807aea873dadce767fc027d3f0d4060ad8d3519e4c800b166e9e3bef5"
FROZEN_EXPECTED_PROJECTION_FIELDS = (
    "scenario", "phase", "second", "time_s", "channel", "prn", "method", "budget",
)
R1_PRESERVED_PATHS = {
    "artifacts/pg_scc_stage0_r1_root_cause_audit/config.json": "ade700892fe3e055e0154ae859ba701a36f11a6ec9245db4b39fd087426eb0e6",
    "artifacts/pg_scc_stage0_r1_root_cause_audit/r1_fail_closed_report.json": "041cc432cdc893e9dba867d6d3dc005e3ee7f2c8d25d542d54d4f374ca68e3f5",
    "artifacts/pg_scc_stage0_r1_root_cause_audit/source_commit.json": "13ab9e544f06903dd6fde04c42e8aefb35a66946bbaf0456e95040fbb777d428",
    "docs/PG_SCC_ROOT_CAUSE_AUDIT_IMPLEMENTATION.md": "ad746e2f481918a9a85e588beeaf709abbcaf3fcb81450b36c590372adad40dc",
    "docs/PG_SCC_ROOT_CAUSE_AUDIT_R1_FAIL_CLOSED.md": "4596439585dc7275774a709b5fd41da59b329e842b5cbc0a078256774b495b08",
    "scripts/run_pg_scc_root_cause_audit.py": "66009f9d14f9c66de4414e46dd3f2f12cd0f3a49cd96c7ae7deb0ea8663d7a76",
    "scripts/verify_pg_scc_root_cause_audit.py": "b9eb52f854e14b5a743c80d0bc5dc38ebb9d7c7eaef245cbe9a6389b81bb573a",
    "tests/test_pg_scc_root_cause_audit.py": "c0bb5f07ffd7f45c136d94d3a46420e9d222b3140e5c5f3270fbb8404438a290",
}
PHASE2_ALLOWED_CHANGED_PATHS = {
    "artifacts/pg_scc_stage0_r2_root_cause_audit/config.json",
    "artifacts/pg_scc_stage0_r2_root_cause_audit/source_commit.json",
    "artifacts/pg_scc_stage0_r2_root_cause_audit/r1_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_root_cause_audit/support_preflight.json",
    "scripts/run_pg_scc_root_cause_audit.py",
    "scripts/verify_pg_scc_root_cause_audit.py",
    "tests/test_pg_scc_root_cause_audit.py",
    "tests/test_pg_scc_r2_preflight.py",
}
CORE_METHODS = {
    "pg_scc_k3", "pg_scc_k5", "pg_scc_k9", "fixed9", "epl3",
    "dense_two_source_glrt", "shuffled_k3",
}
REQUIRED = {
    "README.md", "config.json", "source_commit.json", "r1_failure_binding.json",
    "support_preflight.json", "reproduction_check.json",
    "nested_mask_analysis.csv", "coordinate_contributions.csv",
    "score_dilution_metrics.csv", "dense_teacher_diagnostics.json",
    "synthetic_real_mismatch.csv", "selector_proxy_audit.json",
    "mask_seed_stability.csv", "random_mask_distribution.csv",
    "k3_exploratory_metrics.csv", "awgn_reaudit.json",
    "calibration_uncertainty.json", "bootstrap_intervals.csv",
    "root_cause_verdict.json", "artifact_manifest_sha256.json",
}
PLOTS = {
    "nested_coordinate_map.png", "clean_variance_attack_contribution.png",
    "rss_score_dilution.png", "synthetic_real_delay_doppler.png",
    "learned_random_percentile.png", "seed_mask_stability.png",
    "detector_performance.png", "empirical_noise_awgn_response.png",
    "calibration_threshold_uncertainty.png",
}
ROOT_CAUSES = {
    "SCORE_DILUTION", "NOISY_COORDINATE_ADDITION", "H1_NULL_OVERFIT",
    "DENSE_COVARIANCE_FAILURE", "SYNTHETIC_REAL_PHYSICS_MISMATCH",
    "SELECTOR_PROXY_OBJECTIVE_MISMATCH", "MASK_NON_IDENTIFIABILITY",
    "AWGN_CONTROL_MISSCALE", "CALIBRATION_TAIL_INSUFFICIENCY",
    "GENUINE_LACK_OF_SPARSE_GAIN",
}
VERDICTS = {
    "TERMINATE_PG_SCC", "REPAIRABLE_BUT_REQUIRES_NEW_CONFIRMATION",
    "K3_WORTH_INDEPENDENT_CONFIRMATION",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def function_ast_hash(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def support_hash(values: Iterable[tuple[Any, ...]]) -> str:
    canonical = sorted((list(value) for value in values), key=lambda value: tuple(map(str, value)))
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def project_source_metadata(path: Path) -> list[dict[str, Any]]:
    projection = (
        '.[] | {scenario, phase, second, time_s, prn} '
        '+ (if has("channel") and .channel != null then {channel} else {} end)'
    )
    result = subprocess.run(["jq", "-ce", projection, str(path)], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"metadata_projection:{path}")
    required = {"scenario", "phase", "second", "time_s", "prn"}
    output = []
    for line in result.stdout.splitlines():
        row = json.loads(line)
        if not required.issubset(row) or set(row) - (required | {"channel"}):
            raise RuntimeError(f"metadata_projection_schema:{path}")
        output.append(row)
    return output


def pooled_details(values: list[Mapping[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    pooled: dict[tuple[str, str, int], set[int]] = {}
    for row in values:
        event = (str(row["scenario"]), str(row["phase"]), int(row["second"]))
        pooled.setdefault(event, set()).add(int(row["prn"]))
    return {
        event: {"prns": sorted(prns), "prn_count": len(prns)}
        for event, prns in pooled.items()
    }


def event_identity(event: tuple[str, str, int]) -> dict[str, Any]:
    return {"scenario": event[0], "phase": event[1], "second": event[2]}


def project_frozen_expected_support() -> dict[str, Any]:
    current_hash = digest(FROZEN_EXPECTED_SUPPORT_PATH)
    committed = subprocess.check_output(
        ["git", "show", f"{R1_FAIL_CLOSED_SHA}:{FROZEN_EXPECTED_SUPPORT_PATH.relative_to(ROOT)}"],
        cwd=ROOT,
    )
    if current_hash != FROZEN_EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("frozen_expected_source_hash")
    if hashlib.sha256(committed).hexdigest() != current_hash:
        raise RuntimeError("frozen_expected_source_not_immutable")
    projected = subprocess.run(
        ["cut", "-d,", "-f1-6,11-12", str(FROZEN_EXPECTED_SUPPORT_PATH)],
        text=True, capture_output=True,
    )
    if projected.returncode:
        raise RuntimeError("frozen_expected_projection")
    projected_rows = list(csv.DictReader(projected.stdout.splitlines()))
    if not projected_rows or tuple(projected_rows[0]) != FROZEN_EXPECTED_PROJECTION_FIELDS:
        raise RuntimeError("frozen_expected_projection_schema")
    selected = [
        row for row in projected_rows
        if row["method"] == "pg_scc_k9" and int(row["budget"]) == 9
    ]
    details = pooled_details(selected)
    records = [
        {
            "identity": event_identity(event),
            "prn_count": detail["prn_count"],
            "prns": detail["prns"],
        }
        for event, detail in sorted(details.items()) if detail["prn_count"] >= 4
    ]
    identities = {
        (item["identity"]["scenario"], item["identity"]["phase"], item["identity"]["second"])
        for item in records
    }
    source = {
        "path": str(FROZEN_EXPECTED_SUPPORT_PATH.relative_to(ROOT)),
        "sha256": current_hash,
        "immutable_commit": R1_FAIL_CLOSED_SHA,
        "projection_fields": list(FROZEN_EXPECTED_PROJECTION_FIELDS),
        "filter": {"method": "pg_scc_k9", "budget": 9},
        "eligible_event_count": len(records),
        "eligible_event_support_hash": support_hash(identities),
    }
    return {"source": source, "records": records, "hash": support_hash(identities)}


def reconstruct_support() -> dict[str, Any]:
    source_paths = [CACHE / "clean_features.json", CACHE / "attack_features.json"]
    metadata: list[dict[str, Any]] = []
    sources = []
    for path in source_paths:
        projected = project_source_metadata(path)
        metadata.extend(projected)
        sources.append({
            "path": str(path.relative_to(ROOT)), "sha256": digest(path), "rows": len(projected),
        })
    channel_present = any(row.get("channel") is not None for row in metadata)
    if channel_present and any(row.get("channel") is None for row in metadata):
        raise RuntimeError("metadata_channel_presence")
    key_names = ("scenario", "phase", "second", "time_s", "prn") + (
        ("channel",) if channel_present else ()
    )
    raw = {tuple(row[name] for name in key_names) for row in metadata}
    if len(raw) != len(metadata):
        raise RuntimeError("duplicate_raw_support")
    details = pooled_details(metadata)
    all_events = [
        {
            "eligible": detail["prn_count"] >= 4,
            "identity": event_identity(event),
            "prn_count": detail["prn_count"],
            "prns": detail["prns"],
        }
        for event, detail in sorted(details.items())
    ]
    eligible = [
        {key: value for key, value in item.items() if key != "eligible"}
        for item in all_events if item["eligible"]
    ]
    excluded = [
        {key: value for key, value in item.items() if key != "eligible"}
        for item in all_events if not item["eligible"]
    ]
    frozen = project_frozen_expected_support()
    methods = sorted(CORE_METHODS)
    eligible_identities = {
        (item["identity"]["scenario"], item["identity"]["phase"], item["identity"]["second"])
        for item in eligible
    }
    raw_hash = support_hash(raw)
    eligible_hash = support_hash(eligible_identities)
    return {
        "raw_row_support_key": list(key_names),
        "pooled_event_key": ["scenario", "phase", "second"],
        "minimum_unique_prns": 4,
        "total_raw_event_count": len(raw),
        "common_rows_per_detector": len(raw),
        "eligible_event_count": len(eligible),
        "excluded_event_count": len(excluded),
        "pooled_events": len(eligible),
        "pooled_event_unique_prn_counts": all_events,
        "eligible_pooled_event_identities": [item["identity"] for item in eligible],
        "eligible_pooled_events": eligible,
        "excluded_pooled_event_identities": [item["identity"] for item in excluded],
        "excluded_pooled_events": excluded,
        "per_detector_raw_row_support_hashes": {method: raw_hash for method in methods},
        "per_detector_eligible_event_support_hashes": {method: eligible_hash for method in methods},
        "frozen_pooled_event_expected_count": len(frozen["records"]),
        "reconstructed_pooled_event_count": len(eligible),
        "frozen_pooled_event_count_match": len(eligible) == len(frozen["records"]),
        "frozen_pooled_event_records_match": eligible == frozen["records"],
        "frozen_pooled_event_support_hash": frozen["hash"],
        "frozen_pooled_event_expected_count_source": frozen["source"],
        "metadata_fields_read": [
            "scenario", "phase", "second", "time_s", "prn", "channel_if_present",
        ],
        "metadata_projection": "fixed jq whitelist; non-identity values are not emitted",
        "metadata_sources": sources,
        "protected_score_fields_read": 0,
        "protected_score_fields_projected_or_read": 0,
        "containers_opened_for_metadata_projection": [item["path"] for item in sources]
        + [frozen["source"]["path"]],
        "score_bearing_containers_opened_for_metadata_projection": [{
            "path": frozen["source"]["path"],
            "sha256": frozen["source"]["sha256"],
            "projection_fields": frozen["source"]["projection_fields"],
            "excluded_fields": ["score"],
        }],
    }


def independent_code_identity() -> dict[str, Any]:
    groups = {
        "score": (
            "batch_glrt", "diagnostic_scores", "pooled_events", "metric_bundle", "_fit_all",
            "reproduce", "reproduce_exact", "nested_outputs",
            "residual_components", "_dense_mismatch",
        ),
        "normalization": ("normalize_all", "_surface_array"),
        "selector": (
            "generate_random_masks", "compute_seed_stability", "_selector_audit", "_jaccard",
        ),
        "covariance": ("covariance_variants", "_covariance_audit", "residual_components"),
        "synthetic": ("_selector_audit", "_dense_mismatch", "_awgn_audit"),
        "mask": ("validate_nested_masks", "generate_random_masks"),
        "threshold": ("metric_bundle", "_calibration_audit", "_root_cause_verdict"),
    }
    names = tuple(dict.fromkeys(name for values in groups.values() for name in values))
    runner_path = ROOT / "scripts/run_pg_scc_root_cause_audit.py"
    current_source = runner_path.read_text(encoding="utf-8")
    r1_source = subprocess.check_output(
        ["git", "show", f"{R1_FAIL_CLOSED_SHA}:scripts/run_pg_scc_root_cause_audit.py"],
        cwd=ROOT, text=True,
    )
    functions = {}
    for name in names:
        current = function_ast_hash(current_source, name)
        expected = function_ast_hash(r1_source, name)
        functions[name] = {"current": current, "r1": expected, "match": current == expected}
    artifacts = {}
    for name in ("masks.json", "thresholds.json"):
        expected_bytes = subprocess.check_output(
            ["git", "show", f"{R1_FAIL_CLOSED_SHA}:artifacts/pg_scc_stage0_static_k9/{name}"],
            cwd=ROOT,
        )
        current = digest(FROZEN / name)
        expected = hashlib.sha256(expected_bytes).hexdigest()
        artifacts[name] = {"current": current, "r1": expected, "match": current == expected}
    source_files = {}
    for relative in (
        "src/gnss_doppler_lab/pg_scc.py",
        "src/gnss_doppler_lab/pg_scc_physics.py",
        "src/gnss_doppler_lab/pg_scc_selector.py",
    ):
        expected_bytes = subprocess.check_output(
            ["git", "show", f"{R1_FAIL_CLOSED_SHA}:{relative}"], cwd=ROOT,
        )
        current = digest(ROOT / relative)
        expected = hashlib.sha256(expected_bytes).hexdigest()
        source_files[relative] = {"current": current, "r1": expected, "match": current == expected}
    matches = all(item["match"] for collection in (functions, artifacts, source_files)
                  for item in collection.values())
    return {
        "status": "PASS" if matches else "FAIL",
        "groups": {name: list(values) for name, values in groups.items()},
        "function_ast_sha256": functions,
        "frozen_artifact_sha256": artifacts,
        "frozen_source_sha256": source_files,
    }


def independent_r1_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
    immutable = {}
    for path in R1_PRESERVED_PATHS:
        content = subprocess.check_output(["git", "show", f"{R1_FAIL_CLOSED_SHA}:{path}"], cwd=ROOT)
        immutable[path] = hashlib.sha256(content).hexdigest()
    current_preserved = {
        path: digest(ROOT / path)
        for path in R1_PRESERVED_PATHS
        if path.startswith("artifacts/pg_scc_stage0_r1_root_cause_audit/") or path.startswith("docs/")
    }
    changed: set[str] = set()
    for command in (
        ["git", "diff", "--name-only", R1_FAIL_CLOSED_SHA, "--"],
        ["git", "diff", "--cached", "--name-only", "--"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        changed.update(line for line in result.stdout.splitlines() if line)
    disallowed = sorted(changed - PHASE2_ALLOWED_CHANGED_PATHS)
    code = independent_code_identity()
    declarations = binding.get("explicit_unchanged_declarations", {})
    status = (
        binding.get("r1_fail_closed_commit") == R1_FAIL_CLOSED_SHA
        and binding.get("preserved_file_sha256") == R1_PRESERVED_PATHS
        and immutable == R1_PRESERVED_PATHS
        and all(current_preserved[path] == R1_PRESERVED_PATHS[path] for path in current_preserved)
        and declarations and all(value is True for value in declarations.values())
        and not disallowed and code["status"] == "PASS"
    )
    return {
        "status": "PASS" if status else "FAIL",
        "binding_hashes_verified": len(immutable),
        "preserved_file_sha256": immutable,
        "current_preserved_file_sha256": current_preserved,
        "disallowed_paths_unchanged": not disallowed,
        "disallowed_changed_paths": disallowed,
        "phase2_changed_paths": sorted(changed),
        "unchanged_code_identity": code,
    }

def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_preexecution(root: Path = DEFAULT_ARTIFACT) -> dict[str, Any]:
    """Independently reconstruct every pre-execution support and identity assertion."""
    root = root.resolve()
    errors: list[str] = []
    required = {"config.json", "source_commit.json", "r1_failure_binding.json", "support_preflight.json"}
    errors.extend(f"missing:{name}" for name in sorted(required) if not (root / name).is_file())
    if errors:
        return {
            "schema": "pg_scc_stage0_r2_preexecution_verification.v1",
            "status": "FAIL", "errors": errors, "analysis_executed": False,
        }
    expected_phase1 = {
        "config.json": CONFIG_SHA256,
        "source_commit.json": SOURCE_SHA256,
        "r1_failure_binding.json": R1_FAILURE_BINDING_SHA256,
    }
    errors.extend(
        f"phase1_checksum:{name}" for name, expected in expected_phase1.items()
        if digest(root / name) != expected
    )
    config = load(root / "config.json")
    source = load(root / "source_commit.json")
    binding = load(root / "r1_failure_binding.json")
    preflight = load(root / "support_preflight.json")
    if any(config.get(name) is not False for name in (
        "attack_fit", "attack_based_selection", "post_attack_retuning"
    )):
        errors.append("leakage_guard_config")
    common = config.get("common_support", {})
    if common.get("raw_row_support", {}).get("base_key") != [
        "scenario", "phase", "second", "time_s", "prn"
    ]:
        errors.append("raw_row_key")
    if common.get("pooled_event_identity", {}).get("key") != ["scenario", "phase", "second"]:
        errors.append("pooled_event_key")
    if common.get("eligible_event_filtering", {}).get("minimum_unique_prns_per_pooled_event") != 4:
        errors.append("minimum_prns")
    if source.get("branch") != "research/pg-scc-stage0-r2-root-cause-audit":
        errors.append("source_branch")
    if source.get("protected_score_fields_read_before_preregistration") != 0:
        errors.append("source_protected_access")
    if preflight.get("schema") != "pg_scc_stage0_r2_support_preflight.v1":
        errors.append("preflight_schema")
    if preflight.get("status") != "PASS":
        errors.append("preflight_status")

    reconstructed: dict[str, Any] = {}
    try:
        reconstructed = reconstruct_support()
    except Exception as exc:
        errors.append(f"support_reconstruction:{exc}")
    for key, expected in reconstructed.items():
        if preflight.get(key) != expected:
            errors.append(f"reconstructed_support_mismatch:{key}")
    if reconstructed and not (
        reconstructed["frozen_pooled_event_count_match"]
        and reconstructed["frozen_pooled_event_records_match"]
    ):
        errors.append("independent_frozen_support_mismatch")

    actual_r1 = {name: digest(R1_ARTIFACT / name) for name in R1_ARTIFACT_SHA256}
    expected_r1_artifact_report = {
        "status": "PASS" if actual_r1 == R1_ARTIFACT_SHA256 else "FAIL",
        "sha256": actual_r1,
        "expected_sha256": dict(R1_ARTIFACT_SHA256),
    }
    if preflight.get("r1_artifact_immutability") != expected_r1_artifact_report:
        errors.append("reconstructed_r1_artifact_immutability")
    try:
        r1_identity = independent_r1_identity(binding)
    except Exception as exc:
        r1_identity = {"status": "FAIL"}
        errors.append(f"r1_identity_reconstruction:{exc}")
    if r1_identity.get("status") != "PASS":
        errors.append("independent_r1_identity")
    if r1_identity.get("binding_hashes_verified") != 8:
        errors.append("r1_binding_hash_count")
    if preflight.get("r1_identity") != r1_identity:
        errors.append("reconstructed_r1_identity")
    code_identity = r1_identity.get("unchanged_code_identity", {})
    if preflight.get("frozen_formula_identity") != code_identity:
        errors.append("reconstructed_formula_identity")

    forbidden = [name for name in (
        "root_cause_verdict.json", "reproduction_check.json", "score_dilution_metrics.csv"
    ) if (root / name).exists()]
    errors.extend(f"phase3_artifact_present:{name}" for name in forbidden)
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True
    )
    if branch.returncode or branch.stdout.strip() != "research/pg-scc-stage0-r2-root-cause-audit":
        errors.append("git_branch")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREGISTRATION_SHA, "HEAD"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode:
        errors.append("preregistration_ancestor")
    return {
        "schema": "pg_scc_stage0_r2_preexecution_verification.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "analysis_executed": False,
        "protected_score_fields_read": 0,
        "protected_score_fields_projected_or_read": 0,
        "score_bearing_containers_opened_for_metadata_projection": (
            reconstructed.get("score_bearing_containers_opened_for_metadata_projection", [])
        ),
        "phase1_files_verified": len(expected_phase1),
        "r1_files_verified": len(r1_identity.get("preserved_file_sha256", {})),
        "pooled_events_verified": len(reconstructed.get("pooled_event_unique_prn_counts", [])),
    }



def verify_tree(root: Path, *, require_git: bool = True) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    for name in sorted(REQUIRED):
        if not (root / name).is_file():
            errors.append(f"missing:{name}")
    if not (root / "plots").is_dir():
        errors.append("missing:plots")
    elif require_git:
        for name in sorted(PLOTS):
            if not (root / "plots" / name).is_file():
                errors.append(f"missing:plots/{name}")
    if errors:
        return {"schema": "pg_scc_root_cause_verification.v1", "status": "FAIL", "errors": errors}
    if digest(root / "config.json") != CONFIG_SHA256:
        errors.append("frozen_config_drift")
    if digest(root / "source_commit.json") != SOURCE_SHA256:
        errors.append("frozen_source_commit_drift")
    manifest = load(root / "artifact_manifest_sha256.json")
    actual = {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest_sha256.json"
    }
    if set(manifest) != set(actual):
        errors.append("manifest_file_set_mismatch")
    errors.extend(
        f"checksum:{name}" for name in sorted(set(manifest) & set(actual))
        if manifest[name] != actual[name]
    )
    if require_git:
        reproduction = load(root / "reproduction_check.json")
        verdict = load(root / "root_cause_verdict.json")
        if reproduction.get("status") not in {"PASS", "REPRODUCTION_MISMATCH"}:
            errors.append("reproduction_status")
        causes = verdict.get("root_causes", {})
        if set(causes) != ROOT_CAUSES:
            errors.append("root_cause_set")
        if verdict.get("verdict") not in VERDICTS:
            errors.append("verdict")
        if verdict.get("labels", {}).get("attack") != "POST_HOC_DIAGNOSTIC":
            errors.append("attack_label")
        if verdict.get("labels", {}).get("k3") != "EXPLORATORY_ONLY":
            errors.append("k3_label")
        if any(
            item.get("status") not in {"PASS", "FAIL", "SUPPORTED", "UNSUPPORTED"}
            or not isinstance(item.get("numeric_evidence"), dict)
            for item in causes.values()
        ):
            errors.append("root_cause_status_or_evidence")
        config = load(root / "config.json")
        if any(config.get(name) is not False for name in (
            "attack_fit", "attack_based_selection", "post_attack_retuning"
        )):
            errors.append("leakage_guard_config")
        if config.get("bootstrap", {}).get("family_grouping", {}).get("ds4") != "ds4_transition_only":
            errors.append("ds4_scope")
        if config.get("bootstrap", {}).get("family_grouping", {}).get("ds7") != "ds7_ds8":
            errors.append("ds78_family")
        selector = load(root / "selector_proxy_audit.json")
        if selector.get("seed_count") != 20 or selector.get("attack_based_selection") is not False:
            errors.append("selector_schema")
        seed_rows = rows(root / "mask_seed_stability.csv")
        if len(seed_rows) != 60 or {int(row["budget"]) for row in seed_rows} != {3, 5, 9}:
            errors.append("selector_seed_rows")
        random_rows = rows(root / "random_mask_distribution.csv")
        for budget in (3, 5, 9):
            selected = [row for row in random_rows if int(row["budget"]) == budget]
            if len(selected) < 200 or any(
                row.get("selection_role") != "SYNTHETIC_VALIDATION_ONLY"
                or row.get("attack_label") != "NOT_USED_FOR_RANKING"
                for row in selected
            ):
                errors.append(f"random_mask_rows_k{budget}")
        bootstrap = rows(root / "bootstrap_intervals.csv")
        for row in bootstrap:
            if float(row["block_seconds"]) < 10:
                errors.append("bootstrap_block_seconds")
            if row["family"] == "ds4" and int(row["blocks"]) < 2 and row["status"] != "LIMITED":
                errors.append("ds4_limited")
        freeze = verdict.get("implementation_freeze", {})
        implementation_sha = freeze.get("implementation_sha", "")
        if len(implementation_sha) != 40 or freeze.get("ahead_behind") != [0, 0]:
            errors.append("implementation_freeze_binding")
        repo = root.parents[1]
        if len(implementation_sha) == 40 and subprocess.run(
            ["git", "merge-base", "--is-ancestor", implementation_sha, "HEAD"],
            cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode:
            errors.append("implementation_freeze_not_ancestor")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "scripts/run_pg_scc_root_cause_audit.py",
             "scripts/verify_pg_scc_root_cause_audit.py", "tests/test_pg_scc_root_cause_audit.py"],
            cwd=repo, text=True, capture_output=True,
        )
        if tracked.returncode:
            errors.append("implementation_not_tracked")
    return {
        "schema": "pg_scc_root_cause_verification.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "files_verified": len(actual),
        "analysis_executed": False,
    }


def verify_fresh_clone(repo: Path, revision: str) -> dict[str, Any]:
    """Clone and verify committed bytes only; never invokes the audit producer."""
    temp = Path(tempfile.mkdtemp(prefix="pg-scc-root-cause-verify."))
    try:
        clone = temp / "clone"
        subprocess.run(["git", "clone", "--no-local", "--quiet", str(repo), str(clone)], check=True)
        subprocess.run(["git", "checkout", "--quiet", revision], cwd=clone, check=True)
        report = verify_tree(clone / "artifacts/pg_scc_stage0_r2_root_cause_audit", require_git=True)
        report["fresh_clone"] = True
        report["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip()
        return report
    finally:
        shutil.rmtree(temp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--fresh-clone", action="store_true")
    parser.add_argument("--pre-execution", action="store_true")
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--no-git-schema", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pre_execution and args.fresh_clone:
        raise RuntimeError("pre-execution and fresh-clone modes are mutually exclusive")
    if args.pre_execution:
        report = verify_preexecution(args.artifact)
    elif args.fresh_clone:
        report = verify_fresh_clone(ROOT, args.revision)
    else:
        report = verify_tree(args.artifact, require_git=not args.no_git_schema)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
