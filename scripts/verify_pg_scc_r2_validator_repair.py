#!/usr/bin/env python3
"""Independent verifier for the PG-SCC R2 validator sequencing repair."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/pg_scc_stage0_r2_validator_repair"
BRANCH = "research/pg-scc-stage0-r2-validator-repair"
BASE_SHA = "68ab54677f5d0b4b55cc39279aec631f60f655a9"
PREREGISTRATION_SHA = "c7887316ed981d0e7cde74b2bbadeb1cf83bb233"
PREREGISTRATION_BLOB_SHA256 = "25fcfdb342b733fab7e296f72940a92aabf89fdd03b662bda0845ca8bbd884c0"
VERIFIER_HARDENING_PREREGISTRATION_SHA = "bafbe69365bd99380f9a976cd6dcdd4e951abfab"
VERIFIER_HARDENING_PREREGISTRATION_BLOB_SHA256 = (
    "58c0e3333df0cc25ea74cf3be6d662f684a224e88885cd58e2502a088f7c2aea"
)
VERIFIER_HARDENING_PREREGISTRATION_PATH = (
    "artifacts/pg_scc_stage0_r2_validator_verifier_hardening/preregistration.json"
)
CONFIG_SHA256 = "336802a95e82df1da82822520fe8bd838bf18ce17da6ae29aa5695449f3b67f5"
SOURCE_SHA256 = "2b2c7bc55d031a9fd86f210e249a51d44c0a7f0e92500159e2b60dc05db87f70"
R1_BINDING_SHA256 = "550f3fde25742b571fa0a5206a96d0454300d1fe1732671b9bf655ccbb3f379f"
PREDECESSOR_IMPLEMENTATION_SHA = "9839823c00cafc34fbbf1d6b1dbe069eb2c4e74d"
MANIFEST_FILES = (
    "scripts/run_pg_scc_root_cause_audit.py",
    "scripts/verify_pg_scc_root_cause_audit.py",
    "scripts/verify_pg_scc_r2_repair_followup.py",
    "scripts/verify_pg_scc_r2_semantic_diff.py",
    "scripts/verify_pg_scc_r2_validator_repair.py",
    "scripts/verify_pg_scc_r2_validator_semantic_diff.py",
    "tests/test_pg_scc_root_cause_audit.py",
    "tests/test_pg_scc_r2_preflight.py",
    "tests/test_pg_scc_r2_repair_followup.py",
    "tests/test_pg_scc_r2_validator_repair.py",
    "artifacts/pg_scc_stage0_r2_validator_repair/config.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/source_commit.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/r1_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/support_preflight.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/predecessor_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/validator_root_cause.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/test_report.txt",
    "artifacts/pg_scc_stage0_r2_validator_repair/semantic_diff_audit.json",
    "artifacts/pg_scc_stage0_r2_validator_finalization_followup/preregistration.json",
    "artifacts/pg_scc_stage0_r2_validator_verifier_hardening/preregistration.json",
)
PLOTS = (
    "nested_coordinate_map.png", "clean_variance_attack_contribution.png",
    "rss_score_dilution.png", "synthetic_real_delay_doppler.png",
    "learned_random_percentile.png", "seed_mask_stability.png",
    "detector_performance.png", "empirical_noise_awgn_response.png",
    "calibration_threshold_uncertainty.png",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _load_base_verifier():
    path = ROOT / "scripts/verify_pg_scc_root_cause_audit.py"
    spec = importlib.util.spec_from_file_location("pg_scc_r2_base_verifier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE_SHA256 = SOURCE_SHA256
    return module


def _committed_artifact_entry_names() -> set[str]:
    try:
        raw = subprocess.check_output(
            [
                "git", "show",
                f"{VERIFIER_HARDENING_PREREGISTRATION_SHA}:"
                f"{VERIFIER_HARDENING_PREREGISTRATION_PATH}",
            ],
            cwd=ROOT,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("committed_hardening_preregistration_unavailable") from exc
    if hashlib.sha256(raw).hexdigest() != VERIFIER_HARDENING_PREREGISTRATION_BLOB_SHA256:
        raise RuntimeError("committed_hardening_preregistration_blob_hash")
    try:
        preregistration = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("committed_hardening_preregistration_json") from exc
    contract = preregistration.get("artifact_closed_world_contract")
    values = contract.get("exact_entry_names") if isinstance(contract, dict) else None
    if (
        preregistration.get("schema")
        != "pg_scc_stage0_r2_validator_verifier_hardening_preregistration.v1"
        or not isinstance(values, list)
        or any(not isinstance(name, str) for name in values)
        or values != sorted(set(values))
        or len(values) != 39
        or contract.get("entry_count") != 39
        or contract.get("binding_commit_sha")
        != "5a694a0662376b9e357fa930638657a90e20d7df"
        or "artifact_manifest_sha256.json" in values
    ):
        raise RuntimeError("committed_hardening_artifact_set")
    return set(values)


def _artifact_manifest_errors(root: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = root / "artifact_manifest_sha256.json"
    try:
        manifest = _load(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return ["artifact_manifest_unreadable"]
    expected_names = _committed_artifact_entry_names()
    self_manifest = root / "artifact_manifest_sha256.json"
    actual = {
        str(path.relative_to(root)): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != self_manifest
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_names:
        errors.append("artifact_manifest_expected_key_set")
        manifest = manifest if isinstance(manifest, dict) else {}
    if set(actual) != expected_names:
        errors.append("artifact_actual_expected_file_set")
    if set(manifest) != set(actual):
        errors.append("artifact_manifest_actual_file_set")
    errors.extend(
        f"artifact_checksum:{name}"
        for name in sorted(set(manifest) & set(actual))
        if manifest[name] != actual[name]
    )
    return errors


def _committed_operational_additions() -> set[str]:
    relative = "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json"
    raw = subprocess.check_output(
        ["git", "show", f"{PREREGISTRATION_SHA}:{relative}"], cwd=ROOT,
    )
    if hashlib.sha256(raw).hexdigest() != PREREGISTRATION_BLOB_SHA256:
        raise RuntimeError("committed_preregistration_blob_hash")
    preregistration = json.loads(raw)
    values = preregistration.get("dynamic_operational_path_contract", {}).get(
        "allowed_only_when_added_after_committed_preflight_and_subset_of_this_set"
    )
    if (
        not isinstance(values, list)
        or any(not isinstance(path, str) for path in values)
        or values != sorted(set(values))
    ):
        raise RuntimeError("committed_preregistration_operational_paths")
    return set(values)


def _preexecution_dirty_errors(status: str) -> list[str]:
    operational = _committed_operational_additions()
    return [
        line for line in status.splitlines()
        if line and not (line[:2] == "??" and line[3:] in operational)
    ]


def _manifest_errors(root: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = root / "implementation_manifest_sha256.json"
    manifest = _load(manifest_path)
    expected_outer_keys = {
        "schema", "base_sha", "implementation_freeze_rule", "preregistration_sha",
        "protected_score_fields_read_before_freeze", "files",
    }
    if set(manifest) != expected_outer_keys:
        errors.append("implementation_manifest_schema_key_set")
    if manifest.get("schema") != "pg_scc_stage0_r2_validator_repair_implementation_manifest.v1":
        errors.append("implementation_manifest_schema")
    if manifest.get("base_sha") != PREDECESSOR_IMPLEMENTATION_SHA:
        errors.append("implementation_manifest_base")
    if manifest.get("preregistration_sha") != PREREGISTRATION_SHA:
        errors.append("implementation_manifest_preregistration")
    if manifest.get("implementation_freeze_rule") != "commit containing these exact hashes":
        errors.append("implementation_manifest_freeze_rule")
    if manifest.get("protected_score_fields_read_before_freeze") != 0:
        errors.append("implementation_manifest_protected_access")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(MANIFEST_FILES):
        errors.append("implementation_manifest_key_set")
        files = files if isinstance(files, dict) else {}
    prereg_relative = "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json"
    if files.get(prereg_relative) != PREREGISTRATION_BLOB_SHA256:
        errors.append("implementation_manifest_preregistration_blob")
    repo = root.parents[1]
    for relative in MANIFEST_FILES:
        expected = files.get(relative)
        path = repo / relative
        if not isinstance(expected, str) or not path.is_file() or _digest(path) != expected:
            errors.append(f"implementation_hash:{relative}")
        try:
            committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=repo)
        except subprocess.CalledProcessError:
            errors.append(f"implementation_uncommitted:{relative}")
        else:
            if not isinstance(expected, str) or hashlib.sha256(committed).hexdigest() != expected:
                errors.append(f"implementation_head_hash:{relative}")
    manifest_relative = "artifacts/pg_scc_stage0_r2_validator_repair/implementation_manifest_sha256.json"
    try:
        committed_manifest = subprocess.check_output(
            ["git", "show", f"HEAD:{manifest_relative}"], cwd=repo,
        )
    except subprocess.CalledProcessError:
        errors.append("implementation_manifest_uncommitted")
    else:
        if manifest_path.read_bytes() != committed_manifest:
            errors.append("implementation_manifest_head_bytes")
    return errors


def verify_preexecution(root: Path, implementation_sha: str) -> dict[str, Any]:
    root = root.resolve()
    repo = root.parents[1]
    errors: list[str] = []
    required = {
        "config.json", "source_commit.json", "r1_failure_binding.json",
        "support_preflight.json", "preregistration.json",
        "predecessor_failure_binding.json", "validator_root_cause.json", "semantic_diff_audit.json",
        "implementation_manifest_sha256.json", "pre_run_state.json",
    }
    errors.extend(f"missing:{name}" for name in sorted(required) if not (root / name).is_file())
    if errors:
        return {"schema": "pg_scc_stage0_r2_validator_repair_preexecution.v1", "status": "FAIL", "errors": errors}
    head = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    remote = _git(repo, "rev-parse", f"origin/{BRANCH}")
    if head != implementation_sha or remote != head:
        errors.append("implementation_local_remote_sha")
    if branch != BRANCH:
        errors.append("branch")
    if _git(repo, "merge-base", head, BASE_SHA) != BASE_SHA:
        errors.append("base")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PREREGISTRATION_SHA, head], cwd=repo).returncode:
        errors.append("preregistration_ancestor")
    if _digest(root / "config.json") != CONFIG_SHA256:
        errors.append("config_hash")
    if _digest(root / "source_commit.json") != SOURCE_SHA256:
        errors.append("source_hash")
    if _digest(root / "r1_failure_binding.json") != R1_BINDING_SHA256:
        errors.append("r1_binding_hash")
    prereg = _load(root / "preregistration.json")
    predecessor = _load(root / "predecessor_failure_binding.json")
    semantic = _load(root / "semantic_diff_audit.json")
    support = _load(root / "support_preflight.json")
    pre_run = _load(root / "pre_run_state.json")
    if prereg.get("branch") != BRANCH or prereg.get("base_sha") != BASE_SHA:
        errors.append("preregistration_identity")
    if predecessor.get("authorized_predecessor_failure_metadata", {}).get("implementation_sha") != PREDECESSOR_IMPLEMENTATION_SHA:
        errors.append("predecessor_binding")
    if semantic.get("status") != "PASS" or not semantic.get("non_plot_scientific_functions_equivalent"):
        errors.append("semantic_diff")
    if support.get("status") != "PASS" or support.get("protected_score_fields_projected_or_read") != 0:
        errors.append("support_preflight")
    if pre_run.get("state") != "READY" or pre_run.get("implementation_sha") != head:
        errors.append("pre_run_state")
    errors.extend(_manifest_errors(root))
    dirty = _preexecution_dirty_errors(
        _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    )
    if dirty:
        errors.append("dirty_outside_followup_namespace:" + "|".join(dirty))
    return {
        "ahead_behind": [int(value) for value in _git(repo, "rev-list", "--left-right", "--count", f"{head}...{remote}").split()],
        "analysis_executed": False,
        "errors": sorted(set(errors)),
        "implementation_sha": head,
        "protected_score_fields_read": 0,
        "schema": "pg_scc_stage0_r2_validator_repair_preexecution.v1",
        "status": "PASS" if not errors else "FAIL",
    }


def verify_tree(root: Path) -> dict[str, Any]:
    root = root.resolve()
    repo = root.parents[1]
    errors: list[str] = []
    base_report = _load_base_verifier().verify_tree(root, require_git=True)
    if base_report.get("status") != "PASS":
        errors.extend(f"base_verifier:{item}" for item in base_report.get("errors", []))
    required = {
        "preregistration.json", "predecessor_failure_binding.json", "pre_run_state.json",
        "implementation_manifest_sha256.json", "semantic_diff_audit.json", "attempt_state.json",
    }
    errors.extend(f"missing:{name}" for name in sorted(required) if not (root / name).is_file())
    if errors:
        return {"schema": "pg_scc_stage0_r2_validator_repair_verification.v1", "status": "FAIL", "errors": sorted(set(errors))}
    prereg = _load(root / "preregistration.json")
    predecessor = _load(root / "predecessor_failure_binding.json")
    pre_run = _load(root / "pre_run_state.json")
    semantic = _load(root / "semantic_diff_audit.json")
    attempt = _load(root / "attempt_state.json")
    verdict = _load(root / "root_cause_verdict.json")
    if prereg.get("branch") != BRANCH or prereg.get("base_sha") != BASE_SHA:
        errors.append("preregistration_identity")
    metadata = predecessor.get("authorized_predecessor_failure_metadata", {})
    if metadata.get("implementation_sha") != PREDECESSOR_IMPLEMENTATION_SHA or metadata.get("protected_loader_invoked") is not False:
        errors.append("predecessor_binding")
    if semantic.get("status") != "PASS" or not semantic.get("non_plot_scientific_functions_equivalent"):
        errors.append("semantic_diff")
    if attempt.get("attempt_count") != 1 or attempt.get("state") != "COMPLETED":
        errors.append("attempt_state")
    if attempt.get("scientific_verdict") != verdict.get("verdict"):
        errors.append("attempt_verdict")
    if attempt.get("implementation_sha") != pre_run.get("implementation_sha"):
        errors.append("implementation_lineage")
    if _digest(root / "config.json") != CONFIG_SHA256 or (root / "config.json").read_bytes() != (
        repo / "artifacts/pg_scc_stage0_r2_root_cause_audit/config.json"
    ).read_bytes():
        errors.append("no_retuning_config")
    errors.extend(_manifest_errors(root))
    errors.extend(_artifact_manifest_errors(root))
    for name in PLOTS:
        path = root / "plots" / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"plot:{name}")
    csv_counts: dict[str, int] = {}
    for name in (
        "nested_mask_analysis.csv", "coordinate_contributions.csv",
        "score_dilution_metrics.csv", "synthetic_real_mismatch.csv",
        "mask_seed_stability.csv", "random_mask_distribution.csv",
        "k3_exploratory_metrics.csv", "bootstrap_intervals.csv",
    ):
        with (root / name).open(newline="", encoding="utf-8") as handle:
            csv_counts[name] = sum(1 for _ in csv.DictReader(handle))
        if csv_counts[name] == 0:
            errors.append(f"empty_csv:{name}")
    if csv_counts.get("mask_seed_stability.csv") != 60:
        errors.append("mask_seed_count")
    if csv_counts.get("random_mask_distribution.csv") != 600:
        errors.append("random_mask_count")
    implementation_sha = attempt.get("implementation_sha", "")
    if len(implementation_sha) != 40 or subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_sha, "HEAD"], cwd=repo
    ).returncode:
        errors.append("implementation_not_ancestor")
    return {
        "analysis_executed": False,
        "base_verifier": base_report,
        "csv_row_counts": csv_counts,
        "errors": sorted(set(errors)),
        "files_verified": len(_load(root / "artifact_manifest_sha256.json")),
        "implementation_sha": implementation_sha,
        "no_retuning": "PASS" if "no_retuning_config" not in errors else "FAIL",
        "plots_verified": len(PLOTS),
        "schema": "pg_scc_stage0_r2_validator_repair_verification.v1",
        "scientific_verdict": verdict.get("verdict"),
        "status": "PASS" if not errors else "FAIL",
    }


def verify_fresh_clone(root: Path, revision: str) -> dict[str, Any]:
    repo = root.resolve().parents[1]
    resolved = _git(repo, "rev-parse", revision)
    temp = Path(tempfile.mkdtemp(prefix="pg-scc-r2-validator-repair-verify."))
    try:
        clone = temp / "clone"
        subprocess.run(["git", "clone", "--no-local", "--quiet", str(repo), str(clone)], check=True)
        subprocess.run(["git", "checkout", "--quiet", resolved], cwd=clone, check=True)
        command = [
            sys.executable, str(clone / "scripts/verify_pg_scc_r2_validator_repair.py"),
            "--artifact", str(clone / "artifacts/pg_scc_stage0_r2_validator_repair"),
        ]
        result = subprocess.run(command, cwd=clone, text=True, capture_output=True)
        report = json.loads(result.stdout)
        report["fresh_clone"] = True
        report["revision"] = resolved
        if result.returncode:
            report["status"] = "FAIL"
        return report
    finally:
        shutil.rmtree(temp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--fresh-clone", action="store_true")
    parser.add_argument("--implementation-sha")
    parser.add_argument("--pre-execution", action="store_true")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    if args.pre_execution:
        if not args.implementation_sha:
            parser.error("--implementation-sha is required with --pre-execution")
        report = verify_preexecution(args.artifact, args.implementation_sha)
    elif args.fresh_clone:
        report = verify_fresh_clone(args.artifact, args.revision)
    else:
        report = verify_tree(args.artifact)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
