#!/usr/bin/env python3
"""Verifier for PG-SCC R2 r1 identity repair cycle 2."""
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
DEFAULT_ARTIFACT = ROOT / "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle2"
BRANCH = "research/pg-scc-stage0-r2-r1-identity-repair"
BASE_SHA = "9839823c00cafc34fbbf1d6b1dbe069eb2c4e74d"
PREREGISTRATION_SHA = "a4603a801328666cdf70784154132e213d6d25f6"
CONFIG_SHA256 = "336802a95e82df1da82822520fe8bd838bf18ce17da6ae29aa5695449f3b67f5"
SOURCE_SHA256 = "571b80ec11a9f860317f84c5d1808fddda270e988dfb8d25948df0999de0f8a4"
R1_BINDING_SHA256 = "550f3fde25742b571fa0a5206a96d0454300d1fe1732671b9bf655ccbb3f379f"
PREDECESSOR_ATTEMPT_STATE_SHA256 = "23c5a418ca63fe56b821cba1ebd16efbd2ed5a3ae992e37fc9496ba2f5d9f1fe"
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
    return module


def _load_runner():
    path = ROOT / "scripts/run_pg_scc_root_cause_audit.py"
    spec = importlib.util.spec_from_file_location("pg_scc_r2_identity_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _campaign_verdict(root_cause_verdict: str) -> str:
    return {
        "K3_WORTH_INDEPENDENT_CONFIRMATION": "GO",
        "TERMINATE_PG_SCC": "NO_GO",
        "REPAIRABLE_BUT_REQUIRES_NEW_CONFIRMATION": "INCONCLUSIVE",
    }.get(root_cause_verdict, "INCONCLUSIVE")


def _manifest_errors(root: Path) -> list[str]:
    errors: list[str] = []
    manifest = _load(root / "implementation_manifest_sha256.json")
    if manifest.get("protected_score_fields_read_before_freeze") != 0:
        errors.append("implementation_manifest_protected_access")
    repo = root.parents[1]
    for relative, expected in manifest.get("files", {}).items():
        path = repo / relative
        if not path.is_file() or _digest(path) != expected:
            errors.append(f"implementation_hash:{relative}")
        try:
            committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=repo)
        except subprocess.CalledProcessError:
            errors.append(f"implementation_uncommitted:{relative}")
        else:
            if hashlib.sha256(committed).hexdigest() != expected:
                errors.append(f"implementation_head_hash:{relative}")
    return errors


def verify_preexecution(root: Path, implementation_sha: str) -> dict[str, Any]:
    root = root.resolve()
    repo = root.parents[1]
    errors: list[str] = []
    required = {
        "config.json", "source_commit.json", "r1_failure_binding.json",
        "support_preflight.json", "preregistration.json",
        "predecessor_failure_binding.json", "semantic_diff_audit.json",
        "implementation_manifest_sha256.json", "pre_run_state.json",
        "zero_protected_access.json",
    }
    errors.extend(f"missing:{name}" for name in sorted(required) if not (root / name).is_file())
    if errors:
        return {"schema": "pg_scc_stage0_r2_r1_identity_repair_cycle2_preexecution.v1", "status": "FAIL", "errors": errors}
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
    zero_access = _load(root / "zero_protected_access.json")
    campaign = prereg.get("campaign_identity", {})
    if (
        prereg.get("cycle") != 2
        or campaign.get("branch") != BRANCH
        or campaign.get("required_base_sha") != BASE_SHA
        or campaign.get("artifact_namespace") != str(root.relative_to(repo))
    ):
        errors.append("preregistration_identity")
    predecessor_metadata = predecessor.get("authorized_failure_metadata_only", {})
    if (
        predecessor_metadata.get("attempt_state_sha256") != PREDECESSOR_ATTEMPT_STATE_SHA256
        or predecessor_metadata.get("protected_loader_invoked") is not True
        or predecessor_metadata.get("protected_attempt_count") != 1
    ):
        errors.append("predecessor_binding")
    if (
        semantic.get("status") != "PASS"
        or not semantic.get("non_plot_scientific_functions_equivalent")
        or semantic.get("function_ast_equivalence", {}).get("main", {}).get(
            "manifest_before_schema_validation"
        ) is not True
    ):
        errors.append("semantic_diff")
    if support.get("status") != "PASS" or support.get("protected_score_fields_projected_or_read") != 0:
        errors.append("support_preflight")
    try:
        expected_support = _load_runner().build_metadata_support_preflight_report()
    except Exception as exc:
        errors.append(f"support_reconstruction:{exc}")
    else:
        if support != expected_support:
            errors.append("support_exact_reconstruction")
    if (
        zero_access.get("protected_loader_invoked_for_cycle2_diagnosis") is not False
        or zero_access.get("protected_score_fields_projected_or_read_for_cycle2_diagnosis") != 0
        or zero_access.get("partial_cycle1_scientific_outputs_inspected") is not False
        or zero_access.get("partial_cycle1_scientific_outputs_reused") is not False
    ):
        errors.append("zero_protected_access")
    if (
        pre_run.get("state") != "READY"
        or pre_run.get("implementation_sha") != head
        or pre_run.get("remote_sha") != head
        or pre_run.get("ahead_behind") != [0, 0]
        or pre_run.get("protected_loader_invoked") is not False
    ):
        errors.append("pre_run_state")
    errors.extend(_manifest_errors(root))
    dirty = []
    for line in _git(repo, "status", "--porcelain=v1", "--untracked-files=all").splitlines():
        relative = line[3:]
        if relative.startswith("artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle2/"):
            continue
        dirty.append(line)
    if dirty:
        errors.append("dirty_outside_cycle_namespace:" + "|".join(dirty))
    return {
        "ahead_behind": [int(value) for value in _git(repo, "rev-list", "--left-right", "--count", f"{head}...{remote}").split()],
        "analysis_executed": False,
        "errors": sorted(set(errors)),
        "implementation_sha": head,
        "protected_score_fields_read": 0,
        "schema": "pg_scc_stage0_r2_r1_identity_repair_cycle2_preexecution.v1",
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
        "zero_protected_access.json",
    }
    errors.extend(f"missing:{name}" for name in sorted(required) if not (root / name).is_file())
    if errors:
        return {"schema": "pg_scc_stage0_r2_r1_identity_repair_cycle2_verification.v1", "status": "FAIL", "errors": sorted(set(errors))}
    prereg = _load(root / "preregistration.json")
    predecessor = _load(root / "predecessor_failure_binding.json")
    pre_run = _load(root / "pre_run_state.json")
    semantic = _load(root / "semantic_diff_audit.json")
    attempt = _load(root / "attempt_state.json")
    verdict = _load(root / "root_cause_verdict.json")
    campaign = prereg.get("campaign_identity", {})
    if (
        prereg.get("cycle") != 2
        or campaign.get("branch") != BRANCH
        or campaign.get("required_base_sha") != BASE_SHA
    ):
        errors.append("preregistration_identity")
    metadata = predecessor.get("authorized_failure_metadata_only", {})
    if (
        metadata.get("protected_attempt_count") != 1
        or metadata.get("attempt_state_sha256") != PREDECESSOR_ATTEMPT_STATE_SHA256
        or metadata.get("protected_loader_invoked") is not True
    ):
        errors.append("predecessor_binding")
    if (
        semantic.get("status") != "PASS"
        or not semantic.get("non_plot_scientific_functions_equivalent")
        or semantic.get("function_ast_equivalence", {}).get("main", {}).get(
            "manifest_before_schema_validation"
        ) is not True
    ):
        errors.append("semantic_diff")
    if (
        attempt.get("attempt_count") != 1
        or attempt.get("state") != "COMPLETED"
        or attempt.get("exit_code") != 0
        or attempt.get("protected_loader_invoked") is not True
    ):
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
        "root_cause_verdict": verdict.get("verdict"),
        "schema": "pg_scc_stage0_r2_r1_identity_repair_cycle2_verification.v1",
        "scientific_verdict": _campaign_verdict(str(verdict.get("verdict", ""))),
        "status": "PASS" if not errors else "FAIL",
    }


def verify_fresh_clone(root: Path, revision: str) -> dict[str, Any]:
    repo = root.resolve().parents[1]
    resolved = _git(repo, "rev-parse", revision)
    temp = Path(tempfile.mkdtemp(prefix="pg-scc-r2-r1-identity-cycle2-verify."))
    try:
        clone = temp / "clone"
        subprocess.run(["git", "clone", "--no-local", "--quiet", str(repo), str(clone)], check=True)
        subprocess.run(["git", "checkout", "--quiet", resolved], cwd=clone, check=True)
        command = [
            sys.executable, str(clone / "scripts/verify_pg_scc_r2_r1_identity_cycle2.py"),
            "--artifact", str(clone / "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle2"),
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
