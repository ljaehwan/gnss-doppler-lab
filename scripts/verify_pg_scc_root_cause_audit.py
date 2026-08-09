#!/usr/bin/env python3
"""Independent read-only verifier for PG-SCC root-cause audit artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/pg_scc_stage0_r1_root_cause_audit"
CONFIG_SHA256 = "ade700892fe3e055e0154ae859ba701a36f11a6ec9245db4b39fd087426eb0e6"
SOURCE_SHA256 = "13ab9e544f06903dd6fde04c42e8aefb35a66946bbaf0456e95040fbb777d428"
REQUIRED = {
    "README.md", "config.json", "source_commit.json", "reproduction_check.json",
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

def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))



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
        report = verify_tree(clone / "artifacts/pg_scc_stage0_r1_root_cause_audit", require_git=True)
        report["fresh_clone"] = True
        report["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip()
        return report
    finally:
        shutil.rmtree(temp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--fresh-clone", action="store_true")
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--no-git-schema", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fresh_clone:
        report = verify_fresh_clone(ROOT, args.revision)
    else:
        report = verify_tree(args.artifact, require_git=not args.no_git_schema)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
