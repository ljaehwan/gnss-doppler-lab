#!/usr/bin/env python3
"""Verify the dataset-blocked Q-SET-GNSS Stage-0A artifact without data access."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_REL = Path("artifacts/qset_gnss_stage0a_partial_prn_quantile_feasibility")
BASE_SHA = "8025f331444e600bee97baf28ad3cb7af9410381"
BRANCH = "research/qset-gnss-stage0a-partial-prn-quantile-feasibility"
VERDICT = "BLOCKED_TUNI2025_DATASET_NOT_LOCAL"
PREFLIGHT_COMMIT_SHA = "617a897c9709a5ca46471a99c3906301af9e798e"
REQUIRED = (
    "README.md", "dataset_preflight.json", "dataset_download_plan.json",
    "source_binding.json", "preflight_commit.json", "preregistration.json",
    "preregistration_commit.json", "execution_freeze.json", "freeze_commit.json",
    "receiver_manifests/status.json", "per_prn_support.csv",
    "clean_score_summary.json", "threshold_binding.json",
    "synthetic_dilution_control.json", "scenario_metrics.json",
    "aggregator_comparison.csv", "per_prn_ground_truth_metrics.csv",
    "shortcut_audit.json", "access_audit.json", "deterministic_reproduction.json",
    "final_verdict.json", "plots/dataset_availability.svg",
    "verifier_output.txt", "test_output.txt", "artifact_manifest_sha256.json",
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def compact_manifest(artifact: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            rows.append({"path": path.relative_to(artifact).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": "gnss-doppler-lab.qset-stage0a-manifest.v1", "status": "PASS", "file_count": len(rows), "files": rows}


def seal(artifact: Path) -> None:
    (artifact / "artifact_manifest_sha256.json").write_text(
        json.dumps(compact_manifest(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_access(audit: dict[str, Any]) -> None:
    fields = ("stats", "hashes", "opens", "mmaps", "bytes_read", "downloads", "bytes_downloaded")
    for group in ("raw_iq", "attack_payload", "clean_payload"):
        for field in fields:
            require(int(audit[group].get(field, 0)) == 0, f"nonzero {group}.{field}")
    require(int(audit["raw_iq"]["decodes"]) == 0, "raw decode occurred")
    require(int(audit["attack_payload"]["signal_statistics"]) == 0, "attack statistic occurred")
    require(int(audit["clean_payload"]["bounded_decode_attempts"]) == 0, "clean decode occurred")
    require(all(int(value) == 0 for value in audit["scientific_operations"].values()), "scientific operation occurred")
    require(all(int(value) == 0 for value in audit["forbidden_inputs"].values()), "forbidden feature input")


def validate_download_plan(plan: dict[str, Any]) -> None:
    files = plan["exact_1_3_5_minimum_raw_files"]
    require([row["scenario"] for row in files] == ["C-1", "C-3", "SS-1", "SS-3", "SS-5"], "minimum scenario set")
    total = sum(int(row["size_bytes"]) for row in files)
    require(len(files) == 5 and total == 141925248000, "minimum download total")
    require(total == int(plan["minimum_totals"]["raw_bytes"]), "reported download total")
    require(plan["automatic_download_performed"] is False, "automatic download flag")
    require(all(int(row["size_bytes"]) > int(plan["large_file_threshold_bytes"]) for row in files), "large-file threshold")
    require(plan["minimum_totals"]["safe_headroom_status"] == "INSUFFICIENT_FOR_RAW_PLUS_PLANNING_RESERVE", "headroom gate")


def validate_final(final: dict[str, Any]) -> None:
    require(final["verdict"] == VERDICT, "verdict mismatch")
    require(final["stage0b_authorized"] is False, "Stage-0B authorization")
    require(final["scientific_preregistration_created"] is False, "scientific preregistration claim")
    require(final["implementation_freeze_created"] is False, "freeze claim")
    require(final["attack_evaluation_performed"] is False, "attack evaluation claim")
    require(final["spoofing_detector_validated"] is False, "validation claim")
    require(final["next_state"] == "WAITING_FOR_TUNI2025_DATASET_AND_PROTOCOL_RESOLUTION", "next state")


def validate_artifact(artifact: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED if not (artifact / name).is_file()]
    require(not missing, f"missing artifacts: {missing}")
    preflight = load_json(artifact / "dataset_preflight.json")
    require(preflight["base_sha"] == BASE_SHA and preflight["verdict"] == VERDICT, "preflight binding")
    require(preflight["local_preflight"]["local_tuni2025_raw_file_count"] == 0, "local raw count")
    require(preflight["local_preflight"]["local_tuni2025_raw_bytes"] == 0, "local raw bytes")
    require(preflight["selection_conclusion"]["gps_l1_priority_feasible_for_exact_1_3_5_contract"] is False, "GPS exact-k conclusion")
    source = load_json(artifact / "source_binding.json")
    require(source["base"]["sha"] == BASE_SHA, "base source binding")
    require(source["readme_verification"]["document_count"] == 13, "README count")
    require(source["readme_verification"]["all_downloaded_md5_match_official_metadata"] is True, "README MD5")
    require(source["readme_verification"]["raw_iq_downloaded_or_hashed"] is False, "raw source access")
    require(load_json(artifact / "preflight_commit.json")["commit_sha"] == PREFLIGHT_COMMIT_SHA, "preflight commit binding")
    validate_download_plan(load_json(artifact / "dataset_download_plan.json"))
    validate_access(load_json(artifact / "access_audit.json"))
    validate_final(load_json(artifact / "final_verdict.json"))
    blocked = ("preregistration.json", "execution_freeze.json", "clean_score_summary.json", "threshold_binding.json", "synthetic_dilution_control.json", "scenario_metrics.json")
    require(all(load_json(artifact / name)["status"] == "NOT_RUN_DATASET_BLOCKED" for name in blocked), "blocked status")
    deterministic = load_json(artifact / "deterministic_reproduction.json")
    require(deterministic["status"] == "PASS" and deterministic["run_count"] == 2 and deterministic["byte_identical"] is True, "deterministic verification")
    manifest = load_json(artifact / "artifact_manifest_sha256.json")
    require(manifest == compact_manifest(artifact), "manifest mismatch")
    return {"status": "PASS", "verdict": VERDICT, "base_sha": BASE_SHA, "preflight_commit_sha": PREFLIGHT_COMMIT_SHA, "raw_iq_bytes_read": 0, "attack_bytes_read": 0, "stage0b_authorized": False, "manifest_files": manifest["file_count"], "minimum_download_bytes": 141925248000}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def verify(repo: Path) -> dict[str, Any]:
    result = validate_artifact(repo / ARTIFACT_REL)
    source = load_json(repo / ARTIFACT_REL / "source_binding.json")
    for relative, expected in source["compact_verifier_binding"].items():
        require(sha256_file(repo / relative) == expected, f"code binding mismatch: {relative}")
    head, remote = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", f"origin/{BRANCH}")
    require(head == remote, "local/remote SHA mismatch")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, head], cwd=repo).returncode == 0, "base ancestry")
    result["final_sha"] = head
    return result


def self_test(repo: Path) -> dict[str, Any]:
    artifact, tests = repo / ARTIFACT_REL, {}
    with tempfile.TemporaryDirectory(prefix="qset-stage0a-") as temporary:
        temp = Path(temporary); (temp / "x").write_text("x"); before = compact_manifest(temp); (temp / "x").write_text("tampered")
        tests["manifest_tamper"] = before != compact_manifest(temp)
    access = load_json(artifact / "access_audit.json"); changed = copy.deepcopy(access); changed["attack_payload"]["opens"] = 1
    try: validate_access(changed); tests["attack_access_tamper"] = False
    except VerificationError: tests["attack_access_tamper"] = True
    plan = load_json(artifact / "dataset_download_plan.json"); changed = copy.deepcopy(plan); changed["exact_1_3_5_minimum_raw_files"][0]["size_bytes"] += 1
    try: validate_download_plan(changed); tests["download_total_tamper"] = False
    except VerificationError: tests["download_total_tamper"] = True
    final = load_json(artifact / "final_verdict.json"); changed = copy.deepcopy(final); changed["stage0b_authorized"] = True
    try: validate_final(changed); tests["authorization_tamper"] = False
    except VerificationError: tests["authorization_tamper"] = True
    tests["duplicate_gps_clean_detected"] = load_json(artifact / "dataset_preflight.json")["scenario_role_inventory"][6]["independence"] == "FAIL_BYTE_IDENTICAL_OFFICIAL_MD5"
    tests["exact_1_3_5_set"] = [row["scenario"] for row in plan["exact_1_3_5_minimum_raw_files"]] == ["C-1", "C-3", "SS-1", "SS-3", "SS-5"]
    require(all(tests.values()), f"self-test failure: {tests}")
    return {"status": "PASS", "test_count": len(tests), "tests": tests}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT); parser.add_argument("--seal", action="store_true"); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(); repo = args.repo.resolve()
    if args.seal: seal(repo / ARTIFACT_REL); print("PASS: manifest sealed"); return 0
    result = verify(repo)
    if args.self_test: result["self_test"] = self_test(repo)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except VerificationError as error: print(f"FAIL: {error}"); raise SystemExit(1)
