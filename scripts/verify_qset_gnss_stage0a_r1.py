#!/usr/bin/env python3
"""Independent compact verifier for Q-SET Stage-0A R1 Galileo C-1 preflight."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_REL = Path("artifacts/qset_gnss_stage0a_r1_galileo_c1_receiver_preflight")
BASE_SHA = "3e6e17ed705bad33124cff234f36621dd782b384"
BRANCH = "research/qset-gnss-stage0a-r1-galileo-c1-receiver-preflight"
RAW_SIZE = 29_999_832_000
RAW_MD5 = "4ff0e86938792bf3150c30d5f1481917"
README_MD5 = "317e2f82dc89cfbe36272630e3c4f5e3"
OUTPUT_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/qset-gnss-stage0a-r1-galileo-c1-receiver-preflight")
VERDICTS = {
    "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD",
    "BLOCKED_C1_FORMAT_UNRESOLVED",
    "BLOCKED_GALILEO_E1_RECEIVER_NOT_AVAILABLE",
    "BLOCKED_INSUFFICIENT_GALILEO_PRN_SUPPORT",
    "BLOCKED_RECEIVER_TIME_MAPPING",
    "INCONCLUSIVE_RECEIVER_FEASIBILITY",
}
REQUIRED = (
    "README.md", "preregistration.json", "execution_freeze.json", "freeze_commit.json",
    "final_verdict.json", "source_binding.json", "format_discovery_provenance.json",
    "format_identification.json", "bounded_window_contract.json", "decoder_validation.json",
    "receiver_binary_inventory.json", "receiver_configuration_freeze.json",
    "receiver_run_manifest.json", "acquisition_inventory.csv", "per_prn_tracking_support.csv",
    "time_mapping_validation.json", "support_audit.json", "access_audit.json",
    "deterministic_reproduction.json", "artifact_manifest_sha256.json",
    "verifier_output.txt", "test_output.txt",
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def compact_manifest(artifact: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            rows.append({"path": path.relative_to(artifact).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": "gnss-doppler-lab.qset-r1-artifact-manifest.v1", "status": "PASS", "file_count": len(rows), "files": rows}


def seal(artifact: Path) -> None:
    (artifact / "artifact_manifest_sha256.json").write_text(json.dumps(compact_manifest(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_access(audit: dict[str, Any]) -> None:
    require(audit["status"] == "PASS", "access audit not PASS")
    for group in ("c3", "attack", "other_tuni2025_raw"):
        for field in ("stats", "hashes", "opens", "mmaps", "bytes_read", "downloads"):
            require(int(audit[group][field]) == 0, f"forbidden access {group}.{field}")
    c1 = audit["c1"]
    expected = RAW_SIZE * 2 + 3 * 4_194_304 + 12_000_000_000
    require(int(c1["identity_hash_passes"]) == 2, "identity hash pass count")
    require(int(c1["identity_hash_bytes"]) == RAW_SIZE * 2, "identity hash bytes")
    require(int(c1["format_window_bytes"]) == 3 * 4_194_304, "format window bytes")
    require(int(c1["receiver_decode_bytes"]) == 12_000_000_000, "receiver decode bytes")
    require(int(c1["total_payload_bytes_read_including_pre_freeze_hash"]) == expected, "total C-1 read bytes")


def validate_final(final: dict[str, Any], support: dict[str, Any], fmt: dict[str, Any], mapping: dict[str, Any]) -> None:
    require(final["status"] == "PASS" and final["verdict"] in VERDICTS, "final verdict vocabulary")
    require(final["actual_format"] == ">i2 interleaved I,Q", "final format")
    require(final["qset_training_performed"] is False, "training claim")
    require(final["threshold_calibrated"] is False, "threshold claim")
    require(final["attack_scoring_performed"] is False, "attack scoring claim")
    require(final["detection_performance_claimed"] is False, "detection claim")
    require(final["attack_download_or_access_authorized"] is False, "attack authorization")
    require(final["dynamic_panel"] == support["dynamic_panel"], "dynamic panel binding")
    require(final["acquisition_prn_count"] == support["acquisition_prn_count"], "acquisition count binding")
    require(final["tracking_ge_10s_prn_count"] == support["tracking_ge_10s_prn_count"], "tracking count binding")
    if final["verdict"] == "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD":
        require(fmt["status"] == "PASS", "ready without format pass")
        require(support["acquisition_prn_count"] >= 4 and support["tracking_ge_10s_prn_count"] >= 4, "ready without PRN gate")
        require(support["all_tracking_finite"] and support["physical_consistency_pass"], "ready without physical gate")
        require(mapping["status"] == "PASS", "ready without mapping")
        require(final["c3_clean_download_authorized"] is True and final["next_state"] == "C3_CLEAN_DOWNLOAD_ONLY", "ready scope")
    else:
        require(final["c3_clean_download_authorized"] is False and final["next_state"] == "NOT_AUTHORIZED", "blocked scope")


def validate_large(run: dict[str, Any]) -> dict[str, int]:
    require(Path(run["output_root"]) == OUTPUT_ROOT, "large root drift")
    checked_files = 0; checked_bytes = 0
    for segment in run["segments"]:
        segment_root = OUTPUT_ROOT / segment["segment_id"]
        decoder = segment_root / "c1_4msps_gr_complex.bin"
        require(decoder.is_file(), f"missing decoder output: {decoder}")
        require(decoder.stat().st_size == int(segment["decoder"]["output_size_bytes"]), "decoder size tamper")
        require(sha256_file(decoder) == segment["decoder"]["output_sha256"], "decoder hash tamper")
        checked_files += 1; checked_bytes += decoder.stat().st_size
        receiver_root = segment_root / "receiver"
        for row in segment["receiver"]["output_set"]["files"]:
            path = receiver_root / row["path"]
            require(path.is_file() and path.stat().st_size == int(row["size_bytes"]), f"large output size: {path}")
            require(sha256_file(path) == row["sha256"], f"large output hash: {path}")
            checked_files += 1; checked_bytes += path.stat().st_size
    return {"large_files_checked": checked_files, "large_bytes_checked": checked_bytes}


def validate_artifact(repo: Path, verify_large: bool = False) -> dict[str, Any]:
    artifact = repo / ARTIFACT_REL
    missing = [name for name in REQUIRED if not (artifact / name).is_file()]
    require(not missing, f"missing artifacts: {missing}")
    source = load_json(artifact / "source_binding.json")
    require(source["base_sha"] == BASE_SHA, "base binding")
    require(source["c1"]["size_bytes"] == RAW_SIZE and source["c1"]["md5"] == RAW_MD5, "raw identity")
    require(source["c1"]["readme"]["md5"] == README_MD5, "README identity")
    require(source["c1"]["zenodo_record_id"] == 15470143 and str(source["c1"]["concept_record_id"]) == "15470142", "Zenodo identity")
    for relative, expected in source["code_bindings"].items():
        require(sha256_file(repo / relative) == expected, f"frozen code drift: {relative}")
    provenance = load_json(artifact / "format_discovery_provenance.json")
    require(provenance["not_preregistered_by_r1"] is True and provenance["status"] == "HISTORICAL_PRE_R1_CLEAN_ONLY_OBSERVATION", "format provenance")
    bounded = load_json(artifact / "bounded_window_contract.json")
    require(len(bounded["format_windows"]) == 3 and len(bounded["receiver_windows"]) == 2, "bounded windows")
    require(bounded["full_150s_conversion_forbidden"] is True, "full conversion contract")
    fmt = load_json(artifact / "format_identification.json")
    require(fmt["status"] == "PASS" and fmt["selected_format"] == ">i2 interleaved I,Q", "format result")
    require(fmt["raw_size_bytes"] == RAW_SIZE and fmt["complex_sample_count"] == 7_499_958_000, "format counts")
    decoder = load_json(artifact / "decoder_validation.json")
    require(decoder["status"] == "PASS" and decoder["input_dtype"] == ">i2" and decoder["interleaving"] == "I,Q", "decoder contract")
    require(decoder["unit_vector_pass"] and decoder["streaming_round_trip_pass"], "decoder validation")
    config = load_json(artifact / "receiver_configuration_freeze.json")
    require(config["status"] == "FROZEN_PRE_EXECUTION" and config["channels_capacity"] == 12, "receiver config freeze")
    require(config["dynamic_panel_contract"].startswith("observed PRNs only"), "dynamic panel contract")
    run = load_json(artifact / "receiver_run_manifest.json")
    require(run["status"] == "PASS" and run["worker_count"] == 1 and len(run["segments"]) == 2, "receiver run")
    require(all(item["receiver"]["exit_code"] == 0 and item["receiver"]["terminal_eof"] for item in run["segments"]), "receiver exits")
    support = load_json(artifact / "support_audit.json")
    require(support["status"] == "PASS", "support audit")
    panel = support["dynamic_panel"]
    require(panel == sorted(set(panel)) and all(1 <= int(prn) <= 36 for prn in panel), "dynamic panel values")
    for segment, mask in support["availability_mask"].items():
        require(sorted(int(prn) for prn in mask) == panel, f"mask keys {segment}")
    acquisition = csv_rows(artifact / "acquisition_inventory.csv")
    tracking = csv_rows(artifact / "per_prn_tracking_support.csv")
    require(len(acquisition) == support["acquisition_prn_count"] == len(tracking), "CSV PRN counts")
    require(sum(row["tracking_ge_10s"] == "True" for row in tracking) == support["tracking_ge_10s_prn_count"], "CSV 10-second count")
    mapping = load_json(artifact / "time_mapping_validation.json")
    deterministic = load_json(artifact / "deterministic_reproduction.json")
    require(deterministic["status"] == "PASS" and deterministic["analysis_runs"] == 2 and deterministic["byte_identical"] is True, "deterministic analysis")
    access = load_json(artifact / "access_audit.json"); validate_access(access)
    final = load_json(artifact / "final_verdict.json"); validate_final(final, support, fmt, mapping)
    freeze = load_json(artifact / "freeze_commit.json")
    require(freeze["status"] == "PASS" and final["freeze_sha"] == freeze["commit_sha"] == run["freeze_sha"], "freeze SHA binding")
    manifest = load_json(artifact / "artifact_manifest_sha256.json")
    require(manifest == compact_manifest(artifact), "compact manifest mismatch")
    result = {
        "status": "PASS", "verdict": final["verdict"], "base_sha": BASE_SHA,
        "freeze_sha": freeze["commit_sha"], "manifest_files": manifest["file_count"],
        "c1_bytes_read": access["c1"]["total_payload_bytes_read_including_pre_freeze_hash"],
        "c3_bytes_read": 0, "attack_bytes_read": 0,
        "acquisition_prn_count": final["acquisition_prn_count"],
        "tracking_ge_10s_prn_count": final["tracking_ge_10s_prn_count"],
    }
    if verify_large: result.update(validate_large(run))
    return result


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def verify(repo: Path, verify_large: bool = False) -> dict[str, Any]:
    result = validate_artifact(repo, verify_large)
    head = git(repo, "rev-parse", "HEAD"); remote = git(repo, "rev-parse", f"origin/{BRANCH}")
    require(head == remote, "local/remote SHA mismatch")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, head], cwd=repo).returncode == 0, "base ancestry")
    require(git(repo, "status", "--porcelain") == "", "worktree not clean")
    result["final_sha"] = head
    return result


def self_test(repo: Path) -> dict[str, Any]:
    artifact = repo / ARTIFACT_REL; tests: dict[str, bool] = {}
    manifest = load_json(artifact / "artifact_manifest_sha256.json")
    target = artifact / "README.md"; original = target.read_bytes()
    try:
        target.write_bytes(original + b"tamper")
        tests["artifact_manifest_tamper"] = manifest != compact_manifest(artifact)
    finally:
        target.write_bytes(original)
    access = load_json(artifact / "access_audit.json"); access["attack"]["bytes_read"] = 1
    try: validate_access(access); tests["attack_access_tamper"] = False
    except VerificationError: tests["attack_access_tamper"] = True
    final = load_json(artifact / "final_verdict.json"); final["attack_download_or_access_authorized"] = True
    try: validate_final(final, load_json(artifact / "support_audit.json"), load_json(artifact / "format_identification.json"), load_json(artifact / "time_mapping_validation.json")); tests["attack_authorization_tamper"] = False
    except VerificationError: tests["attack_authorization_tamper"] = True
    require(all(tests.values()), f"self-test failure: {tests}")
    return {"status": "PASS", "tests": tests, "test_count": len(tests)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT); parser.add_argument("--seal", action="store_true")
    parser.add_argument("--self-test", action="store_true"); parser.add_argument("--verify-large", action="store_true")
    args = parser.parse_args(); repo = args.repo.resolve()
    if args.seal:
        seal(repo / ARTIFACT_REL); print("PASS: compact manifest sealed"); return 0
    result = verify(repo, args.verify_large)
    if args.self_test: result["self_test"] = self_test(repo)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except VerificationError as error: print(f"FAIL: {error}"); raise SystemExit(1)
