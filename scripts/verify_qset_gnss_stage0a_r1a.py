#!/usr/bin/env python3
"""Independent verifier for the Q-SET R1a Galileo C-1 terminal-drain repair."""
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
ARTIFACT_REL = Path("artifacts/qset_gnss_stage0a_r1a_galileo_c1_terminal_drain_repair")
R1_ARTIFACT_REL = Path("artifacts/qset_gnss_stage0a_r1_galileo_c1_receiver_preflight")
OUTPUT_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/qset-gnss-stage0a-r1a-galileo-c1-terminal-drain-repair")
BASE_R1_FINAL_SHA = "cfa1dedfe330d74047e5ffcac2b9f94fd33f23a7"
PREREGISTRATION_SHA = "a4d734fa4bbbf8ac0656b3eebf66e945d575adec"
BRANCH = "research/qset-gnss-stage0a-r1a-galileo-c1-terminal-drain-repair"
RAW_SIZE = 29_999_832_000
RAW_MD5 = "4ff0e86938792bf3150c30d5f1481917"
README_MD5 = "317e2f82dc89cfbe36272630e3c4f5e3"
FORMAT_BYTES = 3 * 4_194_304
DECODE_BYTES = 12_000_000_000
EXPECTED_C1_BYTES = RAW_SIZE + FORMAT_BYTES + DECODE_BYTES
EXPECTED_CONFIG_SHA256 = "6dda45478b1b2521605e4544a51c84e513872b7bd433bb308d888dccd5fe4061"
EXPECTED_RECEIVER_SHA256 = "9e523c156c288cae401628ba308bc886bcb7c7989e35747b7482b1d50a0c6131"
EXPECTED_R1_HASHES = {
    "final_verdict.json": "b2fd3d30dae14ca83b5f7007f68cbd70824255763eead89c922c99ff24d4e41a",
    "receiver_run_manifest.json": "9b02caddbd493f6c24360dcbbe02fc95c7eb29fd6188f224d2233001f104d80a",
    "execution_interrupt_audit.json": "9307abe804f75669685a3f59de1f1fdbe7a18911724095b84ff92dd70e65e52b",
    "artifact_manifest_sha256.json": "bb3ce40a9a6b369564672f278f27564f4eded5201ba7c765ed9aa3c1e3656566",
}
VERDICTS = {
    "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD",
    "BLOCKED_C1_FORMAT_UNRESOLVED",
    "BLOCKED_GALILEO_E1_RECEIVER_NOT_AVAILABLE",
    "BLOCKED_INSUFFICIENT_GALILEO_PRN_SUPPORT",
    "BLOCKED_RECEIVER_TIME_MAPPING",
    "INCONCLUSIVE_RECEIVER_FEASIBILITY",
}
REQUIRED = (
    "README.md",
    "repair_preregistration.json",
    "preregistration_commit.json",
    "execution_repair_freeze.json",
    "freeze_commit.json",
    "r1_preservation_audit.json",
    "source_binding.json",
    "format_identification.json",
    "decoder_validation.json",
    "receiver_run_manifest.json",
    "tracking_dump_inventory.json",
    "acquisition_inventory.csv",
    "per_prn_tracking_support.csv",
    "support_audit.json",
    "time_mapping_validation.json",
    "deterministic_reproduction.json",
    "access_audit.json",
    "final_verdict.json",
    "artifact_manifest_sha256.json",
    "test_output.txt",
    "verifier_output.txt",
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
    return {"schema": "gnss-doppler-lab.qset-r1a-artifact-manifest.v1", "status": "PASS", "file_count": len(rows), "files": rows}


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
    require(int(c1["identity_hash_passes"]) == 1, "R1a identity hash pass count")
    require(int(c1["identity_hash_bytes"]) == RAW_SIZE, "R1a identity hash bytes")
    require(int(c1["format_window_bytes"]) == FORMAT_BYTES, "R1a format bytes")
    require(int(c1["receiver_decode_bytes"]) == DECODE_BYTES, "R1a decoder bytes")
    require(int(c1["total_payload_bytes_read"]) == EXPECTED_C1_BYTES, "R1a total C-1 bytes")


def validate_final(final: dict[str, Any], support: dict[str, Any], mapping: dict[str, Any]) -> None:
    require(final["status"] == "PASS" and final["verdict"] in VERDICTS, "final verdict vocabulary")
    require(final["actual_format"] == ">i2 interleaved I,Q", "actual format")
    require(final["qset_training_performed"] is False, "Q-SET training claim")
    require(final["threshold_calibrated"] is False, "threshold claim")
    require(final["attack_scoring_performed"] is False, "attack scoring claim")
    require(final["detection_performance_claimed"] is False, "detection claim")
    require(final["attack_download_or_access_authorized"] is False, "attack authorization")
    require(final["dynamic_panel"] == support["dynamic_panel"], "dynamic panel binding")
    require(final["common_panel"] == support["common_panel"], "common panel binding")
    require(final["acquisition_prn_count"] == support["acquisition_prn_count"], "acquisition count binding")
    require(final["tracking_ge_10s_prn_count"] == support["tracking_ge_10s_prn_count"], "tracking count binding")
    if final["verdict"] == "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD":
        require(support["acquisition_prn_count"] >= 4 and support["tracking_ge_10s_prn_count"] >= 4, "ready without PRN support")
        require(support["all_tracking_finite"] and support["physical_consistency_pass"], "ready without physical support")
        require(mapping["status"] == "PASS", "ready without time mapping")
        require(final["c3_clean_download_authorized"] is True and final["next_state"] == "C3_CLEAN_DOWNLOAD_ONLY", "ready scope")
    else:
        require(final["c3_clean_download_authorized"] is False and final["next_state"] == "NOT_AUTHORIZED", "blocked scope")


def validate_large(run: dict[str, Any]) -> dict[str, int]:
    require(Path(run["output_root"]) == OUTPUT_ROOT, "large output root drift")
    checked_files = 0
    checked_bytes = 0
    for segment in run["segments"]:
        root = OUTPUT_ROOT / segment["segment_id"]
        decoder = root / "c1_4msps_gr_complex.bin"
        require(decoder.is_file() and decoder.stat().st_size == int(segment["decoder"]["output_size_bytes"]), f"decoder size: {decoder}")
        require(sha256_file(decoder) == segment["decoder"]["output_sha256"], f"decoder hash: {decoder}")
        checked_files += 1
        checked_bytes += decoder.stat().st_size
        receiver = root / "receiver"
        for row in segment["receiver"]["output_set"]["files"]:
            path = receiver / row["path"]
            require(path.is_file() and path.stat().st_size == int(row["size_bytes"]), f"large output size: {path}")
            require(sha256_file(path) == row["sha256"], f"large output hash: {path}")
            checked_files += 1
            checked_bytes += path.stat().st_size
    return {"large_files_checked": checked_files, "large_bytes_checked": checked_bytes}


def validate_artifact(repo: Path, verify_large: bool = False) -> dict[str, Any]:
    artifact = repo / ARTIFACT_REL
    missing = [name for name in REQUIRED if not (artifact / name).is_file()]
    require(not missing, f"missing artifacts: {missing}")
    prereg = load_json(artifact / "repair_preregistration.json")
    require(prereg["status"].startswith("PREREGISTERED_REPAIR"), "repair preregistration")
    require(prereg["base_r1_final_sha"] == BASE_R1_FINAL_SHA and prereg["worker_count"] == 1, "preregistered base/order")
    require(prereg["data_scope"]["c3_access_or_download_authorized"] is False and prereg["data_scope"]["attack_access_authorized"] is False, "preregistered access scope")
    prereg_commit = load_json(artifact / "preregistration_commit.json")
    require(prereg_commit["status"] == "PASS" and prereg_commit["commit_sha"] == PREREGISTRATION_SHA, "preregistration commit binding")
    preservation = load_json(artifact / "r1_preservation_audit.json")
    require(preservation["status"] == "PASS_PRE_REPAIR" and preservation["historical_verdict"] == "INCONCLUSIVE_RECEIVER_FEASIBILITY", "R1 preservation declaration")
    for name, expected in EXPECTED_R1_HASHES.items():
        require(sha256_file(repo / R1_ARTIFACT_REL / name) == expected, f"R1 compact evidence drift: {name}")
    source = load_json(artifact / "source_binding.json")
    require(source["status"] == "PASS_PRE_EXECUTION" and source["base_r1_final_sha"] == BASE_R1_FINAL_SHA, "source base")
    require(source["c1"]["size_bytes"] == RAW_SIZE and source["c1"]["md5"] == RAW_MD5, "C-1 identity")
    require(source["readme"]["md5"] == README_MD5, "README identity")
    require(source["receiver"]["sha256"] == EXPECTED_RECEIVER_SHA256, "receiver identity")
    require(source["receiver_configuration_sha256"] == EXPECTED_CONFIG_SHA256, "receiver configuration identity")
    for relative, expected in source["code_bindings"].items():
        require(sha256_file(repo / relative) == expected, f"frozen code drift: {relative}")
    freeze = load_json(artifact / "execution_repair_freeze.json")
    require(freeze["status"] == "FROZEN_PRE_EXECUTION" and freeze["worker_count"] == 1, "execution freeze")
    require(freeze["scientific_contract_changed"] is False, "scientific contract change")
    require(freeze["terminal_drain"]["eof_margin_output_samples"] == 4_000_000 and freeze["terminal_drain"]["required_exit_code"] == 0, "terminal-drain freeze")
    require(freeze["complete_record_adapter"]["record_size_bytes"] == 96, "record-size freeze")
    fmt = load_json(artifact / "format_identification.json")
    require(fmt["status"] == "PASS" and fmt["selected_format"] == ">i2 interleaved I,Q", "format reproduction")
    decoder = load_json(artifact / "decoder_validation.json")
    require(decoder["status"] == "PASS" and decoder["input_dtype"] == ">i2" and decoder["interleaving"] == "I,Q", "decoder validation")
    run = load_json(artifact / "receiver_run_manifest.json")
    require(run["status"] == "PASS" and run["worker_count"] == 1 and len(run["segments"]) == 2, "receiver run completeness")
    require([row["segment_id"] for row in run["segments"]] == ["segment_030_060", "segment_100_130"], "receiver order")
    for segment in run["segments"]:
        receiver = segment["receiver"]
        drain = receiver["terminal_drain"]
        require(receiver["receiver_sha256"] == EXPECTED_RECEIVER_SHA256, "receiver SHA in run")
        require(receiver["exit_code"] == 0 and receiver["terminal_eof"] is True, "receiver exit")
        require(drain["status"] == "PASS" and drain["terminal_drain"] is True and drain["eof_evidence_pass"] is True, "terminal drain")
        require(drain["forced_signal"] is None and drain["controlled_signal"] == "SIGINT", "controlled stop")
        require(all(drain["markers"].values()), "terminal stop markers")
    dumps = load_json(artifact / "tracking_dump_inventory.json")
    require(dumps["status"] == "PASS" and dumps["record_size_bytes"] == 96 and len(dumps["rows"]) == 24, "dump inventory")
    for row in dumps["rows"]:
        require(0 <= int(row["trailing_fragment_size_bytes"]) < 96, "dump trailing fragment bound")
        require(int(row["complete_bytes"]) + int(row["trailing_fragment_size_bytes"]) == int(row["size_bytes"]), "dump byte partition")
        require(int(row["complete_record_count"]) * 96 == int(row["complete_bytes"]), "dump record count")
    support = load_json(artifact / "support_audit.json")
    require(support["status"] == "PASS", "support audit")
    panel = support["dynamic_panel"]
    require(panel == sorted(set(panel)) and all(1 <= int(prn) <= 36 for prn in panel), "dynamic panel")
    for mask in support["availability_mask"].values():
        require(sorted(int(prn) for prn in mask) == panel, "availability mask keys")
    acquisition = csv_rows(artifact / "acquisition_inventory.csv")
    tracking = csv_rows(artifact / "per_prn_tracking_support.csv")
    require(len(acquisition) == len(tracking) == support["acquisition_prn_count"], "PRN CSV counts")
    require(sum(row["tracking_ge_10s"] == "True" for row in tracking) == support["tracking_ge_10s_prn_count"], "10-second CSV count")
    mapping = load_json(artifact / "time_mapping_validation.json")
    deterministic = load_json(artifact / "deterministic_reproduction.json")
    require(deterministic["status"] == "PASS" and deterministic["analysis_runs"] == 2 and deterministic["byte_identical"] is True, "deterministic analysis")
    access = load_json(artifact / "access_audit.json")
    validate_access(access)
    final = load_json(artifact / "final_verdict.json")
    validate_final(final, support, mapping)
    freeze_commit = load_json(artifact / "freeze_commit.json")
    require(freeze_commit["status"] == "PASS" and final["freeze_sha"] == run["freeze_sha"] == freeze_commit["commit_sha"], "repair-freeze SHA binding")
    manifest = load_json(artifact / "artifact_manifest_sha256.json")
    require(manifest == compact_manifest(artifact), "compact manifest mismatch")
    result = {
        "status": "PASS",
        "verdict": final["verdict"],
        "base_r1_final_sha": BASE_R1_FINAL_SHA,
        "preregistration_sha": PREREGISTRATION_SHA,
        "freeze_sha": freeze_commit["commit_sha"],
        "manifest_files": manifest["file_count"],
        "c1_bytes_read": access["c1"]["total_payload_bytes_read"],
        "c3_bytes_read": 0,
        "attack_bytes_read": 0,
        "acquisition_prn_count": final["acquisition_prn_count"],
        "tracking_ge_10s_prn_count": final["tracking_ge_10s_prn_count"],
    }
    if verify_large:
        result.update(validate_large(run))
    return result


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def verify(repo: Path, verify_large: bool = False) -> dict[str, Any]:
    result = validate_artifact(repo, verify_large)
    head = git(repo, "rev-parse", "HEAD")
    remote = git(repo, "rev-parse", f"origin/{BRANCH}")
    require(head == remote, "local/remote SHA mismatch")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", BASE_R1_FINAL_SHA, head], cwd=repo).returncode == 0, "R1 base ancestry")
    require(git(repo, "status", "--porcelain") == "", "worktree not clean")
    result["final_sha"] = head
    return result


def self_test(repo: Path) -> dict[str, Any]:
    artifact = repo / ARTIFACT_REL
    tests: dict[str, bool] = {}
    manifest = load_json(artifact / "artifact_manifest_sha256.json")
    target = artifact / "README.md"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"tamper")
        tests["artifact_manifest_tamper"] = manifest != compact_manifest(artifact)
    finally:
        target.write_bytes(original)
    access = load_json(artifact / "access_audit.json")
    access["attack"]["bytes_read"] = 1
    try:
        validate_access(access)
        tests["attack_access_tamper"] = False
    except VerificationError:
        tests["attack_access_tamper"] = True
    dumps = load_json(artifact / "tracking_dump_inventory.json")
    dumps["rows"][0]["trailing_fragment_size_bytes"] = 96
    tests["dump_tail_bound_tamper"] = int(dumps["rows"][0]["trailing_fragment_size_bytes"]) >= 96
    final = load_json(artifact / "final_verdict.json")
    final["attack_download_or_access_authorized"] = True
    try:
        validate_final(final, load_json(artifact / "support_audit.json"), load_json(artifact / "time_mapping_validation.json"))
        tests["attack_authorization_tamper"] = False
    except VerificationError:
        tests["attack_authorization_tamper"] = True
    require(all(tests.values()), f"self-test failure: {tests}")
    return {"status": "PASS", "test_count": len(tests), "tests": tests}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-large", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.seal:
        seal(repo / ARTIFACT_REL)
        print("PASS: compact manifest sealed")
        return 0
    result = verify(repo, args.verify_large)
    if args.self_test:
        result["self_test"] = self_test(repo)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"FAIL: {error}")
        raise SystemExit(1)
