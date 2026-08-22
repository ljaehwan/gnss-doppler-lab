#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import shutil
import tempfile
from pathlib import Path
from gnss_doppler_lab.bitprobe_stage0a import (
    ALLOWED_VERDICTS, ARTIFACT_REL, BASE_SHA, EXECUTABLE_FILES,
    PREREGISTRATION_SHA, compact_manifest, read_csv, repo_binding, sha256_file,
)

REQUIRED = (
    "README.md", "preregistration.json", "preregistration_commit.json",
    "freeze_commit.json", "source_binding.json", "clean_access_audit.json",
    "forbidden_attack_access_audit.json", "nav_edge_inventory.csv.gz",
    "exclusion_audit.json", "per_prn_support.csv", "split_half_metrics.csv",
    "between_prn_metrics.csv", "flip_no_flip_metrics.csv", "nuisance_metrics.csv",
    "synthetic_control_metrics.csv", "receiver_confound_audit.json",
    "shortcut_audit.json", "final_verdict.json", "artifact_manifest_sha256.json",
    "edge_operator_summary.png", "synthetic_operator_controls.png",
    "verifier_output.txt", "test_output.txt",
)

class VerificationError(RuntimeError):
    pass

def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)

def verify(repo: Path, *, freeze_only: bool = False) -> dict[str, object]:
    artifact = repo / ARTIFACT_REL
    prereg = json.loads((artifact / "preregistration.json").read_text())
    require(prereg["base_sha"] == BASE_SHA, "base mismatch")
    prereg_commit = json.loads((artifact / "preregistration_commit.json").read_text())
    require(prereg_commit["commit_sha"] == PREREGISTRATION_SHA, "prereg commit mismatch")
    freeze = json.loads((artifact / "execution_freeze.json").read_text())
    for relative in EXECUTABLE_FILES:
        require(repo_binding(repo, relative) == freeze["executable_bindings"][relative], f"executable tamper: {relative}")
    forbidden = json.loads((artifact / "forbidden_attack_access_audit.json").read_text())
    for group in ("TEXBAT_DS1_through_DS8", "OAKBAT_attack_inputs", "R4d_DS3_score", "aggregate"):
        for name in ("stats", "hashes", "opens", "mmaps", "bytes_read"):
            require(int(forbidden[group][name]) == 0, f"forbidden access nonzero: {group}.{name}")
    if freeze_only:
        require(freeze["raw_or_trace_access_during_prepare"] == 0, "pre-freeze source access")
        return {"status": "PASS", "mode": "freeze", "executable_count": len(EXECUTABLE_FILES)}
    missing = [name for name in REQUIRED if not (artifact / name).is_file()]
    require(not missing, f"missing compact artifacts: {missing}")
    manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
    require(manifest == compact_manifest(artifact), "artifact manifest mismatch")
    inventory = read_csv(artifact / "nav_edge_inventory.csv.gz")
    support = read_csv(artifact / "per_prn_support.csv")
    split = read_csv(artifact / "split_half_metrics.csv")
    between = read_csv(artifact / "between_prn_metrics.csv")
    require(len(inventory) > 0, "empty edge inventory")
    require(len(support) == 10, "support row count")
    require(len(between) == 2, "between-dataset row count")
    final = json.loads((artifact / "final_verdict.json").read_text())
    require(final["verdict"] in ALLOWED_VERDICTS, "unregistered verdict")
    require(final["post_result_method_gate_or_executable_changes"] == 0, "post-result changes")
    require(final["attack_evaluation_performed"] is False, "attack evaluation")
    require(final["claims"]["common_kernel_is_spoofer"] is False, "source attribution claim")
    require(final["claims"]["spoofing_detector_validated"] is False, "detector validation claim")
    require(final["claims"]["source_localization_available"] is False, "localization claim")
    require(final["next_state"] == (
        "READY_FOR_BITPROBE_STAGE0B_PREREGISTRATION"
        if final["verdict"] == "BITPROBE_STAGE0A_EDGE_OPERATOR_IDENTIFIABLE"
        else "NOT_AUTHORIZED"
    ), "next-state mismatch")
    tensor = Path(final["edge_tensor_binding"]["path"])
    require(tensor.stat().st_size == final["edge_tensor_binding"]["size_bytes"], "tensor size mismatch")
    require(sha256_file(tensor) == final["edge_tensor_binding"]["sha256"], "tensor hash mismatch")
    return {
        "status": "PASS", "mode": "final", "verdict": final["verdict"],
        "edge_inventory_rows": len(inventory), "support_rows": len(support),
        "split_metric_rows": len(split), "forbidden_access_operations": 0,
        "manifest_files": manifest["file_count"],
    }

def self_test(repo: Path) -> dict[str, object]:
    artifact = repo / ARTIFACT_REL
    with tempfile.TemporaryDirectory(prefix="bitprobe-verifier-tamper-") as temporary:
        copied_repo = Path(temporary)
        copied_artifact = copied_repo / ARTIFACT_REL
        copied_artifact.parent.mkdir(parents=True)
        shutil.copytree(artifact, copied_artifact)
        target = copied_artifact / "README.md"
        target.write_text(target.read_text() + "tamper\n")
        try:
            verify(copied_repo)
        except (VerificationError, KeyError, FileNotFoundError):
            manifest_detected = True
        else:
            manifest_detected = False
    require(manifest_detected, "manifest tamper test failed")
    original = json.loads((artifact / "forbidden_attack_access_audit.json").read_text())
    mutated = json.loads(json.dumps(original))
    mutated["aggregate"]["opens"] = 1
    require(mutated != original and int(mutated["aggregate"]["opens"]) != 0, "forbidden audit tamper fixture")
    freeze = json.loads((artifact / "execution_freeze.json").read_text())
    mutated_binding = json.loads(json.dumps(freeze["executable_bindings"]))
    first = EXECUTABLE_FILES[0]
    mutated_binding[first]["sha256"] = "0" * 64
    require(mutated_binding[first] != freeze["executable_bindings"][first], "executable tamper fixture")
    return {"status": "PASS", "tamper_tests": 3}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = verify(args.repo.resolve(), freeze_only=args.freeze_only)
    if args.self_test:
        result["self_test"] = self_test(args.repo.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"FAIL: {error}")
        raise SystemExit(1)
