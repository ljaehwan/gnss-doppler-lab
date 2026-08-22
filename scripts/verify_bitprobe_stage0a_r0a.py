#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab.bitprobe_stage0a_r0a import (
    ALLOWED_VERDICTS,
    ARTIFACT_REL,
    BASE_SHA,
    BRANCH,
    EXECUTABLE_FILES,
    FROZEN_FILES,
    ORIGINAL_ARTIFACT_REL,
    ORIGINAL_ARTIFACT_TREE_SHA,
    ORIGINAL_MANIFEST_FILE_SHA256,
    PREREGISTRATION_COMMIT_SHA,
    RepairError,
    TENSOR_PATH,
    TENSOR_SHA256,
    TENSOR_SIZE,
    analysis_bytes,
    binding,
    canonical_json,
    compact_manifest,
    directory_binding,
    exact_prn_permutation,
    git,
    load_inventory,
    load_repair_preregistration,
    load_tensor_once,
    repaired_analysis,
    sha256_bytes,
    sha256_file,
)


REQUIRED_FINAL = (
    "README.md", "repair_preregistration.json", "repair_preregistration_commit.json",
    "execution_freeze.json", "freeze_commit.json", "original_artifact_binding.json",
    "frozen_tensor_binding.json", "input_access_audit.json",
    "corrected_split_half_metrics.csv", "corrected_between_prn_metrics.csv",
    "corrected_bootstrap_metrics.json", "old_new_metric_comparison.csv",
    "flip_no_flip_reproduction.json", "nuisance_reproduction.json",
    "synthetic_reproduction.json", "deterministic_reanalysis.json",
    "inference_contract_tests.json", "post_result_change_audit.json",
    "final_verdict.json", "artifact_manifest_sha256.json",
    "verifier_output.txt", "test_output.txt",
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _check_freeze(repo: Path, artifact: Path) -> dict[str, object]:
    prereg = load_repair_preregistration(repo)
    commit = json.loads((artifact / "repair_preregistration_commit.json").read_text())
    require(commit["commit_sha"] == PREREGISTRATION_COMMIT_SHA, "repair preregistration commit mismatch")
    freeze = json.loads((artifact / "execution_freeze.json").read_text())
    require(freeze["base_sha"] == BASE_SHA, "freeze base mismatch")
    require(freeze["repair_preregistration_commit_sha"] == PREREGISTRATION_COMMIT_SHA, "freeze prereg mismatch")
    for relative in EXECUTABLE_FILES:
        require(binding(repo / relative, relative) == freeze["executable_bindings"][relative], f"executable binding mismatch: {relative}")
    for relative in FROZEN_FILES:
        require(binding(repo / relative, relative) == freeze["frozen_bindings"][relative], f"frozen binding mismatch: {relative}")
    expected_config = sha256_bytes(canonical_json({
        "chronological_half_bootstrap_contract": prereg["chronological_half_bootstrap_contract"],
        "exact_prn_permutation_contract": prereg["exact_prn_permutation_contract"],
        "frozen_reproduction_contract": prereg["frozen_reproduction_contract"],
        "post_result_contract": prereg["post_result_contract"],
    }).encode())
    require(expected_config == freeze["repair_configuration_sha256"], "repair configuration mismatch")
    require(freeze["tensor_operations_before_freeze"] == 0, "pre-freeze tensor access")
    original_binding = json.loads((artifact / "original_artifact_binding.json").read_text())
    require(original_binding["git_tree_sha"] == ORIGINAL_ARTIFACT_TREE_SHA, "original tree SHA mismatch")
    require(original_binding["manifest_file_sha256"] == ORIGINAL_MANIFEST_FILE_SHA256, "original manifest binding mismatch")
    require(git(repo, "rev-parse", f"HEAD:{ORIGINAL_ARTIFACT_REL}") == ORIGINAL_ARTIFACT_TREE_SHA, "original Git tree changed")
    require(directory_binding(repo / ORIGINAL_ARTIFACT_REL) == original_binding["directory_binding"], "original artifact bytes changed")
    audit = json.loads((artifact / "input_access_audit.json").read_text())
    for group in ("clean_raw_iq", "trace", "attacks"):
        for field in ("stats", "hashes", "opens", "mmaps", "bytes_read"):
            require(int(audit[group][field]) == 0, f"forbidden access nonzero: {group}.{field}")
    return freeze


def verify(repo: Path) -> dict[str, object]:
    artifact = repo / ARTIFACT_REL
    freeze = _check_freeze(repo, artifact)
    final_path = artifact / "final_verdict.json"
    if not final_path.is_file():
        return {
            "status": "PASS", "mode": "freeze", "base_sha": BASE_SHA,
            "repair_preregistration_commit_sha": PREREGISTRATION_COMMIT_SHA,
            "executable_count": len(EXECUTABLE_FILES), "tensor_access_before_freeze": 0,
        }
    missing = [name for name in REQUIRED_FINAL if not (artifact / name).is_file()]
    require(not missing, f"missing final artifacts: {missing}")
    manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
    require(manifest == compact_manifest(artifact), "artifact manifest mismatch")
    tensor_binding = json.loads((artifact / "frozen_tensor_binding.json").read_text())
    require(tensor_binding["path"] == str(TENSOR_PATH), "tensor path mismatch")
    require(tensor_binding["size_bytes"] == TENSOR_SIZE, "tensor recorded size mismatch")
    require(tensor_binding["sha256"] == TENSOR_SHA256, "tensor recorded hash mismatch")
    vectors, observed_tensor = load_tensor_once()
    require(observed_tensor["size_bytes"] == TENSOR_SIZE and observed_tensor["sha256"] == TENSOR_SHA256, "tensor physical binding mismatch")
    rows = load_inventory(repo)
    reproduced = repaired_analysis(repo, rows, vectors)
    for name, payload in analysis_bytes(reproduced).items():
        require((artifact / name).read_bytes() == payload, f"repaired metric recalculation mismatch: {name}")
    deterministic = json.loads((artifact / "deterministic_reanalysis.json").read_text())
    require(deterministic["analysis_run_count"] == 2 and deterministic["byte_identical"] is True, "two-run determinism mismatch")
    bootstrap = json.loads((artifact / "corrected_bootstrap_metrics.json").read_text())
    for dataset, value in bootstrap.items():
        require(value["replicate_count"] == 500 and value["effect_replicate_count"] == 500, f"bootstrap count mismatch: {dataset}")
        require(value["permutation"]["permutation_count"] == 120, f"permutation count mismatch: {dataset}")
        require(value["permutation"]["same_count"] == 5, f"same shape mismatch: {dataset}")
        require(value["permutation"]["different_count"] == 20, f"different shape mismatch: {dataset}")
        require(value["permutation"]["all_shapes_valid"] is True, f"permutation shape audit: {dataset}")
    access = json.loads((artifact / "input_access_audit.json").read_text())
    require(access["frozen_tensor"]["bytes_read"] == TENSOR_SIZE, "analysis tensor byte count mismatch")
    require(access["frozen_tensor"]["stats"] == 1 and access["frozen_tensor"]["hashes"] == 1 and access["frozen_tensor"]["opens"] == 1 and access["frozen_tensor"]["mmaps"] == 0, "analysis tensor operation count mismatch")
    flip = json.loads((artifact / "flip_no_flip_reproduction.json").read_text())
    require(flip["values_match_original"] is True, "flip values changed")
    require(flip["new_mismatch_found"] is True and flip["repair_attempted"] is False, "flip fail-closed audit mismatch")
    final = json.loads(final_path.read_text())
    require(final["base_sha"] == BASE_SHA, "final base mismatch")
    require(final["repair_preregistration_commit_sha"] == PREREGISTRATION_COMMIT_SHA, "final prereg mismatch")
    require(final["freeze_sha"] == json.loads((artifact / "freeze_commit.json").read_text())["freeze_sha"], "final freeze mismatch")
    require(final["substantive_verdict"] in ALLOWED_VERDICTS, "unregistered substantive verdict")
    require(final["substantive_verdict"] == "INCONCLUSIVE_BITPROBE_STAGE0A_R0A_INFERENCE_REPAIR", "new flip mismatch must fail closed")
    require(final["repair_status"] == "FAIL" and final["stage0b_authorized"] is False, "repair/authorization mismatch")
    require(final["post_result_method_feature_gate_threshold_changes"] == 0, "post-result method change")
    require(final["raw_iq_operations"] == 0 and final["trace_operations"] == 0 and final["attack_operations"] == 0, "forbidden access in verdict")
    require(final["original_artifact_byte_identical"] is True, "original artifact not preserved")
    head = git(repo, "rev-parse", "HEAD")
    remote = git(repo, "rev-parse", f"origin/{BRANCH}")
    require(head == remote, "local/remote final SHA mismatch")
    return {
        "status": "PASS", "mode": "final", "final_sha": head,
        "base_sha": BASE_SHA, "repair_preregistration_commit_sha": PREREGISTRATION_COMMIT_SHA,
        "freeze_sha": final["freeze_sha"], "substantive_verdict": final["substantive_verdict"],
        "repair_status": final["repair_status"], "stage0b_authorized": final["stage0b_authorized"],
        "original_artifact_byte_identical": True, "raw_trace_attack_operations": 0,
        "manifest_files": manifest["file_count"], "tensor_verification_bytes": TENSOR_SIZE,
    }


def self_test(repo: Path) -> dict[str, object]:
    artifact = repo / ARTIFACT_REL
    tests: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="bitprobe-r0a-verifier-") as temporary:
        root = Path(temporary)
        (root / "a.txt").write_text("a\n")
        frozen = compact_manifest(root)
        (root / "a.txt").write_text("b\n")
        tests["manifest_tamper"] = frozen != compact_manifest(root)
    freeze = json.loads((artifact / "execution_freeze.json").read_text())
    mutated_code = copy.deepcopy(freeze["executable_bindings"])
    first = EXECUTABLE_FILES[0]
    mutated_code[first]["sha256"] = "0" * 64
    tests["code_tamper"] = mutated_code[first] != binding(repo / first, first)
    access = json.loads((artifact / "input_access_audit.json").read_text())
    mutated_access = copy.deepcopy(access); mutated_access["attacks"]["opens"] = 1
    tests["access_tamper"] = mutated_access["attacks"] != access["attacks"]
    original_binding = json.loads((artifact / "original_artifact_binding.json").read_text())
    mutated_original = copy.deepcopy(original_binding); mutated_original["git_tree_sha"] = "0" * 40
    tests["original_artifact_tamper"] = mutated_original["git_tree_sha"] != ORIGINAL_ARTIFACT_TREE_SHA
    tensor = json.loads((artifact / "frozen_tensor_binding.json").read_text())
    mutated_tensor = copy.deepcopy(tensor); mutated_tensor["sha256"] = "0" * 64
    tests["tensor_binding_tamper"] = mutated_tensor["sha256"] != TENSOR_SHA256
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    perm = exact_prn_permutation(matrix)
    tests["permutation_shape"] = perm["same_count"] == 2 and perm["different_count"] == 2 and perm["all_shapes_valid"]
    require(all(tests.values()), f"self-test failure: {tests}")
    return {"status": "PASS", "test_count": len(tests), "tests": tests}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify BITPROBE Stage-0A R0a inference-contract repair")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help="repository root (default: derived from this script)")
    parser.add_argument("--self-test", action="store_true", help="also run deterministic verifier tamper fixtures")
    args = parser.parse_args()
    result = verify(args.repo.resolve())
    if args.self_test:
        result["self_test"] = self_test(args.repo.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerificationError, RepairError) as error:
        print(f"FAIL: {error}")
        raise SystemExit(1)
