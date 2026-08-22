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

from gnss_doppler_lab.bitprobe_stage0a_r0b import (
    ALLOWED_VERDICTS, ARTIFACT_REL, BASE_SHA, BRANCH, CONTRACT_COMMIT_SHA,
    EXECUTABLE_FILES, GRID_FIELDS, SOURCE_FILES, SensitivityError, analysis_bytes,
    analyze, binding, canonical_json, compact_manifest, git, load_contract,
    sha256_bytes, verify_sources,
)

REQUIRED_FINAL = (
    "README.md", "relaxed_sensitivity_contract.json", "contract_commit.json",
    "execution_freeze.json", "freeze_commit.json", "source_metric_binding.json",
    "tier_results.json", "threshold_grid.csv.gz", "minimum_relaxation_required.json",
    "gate_margin.csv", "leave_one_prn_out.json", "input_access_audit.json",
    "formal_verdict_preservation.json", "deterministic_reproduction.json",
    "final_verdict.json", "artifact_manifest_sha256.json", "verifier_output.txt",
    "test_output.txt", "threshold_sensitivity_heatmap.png",
    "dataset_gate_margin.png", "tier_comparison.png",
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _freeze(repo: Path, artifact: Path) -> dict[str, object]:
    contract = load_contract(repo)
    commit = json.loads((artifact / "contract_commit.json").read_text())
    require(commit["commit_sha"] == CONTRACT_COMMIT_SHA, "contract commit mismatch")
    freeze = json.loads((artifact / "execution_freeze.json").read_text())
    require(freeze["base_sha"] == BASE_SHA and freeze["contract_commit_sha"] == CONTRACT_COMMIT_SHA, "freeze ancestry mismatch")
    for relative in EXECUTABLE_FILES:
        require(binding(repo / relative, relative) == freeze["executable_bindings"][relative], f"executable tamper: {relative}")
    sources = verify_sources(repo)
    require(sources == freeze["source_bindings"], "source metric binding mismatch")
    expected_config = sha256_bytes(canonical_json({
        "tiers": contract["tiers"], "threshold_grid": contract["threshold_grid"],
        "minimum_relaxation_contract": contract["minimum_relaxation_contract"],
        "verdict_contract": contract["verdict_contract"],
    }).encode())
    require(expected_config == freeze["configuration_sha256"], "configuration binding mismatch")
    audit = json.loads((artifact / "input_access_audit.json").read_text())
    for group in ("raw_iq", "trace", "frozen_tensor", "attacks"):
        require(all(int(audit[group][field]) == 0 for field in ("stats", "hashes", "opens", "mmaps", "bytes_read")), f"forbidden access: {group}")
    return freeze


def verify(repo: Path) -> dict[str, object]:
    artifact = repo / ARTIFACT_REL
    freeze = _freeze(repo, artifact)
    if not (artifact / "final_verdict.json").is_file():
        return {"status": "PASS", "mode": "freeze", "base_sha": BASE_SHA, "contract_commit_sha": CONTRACT_COMMIT_SHA, "forbidden_access_operations": 0}
    missing = [name for name in REQUIRED_FINAL if not (artifact / name).is_file()]
    require(not missing, f"missing artifacts: {missing}")
    manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
    require(manifest == compact_manifest(artifact), "manifest mismatch")
    reproduced = analyze(repo)
    for name, payload in analysis_bytes(reproduced).items():
        require((artifact / name).read_bytes() == payload, f"deterministic metric mismatch: {name}")
    deterministic = json.loads((artifact / "deterministic_reproduction.json").read_text())
    require(deterministic["run_count"] == 2 and deterministic["byte_identical"] is True, "deterministic rerun failure")
    import gzip, csv, io
    rows = list(csv.DictReader(io.StringIO(gzip.decompress((artifact / "threshold_grid.csv.gz").read_bytes()).decode())))
    require(len(rows) == 4608, "threshold grid row count")
    require(set(rows[0]) == set(GRID_FIELDS), "threshold grid schema")
    final = json.loads((artifact / "final_verdict.json").read_text())
    require(final["exploratory_verdict"] in ALLOWED_VERDICTS, "unregistered verdict")
    require(final["exploratory_verdict"] == reproduced["verdict"], "verdict reproduction mismatch")
    require(final["post_hoc"] is True and final["confirmatory"] is False, "post-hoc flags")
    require(final["formal_stage0a_verdict_changed"] is False and final["stage0b_authorized"] is False, "formal authorization flags")
    require(final["spoofing_detector_validated"] is False and final["flip_specificity_advisory_only"] is True, "claim/advisory flags")
    require(final["source_metric_changes"] == 0, "source metric change")
    require(final["raw_iq_operations"] == 0 and final["trace_operations"] == 0 and final["frozen_tensor_operations"] == 0 and final["attack_operations"] == 0, "forbidden operation count")
    head, remote = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", f"origin/{BRANCH}")
    require(head == remote, "local/remote mismatch")
    return {
        "status": "PASS", "mode": "final", "final_sha": head,
        "base_sha": BASE_SHA, "contract_commit_sha": CONTRACT_COMMIT_SHA,
        "freeze_sha": final["freeze_sha"], "exploratory_verdict": final["exploratory_verdict"],
        "formal_stage0a_verdict_changed": False, "stage0b_authorized": False,
        "forbidden_access_operations": 0, "threshold_grid_rows": len(rows),
        "manifest_files": manifest["file_count"],
    }


def self_test(repo: Path) -> dict[str, object]:
    artifact = repo / ARTIFACT_REL
    tests = {}
    with tempfile.TemporaryDirectory(prefix="bitprobe-r0b-") as temporary:
        root = Path(temporary); (root / "x").write_text("x")
        frozen = compact_manifest(root); (root / "x").write_text("y")
        tests["manifest_tamper"] = frozen != compact_manifest(root)
    freeze = json.loads((artifact / "execution_freeze.json").read_text())
    mutated = copy.deepcopy(freeze["source_bindings"]); key = sorted(mutated)[0]; mutated[key]["sha256"] = "0" * 64
    tests["source_metric_tamper"] = mutated != verify_sources(repo)
    access = json.loads((artifact / "input_access_audit.json").read_text()); changed = copy.deepcopy(access); changed["frozen_tensor"]["opens"] = 1
    tests["tensor_access_tamper"] = changed["frozen_tensor"] != access["frozen_tensor"]
    result = analyze(repo)
    tests["grid_cardinality"] = len(result["grid_rows"]) == 4608
    tests["formal_preservation"] = result["verdict"] in ALLOWED_VERDICTS
    require(all(tests.values()), f"self-test failure: {tests}")
    return {"status": "PASS", "test_count": len(tests), "tests": tests}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify BITPROBE Stage-0A R0b relaxed-gate sensitivity")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = verify(args.repo.resolve())
    if args.self_test:
        result["self_test"] = self_test(args.repo.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerificationError, SensitivityError) as error:
        print(f"FAIL: {error}")
        raise SystemExit(1)
