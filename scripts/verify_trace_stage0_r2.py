#!/usr/bin/env python3
"""Verify TRACE-R2 Git scope, fail-closed gates, and artifact integrity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2_native_1ms_dump"
BASE = "ba3355ac6b61f088fa37d579d6af02947ccf34c1"
EXPECTED_MAIN = "461eb4dc7bb794e719295daf028f6811658ba37f"
BRANCH = "research/trace-stage0-r2-native-1ms-dump"
ALLOWED_PREFIXES = (
    "artifacts/trace_stage0_r2_native_1ms_dump/",
    "docs/TRACE_STAGE0_R2_NATIVE_1MS.md",
    "scripts/evaluate_trace_r2_phase_a.py",
    "scripts/evaluate_trace_r2_phase_b.py",
    "scripts/finalize_trace_r2_fail_closed.py",
    "scripts/run_trace_stage0_r2.py",
    "scripts/validate_trace_native_1ms_dump.py",
    "scripts/verify_trace_stage0_r2.py",
    "src/gnss_doppler_lab/trace_native_1ms.py",
    "tests/test_trace_native_1ms.py",
)
REQUIRED = (
    "README.md", "config.json", "preregistration.json", "source_commit.json",
    "receiver_source_provenance.json", "receiver_patch.diff", "receiver_build_manifest.json",
    "native_dump_schema.json", "action_mapping_validation.json", "smoke_replay_results.json",
    "raw_source_binding.json", "replay_inventory.json", "clean_split_audit.json", "thresholds.json",
    "scenario_metrics.csv", "ablation_metrics.csv", "per_epoch_scores.csv.gz",
    "per_prn_action_response.csv.gz", "external_static_fpr.csv", "action_shuffle_metrics.json",
    "physical_controls.json", "bootstrap_intervals.csv", "final_verdict.json",
    "artifact_manifest_sha256.json", "runner_runs.json", "test_results.json",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_git_scope() -> None:
    assert git("branch", "--show-current") == BRANCH
    assert git("merge-base", "HEAD", BASE) == BASE
    assert git("rev-parse", "origin/research/trace-stage0-r1-native-cadence") == BASE
    assert git("rev-parse", "main") == EXPECTED_MAIN
    assert git("rev-parse", "origin/main") == EXPECTED_MAIN
    changed = git("diff", "--name-only", f"{BASE}...HEAD").splitlines()
    for path in changed:
        assert any(path == prefix or path.startswith(prefix) for prefix in ALLOWED_PREFIXES), path
    tracked = git("ls-files").splitlines()
    for name in tracked:
        path = ROOT / name
        if path.is_file():
            assert path.stat().st_size < 25 * 1024 * 1024, f"large tracked file: {name}"
        assert not name.endswith(".bin") or not name.startswith("artifacts/trace_stage0_r2_native_1ms_dump/")


def verify_final() -> None:
    missing = [name for name in REQUIRED if not (ARTIFACT / name).is_file()]
    assert not missing, missing
    verdict = json.loads((ARTIFACT / "final_verdict.json").read_text())
    allowed = {
        "GO_TRACE_PHYSICAL_HYPOTHESIS",
        "NO_GO_TRACE_PHYSICAL_HYPOTHESIS",
        "INCONCLUSIVE_INPUT_OR_RECEIVER",
    }
    assert verdict["verdict"] in allowed
    smoke = json.loads((ARTIFACT / "smoke_replay_results.json").read_text())
    if verdict["verdict"] == "INCONCLUSIVE_INPUT_OR_RECEIVER":
        labels = set(verdict["failure_labels"])
        assert labels
        assert labels <= {
            "NATIVE_1MS_DUMP_IMPLEMENTATION_INVALID",
            "ACTION_MAPPING_UNRESOLVED",
            "INSUFFICIENT_MULTI_PRN_SUPPORT",
            "RAW_SOURCE_BINDING_FAILED",
        }
        assert verdict["phase_b_run"] is False
        assert verdict["attack_scores_computed"] is False
        assert verdict["performance_claimed"] is False
        for filename in ("scenario_metrics.csv", "ablation_metrics.csv"):
            with (ARTIFACT / filename).open(newline="") as stream:
                assert all(row["status"] == "UNAVAILABLE" for row in csv.DictReader(stream))
    else:
        assert smoke["status"] == "PASS"
        assert smoke["phase_b_authorized"] is True
    manifest = json.loads((ARTIFACT / "artifact_manifest_sha256.json").read_text())
    for relative, expected in manifest["files"].items():
        assert sha256(ARTIFACT / relative) == expected, relative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "final"), required=True)
    args = parser.parse_args()
    verify_git_scope()
    assert (ARTIFACT / "preregistration.json").is_file()
    assert (ARTIFACT / "receiver_patch.diff").is_file()
    if args.phase == "final":
        verify_final()
    print(f"TRACE-R2 {args.phase} verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
