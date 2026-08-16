#!/usr/bin/env python3
"""Independent structural/checksum verifier for TRACE Stage-0 artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_static"
ALLOWED = {"GO_FOR_TRACE_STAGE1", "NO_GO_ACTION_EQUIVARIANCE", "INCONCLUSIVE_INPUT_OR_ALIGNMENT"}
PREFREEZE = (
    "README.md", "config.json", "preregistration.json", "input_inventory.json",
    "source_binding.json", "timeline_inventory.json", "alignment_validation.json",
    "clean_split_audit.json", "normal_model_summary.json", "thresholds.json",
)
FINAL = PREFREEZE + (
    "source_commit.json", "test_results.json", "confirmation_freeze.json",
    "synthetic_physics_metrics.json", "scenario_metrics.csv", "ablation_metrics.csv",
    "per_epoch_scores.csv.gz", "action_response_diagnostics.csv.gz",
    "external_static_fpr.csv", "action_shuffle_metrics.json", "physical_controls.json",
    "bootstrap_intervals.csv", "final_verdict.json", "runner_runs.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def manifest_files() -> list[Path]:
    return sorted(path for path in ARTIFACT.rglob("*") if path.is_file() and path.name != "artifact_manifest_sha256.json" and "work" not in path.parts)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("prefreeze", "final"), default="final"); parser.add_argument("--write-manifest", action="store_true"); args = parser.parse_args()
    required = PREFREEZE if args.phase == "prefreeze" else FINAL
    failures = []
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "research/trace-stage0-static": failures.append(f"wrong branch: {branch}")
    for name in required:
        if not (ARTIFACT / name).is_file(): failures.append(f"missing {name}")
    if "Configuration frozen before this TRACE evaluation." not in (ARTIFACT / "README.md").read_text(): failures.append("required freeze wording missing")
    alignment = json.loads((ARTIFACT / "alignment_validation.json").read_text())
    if alignment.get("status") == "UNRESOLVED" and alignment.get("reason") != "TRACKER_ACTION_ALIGNMENT_UNRESOLVED": failures.append("wrong fail-closed alignment reason")
    if args.phase == "final" and not failures:
        verdict = json.loads((ARTIFACT / "final_verdict.json").read_text())
        if verdict.get("verdict") not in ALLOWED: failures.append("invalid final verdict")
        if alignment.get("status") == "UNRESOLVED" and verdict.get("verdict") != "INCONCLUSIVE_INPUT_OR_ALIGNMENT": failures.append("unresolved alignment did not force inconclusive")
        if verdict.get("verdict") == "INCONCLUSIVE_INPUT_OR_ALIGNMENT" and verdict.get("performance_claimed") is not False: failures.append("inconclusive result claims performance")
        for name in ("per_epoch_scores.csv.gz", "action_response_diagnostics.csv.gz"):
            try:
                with gzip.open(ARTIFACT / name, "rt") as stream: next(stream)
            except Exception as exc: failures.append(f"invalid gzip {name}: {exc}")
    if args.write_manifest:
        payload = {str(path.relative_to(ARTIFACT)): sha256(path) for path in manifest_files()}
        (ARTIFACT / "artifact_manifest_sha256.json").write_text(json.dumps({"schema": "gnss-doppler-lab.trace-artifact-manifest.v1", "files": payload}, indent=2, sort_keys=True) + "\n")
    manifest_path = ARTIFACT / "artifact_manifest_sha256.json"
    if args.phase == "final" and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text()).get("files", {})
        for relative, expected in manifest.items():
            path = ARTIFACT / relative
            if not path.is_file() or sha256(path) != expected: failures.append(f"checksum mismatch {relative}")
    result = {"phase": args.phase, "passed": not failures, "failures": failures, "branch": branch}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__": raise SystemExit(main())
