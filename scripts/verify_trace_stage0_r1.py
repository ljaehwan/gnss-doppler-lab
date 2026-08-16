#!/usr/bin/env python3
"""Independent structural and checksum verifier for TRACE Stage-0 R1."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r1_native_cadence"
BASE = "ab8770b021020062d86fcd240ce7ecee76466072"
BRANCH = "research/trace-stage0-r1-native-cadence"
PREFREEZE = (
    "README.md", "config.json", "preregistration.json", "cadence_contract.json",
    "cadence_support.csv", "cadence_transition_by_prn.csv", "action_mapping_validation.json",
    "input_inventory.json", "source_binding.json",
)
FINAL = PREFREEZE + (
    "source_commit.json", "clean_split_audit.json", "normal_model_summary.json", "thresholds.json",
    "scenario_metrics.csv", "ablation_metrics.csv", "per_epoch_scores.csv.gz", "external_static_fpr.csv",
    "action_shuffle_metrics.json", "synthetic_physics_metrics.json", "physical_controls.json",
    "bootstrap_intervals.csv", "b0_lineage.json", "final_verdict.json", "test_results.json",
    "runner_runs.json", "artifact_manifest_sha256.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prefreeze", "final"), default="final")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != BRANCH:
        failures.append(f"wrong branch: {branch}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT).returncode:
        failures.append("required base is not an ancestor")
    if subprocess.check_output(["git", "diff", "--name-only", BASE, "--", "artifacts/trace_stage0_static"], cwd=ROOT, text=True).strip():
        failures.append("protected prior TRACE artifact changed")
    required = PREFREEZE if args.phase == "prefreeze" else FINAL
    for name in required:
        if not (ARTIFACT / name).is_file():
            failures.append(f"missing {name}")
    if (ARTIFACT / "README.md").is_file() and "Configuration frozen before this TRACE-R1 evaluation." not in (ARTIFACT / "README.md").read_text():
        failures.append("freeze wording missing")
    if (ARTIFACT / "action_mapping_validation.json").is_file():
        mapping = json.loads((ARTIFACT / "action_mapping_validation.json").read_text())
        if mapping.get("retained_row_t_to_t_plus_1") is not False:
            failures.append("retained mapping result unexpectedly changed")
        if mapping.get("nco_update_cadence") != "1_ms_with_20_ms_dump_decimation":
            failures.append("wrong audited receiver cadence")
    if args.phase == "final" and not failures:
        verdict = json.loads((ARTIFACT / "final_verdict.json").read_text())
        if verdict.get("verdict") != "NEEDS_TRACE_SPECIFIC_RECEIVER_DUMP":
            failures.append("invalid fail-closed verdict")
        if verdict.get("attack_scores_computed") is not False or verdict.get("performance_claimed") is not False:
            failures.append("fail-closed result claims performance")
        with gzip.open(ARTIFACT / "per_epoch_scores.csv.gz", "rt") as stream:
            lines = stream.readlines()
        if len(lines) != 1:
            failures.append("attack score rows exist despite failed mapping gate")
    if args.write_manifest and not failures:
        files = sorted(path for path in ARTIFACT.rglob("*") if path.is_file() and path.name != "artifact_manifest_sha256.json")
        payload = {str(path.relative_to(ARTIFACT)): sha256(path) for path in files}
        (ARTIFACT / "artifact_manifest_sha256.json").write_text(json.dumps({"schema": "gnss-doppler-lab.trace-r1-artifact-manifest.v1", "files": payload}, indent=2, sort_keys=True) + "\n")
    manifest_path = ARTIFACT / "artifact_manifest_sha256.json"
    if args.phase == "final" and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text()).get("files", {})
        for relative, expected in manifest.items():
            path = ARTIFACT / relative
            if not path.is_file() or sha256(path) != expected:
                failures.append(f"checksum mismatch {relative}")
    result = {"phase": args.phase, "passed": not failures, "failures": failures, "branch": branch}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
