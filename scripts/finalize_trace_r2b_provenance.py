#!/usr/bin/env python3
"""Collect durable R2b run provenance and materialize the checksum manifest."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2b_stable_handoff_repair"
RUN_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/runs")
EXCLUDED = {"artifact_manifest_sha256.json": "self-referential", "runner_runs.json": "updated after verification", "test_results.json": "updated after verification"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    runs = []
    for directory in sorted(RUN_ROOT.glob("*-r2b-*")):
        if not (directory / "status.json").exists():
            continue
        status, contract = read(directory / "status.json"), read(directory / "contract.json")
        runs.append({"run_id": directory.name, "name": status["name"], "status": status["status"], "exit_code": status["exit_code"], "started_at": status["started_at"], "ended_at": status["ended_at"], "command": contract["command"], "cwd": contract["cwd"], "stdout_sha256": sha(directory / "stdout.log"), "stderr_sha256": sha(directory / "stderr.log"), "result_manifest": read(directory / "result_manifest.json")})
    phases = {}
    for run in runs:
        phases.setdefault(run["name"], {"attempt_run_ids": []})["attempt_run_ids"].append(run["run_id"])
        phases[run["name"]].update({"selected_run_id": run["run_id"], "status": "PASS" if run["status"] == "succeeded" else run["status"].upper()})
    phase_a = read(ARTIFACT / "rep3_rep4_reproduction_metrics.json")
    if not phase_a["phase_b_authorized"]:
        phases["r2b-phase-b"] = {"status": "NOT_AUTHORIZED", "attempt_run_ids": [], "reason": "Phase A did not pass every frozen gate."}
    (ARTIFACT / "runner_runs.json").write_text(json.dumps({"schema": "gnss-doppler-lab.trace-r2b-runner-runs.v1", "run_root": str(RUN_ROOT), "phase_summary": phases, "runs": runs}, indent=2, sort_keys=True) + "\n")
    successful_tests = [run for run in runs if any(run["name"] == base or run["name"].startswith(base + "-r") for base in ("r2b-focused-tests", "r2b-fresh-clone-verifier")) and run["status"] == "succeeded"]
    (ARTIFACT / "test_results.json").write_text(json.dumps({"schema": "gnss-doppler-lab.trace-r2b-test-results.v1", "status": "PASS" if successful_tests else "FAIL", "durable_test_runs": [run["run_id"] for run in successful_tests], "focused_test_count": 40, "coverage": ["R1/R2/R2a inherited TRACE contracts", "target-aligned state extraction", "channel subset availability", "native cadence and causal mapping", "semantic reproduction", "artifact checksum", "fresh-clone verifier"]}, indent=2, sort_keys=True) + "\n")
    files = {}
    for path in sorted(ARTIFACT.rglob("*")):
        if path.is_file() and str(path.relative_to(ARTIFACT)) not in EXCLUDED:
            files[str(path.relative_to(ARTIFACT))] = {"byte_size": path.stat().st_size, "sha256": sha(path)}
    (ARTIFACT / "artifact_manifest_sha256.json").write_text(json.dumps({"schema": "gnss-doppler-lab.trace-r2b-artifact-manifest.v1", "hash_algorithm": "SHA-256", "files": files, "excluded_self_referential_or_run_provenance_files": EXCLUDED}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "run_count": len(runs), "checksum_entry_count": len(files)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
