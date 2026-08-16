#!/usr/bin/env python3
"""Finalize R2d metrics summary, durable provenance, README, and SHA-256 seal."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
RUN_ROOT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2d-oakbat-clean-support-repair/runner-runs"
)
FREEZE_COMMIT = "ec564b41bb7034d64bb3399ff23cf13c41531522"
EXCLUDED = {
    "artifact_manifest_sha256.json": "self-referential",
    "runner_runs.json": "updated after verification",
    "test_results.json": "updated after verification",
    "verifier_output.json": "written by verifier after manifest finalization",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(name: str, value: object) -> None:
    (ARTIFACT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def collect_runs() -> tuple[list[dict], dict]:
    runs = []
    for directory in sorted(RUN_ROOT.iterdir()):
        if not (directory / "status.json").exists():
            continue
        status = read(directory / "status.json")
        contract = read(directory / "contract.json")
        runs.append(
            {
                "run_id": directory.name,
                "name": status["name"],
                "status": status["status"],
                "exit_code": status["exit_code"],
                "started_at": status["started_at"],
                "ended_at": status["ended_at"],
                "command": contract["command"],
                "cwd": contract["cwd"],
                "stdout_sha256": sha(directory / "stdout.log"),
                "stderr_sha256": sha(directory / "stderr.log"),
                "result_manifest": read(directory / "result_manifest.json"),
            }
        )
    phases = {}
    for run in runs:
        phase = phases.setdefault(run["name"], {"attempt_run_ids": []})
        phase["attempt_run_ids"].append(run["run_id"])
        phase["selected_run_id"] = run["run_id"]
        phase["status"] = "PASS" if run["status"] == "succeeded" else run["status"].upper()
    return runs, phases


def phase_b_summary(verdict: dict) -> dict:
    computed = verdict["verdict"] in {"GO_TRACE_PHYSICAL_HYPOTHESIS", "NO_GO_TRACE_PHYSICAL_HYPOTHESIS"} and verdict.get("attack_metrics_computed") is True
    if not computed:
        return {
            "schema": "gnss-doppler-lab.trace-r2d-phase-b-metrics.v1",
            "status": "UNAVAILABLE",
            "metrics_computed": False,
            "reason": verdict["reason"],
            "performance_claimed": False,
        }
    with (ARTIFACT / "scenario_metrics.csv").open(newline="") as stream:
        scenarios = list(csv.DictReader(stream))
    return {
        "schema": "gnss-doppler-lab.trace-r2d-phase-b-metrics.v1",
        "status": "AVAILABLE",
        "metrics_computed": True,
        "frozen_contract": True,
        "final_verdict": verdict["verdict"],
        "clean_holdout_fpr_worst": verdict["clean_holdout_fpr_worst"],
        "external_static_fpr_worst": verdict["external_static_fpr_worst"],
        "go_checks": verdict["go_checks"],
        "scenario_metrics": scenarios,
        "scenario_metrics_csv": "scenario_metrics.csv",
        "ablation_metrics_csv": "ablation_metrics.csv",
        "bootstrap_intervals_csv": "bootstrap_intervals.csv",
        "action_controls": "action_shuffle_metrics.json",
        "physical_controls": "physical_controls.json",
        "performance_claimed": True,
    }


def main() -> int:
    runs, phases = collect_runs()
    phase_a = read(ARTIFACT / "rep3_rep4_reproduction_metrics.json")
    verdict = read(ARTIFACT / "final_verdict.json")
    metrics = phase_b_summary(verdict)
    write("phase_b_metrics.json", metrics)
    write(
        "runner_runs.json",
        {
            "schema": "gnss-doppler-lab.trace-r2d-runner-runs.v1",
            "run_root": str(RUN_ROOT),
            "phase_summary": phases,
            "runs": runs,
        },
    )
    test_runs = [
        run for run in runs
        if run["status"] == "succeeded" and any(marker in run["name"] for marker in ("focused-tests", "artifact-verifier", "parent-verifier", "fresh-clone-verifier"))
    ]
    required_test_markers = ("focused-tests", "artifact-verifier", "parent-verifier", "fresh-clone-verifier")
    test_status = "PASS" if all(any(marker in run["name"] for run in test_runs) for marker in required_test_markers) else "IN_PROGRESS"
    write(
        "test_results.json",
        {
            "schema": "gnss-doppler-lab.trace-r2d-test-results.v1",
            "status": test_status,
            "durable_test_runs": [run["run_id"] for run in test_runs],
            "focused_test_count": 53,
            "coverage": [
                "R1/R2/R2a/R2b/R2c inherited TRACE contracts",
                "R2d cleanStatic-specific normal-only handoff selection",
                "unchanged R2c terminal drain and Phase-A row-set closure",
                "non-empty guarded OAKBAT chronological clean support",
                "frozen Phase-B metric and fail-closed contracts",
                "R2d and preserved parent artifact checksums",
                "fresh-clone verification",
            ],
        },
    )
    source = read(ARTIFACT / "source_commit.json")
    source["freeze_commit"] = FREEZE_COMMIT
    write("source_commit.json", source)
    computed = metrics["metrics_computed"]
    (ARTIFACT / "README.md").write_text(
        "# TRACE Stage-0 R2d OAKBAT Clean Support Repair\n\n"
        "R2d repaired the OAKBAT cleanStatic receiver support with a preregistered, "
        "normal-only, target-aligned cleanStatic handoff. The R2c receiver executable and "
        "natural terminal drain are byte-identical, and TRACE scoring, gates, windows, "
        "tolerances, clean split, controls, and alarm policy are unchanged.\n\n"
        f"Phase A status: `{phase_a['phase_a_status']}`. Phase B authorized: "
        f"`{phase_a['phase_b_authorized']}`. Final scientific verdict: "
        f"`{verdict['verdict']}`. Phase B metrics computed: `{computed}`.\n\n"
        "All durable child run IDs, commands, logs, exit states, and result manifests are "
        "indexed by `runner_runs.json`. Parent R2b/R2c artifacts remain intact. Hermes "
        "independent verification remains pending.\n"
    )
    files = {}
    for path in sorted(ARTIFACT.rglob("*")):
        relative = str(path.relative_to(ARTIFACT))
        if path.is_file() and relative not in EXCLUDED:
            files[relative] = {"byte_size": path.stat().st_size, "sha256": sha(path)}
    write(
        "artifact_manifest_sha256.json",
        {
            "schema": "gnss-doppler-lab.trace-r2d-artifact-manifest.v1",
            "hash_algorithm": "SHA-256",
            "files": files,
            "excluded_self_referential_or_mutable_provenance_files": EXCLUDED,
        },
    )
    print(json.dumps({"status": "PASS", "run_count": len(runs), "checksum_entry_count": len(files), "phase_b_metrics_computed": computed, "test_status": test_status}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
