#!/usr/bin/env python3
"""Collect durable-run provenance, test evidence, and artifact checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2a_reproducibility_repair"
RUN_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/runs")
PHASES = [
    "r2a-preflight-base-and-artifact-inventory",
    "r2a-posthoc-rep1-rep2-root-cause-audit",
    "r2a-receiver-nondeterminism-source-audit",
    "r2a-repair-canonical-serialization",
    "r2a-repair-deterministic-assignment",
    "r2a-repair-frozen-acquisition-handoff",
    "r2a-preregistration-freeze",
    "r2a-phase-a-cleanstatic-rep3",
    "r2a-phase-a-cleanstatic-rep4",
    "r2a-phase-a-ds3-smoke",
    "r2a-phase-a-os3-smoke",
    "r2a-phase-a-semantic-reproduction-evaluate",
    "r2a-phase-b-texbat-core",
    "r2a-phase-b-oakbat-core",
    "r2a-controls-bootstrap-finalize",
    "r2a-tests-and-fresh-clone-verifier",
]
MANIFEST_EXCLUSIONS = {
    "artifact_manifest_sha256.json": "self-referential checksum manifest",
    "runner_runs.json": "updated after the final durable verifier run",
    "test_results.json": "updated after the final fresh-clone verifier run",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def collect_runs() -> tuple[list[dict], dict[str, dict]]:
    runs = []
    for directory in sorted(RUN_ROOT.glob("*-r2a-*")):
        if not directory.is_dir() or not (directory / "status.json").exists():
            continue
        status = read_json(directory / "status.json")
        contract = read_json(directory / "contract.json")
        result = read_json(directory / "result_manifest.json") if (directory / "result_manifest.json").exists() else None
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
                "stdout_sha256": sha256_file(directory / "stdout.log"),
                "stderr_sha256": sha256_file(directory / "stderr.log"),
                "result_manifest": result,
            }
        )
    phase_a = read_json(ARTIFACT / "rep3_rep4_reproduction_metrics.json")
    summary = {}
    for phase in PHASES:
        attempts = [run for run in runs if run["name"] == phase or run["name"].startswith(phase + "-r")]
        if attempts:
            final = attempts[-1]
            summary[phase] = {
                "status": "PASS" if final["status"] == "succeeded" else "FAIL",
                "selected_run_id": final["run_id"],
                "attempt_run_ids": [run["run_id"] for run in attempts],
            }
        elif phase.startswith("r2a-phase-b-") and not phase_a["phase_b_authorized"]:
            summary[phase] = {
                "status": "NOT_AUTHORIZED",
                "reason": "Phase A semantic reproduction did not pass every preregistered source, causal, semantic, score, support, action-mapping, and raw-timeline gate.",
                "attempt_run_ids": [],
            }
        else:
            summary[phase] = {"status": "UNAVAILABLE", "reason": "No durable runner record found.", "attempt_run_ids": []}
    return runs, summary


def write_runner_and_tests(runs: list[dict], phase_summary: dict[str, dict]) -> None:
    (ARTIFACT / "runner_runs.json").write_text(
        json.dumps(
            {
                "schema": "gnss-doppler-lab.trace-r2a-runner-runs.v1",
                "run_root": str(RUN_ROOT),
                "phase_summary": phase_summary,
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    verifier = phase_summary["r2a-tests-and-fresh-clone-verifier"]
    (ARTIFACT / "test_results.json").write_text(
        json.dumps(
            {
                "schema": "gnss-doppler-lab.trace-r2a-test-results.v1",
                "status": verifier["status"],
                "durable_verifier": verifier,
                "coverage": [
                    "canonical key/order",
                    "serialization padding and exact scalar layout",
                    "current/next action semantics",
                    "semantic hash",
                    "common-support alignment",
                    "acquisition/tracking handoff",
                    "clean-only calibration",
                    "causal mapping",
                    "PRN permutation invariance",
                    "variable PRN count",
                    "timeline/onset",
                    "deterministic score reproduction",
                    "artifact checksum",
                    "fresh-clone verifier",
                ],
                "note": "Exact commands, stdout/stderr hashes, exit code, and result manifest are preserved in runner_runs.json and the durable run directory.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_checksum_manifest() -> None:
    entries = {}
    for path in sorted(ARTIFACT.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(ARTIFACT))
        if relative in MANIFEST_EXCLUSIONS:
            continue
        entries[relative] = {"byte_size": path.stat().st_size, "sha256": sha256_file(path)}
    (ARTIFACT / "artifact_manifest_sha256.json").write_text(
        json.dumps(
            {
                "schema": "gnss-doppler-lab.trace-r2a-artifact-manifest.v1",
                "hash_algorithm": "SHA-256",
                "files": entries,
                "excluded_self_referential_or_run_provenance_files": MANIFEST_EXCLUSIONS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    runs, phase_summary = collect_runs()
    write_runner_and_tests(runs, phase_summary)
    write_checksum_manifest()
    print(json.dumps({"status": "PASS", "run_count": len(runs), "checksum_entry_count": len(read_json(ARTIFACT / "artifact_manifest_sha256.json")["files"])}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
