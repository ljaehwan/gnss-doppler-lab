#!/usr/bin/env python3
"""Build a compact, read-only evidence bundle for frozen GCSPO Stage-0 R2 artifacts.

This script copies/verifies existing SSD artifacts only. It must not rerun GCSPO,
retrain cleanStatic, regenerate the source manifest, or modify SSD files.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcspo_stage0_r2_runner_simulation")
RUNS = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/runs")
DEST = REPO / "artifacts/gcspo_stage0_r2_runner_simulation_evidence"
EXPECTED_MANIFEST_HASH = "ad6bbcd34c3889aa393d8699eec4e48c2dcc59095a5a0e3e632442b0bc7205cd"
EXPECTED_FILE_COUNT = 68
MAX_DIRECT_FILE = 25 * 1024 * 1024
LOG_DIRECT_LIMIT = 256 * 1024

CORE_FILES = [
    "README.md",
    "config.json",
    "source_commit.json",
    "run_inventory.json",
    "data_inventory.json",
    "clean_reproduction_check.json",
    "thresholds.json",
    "exact_support_audit.json",
    "common_support_inventory.csv",
    "scenario_metrics.csv",
    "ablation_metrics.csv",
    "external_static_fpr.csv",
    "relation_destruction_metrics.json",
    "physical_controls_audit.json",
    "bootstrap_intervals.csv",
    "shared_state_onset_summary.json",
    "final_verdict.json",
    "artifact_manifest_sha256.json",
]

OPTIONAL_FILES = [
    "b0_full_exact_comparison.json",
    "phase_outputs/preflight/preflight.json",
    "phase_outputs/cleanstatic-normal-model/summary.json",
    "phase_outputs/cleanstatic-normal-model/a5-cuda-process-repair.json",
    "phase_outputs/cleanstatic-normal-model/config-byte-repair.json",
    "phase_outputs/ds3-evaluation/summary.json",
    "phase_outputs/ds3-evaluation/common_support_inventory.csv",
    "phase_outputs/ds3-evaluation/exact_support_audit.json",
    "phase_outputs/ds7-evaluation/summary.json",
    "phase_outputs/ds7-evaluation/common_support_inventory.csv",
    "phase_outputs/ds7-evaluation/exact_support_audit.json",
    "phase_outputs/ds4-ds8-conditional-evaluation/conditional_dispositions.json",
    "phase_outputs/ds4-ds8-conditional-evaluation/replay_overlap_audit.json",
    "phase_outputs/ds4-ds8-conditional-evaluation/unavailable_scenarios.csv",
    "phase_outputs/relation-destruction-physical-controls/summary.json",
    "phase_outputs/relation-destruction-physical-controls/relation_rows.csv",
    "phase_outputs/final-statistics-plots-verification/verification.json",
    "clean_physical_controls.csv",
    "clean_only_report.json",
    "clean_ablation_report.json",
    "clean_a5_report.json",
    "clean_b0_report.json",
]

PER_EPOCH_FILES = ["per_epoch_scores.csv", "shared_state_estimates.csv"]
REQUIRED_PHASES = {
    "preflight": "gcspo-r2-preflight",
    "cleanstatic-normal-model": "gcspo-r2-cleanstatic-normal-model",
    "ds3-evaluation": "gcspo-r2-ds3-evaluation",
    "ds7-evaluation": "gcspo-r2-ds7-evaluation",
    "ds4-ds8-conditional-evaluation": "gcspo-r2-ds4-ds8-conditional-evaluation",
    "relation-destruction-physical-controls": "gcspo-r2-relation-destruction-physical-controls",
    "final-statistics-plots-verification": "gcspo-r2-final-statistics-plots-verification",
}
RUN_SMALL_FILES = ["contract.json", "env.json", "git.json", "heartbeat.json", "status.json", "result_manifest.json"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(DEST).as_posix()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_direct(src_rel: str, *, required: bool, allow_gzip_large: bool = False):
    src = SRC / src_rel
    if not src.exists():
        if required:
            raise FileNotFoundError(src)
        return None
    if src.stat().st_size == 0:
        return {"path": src_rel, "status": "skipped_empty"}
    dst_rel = src_rel
    dst = DEST / dst_rel
    original = {"source_relative_path": src_rel, "source_sha256": sha256_file(src), "source_size_bytes": src.stat().st_size}
    if allow_gzip_large and src.stat().st_size > MAX_DIRECT_FILE // 5:
        dst = DEST / (dst_rel + ".gz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as inf, gzip.GzipFile(filename="", mode="wb", fileobj=dst.open("wb"), mtime=0, compresslevel=9) as outf:
            shutil.copyfileobj(inf, outf)
        return {**original, "bundle_path": rel(dst), "bundle_sha256": sha256_file(dst), "bundle_size_bytes": dst.stat().st_size, "encoding": "gzip-deterministic"}
    if src.stat().st_size > MAX_DIRECT_FILE:
        return summarize_table(src, src_rel, original)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {**original, "bundle_path": rel(dst), "bundle_sha256": sha256_file(dst), "bundle_size_bytes": dst.stat().st_size, "encoding": "identity"}


def summarize_table(src: Path, src_rel: str, original: dict):
    dst_rel = src_rel + ".summary.json"
    counts = Counter()
    first, last = [], []
    row_count = 0
    fieldnames = None
    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            row_count += 1
            key = tuple((k, row.get(k, "")) for k in ("scenario", "method", "phase") if k in row)
            counts[key] += 1
            if len(first) < 20:
                first.append(row)
            last.append(row)
            if len(last) > 20:
                last.pop(0)
    summary = {
        **original,
        "bundle_status": "not_included_too_large",
        "local_absolute_path": str(src),
        "row_count": row_count,
        "column_schema": fieldnames,
        "row_counts_by_available_scenario_method_phase": [
            {**{k: v for k, v in key}, "rows": n} for key, n in sorted(counts.items())
        ],
        "first_20_rows": first,
        "last_20_rows": last,
    }
    dst = DEST / dst_rel
    write_json(dst, summary)
    return {**original, "bundle_path": rel(dst), "bundle_sha256": sha256_file(dst), "bundle_size_bytes": dst.stat().st_size, "encoding": "summary-only"}


def deterministic_gzip_or_summary(src_rel: str):
    src = SRC / src_rel
    original = {"source_relative_path": src_rel, "source_sha256": sha256_file(src), "source_size_bytes": src.stat().st_size}
    dst = DEST / (src_rel + ".gz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as inf, gzip.GzipFile(filename="", mode="wb", fileobj=dst.open("wb"), mtime=0, compresslevel=9) as outf:
        shutil.copyfileobj(inf, outf)
    if dst.stat().st_size <= MAX_DIRECT_FILE:
        return {**original, "bundle_path": rel(dst), "bundle_sha256": sha256_file(dst), "bundle_size_bytes": dst.stat().st_size, "encoding": "gzip-deterministic"}
    dst.unlink()
    return summarize_table(src, src_rel, original)


def verify_source_manifest():
    manifest_path = SRC / "artifact_manifest_sha256.json"
    actual_manifest_hash = sha256_file(manifest_path)
    manifest = load_json(manifest_path)
    entries = manifest["files"]
    result = {
        "expected_manifest_sha256": EXPECTED_MANIFEST_HASH,
        "actual_manifest_sha256": actual_manifest_hash,
        "expected_file_count": EXPECTED_FILE_COUNT,
        "actual_file_count": len(entries),
        "missing": [],
        "hash_mismatches": [],
        "status": "PASS",
    }
    if actual_manifest_hash != EXPECTED_MANIFEST_HASH or len(entries) != EXPECTED_FILE_COUNT:
        result["status"] = "SOURCE_ARTIFACT_CHANGED"
    for e in entries:
        p = SRC / e["path"]
        if not p.exists():
            result["missing"].append(e["path"])
            result["status"] = "SOURCE_ARTIFACT_CHANGED"
            continue
        h = sha256_file(p)
        if h != e["sha256"]:
            mismatch = {"path": e["path"], "expected": e["sha256"], "actual": h}
            result["hash_mismatches"].append(mismatch)
            # fail closed for scientific CSV/JSON. PNG byte drift is noted but not fatal.
            if p.suffix.lower() in {".csv", ".json", ".jsonl"}:
                result["status"] = "SOURCE_ARTIFACT_CHANGED"
    return result


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def summarize_log(path: Path, dst_dir: Path):
    if not path.exists():
        return {"name": path.name, "status": "missing"}
    meta = {"name": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    if path.stat().st_size <= LOG_DIRECT_LIMIT:
        dst = dst_dir / path.name
        shutil.copy2(path, dst)
        return {**meta, "bundle_path": rel(dst), "encoding": "identity"}
    lines = path.read_text(errors="replace").splitlines()
    dst = dst_dir / (path.name + ".summary.json")
    write_json(dst, {**meta, "first_100_lines": lines[:100], "last_200_lines": lines[-200:]})
    return {**meta, "bundle_path": rel(dst), "encoding": "summary-only"}


def build_runner_evidence():
    inventory = load_json(SRC / "run_inventory.json")
    attempts = inventory["attempts"]
    by_phase = defaultdict(list)
    for a in attempts:
        by_phase[a["phase"]].append(a)
    summary = {"required_phases": {}, "all_required_latest_successful": True, "failed_attempts_preserved": []}
    for short, phase in REQUIRED_PHASES.items():
        phase_attempts = sorted(by_phase.get(phase, []), key=lambda a: a.get("created_at", ""))
        successes = [a for a in phase_attempts if a.get("status") == "succeeded" and a.get("exit_code") == 0]
        latest = successes[-1] if successes else None
        if not latest:
            summary["all_required_latest_successful"] = False
            summary["required_phases"][short] = {"phase": phase, "status": "NO_SUCCESSFUL_RUN"}
            continue
        run_id = latest["run_id"]
        run_src = RUNS / run_id
        run_dst = DEST / "runner_runs" / run_id
        run_dst.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in RUN_SMALL_FILES:
            p = run_src / name
            if p.exists() and p.stat().st_size > 0:
                shutil.copy2(p, run_dst / name)
                copied.append({"path": f"runner_runs/{run_id}/{name}", "sha256": sha256_file(run_dst / name), "size_bytes": (run_dst / name).stat().st_size})
        logs = [summarize_log(run_src / "stdout.log", run_dst), summarize_log(run_src / "stderr.log", run_dst)]
        status = load_json(run_src / "status.json") if (run_src / "status.json").exists() else {}
        contract = load_json(run_src / "contract.json") if (run_src / "contract.json").exists() else {}
        heartbeat_exists = (run_src / "heartbeat.json").exists() and (run_src / "heartbeat.json").stat().st_size > 0
        artifacts = latest.get("artifacts", [])
        artifact_existence = [{"path": p, "exists": (SRC / p).exists(), "size_bytes": (SRC / p).stat().st_size if (SRC / p).exists() else None} for p in artifacts]
        prior_failures = [a for a in phase_attempts if a.get("status") != "succeeded" or a.get("exit_code") != 0]
        summary["failed_attempts_preserved"].extend({"phase": phase, "run_id": a["run_id"], "status": a.get("status"), "exit_code": a.get("exit_code")} for a in prior_failures)
        contract_repo_path = contract.get("repo") or contract.get("cwd")
        phase_ok = (
            status.get("status") == "succeeded" and status.get("exit_code") == 0 and heartbeat_exists
            and bool(contract.get("command")) and bool(contract_repo_path)
            and all(x["exists"] and x.get("size_bytes", 0) > 0 for x in artifact_existence)
        )
        if not phase_ok:
            summary["all_required_latest_successful"] = False
        summary["required_phases"][short] = {
            "phase": phase,
            "latest_successful_run_id": run_id,
            "status": status.get("status"),
            "exit_code": status.get("exit_code"),
            "final_heartbeat_exists": heartbeat_exists,
            "contract_command_exists": bool(contract.get("command")),
            "contract_repository_path": contract_repo_path,
            "artifact_phase_outputs": artifact_existence,
            "copied_files": copied,
            "logs": logs,
            "prior_failed_attempt_count": len(prior_failures),
            "phase_terminal_evidence_ok": phase_ok,
        }
    write_json(DEST / "runner_phase_evidence.json", summary)
    return summary


def build_provenance_audit():
    src_file = REPO / "src/gnss_doppler_lab/gcspo_r2_runner.py"
    text = src_file.read_text(encoding="utf-8")
    access_gate_direct = 'gate.set_remote_sync(local_sha=freeze_sha, remote_sha=freeze_sha, ahead=0, behind=0, clean=True)' in text
    preflight = load_json(SRC / "phase_outputs/preflight/preflight.json") if (SRC / "phase_outputs/preflight/preflight.json").exists() else {}
    result = {
        "audited_file": "src/gnss_doppler_lab/gcspo_r2_runner.py",
        "audited_file_sha256_current": sha256_file(src_file),
        "access_gate_remote_sync_sets_local_and_remote_to_freeze_sha_without_live_remote_lookup": access_gate_direct,
        "metric_effect": "none identified",
        "provenance_claim": "weakened" if access_gate_direct else "not_weakened_by_this_pattern",
        "live_remote_sync": "not independently demonstrated at execution time" if access_gate_direct else "not assessed",
        "source_commit_preflight": preflight,
        "execution_dirty_status": preflight.get("dirty_status", "UNVERIFIED"),
        "execution_head": preflight.get("head_at_preflight", "UNVERIFIED"),
        "execution_working_diff_sha256": preflight.get("working_diff_sha256", "UNVERIFIED"),
        "final_commit_contains_same_runner_file": "UNVERIFIED",
    }
    try:
        final_commit = git("rev-parse", "HEAD")
        result["current_head"] = final_commit
        if result["execution_head"] != "UNVERIFIED":
            old = subprocess.check_output(["git", "show", f"{result['execution_head']}:src/gnss_doppler_lab/gcspo_r2_runner.py"], cwd=REPO)
            cur = src_file.read_bytes()
            result["runner_file_hash_at_execution_head"] = sha256_bytes(old)
            result["runner_file_hash_current"] = sha256_bytes(cur)
            result["final_commit_contains_same_runner_file"] = sha256_bytes(old) == sha256_bytes(cur)
    except Exception as exc:
        result["final_commit_contains_same_runner_file_error"] = repr(exc)
    write_json(DEST / "provenance_audit.json", result)
    return result


def build_delivery_status():
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    remote_ref = f"origin/{branch}"
    remote = git("rev-parse", remote_ref)
    ahead, behind = git("rev-list", "--left-right", "--count", f"HEAD...{remote_ref}").split()
    obj = {
        "initial_push_attempt": "failed due missing credentials",
        "later_delivery": "succeeded",
        "current_remote_branch": branch,
        "current_local_commit_sha": head,
        "current_remote_commit_sha": remote,
        "ahead": int(ahead),
        "behind": int(behind),
        "science_verdict_is_separate": True,
        "scientific_verdict": load_json(SRC / "final_verdict.json"),
        "evidence_bundle_commit_sha": "PENDING_COMMIT",
    }
    write_json(DEST / "delivery_status.json", obj)
    return obj


def main():
    source_check = verify_source_manifest()
    if source_check["status"] != "PASS":
        write_json(REPO / "SOURCE_ARTIFACT_CHANGED.json", source_check)
        raise SystemExit("SOURCE_ARTIFACT_CHANGED")
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    copied = []
    for f in CORE_FILES:
        copied.append(copy_direct(f, required=True, allow_gzip_large=False))
    for f in OPTIONAL_FILES:
        copied.append(copy_direct(f, required=False, allow_gzip_large=True))
    for f in PER_EPOCH_FILES:
        copied.append(deterministic_gzip_or_summary(f))
    runner = build_runner_evidence()
    provenance = build_provenance_audit()
    delivery = build_delivery_status()
    # Copy source check after DEST exists.
    write_json(DEST / "source_artifact_integrity.json", source_check)
    bundle_files = []
    total = 0
    for p in sorted(DEST.rglob("*")):
        if p.is_file():
            total += p.stat().st_size
            if p.name == "evidence_bundle_manifest.json":
                continue
            bundle_files.append({"path": rel(p), "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    manifest = {
        "schema": "gnss-doppler-lab.gcspo-stage0-r2.evidence-bundle.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_artifact_root": str(SRC),
        "source_manifest_sha256": source_check["actual_manifest_sha256"],
        "source_file_count": source_check["actual_file_count"],
        "bundle_root_relative": "artifacts/gcspo_stage0_r2_runner_simulation_evidence",
        "copied_or_summarized_sources": [x for x in copied if x],
        "runner_evidence_status": runner,
        "provenance_audit_summary": provenance,
        "delivery_status_summary": delivery,
        "bundle_file_count": len(bundle_files),
        "bundle_file_count_including_manifest": len(bundle_files) + 1,
        "bundle_size_bytes": total,
        "manifest_self_hash_policy": "evidence_bundle_manifest.json is excluded from its own files list to avoid a self-referential hash.",
        "files": bundle_files,
        "final_evidence_judgement": "EVIDENCE_VERIFIED" if runner["all_required_latest_successful"] and source_check["status"] == "PASS" else "EVIDENCE_INCOMPLETE",
        "science_verdict_preserved": {
            "detector": "NO-GO under current configuration",
            "neural_stage1": "not allowed",
            "shared_pull_off_physics": "incomplete",
            "paper_model_continuation": "not recommended",
        },
    }
    write_json(DEST / "evidence_bundle_manifest.json", manifest)
    print(json.dumps({"status": manifest["final_evidence_judgement"], "bundle_size_bytes": total, "bundle_file_count": len(bundle_files)}, indent=2))


if __name__ == "__main__":
    main()
