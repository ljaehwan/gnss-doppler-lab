#!/usr/bin/env python3
"""Freeze the repaired R2b receiver, configs, handoffs, and source provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_trace_stage0_r2b  # noqa: F401
import run_trace_stage0_r2a as driver
from gnss_doppler_lab.trace_native_1ms import sha256_file

ARTIFACT = driver.ARTIFACT
SOURCE = driver.SSD_ROOT / "receiver-source-r1"
BASE = "1ddd4562723040fd66cb334b578a5b69455625f4"


def write(name: str, value: object) -> None:
    path = ARTIFACT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    frozen = ARTIFACT / "frozen_configs"
    frozen.mkdir(parents=True, exist_ok=True)
    config_hashes = {}
    for name, spec in driver.SCENARIOS.items():
        path = frozen / f"{spec['slug']}.conf"
        path.write_text(driver.frozen_config_text(name))
        config_hashes[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    phase_b_hashes = {}
    phase_b_dir = frozen / "phase_b"
    phase_b_dir.mkdir(exist_ok=True)
    for name, spec in driver.PHASE_B_SCENARIOS.items():
        path = phase_b_dir / f"{spec['slug']}.conf"
        path.write_text(driver.frozen_phase_b_config_text(name))
        phase_b_hashes[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    diff = subprocess.check_output(["git", "-c", f"safe.directory={SOURCE}", "-C", str(SOURCE), "diff", "--binary", "--no-ext-diff", BASE], text=True)
    repair = ARTIFACT / "receiver_repair.diff"
    repair.write_text(diff)
    handoffs = json.loads((ARTIFACT / "handoffs/manifest.json").read_text())
    executable_hash = sha256_file(driver.RECEIVER)
    patch_hash = sha256_file(repair)
    semantic = json.loads((ROOT / "artifacts/trace_stage0_r2a_reproducibility_repair/semantic_reproduction_contract.json").read_text())
    semantic["schema"] = "gnss-doppler-lab.trace-r2b-semantic-reproduction-contract.v1"
    semantic["frozen_before_rep3"] = True
    write("semantic_reproduction_contract.json", semantic)
    source_hashes = {"receiver_executable_sha256": executable_hash, "receiver_patch_sha256": patch_hash, "receiver_configs": {"phase_a": config_hashes, "phase_b": phase_b_hashes}, "handoffs": {name: row["handoff_sha256"] for name, row in handoffs["scenarios"].items()}}
    plan = json.loads((ARTIFACT / "repair_plan_preregistered.json").read_text())
    write("preregistration.json", {"schema": "gnss-doppler-lab.trace-r2b-preregistration.v1", "scientific_objective": "Restore exact target-aligned receiver state and repeat unchanged frozen Phase A.", "phase_a_replays": ["TEXBAT.cleanStatic.rep3", "TEXBAT.cleanStatic.rep4", "TEXBAT.DS3.smoke", "OAKBAT.OS3.smoke"], "phase_a_pass_requires": ["source gate PASS", "causal gate PASS", "semantic reproduction gate PASS", "rep3/rep4 common-support TRACE score within 1e-12", "DS3/OS3 native 1 ms and >=4 PRN support", "action mapping mismatch 0", "raw source/timeline binding PASS"], "phase_b_scope_if_authorized": list(driver.PHASE_B_SCENARIOS), "phase_b_scorer": {"implementation": "scripts/evaluate_trace_r2_phase_b.py", "adapter": "scripts/evaluate_trace_r2b_phase_b.py", "frozen_r2_sha256": sha256_file(ROOT / "scripts/evaluate_trace_r2_phase_b.py")}, "source_hashes": source_hashes, "semantic_contract": "semantic_reproduction_contract.json", "repair_plan": plan, "attack_data_used_for_tolerance_or_repair_selection": False})
    write("receiver_build_manifest.json", {"schema": "gnss-doppler-lab.trace-r2b-receiver-build-manifest.v1", "status": "PASS", "receiver_base_commit": BASE, "receiver_source_path": str(SOURCE), "receiver_build_path": str(driver.SSD_ROOT / "receiver-build"), "receiver_executable": {"path": str(driver.RECEIVER), "byte_size": driver.RECEIVER.stat().st_size, "sha256": executable_hash}, "receiver_repair_diff": {"path": str(repair.relative_to(ROOT)), "byte_size": repair.stat().st_size, "sha256": patch_hash}, "build_command": "cmake --build <R2b receiver-build> --target gnss-sdr -j 4"})
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write("source_commit.json", {"schema": "gnss-doppler-lab.trace-r2b-source-freeze.v1", "research_branch": "research/trace-stage0-r2b-stable-handoff-repair", "pinned_parent_commit": head, "task_base_commit": "37f3118cd5ca2de6aa507f2e9a15b0f48fc4f52c", "receiver_base_commit": BASE, "receiver_patch_sha256": patch_hash, "receiver_executable_sha256": executable_hash, "preregistration_commit": "TO_BE_RECORDED_AFTER_FREEZE_COMMIT"})
    write("handoff_manifest.json", handoffs)
    write("config.json", {"schema": "gnss-doppler-lab.trace-r2b-config.v1", "frozen_trace_r2_score_policy": json.loads((ROOT / "artifacts/trace_stage0_r2_native_1ms_dump/config.json").read_text())["frozen_trace_r1"], "native_dump_schema": {"version": 2, "header_bytes": 192, "record_bytes": 416, "tap_count": 9, "integration_s": 0.001}, "frozen_receiver_configs": {"phase_a": config_hashes, "phase_b": phase_b_hashes}, "semantic_tolerances": semantic["semantic_tolerances"]})
    print(json.dumps({"status": "PASS", "receiver_executable_sha256": executable_hash, "receiver_patch_sha256": patch_hash, "config_hashes": config_hashes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
