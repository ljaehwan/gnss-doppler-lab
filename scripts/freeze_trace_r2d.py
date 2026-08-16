#!/usr/bin/env python3
"""Freeze the repaired R2d handoff/config mapping and inherited R2c receiver."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gnss_doppler_lab.trace_native_1ms import sha256_file
import run_trace_stage0_r2d as r2d

driver = r2d.driver
ARTIFACT = driver.ARTIFACT
PARENT = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair"
BASE = "70a3ed273b5d396f6befe06f7aba8d8a3304cd65"
PREREGISTRATION_COMMIT = "4d6c3a290d6d83b0aa2352de16954d21a94b9fab"


def write(name: str, value: object) -> None:
    path = ARTIFACT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    if subprocess.check_output(["git", "merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "HEAD"], cwd=ROOT).decode() != "":
        pass
    handoffs = json.loads((ARTIFACT / "handoffs/manifest.json").read_text())
    clean = handoffs["scenarios"]["OAKBAT.cleanStatic"]
    if not clean["normal_only"] or clean["attack_data_used"]:
        raise ValueError("cleanStatic handoff is not sealed normal-only support")
    mirror = json.loads((ARTIFACT / "handoff_path_mirror_manifest.json").read_text())
    if mirror["status"] != "PASS" or not mirror["sha256_identity"]:
        raise ValueError("short runtime handoff mirror did not pass byte-identity checks")
    frozen = ARTIFACT / "frozen_configs"
    frozen.mkdir(parents=True, exist_ok=True)
    phase_a = {}
    for name, spec in driver.SCENARIOS.items():
        path = frozen / f"{spec['slug']}.conf"
        path.write_text(r2d.frozen_config_text(name))
        phase_a[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    phase_b_dir = frozen / "phase_b"
    phase_b_dir.mkdir(exist_ok=True)
    phase_b = {}
    for name, spec in driver.PHASE_B_SCENARIOS.items():
        path = phase_b_dir / f"{spec['slug']}.conf"
        path.write_text(r2d.frozen_phase_b_config_text(name))
        phase_b[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    shutil.copy2(PARENT / "receiver_repair.diff", ARTIFACT / "receiver_repair.diff")
    shutil.copy2(PARENT / "semantic_reproduction_contract.json", ARTIFACT / "semantic_reproduction_contract.json")
    semantic = json.loads((ARTIFACT / "semantic_reproduction_contract.json").read_text())
    semantic["schema"] = "gnss-doppler-lab.trace-r2d-semantic-reproduction-contract.v1"
    semantic["r2c_terminal_drain_semantics_preserved"] = True
    (ARTIFACT / "semantic_reproduction_contract.json").write_text(json.dumps(semantic, indent=2, sort_keys=True, allow_nan=False) + "\n")
    executable_hash = sha256_file(driver.RECEIVER)
    patch_hash = sha256_file(ARTIFACT / "receiver_repair.diff")
    parent_build = json.loads((PARENT / "receiver_build_manifest.json").read_text())
    if executable_hash != parent_build["receiver_executable"]["sha256"] or patch_hash != parent_build["receiver_repair_diff"]["sha256"]:
        raise ValueError("R2c receiver executable or patch changed")
    write(
        "receiver_build_manifest.json",
        {
            "schema": "gnss-doppler-lab.trace-r2d-receiver-build-manifest.v1",
            "status": "PASS_REUSED_BYTE_IDENTICAL_R2C",
            "receiver_base_commit": parent_build["receiver_base_commit"],
            "receiver_source_path": parent_build["receiver_source_path"],
            "receiver_build_path": parent_build["receiver_build_path"],
            "receiver_executable": {"path": str(driver.RECEIVER), "byte_size": driver.RECEIVER.stat().st_size, "sha256": executable_hash},
            "receiver_repair_diff": {"path": str((ARTIFACT / "receiver_repair.diff").relative_to(ROOT)), "byte_size": (ARTIFACT / "receiver_repair.diff").stat().st_size, "sha256": patch_hash},
            "terminal_drain_semantics": "unchanged byte-identical R2c executable and patch",
            "build_step": "No rebuild: verified byte-identical reuse avoids changing the terminal-drain implementation.",
        },
    )
    plan = json.loads((ARTIFACT / "repair_plan_preregistered.json").read_text())
    prereg = json.loads((ARTIFACT / "preregistration.json").read_text())
    prereg["status"] = "FROZEN_AFTER_PREREGISTERED_NORMAL_ONLY_HANDOFF"
    prereg["phase_a_pass_requires"] = json.loads((PARENT / "preregistration.json").read_text())["phase_a_pass_requires"]
    prereg["source_hashes"] = {
        "receiver_executable_sha256": executable_hash,
        "receiver_patch_sha256": patch_hash,
        "receiver_configs": {"phase_a": phase_a, "phase_b": phase_b},
        "handoffs": {name: item["handoff_sha256"] for name, item in handoffs["scenarios"].items()},
    }
    prereg["repair_plan"] = plan
    prereg["config_path_length_amendment"] = "config_path_length_amendment_preregistered.json"
    prereg["handoff_path_mirror"] = "handoff_path_mirror_manifest.json"
    write("preregistration.json", prereg)
    write("handoff_manifest.json", handoffs)
    parent_config = json.loads((PARENT / "config.json").read_text())
    write(
        "config.json",
        {
            "schema": "gnss-doppler-lab.trace-r2d-config.v1",
            "frozen_trace_r2_score_policy": parent_config["frozen_trace_r2_score_policy"],
            "semantic_tolerances": parent_config["semantic_tolerances"],
            "native_dump_schema": parent_config["native_dump_schema"],
            "finite_source_terminal_policy": parent_config["finite_source_terminal_policy"],
            "frozen_receiver_configs": {"phase_a": phase_a, "phase_b": phase_b},
            "only_support_mapping_change": {
                "scenario": "OAKBAT.cleanStatic",
                "seconds_to_skip_before": 90.0,
                "seconds_to_skip_after": 0.0,
                "handoff_before": "oakbat_os3.csv",
                "handoff_after": "oakbat_cleanstatic.csv",
            },
            "runtime_handoff_path_mirror": mirror,
        },
    )
    write(
        "source_commit.json",
        {
            "schema": "gnss-doppler-lab.trace-r2d-source-freeze.v1",
            "research_branch": "research/trace-stage0-r2d-oakbat-clean-support-repair",
            "task_base_commit": BASE,
            "parent_r2c_commit": BASE,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "freeze_commit": "TO_BE_RECORDED_AFTER_SUPPORT_FREEZE_COMMIT",
            "receiver_executable_sha256": executable_hash,
            "receiver_patch_sha256": patch_hash,
        },
    )
    print(json.dumps({"status": "PASS", "clean_handoff_sha256": clean["handoff_sha256"], "receiver_executable_sha256": executable_hash, "phase_a_config_count": len(phase_a), "phase_b_config_count": len(phase_b)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
