#!/usr/bin/env python3
"""Materialize the TRACE-R2a code/config/semantic preregistration freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gnss_doppler_lab.trace_native_1ms import sha256_file
from run_trace_stage0_r2a import ARTIFACT, PHASE_B_SCENARIOS, RECEIVER, SCENARIOS, frozen_config_text, frozen_phase_b_config_text

RECEIVER_SOURCE = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2a-reproducibility-repair/receiver-source")


def dump_json(name: str, value: object) -> None:
    path = ARTIFACT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def command_output(command: list[str], cwd: Path = ROOT) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True)


def main() -> int:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    frozen_configs = ARTIFACT / "frozen_configs"
    frozen_configs.mkdir(exist_ok=True)
    config_hashes = {}
    for name, spec in SCENARIOS.items():
        path = frozen_configs / f"{spec['slug']}.conf"
        path.write_text(frozen_config_text(name))
        config_hashes[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    phase_b_config_hashes = {}
    phase_b_configs = frozen_configs / "phase_b"
    phase_b_configs.mkdir(exist_ok=True)
    for name, spec in PHASE_B_SCENARIOS.items():
        path = phase_b_configs / f"{spec['slug']}.conf"
        path.write_text(frozen_phase_b_config_text(name))
        phase_b_config_hashes[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}

    receiver_diff = command_output(
        [
            "git",
            "-c",
            f"safe.directory={RECEIVER_SOURCE}",
            "diff",
            "--binary",
            "--no-ext-diff",
            "1ddd4562723040fd66cb334b578a5b69455625f4",
        ],
        RECEIVER_SOURCE,
    )
    repair_path = ARTIFACT / "receiver_repair.diff"
    repair_path.write_text(receiver_diff)
    receiver_patch_hash = sha256_file(repair_path)
    executable_hash = sha256_file(RECEIVER)
    phase_b_scorer_hashes = {
        "frozen_r2_scorer": sha256_file(ROOT / "scripts/evaluate_trace_r2_phase_b.py"),
        "r2a_path_only_adapter": sha256_file(ROOT / "scripts/evaluate_trace_r2a_phase_b.py"),
    }
    raw_binding = json.loads((ARTIFACT / "raw_source_binding.json").read_text())
    if raw_binding.get("status") != "PASS" or set(raw_binding.get("datasets", {})) != set(PHASE_B_SCENARIOS):
        raise ValueError("raw source freeze must PASS for every preregistered Phase-B dataset")
    handoffs = json.loads((ARTIFACT / "handoffs/manifest.json").read_text())
    base_sha = command_output(["git", "rev-parse", "HEAD"]).strip()
    repo_diff = command_output(["git", "diff", "--binary", "HEAD"])
    untracked = command_output(["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
    untracked_hashes = {path: sha256_file(ROOT / path) for path in sorted(untracked) if (ROOT / path).is_file()}

    semantic_contract = {
        "schema": "gnss-doppler-lab.trace-r2a-semantic-reproduction-contract.v1",
        "frozen_before_rep3": True,
        "canonical_key": ["dataset", "prn", "raw_interval_start_sample", "raw_interval_end_sample"],
        "metadata_excluded_from_physical_identity": ["channel", "tracking_session_id", "file_order", "local_loop_sequence"],
        "stable_normal_region": {
            "quality": "valid_tracking=1, cn0_db_hz>=28, carrier_lock_test>=0.85, prompt finite/nonzero",
            "minimum_common_prns_per_rounded_ms_epoch": 4,
            "minimum_common_epoch_count": 1000,
        },
        "source_gate": {
            "raw_iq_sha256_identical": True,
            "receiver_executable_sha256_identical": True,
            "receiver_config_sha256_identical": True,
            "receiver_patch_sha256_identical": True,
            "raw_sample_range_identical": True,
        },
        "causal_gate": {
            "sequence_mismatch_count": 0,
            "action_value_mapping_mismatch_count": 0,
            "consume_span_mismatch_count": 0,
            "finite_failure_count": 0,
            "zero_tap_row_count": 0,
            "zero_action_row_count": 0,
            "repeated_placeholder_row_count": 0,
        },
        "semantic_tolerances": {
            "float32_complex_taps_absolute": 0.0,
            "float64_actions_and_state_absolute": 0.0,
            "integer_fields": "exact",
            "trace_score_absolute": 1e-12,
            "trace_score_relative": 0.0,
            "basis": "Same executable, CPU, explicit scalar serialization, fixed PRN/Doppler/raw-start handoff, and fixed floating reduction order should be bitwise deterministic; the 1e-12 derived-score allowance only covers deterministic float64 recomputation order in the evaluator.",
        },
        "decision_gate": {
            "threshold_policy": "clean calibration q99 (q99.5 also reported), frozen R2 policy",
            "threshold_crossing_identical": True,
            "alarm_decision_identical": True,
            "metadata_only_differences_must_be_separately_proven": True,
            "whole_file_sha_match_required": False,
            "physical_semantic_match_required": True,
        },
        "phase_b_authorization": "Only if every Phase-A source, causal, semantic, score, support, action-mapping, and raw-timeline condition passes.",
    }
    dump_json("semantic_reproduction_contract.json", semantic_contract)
    dump_json(
        "config.json",
        {
            "schema": "gnss-doppler-lab.trace-r2a-config.v1",
            "frozen_trace_r2_score_policy": json.loads((ROOT / "artifacts/trace_stage0_r2_native_1ms_dump/config.json").read_text())["frozen_trace_r1"],
            "native_dump_schema": {"version": 2, "header_bytes": 192, "record_bytes": 416, "tap_count": 9, "integration_s": 0.001},
            "frozen_receiver_configs": {"phase_a": config_hashes, "phase_b": phase_b_config_hashes},
            "semantic_tolerances": semantic_contract["semantic_tolerances"],
        },
    )
    dump_json(
        "preregistration.json",
        {
            "schema": "gnss-doppler-lab.trace-r2a-preregistration.v1",
            "scientific_objective": "Repair cleanStatic receiver reproducibility without changing frozen TRACE scoring or performance gates.",
            "post_hoc_root_cause_audit_completed_before_repair": True,
            "attack_data_used_for_tolerance_or_repair_selection": False,
            "phase_a_replays": ["TEXBAT.cleanStatic.rep3", "TEXBAT.cleanStatic.rep4", "TEXBAT.DS3.smoke", "OAKBAT.OS3.smoke"],
            "phase_a_pass_requires": ["source gate PASS", "causal gate PASS", "semantic reproduction gate PASS", "rep3/rep4 common-support TRACE score within 1e-12", "DS3/OS3 native 1 ms and >=4 PRN support", "action mapping mismatch 0", "raw source/timeline binding PASS"],
            "phase_b_scope_if_authorized": ["TEXBAT.cleanStatic", "TEXBAT.DS3", "TEXBAT.DS7", "OAKBAT.cleanStatic", "OAKBAT.OS3", "OAKBAT.OS4"],
            "frozen_components": ["predictor", "action-conditioned transition", "features", "complex normalization", "score", "whitening", "window", "threshold policy", "ablations", "action shuffle", "scenario gate", "GO/NO-GO criteria"],
            "phase_b_scorer": {"implementation": "scripts/evaluate_trace_r2_phase_b.py", "adapter": "scripts/evaluate_trace_r2a_phase_b.py", "sha256": phase_b_scorer_hashes, "adapter_scope": "artifact and dump roots plus Phase-A inventory handoff only; no scoring math"},
            "receiver_repair": ["retain R2 explicit scalar serialization", "fix channel-to-PRN map", "load pre-onset frozen PRN/Doppler/first-raw-interval handoff", "fail closed if target raw interval is missed"],
            "phase_b_handoff_procedure_if_authorized": "Reuse only the already frozen pre-onset family handoffs: TEXBAT cleanStatic uses its 0 s clean handoff; DS3 and DS7 use the TEXBAT DS3 handoff derived from the 90 s pre-onset slice; OAKBAT cleanStatic, OS3, and OS4 use the OAKBAT OS3 handoff derived from the 90 s pre-onset slice. All Phase-B configs begin at the matching family timeline and were hashed before Phase A. No new acquisition pass, post-onset row, or attack metric may enter handoff selection.",
            "source_hashes": {"receiver_executable_sha256": executable_hash, "receiver_patch_sha256": receiver_patch_hash, "phase_b_scorer": phase_b_scorer_hashes, "receiver_configs": {"phase_a": config_hashes, "phase_b": phase_b_config_hashes}, "raw_iq": {name: row["fresh_sha256"] for name, row in raw_binding["datasets"].items()}, "handoffs": {name: row["handoff_sha256"] for name, row in handoffs["scenarios"].items()}},
            "repository_freeze_basis": {"parent_commit": base_sha, "tracked_diff_sha256": sha_text(repo_diff), "untracked_file_sha256": untracked_hashes},
            "semantic_contract": "semantic_reproduction_contract.json",
        },
    )
    dump_json(
        "r2_existing_failure_preservation.json",
        {
            "schema": "gnss-doppler-lab.trace-r2a-existing-failure-preservation.v1",
            "preserved_verdict": "INCONCLUSIVE_INPUT_OR_RECEIVER",
            "r2_final_verdict_path": "artifacts/trace_stage0_r2_native_1ms_dump/final_verdict.json",
            "r2_final_verdict_sha256": sha256_file(ROOT / "artifacts/trace_stage0_r2_native_1ms_dump/final_verdict.json"),
            "r2_artifacts_modified": False,
            "failure": "Same receiver executable/config/raw slice produced different per-channel hashes and acquisition/tracking timing.",
        },
    )
    dump_json(
        "receiver_build_manifest.json",
        {
            "schema": "gnss-doppler-lab.trace-r2a-receiver-build-manifest.v1",
            "status": "PASS",
            "receiver_base_commit": "1ddd4562723040fd66cb334b578a5b69455625f4",
            "receiver_source_path": str(RECEIVER_SOURCE),
            "receiver_build_path": str(RECEIVER.parent.parent.parent),
            "receiver_executable": {"path": str(RECEIVER), "byte_size": RECEIVER.stat().st_size, "sha256": executable_hash},
            "receiver_repair_diff": {"path": str(repair_path.relative_to(ROOT)), "sha256": receiver_patch_hash, "byte_size": repair_path.stat().st_size},
            "build_command": "env PYTHONPATH=/usr/lib/python3/dist-packages GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=* cmake --build <R2a receiver-build> --target gnss-sdr -j 4",
        },
    )
    dump_json(
        "source_commit.json",
        {
            "schema": "gnss-doppler-lab.trace-r2a-source-freeze.v1",
            "research_branch": "research/trace-stage0-r2a-reproducibility-repair",
            "pinned_parent_commit": base_sha,
            "pinned_base_remote_commit": "948658aa7b117a40b102a48193022e6c403cbc00",
            "receiver_base_commit": "1ddd4562723040fd66cb334b578a5b69455625f4",
            "receiver_patch_sha256": receiver_patch_hash,
            "receiver_executable_sha256": executable_hash,
            "preregistration_commit": "TO_BE_RECORDED_BY_PRE_REPLAY_PROVENANCE_COMMIT",
        },
    )
    (ARTIFACT / "README.md").write_text(
        "# TRACE Stage-0 R2a reproducibility repair\n\n"
        "This artifact tree preserves the R2 failure, performs the post-hoc canonical PRN/raw-sample audit, freezes the receiver handoff repair, and gates Phase B on Phase A semantic reproduction. Large native dumps remain under the R2a SSD root.\n"
    )
    print(json.dumps({"status": "PASS", "receiver_executable_sha256": executable_hash, "receiver_patch_sha256": receiver_patch_hash, "frozen_configs": {"phase_a": config_hashes, "phase_b": phase_b_config_hashes}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
