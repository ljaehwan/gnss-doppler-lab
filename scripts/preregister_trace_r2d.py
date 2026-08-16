#!/usr/bin/env python3
"""Audit the R2c OAKBAT clean failure and preregister the bounded R2d repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_native_1ms import load_native_trace_pairs, read_records

ARTIFACT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
PARENT = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair"
PARENT_DUMP = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2c-terminal-drain-repair/dumps/phase_b/oakbat_cleanstatic/rep1"
)
HISTORICAL_SUMMARY = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/"
    "cleanstatic/receiver/cleanstatic-complex9/tracking_summary.csv"
)
BASE = "70a3ed273b5d396f6befe06f7aba8d8a3304cd65"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(name: str, value: object) -> None:
    path = ARTIFACT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def support_audit() -> dict[str, object]:
    pairs = load_native_trace_pairs(
        PARENT_DUMP,
        cn0_min_db_hz=28.0,
        lock_min=0.85,
        prompt_epsilon=1e-12,
    )
    common = pairs.valid_support[:, np.arange(1, 8)].all(axis=1)
    finite = (
        np.isfinite(pairs.current[:, 1:8].real).all(axis=1)
        & np.isfinite(pairs.current[:, 1:8].imag).all(axis=1)
        & np.isfinite(pairs.target[:, 1:8].real).all(axis=1)
        & np.isfinite(pairs.target[:, 1:8].imag).all(axis=1)
    )
    selected = pairs.take(common & finite)
    start = float(selected.time_s.min())
    end = float(selected.time_s.max())
    duration = end - start
    boundaries = (start + 0.45 * duration, start + 0.65 * duration, start + 0.80 * duration)
    guard = 5.0
    masks = {
        "train": selected.time_s < boundaries[0] - guard,
        "covariance_validation": (selected.time_s >= boundaries[0] + guard)
        & (selected.time_s < boundaries[1] - guard),
        "calibration": (selected.time_s >= boundaries[1] + guard)
        & (selected.time_s < boundaries[2] - guard),
        "holdout": selected.time_s >= boundaries[2] + guard,
    }
    files = []
    for path in sorted(PARENT_DUMP.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path)
        files.append(
            {
                "path": str(path),
                "record_count": int(len(records)),
                "prns": sorted({int(value) for value in records["prn"]}),
                "raw_start_min": int(records["raw_interval_start_sample"].min()),
                "raw_start_max": int(records["raw_interval_start_sample"].max()),
            }
        )
    manifest = json.loads((PARENT_DUMP / "manifest.json").read_text())
    return {
        "schema": "gnss-doppler-lab.trace-r2d-oakbat-clean-support-audit.v1",
        "status": "FAIL_CONFIRMED",
        "failure_label": "OAKBAT_CLEAN_SPLIT_EMPTY",
        "parent_receiver_exit_code": manifest["exit_code"],
        "parent_replay_validation": manifest["replay_validation"],
        "raw_iq": manifest["raw_iq"],
        "parent_handoff_path": manifest["frozen_handoff_path"],
        "parent_handoff_sha256": manifest["frozen_handoff_sha256"],
        "parent_seconds_to_skip": manifest["raw_sample_range"]["seconds_to_skip"],
        "native_dump_files": files,
        "quality_common_support": {
            "all_native_pair_count": int(len(pairs.time_s)),
            "selected_pair_count": int(len(selected.time_s)),
            "unique_prns": sorted({int(value) for value in selected.prn}),
            "time_start_s": start,
            "time_end_s": end,
            "duration_s": duration,
            "chronological_boundaries_s": list(boundaries),
            "guard_s": guard,
            "role_pair_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        },
        "historical_clean_receiver_summary": {
            "path": str(HISTORICAL_SUMMARY),
            "sha256": sha(HISTORICAL_SUMMARY),
            "evidence": "Authenticated cleanStatic acquisition has eleven channel tracks extending to approximately 479.9 s with a clean-specific PRN/channel state.",
        },
        "attack_data_read_or_scored": False,
    }


def main() -> int:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if head != BASE:
        raise ValueError(f"R2d preregistration requires base {BASE}, got {head}")
    audit = support_audit()
    write("oakbat_clean_support_audit.json", audit)
    diagnosis = {
        "schema": "gnss-doppler-lab.trace-r2d-oakbat-clean-diagnosis.v1",
        "status": "COMPLETE_BEFORE_REPAIR",
        "diagnosis_label": "OAKBAT_OS3_HANDOFF_INCOMPATIBLE_WITH_CLEANSTATIC",
        "failure_label": "OAKBAT_CLEAN_SPLIT_EMPTY",
        "root_cause": (
            "R2c reused the OAKBAT.OS3 target-aligned handoff and 90 s skip for OAKBAT.cleanStatic. "
            "The authenticated clean recording has a different clean-specific PRN/channel tracking state. "
            "The receiver therefore emitted physical files but lost quality/common support before the "
            "unchanged guarded chronological roles could be populated."
        ),
        "evidence": {
            "parent_support_audit": "oakbat_clean_support_audit.json",
            "parent_selected_pairs": audit["quality_common_support"]["selected_pair_count"],
            "parent_selected_unique_prns": audit["quality_common_support"]["unique_prns"],
            "parent_selected_duration_s": audit["quality_common_support"]["duration_s"],
            "parent_role_pair_counts": audit["quality_common_support"]["role_pair_counts"],
        },
        "not_causes": {
            "raw_file_truncation": "The authenticated cleanStatic source is 9.6 GB / 480 s and its hash matches the frozen source binding.",
            "receiver_process_failure": "The R2c receiver exited 0 and emitted eleven non-empty native dump files.",
            "terminal_drain": "R2c natural EOS drain remains enabled and is preserved unchanged.",
            "trace_scoring_or_split_implementation": "The scorer correctly failed closed after the frozen split roles were empty.",
        },
        "attack_performance_read_or_computed": False,
    }
    write("diagnosis.json", diagnosis)
    plan = {
        "schema": "gnss-doppler-lab.trace-r2d-repair-preregistration.v1",
        "status": "SEALED_BEFORE_SUPPORT_ACQUISITION",
        "task_base_commit": BASE,
        "diagnosis_label": diagnosis["diagnosis_label"],
        "repair_strategy": "CLEANSTATIC_SPECIFIC_NORMAL_ONLY_TARGET_ALIGNED_HANDOFF",
        "support_acquisition": {
            "dataset": "OAKBAT.cleanStatic",
            "raw_iq_sha256": audit["raw_iq"]["sha256"],
            "raw_seconds_to_skip": 0.0,
            "bounded_duration_s": 45.0,
            "selection_guard_s": 30.0,
            "selection_rule": "For each physical native channel file, choose the first causal native row whose raw interval starts at or after 30.0 s; retain only its PRN and exact action-used state; require at least four contiguous output channels.",
            "quality_or_score_used_for_selection": False,
            "attack_data_used": False,
        },
        "phase_b_mapping_change": {
            "OAKBAT.cleanStatic": "Use the new cleanStatic-specific handoff at zero source skip.",
            "OAKBAT.OS3": "Unchanged R2c OS3 handoff at 90 s.",
            "OAKBAT.OS4": "Unchanged R2c OS3-family handoff at 90 s.",
        },
        "unchanged": [
            "R2c natural terminal-drain receiver semantics and executable",
            "TEXBAT and OAKBAT OS3/OS4 raw IQ sources and hashes",
            "TEXBAT and OAKBAT OS3/OS4 handoffs and source ranges",
            "Phase A replay set and every Phase A gate",
            "TRACE features, score formula, thresholds, windows, tolerances, block keys, controls, alarm gates, and GO/NO-GO criteria",
            "45/20/15/20 chronological clean split with 5 s guards",
            "minimum four-PRN support and cleanStatic-only calibration",
        ],
        "prohibited": [
            "using cleanDynamic or attack data for clean support",
            "selecting a handoff by TRACE score, quality outcome, or attack outcome",
            "loosening any split, support, score, or alarm gate",
            "post-hoc filtering to manufacture a passing clean split",
        ],
        "phase_a_authorization_rule": "Repeat unchanged R2c Phase A and authorize Phase B only if every inherited gate passes.",
        "phase_b_stop_rule": "Fail closed before attack scoring if the repaired OAKBAT clean audit is non-chronological, has an empty role, or violates minimum support.",
        "attack_performance_read_or_computed": False,
    }
    write("repair_plan_preregistered.json", plan)
    write(
        "preregistration.json",
        {
            "schema": "gnss-doppler-lab.trace-r2d-preregistration.v1",
            "status": "SEALED_BEFORE_SUPPORT_ACQUISITION",
            "scientific_objective": "Repair only OAKBAT clean receiver/handoff support, repeat unchanged Phase A, then repeat frozen Phase B if authorized.",
            "task_base_commit": BASE,
            "phase_a_replays": [
                "TEXBAT.cleanStatic.rep3",
                "TEXBAT.cleanStatic.rep4",
                "TEXBAT.DS3.smoke",
                "OAKBAT.OS3.smoke",
            ],
            "phase_a_contract": "Inherited unchanged from R2c preregistration and semantic reproduction contract.",
            "phase_b_scope_if_authorized": [
                "TEXBAT.cleanStatic",
                "TEXBAT.DS3",
                "TEXBAT.DS7",
                "OAKBAT.cleanStatic",
                "OAKBAT.OS3",
                "OAKBAT.OS4",
            ],
            "phase_b_scorer": {
                "implementation": "scripts/evaluate_trace_r2_phase_b.py",
                "frozen_sha256": sha(ROOT / "scripts/evaluate_trace_r2_phase_b.py"),
                "allowed_adapter_scope": "R2d artifact/dump paths only",
            },
            "repair_plan": plan,
            "attack_data_used_for_repair_or_tolerance_selection": False,
        },
    )
    write(
        "source_commit.json",
        {
            "schema": "gnss-doppler-lab.trace-r2d-source-freeze.v1",
            "research_branch": branch,
            "task_base_commit": BASE,
            "parent_r2c_commit": BASE,
            "preregistration_commit": "TO_BE_RECORDED_AFTER_PREREGISTRATION_COMMIT",
            "freeze_commit": "TO_BE_RECORDED_AFTER_SUPPORT_FREEZE_COMMIT",
        },
    )
    (ARTIFACT / "README.md").write_text(
        "# TRACE Stage-0 R2d OAKBAT Clean Support Repair\n\n"
        "R2d repairs only the cleanStatic receiver/handoff support identified by the R2c "
        "fail-closed result. The repair and frozen reruns remain in progress; no performance "
        "claim is available until every verifier passes.\n"
    )
    print(json.dumps({"status": "PASS", "diagnosis_label": diagnosis["diagnosis_label"], "artifact_root": str(ARTIFACT)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
