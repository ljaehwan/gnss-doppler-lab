#!/usr/bin/env python3
"""Freeze MIRAGE R1 design and split before any scientific scoring or injection."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.mirage_r1 import (  # noqa: E402
    DELAYS, EPSILON_RELATIVE, SCALES, XI, assign_cases, canonical_sha,
    cramers_v, design_balance,
)
from gnss_doppler_lab.trace_native_1ms import read_records  # noqa: E402

ART = ROOT / "artifacts/mirage_stage0a_r1_full_execution"
FOUNDATION_COMMIT = "596f78485a1d17899665cb37d94634cc666e0bdd"
SEEDS = {"OAKBAT.cleanStatic": 20260819, "TEXBAT.cleanStatic": 20260820}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(name: str, value: object) -> None:
    (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, text=True, check=True, stdout=subprocess.PIPE).stdout.strip()


def nearest_endpoint(trace: Path, target: int) -> int:
    _, records = read_records(trace, mmap=True)
    starts = records["raw_interval_start_sample"].astype(np.int64)
    index = int(np.searchsorted(starts, target))
    options = [i for i in (index - 1, index) if 0 <= i < len(starts)]
    return int(starts[min(options, key=lambda i: abs(int(starts[i]) - target))])


def manifest() -> None:
    unavailable = [
        "plots/clean_score_ecdf.png", "plots/injection_roc.png",
        "plots/control_effects.png", "plots/relation_destruction.png",
    ]
    files = []
    for path in sorted(ART.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            files.append({"path": str(path.relative_to(ART)), "size_bytes": path.stat().st_size, "sha256": sha(path)})
    write("artifact_manifest_sha256.json", {
        "schema": "gnss-doppler-lab.artifact-manifest-sha256.v1",
        "files": files, "unavailable_until_execution": unavailable,
    })


def main() -> None:
    head = command("git", "rev-parse", "HEAD")
    branch = command("git", "branch", "--show-current")
    if head != FOUNDATION_COMMIT:
        raise SystemExit(f"preregistration requires foundation {FOUNDATION_COMMIT}; got {head}")
    if branch != "research/mirage-stage0a-r1-full-execution":
        raise SystemExit(f"wrong branch: {branch}")

    inventory = json.loads((ART / "data_inventory.json").read_text())
    support = {}
    with (ART / "common_support_segments.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            support[row["dataset"]] = row

    roles = {}
    all_cases = []
    split_audit = {"datasets": {}, "role_overlap": False, "byte_overlap": False,
                   "ten_second_block_role_overlap": False, "status": "PASS"}
    for dataset in ("OAKBAT.cleanStatic", "TEXBAT.cleanStatic"):
        fs = int(inventory["raw_sources"][dataset]["sample_rate_hz"])
        base = int(support[dataset]["raw_start_sample"])
        role_offsets = {
            "train": (0, 60), "guard_1": (60, 70), "calibration": (70, 100),
            "guard_2": (100, 110), "holdout": (110, 140), "guard_3": (140, 150),
            "replay_acquisition_preroll_unscored": (150, 180), "controlled_injection": (180, 210),
        }
        dataset_roles = {}
        for name, (lo_s, hi_s) in role_offsets.items():
            dataset_roles[name] = {
                "raw_start_sample": base + lo_s * fs,
                "raw_end_sample_exclusive": base + hi_s * fs,
                "byte_start": (base + lo_s * fs) * 4,
                "byte_end_exclusive": (base + hi_s * fs) * 4,
                "duration_s": hi_s - lo_s,
                "ten_second_blocks": list(range(lo_s // 10, hi_s // 10)),
            }
        roles[dataset] = dataset_roles
        traces = inventory["trace_sources"][dataset]
        reference_trace = Path(traces[0]["path"])
        injection_lo = dataset_roles["controlled_injection"]["raw_start_sample"]
        targets = [injection_lo + int(round((.5 + 2.4 * i) * fs)) for i in range(12)]
        anchors = [nearest_endpoint(reference_trace, target) for target in targets]
        prns = [int(item["prn"]) for item in traces]
        cases = assign_cases(SEEDS[dataset], dataset, prns, anchors)
        all_cases.extend(cases)
        split_audit["datasets"][dataset] = {
            "sample_rate_hz": fs, "common_support": [base, int(support[dataset]["raw_end_sample_exclusive"])],
            "roles": dataset_roles, "injection_anchors": anchors,
            "clean_scoring_cadence_ms": 500, "role_segments_minimum": 3,
            "scientific_role_duration_s": 180, "additional_unscored_replay_preroll_s": 30,
            "valid_prns": prns, "status": "PASS",
        }

    balances = {}
    for dataset in SEEDS:
        subset = [row for row in all_cases if row["dataset"] == dataset]
        audit = design_balance(subset)
        singles = [row | {"target_prn": row["target_prns"][0]} for row in subset if row["mode"] == "single_prn"]
        fours = [row | {"excluded_prn": next(p for p in split_audit["datasets"][dataset]["valid_prns"]
                                             if p not in row["target_prns"])}
                 for row in subset if row["mode"] == "simultaneous_four_prn"]
        audit["prn_factor_cramers_v_single"] = {
            factor: cramers_v(singles, "target_prn", factor)
            for factor in ("rho_db", "delay_chips", "doppler_hz", "phase_rad")
        }
        audit["excluded_prn_factor_cramers_v_four"] = {
            factor: cramers_v(fours, "excluded_prn", factor)
            for factor in ("rho_db", "delay_chips", "doppler_hz", "phase_rad")
        }
        audit["target_prn_counts"] = {str(p): sum(p in r["target_prns"] for r in singles)
                                      for p in split_audit["datasets"][dataset]["valid_prns"]}
        audit["seed"] = SEEDS[dataset]
        balances[dataset] = audit
        if audit["status"] != "PASS":
            raise SystemExit(f"factor balance failed for {dataset}")

    design_document = {
        "schema": "gnss-doppler-lab.mirage-r1-injection-design.v1",
        "candidate_space_size": 360, "selection": "seeded balanced maximin over factor contingency",
        "configuration_frozen_before_this_MIRAGE_controlled_evaluation": True,
        "cases": all_cases,
    }
    write("injection_design.json", design_document)
    write("injection_design_balance.json", {
        "same_seed_reproducible": True, "different_seed_changes_design": True,
        "datasets": balances, "status": "PASS",
    })
    write("injection_design_sha256.json", {"canonical_json_sha256": canonical_sha(design_document),
                                             "case_count": len(all_cases), "status": "PASS"})
    write("clean_split_audit.json", split_audit)
    write("caf_grid.json", {
        "delay_chips": DELAYS.tolist(), "normalized_doppler_xi": XI.tolist(),
        "integration_scales_ms": [int(1000 * scale) for scale in SCALES],
        "doppler_hz_by_scale": {str(int(scale * 1000)): (XI / scale).tolist() for scale in SCALES},
        "full_epoch_ms": 500, "clean_cadence_ms": 500,
        "subwindows": "causal and ending at the Full epoch endpoint",
        "complex_values_preserved": True, "actual_raw_iq_and_applied_tracker_state_required": True,
    })
    config = {
        "schema": "gnss-doppler-lab.mirage-r1-config.v1", "foundation_commit": FOUNDATION_COMMIT,
        "epsilon_relative": EPSILON_RELATIVE, "minor_numeric_floor_relative": 1e-6,
        "workers_max": 4, "receiver_replays_max_concurrent": 1, "memory_target_gib": 24,
        "temp_storage_limit_gb": 250, "case_retries_max": 2, "heartbeat_seconds": 60,
        "receiver": inventory["receiver"], "raw_sources": inventory["raw_sources"],
        "roles": roles, "attack_data_accessed": False, "scientific_results_computed": False,
    }
    write("config.json", config)
    prereg = {
        "schema": "gnss-doppler-lab.mirage-r1-preregistration.v1",
        "freeze_label": "MIRAGE_R1_PREREGISTRATION_FREEZE",
        "statement": "configuration frozen before this MIRAGE controlled evaluation",
        "results_seen_before_freeze": False, "case_count": 84,
        "score": {
            "minor": "squared relative normalized adjacent complex minors",
            "reference": "clean-train minor-specific median and 1.4826*MAD with relative floor",
            "tail": "clean-train empirical right-tail surprise per scale",
            "node": "maximum over 20/100/500 ms scale surprises",
            "full": "unweighted median over currently valid PRNs; minimum four; no PRN identity feature",
        },
        "thresholds": {"primary": "dataset-specific clean calibration q99", "diagnostic": "q99.5",
                       "target_empirical_fpr": .01, "threshold_rescue_forbidden": True},
        "injection": {"duration_s": 2, "ramp_s": .25, "steady_s": 1.5, "release_s": .25,
                      "evaluation": "steady only", "raw_iq_injection": True,
                      "pinned_receiver_replay": True, "replay_tracker_state_recorrelation": True},
        "controls": ["authentic_only", "common_gain_scaling", "global_phase_rotation",
                     "matched_output_rms_gain", "empirical_raw_iq_awgn", "cn0_degradation_3db",
                     "collapsed_source", "single_source_delay_drift", "single_source_doppler_drift",
                     "one_prn_secondary_path", "prn_drop_add"],
        "ablations": ["E0_total_energy", "E1_magnitude_distortion", "E2_complex_svd_second_energy",
                      "E3_20ms_complex_minor", "E4_100ms_complex_minor", "E5_500ms_complex_minor",
                      "E6_magnitude_minor", "E7_single_prn_node", "Full"],
        "alignment_gate": {"selected_prn_epochs_min": 500, "per_dataset_min": 200,
                           "complex_cosine_median_min": .995, "magnitude_spearman_median_min": .99,
                           "center_error_abs_chips_max": .125, "center_pass_fraction_min": .95},
        "verdicts": ["GO_FOR_FROZEN_STAGE0B_REAL_STATIC_EVALUATION",
                     "NO_GO_MIRAGE_PHYSICAL_HYPOTHESIS", "INCONCLUSIVE_INPUT_OR_SUPPORT",
                     "INCONCLUSIVE_RAW_RECORRELATION_ALIGNMENT", "INCONCLUSIVE_EXECUTION_FAILURE"],
        "go_gates": {
            "clean_holdout_q99_fpr_max": .015, "worst_clean_segment_fpr_max": .05,
            "strong_single_detection_min_each": .75, "strong_four_detection_min_each": .75,
            "paired_effect_ci_lower_min": 0, "abs_score_rms_spearman_max": .3,
            "abs_score_cn0_spearman_max": .3, "single_prn_contribution_max": .5,
            "full_vs_best_scale_max_loss_fraction": .05, "positive_effect_scales_min": 2,
            "valid_case_coverage_min": .8, "clipping_shortcut_forbidden": True,
        },
        "forbidden": ["DS1-DS8", "OS1-OS4", "real_attack_evaluation", "attack_labels", "neural_models",
                      "analytic_triangular_acf_fit", "B0_M1_fusion", "post_result_changes"],
    }
    write("preregistration.json", prereg)
    code_paths = [ROOT / "src/gnss_doppler_lab/mirage_r1.py", ROOT / "tests/test_mirage_r1.py", Path(__file__)]
    write("preregistration_freeze.json", {
        "freeze_label": "MIRAGE_R1_PREREGISTRATION_FREEZE", "generation_commit": head,
        "branch": branch, "configuration_sha256": canonical_sha(config),
        "design_sha256": canonical_sha(design_document),
        "preregistration_sha256": canonical_sha(prereg),
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code_paths},
        "receiver_binary_sha256": sha(Path(inventory["receiver"]["path"])),
        "raw_iq_sha256": {dataset: item["expected_sha256"] for dataset, item in inventory["raw_sources"].items()},
        "scientific_results_computed": False, "injection_executed": False,
        "remote_push_required_before_execution": True, "status": "FROZEN_PENDING_COMMIT_AND_PUSH",
    })
    with (ART / "case_execution_status.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "dataset", "mode", "status", "attempts", "reason"], lineterminator="\n")
        writer.writeheader()
        for row in all_cases:
            writer.writerow({"case_id": row["case_id"], "dataset": row["dataset"], "mode": row["mode"],
                             "status": "PENDING_PREREGISTRATION_PUSH", "attempts": 0, "reason": ""})
    write("runner_phase_evidence.json", {
        "phases": {"extended_support": "PASS", "preregistration": "FROZEN_PENDING_COMMIT_AND_PUSH",
                   "alignment": "NOT_RUN", "clean": "NOT_RUN", "receiver_in_loop": "NOT_RUN",
                   "controls": "NOT_RUN", "verdict": "NOT_RUN"},
        "attack_data_accessed": False, "injection_executed": False,
    })
    (ART / "CURRENT_STATE.md").write_text(
        "# MIRAGE Stage-0A R1 current state\n\n"
        "The extended-support foundation is complete. The R1 configuration, chronological roles, "
        "84-case balanced design, CAF/minor score, controls, and verdict gates are frozen before "
        "this MIRAGE controlled evaluation. Scientific scoring and injection have not run.\n"
    )
    manifest()
    print(f"FROZEN_PENDING_COMMIT_AND_PUSH cases={len(all_cases)} design={canonical_sha(design_document)}")


if __name__ == "__main__":
    main()
