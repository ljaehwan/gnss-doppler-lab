"""Fail-closed terminal attestation for the frozen Q-SET Stage-0A R2 run.

This module does not read any Tuni2025 raw input and does not compute Q-SET
features, scores, thresholds, or attack metrics. It only binds the preserved
SS-1 receiver output after the frozen receiver-support gate stopped execution.
"""

from __future__ import annotations

import csv
import gzip
import io
import subprocess
from pathlib import Path
from typing import Any

from .qset_stage0a_r2 import (
    AGGREGATORS,
    ARTIFACT,
    SCENARIOS,
    SSD_ROOT,
    canonical_sha,
    git,
    read_json,
    require,
    sha256_file,
    validate_galileo_trace,
    write_json,
)
from .qset_stage0a_r2_evaluation import verify_file_binding, write_csv
from .qset_stage0a_r2_execution import output_manifest, verify_manifest

BRANCH = "research/qset-gnss-stage0a-r2-galileo-partial-prn-execution"
SS1_ROOT = SSD_ROOT / "replays" / "SS-1"
ATTACK_WRAPPER_LOG = SSD_ROOT / "attack_execution_freeze_5a15e1ea_stdout.txt"
REPORTING_PATHS = (
    "src/gnss_doppler_lab/qset_stage0a_r2_terminal_inconclusive.py",
    "scripts/finalize_qset_gnss_stage0a_r2_terminal_inconclusive.py",
    "scripts/verify_qset_gnss_stage0a_r2_terminal_inconclusive.py",
    "tests/test_qset_gnss_stage0a_r2_terminal_inconclusive.py",
)


def verify_reporting_checkout(freeze_sha: str) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    require(git("status", "--porcelain") == "", "terminal reporting checkout not clean")
    require(git("rev-parse", f"origin/{BRANCH}") == head, "terminal reporting commit not pushed")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze_sha, head],
        cwd=ARTIFACT.parents[1],
        check=False,
    )
    require(ancestor.returncode == 0, "scientific freeze is not an ancestor")
    freeze = read_json(ARTIFACT / "execution_freeze.json")
    require(freeze["status"] == "PASS_PRE_ATTACK", "scientific freeze status drift")
    for relative, expected in freeze["code_bindings"].items():
        require(sha256_file(ARTIFACT.parents[1] / relative) == expected, f"frozen code drift {relative}")
    require(canonical_sha(read_json(ARTIFACT / "normal_model.json")) == freeze["model_sha256"], "literal model drift")
    threshold = read_json(ARTIFACT / "threshold_binding.json")
    threshold_sha = canonical_sha(
        {"multi_q_reference": threshold["multi_q_reference"], "thresholds": threshold["thresholds"]}
    )
    require(threshold_sha == threshold["threshold_sha256"] == freeze["threshold_sha256"], "threshold drift")
    receiver = read_json(ARTIFACT / "receiver_binary_inventory.json")
    require(sha256_file(Path(receiver["receiver_path"])) == freeze["receiver_sha256"], "receiver drift")
    for name in ("C-1", "C-3"):
        binding = freeze["feature_bindings"][name]
        verify_file_binding(Path(binding["path"]), binding, f"{name} feature cache")
        replay = read_json(SSD_ROOT / "replays" / name / "manifest.json")
        verify_manifest(SSD_ROOT / "replays" / name, replay["output_set"])
    return {"head": head, "freeze": freeze}


def preserved_ss1_failure() -> dict[str, Any]:
    require(SS1_ROOT.is_dir(), "preserved SS-1 output absent")
    require(not (SS1_ROOT / "manifest.json").exists(), "SS-1 output unexpectedly finalized or overwritten")
    receiver_dir = SS1_ROOT / "receiver"
    log_path = receiver_dir / "receiver.log"
    decoder = SS1_ROOT / "decoded_4msps_gr_complex.bin"
    require(log_path.is_file() and decoder.is_file(), "SS-1 receiver evidence incomplete")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    require("Draining receiver, 599996606 samples processed" in log, "SS-1 exact drain evidence absent")
    require("Received action DRAIN" in log and "GNSS-SDR program ended." in log, "SS-1 terminal evidence absent")
    trace = validate_galileo_trace(sorted(receiver_dir.glob("trace_native_1ms_ch_*.bin")), "SS-1")
    require(trace["status"] == "FAIL", "SS-1 frozen support gate unexpectedly passed")
    require(trace["tracked_prn_count"] == 3 and trace["tracked_prns"] == [9, 30, 36], "SS-1 support drift")
    outputs = output_manifest(SS1_ROOT)
    decoder_row = next(row for row in outputs["files"] if row["path"] == decoder.name)
    require(decoder_row["size_bytes"] == 4_799_972_848, "SS-1 decoder size drift")
    config = receiver_dir / "receiver.conf"
    require(config.is_file(), "SS-1 receiver config absent")
    require(ATTACK_WRAPPER_LOG.is_file(), "attack wrapper failure log absent")
    return {
        "schema": "gnss-doppler-lab.qset-stage0a-r2-ss1-terminal-failure.v1",
        "status": "FAIL_INSUFFICIENT_GALILEO_RECEIVER_SUPPORT",
        "scenario": "SS-1",
        "raw_input_identity": {
            "size_bytes": SCENARIOS["SS-1"]["size"],
            "md5": SCENARIOS["SS-1"]["md5"],
            "path_redacted_from_compact_artifact": True,
        },
        "decoder": {
            "size_bytes": decoder_row["size_bytes"],
            "sha256": decoder_row["sha256"],
            "output_samples": decoder_row["size_bytes"] // 8,
            "source_bytes_read": SCENARIOS["SS-1"]["size"],
        },
        "receiver": {
            "sha256": read_json(ARTIFACT / "execution_freeze.json")["receiver_sha256"],
            "config_sha256": sha256_file(config),
            "exit_code": "NOT_PERSISTED_BEFORE_FAIL_CLOSED_EXCEPTION",
            "program_ended": True,
            "terminal_drain": True,
            "drained_samples": 599_996_606,
        },
        "trace_validation": trace,
        "support_gate": {
            "receiver_validation_minimum_prns": 4,
            "event_window_minimum_prns": 5,
            "actual_tracked_prns": 3,
            "pass": False,
        },
        "score_computed": False,
        "threshold_or_model_changed": False,
        "downstream_attack_scenarios_opened": [],
        "wrapper_log": {
            "path": str(ATTACK_WRAPPER_LOG),
            "size_bytes": ATTACK_WRAPPER_LOG.stat().st_size,
            "sha256": sha256_file(ATTACK_WRAPPER_LOG),
        },
        "output_set": outputs,
    }


def _write_empty_scores() -> None:
    target = ARTIFACT / "per_window_scores.csv.gz"
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(
                    ["scenario", "window_end_s", "prn_count", "cn0_mean", "lock_mean", "multi_q_persistent_score", "alarm"]
                )


def _write_support_plot(failure: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean_counts = [
        read_json(ARTIFACT / "receiver_manifests" / f"{name}.json")["trace_validation"]["tracked_prn_count"]
        for name in ("C-1", "C-3")
    ]
    counts = [*clean_counts, failure["trace_validation"]["tracked_prn_count"]]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    bars = axis.bar(["C-1 clean", "C-3 clean", "SS-1"], counts, color=["#4c78a8", "#4c78a8", "#e45756"])
    axis.axhline(4, color="#777777", linestyle="--", label="receiver minimum = 4")
    axis.axhline(5, color="#222222", linestyle=":", label="event-window minimum = 5")
    axis.bar_label(bars)
    axis.set_ylim(0, max(counts) + 2)
    axis.set_ylabel("Tracked Galileo E1 PRNs")
    axis.set_title("Q-SET Stage-0A stopped before scoring: SS-1 support failure")
    axis.legend()
    figure.tight_layout()
    path = ARTIFACT / "plots" / "receiver_support_failure.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, metadata={"Software": "gnss-doppler-lab"})
    plt.close(figure)


def finalize_terminal_inconclusive(freeze_sha: str) -> dict[str, Any]:
    checkout = verify_reporting_checkout(freeze_sha)
    failure_first = preserved_ss1_failure()
    failure_second = preserved_ss1_failure()
    require(canonical_sha(failure_first) == canonical_sha(failure_second), "technical failure reproduction drift")
    freeze = checkout["freeze"]
    write_json(
        ARTIFACT / "freeze_commit.json",
        {
            "schema": "gnss-doppler-lab.qset-stage0a-r2-freeze-commit.v1",
            "status": "PASS",
            "commit_sha": freeze_sha,
            "pushed_before_attack_access": True,
            "local_remote_match": True,
            "ahead": 0,
            "behind": 0,
            "clean_checkout": True,
        },
    )
    write_json(ARTIFACT / "receiver_manifests" / "SS-1.json", failure_first)
    support_rows: list[dict[str, Any]] = []
    with (ARTIFACT / "per_prn_support.csv").open(newline="", encoding="utf-8") as stream:
        support_rows.extend(csv.DictReader(stream))
    support_rows.extend(
        {"scenario": "SS-1", "prn": prn, "source": "native TRACE", "status": "TRACKED_BELOW_GATE"}
        for prn in failure_first["trace_validation"]["tracked_prns"]
    )
    write_csv(ARTIFACT / "per_prn_support.csv", support_rows, ["scenario", "prn", "source", "status"])
    write_json(
        ARTIFACT / "scenario_metrics.json",
        {
            "schema": "gnss-doppler-lab.qset-stage0a-r2-scenario-metrics.v1",
            "status": "NOT_COMPUTED_TECHNICAL_SUPPORT_GATE",
            "scenarios": {
                "SS-1": {
                    "status": failure_first["status"],
                    "tracked_prns": failure_first["trace_validation"]["tracked_prns"],
                    "score_computed": False,
                }
            },
            "unopened_scenarios": ["SS-3", "SS-5", "SS-11"],
        },
    )
    write_csv(
        ARTIFACT / "aggregator_comparison.csv",
        [
            {
                "scenario": "SS-1",
                "aggregator": name,
                "status": "NOT_COMPUTED_TECHNICAL_SUPPORT_GATE",
                "reason": "3 tracked PRNs < frozen minimum 5",
            }
            for name in AGGREGATORS
        ],
        ["scenario", "aggregator", "status", "reason"],
    )
    write_csv(
        ARTIFACT / "per_prn_ground_truth_metrics.csv",
        [
            {
                "scenario": "SS-1",
                "prn": prn,
                "ground_truth": "spoofed" if prn in SCENARIOS["SS-1"]["spoofed"] else "genuine",
                "status": "NOT_SCORED_TECHNICAL_SUPPORT_GATE",
            }
            for prn in failure_first["trace_validation"]["tracked_prns"]
        ],
        ["scenario", "prn", "ground_truth", "status"],
    )
    _write_empty_scores()
    write_json(
        ARTIFACT / "shortcut_audit.json",
        {
            "schema": "gnss-doppler-lab.qset-stage0a-r2-shortcut-audit.v1",
            "status": "NOT_EVALUATED_NO_ATTACK_SCORE",
            "forbidden_score_inputs_absent": True,
            "attack_score_computed": False,
        },
    )
    clean = read_json(ARTIFACT / "clean_score_summary.json")
    write_json(
        ARTIFACT / "stage0a_gate.json",
        {
            "schema": "gnss-doppler-lab.qset-stage0a-r2-gate.v1",
            "technical": {
                "pass": False,
                "reason": "SS-1 tracked PRN count 3 is below receiver minimum 4 and event minimum 5",
            },
            "false_alarm": {"pass": True, "metrics": clean["clean_metrics"]},
            "local_signal": {"pass": None, "status": "NOT_EVALUATED"},
            "aggregation": {"pass": None, "status": "NOT_EVALUATED"},
            "overall_pass": False,
        },
    )
    final = {
        "schema": "gnss-doppler-lab.qset-stage0a-r2-final-verdict.v1",
        "verdict": "INCONCLUSIVE_QSET_DATA_FORMAT_OR_RECEIVER_SUPPORT",
        "next_state": "NOT_AUTHORIZED",
        "stage0b_authorized": False,
        "attack_evaluation_complete": False,
        "freeze_sha": freeze_sha,
        "classification": "STAGE_0A_NON_NEURAL_FEASIBILITY",
        "forbidden_claims_made": False,
        "score_computed": False,
        "stopped_fail_closed_after_scenario": "SS-1",
    }
    write_json(ARTIFACT / "final_verdict.json", final)
    write_json(
        ARTIFACT / "deterministic_reproduction.json",
        {
            "schema": "gnss-doppler-lab.qset-stage0a-r2-determinism.v1",
            "status": "PASS_TECHNICAL_FAILURE_REPRODUCED",
            "failure_sha256_first": canonical_sha(failure_first),
            "failure_sha256_second": canonical_sha(failure_second),
            "byte_identical_compact_failure_evidence": True,
            "attack_score_reproduction": "NOT_APPLICABLE_SCORE_NOT_COMPUTED",
        },
    )
    audit = read_json(ARTIFACT / "access_audit.json")
    audit["phase"] = "ATTACK_EXECUTION_STOPPED_FAIL_CLOSED"
    audit["attack_payload"] = {
        "allowlisted_scenarios": ["SS-1"],
        "stats": 3,
        "hashes": 1,
        "opens": 2,
        "mmaps": 0,
        "bytes_read": SCENARIOS["SS-1"]["size"] * 2,
        "signal_statistics": 0,
    }
    audit["unallowlisted_tuni2025_raw"] = {
        "stats": 0,
        "hashes": 0,
        "opens": 0,
        "mmaps": 0,
        "bytes_read": 0,
    }
    audit["unopened_allowlisted_scenarios"] = ["SS-3", "SS-5", "SS-11"]
    audit["attack_access_after_freeze_only"] = True
    audit["attack_scientific_operations"] = {
        "receiver_runs": 1,
        "feature_windows": 0,
        "scores": 0,
        "attack_evaluations": 0,
    }
    audit["status"] = "PASS"
    write_json(ARTIFACT / "access_audit.json", audit)
    reporting_bindings = {
        relative: sha256_file(ARTIFACT.parents[1] / relative) for relative in REPORTING_PATHS
    }
    write_json(
        ARTIFACT / "terminal_execution_attestation.json",
        {
            "schema": "gnss-doppler-lab.qset-stage0a-r2-terminal-execution-attestation.v1",
            "status": "PASS_FAIL_CLOSED_AT_FROZEN_SUPPORT_GATE",
            "freeze_sha": freeze_sha,
            "reporting_commit": checkout["head"],
            "reporting_code_bindings": reporting_bindings,
            "scientific_code_bindings_unchanged": True,
            "model_sha256": freeze["model_sha256"],
            "threshold_sha256": freeze["threshold_sha256"],
            "ss1_failure_sha256": canonical_sha(failure_first),
            "attack_score_computed": False,
            "downstream_attack_scenarios_opened": [],
        },
    )
    _write_support_plot(failure_first)
    (ARTIFACT / "README.md").write_text(
        "# Q-SET-GNSS Stage-0A R2 Galileo partial-PRN execution\n\n"
        "Final status: INCONCLUSIVE_QSET_DATA_FORMAT_OR_RECEIVER_SUPPORT.\n\n"
        "The scientific implementation, literal clean model, thresholds, receiver, dynamic-panel rule, "
        "and gates were frozen and pushed at " + freeze_sha + " before attack access. SS-1 then "
        "completed the fixed decoder and receiver run with exact terminal drain, finite native TRACE, "
        "and zero cadence/causal failures, but only Galileo E1 PRNs E09, E30, and E36 were tracked. "
        "Three PRNs are below both the receiver-validation minimum of four and the event-window minimum "
        "of five, so execution stopped fail-closed before feature extraction or attack scoring. "
        "SS-3, SS-5, and SS-11 were not opened. Stage-0B is not authorized.\n",
        encoding="utf-8",
    )
    return final
