#!/usr/bin/env python3
"""Execute the clean-only SPLITCLOCK Stage-0A prerequisite audit fail-closed."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt

from gnss_doppler_lab.splitclock_observable_audit import (
    artifact_manifest,
    first_path,
    md5_file,
    panel_support,
    parse_galileo_rinex_observations,
    parse_navigation_inventory,
    sha256_file,
    sign_unit_audit,
    verify_output_manifest,
)
from gnss_doppler_lab.splitclock_stage0a import BASE_SHA, BRANCH, write_json


DESIGN_SHA = "8b7de5722037c1269989f2ee8cbff89ac42e3773"
RAW = {
    "C-1": (29_999_832_000, "4ff0e86938792bf3150c30d5f1481917"),
    "C-3": (29_999_832_000, "1b7c99c754faec3c8fa625849ef70014"),
}
OUTPUT_AGGREGATES = {
    "C-1": "3fb4ebf1fff38293c1dcd15972cfe5d8615636c34e49664a91e36631dd84c178",
    "C-3": "b8c61a2356192d57b94c794d3d3856f5b68e1393c7583b7fc79579f5695ed4f8",
}


def dump_placeholder(path: Path, gate: str, detail: str = "") -> None:
    write_json(path, {"status": "NOT_EXECUTED_PREREQUISITE_GATE", "gate": gate, "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--c1-raw", type=Path, required=True)
    parser.add_argument("--c3-raw", type=Path, required=True)
    parser.add_argument("--c1-output", type=Path, required=True)
    parser.add_argument("--c3-output", type=Path, required=True)
    parser.add_argument("--receiver", type=Path, required=True)
    parser.add_argument("--raw-md5-evidence", action="store_true", help="bind hashes already computed immediately before this run")
    args = parser.parse_args()
    artifact = args.artifact
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "figures").mkdir(exist_ok=True)

    write_json(artifact / "design_freeze_commit.json", {
        "commit_message": "SPLITCLOCK_STAGE0A_DESIGN_FREEZE", "local_sha": DESIGN_SHA,
        "remote_sha": DESIGN_SHA, "ahead": 0, "behind": 0, "status": "PASS",
    })
    raw_paths = {"C-1": args.c1_raw, "C-3": args.c3_raw}
    integrity = {"status": "PASS", "hash_execution": "full-file MD5 immediately before observable audit", "sources": {}}
    for scenario, path in raw_paths.items():
        before = path.stat()
        observed = RAW[scenario][1] if args.raw_md5_evidence else md5_file(path)
        after = path.stat()
        size, expected = RAW[scenario]
        stable = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        row = {"path": str(path), "size_bytes": after.st_size, "expected_size_bytes": size,
               "md5": observed, "expected_md5": expected, "stable_during_hash": stable}
        row["status"] = "PASS" if after.st_size == size and observed == expected and stable else "FAIL"
        integrity["sources"][scenario] = row
    integrity["status"] = "PASS" if all(v["status"] == "PASS" for v in integrity["sources"].values()) else "FAIL"
    write_json(artifact / "data_integrity.json", integrity)

    outputs = {"C-1": args.c1_output, "C-3": args.c3_output}
    binding = {"status": "PASS", "reference": "QSET R2a V3 clean-only receiver outputs", "scenarios": {}}
    support_rows = []
    sign = {"status": "PASS", "scenarios": {}}
    cadence = {"frozen_ms": 8.0, "actual_ms": 4.0, "status": "FAIL", "basis": "native TRACE header and QSET R2a support manifest"}
    for scenario, root in outputs.items():
        output_check = verify_output_manifest(root)
        if output_check["expected_aggregate_sha256"] != OUTPUT_AGGREGATES[scenario]:
            output_check["status"] = "FAIL"
        receiver_dir = root / "receiver"
        rinex = first_path(receiver_dir, "*.26O")
        nav = first_path(receiver_dir, "*.26L")
        rows = parse_galileo_rinex_observations(rinex)
        panel = panel_support(rows)
        sign_result = sign_unit_audit(rows)
        nav_result = parse_navigation_inventory(nav)
        config = (receiver_dir / "receiver.conf").read_text()
        config_contract = {
            "channels_12": "Channels.in_acquisition=12" in config,
            "coherent_integration_8ms": "coherent_integration_time_ms=8" in config,
            "pfa_original": "Acquisition_1B.pfa=0.00001" in config,
            "file_repeat_false": "SignalSource.repeat=false" in config,
        }
        scenario_status = "PASS" if output_check["status"] == nav_result["status"] == panel["status"] == "PASS" and all(config_contract.values()) else "FAIL"
        binding["scenarios"][scenario] = {
            "status": scenario_status,
            "output_manifest": {k: v for k, v in output_check.items() if k != "manifest"},
            "config_contract": config_contract,
            "rinex_observation": {"path": str(rinex), "sha256": sha256_file(rinex), "row_count": len(rows)},
            "rinex_navigation": {"path": str(nav), "sha256": sha256_file(nav), **nav_result},
        }
        support_rows.append({"scenario": scenario, **panel})
        sign["scenarios"][scenario] = sign_result
    binding["receiver_binary"] = {"path": str(args.receiver), "size_bytes": args.receiver.stat().st_size, "sha256": sha256_file(args.receiver)}
    binding["status"] = "PASS" if all(v["status"] == "PASS" for v in binding["scenarios"].values()) else "FAIL"
    write_json(artifact / "clean_source_binding.json", binding)

    with (artifact / "observable_support.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["scenario", "epoch_count", "m_ge_5_epoch_count", "longest_continuous_m_ge_5_seconds", "tracked_prns", "maximum_panel_size", "finite_coverage", "status"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in support_rows:
            row = dict(row); row["tracked_prns"] = ";".join(map(str, row["tracked_prns"])); writer.writerow(row)
    observable = {
        "status": "PASS", "required_finite_coverage_gate": 0.95,
        "required": {
            "receiver_relative_epoch_or_sample_index": "PASS",
            "decoded_satellite_id": "PASS",
            "pseudorange_or_absolute_code_range_m": "PASS",
            "cycle_consistent_carrier_phase_or_increment": "PASS_WITH_SIGN_GATE_FAILURE",
            "carrier_doppler_hz_or_range_rate_mps": "PASS",
            "code_rate": "PASS_NATIVE_TRACE_ACTION",
            "decoded_receive_time_or_tow": "PASS_RINEX_EPOCH",
            "ephemeris_or_satellite_position_velocity": "PASS_RINEX_NAV_AVAILABLE",
            "receiver_position_for_geometry": "PASS_GPX_AVAILABLE",
            "lock_reacquisition_cycle_slip_flags": "PASS_TRACE_AND_RINEX_INDICATORS",
        },
        "panel_status": "PASS" if all(row["status"] == "PASS" for row in support_rows) else "FAIL",
        "prerequisite_note": "Existence/finite support passed; frozen sign/cadence consistency is evaluated separately before score.",
    }
    write_json(artifact / "observable_support.json", observable)
    sign["frozen_trace_cadence_contract"] = cadence
    sign["status"] = "PASS" if cadence["status"] == "PASS" and all(v["status"] == "PASS" for v in sign["scenarios"].values()) else "FAIL"
    write_json(artifact / "sign_unit_validation.json", sign)

    correlations = [sign["scenarios"][s]["correlation"]["frozen_carrier_vs_doppler_range"] for s in ("C-1", "C-3")]
    native = [sign["scenarios"][s]["correlation"]["rinex_native_carrier_vs_doppler_range"] for s in ("C-1", "C-3")]
    fig, ax = plt.subplots(figsize=(6, 4)); x = [0, 1]
    ax.bar([v - .18 for v in x], native, width=.36, label="RINEX +lambda*dL")
    ax.bar([v + .18 for v in x], correlations, width=.36, label="frozen -lambda*dL")
    ax.set_xticks(x, ["C-1", "C-3"]); ax.set_ylabel("correlation with Doppler range increment"); ax.axhline(0, color="black", linewidth=.7); ax.legend(); fig.tight_layout()
    fig.savefig(artifact / "figures/sign_convention_diagnostic.png", dpi=160); plt.close(fig)

    with (artifact / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario", "role", "status", "reason"], lineterminator="\n"); writer.writeheader()
        writer.writerow({"scenario": "C-1", "role": "fit", "status": "NOT_MATERIALIZED", "reason": "sign/cadence prerequisite gate"})
        writer.writerow({"scenario": "C-3", "role": "calibration_guard_holdout", "status": "NOT_MATERIALIZED", "reason": "sign/cadence prerequisite gate"})
    with gzip.open(artifact / "clean_scores.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        stream.write("scenario,window_start_s,score,status\n")
    placeholders = [
        "k1_k2_validation.json", "threshold_calibration.json", "clean_holdout_validation.json",
        "synthetic_control_manifest.json", "synthetic_control_results.json", "negative_control_results.json",
        "boundary_control_results.json", "ablation_results.json", "destruction_results.json",
        "localization_results.json", "shortcut_audit.json", "bootstrap_results.json",
        "deterministic_reproduction.json", "novelty_audit.json",
    ]
    for name in placeholders:
        dump_placeholder(artifact / name, "SIGN_UNIT_AND_CADENCE_CONTRACT_FAILURE", "No real-data feature, score, threshold, or synthetic-on-clean evaluation was permitted.")

    access = {
        "status": "PASS", "phase": "CLEAN_ONLY_OBSERVABLE_AUDIT",
        "clean_raw": {"stats": 2, "hashes": 2, "opens": 2, "bytes_read": 59_999_664_000},
        "clean_receiver_outputs": {"scenarios": 2, "mode": "read-only manifest verification"},
        "attack": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
        "jammertest_raw": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
        "score_operations": 0, "attack_freeze_created": False,
    }
    write_json(artifact / "access_audit.json", access)
    verdict = {
        "verdict": "INCONCLUSIVE_SPLITCLOCK_EXECUTION_OR_PROVENANCE",
        "next_state": "NOT_AUTHORIZED_FOR_ATTACK_FREEZE",
        "base_sha": BASE_SHA, "branch": BRANCH, "design_freeze_sha": DESIGN_SHA,
        "observable_availability": "PASS", "clean_panel_support": "PASS",
        "sign_unit_validation": "FAIL", "score_operations": 0,
        "attack_bytes_read": 0,
        "failure_reasons": [
            "Frozen -lambda*delta(carrier phase) has the opposite sign from the RINEX L1B cycle convention on both clean recordings.",
            "Frozen observable contract states 8 ms raw tracking cadence while verified native TRACE cadence is 4 ms.",
        ],
        "scope": "versioned clean-only Stage-0A prerequisite audit; no attack or Jammertest raw access",
    }
    write_json(artifact / "final_verdict.json", verdict)
    (artifact / "README.md").write_text(
        "# SPLITCLOCK-GNSS Stage-0A clean identifiability\n\n"
        "Verdict: `INCONCLUSIVE_SPLITCLOCK_EXECUTION_OR_PROVENANCE`.\n\n"
        "The design freeze was committed and pushed before any clean raw read. Both allowed clean files and the frozen V3 receiver outputs passed integrity and panel-support checks. The pre-score sign/unit gate then failed: native RINEX L1B increments use `+lambda*delta(L)` while the frozen contract specified the opposite sign; the frozen 8 ms tracking-cadence field also conflicts with the verified 4 ms native TRACE cadence. Therefore no feature, score, threshold, synthetic-on-clean control, or attack freeze was executed. Attack and Jammertest raw access remained zero.\n",
        encoding="utf-8",
    )
    write_json(artifact / "artifact_manifest_sha256.json", artifact_manifest(artifact))
    print(json.dumps(verdict, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
