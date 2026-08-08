#!/usr/bin/env python3
"""Independent verifier for the R2a cleanStatic L20 foundation artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.acaf_nf_stage1_r2a_l20_foundation_audit import (  # noqa: E402
    FS_HZ,
    SUPPORT_SAMPLES,
    State,
    clean_only_guard,
    complex_caf_surface,
    r14_l20_aggregate,
)

DEFAULT = ROOT / "artifacts/acaf_nf_stage1_r2a_l20_foundation_audit"
REQUIRED = (
    "README.md", "config.json", "source_binding.json", "r14_r2_equivalence.json",
    "tracker_state_alignment.csv", "state_combination_metrics.csv", "full_cleanstatic_l20_metrics.json",
    "per_prn_metrics.csv", "per_time_segment_metrics.csv", "failed_window_diagnostics.csv",
    "foundation_verdict.json", "execution_validity.json", "verification_report.json", "test_report.txt",
    "plots/l20_doppler_by_prn.png", "plots/l20_doppler_over_time.png", "plots/grid_boundary_diagnostics.png",
)
MAT_FIELDS = (
    "PRN_start_sample_count", "PRN", "carrier_doppler_hz", "code_freq_chips", "aux1",
    "Prompt_I", "Prompt_Q", "CN0_SNV_dB_Hz", "carrier_lock_test",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def recompute_window(raw: np.memmap, mat_path: Path, channel: int, anchor: int) -> str:
    with h5py.File(mat_path, "r") as handle:
        arrays = {name: np.asarray(handle[name]).reshape(-1) for name in MAT_FIELDS}
    surfaces = []
    for cur in range(anchor - 19, anchor + 1):
        state_row = cur - 1
        state = State(
            channel=channel, prn=int(arrays["PRN"][cur]), tracker_row=cur, state_row=state_row,
            raw_start_sample=int(arrays["PRN_start_sample_count"][state_row]),
            code_freq_chips=float(arrays["code_freq_chips"][state_row]),
            carrier_doppler_hz=float(arrays["carrier_doppler_hz"][state_row]),
            aux1=float(arrays["aux1"][state_row]), prompt_i=float(arrays["Prompt_I"][cur]),
            prompt_q=float(arrays["Prompt_Q"][cur]), cn0_db_hz=float(arrays["CN0_SNV_dB_Hz"][cur]),
            carrier_lock=float(arrays["carrier_lock_test"][cur]),
        )
        start = state.raw_start_sample
        packed = np.asarray(raw[2 * start:2 * (start + SUPPORT_SAMPLES)]).reshape(-1, 2)
        iq = packed[:, 0].astype(np.float64) + 1j * packed[:, 1].astype(np.float64)
        surfaces.append(complex_caf_surface(iq, state))
    aggregate = r14_l20_aggregate(surfaces)
    return hashlib.sha256(np.ascontiguousarray(aggregate).view(np.uint8)).hexdigest()


def verify(root: Path) -> dict:
    errors: list[str] = []
    for name in REQUIRED:
        if not (root / name).is_file():
            errors.append(f"missing:{name}")
    if errors:
        return {"schema": "acaf_nf_stage1_r2a_verification.v1", "status": "FAIL", "errors": errors}
    config = load(root / "config.json")
    source = load(root / "source_binding.json")
    metrics = load(root / "full_cleanstatic_l20_metrics.json")
    verdict = load(root / "foundation_verdict.json")
    execution = load(root / "execution_validity.json")
    equivalence = load(root / "r14_r2_equivalence.json")
    clean_only_guard("cleanStatic", [source["raw"]["path"], source["fresh_receiver"]["manifest"]])
    raw_path = Path(source["raw"]["path"])
    manifest_path = Path(source["fresh_receiver"]["manifest"])
    if raw_path.stat().st_size != source["raw"]["size_bytes"]:
        errors.append("raw_size")
    if digest(manifest_path) != source["fresh_receiver"]["manifest_sha256"]:
        errors.append("manifest_sha256")
    r14_manifest_path = Path(source["r14_receiver"]["manifest"])
    if digest(r14_manifest_path) != source["r14_receiver"]["manifest_sha256"]:
        errors.append("r14_manifest_sha256")
    r14_manifest = load(r14_manifest_path)
    if (r14_manifest["receiver"]["executable_sha256"] != source["r14_receiver"]["executable_sha256"]
            or r14_manifest["receiver"]["config_sha256"] != source["r14_receiver"]["config_sha256"]):
        errors.append("r14_receiver_binding")
    if (source["r14_receiver"]["executable_sha256"] == source["fresh_receiver"]["executable_sha256"]
            or source["r14_receiver"]["config_sha256"] == source["fresh_receiver"]["config_sha256"]):
        errors.append("independent_replay_classification")
    manifest = load(manifest_path)
    tracker_dir = manifest_path.parent / "raw"
    mat_by_channel = {}
    for name, expected in manifest["tracking"]["mat_inventory"].items():
        path = tracker_dir / name
        match = re.search(r"(\d+)\.mat$", name)
        if match:
            mat_by_channel[int(match.group(1))] = path
        if digest(path) != expected:
            errors.append(f"mat_sha256:{name}")
    raw = np.memmap(raw_path, dtype="<i2", mode="r")
    recomputed = []
    for sample in config["verification_samples"]:
        actual = recompute_window(raw, mat_by_channel[int(sample["channel"])], int(sample["channel"]), int(sample["anchor_row"]))
        recomputed.append({**sample, "recomputed_aggregate_sha256": actual, "match": actual == sample["aggregate_sha256"]})
        if actual != sample["aggregate_sha256"]:
            errors.append(f"raw_recompute:{sample['channel']}:{sample['anchor_row']}")
    prn = rows(root / "per_prn_metrics.csv")
    total = sum(int(row["l20_windows"]) for row in prn)
    within50 = sum(int(row["l20_windows"]) * float(row["within_50_fraction"]) for row in prn) / total
    within100 = sum(int(row["l20_windows"]) * float(row["within_100_fraction"]) for row in prn) / total
    if total != metrics["l20_windows"]:
        errors.append("per_prn_window_total")
    if abs(within50 - metrics["within_50_fraction"]) > 1e-12:
        errors.append("per_prn_within50")
    if abs(within100 - metrics["within_100_fraction"]) > 1e-12:
        errors.append("per_prn_within100")
    expected_gates = {
        "sample_windows_ge_1000": metrics["l20_windows"] >= 1000,
        "prn_channels_ge_8": metrics["evaluated_prn_channels"] >= 8,
        "prompt_pooled_spearman_ge_0_999": metrics["prompt_reproduction"]["pooled_spearman"] >= .999,
        "prompt_p99_relative_error_le_0_01": metrics["prompt_reproduction"]["p99_relative_error"] <= .01,
        "delay_within_0_125_ge_0_95": metrics["delay_within_0_125_fraction"] >= .95,
        "l20_doppler_within_50_ge_0_95": metrics["within_50_fraction"] >= .95,
        "grid_boundary_le_0_01": metrics["overall_grid_boundary_fraction"] <= .01,
        "raw_provenance_authenticated": source["status"] == "PASS",
        "causal_alignment_verified": equivalence["status"] == "PASS",
    }
    if expected_gates != verdict["gates"]:
        errors.append("gate_recompute")
    expected_verdict = "FOUNDATION_VALID" if all(expected_gates.values()) else (
        "FOUNDATION_INVALID" if metrics["l20_windows"] >= 1000 and metrics["evaluated_prn_channels"] >= 8 else "FOUNDATION_INCONCLUSIVE"
    )
    if verdict["verdict"] != expected_verdict:
        errors.append("verdict_recompute")
    if execution["attack_iq_bytes_read"] != 0 or execution["attack_iq_files_opened"] != []:
        errors.append("attack_access")
    if execution["model_training_executed"] or execution["threshold_fitting_executed"] or execution["thresholds_changed"]:
        errors.append("fit_or_threshold")
    if verdict["deterministic_replay_equivalence"]["gate_name_retained"] != "r14_common_reproduced_1e_6":
        errors.append("r14_gate_removed")
    return {
        "schema": "acaf_nf_stage1_r2a_verification.v1", "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)), "raw_recomputed_windows": recomputed,
        "recomputed": {"l20_windows": total, "within_50_fraction": within50,
                       "within_100_fraction": within100, "gates": expected_gates, "verdict": expected_verdict},
        "artifact_sha256": {
            name: digest(root / name) for name in REQUIRED if name != "verification_report.json"
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT)
    args = parser.parse_args()
    report = verify(args.artifact)
    (args.artifact / "verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
