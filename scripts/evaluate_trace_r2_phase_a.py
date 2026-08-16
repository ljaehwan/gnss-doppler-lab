#!/usr/bin/env python3
"""Aggregate the preregistered TRACE-R2 Phase-A smoke gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_native_1ms import read_header, read_records, sha256_file, validate_dump_files

ARTIFACT = ROOT / "artifacts/trace_stage0_r2_native_1ms_dump"
SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2-native-1ms-dump")
SCENARIOS = {
    "TEXBAT.cleanStatic": {
        "slug": "texbat_cleanstatic",
        "raw_start_sample": 0,
        "raw_end_sample": 1_125_000_000,
        "windows": {"stable_normal_5_to_10_s": (35.0, 45.0)},
    },
    "TEXBAT.DS3": {
        "slug": "texbat_ds3",
        "raw_start_sample": 2_250_000_000,
        "raw_end_sample": 3_375_000_000,
        "windows": {"pre_onset": (113.9, 118.9), "post_onset": (118.9, 128.9)},
    },
    "OAKBAT.OS3": {
        "slug": "oakbat_os3",
        "raw_start_sample": 450_000_000,
        "raw_end_sample": 675_000_000,
        "windows": {"pre_onset": (115.0, 120.0), "post_onset": (120.0, 130.0)},
    },
}


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def support_in_window(paths: list[Path], start_s: float, end_s: float) -> dict[str, object]:
    epochs: dict[int, set[int]] = {}
    rows = 0
    for path in paths:
        _, records = read_records(path)
        quality = (
            (records["valid_tracking"] == 1)
            & (records["valid_lock"] == 1)
            & (records["receiver_state"] == 4)
            & (records["receiver_timestamp_s"] >= start_s)
            & (records["receiver_timestamp_s"] < end_s)
        )
        rows += int(quality.sum())
        epoch = np.rint(records["receiver_timestamp_s"][quality] * 1000.0).astype(np.int64)
        for value, prn in zip(epoch, records["prn"][quality], strict=True):
            epochs.setdefault(int(value), set()).add(int(prn))
    maximum = max((len(prns) for prns in epochs.values()), default=0)
    epochs_ge4 = sum(len(prns) >= 4 for prns in epochs.values())
    return {
        "start_s": start_s,
        "end_s": end_s,
        "valid_locked_state4_rows": rows,
        "maximum_prns_same_rounded_ms_epoch": maximum,
        "epochs_with_at_least_4_prns": epochs_ge4,
        "status": "PASS" if maximum >= 4 and epochs_ge4 > 0 else "FAIL",
    }


def deterministic_hashes(first: Path, second: Path) -> dict[str, object]:
    first_manifest = json.loads((first / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    first_hashes = {Path(item["path"]).name: item["sha256"] for item in first_manifest["dump_files"]}
    second_hashes = {Path(item["path"]).name: item["sha256"] for item in second_manifest["dump_files"]}
    return {
        "first": first_hashes,
        "second": second_hashes,
        "exact_file_hash_match": first_hashes == second_hashes,
        "status": "PASS" if first_hashes == second_hashes else "FAIL",
    }


def main() -> int:
    failures: set[str] = set()
    scenario_results: dict[str, object] = {}
    schema_signatures: set[tuple[object, ...]] = set()
    action_totals = {
        "causal_pair_count": 0,
        "causal_sequence_mismatch_count": 0,
        "causal_value_mismatch_count": 0,
        "consume_span_mismatch_count": 0,
    }
    raw_path = ARTIFACT / "raw_source_binding.json"
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {"status": "FAIL", "datasets": {}}
    for name, spec in SCENARIOS.items():
        directory = SSD / "dumps/phase_a" / spec["slug"] / "rep1"
        paths = sorted(directory.glob("trace_native_1ms_ch_*.bin"))
        try:
            validation = validate_dump_files(paths, expected_scenario_id=name, minimum_prns=4)
        except (OSError, ValueError) as error:
            validation = {
                "status": "FAIL",
                "failure_labels": ["NATIVE_1MS_DUMP_IMPLEMENTATION_INVALID"],
                "error": str(error),
            }
        failures.update(validation.get("failure_labels", []))
        for key in action_totals:
            action_totals[key] += int(validation.get(key, 0))
        windows = {
            window: support_in_window(paths, *bounds) if paths else {"status": "FAIL", "start_s": bounds[0], "end_s": bounds[1]}
            for window, bounds in spec["windows"].items()
        }
        if any(payload["status"] != "PASS" for payload in windows.values()):
            failures.add("INSUFFICIENT_MULTI_PRN_SUPPORT")
        for path in paths:
            header = read_header(path)
            schema_signatures.add(
                (
                    header.schema_version,
                    header.header_size,
                    header.record_size,
                    header.sample_rate_hz,
                    header.tap_spacing_chips,
                    header.coherent_integration_s,
                    header.receiver_source_base_commit,
                    header.tap_offsets_chips,
                )
            )
        manifest = json.loads((directory / "manifest.json").read_text()) if (directory / "manifest.json").exists() else None
        manifest_checks: dict[str, object] = {"status": "FAIL"}
        if manifest is not None:
            config_path = Path(manifest["receiver_config_path"])
            recorded_dumps = {
                Path(item["path"]).name: item["sha256"] for item in manifest.get("dump_files", [])
            }
            observed_dumps = {
                Path(item["path"]).name: item["sha256"]
                for item in validation.get("file_summaries", [])
            }
            bound_raw = raw.get("datasets", {}).get(name, {})
            checks = {
                "receiver_exit_zero": manifest.get("exit_code") == 0,
                "receiver_stable": manifest.get("receiver_stable_during_run") is True,
                "raw_stable": manifest.get("raw_iq_stable_during_run") is True,
                "raw_sha_matches_authenticated_audit": (
                    manifest.get("raw_iq", {}).get("sha256") == bound_raw.get("fresh_sha256")
                    and bound_raw.get("status") == "PASS"
                ),
                "raw_path_matches_authenticated_audit": (
                    manifest.get("raw_iq", {}).get("path") == bound_raw.get("raw_iq_path")
                ),
                "raw_sample_range_exact": manifest.get("raw_sample_range", {}).get("start_inclusive") == spec["raw_start_sample"]
                and manifest.get("raw_sample_range", {}).get("end_exclusive") == spec["raw_end_sample"],
                "config_hash_matches": config_path.is_file()
                and sha256_file(config_path) == manifest.get("receiver_config_sha256"),
                "dump_hashes_match": bool(recorded_dumps) and recorded_dumps == observed_dumps,
            }
            manifest_checks = {**checks, "status": "PASS" if all(checks.values()) else "FAIL"}
        if manifest_checks["status"] != "PASS":
            failures.add("RAW_SOURCE_BINDING_FAILED")
        scenario_results[name] = {
            "directory": str(directory),
            "manifest": manifest,
            "manifest_checks": manifest_checks,
            "validation": validation,
            "support_windows": windows,
        }

    # Sample rate is dataset-specific; compare every other schema component.
    schema_without_fs = {(v, h, r, s, c, commit, offsets) for v, h, r, _fs, s, c, commit, offsets in schema_signatures}
    same_schema = len(schema_without_fs) == 1 and len(schema_signatures) >= 2
    if not same_schema:
        failures.add("NATIVE_1MS_DUMP_IMPLEMENTATION_INVALID")

    phase_raw_pass = all(raw.get("datasets", {}).get(name, {}).get("status") == "PASS" for name in SCENARIOS)
    if not phase_raw_pass:
        failures.add("RAW_SOURCE_BINDING_FAILED")

    clean_rep1 = SSD / "dumps/phase_a/texbat_cleanstatic/rep1"
    clean_rep2 = SSD / "dumps/phase_a/texbat_cleanstatic/rep2"
    try:
        deterministic = deterministic_hashes(clean_rep1, clean_rep2)
    except OSError as error:
        deterministic = {"status": "FAIL", "error": str(error), "exact_file_hash_match": False}
    if deterministic["status"] != "PASS":
        failures.add("NATIVE_1MS_DUMP_IMPLEMENTATION_INVALID")

    phase_status = "PASS" if not failures else "FAIL"
    action_mapping = {
        "schema": "gnss-doppler-lab.trace-r2-action-mapping-validation.v1",
        "status": "PASS" if not any(action_totals[key] for key in action_totals if key != "causal_pair_count") and action_totals["causal_pair_count"] else "FAIL",
        "default_causal_tuple": "(complex_t, action_computed_for_next_interval_t, complex_t+1)",
        **action_totals,
        "proof": "exact source sequence, exact action equality, and next-buffer consume-span equality",
    }
    if action_mapping["status"] != "PASS":
        failures.add("ACTION_MAPPING_UNRESOLVED")
        phase_status = "FAIL"
    smoke = {
        "schema": "gnss-doppler-lab.trace-r2-smoke-replay-results.v1",
        "status": phase_status,
        "failure_labels": sorted(failures),
        "scenarios": scenario_results,
        "same_schema_across_scenarios": same_schema,
        "raw_source_binding_pass": phase_raw_pass,
        "deterministic_reproduction": deterministic,
        "phase_b_authorized": phase_status == "PASS",
    }
    dump_json(ARTIFACT / "action_mapping_validation.json", action_mapping)
    dump_json(ARTIFACT / "smoke_replay_results.json", smoke)
    print(json.dumps(smoke, indent=2, sort_keys=True), flush=True)
    return 0 if phase_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
