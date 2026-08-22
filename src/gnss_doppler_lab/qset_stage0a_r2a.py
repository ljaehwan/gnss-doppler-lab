"""Locked Q-SET R2a SS-1 receiver-support root-cause audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import numpy as np

from . import qset_stage0a_r2 as r2
from .trace_native_1ms import MEASUREMENT_FIELDS, complex_taps, read_records, sha256_file

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/qset_gnss_stage0a_r2a_ss1_receiver_support_audit"
SSD_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/qset-gnss-stage0a-r2a-ss1-receiver-support-audit")
R2_ARTIFACT = ROOT / "artifacts/qset_gnss_stage0a_r2_galileo_partial_prn_execution"
R2_SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/qset-gnss-stage0a-r2-galileo-partial-prn-execution")
RECEIVER = R2_SSD / "receiver-build-v7/src/main/gnss-sdr"
SOURCE = R2_SSD / "receiver-source-v7"
BASE_SHA = "0df61a01485e6d7ac5cee21b861533541ec29859"
BRANCH = "research/qset-gnss-stage0a-r2a-ss1-receiver-support-audit"
R2_OUTPUTS = {"C-1": "b01c45a8c435c1fd3634346ae685a31a3edcb3e60510be2850e39d6c85e72384", "C-3": "e6205442e56b73d8ea92a525ff17ef97fcbed5bd0b082decb74648e48d5ad446", "SS-1": "45e353a3e1d51f719ec746ee5f20b6c644f06d276456d8bcb0860d6d897e1e4c"}
SS1_DECODED_SHA256 = "b33689e2cdbbf750ce1e9bfb19a244b85f8e299a9a3f7cf2f52d63be146587f0"
RECEIVER_SHA256 = "3ed059f699201807cb86eb54a24ba00fde7e248f37a8081306cbc76d6be1b06f"
MIN_EPOCHS = 125
MIN_PANEL = 5
PRN9_STABLE_WINDOWS = 60
RUN_NAMESPACE = "variants-repair-v1"
VARIANTS = {
    "V0": {"in_acquisition": 4, "pfa": "0.00001", "coherent_ms": 4, "execution": "REUSE_VERIFIED_R2_BASELINE"},
    "V1": {"in_acquisition": 12, "pfa": "0.00001", "coherent_ms": 4, "execution": "RUN"},
    "V2": {"in_acquisition": 12, "pfa": "0.0001", "coherent_ms": 4, "execution": "RUN"},
    "V3": {"in_acquisition": 12, "pfa": "0.00001", "coherent_ms": 8, "execution": "RUN"},
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def file_binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def output_manifest(root: Path) -> dict[str, Any]:
    rows = [file_binding(path) | {"path": path.relative_to(root).as_posix()} for path in sorted(root.rglob("*")) if path.is_file() and path.name != "manifest.json"]
    return {"files": rows, "file_count": len(rows), "aggregate_sha256": canonical_sha(rows)}


def verify_output_manifest(root: Path, manifest: dict[str, Any]) -> None:
    actual = output_manifest(root)
    require(actual == manifest, f"output manifest drift: {root}")


def support_from_traces(receiver_dir: Path, scenario: str = "SS-1") -> dict[str, Any]:
    paths = sorted(receiver_dir.glob("trace_native_1ms_ch_*.bin"))
    require(len(paths) == 12, f"expected 12 TRACE channels, got {len(paths)}")
    per_second: dict[int, dict[int, int]] = {}
    files: list[dict[str, Any]] = []
    tracked: set[int] = set()
    finite_failures = cadence_failures = causal_failures = 0
    per_prn_records: dict[int, int] = {}
    per_prn_start: dict[int, float] = {}
    per_prn_end: dict[int, float] = {}
    per_prn_cn0: dict[int, list[float]] = {}
    per_prn_lock: dict[int, list[float]] = {}
    for path in paths:
        header, records = read_records(path)
        require(header.scenario_id == scenario, f"scenario header drift: {path}")
        row = file_binding(path) | {"record_count": int(len(records))}
        if not len(records):
            row["status"] = "EMPTY_OPTIONAL_CHANNEL"
            files.append(row)
            continue
        taps = complex_taps(records)
        measurements = np.column_stack([records[name] for name in MEASUREMENT_FIELDS])
        finite = np.isfinite(taps.real).all(axis=1) & np.isfinite(taps.imag).all(axis=1) & np.isfinite(measurements).all(axis=1)
        finite_failures += int((~finite).sum())
        tracked.update(int(value) for value in np.unique(records["prn"]))
        same = (records["tracking_session_id"][1:] == records["tracking_session_id"][:-1]) & (records["prn"][1:] == records["prn"][:-1])
        indices = np.flatnonzero(same) + 1
        if len(indices):
            prior = indices - 1
            dt = (records["raw_interval_start_sample"][indices] - records["raw_interval_start_sample"][prior]).astype(float) / r2.OUTPUT_FS
            cadence_failures += int(np.sum((dt < 0.0035) | (dt > 0.0045)))
            causal_failures += int(np.sum(records["action_used_source_loop_sequence"][indices] != records["loop_sequence"][prior]))
        valid = finite & (records["valid_tracking"] == 1) & (records["valid_lock"] == 1)
        raw_time = (records["raw_interval_start_sample"].astype(float) * r2.RESAMPLER_RATIO - r2.RESAMPLER_GROUP_DELAY_RAW_SAMPLES) / r2.RAW_FS
        seconds = np.floor(raw_time).astype(int) + 1
        for index in np.flatnonzero(valid & (seconds >= 1) & (seconds <= 149)):
            prn = int(records["prn"][index]); second = int(seconds[index]); timestamp = float(raw_time[index])
            per_second.setdefault(second, {})[prn] = per_second.setdefault(second, {}).get(prn, 0) + 1
            per_prn_records[prn] = per_prn_records.get(prn, 0) + 1
            per_prn_start[prn] = min(per_prn_start.get(prn, timestamp), timestamp)
            per_prn_end[prn] = max(per_prn_end.get(prn, timestamp), timestamp)
            per_prn_cn0.setdefault(prn, []).append(float(records["cn0_db_hz"][index]))
            per_prn_lock.setdefault(prn, []).append(float(records["carrier_lock_test"][index]))
        row.update({"status": "PASS", "prns": sorted(set(int(value) for value in records["prn"]))})
        files.append(row)
    panels = {second: sorted(prn for prn, count in counts.items() if count >= MIN_EPOCHS) for second, counts in sorted(per_second.items())}
    m5 = [second for second, panel in panels.items() if len(panel) >= MIN_PANEL]
    longest = current = 0; prior = None
    for second in m5:
        current = current + 1 if prior is not None and second == prior + 1 else 1
        longest = max(longest, current); prior = second
    churn = 0
    ordered = sorted(panels)
    for left, right in zip(ordered, ordered[1:]):
        churn += len(set(panels[left]).symmetric_difference(panels[right]))
    per_prn = []
    for prn in sorted(per_prn_records):
        qualifying = [second for second, panel in panels.items() if prn in panel]
        per_prn.append({"prn": prn, "valid_records": per_prn_records[prn], "qualifying_one_second_windows": len(qualifying), "first_valid_s": per_prn_start[prn], "last_valid_s": per_prn_end[prn], "duration_span_s": per_prn_end[prn] - per_prn_start[prn], "median_cn0_db_hz": float(np.median(per_prn_cn0[prn])), "median_carrier_lock_test": float(np.median(per_prn_lock[prn]))})
    prn9_windows = next((row["qualifying_one_second_windows"] for row in per_prn if row["prn"] == 9), 0)
    status = "PASS" if finite_failures == cadence_failures == causal_failures == 0 else "FAIL"
    return {"status": status, "files": files, "tracked_prns": sorted(tracked), "tracked_prn_count": len(tracked), "per_prn": per_prn, "panels": panels, "median_panel_size": float(np.median([len(panel) for panel in panels.values()])) if panels else 0.0, "m_ge_5_windows": len(m5), "longest_m_ge_5_run": longest, "prn9_stable_windows": prn9_windows, "prn9_stable": prn9_windows >= PRN9_STABLE_WINDOWS, "panel_churn": churn, "finite_failures": finite_failures, "cadence_failures": cadence_failures, "causal_failures": causal_failures}


def log_events(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = [{"channel": int(ch), "prn": int(prn)} for ch, prn in re.findall(r"Tracking of Galileo E1C signal started on channel (\d+) for satellite Galileo PRN E(\d+)", text)]
    preambles = sorted(set(int(v) for v in re.findall(r"Preamble detection for Galileo satellite Galileo PRN E(\d+)", text)))
    telemetry = sorted(set(int(v) for v in re.findall(r"I/NAV .*?Galileo PRN E(\d+)", text)))
    initial = [{"channel": int(ch), "prn": int(prn)} for ch, prn in re.findall(r"Channel (\d+) assigned to Galileo PRN E(\d+)", text)]
    return {"initial_assignments": initial, "tracking_starts": starts, "acquisition_success_prns": sorted(set(row["prn"] for row in starts)), "preamble_prns": preambles, "telemetry_prns": telemetry, "loss_of_lock_events": text.count("Loss of lock"), "synchronization_timeout_events": text.count("Synchronization time limit"), "terminal_drain": "Draining receiver" in text and "Received action DRAIN" in text, "program_ended": "GNSS-SDR program ended" in text, "individual_failed_acquisition_attempts_observable": False, "failure_observability_reason": "GNSSFlowgraph acquisition failure and reassignment messages are DLOG(INFO), compiled out in the frozen Release receiver"}


def render_variant_config(variant: str, receiver_dir: Path, scenario: str = "SS-1") -> str:
    spec = VARIANTS[variant]
    require(scenario in ("SS-1", "C-1", "C-3"), "scenario outside R2a allowlist")
    baseline = R2_SSD / f"replays/{scenario}/receiver/receiver.conf"
    text = baseline.read_text(encoding="utf-8")
    decoded = R2_SSD / f"replays/{scenario}/decoded_4msps_gr_complex.bin"
    expected = read_json(R2_ARTIFACT / f"receiver_manifests/{scenario}.json")["decoder"]["sha256"]
    require(sha256_file(decoded) == expected, f"{scenario} decoded IQ drift")
    replacements = {
        r"^Channels\.in_acquisition=.*$": f"Channels.in_acquisition={spec['in_acquisition']}",
        r"^Acquisition_1B\.pfa=.*$": f"Acquisition_1B.pfa={spec['pfa']}",
        r"^Acquisition_1B\.coherent_integration_time_ms=.*$": f"Acquisition_1B.coherent_integration_time_ms={spec['coherent_ms']}",
        r"^Tracking_1B\.trace_dump_filename=.*$": f"Tracking_1B.trace_dump_filename={receiver_dir / 'trace_native_1ms_ch_'}",
    }
    for pattern, replacement in replacements.items():
        text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
        require(count == 1, f"config replacement count {pattern}: {count}")
    require(str(decoded) in text, "decoded IQ binding absent")
    return text



def run_clean_replay(selected: str, scenario: str) -> dict[str, Any]:
    require(selected in ("V1", "V2", "V3") and scenario in ("C-1", "C-3"), "clean replay outside frozen contract")
    root = SSD_ROOT / "clean-regression" / selected / scenario; manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        prior = read_json(manifest_path); verify_output_manifest(root, prior["output_set"]); return prior
    require(not root.exists(), f"incomplete clean output exists; refusing overwrite: {root}")
    receiver_dir = root / "receiver"; receiver_dir.mkdir(parents=True)
    config = receiver_dir / "receiver.conf"; config.write_text(render_variant_config(selected, receiver_dir, scenario), encoding="utf-8")
    command = [str(RECEIVER), f"--config_file={config}", "--keyboard=false", "--logtostderr=true", "--logbufsecs=0"]
    log = receiver_dir / "receiver.log"; started = time.time()
    with log.open("wb") as stream:
        completed = subprocess.run(command, cwd=receiver_dir, stdout=stream, stderr=subprocess.STDOUT, timeout=7200)
    support = support_from_traces(receiver_dir, scenario); events = log_events(log)
    terminal = completed.returncode == 0 and events["terminal_drain"] and events["program_ended"]
    result = {"status": "PASS" if terminal and support["status"] == "PASS" else "FAIL", "variant": selected, "scenario": scenario, "parameters": VARIANTS[selected], "exit_code": completed.returncode, "terminal_drain": terminal, "elapsed_s": time.time() - started, "config": file_binding(config), "support": support, "events": events}
    require(result["status"] == "PASS", f"clean regression replay failed: {scenario}")
    write_json(manifest_path, result); result["output_set"] = output_manifest(root); write_json(manifest_path, result); return result


def score_clean_regression(selected: str) -> dict[str, Any]:
    replays = {name: run_clean_replay(selected, name) for name in ("C-1", "C-3")}
    model = read_json(R2_ARTIFACT / "normal_model.json"); threshold_binding = read_json(R2_ARTIFACT / "threshold_binding.json")
    summary = read_json(R2_ARTIFACT / "clean_score_summary.json")
    require(canonical_sha(model) == summary["model_sha256"], "R2 model binding drift")
    expected_threshold_sha = summary["threshold_sha256"]
    require(canonical_sha({"multi_q_reference": threshold_binding["multi_q_reference"], "thresholds": threshold_binding["thresholds"]}) == expected_threshold_sha, "R2 threshold binding drift")
    metrics: dict[str, Any] = {}
    for name in ("C-1", "C-3"):
        rows = r2.extract_window_features(SSD_ROOT / "clean-regression" / selected / name / "receiver", name, 149.99916)
        windows = r2.dynamic_windows(rows, model)
        for window in windows: window["aggregates"] = r2.aggregate_scores(window["scores"], threshold_binding["multi_q_reference"])
        raw = [window["aggregates"]["MULTI_Q"] for window in windows]; ends = [window["window_end_s"] for window in windows]
        continuous, warmup = r2.persistence(raw, ends); eligible = (~warmup) & np.isfinite(continuous); alarms = continuous[eligible] > threshold_binding["thresholds"]["MULTI_Q"]["threshold"]
        count = int(np.sum(eligible)); alarm_count = int(np.sum(alarms))
        metrics[name] = {"feature_rows": len(rows), "scoreable_windows": len(windows), "eligible_windows": count, "alarm_count": alarm_count, "empirical_fpr": float(np.mean(alarms)) if count else 1.0, "wilson_95_upper": r2.wilson_upper(alarm_count, count), "minimum_panel": min((len(window["prns"]) for window in windows), default=0)}
    passed = all(value["scoreable_windows"] >= 100 and value["minimum_panel"] >= 5 and value["empirical_fpr"] <= 0.01 and value["wilson_95_upper"] <= 0.05 for value in metrics.values())
    return {"schema": "gnss-doppler-lab.qset-stage0a-r2a-clean-regression.v1", "status": "PASS" if passed else "FAIL", "selected_variant": selected, "model_sha256": canonical_sha(model), "threshold_sha256": expected_threshold_sha, "refit": False, "metrics": metrics, "replays": replays}


def run_variant(variant: str) -> dict[str, Any]:
    require(variant in ("V1", "V2", "V3"), "V0 is preserved baseline and must not rerun")
    require(sha256_file(RECEIVER) == RECEIVER_SHA256, "receiver drift")
    root = SSD_ROOT / RUN_NAMESPACE / variant
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        prior = read_json(manifest_path); verify_output_manifest(root, prior["output_set"])
        require(prior["status"] == "PASS", f"cached {variant} failed")
        return prior
    require(not root.exists(), f"incomplete output exists; refusing overwrite: {root}")
    free = os.statvfs(SSD_ROOT.parent); require(free.f_bavail * free.f_frsize >= 5_000_000_000, "insufficient disk")
    receiver_dir = root / "receiver"; receiver_dir.mkdir(parents=True)
    config = receiver_dir / "receiver.conf"; config.write_text(render_variant_config(variant, receiver_dir), encoding="utf-8")
    command = [str(RECEIVER), f"--config_file={config}", "--keyboard=false", "--logtostderr=true", "--logbufsecs=0"]
    log = receiver_dir / "receiver.log"; started = time.time()
    with log.open("wb") as stream:
        completed = subprocess.run(command, cwd=receiver_dir, stdout=stream, stderr=subprocess.STDOUT, timeout=7200)
    support = support_from_traces(receiver_dir); events = log_events(log)
    terminal = completed.returncode == 0 and events["terminal_drain"] and events["program_ended"]
    result = {"schema": "gnss-doppler-lab.qset-stage0a-r2a-variant-run.v1", "status": "PASS" if terminal and support["status"] == "PASS" else "FAIL", "variant": variant, "parameters": VARIANTS[variant], "input": file_binding(R2_SSD / "replays/SS-1/decoded_4msps_gr_complex.bin"), "receiver": file_binding(RECEIVER), "config": file_binding(config), "command": command, "exit_code": completed.returncode, "elapsed_s": time.time() - started, "terminal_drain": terminal, "support": support, "events": events}
    require(result["status"] == "PASS", f"{variant} receiver execution failed closed")
    write_json(manifest_path, result); result["output_set"] = output_manifest(root); write_json(manifest_path, result)
    return result


def baseline_result() -> dict[str, Any]:
    root = R2_SSD / "replays/SS-1"; original = read_json(R2_ARTIFACT / "receiver_manifests/SS-1.json")
    require(original["output_set"]["aggregate_sha256"] == R2_OUTPUTS["SS-1"], "R2 SS-1 manifest binding drift")
    verify_output_manifest(root, original["output_set"])
    support = support_from_traces(root / "receiver")
    events = log_events(root / "receiver/receiver.log")
    return {"schema": "gnss-doppler-lab.qset-stage0a-r2a-preserved-baseline.v1", "status": "PASS", "variant": "V0", "execution": "REUSED_VERIFIED_R2_BASELINE", "parameters": VARIANTS["V0"], "r2_output_set_sha256": R2_OUTPUTS["SS-1"], "receiver": original["receiver"], "support": support, "events": events}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def make_preregistration() -> dict[str, Any]:
    source_files = [SOURCE / "src/core/receiver/gnss_flowgraph.cc", SOURCE / "src/algorithms/acquisition/adapters/galileo_e1_pcps_ambiguous_acquisition.cc", SOURCE / "src/algorithms/acquisition/libs/acq_conf.cc"]
    variants = [{"variant": name, **spec, "supported": True, "support_basis": "Channels.in_acquisition is clamped to channel count; PFA source range is (0,1]; coherent time must be a 4-ms multiple and 8-ms adapter test exists"} for name, spec in VARIANTS.items()]
    return {"schema": "gnss-doppler-lab.qset-stage0a-r2a-preregistration.v1", "status": "FROZEN_PRE_SS1_VARIANT_EXECUTION", "base_sha": BASE_SHA, "branch": BRANCH, "objective": "distinguish SS-1 dataset support limit from acquisition/channel scheduling limitation", "prohibited": ["spoofing score", "morphology", "ROC/AUC", "threshold/model/feature/window/aggregator/persistence changes", "minimum-five relaxation", "SS-3/SS-5/SS-11/SS-12/SS-13 raw access"], "inputs": {"R2_C1_C3_SS1_outputs_read_only": R2_OUTPUTS, "SS1_decoded_sha256": SS1_DECODED_SHA256, "receiver_sha256": RECEIVER_SHA256}, "variants": variants, "execution_order": ["V0_REUSE", "V1", "V2", "V3"], "success_gate": {"m_ge_5_windows_minimum": 60, "prn9_stable_windows_minimum": PRN9_STABLE_WINDOWS, "finite_failures": 0, "cadence_failures": 0, "causal_failures": 0, "terminal_drain": True, "real_valid_tracking": True}, "selection_order": ["gate pass", "maximum M>=5 windows", "maximum median panel", "minimum churn", "closest to baseline"], "clean_regression": "selected variant only, C-1/C-3, existing R2 model/threshold, no refit", "source_bindings": [file_binding(path) for path in source_files], "executable_bindings": [file_binding(ROOT / path) for path in ("src/gnss_doppler_lab/qset_stage0a_r2a.py", "scripts/run_qset_gnss_stage0a_r2a.py", "scripts/verify_qset_gnss_stage0a_r2a.py", "tests/test_qset_gnss_stage0a_r2a.py")], "result_driven_variants_forbidden": True, "score_operations_before_selection": 0}


def manifest_for_artifact() -> dict[str, Any]:
    excluded = {"artifact_manifest_sha256.json", "verifier_output.txt", "fresh_clone_verifier_output.txt"}
    rows = [file_binding(path) | {"path": path.relative_to(ARTIFACT).as_posix()} for path in sorted(ARTIFACT.rglob("*")) if path.is_file() and path.name not in excluded]
    return {"schema": "gnss-doppler-lab.qset-stage0a-r2a-artifact-manifest.v1", "status": "PASS", "files": rows, "aggregate_sha256": canonical_sha(rows)}
