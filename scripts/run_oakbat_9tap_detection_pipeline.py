#!/usr/bin/env python3
"""Serial OAKBAT 5 MHz receiver/feature adapter for the frozen champion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
from gnss_doppler_lab.gnss_sdr import export_tracking_csv, parse_acquired_prns, parse_receiver_reported_prns
from gnss_doppler_lab.tracking_feature_windows import export_receiver_run_tap_feature_csv
from gnss_doppler_lab.normal_multi_prn_dataset import export_tap_multi_prn_dataset

SAMPLE_RATE=5_000_000; DURATION_S=480.0; EXPECTED_COMPLEX_SAMPLES=2_400_000_000
AVAILABLE_SCALAR_INT16_ITEMS=4_800_000_000
CONFIGURED_SIGNAL_SOURCE_SAMPLES=0
SIGNAL_SOURCE_SAMPLES_SEMANTICS="auto_until_eof"
EXPECTED_IQ_BYTES=9_600_000_000; ONSET_S=120.0; GUARD_S=10.0; CHANNELS=11
FROZEN_CHECKPOINT_SHA256="f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b"
FROZEN_FEATURE_COLUMNS=["tap_E4_rel_prompt_mean","tap_E3_rel_prompt_mean","tap_E2_rel_prompt_mean","tap_E_rel_prompt_mean","tap_P_rel_prompt_mean","tap_L_rel_prompt_mean","tap_L2_rel_prompt_mean","tap_L3_rel_prompt_mean","tap_L4_rel_prompt_mean"]
TIMING_CONTRACT={'score_time_field':'window_start_s','window_duration_s':1.0,'window_availability_offset_s':1.0,'interpretation':'scores timestamped at frozen window start become available one full window later'}
DEFAULT_MIN_FREE_BYTES=20*1024**3
ESTIMATED_OUTPUT_BYTES_PER_SCENARIO=2*1024**3
COVERAGE_TAIL_TOLERANCE_S=2.0
COVERAGE_LATE_TOLERANCE_S=1.0
RAW_FEATURE_REQUIRED_COLUMNS={"run_id","prn","window_start_s","window_end_s","window_mid_s","tap_count","tap_layout"}
NODE_REQUIRED_COLUMNS={"run_id","prn","window_bin_s"}
FEATURE_CONTRACT={"tap_count":9,"tap_spacing_chips":0.125,"window_s":1.0,"stride_s":0.5,"min_epochs":4,"min_prns_per_graph":2,"feature_mode":"normalized_dmcpd","node_feature_columns":FROZEN_FEATURE_COLUMNS}
CLEAN_SCENARIOS={"cleanStatic","cleanDynamic"}
SCENARIOS={"os1":"os1.bin","os2":"os2.bin","os3":"os3.bin","os4":"os4.bin","cleanStatic":"cleanStatic_gps.bin","cleanDynamic":"cleanDynamic_gps.bin"}


def sha256(path: Path, block: int=8*1024*1024) -> str:
    d=hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(block),b""): d.update(chunk)
    return d.hexdigest()


def iq_file_identity(path: Path) -> dict[str, object]:
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'IQ changed while hashing: {path}')
    return {'path':str(path.resolve()), 'size_bytes':after.st_size, 'sha256':digest}


def preflight_output_space(output_root: Path, *, scenario_count: int,
                           minimum_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
                           estimated_bytes_per_scenario: int = ESTIMATED_OUTPUT_BYTES_PER_SCENARIO) -> dict[str, int]:
    if scenario_count < 1 or minimum_free_bytes < 0 or estimated_bytes_per_scenario < 0:
        raise ValueError('disk preflight values must be non-negative and scenario_count positive')
    output_root.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(output_root).free)
    required = max(int(minimum_free_bytes), int(scenario_count) * int(estimated_bytes_per_scenario))
    if free < required:
        raise RuntimeError(f'insufficient output-root disk space: {free} bytes free, {required} required')
    return {'free_bytes':free, 'required_free_bytes':required,
            'minimum_free_bytes':int(minimum_free_bytes),
            'estimated_bytes_per_scenario':int(estimated_bytes_per_scenario)}


def validate_iq(path: Path) -> None:
    if not path.is_file(): raise FileNotFoundError(path)
    if path.stat().st_size != EXPECTED_IQ_BYTES:
        raise ValueError(f"OAKBAT IQ must have exact size {EXPECTED_IQ_BYTES} bytes (480 s, 5 MHz, interleaved int16 IQ); got {path.stat().st_size}: {path}")


def receiver_config(iq: Path, run_dir: Path, *, samples: int=CONFIGURED_SIGNAL_SOURCE_SAMPLES) -> str:
    prefix=run_dir/"raw"/"epl_tracking_ch_"
    return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={SAMPLE_RATE}
SignalSource.implementation=File_Signal_Source
SignalSource.filename={iq.resolve()}
SignalSource.item_type=ishort
SignalSource.sampling_frequency={SAMPLE_RATE}
SignalSource.samples={samples}
SignalSource.repeat=false
SignalSource.dump=false
SignalSource.enable_throttle_control=false
SignalConditioner.implementation=Signal_Conditioner
DataTypeAdapter.implementation=Ishort_To_Complex
InputFilter.implementation=Pass_Through
InputFilter.input_item_type=gr_complex
InputFilter.output_item_type=gr_complex
Resampler.implementation=Pass_Through
Resampler.item_type=gr_complex
Channels_1C.count={CHANNELS}
Channels.in_acquisition={CHANNELS}
Channel.signal=1C
Acquisition_1C.implementation=GPS_L1_CA_PCPS_Acquisition
Acquisition_1C.item_type=gr_complex
Acquisition_1C.coherent_integration_time_ms=1
Acquisition_1C.threshold=2.5
Acquisition_1C.doppler_max=10000
Acquisition_1C.doppler_step=100
Tracking_1C.implementation=GPS_L1_CA_DLL_PLL_Tracking
Tracking_1C.item_type=gr_complex
Tracking_1C.pll_bw_hz=20.0
Tracking_1C.dll_bw_hz=1.5
Tracking_1C.order=3
Tracking_1C.dump=true
Tracking_1C.dump_filename={prefix.resolve()}
Tracking_1C.tap_count=9
Tracking_1C.tap_spacing_chips=0.125
TelemetryDecoder_1C.implementation=GPS_L1_CA_Telemetry_Decoder
TelemetryDecoder_1C.dump=false
Observables.implementation=Hybrid_Observables
Observables.dump=true
Observables.dump_filename={(run_dir/'raw'/'observables.dat').resolve()}
PVT.implementation=RTKLIB_PVT
PVT.positioning_mode=Single
PVT.output_rate_ms=100
PVT.display_rate_ms=500
PVT.flag_rtcm_server=false
PVT.flag_rtcm_tty_port=false
PVT.dump=false
"""


def resolve_executable(exe: str | Path) -> Path:
    text=str(exe); found=shutil.which(text) if "/" not in text else None
    path=Path(found or text).resolve()
    if not path.is_file(): raise FileNotFoundError(f"GNSS-SDR executable not found: {exe}")
    return path


def _artifact(path: Path, base: Path | None=None) -> dict[str, object]:
    return {"path":str(path.relative_to(base) if base else path.resolve()),"size_bytes":path.stat().st_size,"sha256":sha256(path)}


class TrackingCoverageError(ValueError):
    def __init__(self, message: str, coverage: dict[str, object]):
        super().__init__(message)
        self.coverage = coverage


def _tracking_table(path: Path, time_field: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"tracking coverage table is unreadable: {path}: {exc}") from exc
    required = {time_field, "prn"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"tracking coverage table has invalid schema: {path}")
    times = pd.to_numeric(frame[time_field], errors="coerce")
    prns = frame["prn"].astype(str)
    valid = times.notna() & np.isfinite(times) & prns.str.fullmatch(r"G(?:0[1-9]|[12][0-9]|3[0-2])")
    if not valid.all():
        raise ValueError(f"tracking coverage contains non-finite rows or invalid GPS PRNs: {path}")
    return frame.assign(**{time_field: times})


def receiver_tracking_coverage(run_dir: Path) -> dict[str, object]:
    """Authenticate near-full 480 s coverage from valid long-form and summary rows."""
    run_dir = Path(run_dir)
    tracking = _tracking_table(run_dir / "tracking.csv", "time_s")
    summary = _tracking_table(run_dir / "tracking_summary.csv", "end_time_s")
    lower = DURATION_S - COVERAGE_TAIL_TOLERANCE_S
    upper = DURATION_S + COVERAGE_LATE_TOLERANCE_S
    csv_max = float(tracking["time_s"].max())
    summary_max = float(summary["end_time_s"].max())
    csv_prns = set(tracking["prn"].astype(str))
    summary_prns = set(summary["prn"].astype(str))
    common_prns = sorted(csv_prns & summary_prns)
    coverage = {
        "expected_duration_s": DURATION_S,
        "tail_tolerance_s": COVERAGE_TAIL_TOLERANCE_S,
        "required_min_time_s": lower,
        "allowed_max_time_s": upper,
        "tracking_csv_max_time_s": csv_max,
        "tracking_summary_max_time_s": summary_max,
        "valid_tracking_row_count": int(len(tracking)),
        "valid_tracking_summary_row_count": int(len(summary)),
        "valid_tracking_prns": common_prns,
    }
    if not common_prns or not (lower <= csv_max <= upper) or not (lower <= summary_max <= upper):
        raise TrackingCoverageError(
            f"tracking coverage outside [{lower}, {upper}]s: tracking.csv={csv_max}s, tracking_summary.csv={summary_max}s",
            coverage,
        )
    return coverage


def validate_receiver_manifest_coverage(manifest: Path) -> dict[str, object]:
    try:
        doc = json.loads(Path(manifest).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid receiver manifest for coverage: {exc}") from exc
    actual = receiver_tracking_coverage(Path(manifest).parent)
    if doc.get("status") != "complete" or doc.get("tracking", {}).get("coverage") != actual:
        raise ValueError("receiver manifest coverage is missing, stale, or incomplete")
    return actual


def receiver_cache_contract(scenario: str, iq: Path, run_dir: Path, exe: str | Path, iq_identity: dict[str, object] | None=None) -> dict[str, object]:
    exe_path=resolve_executable(exe); config=run_dir/"receiver.conf"
    identity=iq_identity or iq_file_identity(iq)
    outputs={name:_artifact(run_dir/name,run_dir) for name in ("receiver.log","tracking.csv","tracking_summary.csv")}
    mats=sorted((run_dir/"raw").glob("epl_tracking_ch_*.mat"))
    if not mats: raise ValueError("receiver cache missing Method-A tracking output")
    outputs["tracking_mat_files"]=[_artifact(x,run_dir) for x in mats]
    return {
      "schema_version":4,"status":"complete","receiver_run_id":f"oakbat-{scenario}-method-a-9tap","source_rf_run_id":f"oakbat-{scenario}",
      "source":{"dataset":"OAKBAT","scenario_id":scenario,"iq":str(iq.resolve()),"iq_size_bytes":identity["size_bytes"],"iq_sha256":identity["sha256"],"sample_rate_hz":SAMPLE_RATE,"duration_s":DURATION_S,"sample_format":"interleaved_int16_iq","available_scalar_int16_items":AVAILABLE_SCALAR_INT16_ITEMS,"expected_complex_samples":EXPECTED_COMPLEX_SAMPLES,"configured_signal_source_samples":CONFIGURED_SIGNAL_SOURCE_SAMPLES,"signal_source_samples_semantics":SIGNAL_SOURCE_SAMPLES_SEMANTICS,"signal_source_repeat":False},
      "receiver":{"name":"GNSS-SDR Method-A","executable":str(exe_path),"executable_sha256":sha256(exe_path),"config":config.name,"config_sha256":sha256(config)},
      "tracking":{"tap_count":9,"tap_spacing_chips":0.125,"raw_directory":"raw","coverage":receiver_tracking_coverage(run_dir)},"completed_outputs":outputs,
    }


def validate_cached_receiver(manifest: Path, scenario: str, iq: Path, exe: str | Path, iq_identity: dict[str, object] | None=None) -> Path:
    try: doc=json.loads(manifest.read_text())
    except (OSError,json.JSONDecodeError) as exc: raise ValueError(f"invalid cached receiver manifest: {exc}") from exc
    run_dir=manifest.parent
    config=run_dir/"receiver.conf"
    desired_hash=hashlib.sha256(receiver_config(iq,run_dir).encode("utf-8")).hexdigest()
    if not config.is_file() or sha256(config) != desired_hash:
        raise ValueError("stale cached receiver config contract; rerun with --force-receiver")
    try: expected=receiver_cache_contract(scenario,iq,run_dir,exe,iq_identity)
    except (OSError,ValueError) as exc: raise ValueError(f"stale or incomplete receiver cache: {exc}") from exc
    for section,keys in {"source":["iq","iq_size_bytes","iq_sha256","sample_rate_hz","duration_s","sample_format","available_scalar_int16_items","expected_complex_samples","configured_signal_source_samples","signal_source_samples_semantics","signal_source_repeat"],"receiver":["executable","executable_sha256","config","config_sha256"],"tracking":["tap_count","tap_spacing_chips","raw_directory","coverage"]}.items():
        if not isinstance(doc.get(section),dict) or any(doc[section].get(k)!=expected[section][k] for k in keys):
            raise ValueError(f"stale cached receiver {section} contract; rerun with --force-receiver")
    if doc.get("status")!="complete" or doc.get("completed_outputs")!=expected["completed_outputs"]:
        raise ValueError("stale or incomplete cached receiver outputs; rerun with --force-receiver")
    return manifest


def _write_receiver_failure(run_dir: Path, metadata: dict[str, object]) -> None:
    metadata = {'schema':'gnss-doppler-lab.receiver-failure.v1', 'status':'failed', **metadata}
    (run_dir/'receiver_failure.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')


def _terminate_process_group(process: subprocess.Popen, log_handle) -> bool:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
    log_handle.flush()
    return process.poll() is not None


def run_receiver(scenario: str, iq: Path, out: Path, *, exe: str, timeout_s: int, force: bool) -> Path:
    validate_iq(iq)
    identity_before = iq_file_identity(iq)
    run_dir = out / "receiver" / f"oakbat-{scenario}-method-a-9tap"
    manifest = run_dir / "manifest.json"
    if manifest.exists() and not force:
        return validate_cached_receiver(manifest, scenario, iq, exe, identity_before)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "raw").mkdir(parents=True)
    config = run_dir / "receiver.conf"
    log_path = run_dir / "receiver.log"
    config.write_text(receiver_config(iq, run_dir), encoding="utf-8")
    exe_path = resolve_executable(exe)

    version_command = [str(exe_path), "--version"]
    try:
        version_result = subprocess.run(
            version_command, capture_output=True, text=True, timeout=30, check=False
        )
    except subprocess.TimeoutExpired as exc:
        message = "GNSS-SDR version probe timed out after 30s"
        log_path.write_text(f"[OAKBAT PIPELINE FAILURE] {message}\n", encoding="utf-8")
        _write_receiver_failure(run_dir, {
            "failure_kind": "version_timeout", "scenario": scenario,
            "version_command": version_command, "version_timeout_s": 30,
            "receiver_log": str(log_path.resolve()), "error": str(exc),
        })
        raise RuntimeError(message) from exc
    if version_result.returncode != 0:
        version_output = (version_result.stdout or "") + (version_result.stderr or "")
        message = f"GNSS-SDR version probe failed with rc={version_result.returncode}"
        log_path.write_text(version_output + f"\n[OAKBAT PIPELINE FAILURE] {message}\n", encoding="utf-8")
        _write_receiver_failure(run_dir, {
            "failure_kind": "version_nonzero", "scenario": scenario,
            "version_command": version_command, "exit_code": version_result.returncode,
            "receiver_log": str(log_path.resolve()), "error": message,
            "version_stdout": version_result.stdout, "version_stderr": version_result.stderr,
        })
        raise RuntimeError(message)
    version_text = version_result.stdout or version_result.stderr

    command = [str(exe_path), f"--config_file={config.resolve()}", "--keyboard=false"]
    timed_out = False
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command, cwd=run_dir, stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True, shell=False,
        )
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        terminated = _terminate_process_group(process, log_handle)

    # Retain and compare the source identity immediately after GNSS-SDR exits.
    identity_after = iq_file_identity(iq)
    common = {
        "scenario": scenario, "command": command, "timeout_s": timeout_s,
        "exit_code": process.returncode, "process_group_terminated": terminated,
        "iq_identity_before": identity_before, "iq_identity_after": identity_after,
        "receiver_log": str(log_path.resolve()),
    }
    if timed_out:
        _write_receiver_failure(run_dir, {"failure_kind": "timeout", **common})
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[OAKBAT PIPELINE FAILURE] timeout after {timeout_s}s; process group terminated={terminated}\n")
        raise RuntimeError(f"GNSS-SDR timed out for {scenario} after {timeout_s}s")
    if identity_after != identity_before:
        _write_receiver_failure(run_dir, {"failure_kind": "iq_identity_changed", **common})
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n[OAKBAT PIPELINE FAILURE] IQ identity changed during receiver execution\n")
        raise RuntimeError(f"OAKBAT IQ changed during receiver execution for {scenario}")
    if process.returncode:
        _write_receiver_failure(run_dir, {"failure_kind": "nonzero_exit", **common})
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[OAKBAT PIPELINE FAILURE] nonzero exit rc={process.returncode}; process group terminated={terminated}\n")
        raise RuntimeError(f"GNSS-SDR failed for {scenario} rc={process.returncode}")

    log = log_path.read_text(encoding="utf-8", errors="replace")
    mats = sorted((run_dir / "raw").glob("epl_tracking_ch_*.mat"))
    if not mats:
        _write_receiver_failure(run_dir, {"failure_kind": "missing_tracking_output", **common})
        raise RuntimeError(f"GNSS-SDR produced no Method-A tracking files for {scenario}")
    try:
        report = export_tracking_csv(
            mats, run_dir / "tracking.csv", run_dir / "tracking_summary.csv",
            sample_rate_hz=SAMPLE_RATE,
        )
    except Exception as exc:
        _write_receiver_failure(run_dir, {
            "failure_kind": "tracking_export_error", **common,
            "tracking_mat_count": len(mats), "error": f"{type(exc).__name__}: {exc}",
        })
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[OAKBAT PIPELINE FAILURE] tracking export failed: {type(exc).__name__}: {exc}\n")
        raise RuntimeError(f"GNSS-SDR tracking export failed for {scenario}: {exc}") from exc

    # Export reads receiver artifacts after the process exits; guard that phase too.
    identity_after_export = iq_file_identity(iq)
    if identity_after_export != identity_before:
        export_common = {**common, "iq_identity_after": identity_after_export,
                         "iq_identity_after_receiver": identity_after}
        _write_receiver_failure(run_dir, {
            "failure_kind": "iq_identity_changed_after_tracking_export", **export_common,
        })
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n[OAKBAT PIPELINE FAILURE] IQ identity changed during tracking export\n")
        raise RuntimeError(f"OAKBAT IQ changed during tracking export for {scenario}")

    try:
        coverage = receiver_tracking_coverage(run_dir)
    except (TrackingCoverageError, ValueError) as exc:
        detail = getattr(exc, "coverage", {
            "expected_duration_s": DURATION_S,
            "required_min_time_s": DURATION_S - COVERAGE_TAIL_TOLERANCE_S,
        })
        _write_receiver_failure(run_dir, {
            "failure_kind": "incomplete_tracking_coverage", **common,
            "coverage": detail, "error": str(exc),
        })
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[OAKBAT PIPELINE FAILURE] incomplete tracking coverage: {exc}\n")
        raise RuntimeError(f"GNSS-SDR tracking coverage incomplete for {scenario}: {exc}") from exc

    doc = receiver_cache_contract(scenario, iq, run_dir, exe_path, identity_before)
    if doc["tracking"]["coverage"] != coverage:
        raise RuntimeError("receiver coverage changed before manifest publication")
    doc["receiver"].update({
        "version": version_text.strip().splitlines()[0] if version_text else "unknown",
        "command": command, "exit_code": process.returncode,
    })
    doc["acquisition"] = {
        "channel_count": CHANNELS, "tracked_prns": parse_acquired_prns(log),
        "receiver_reported_prns": parse_receiver_reported_prns(log),
    }
    doc["tracking"].update(report)
    doc["tracking"].update({"csv": "tracking.csv", "summary_csv": "tracking_summary.csv"})
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return manifest
def _finite_csv(path: Path, required: set[str]) -> pd.DataFrame:
    try: frame=pd.read_csv(path)
    except Exception as exc: raise ValueError(f"stale feature CSV: {exc}") from exc
    if frame.empty or not required.issubset(frame.columns): raise ValueError(f"stale feature schema: {path}")
    if "tap_count" in frame and not frame["tap_count"].eq(9).all(): raise ValueError(f"stale feature tap-count contract: {path}")
    if "tap_layout" in frame and not frame["tap_layout"].eq("E4,E3,E2,E,P,L,L2,L3,L4").all(): raise ValueError(f"stale feature tap-layout contract: {path}")
    numeric=frame.select_dtypes(include=[np.number])
    if frame.isna().any().any() or numeric.empty or not np.isfinite(numeric.to_numpy(float)).all(): raise ValueError(f"feature cache contains non-finite/null values: {path}")
    return frame


def write_feature_cache_contract(out: Path, receiver_manifest: Path, features: Path, node_csv: Path) -> Path:
    validate_receiver_manifest_coverage(receiver_manifest)
    _finite_csv(features,RAW_FEATURE_REQUIRED_COLUMNS)
    node=_finite_csv(node_csv,NODE_REQUIRED_COLUMNS)
    if not set(FROZEN_FEATURE_COLUMNS).issubset(node.columns):
        raise ValueError("node feature schema does not contain the frozen model features")
    doc={"schema":"gnss-doppler-lab.oakbat-feature-cache.v1","receiver_manifest":{"path":str(receiver_manifest.resolve()),"sha256":sha256(receiver_manifest)},"feature_contract":FEATURE_CONTRACT,"features":_artifact(features),"node_table":_artifact(node_csv)}
    path=out/"oakbat_feature_cache_manifest.json"; path.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n"); return path


def validate_cached_features(out: Path, receiver_manifest: Path) -> Path:
    path=out/"oakbat_feature_cache_manifest.json"
    try: doc=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError) as exc: raise ValueError(f"stale feature cache manifest: {exc}") from exc
    if doc.get("schema")!="gnss-doppler-lab.oakbat-feature-cache.v1" or doc.get("feature_contract")!=FEATURE_CONTRACT: raise ValueError("stale feature cache contract")
    rel=doc.get("receiver_manifest",{})
    if rel!={"path":str(receiver_manifest.resolve()),"sha256":sha256(receiver_manifest)}: raise ValueError("stale feature cache receiver relationship")
    features=Path(doc["features"]["path"]); node=Path(doc["node_table"]["path"])
    for key,target in (("features",features),("node_table",node)):
        if not target.is_file() or doc[key]!=_artifact(target): raise ValueError(f"stale feature cache hash: {target}")
    validate_receiver_manifest_coverage(receiver_manifest)
    _finite_csv(features,RAW_FEATURE_REQUIRED_COLUMNS)
    node_frame=_finite_csv(node,NODE_REQUIRED_COLUMNS)
    if not set(FROZEN_FEATURE_COLUMNS).issubset(node_frame.columns):
        raise ValueError("stale node feature schema does not contain the frozen model features")
    return node


def build_features(scenario: str, out: Path, receiver_manifest: Path, *, force: bool) -> Path:
    validate_receiver_manifest_coverage(receiver_manifest)
    features=out/"tap9_tracking_features_w1.0_s0.5.csv"; dataset=out/"multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd"; node=dataset/"normal_prn_node_windows.csv"; cache=out/"oakbat_feature_cache_manifest.json"
    if not force and (features.exists() or node.exists() or cache.exists()): return validate_cached_features(out,receiver_manifest)
    if force:
        for target in (features,cache): target.unlink(missing_ok=True)
        if dataset.exists(): shutil.rmtree(dataset)
    export_receiver_run_tap_feature_csv(receiver_manifest.parent,output_path=features,tap_count=9,window_s=1.0,stride_s=0.5,min_epochs=4,label=f"oakbat_{scenario}_9tap")
    _finite_csv(features,RAW_FEATURE_REQUIRED_COLUMNS)
    node_csv,_,_=export_tap_multi_prn_dataset(features,output_dir=dataset,stride_s=0.5,min_prns_per_graph=2,feature_mode="normalized_dmcpd")
    write_feature_cache_contract(out,receiver_manifest,features,Path(node_csv)); return Path(node_csv)


def validate_frozen_inputs(model_dir: Path, calibration_json: Path) -> None:
    checkpoint=model_dir/"prn_local_gru_predictor.pt"
    if not checkpoint.is_file() or sha256(checkpoint)!=FROZEN_CHECKPOINT_SHA256: raise ValueError("frozen checkpoint missing or SHA256 mismatch")
    import importlib.util
    spec=importlib.util.spec_from_file_location("oakbat_gate_validation",ROOT/"scripts/eval_btail_support_gate.py"); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    calibration=mod.load_frozen_calibration(calibration_json)
    if calibration.node_thresholds != mod.FROZEN_CALIBRATION_CONTRACT["node_thresholds"]: raise ValueError("frozen calibration contract mismatch")


def validate_score_provenance(summary_path: Path, calibration_json: Path) -> None:
    summary=json.loads(summary_path.read_text()); calibration=json.loads(calibration_json.read_text())
    if summary.get("checkpoint_provenance",{}).get("checkpoint_sha256")!=calibration.get("checkpoint_sha256"): raise ValueError("score/checkpoint calibration mismatch")


def provenance_manifest(*,scenario:str,iq:Path,iq_sha256:str,receiver_manifest:Path,node_csv:Path,score_summary:Path,gate_summary:Path,checkpoint:Path|None=None,calibration_json:Path|None=None) -> dict[str,object]:
    outputs={"score_summary":score_summary,"gate_summary":gate_summary,"receiver_manifest":receiver_manifest,"node_csv":node_csv}
    prefix=score_summary.name.replace("_prn_local_onset_summary.json", "")
    for label,suffix in (("prn_scores","_prn_local_scores.csv"),("event_scores","_prn_local_event_scores.csv"),("score_plot","_prn_local_score_vs_time.png")):
        outputs[label]=score_summary.parent/f"{prefix}{suffix}"
    outputs["gate_event_scores"]=gate_summary.parent/f"{scenario}_event_scores.csv"
    artifacts={name:_artifact(path) for name,path in outputs.items() if path.is_file()}
    frozen={"checkpoint_sha256":sha256(checkpoint) if checkpoint and checkpoint.is_file() else FROZEN_CHECKPOINT_SHA256}
    if calibration_json and calibration_json.is_file():
        cal=json.loads(calibration_json.read_text()); frozen.update({"calibration_path":str(calibration_json.resolve()),"calibration_sha256":sha256(calibration_json),"calibration_constants":cal})
    return {"schema":"gnss-doppler-lab.oakbat-frozen-champion-adapter.v1","source":{"dataset":"OAKBAT","scenario":scenario,"iq":str(iq.resolve()),"iq_sha256":iq_sha256,"sample_rate_hz":SAMPLE_RATE,"duration_s":DURATION_S,"sample_format":"interleaved_int16_iq"},"frozen_detector":frozen,"adapter":{"receiver_manifest":str(receiver_manifest),"tap_count":9,"tap_spacing_chips":0.125,"feature_mode":"normalized_dmcpd","node_csv":str(node_csv),"feature_contract":FEATURE_CONTRACT,"timing_contract":TIMING_CONTRACT},"evaluation":({"kind":"clean_negative_control"} if scenario in CLEAN_SCENARIOS else {"onset_s":ONSET_S,"guard_s":GUARD_S}),"outputs":{"score_summary":str(score_summary),"gate_summary":str(gate_summary),"artifacts":artifacts}}


def build_parser() -> argparse.ArgumentParser:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root",default="/home/ubuntu/unraid_hdd/oakbat/gps_l1ca/raw"); ap.add_argument("--out-root",default=str(ROOT/"artifacts/oakbat_9tap_frozen_champion"))
    ap.add_argument("--scenarios",nargs="+",choices=list(SCENARIOS),default=["cleanStatic","cleanDynamic","os1","os2","os3","os4"])
    ap.add_argument("--exe",default=os.environ.get("GNSS_SDR_METHOD_A_EXE",str(ROOT/".tools/gnss-sdr-method-a-9tap"))); ap.add_argument("--model-dir",default=str(ROOT/"artifacts/ai_morph_gru_cleanStatic_q70_frame")); ap.add_argument("--calibration-json",default=str(ROOT/"configs/detectors/texbat_btail_gate_v1.json"))
    ap.add_argument("--timeout-s",type=int,default=7200); ap.add_argument("--min-free-gib",type=float,default=20.0,help="Minimum free output-root GiB; scenario estimate may require more."); ap.add_argument("--force-receiver",action="store_true"); ap.add_argument("--force-features",action="store_true"); return ap


def main() -> None:
    args=build_parser().parse_args(); model_dir=Path(args.model_dir); calibration=Path(args.calibration_json); validate_frozen_inputs(model_dir,calibration)
    out_root=Path(args.out_root); raw_root=Path(args.raw_root); out_root.mkdir(parents=True,exist_ok=True); preflight_output_space(out_root,scenario_count=len(args.scenarios),minimum_free_bytes=int(args.min_free_gib*1024**3)); records={}
    for scenario in args.scenarios:
        iq=raw_root/SCENARIOS[scenario]; validate_iq(iq); out=out_root/scenario; out.mkdir(parents=True,exist_ok=True)
        receiver=run_receiver(scenario,iq,out,exe=args.exe,timeout_s=args.timeout_s,force=args.force_receiver); node=build_features(scenario,out,receiver,force=args.force_features)
        cmd=[sys.executable,str(ROOT/"scripts/score_texbat_prn_node_gru.py"),"--node-csv",str(node),"--model-dir",args.model_dir,"--out-dir",str(out),"--scenario",scenario,"--output-prefix","oakbat","--dataset-prefix","OAKBAT"]
        if scenario in CLEAN_SCENARIOS: cmd.append("--clean-only")
        else: cmd += ["--onset-s",str(ONSET_S)]
        subprocess.run(cmd,check=True,timeout=args.timeout_s); summary=out/f"oakbat_{scenario}_prn_local_onset_summary.json"; validate_score_provenance(summary,calibration); records[scenario]=(iq,receiver,node)
    gate_dir=out_root/"gate"; attacks=[s for s in args.scenarios if s not in CLEAN_SCENARIOS]
    gate_cmd=[sys.executable,str(ROOT/"scripts/eval_btail_support_gate.py"),"--score-root",str(out_root),"--out-dir",str(gate_dir),"--scenarios",",".join(args.scenarios),"--calibration-json",str(calibration),"--score-prefix","oakbat","--onsets-json",json.dumps({s:ONSET_S for s in attacks}),"--onset-buffer-s",str(GUARD_S)]
    subprocess.run(gate_cmd,check=True,timeout=args.timeout_s)
    checkpoint=model_dir/"prn_local_gru_predictor.pt"
    for scenario,(iq,receiver,node) in records.items():
        summary=out_root/scenario/f"oakbat_{scenario}_prn_local_onset_summary.json"; receiver_doc=json.loads(receiver.read_text()); doc=provenance_manifest(scenario=scenario,iq=iq,iq_sha256=receiver_doc["source"]["iq_sha256"],receiver_manifest=receiver,node_csv=node,score_summary=summary,gate_summary=gate_dir/"summary.json",checkpoint=checkpoint,calibration_json=calibration)
        (out_root/scenario/"oakbat_frozen_champion_manifest.json").write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"scenarios":args.scenarios,"gate_summary":str(gate_dir/"summary.json")},indent=2))

if __name__=="__main__": main()
