"""Immutable replay and clean-freeze execution for Q-SET Stage-0A R2."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .qset_stage0a_r2 import *  # noqa: F401,F403


def output_manifest(root: Path, *, include_decoder: bool = True) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and (include_decoder or path.name != "decoded_4msps_gr_complex.bin"):
            files.append({"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": files, "file_count": len(files), "aggregate_sha256": canonical_sha(files)}


def verify_manifest(root: Path, manifest: dict[str, Any]) -> None:
    for row in manifest["files"]:
        path = root / row["path"]
        require(path.is_file(), f"missing output {path}")
        require(path.stat().st_size == int(row["size_bytes"]), f"output size drift {path}")
        require(sha256_file(path) == row["sha256"], f"output hash drift {path}")
    require(canonical_sha(manifest["files"]) == manifest["aggregate_sha256"], "aggregate manifest drift")


def build_receiver() -> dict[str, Any]:
    SSD_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = SSD_ROOT / "receiver_build_manifest.json"
    if manifest_path.exists():
        prior = read_json(manifest_path)
        require(RECEIVER.is_file() and sha256_file(RECEIVER) == prior["receiver_sha256"], "cached receiver drift")
        return prior
    require(not RECEIVER_SOURCE.exists() and not RECEIVER_BUILD.exists(), "partial receiver build exists; refusing overwrite")
    require(sha256_file(R2C_PATCH) == R2C_PATCH_SHA256, "R2c receiver patch drift")
    require(sha256_file(CONFIG_TEMPLATE) == CONFIG_SHA256, "receiver config drift")
    source_status = subprocess.run(["git", "-c", f"safe.directory={R2C_SOURCE}", "-C", str(R2C_SOURCE), "rev-parse", "HEAD"], text=True, capture_output=True)
    require(source_status.returncode == 0 and source_status.stdout.strip() == BASE_RECEIVER_COMMIT, "R2c source base drift")
    subprocess.run(["cp", "-a", "--reflink=auto", str(R2C_SOURCE), str(RECEIVER_SOURCE)], check=True)
    subprocess.run(["git", "-c", f"safe.directory={RECEIVER_SOURCE}", "-C", str(RECEIVER_SOURCE), "apply", str(GALILEO_PATCH)], check=True)
    subprocess.run(["git", "-c", f"safe.directory={RECEIVER_SOURCE}", "-C", str(RECEIVER_SOURCE), "diff", "--check"], check=True)
    configure = [
        "cmake", "-S", str(RECEIVER_SOURCE), "-B", str(RECEIVER_BUILD), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={SSD_ROOT / 'receiver-install'}", f"-DPYTHON_EXECUTABLE={ROOT / '.venv/bin/python'}",
        f"-DPython3_EXECUTABLE={ROOT / '.venv/bin/python'}", "-DENABLE_LOG=ON", "-DENABLE_UHD=ON", "-DENABLE_ZMQ=ON",
        "-DENABLE_UNIT_TESTING=ON", "-DENABLE_UNIT_TESTING_EXTRA=OFF", "-DENABLE_SYSTEM_TESTING=OFF", "-DENABLE_SYSTEM_TESTING_EXTRA=OFF",
    ]
    environment = dict(os.environ); environment["PYTHONPATH"] = "/usr/lib/python3/dist-packages"
    subprocess.run(configure, check=True, env=environment)
    subprocess.run(["cmake", "--build", str(RECEIVER_BUILD), "--target", "gnss-sdr", "-j", "12"], check=True, env=environment)
    require(RECEIVER.is_file(), "receiver build produced no executable")
    version = subprocess.run([str(RECEIVER), "--version"], text=True, capture_output=True, check=True)
    diff = subprocess.run(["git", "-c", f"safe.directory={RECEIVER_SOURCE}", "-C", str(RECEIVER_SOURCE), "diff", "--binary", "--no-ext-diff", BASE_RECEIVER_COMMIT], capture_output=True, check=True).stdout
    result = {
        "schema": "gnss-doppler-lab.qset-stage0a-r2-receiver-build.v1", "status": "PASS",
        "receiver_base_commit": BASE_RECEIVER_COMMIT, "r2c_patch_sha256": R2C_PATCH_SHA256,
        "galileo_patch_sha256": sha256_file(GALILEO_PATCH), "combined_source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "receiver_path": str(RECEIVER), "receiver_size_bytes": RECEIVER.stat().st_size, "receiver_sha256": sha256_file(RECEIVER),
        "version": (version.stdout + version.stderr).strip(), "configure_command": configure,
        "build_command": ["cmake", "--build", str(RECEIVER_BUILD), "--target", "gnss-sdr", "-j", "12"],
    }
    write_json(manifest_path, result)
    return result


def render_config(input_path: Path, trace_prefix: Path, samples: int, scenario: str) -> str:
    text = CONFIG_TEMPLATE.read_text(encoding="utf-8")
    values = {"@@INPUT@@": str(input_path), "@@TRACE_PREFIX@@": str(trace_prefix), "@@SAMPLES@@": str(samples), "@@SCENARIO@@": scenario, "@@SOURCE_COMMIT@@": BASE_RECEIVER_COMMIT}
    for token, value in values.items():
        require(token in text, f"missing config token {token}")
        text = text.replace(token, value)
    require("@@" not in text, "unrendered config token")
    return text


def verify_raw(name: str, path: Path) -> dict[str, Any]:
    spec = SCENARIOS[name]
    before = path.stat(); require(before.st_size == spec["size"], f"{name} raw size mismatch")
    actual = md5_file(path); after = path.stat(); require(actual == spec["md5"], f"{name} raw MD5 mismatch")
    require((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino), f"{name} raw mutated during hash")
    return {"path": str(path), "size_bytes": before.st_size, "md5": actual, "stable_during_hash": True}


def _load_r1() -> Any:
    path = ROOT / "scripts/run_qset_gnss_stage0a_r1.py"
    spec = importlib.util.spec_from_file_location("qset_r1_decoder", path)
    require(spec is not None and spec.loader is not None, "cannot load R1 decoder")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def replay_scenario(name: str, path: Path, receiver_build: dict[str, Any]) -> dict[str, Any]:
    scenario_root = SSD_ROOT / "replays" / name; manifest_path = scenario_root / "manifest.json"
    if manifest_path.exists():
        prior = read_json(manifest_path); verify_manifest(scenario_root, prior["output_set"])
        require(prior["status"] == "PASS" and prior["raw_input"]["md5"] == SCENARIOS[name]["md5"], f"cached replay invalid {name}")
        return prior
    require(not scenario_root.exists(), f"incomplete replay exists; refusing overwrite: {scenario_root}")
    free = os.statvfs(SSD_ROOT); require(free.f_bavail * free.f_frsize >= 15_000_000_000, "insufficient disk for replay")
    scenario_root.mkdir(parents=True); raw_identity = verify_raw(name, path); r1 = _load_r1()
    decoded = scenario_root / "decoded_4msps_gr_complex.bin"; raw_samples = int(SCENARIOS[name]["size"] // BYTES_PER_COMPLEX)
    decoder = r1.stream_decode_resample(path, decoded, 0, raw_samples)
    receiver_dir = scenario_root / "receiver"; receiver_dir.mkdir(); config_path = receiver_dir / "receiver.conf"
    config_path.write_text(render_config(decoded, receiver_dir / "trace_native_1ms_ch_", int(decoder["output_samples"]), name), encoding="utf-8")
    command = [str(RECEIVER), f"--config_file={config_path}", "--keyboard=false", "--logtostderr=true", "--logbufsecs=0"]
    log_path = receiver_dir / "receiver.log"; started = time.time()
    with log_path.open("wb") as log:
        completed = subprocess.run(command, cwd=receiver_dir, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    elapsed = time.time() - started; log_text = log_path.read_text(encoding="utf-8", errors="replace")
    trace_validation = validate_galileo_trace(sorted(receiver_dir.glob("trace_native_1ms_ch_*.bin")), name)
    terminal = completed.returncode == 0 and "Draining receiver" in log_text and "Received action DRAIN" in log_text
    status = "PASS" if terminal and trace_validation["status"] == "PASS" else "FAIL"
    preliminary = {"schema": "gnss-doppler-lab.qset-stage0a-r2-replay.v1", "status": status, "scenario": name, "raw_input": raw_identity, "decoder": decoder, "receiver": {"path": str(RECEIVER), "sha256": receiver_build["receiver_sha256"], "config_sha256": sha256_file(config_path), "command": command, "exit_code": completed.returncode, "elapsed_s": elapsed, "terminal_drain": terminal}, "trace_validation": trace_validation}
    require(status == "PASS", f"receiver replay failed closed: {name}")
    write_json(manifest_path, preliminary); preliminary["output_set"] = output_manifest(scenario_root); write_json(manifest_path, preliminary)
    return preliminary


def save_feature_cache(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = SSD_ROOT / "features" / f"{name}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"prn": np.asarray([row["prn"] for row in rows], dtype=np.int16), "window_start_s": np.asarray([row["window_start_s"] for row in rows], dtype=np.int16), "window_end_s": np.asarray([row["window_end_s"] for row in rows], dtype=np.int16), "epoch_count": np.asarray([row["epoch_count"] for row in rows], dtype=np.int32), "features": np.asarray([row["feature"] for row in rows]), "cn0_median": np.asarray([row["cn0_median"] for row in rows]), "lock_median": np.asarray([row["lock_median"] for row in rows])}
    np.savez_compressed(path, **arrays)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "row_count": len(rows)}


def load_feature_cache(name: str) -> list[dict[str, Any]]:
    path = SSD_ROOT / "features" / f"{name}.npz"; require(path.is_file(), f"missing feature cache {name}")
    with np.load(path, allow_pickle=False) as data:
        return [{"scenario": name, "prn": int(data["prn"][i]), "window_start_s": int(data["window_start_s"][i]), "window_end_s": int(data["window_end_s"][i]), "epoch_count": int(data["epoch_count"][i]), "feature": data["features"][i], "cn0_median": float(data["cn0_median"][i]), "lock_median": float(data["lock_median"][i])} for i in range(len(data["prn"]))]


def analyze_clean() -> dict[str, Any]:
    c1_rows = load_feature_cache("C-1"); c3_rows = load_feature_cache("C-3")
    development = [row for row in c1_rows if 1 <= int(row["window_end_s"]) <= 100]; calibration_rows = [row for row in c1_rows if 111 <= int(row["window_end_s"]) <= 149]
    require(development and calibration_rows, "clean split has no PRN rows")
    model = fit_robust_model(np.asarray([row["feature"] for row in development])); calibration_windows = dynamic_windows(calibration_rows, model); holdout_windows = dynamic_windows(c3_rows, model)
    require(len(holdout_windows) >= 100, "clean holdout support below 100 windows")
    multi_ref = fit_multi_q_reference(window["scores"] for window in calibration_windows)
    for windows in (calibration_windows, holdout_windows):
        for window in windows: window["aggregates"] = aggregate_scores(window["scores"], multi_ref)
    thresholds = {aggregator: calibrate_threshold([window["aggregates"][aggregator] for window in calibration_windows], [window["window_end_s"] for window in calibration_windows]) for aggregator in AGGREGATORS}
    metrics: dict[str, Any] = {}
    for aggregator in AGGREGATORS:
        raw = [window["aggregates"][aggregator] for window in holdout_windows]; ends = [window["window_end_s"] for window in holdout_windows]; continuous, warmup = persistence(raw, ends); eligible = ~warmup & np.isfinite(continuous); alarms = continuous[eligible] > thresholds[aggregator]["threshold"]
        metrics[aggregator] = {"eligible_windows": int(np.sum(eligible)), "alarm_count": int(np.sum(alarms)), "empirical_fpr": float(np.mean(alarms)) if len(alarms) else 1.0, "wilson_95_upper": wilson_upper(int(np.sum(alarms)), len(alarms))}
    status = "PASS" if metrics["MULTI_Q"]["empirical_fpr"] <= 0.01 and metrics["MULTI_Q"]["wilson_95_upper"] <= 0.05 else "FAIL"
    return {"schema": "gnss-doppler-lab.qset-stage0a-r2-clean-analysis.v1", "status": status, "model": model, "multi_q_reference": multi_ref, "thresholds": thresholds, "clean_metrics": metrics, "development_prn_windows": len(development), "calibration_event_windows": len(calibration_windows), "holdout_event_windows": len(holdout_windows), "model_sha256": canonical_sha(model), "threshold_sha256": canonical_sha({"multi_q_reference": multi_ref, "thresholds": thresholds})}


def synthetic_dilution(clean_analysis: dict[str, Any]) -> dict[str, Any]:
    windows = dynamic_windows(load_feature_cache("C-3"), clean_analysis["model"]); require(windows, "no clean windows for synthetic dilution"); rows = []
    for seed in (11, 29, 47, 83):
        rng = np.random.default_rng(seed)
        for shift in (0.5, 1.0, 2.0, 4.0):
            for window in windows[:30]:
                base = np.asarray(window["scores"], dtype=float); m = len(base)
                for requested in (1, 2, 3, 5, m):
                    k = min(int(requested), m); selected = rng.choice(m, size=k, replace=False); changed = base.copy(); changed[selected] += shift; before = aggregate_scores(base, clean_analysis["multi_q_reference"]); after = aggregate_scores(changed, clean_analysis["multi_q_reference"])
                    rows.append({"seed": seed, "shift": shift, "M": m, "k": k, **{f"delta_{key}": after[key] - before[key] for key in AGGREGATORS}})
    mean_error = max(abs(row["delta_MEAN"] - row["shift"] * row["k"] / row["M"]) for row in rows); responsive_max = any(row["delta_MAX"] > 0 for row in rows if row["k"] == 1)
    return {"schema": "gnss-doppler-lab.qset-stage0a-r2-synthetic-dilution.v1", "status": "PASS" if mean_error < 1e-12 and responsive_max else "FAIL", "claimed_as_detection_evidence": False, "runs": len(rows), "maximum_mean_k_over_m_error": mean_error, "max_k1_responsive": responsive_max, "summary": {key: float(np.mean([row[f'delta_{key}'] for row in rows])) for key in AGGREGATORS}}


def run_clean() -> dict[str, Any]:
    receiver_build = build_receiver(); manifests = {}
    for name in ("C-1", "C-3"):
        replay = replay_scenario(name, scenario_path(name), receiver_build); manifests[name] = replay
        rows = extract_window_features(SSD_ROOT / "replays" / name / "receiver", name, SCENARIOS[name]["size"] / BYTES_PER_COMPLEX / RAW_FS); require(rows, f"no feature rows for {name}"); save_feature_cache(name, rows)
    clean = analyze_clean(); synthetic = synthetic_dilution(clean); require(clean["status"] == "PASS" and synthetic["status"] == "PASS", "clean freeze gates failed")
    return {"receiver_build": receiver_build, "replays": manifests, "clean": clean, "synthetic": synthetic}

