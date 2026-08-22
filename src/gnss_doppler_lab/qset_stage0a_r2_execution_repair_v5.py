"""Pre-score Galileo VEML nine-tap pointer and empty TRACE repairs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any

from .qset_stage0a_r2_execution_repair_v4 import *  # noqa: F401,F403
from . import qset_stage0a_r2 as core
from . import qset_stage0a_r2_evaluation as evaluation
from . import qset_stage0a_r2_execution as execution

V5_SOURCE = SSD_ROOT / "receiver-source-v5"
V5_BUILD = SSD_ROOT / "receiver-build-v5"
V5_RECEIVER = V5_BUILD / "src/main/gnss-sdr"
V5_PATCH = ROOT / "patches/qset_galileo_e1_veml_9tap_pointer_repair.patch"
V5_REPAIR_PATHS = (
    "src/gnss_doppler_lab/qset_stage0a_r2_execution_repair_v5.py",
    "scripts/run_qset_gnss_stage0a_r2_repaired_v5.py",
    "tests/test_qset_gnss_stage0a_r2_execution_repair_v5.py",
    "patches/qset_galileo_e1_veml_9tap_pointer_repair.patch",
)


def activate_v5_receiver_paths() -> None:
    for module in (core, execution, evaluation):
        module.RECEIVER_SOURCE = V5_SOURCE
        module.RECEIVER_BUILD = V5_BUILD
        module.RECEIVER = V5_RECEIVER


def _source_diff(source: Path) -> bytes:
    return subprocess.run(
        ["git", "-c", f"safe.directory={source}", "-C", str(source), "diff", "--binary", "--no-ext-diff", BASE_RECEIVER_COMMIT],
        capture_output=True,
        check=True,
    ).stdout


def build_receiver_repaired_v5() -> dict[str, Any]:
    runtime = bind_gnuradio_runtime()
    activate_v5_receiver_paths()
    manifest_path = SSD_ROOT / "receiver_build_manifest_v5.json"
    if manifest_path.exists():
        prior = read_json(manifest_path)
        require(V5_RECEIVER.is_file() and sha256_file(V5_RECEIVER) == prior["receiver_sha256"], "cached v5 receiver drift")
        return prior
    require(not V5_SOURCE.exists() and not V5_BUILD.exists(), "partial v5 receiver build exists; refusing overwrite")
    prior_manifest = read_json(SSD_ROOT / "receiver_build_manifest.json")
    require(sha256_file(RECEIVER) == prior_manifest["receiver_sha256"], "preserved pre-v5 receiver drift")
    require(hashlib.sha256(_source_diff(RECEIVER_SOURCE)).hexdigest() == prior_manifest["combined_source_diff_sha256"], "pre-v5 source diff drift")
    require(sha256_file(V5_PATCH) == "5680ed7984edce60d4e8a7296b6450ca0a9ea767dfbd218b32eea4d1726663c0", "v5 patch drift")
    subprocess.run(["cp", "-a", "--reflink=auto", str(RECEIVER_SOURCE), str(V5_SOURCE)], check=True)
    subprocess.run(["git", "-c", f"safe.directory={V5_SOURCE}", "-C", str(V5_SOURCE), "apply", str(V5_PATCH)], check=True)
    subprocess.run(["git", "-c", f"safe.directory={V5_SOURCE}", "-C", str(V5_SOURCE), "diff", "--check"], check=True)
    configure = [
        "cmake", "-S", str(V5_SOURCE), "-B", str(V5_BUILD), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={SSD_ROOT / 'receiver-install-v5'}", "-DPYTHON_EXECUTABLE=/usr/bin/python3",
        "-DPython3_EXECUTABLE=/usr/bin/python3", "-DENABLE_LOG=ON", "-DENABLE_UHD=ON", "-DENABLE_ZMQ=ON",
        "-DENABLE_UNIT_TESTING=ON", "-DENABLE_UNIT_TESTING_EXTRA=OFF", "-DENABLE_SYSTEM_TESTING=OFF", "-DENABLE_SYSTEM_TESTING_EXTRA=OFF",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SYSTEM_DIST_PACKAGES)
    environment["GIT_CONFIG_COUNT"] = "1"
    environment["GIT_CONFIG_KEY_0"] = "safe.directory"
    environment["GIT_CONFIG_VALUE_0"] = "*"
    subprocess.run(configure, check=True, env=environment)
    subprocess.run(["cmake", "--build", str(V5_BUILD), "--target", "gnss-sdr", "-j", "12"], check=True, env=environment)
    require(V5_RECEIVER.is_file(), "v5 receiver build produced no executable")
    version = subprocess.run([str(V5_RECEIVER), "--version"], text=True, capture_output=True, check=True)
    result = {
        "schema": "gnss-doppler-lab.qset-stage0a-r2-receiver-build-v5.v1",
        "status": "PASS",
        "runtime": runtime,
        "receiver_base_commit": BASE_RECEIVER_COMMIT,
        "r2c_patch_sha256": R2C_PATCH_SHA256,
        "galileo_patch_sha256": sha256_file(GALILEO_PATCH),
        "veml_pointer_repair_patch_sha256": sha256_file(V5_PATCH),
        "previous_receiver_sha256": prior_manifest["receiver_sha256"],
        "combined_source_diff_sha256": hashlib.sha256(_source_diff(V5_SOURCE)).hexdigest(),
        "receiver_path": str(V5_RECEIVER),
        "receiver_size_bytes": V5_RECEIVER.stat().st_size,
        "receiver_sha256": sha256_file(V5_RECEIVER),
        "version": (version.stdout + version.stderr).strip(),
        "configure_command": configure,
        "build_command": ["cmake", "--build", str(V5_BUILD), "--target", "gnss-sdr", "-j", "12"],
        "scientific_changes": False,
        "receiver_source_changes": True,
        "receiver_loop_changes": False,
        "repair_scope": "bind frozen outer +/-0.5-chip taps to the pre-existing Galileo VEML pointers; no loop, correlator, tap, threshold, feature, or gate change",
    }
    write_json(manifest_path, result)
    return result


def preserve_segfault_attempt(root: Path, preserved: Path, log_path: Path, expected_decoder_size: int = 4_799_972_848) -> dict[str, Any]:
    require(root.is_dir(), "segfault attempt root absent")
    decoder = root / "decoded_4msps_gr_complex.bin"
    receiver_dir = root / "receiver"
    receiver_log = receiver_dir / "receiver.log"
    traces = sorted(receiver_dir.glob("trace_native_1ms_ch_*.bin"))
    require(decoder.is_file() and decoder.stat().st_size == expected_decoder_size, "segfault attempt decoder evidence incomplete")
    require(receiver_log.is_file() and len(traces) == TRACE_CHANNELS, "segfault attempt receiver evidence incomplete")
    require(all(path.stat().st_size == 0 for path in traces), "segfault attempt expected all TRACE channels empty")
    text = receiver_log.read_text(encoding="utf-8", errors="replace")
    require("Flowgraph started" in text and "Draining receiver" not in text and "Received action DRAIN" not in text, "segfault attempt terminal evidence drift")
    require(log_path.is_file() and not preserved.exists(), "segfault attempt preservation precondition")
    output_set = output_manifest(root)
    log_binding = {"path": str(log_path), "size_bytes": log_path.stat().st_size, "sha256": sha256_file(log_path)}
    preserved.parent.mkdir(parents=True, exist_ok=True)
    root.rename(preserved)
    result = {
        "schema": "gnss-doppler-lab.qset-stage0a-r2-segfault-attempt-preservation.v1",
        "status": "PRESERVED_PRE_SCORE_RECEIVER_SIGSEGV_ATTEMPT",
        "original_path": str(root),
        "preserved_path": str(preserved),
        "output_set": output_set,
        "failure_log": log_binding,
        "receiver_exit_code": "UNRECOVERED_EXCEPTION_MASKED_RETURN_CODE",
        "kernel_evidence": {"signal": "SIGSEGV", "pid": 694093, "timestamp": "2026-08-22T23:09:24+09:00", "function": "dll_pll_veml_tracking::general_work", "instruction_offset": "0x396015"},
        "root_cause": "trace_9tap left d_Very_Early/d_Very_Late null while Galileo E1 d_veml dereferenced them",
        "terminal_drain": False,
        "score_computed": False,
        "attack_accessed": False,
    }
    write_json(preserved / "attempt_preservation.json", result)
    return result


def clean_execution_repaired_v5() -> dict[str, Any]:
    runtime = bind_gnuradio_runtime()
    activate_v5_receiver_paths()
    failed = SSD_ROOT / "replays" / "C-1"
    preserved = SSD_ROOT / "historical-incomplete" / "C-1-galileo-veml-null-pointer"
    log_path = SSD_ROOT / "clean_execution_v4_stdout.txt"
    if failed.exists():
        segfault_preservation = preserve_segfault_attempt(failed, preserved, log_path)
    else:
        require((preserved / "attempt_preservation.json").is_file(), "expected preserved segfault attempt absent")
        segfault_preservation = read_json(preserved / "attempt_preservation.json")
    empty_preservation = read_json(SSD_ROOT / "historical-incomplete" / "C-1-missing-gnuradio-before-decoder" / "attempt_preservation.json")
    receiver_build = build_receiver_repaired_v5()
    manifests: dict[str, Any] = {}
    for name in ("C-1", "C-3"):
        replay = replay_scenario(name, scenario_path(name), receiver_build)
        manifests[name] = replay
        rows = extract_window_features(SSD_ROOT / "replays" / name / "receiver", name, SCENARIOS[name]["size"] / BYTES_PER_COMPLEX / RAW_FS)
        require(rows, f"no feature rows for {name}")
        save_feature_cache(name, rows)
    clean = analyze_clean()
    synthetic = synthetic_dilution(clean)
    require(synthetic["status"] == "PASS", "synthetic implementation sanity failed")
    return {
        "receiver_build": receiver_build,
        "replays": manifests,
        "clean": clean,
        "synthetic": synthetic,
        "gnuradio_runtime_repair": runtime,
        "preserved_failed_attempt": empty_preservation,
        "preserved_segfault_attempt": segfault_preservation,
    }


def freeze_clean_artifacts_repaired_v5(result: dict[str, Any]) -> None:
    freeze_clean_artifacts_repaired_v4(result)
    freeze = read_json(ARTIFACT / "execution_freeze.json")
    freeze["code_bindings"].update({relative: sha256_file(ROOT / relative) for relative in V5_REPAIR_PATHS})
    freeze["prefreeze_engineering_repair_v5"] = {
        "trigger": "Galileo E1 tracking SIGSEGV at the first VEML accumulator dereference with frozen nine-tap TRACE",
        "stage": "after C-1 decoder and receiver start, before manifest completion, feature extraction, or score",
        "root_cause": result["preserved_segfault_attempt"]["root_cause"],
        "resolution": result["receiver_build"]["repair_scope"],
        "empty_trace_adapter": "zero-byte existing channels are SHA-bound as EMPTY_OPTIONAL_CHANNEL and excluded from support; missing channels fail closed",
        "preserved_attempt": result["preserved_segfault_attempt"],
        "scientific_changes": False,
        "receiver_source_changes": True,
        "receiver_loop_changes": False,
        "attack_accessed": False,
    }
    write_json(ARTIFACT / "execution_freeze.json", freeze)
    audit = read_json(ARTIFACT / "access_audit.json")
    audit["prefreeze_failed_clean_attempt_v5"] = {
        "scenario": "C-1",
        "identity_hash_bytes": SCENARIOS["C-1"]["size"],
        "decoder_bytes_read": SCENARIOS["C-1"]["size"],
        "receiver_runs": 1,
        "feature_windows": 0,
        "scores": 0,
        "attack_bytes_read": 0,
        "preserved": True,
        "failure": "SIGSEGV",
    }
    write_json(ARTIFACT / "access_audit.json", audit)
