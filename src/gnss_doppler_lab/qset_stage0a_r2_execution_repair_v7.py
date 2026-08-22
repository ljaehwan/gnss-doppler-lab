"""Pre-score exact-boundary terminal-drain repair for Q-SET R2."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any

from .qset_stage0a_r2_execution_repair_v6 import *  # noqa: F401,F403
from . import qset_stage0a_r2 as core
from . import qset_stage0a_r2_evaluation as evaluation
from . import qset_stage0a_r2_execution as execution

V7_SOURCE = SSD_ROOT / "receiver-source-v7"
V7_BUILD = SSD_ROOT / "receiver-build-v7"
V7_RECEIVER = V7_BUILD / "src/main/gnss-sdr"
V7_PATCH = ROOT / "patches/qset_terminal_drain_exact_boundary_repair.patch"
V7_PATCH_SHA256 = "5f1208df9d44a41048b8e01b6e69f28f48c99203316df8d112649890c1074c56"
V7_REPAIR_PATHS = (
    "src/gnss_doppler_lab/qset_stage0a_r2_execution_repair_v7.py",
    "scripts/run_qset_gnss_stage0a_r2_repaired_v7.py",
    "tests/test_qset_gnss_stage0a_r2_execution_repair_v7.py",
    "patches/qset_terminal_drain_exact_boundary_repair.patch",
)


def activate_v7_receiver_paths() -> None:
    for module in (core, execution, evaluation):
        module.RECEIVER_SOURCE = V7_SOURCE
        module.RECEIVER_BUILD = V7_BUILD
        module.RECEIVER = V7_RECEIVER


def _source_diff_v7(source: Path) -> bytes:
    return subprocess.run(
        ["git", "-c", f"safe.directory={source}", "-C", str(source), "diff", "--binary", "--no-ext-diff", BASE_RECEIVER_COMMIT],
        capture_output=True,
        check=True,
    ).stdout


def build_receiver_repaired_v7() -> dict[str, Any]:
    runtime = bind_gnuradio_runtime()
    activate_v7_receiver_paths()
    manifest_path = SSD_ROOT / "receiver_build_manifest_v7.json"
    if manifest_path.exists():
        prior = read_json(manifest_path)
        require(V7_RECEIVER.is_file() and sha256_file(V7_RECEIVER) == prior["receiver_sha256"], "cached v7 receiver drift")
        return prior
    require(not V7_SOURCE.exists() and not V7_BUILD.exists(), "partial v7 receiver build exists; refusing overwrite")
    prior_manifest = read_json(SSD_ROOT / "receiver_build_manifest_v6.json")
    require(V6_RECEIVER.is_file() and sha256_file(V6_RECEIVER) == prior_manifest["receiver_sha256"], "preserved v6 receiver drift")
    require(hashlib.sha256(_source_diff_v7(V6_SOURCE)).hexdigest() == prior_manifest["combined_source_diff_sha256"], "v6 source diff drift")
    require(sha256_file(V7_PATCH) == V7_PATCH_SHA256, "v7 patch drift")
    subprocess.run(["cp", "-a", "--reflink=auto", str(V6_SOURCE), str(V7_SOURCE)], check=True)
    subprocess.run(["git", "-c", f"safe.directory={V7_SOURCE}", "-C", str(V7_SOURCE), "apply", str(V7_PATCH)], check=True)
    subprocess.run(["git", "-c", f"safe.directory={V7_SOURCE}", "-C", str(V7_SOURCE), "diff", "--check"], check=True)
    configure = [
        "cmake", "-S", str(V7_SOURCE), "-B", str(V7_BUILD), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={SSD_ROOT / 'receiver-install-v7'}", "-DPYTHON_EXECUTABLE=/usr/bin/python3",
        "-DPython3_EXECUTABLE=/usr/bin/python3", "-DENABLE_LOG=ON", "-DENABLE_UHD=ON", "-DENABLE_ZMQ=ON",
        "-DENABLE_UNIT_TESTING=ON", "-DENABLE_UNIT_TESTING_EXTRA=OFF", "-DENABLE_SYSTEM_TESTING=OFF", "-DENABLE_SYSTEM_TESTING_EXTRA=OFF",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SYSTEM_DIST_PACKAGES)
    environment["GIT_CONFIG_COUNT"] = "1"
    environment["GIT_CONFIG_KEY_0"] = "safe.directory"
    environment["GIT_CONFIG_VALUE_0"] = "*"
    subprocess.run(configure, check=True, env=environment)
    subprocess.run(["cmake", "--build", str(V7_BUILD), "--target", "gnss-sdr", "-j", "12"], check=True, env=environment)
    require(V7_RECEIVER.is_file(), "v7 receiver build produced no executable")
    version = subprocess.run([str(V7_RECEIVER), "--version"], text=True, capture_output=True, check=True)
    result = {
        "schema": "gnss-doppler-lab.qset-stage0a-r2-receiver-build-v7.v1",
        "status": "PASS",
        "runtime": runtime,
        "receiver_base_commit": BASE_RECEIVER_COMMIT,
        "r2c_patch_sha256": R2C_PATCH_SHA256,
        "galileo_patch_sha256": sha256_file(GALILEO_PATCH),
        "veml_trace_loop_preservation_patch_sha256": sha256_file(V6_PATCH),
        "terminal_drain_exact_boundary_patch_sha256": sha256_file(V7_PATCH),
        "previous_receiver_sha256": prior_manifest["receiver_sha256"],
        "combined_source_diff_sha256": hashlib.sha256(_source_diff_v7(V7_SOURCE)).hexdigest(),
        "receiver_path": str(V7_RECEIVER),
        "receiver_size_bytes": V7_RECEIVER.stat().st_size,
        "receiver_sha256": sha256_file(V7_RECEIVER),
        "version": (version.stdout + version.stderr).strip(),
        "configure_command": configure,
        "build_command": ["cmake", "--build", str(V7_BUILD), "--target", "gnss-sdr", "-j", "12"],
        "scientific_changes": False,
        "receiver_source_changes": True,
        "receiver_loop_changes": False,
        "final_sample_forwarded": True,
        "repair_scope": "queue terminal DRAIN after copying the exact-boundary final chunk while returning its full item count to downstream blocks",
        "abandoned_v5": prior_manifest["abandoned_v5"],
    }
    write_json(manifest_path, result)
    return result


def preserve_exact_boundary_attempt(
    root: Path,
    preserved: Path,
    wrapper_log: Path,
    expected_decoder_size: int = 4_799_972_848,
) -> dict[str, Any]:
    require(root.is_dir() and not preserved.exists(), "exact-boundary preservation precondition")
    decoder = root / "decoded_4msps_gr_complex.bin"
    receiver_dir = root / "receiver"
    receiver_log = receiver_dir / "receiver.log"
    traces = sorted(receiver_dir.glob("trace_native_1ms_ch_*.bin"))
    require(decoder.is_file() and decoder.stat().st_size == expected_decoder_size, "exact-boundary decoder evidence incomplete")
    require(receiver_log.is_file() and len(traces) == TRACE_CHANNELS, "exact-boundary receiver evidence incomplete")
    text = receiver_log.read_text(encoding="utf-8", errors="replace")
    require("Current receiver time: 2 min 30 s" in text, "receiver did not reach frozen input boundary")
    require("Draining receiver" not in text and "Received action DRAIN" not in text, "attempt unexpectedly terminal-drained")
    require("Received action STOP" in text and "Flowgraph stopped" in text, "operator stop evidence absent")
    require(wrapper_log.is_file() and "KeyboardInterrupt" in wrapper_log.read_text(encoding="utf-8", errors="replace"), "wrapper interruption evidence absent")
    output_set = output_manifest(root)
    wrapper_binding = {"path": str(wrapper_log), "size_bytes": wrapper_log.stat().st_size, "sha256": sha256_file(wrapper_log)}
    preserved.parent.mkdir(parents=True, exist_ok=True)
    root.rename(preserved)
    result = {
        "schema": "gnss-doppler-lab.qset-stage0a-r2-exact-boundary-attempt-preservation.v1",
        "status": "PRESERVED_PRE_SCORE_EXACT_BOUNDARY_DRAIN_DEADLOCK",
        "original_path": str(root),
        "preserved_path": str(preserved),
        "output_set": output_set,
        "wrapper_log": wrapper_binding,
        "receiver_exit_code": "OPERATOR_SIGINT_AFTER_BOUNDED_EOF_DEADLOCK",
        "receiver_reached_final_sample": True,
        "terminal_drain": False,
        "feature_rows_computed": 0,
        "clean_score_computed": False,
        "attack_accessed": False,
        "root_cause": "valve queued DRAIN only on a later work call that cannot occur after exact upstream EOF",
    }
    write_json(preserved / "attempt_preservation.json", result)
    return result


def clean_execution_repaired_v7() -> dict[str, Any]:
    runtime = bind_gnuradio_runtime()
    activate_v7_receiver_paths()
    failed = SSD_ROOT / "replays" / "C-1"
    preserved = SSD_ROOT / "historical-incomplete" / "C-1-terminal-drain-exact-boundary"
    wrapper_log = SSD_ROOT / "clean_execution_v6_stdout.txt"
    if failed.exists() and not (failed / "manifest.json").is_file():
        boundary_preservation = preserve_exact_boundary_attempt(failed, preserved, wrapper_log)
    else:
        require((preserved / "attempt_preservation.json").is_file(), "expected preserved boundary attempt absent")
        boundary_preservation = read_json(preserved / "attempt_preservation.json")
        require(boundary_preservation["status"] == "PRESERVED_PRE_SCORE_EXACT_BOUNDARY_DRAIN_DEADLOCK", "boundary preservation drift")
    empty_preservation = read_json(SSD_ROOT / "historical-incomplete" / "C-1-missing-gnuradio-before-decoder" / "attempt_preservation.json")
    segfault_preservation = read_json(SSD_ROOT / "historical-incomplete" / "C-1-galileo-veml-null-pointer" / "attempt_preservation.json")
    receiver_build = build_receiver_repaired_v7()
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
        "preserved_boundary_attempt": boundary_preservation,
    }


def freeze_clean_artifacts_repaired_v7(result: dict[str, Any]) -> None:
    freeze_clean_artifacts_repaired_v4(result)
    freeze = read_json(ARTIFACT / "execution_freeze.json")
    freeze["code_bindings"].update({relative: sha256_file(ROOT / relative) for relative in (*V5_REPAIR_PATHS, *V6_REPAIR_PATHS, *V7_REPAIR_PATHS)})
    freeze["prefreeze_engineering_repair_v5"] = result["receiver_build"]["abandoned_v5"]
    freeze["prefreeze_engineering_repair_v6"] = {
        "trigger": "Galileo E1 VEML null dereference under frozen nine-tap TRACE",
        "resolution": "nine serialized TRACE taps plus four loop-only taps at original configured VEML spacings",
        "preserved_attempt": result["preserved_segfault_attempt"],
        "scientific_changes": False,
        "receiver_loop_changes": False,
        "attack_accessed": False,
    }
    freeze["prefreeze_engineering_repair_v7"] = {
        "trigger": "exact-boundary EOF prevented the later valve work call that formerly queued terminal DRAIN",
        "stage": "after complete C-1 receiver sample processing, before replay manifest, feature extraction, or any clean score",
        "resolution": result["receiver_build"]["repair_scope"],
        "preserved_attempt": result["preserved_boundary_attempt"],
        "scientific_changes": False,
        "receiver_loop_changes": False,
        "configuration_changes": False,
        "final_sample_forwarded": True,
        "attack_accessed": False,
    }
    write_json(ARTIFACT / "execution_freeze.json", freeze)
    audit = read_json(ARTIFACT / "access_audit.json")
    audit["prefreeze_failed_clean_attempt_v5"] = {
        "scenario": "C-1", "identity_hash_bytes": SCENARIOS["C-1"]["size"], "decoder_bytes_read": SCENARIOS["C-1"]["size"],
        "receiver_runs": 1, "feature_windows": 0, "scores": 0, "attack_bytes_read": 0, "preserved": True, "failure": "SIGSEGV",
    }
    audit["prefreeze_failed_clean_attempt_v6"] = {
        "scenario": "C-1", "identity_hash_bytes": SCENARIOS["C-1"]["size"], "decoder_bytes_read": SCENARIOS["C-1"]["size"],
        "receiver_runs": 1, "feature_windows": 0, "scores": 0, "attack_bytes_read": 0, "preserved": True,
        "failure": "EXACT_BOUNDARY_TERMINAL_DRAIN_DEADLOCK",
    }
    audit["abandoned_pointer_only_build_v5"] = result["receiver_build"]["abandoned_v5"]
    write_json(ARTIFACT / "access_audit.json", audit)
