"""Pre-score Galileo E1 TRACE/VEML loop-preservation repair.

The frozen nine TRACE taps remain the first nine correlators.  Galileo VEML
uses four additional, loop-only correlators at the receiver configuration's
original VE/E/L/VL spacings, so neither the exported scientific observation
nor the receiver discriminator geometry is changed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any

from .qset_stage0a_r2_execution_repair_v5 import *  # noqa: F401,F403
from . import qset_stage0a_r2 as core
from . import qset_stage0a_r2_evaluation as evaluation
from . import qset_stage0a_r2_execution as execution

ORIGINAL_SOURCE = RECEIVER_SOURCE
ORIGINAL_RECEIVER = RECEIVER
V6_SOURCE = SSD_ROOT / "receiver-source-v6"
V6_BUILD = SSD_ROOT / "receiver-build-v6"
V6_RECEIVER = V6_BUILD / "src/main/gnss-sdr"
V6_PATCH = ROOT / "patches/qset_galileo_e1_trace9_veml_preservation_repair.patch"
V6_PATCH_SHA256 = "5cc1ac60c10f510f447f848a5151e0a4f007121dd11d4a885f1f9f8d2fd95bc0"
V6_REPAIR_PATHS = (
    "src/gnss_doppler_lab/qset_stage0a_r2_execution_repair_v6.py",
    "scripts/run_qset_gnss_stage0a_r2_repaired_v6.py",
    "tests/test_qset_gnss_stage0a_r2_execution_repair_v6.py",
    "patches/qset_galileo_e1_trace9_veml_preservation_repair.patch",
)


def activate_v6_receiver_paths() -> None:
    for module in (core, execution, evaluation):
        module.RECEIVER_SOURCE = V6_SOURCE
        module.RECEIVER_BUILD = V6_BUILD
        module.RECEIVER = V6_RECEIVER


def _source_diff_v6(source: Path) -> bytes:
    return subprocess.run(
        ["git", "-c", f"safe.directory={source}", "-C", str(source), "diff", "--binary", "--no-ext-diff", BASE_RECEIVER_COMMIT],
        capture_output=True,
        check=True,
    ).stdout


def abandoned_v5_audit() -> dict[str, Any]:
    """Bind the stopped, never-executed pointer-only proposal without deleting it."""
    require(V5_SOURCE.is_dir() and V5_BUILD.is_dir(), "expected preserved v5 partial build absent")
    require(not (SSD_ROOT / "receiver_build_manifest_v5.json").exists(), "unexpected completed v5 manifest")
    require(not V5_RECEIVER.exists(), "unexpected executable from abandoned v5 build")
    return {
        "status": "ABANDONED_PRE_EXECUTION_LOOP_SPACING_MISMATCH",
        "source_path": str(V5_SOURCE),
        "build_path": str(V5_BUILD),
        "source_diff_sha256": hashlib.sha256(_source_diff_v6(V5_SOURCE)).hexdigest(),
        "proposal_patch_sha256": sha256_file(V5_PATCH),
        "receiver_executable_created": False,
        "receiver_executed": False,
        "clean_score_computed": False,
        "attack_accessed": False,
        "reason": "pointer-only mapping would have replaced frozen Galileo VEML loop spacings with outer TRACE tap spacing",
    }


def build_receiver_repaired_v6() -> dict[str, Any]:
    runtime = bind_gnuradio_runtime()
    activate_v6_receiver_paths()
    manifest_path = SSD_ROOT / "receiver_build_manifest_v6.json"
    if manifest_path.exists():
        prior = read_json(manifest_path)
        require(V6_RECEIVER.is_file() and sha256_file(V6_RECEIVER) == prior["receiver_sha256"], "cached v6 receiver drift")
        return prior
    require(not V6_SOURCE.exists() and not V6_BUILD.exists(), "partial v6 receiver build exists; refusing overwrite")
    prior_manifest = read_json(SSD_ROOT / "receiver_build_manifest.json")
    require(sha256_file(ORIGINAL_RECEIVER) == prior_manifest["receiver_sha256"], "preserved pre-v6 receiver drift")
    require(hashlib.sha256(_source_diff_v6(ORIGINAL_SOURCE)).hexdigest() == prior_manifest["combined_source_diff_sha256"], "pre-v6 source diff drift")
    require(sha256_file(V6_PATCH) == V6_PATCH_SHA256, "v6 patch drift")
    abandoned = abandoned_v5_audit()
    subprocess.run(["cp", "-a", "--reflink=auto", str(ORIGINAL_SOURCE), str(V6_SOURCE)], check=True)
    subprocess.run(["git", "-c", f"safe.directory={V6_SOURCE}", "-C", str(V6_SOURCE), "apply", str(V6_PATCH)], check=True)
    subprocess.run(["git", "-c", f"safe.directory={V6_SOURCE}", "-C", str(V6_SOURCE), "diff", "--check"], check=True)
    configure = [
        "cmake", "-S", str(V6_SOURCE), "-B", str(V6_BUILD), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={SSD_ROOT / 'receiver-install-v6'}", "-DPYTHON_EXECUTABLE=/usr/bin/python3",
        "-DPython3_EXECUTABLE=/usr/bin/python3", "-DENABLE_LOG=ON", "-DENABLE_UHD=ON", "-DENABLE_ZMQ=ON",
        "-DENABLE_UNIT_TESTING=ON", "-DENABLE_UNIT_TESTING_EXTRA=OFF", "-DENABLE_SYSTEM_TESTING=OFF", "-DENABLE_SYSTEM_TESTING_EXTRA=OFF",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SYSTEM_DIST_PACKAGES)
    environment["GIT_CONFIG_COUNT"] = "1"
    environment["GIT_CONFIG_KEY_0"] = "safe.directory"
    environment["GIT_CONFIG_VALUE_0"] = "*"
    subprocess.run(configure, check=True, env=environment)
    subprocess.run(["cmake", "--build", str(V6_BUILD), "--target", "gnss-sdr", "-j", "12"], check=True, env=environment)
    require(V6_RECEIVER.is_file(), "v6 receiver build produced no executable")
    version = subprocess.run([str(V6_RECEIVER), "--version"], text=True, capture_output=True, check=True)
    result = {
        "schema": "gnss-doppler-lab.qset-stage0a-r2-receiver-build-v6.v1",
        "status": "PASS",
        "runtime": runtime,
        "receiver_base_commit": BASE_RECEIVER_COMMIT,
        "r2c_patch_sha256": R2C_PATCH_SHA256,
        "galileo_patch_sha256": sha256_file(GALILEO_PATCH),
        "veml_trace_loop_preservation_patch_sha256": sha256_file(V6_PATCH),
        "previous_receiver_sha256": prior_manifest["receiver_sha256"],
        "combined_source_diff_sha256": hashlib.sha256(_source_diff_v6(V6_SOURCE)).hexdigest(),
        "receiver_path": str(V6_RECEIVER),
        "receiver_size_bytes": V6_RECEIVER.stat().st_size,
        "receiver_sha256": sha256_file(V6_RECEIVER),
        "version": (version.stdout + version.stderr).strip(),
        "configure_command": configure,
        "build_command": ["cmake", "--build", str(V6_BUILD), "--target", "gnss-sdr", "-j", "12"],
        "scientific_changes": False,
        "receiver_source_changes": True,
        "receiver_loop_changes": False,
        "trace_taps": 9,
        "galileo_veml_loop_only_taps": 4,
        "repair_scope": "retain frozen nine TRACE correlators and add four non-serialized Galileo VEML correlators at the existing configured loop spacings",
        "abandoned_v5": abandoned,
    }
    write_json(manifest_path, result)
    return result


def clean_execution_repaired_v6() -> dict[str, Any]:
    runtime = bind_gnuradio_runtime()
    activate_v6_receiver_paths()
    failed = SSD_ROOT / "replays" / "C-1"
    preserved = SSD_ROOT / "historical-incomplete" / "C-1-galileo-veml-null-pointer"
    log_path = SSD_ROOT / "clean_execution_v4_stdout.txt"
    if failed.exists():
        segfault_preservation = preserve_segfault_attempt(failed, preserved, log_path)
    else:
        require((preserved / "attempt_preservation.json").is_file(), "expected preserved segfault attempt absent")
        segfault_preservation = read_json(preserved / "attempt_preservation.json")
        require(segfault_preservation["status"] == "PRESERVED_PRE_SCORE_RECEIVER_SIGSEGV_ATTEMPT", "preserved segfault evidence drift")
    empty_preservation = read_json(SSD_ROOT / "historical-incomplete" / "C-1-missing-gnuradio-before-decoder" / "attempt_preservation.json")
    receiver_build = build_receiver_repaired_v6()
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


def freeze_clean_artifacts_repaired_v6(result: dict[str, Any]) -> None:
    freeze_clean_artifacts_repaired_v4(result)
    freeze = read_json(ARTIFACT / "execution_freeze.json")
    freeze["code_bindings"].update({relative: sha256_file(ROOT / relative) for relative in (*V5_REPAIR_PATHS, *V6_REPAIR_PATHS)})
    freeze["prefreeze_engineering_repair_v5"] = result["receiver_build"]["abandoned_v5"]
    freeze["prefreeze_engineering_repair_v6"] = {
        "trigger": "Galileo E1 tracking SIGSEGV at first VEML accumulator dereference with frozen nine-tap TRACE",
        "stage": "after C-1 decoding and receiver start, before manifest completion, feature extraction, or any clean score",
        "root_cause": result["preserved_segfault_attempt"]["root_cause"],
        "resolution": result["receiver_build"]["repair_scope"],
        "trace_contract": "first nine correlators remain frozen TRACE taps; four extra correlators are loop-only and never serialized",
        "loop_contract": "Galileo VE/E/L/VL retain the preregistered receiver configuration spacings, including narrow-mode transition",
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
    audit["abandoned_pointer_only_build_v5"] = result["receiver_build"]["abandoned_v5"]
    write_json(ARTIFACT / "access_audit.json", audit)
