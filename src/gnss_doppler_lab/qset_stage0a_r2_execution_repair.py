"""Pre-result repair for the worktree-local venv path used by receiver CMake."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

from .qset_stage0a_r2_evaluation import *  # noqa: F401,F403

SHARED_PYTHON = Path("/home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python")
REPAIR_PATHS = (
    "src/gnss_doppler_lab/qset_stage0a_r2_execution_repair.py",
    "scripts/run_qset_gnss_stage0a_r2_repaired.py",
)


def build_receiver_repaired() -> dict[str, Any]:
    SSD_ROOT.mkdir(parents=True, exist_ok=True); manifest_path = SSD_ROOT / "receiver_build_manifest.json"
    if manifest_path.exists():
        prior = read_json(manifest_path); require(RECEIVER.is_file() and sha256_file(RECEIVER) == prior["receiver_sha256"], "cached receiver drift"); return prior
    require(RECEIVER_SOURCE.is_dir(), "preserved pre-result receiver source absent")
    require(SHARED_PYTHON.is_file(), "shared project venv absent")
    head = subprocess.run(["git", "-c", f"safe.directory={RECEIVER_SOURCE}", "-C", str(RECEIVER_SOURCE), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    require(head == BASE_RECEIVER_COMMIT, "preserved source base drift")
    source_text = (RECEIVER_SOURCE / "src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc").read_text(encoding="utf-8")
    require("pilot-tracked Galileo 1B" in source_text and "supported_trace_signal" in source_text, "Galileo patch absent from preserved source")
    subprocess.run(["git", "-c", f"safe.directory={RECEIVER_SOURCE}", "-C", str(RECEIVER_SOURCE), "diff", "--check"], check=True)
    configure = ["cmake", "-S", str(RECEIVER_SOURCE), "-B", str(RECEIVER_BUILD), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={SSD_ROOT / 'receiver-install'}", f"-DPYTHON_EXECUTABLE={SHARED_PYTHON}", f"-DPython3_EXECUTABLE={SHARED_PYTHON}", "-DENABLE_LOG=ON", "-DENABLE_UHD=ON", "-DENABLE_ZMQ=ON", "-DENABLE_UNIT_TESTING=ON", "-DENABLE_UNIT_TESTING_EXTRA=OFF", "-DENABLE_SYSTEM_TESTING=OFF", "-DENABLE_SYSTEM_TESTING_EXTRA=OFF"]
    environment = dict(os.environ); environment["PYTHONPATH"] = "/usr/lib/python3/dist-packages"
    subprocess.run(configure, check=True, env=environment); subprocess.run(["cmake", "--build", str(RECEIVER_BUILD), "--target", "gnss-sdr", "-j", "12"], check=True, env=environment)
    require(RECEIVER.is_file(), "receiver build produced no executable")
    version = subprocess.run([str(RECEIVER), "--version"], text=True, capture_output=True, check=True)
    diff = subprocess.run(["git", "-c", f"safe.directory={RECEIVER_SOURCE}", "-C", str(RECEIVER_SOURCE), "diff", "--binary", "--no-ext-diff", BASE_RECEIVER_COMMIT], capture_output=True, check=True).stdout
    result = {"schema": "gnss-doppler-lab.qset-stage0a-r2-receiver-build.v1", "status": "PASS", "receiver_base_commit": BASE_RECEIVER_COMMIT, "r2c_patch_sha256": R2C_PATCH_SHA256, "galileo_patch_sha256": sha256_file(GALILEO_PATCH), "combined_source_diff_sha256": hashlib.sha256(diff).hexdigest(), "receiver_path": str(RECEIVER), "receiver_size_bytes": RECEIVER.stat().st_size, "receiver_sha256": sha256_file(RECEIVER), "version": (version.stdout + version.stderr).strip(), "configure_command": configure, "build_command": ["cmake", "--build", str(RECEIVER_BUILD), "--target", "gnss-sdr", "-j", "12"], "prefreeze_engineering_repair": "resume preserved configure directory with absolute shared venv path; no receiver/scientific change"}
    write_json(manifest_path, result); return result


def clean_execution_repaired() -> dict[str, Any]:
    receiver_build = build_receiver_repaired(); manifests = {}
    for name in ("C-1", "C-3"):
        replay = replay_scenario(name, scenario_path(name), receiver_build); manifests[name] = replay
        rows = extract_window_features(SSD_ROOT / "replays" / name / "receiver", name, SCENARIOS[name]["size"] / BYTES_PER_COMPLEX / RAW_FS); require(rows, f"no feature rows for {name}"); save_feature_cache(name, rows)
    clean = analyze_clean(); synthetic = synthetic_dilution(clean); require(synthetic["status"] == "PASS", "synthetic implementation sanity failed")
    return {"receiver_build": receiver_build, "replays": manifests, "clean": clean, "synthetic": synthetic}


def freeze_clean_artifacts_repaired(result: dict[str, Any]) -> None:
    freeze_clean_artifacts(result)
    freeze = read_json(ARTIFACT / "execution_freeze.json")
    freeze["code_bindings"].update({relative: sha256_file(ROOT / relative) for relative in REPAIR_PATHS})
    freeze["prefreeze_engineering_repair"] = {"trigger": "worktree-local .venv path did not exist", "stage": "CMake configure before clean receiver replay or scoring", "scientific_changes": False, "receiver_source_changes": False, "preserved_partial_build_resumed": True}
    write_json(ARTIFACT / "execution_freeze.json", freeze)
