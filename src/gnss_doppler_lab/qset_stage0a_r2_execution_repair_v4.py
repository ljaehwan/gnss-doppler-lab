"""Pre-result GNURadio runtime binding and preserved empty-attempt repair."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .qset_stage0a_r2_execution_repair_v2 import *  # noqa: F401,F403

SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
SYSTEM_PYTHON = Path("/usr/bin/python3")
V4_REPAIR_PATHS = (
    "src/gnss_doppler_lab/qset_stage0a_r2_execution_repair_v4.py",
    "scripts/run_qset_gnss_stage0a_r2_repaired_v4.py",
    "tests/test_qset_gnss_stage0a_r2_execution_repair_v4.py",
)


def bind_gnuradio_runtime() -> dict[str, Any]:
    require(Path(sys.executable).resolve() == SYSTEM_PYTHON.resolve(), "GNURadio execution must use /usr/bin/python3")
    require(SYSTEM_DIST_PACKAGES.is_dir(), "system dist-packages absent")
    text = str(SYSTEM_DIST_PACKAGES)
    if text not in sys.path:
        sys.path.insert(0, text)
    from gnuradio import gr
    return {
        "status": "PASS",
        "python_executable": sys.executable,
        "system_dist_packages": text,
        "gnuradio_version": gr.version(),
    }


def preserve_empty_attempt(failed: Path, preserved: Path, log_path: Path) -> dict[str, Any]:
    require(failed.is_dir(), "failed attempt directory absent")
    require(not any(failed.iterdir()), "failed attempt is not empty; refusing mutation")
    require(not preserved.exists(), "preservation target already exists")
    require(log_path.is_file(), "failed attempt log absent")
    log_binding = {"path": str(log_path), "size_bytes": log_path.stat().st_size, "sha256": sha256_file(log_path)}
    preserved.parent.mkdir(parents=True, exist_ok=True)
    failed.rename(preserved)
    result = {
        "schema": "gnss-doppler-lab.qset-stage0a-r2-prefreeze-attempt-preservation.v1",
        "status": "PRESERVED_EMPTY_PRE_SCORE_ATTEMPT",
        "original_path": str(failed),
        "preserved_path": str(preserved),
        "original_entries": [],
        "trigger": "ModuleNotFoundError: No module named 'gnuradio' before decoder creation",
        "clean_score_computed": False,
        "attack_accessed": False,
        "log": log_binding,
    }
    write_json(preserved / "attempt_preservation.json", result)
    return result


def build_receiver_repaired_v4() -> dict[str, Any]:
    bind_gnuradio_runtime()
    return build_receiver_repaired_v2()


def clean_execution_repaired_v4() -> dict[str, Any]:
    runtime = bind_gnuradio_runtime()
    failed = SSD_ROOT / "replays" / "C-1"
    preserved = SSD_ROOT / "historical-incomplete" / "C-1-missing-gnuradio-before-decoder"
    log_path = SSD_ROOT / "clean_execution_stdout.txt"
    if failed.exists():
        preservation = preserve_empty_attempt(failed, preserved, log_path)
    else:
        require((preserved / "attempt_preservation.json").is_file(), "expected preserved pre-score attempt absent")
        preservation = read_json(preserved / "attempt_preservation.json")
        require(preservation["status"] == "PRESERVED_EMPTY_PRE_SCORE_ATTEMPT", "preserved attempt drift")
    result = clean_execution_repaired_v2()
    result["gnuradio_runtime_repair"] = runtime
    result["preserved_failed_attempt"] = preservation
    return result


def freeze_clean_artifacts_repaired_v4(result: dict[str, Any]) -> None:
    freeze_clean_artifacts_repaired_v2(result)
    freeze = read_json(ARTIFACT / "execution_freeze.json")
    freeze["code_bindings"].update({relative: sha256_file(ROOT / relative) for relative in V4_REPAIR_PATHS})
    freeze["prefreeze_engineering_repair_v4"] = {
        "trigger": "project venv did not expose system GNURadio Python modules",
        "stage": "after C-1 identity MD5, before decoder output, receiver replay, feature extraction, or score",
        "resolution": "bind /usr/lib/python3/dist-packages and preserve the empty failed attempt under a versioned historical path",
        "runtime": result["gnuradio_runtime_repair"],
        "preserved_attempt": result["preserved_failed_attempt"],
        "scientific_changes": False,
        "receiver_source_changes": False,
        "attack_accessed": False,
    }
    write_json(ARTIFACT / "execution_freeze.json", freeze)
    audit = read_json(ARTIFACT / "access_audit.json")
    audit["prefreeze_failed_clean_attempt"] = {"scenario": "C-1", "identity_hash_bytes": SCENARIOS["C-1"]["size"], "decoder_bytes_read": 0, "receiver_runs": 0, "feature_windows": 0, "scores": 0, "attack_bytes_read": 0, "preserved": True}
    write_json(ARTIFACT / "access_audit.json", audit)
