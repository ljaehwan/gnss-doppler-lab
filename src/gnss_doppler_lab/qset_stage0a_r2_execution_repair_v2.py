"""Pre-result build-environment repair for preserved third-party Git caches."""

from __future__ import annotations

import os
from typing import Any

from .qset_stage0a_r2_execution_repair import *  # noqa: F401,F403

V2_REPAIR_PATHS = (
    "src/gnss_doppler_lab/qset_stage0a_r2_execution_repair_v2.py",
    "scripts/run_qset_gnss_stage0a_r2_repaired_v2.py",
)


def _safe_git_environment() -> None:
    os.environ["GIT_CONFIG_COUNT"] = "1"
    os.environ["GIT_CONFIG_KEY_0"] = "safe.directory"
    os.environ["GIT_CONFIG_VALUE_0"] = "*"


def build_receiver_repaired_v2() -> dict[str, Any]:
    _safe_git_environment()
    return build_receiver_repaired()


def clean_execution_repaired_v2() -> dict[str, Any]:
    receiver_build = build_receiver_repaired_v2(); manifests = {}
    for name in ("C-1", "C-3"):
        replay = replay_scenario(name, scenario_path(name), receiver_build); manifests[name] = replay
        rows = extract_window_features(SSD_ROOT / "replays" / name / "receiver", name, SCENARIOS[name]["size"] / BYTES_PER_COMPLEX / RAW_FS); require(rows, f"no feature rows for {name}"); save_feature_cache(name, rows)
    clean = analyze_clean(); synthetic = synthetic_dilution(clean); require(synthetic["status"] == "PASS", "synthetic implementation sanity failed")
    return {"receiver_build": receiver_build, "replays": manifests, "clean": clean, "synthetic": synthetic}


def freeze_clean_artifacts_repaired_v2(result: dict[str, Any]) -> None:
    freeze_clean_artifacts_repaired(result)
    freeze = read_json(ARTIFACT / "execution_freeze.json")
    freeze["code_bindings"].update({relative: sha256_file(ROOT / relative) for relative in V2_REPAIR_PATHS})
    freeze["prefreeze_engineering_repair_v2"] = {"trigger": "preserved third-party Git caches failed ownership safety check", "stage": "dependency build before clean receiver replay or scoring", "resolution": "build-subprocess-only safe.directory=* matching the verified R2c build environment", "scientific_changes": False, "receiver_source_changes": False}
    write_json(ARTIFACT / "execution_freeze.json", freeze)
