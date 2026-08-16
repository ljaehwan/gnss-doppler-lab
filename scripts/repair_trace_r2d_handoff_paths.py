#!/usr/bin/env python3
"""Materialize SHA-identical short handoff paths and preserve the failed rep3."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gnss_doppler_lab.trace_native_1ms import sha256_file
import run_trace_stage0_r2d as r2d

ARTIFACT = r2d.driver.ARTIFACT
SOURCE = ARTIFACT / "handoffs"
MIRROR = r2d.driver.HANDOFF_ROOT
FAILED = r2d.driver.SSD_ROOT / "dumps/phase_a/texbat_cleanstatic/rep3"
ARCHIVE = r2d.driver.SSD_ROOT / "dumps/phase_a/texbat_cleanstatic/rep3-path-truncation-failed-20260816T135409Z"


def main() -> int:
    amendment = json.loads((ARTIFACT / "config_path_length_amendment_preregistered.json").read_text())
    if amendment["status"] != "PREREGISTERED_BEFORE_PHASE_A_RETRY":
        raise ValueError("path-length repair lacks preregistration")
    MIRROR.mkdir(parents=True, exist_ok=True)
    files = {}
    for source in sorted(SOURCE.glob("*.csv")):
        destination = MIRROR / source.name
        shutil.copy2(source, destination)
        if sha256_file(source) != sha256_file(destination):
            raise ValueError(f"handoff mirror differs: {source.name}")
        files[source.name] = {
            "committed_path": str(source),
            "runtime_path": str(destination),
            "sha256": sha256_file(source),
            "byte_size": source.stat().st_size,
        }
    shutil.copy2(SOURCE / "manifest.json", MIRROR / "manifest.json")
    if ARCHIVE.exists():
        raise FileExistsError(f"archive already exists: {ARCHIVE}")
    if not FAILED.exists():
        raise FileNotFoundError(f"missing preserved failed attempt: {FAILED}")
    FAILED.rename(ARCHIVE)
    payload = {
        "schema": "gnss-doppler-lab.trace-r2d-handoff-path-mirror.v1",
        "status": "PASS",
        "amendment": "config_path_length_amendment_preregistered.json",
        "runtime_root": str(MIRROR),
        "files": files,
        "sha256_identity": True,
        "failed_attempt_archived_at": str(ARCHIVE),
        "attack_data_read_or_scored": False,
    }
    (ARTIFACT / "handoff_path_mirror_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "mirrored_file_count": len(files), "runtime_root": str(MIRROR), "archived_failed_attempt": str(ARCHIVE)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
