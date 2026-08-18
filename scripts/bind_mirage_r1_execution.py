#!/usr/bin/env python3
"""Bind the complete R1 executor before any scientific result is computed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mirage_stage0a_r1_full_execution"
PREREG = "bd27782c54c6f6df603f9a08b831013826d24046"
FILES = [
    "src/gnss_doppler_lab/mirage_r1.py",
    "src/gnss_doppler_lab/mirage_r1_executor.py",
    "scripts/preregister_mirage_r1.py",
    "scripts/run_mirage_r1_full.py",
    "scripts/bind_mirage_r1_execution.py",
    "scripts/verify_mirage_r1_full.py",
    "tests/test_mirage_r1.py",
    "tests/test_mirage_r1_executor.py",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
    if head != PREREG:
        raise SystemExit(f"binding must run at exact pushed preregistration {PREREG}; got {head}")
    if subprocess.run(["git", "rev-parse", f"origin/research/mirage-stage0a-r1-full-execution"], cwd=ROOT,
                      check=True, text=True, stdout=subprocess.PIPE).stdout.strip() != head:
        raise SystemExit("preregistration local/remote mismatch")
    binding = {
        "schema": "gnss-doppler-lab.mirage-r1-execution-code-binding.v1",
        "preregistration_commit": PREREG, "code_sha256": {name: sha(ROOT / name) for name in FILES},
        "case_score": "median of three nonoverlapping 500 ms Full epochs in the 1.5 s steady interval",
        "control_execution": {
            "authentic": "actual clean raw-IQ recorrelation with original applied tracker state",
            "primary": "actual raw-IQ injection, pinned receiver replay, replay-state raw-IQ recorrelation",
            "gain_phase_matched_collapsed": "exact complex linear transforms of empirically reconstructed taps",
            "awgn_cn0": "actual clean raw-IQ plus deterministic empirical-variance AWGN, recorrelation with applied tracker state",
            "delay_doppler_drift": "single-source transforms of empirical complex-tap sequences",
            "one_prn_drop_add": "multi-PRN aggregation relation controls",
        },
        "results_seen": False, "scientific_results_computed": False, "cases_executed": 0,
        "receiver_replays_max_concurrent": 1, "caf_workers_max": 4, "status": "FROZEN",
    }
    dump(ART / "execution_code_binding.json", binding)
    freeze = json.loads((ART / "preregistration_freeze.json").read_text())
    freeze["execution_code_binding_sha256"] = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    freeze["status"] = "FROZEN_EXECUTION_CODE_BOUND_PENDING_COMMIT_AND_PUSH"
    dump(ART / "preregistration_freeze.json", freeze)
    files = []
    for path in sorted(ART.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            files.append({"path": str(path.relative_to(ART)), "size_bytes": path.stat().st_size, "sha256": sha(path)})
    dump(ART / "artifact_manifest_sha256.json", {"schema": "gnss-doppler-lab.artifact-manifest-sha256.v1", "files": files})
    print("FROZEN_EXECUTION_CODE_BOUND_PENDING_COMMIT_AND_PUSH")


if __name__ == "__main__": main()
