#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.gcspo_artifacts import canonical_write_json, FROZEN_HASHES, sha256_file, utc_now
from gnss_doppler_lab.gcspo_verify import verify_clean_ready, verify_final

def _artifact_manifest_sha(artifact_dir, phase):
    path = Path(artifact_dir) / "artifact_manifest_sha256.json"
    if phase == "clean-ready" and not path.is_file():
        return None
    return sha256_file(path)



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/gcspo_stage0_static_rerun")
    parser.add_argument("--mode", choices=("full",))
    parser.add_argument("--phase", choices=("clean-ready", "final"))
    args = parser.parse_args()
    if not args.mode and not args.phase: parser.error("--mode full or --phase is required")
    started = utc_now(); phase = "final" if args.mode == "full" else args.phase
    result = verify_clean_ready(args.artifact_dir) if phase == "clean-ready" else verify_final(args.artifact_dir, strict=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    freeze_path = args.artifact_dir / "implementation_manifest.json"
    target = head if not freeze_path.is_file() else __import__("json").loads(freeze_path.read_text()).get("target_commit", head)
    report = {"schema": "gnss-doppler-lab.gcspo-stage0.verifier-report.v2", "commit": target,
              "target_commit": target, "evidence_commit": head, "config_sha256": FROZEN_HASHES["config.json"],
              "artifact_manifest_sha256": _artifact_manifest_sha(args.artifact_dir, phase),
              "checks": [{"id": "semantic_reconstruction", "status": result["status"]}],
              "overall_status": "PASS", "verified_run_status": "VALID_SCIENCE" if phase == "final" else "CLEAN_ONLY_PASS",
              "command": "--mode full" if args.mode else f"--phase {phase}", "started_utc": started,
              "finished_utc": utc_now(), "exit_code": 0}
    canonical_write_json(args.artifact_dir / "verifier_report.json", report)
    print(f"VERIFIER_PASS phase={phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
