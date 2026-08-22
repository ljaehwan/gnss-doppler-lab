#!/usr/bin/env python3
"""CLI for the frozen Q-SET-GNSS Stage-0A R2 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.qset_stage0a_r2_evaluation import clean_execution, freeze_clean_artifacts, run_attacks
from gnss_doppler_lab.qset_stage0a_r2_execution import build_receiver
from gnss_doppler_lab.qset_stage0a_r2_finalize import compact_manifest, finalize_attack_artifacts
from gnss_doppler_lab.qset_stage0a_r2 import ARTIFACT, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-receiver")
    sub.add_parser("run-clean")
    attack = sub.add_parser("run-attacks"); attack.add_argument("--freeze-sha", required=True)
    sub.add_parser("write-manifest")
    args = parser.parse_args()
    if args.command == "build-receiver":
        result = build_receiver()
    elif args.command == "run-clean":
        result = clean_execution(); freeze_clean_artifacts(result)
        result = {"status": "PASS_CLEAN_EXECUTION", "clean_status": result["clean"]["status"], "receiver_sha256": result["receiver_build"]["receiver_sha256"], "threshold_sha256": result["clean"]["threshold_sha256"]}
    elif args.command == "run-attacks":
        payload = run_attacks(args.freeze_sha); result = finalize_attack_artifacts(payload, args.freeze_sha)
    else:
        result = compact_manifest(); write_json(ARTIFACT / "artifact_manifest_sha256.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
