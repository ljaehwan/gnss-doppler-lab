#!/usr/bin/env python3
"""R2 CLI with both pre-result build-environment repairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.qset_stage0a_r2_execution_repair_v2 import build_receiver_repaired_v2, clean_execution_repaired_v2, freeze_clean_artifacts_repaired_v2
from gnss_doppler_lab.qset_stage0a_r2_evaluation import run_attacks
from gnss_doppler_lab.qset_stage0a_r2_finalize import compact_manifest, finalize_attack_artifacts
from gnss_doppler_lab.qset_stage0a_r2 import ARTIFACT, write_json


def main() -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("build-receiver"); sub.add_parser("run-clean"); attack = sub.add_parser("run-attacks"); attack.add_argument("--freeze-sha", required=True); sub.add_parser("write-manifest"); args = parser.parse_args()
    if args.command == "build-receiver": result = build_receiver_repaired_v2()
    elif args.command == "run-clean":
        full = clean_execution_repaired_v2(); freeze_clean_artifacts_repaired_v2(full); result = {"status": "PASS_CLEAN_EXECUTION", "clean_status": full["clean"]["status"], "receiver_sha256": full["receiver_build"]["receiver_sha256"], "threshold_sha256": full["clean"]["threshold_sha256"]}
    elif args.command == "run-attacks":
        payload = run_attacks(args.freeze_sha); result = finalize_attack_artifacts(payload, args.freeze_sha)
    else:
        result = compact_manifest(); write_json(ARTIFACT / "artifact_manifest_sha256.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
