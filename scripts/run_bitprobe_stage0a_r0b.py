#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab.bitprobe_stage0a_r0b import execute, prepare_freeze


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BITPROBE Stage-0A R0b relaxed-gate sensitivity")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-freeze")
    analysis = commands.add_parser("execute")
    analysis.add_argument("--freeze-sha", required=True)
    args = parser.parse_args()
    if args.command == "prepare-freeze":
        prepare_freeze(args.repo.resolve())
        print("PASS: R0b sensitivity freeze prepared with zero raw/TRACE/tensor/attack access")
    else:
        print(json.dumps(execute(args.repo.resolve(), args.freeze_sha), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
