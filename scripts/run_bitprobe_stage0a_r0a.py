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

from gnss_doppler_lab.bitprobe_stage0a_r0a import execute_repair, prepare_freeze


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen BITPROBE Stage-0A R0a inference repair")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-freeze", help="write pre-tensor-access freeze artifacts")
    execute = subparsers.add_parser("execute", help="run two repaired analyses from the frozen tensor")
    execute.add_argument("--freeze-sha", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.command == "prepare-freeze":
        prepare_freeze(repo)
        print("PASS: R0a execution freeze prepared with zero tensor/raw/TRACE/attack access")
    else:
        print(json.dumps(execute_repair(repo, args.freeze_sha), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
