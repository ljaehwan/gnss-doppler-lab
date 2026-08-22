#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from gnss_doppler_lab.bitprobe_stage0a import execute_clean_stage0a, prepare_freeze

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-freeze")
    execute = sub.add_parser("execute")
    execute.add_argument("--freeze-sha", required=True)
    args = parser.parse_args()
    if args.command == "prepare-freeze":
        prepare_freeze(args.repo.resolve())
        print("PASS: BITPROBE Stage-0A implementation freeze prepared without raw/TRACE access")
    else:
        verdict = execute_clean_stage0a(args.repo.resolve(), args.freeze_sha)
        print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
