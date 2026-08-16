#!/usr/bin/env python3
"""Validate authenticated native-1-ms TRACE receiver dump files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_native_1ms import validate_dump_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--scenario-id")
    parser.add_argument("--minimum-prns", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.dump_dir.glob("trace_native_1ms_ch_*.bin"))
    result = validate_dump_files(
        paths,
        expected_scenario_id=args.scenario_id,
        minimum_prns=args.minimum_prns,
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
