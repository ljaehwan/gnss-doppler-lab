#!/usr/bin/env python3
"""Verify the exact committed Round-8 successor freeze read-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_round8_verify import verify_round8_freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-freeze-commit", required=True)
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=ROOT / "artifacts/gcspo_stage0_static_rerun",
    )
    args = parser.parse_args()
    result = verify_round8_freeze(args.artifact_dir, args.expected_freeze_commit)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
