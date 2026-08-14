#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_successor_freeze import ARTIFACT_RELATIVE, verify_successor_freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-wrapper-commit", required=True)
    args = parser.parse_args()
    result = verify_successor_freeze(ROOT, ROOT / ARTIFACT_RELATIVE, args.expected_wrapper_commit)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
