#!/usr/bin/env python3
"""Independent compact verifier for committed SPLITCLOCK R1 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gnss_doppler_lab.splitclock_r1_experiment import verify_artifact


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("artifact", type=Path); args = parser.parse_args(); errors = verify_artifact(args.artifact)
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, sort_keys=True)); return 0 if not errors else 1


if __name__ == "__main__": raise SystemExit(main())
