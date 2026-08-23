#!/usr/bin/env python3
"""Verify the committed Jammertest 2025 CRPA Stage-0C artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnss_doppler_lab.jammertest_crpa_stage0c_execution import verify_artifact


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/jammertest2025_crpa_stage0c_spatial_discrimination"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    errors = verify_artifact(args.artifact.resolve())
    print(json.dumps({
        "artifact": str(args.artifact.resolve()),
        "error_count": len(errors),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
