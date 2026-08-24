#!/usr/bin/env python3
"""Verify the committed Stage-0D artifact without raw-IQ access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnss_doppler_lab.jammertest_crpa_stage0d_execution import verify_artifact


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/jammertest2025_crpa_stage0d_true_spoof_discrimination"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    errors = verify_artifact(args.artifact.resolve())
    print(json.dumps({
        "artifact": str(args.artifact.resolve()),
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors,
    }, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
