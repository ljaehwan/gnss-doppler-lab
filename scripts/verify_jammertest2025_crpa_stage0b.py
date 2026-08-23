#!/usr/bin/env python3
"""Verify the compact Stage-0B artifact without opening raw payloads."""

from __future__ import annotations

import argparse
from pathlib import Path

from gnss_doppler_lab.jammertest_crpa_stage0b import read_json, verify_artifact


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/jammertest2025_crpa_stage0b_bounded_validation"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    errors = verify_artifact(args.artifact.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    verdict = read_json(args.artifact.resolve() / "final_verdict.json")
    access = read_json(args.artifact.resolve() / "access_audit.json")
    print("JAMMERTEST2025_CRPA_STAGE0B_VERIFY_PASS")
    print(f"verdict={verdict['verdict']}")
    print(f"spatial_gate_passed={str(verdict['spatial_gate_passed']).lower()}")
    print(f"classification_run={str(verdict['classification_run']).lower()}")
    print(f"downloaded_payload_bytes={access['downloaded_payload_bytes']}")
    print("forbidden_payload_bytes=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
