#!/usr/bin/env python3
"""Prepare and execute the frozen CRID R4b clean Phase A."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.crid_r4b_phase_a import (  # noqa: E402
    BindingError,
    R4B_ART,
    R4B_SSD,
    analyze_phase_a,
    authorize_threshold,
    preflight_inputs,
    prepare_freeze,
    run_all_replays,
)


ARTIFACT = ROOT / R4B_ART


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-freeze")
    for name in ("preflight", "authorize-threshold", "replay", "analyze"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--freeze-sha", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare-freeze":
            result = prepare_freeze(ROOT, ARTIFACT)
        elif args.command == "preflight":
            result = preflight_inputs(ROOT, ARTIFACT, R4B_SSD, args.freeze_sha)
        elif args.command == "authorize-threshold":
            result = authorize_threshold(ROOT, ARTIFACT, R4B_SSD, args.freeze_sha)
        elif args.command == "replay":
            result = run_all_replays(ROOT, ARTIFACT, R4B_SSD, args.freeze_sha)
        else:
            result = analyze_phase_a(ROOT, ARTIFACT, R4B_SSD, args.freeze_sha)
    except (BindingError, FileNotFoundError, KeyError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema": "gnss-doppler-lab.crid-r4b-runner-error.v1",
                    "status": "FAIL_CLOSED",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 4 if "INCONCLUSIVE_RESOURCE" in str(exc) else 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
