#!/usr/bin/env python3
"""Prepare and execute the frozen exploratory CRID R4c TEXBAT DS3 audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.crid_r4c_ds3 import (  # noqa: E402
    ARTIFACT_REL,
    SSD_ROOT,
    analyze,
    inventory_ds3,
    prepare_freeze,
    run_replays,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-freeze")
    for name in ("inventory", "replay", "analyze"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--freeze-sha", required=True)
    args = parser.parse_args()
    artifact = ROOT / ARTIFACT_REL
    if args.command == "prepare-freeze":
        result = prepare_freeze(ROOT, artifact)
    elif args.command == "inventory":
        result = inventory_ds3(ROOT, artifact, SSD_ROOT, args.freeze_sha)
    elif args.command == "replay":
        result = run_replays(ROOT, artifact, SSD_ROOT, args.freeze_sha)
    else:
        result = analyze(ROOT, artifact, SSD_ROOT, args.freeze_sha)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
