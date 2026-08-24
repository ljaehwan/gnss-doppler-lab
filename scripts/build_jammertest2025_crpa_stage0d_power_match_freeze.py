#!/usr/bin/env python3
"""Build the Stage-0D received-power matching freeze before spatial scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnss_doppler_lab.jammertest_crpa_stage0d_power import build_power_match_freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-npy", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    result = build_power_match_freeze(args.raw_npy.resolve(), args.artifact.resolve())
    print(json.dumps({
        "status": result["status"],
        "primary_gate_pass": result["primary_gate_pass"],
        "primary": result["calipers"]["0.25"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
