#!/usr/bin/env python3
"""Build the Stage-0C label-only design freeze before opening IQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnss_doppler_lab.jammertest_crpa_stage0c import build_design


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = build_design(args.split_root.resolve(), args.output.resolve())
    print(json.dumps({
        "status": design["status"],
        "primary_feasible": design["primary"]["audit"]["feasible"],
        "sensitivity_a_feasible": design["sensitivity_a"]["audit"]["feasible"],
        "sensitivity_b_feasible": design["sensitivity_b"]["audit"]["feasible"],
        "iq_feature_bytes_read": design["iq_feature_bytes_read"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
