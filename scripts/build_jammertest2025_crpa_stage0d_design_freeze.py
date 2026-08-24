#!/usr/bin/env python3
"""Create the label-only Stage-0D design freeze before IQ access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnss_doppler_lab.jammertest_crpa_stage0d import build_design


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    result = build_design(args.split_root.resolve(), args.artifact.resolve())
    print(json.dumps({"status": result["status"], "iq_bytes_read": result["iq_bytes_read"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
