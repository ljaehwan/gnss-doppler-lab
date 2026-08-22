#!/usr/bin/env python3
"""Finalize a frozen Q-SET R2 technical support failure without scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.qset_stage0a_r2_terminal_inconclusive import finalize_terminal_inconclusive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(finalize_terminal_inconclusive(args.freeze_sha), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
