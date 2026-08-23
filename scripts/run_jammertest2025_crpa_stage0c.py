#!/usr/bin/env python3
"""Frozen Stage-0C execution skeleton.

Implementation is intentionally blocked in the label-only design-freeze
commit.  The executable is completed only after that commit is pushed.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-npy", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.parse_args()
    raise SystemExit("LABEL_ONLY_DESIGN_FREEZE_IMPLEMENTATION_NOT_YET_AUTHORIZED")


if __name__ == "__main__":
    raise SystemExit(main())
