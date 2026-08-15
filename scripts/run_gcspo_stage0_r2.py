#!/usr/bin/env python3
"""Runner-compatible phase entrypoint for the frozen GCSPO Stage-0 R2 evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_r2_runner import ARTIFACT_ROOT, run_phase  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "preflight",
            "cleanstatic-normal-model",
            "ds3-evaluation",
            "ds7-evaluation",
            "ds4-ds8-conditional-evaluation",
            "relation-destruction-physical-controls",
            "final-statistics-plots-verification",
            "refresh-inventory-manifest",
        ),
    )
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    run_phase(args.phase, args.artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
