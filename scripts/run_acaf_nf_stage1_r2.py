#!/usr/bin/env python3
"""Run ACAF-NF Stage-1 R2 full-normal checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

from gnss_doppler_lab.acaf_nf_stage1_r2 import checkpoint1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=("1",), default="1")
    parser.add_argument("--output", type=Path, default=Path("artifacts/acaf_nf_stage1_r2_full_normal"))
    parser.add_argument(
        "--r1-artifact",
        type=Path,
        default=Path("/home/ubuntu/orca/workspaces/gnss-doppler-lab/research-acaf-nf-stage1-r1-continuous-tracker/artifacts/acaf_nf_stage1_r1_continuous_tracker"),
    )
    args = parser.parse_args()
    checkpoint1(args.output, args.r1_artifact)


if __name__ == "__main__":
    main()
