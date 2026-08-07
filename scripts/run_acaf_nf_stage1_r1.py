#!/usr/bin/env python3
"""Run the Stage-1 R1 pipeline."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from gnss_doppler_lab.acaf_nf_stage1_r1 import run_stage1_r1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/acaf_nf_stage1_r1_continuous_tracker")
    parser.add_argument(
        "--source-binding",
        default="configs/acaf_nf_stage1_source_binding.json",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("A", "B", "C", "D"),
        default="A",
        help="Checkpoint selector: A cadence, B cleanStatic validation, C attack binding, D actual evaluation.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
    )
    args = parser.parse_args()

    if args.checkpoint == "A":
        cmd = [
            sys.executable,
            str(Path("scripts/build_acaf_nf_continuous_tracker.py")),
            "--checkpoint",
            "A",
            "--source-binding",
            args.source_binding,
            "--output",
            args.output,
        ]
        if args.scenario:
            for scenario in args.scenario:
                cmd.extend(["--scenario", scenario])
        result = subprocess.run(cmd, check=True)
    elif args.checkpoint == "B":
        if args.scenario and args.scenario != ["cleanStatic"]:
            raise ValueError("--scenario for checkpoint B must be cleanStatic")
        cmd = [
            sys.executable,
            str(Path("scripts/build_acaf_nf_continuous_tracker.py")),
            "--checkpoint",
            "B",
            "--source-binding",
            args.source_binding,
            "--output",
            args.output,
            "--scenario",
            args.scenario[0] if args.scenario else "cleanStatic",
        ]
        result = subprocess.run(cmd, check=True)
    elif args.checkpoint == "C":
        expected = ["ds3", "ds4", "ds7", "ds8"]
        if args.scenario and args.scenario != expected:
            raise ValueError("--scenario for checkpoint C must be ds3,ds4,ds7,ds8")
        cmd = [sys.executable, str(Path("scripts/build_acaf_nf_continuous_tracker.py")), "--checkpoint", "C",
               "--source-binding", args.source_binding, "--output", args.output]
        for scenario in expected: cmd.extend(["--scenario", scenario])
        result = subprocess.run(cmd, check=True)
    elif args.checkpoint == "D":
        if args.scenario: raise ValueError("--scenario is not accepted for checkpoint D")
        run_stage1_r1(Path(args.output), Path(args.source_binding))
        result = subprocess.CompletedProcess([], 0)
    else:
        raise ValueError(f"Unsupported checkpoint: {args.checkpoint}")

    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
