#!/usr/bin/env python3
"""VS Code-friendly entry point for normal GPS IQ generation."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab.cli import main  # noqa: E402


def default_arguments() -> list[str]:
    """Return the default smoke scenario used by VS Code's Run button."""
    executable = REPO_ROOT / ".tools" / "gps-sdr-sim-src" / "gps-sdr-sim"
    return [
        "generate",
        str(REPO_ROOT / "configs" / "gps_l1ca_static.example.yaml"),
        "--executable",
        str(executable),
    ]


if __name__ == "__main__":
    arguments = sys.argv[1:] or default_arguments()
    raise SystemExit(main(arguments))
