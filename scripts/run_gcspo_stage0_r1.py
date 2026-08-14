#!/usr/bin/env python3
"""Execute the preregistered R1 protected evaluation exactly once."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_r1_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
