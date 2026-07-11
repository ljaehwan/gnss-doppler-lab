#!/usr/bin/env python3
"""Create a visual dashboard from GPS L1 C/A signed 8-bit IQ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gnss_doppler_lab.iq_visualization import load_s8_iq, render_iq_dashboard, summarize_iq


def latest_iq_file() -> Path:
    candidates = list((REPO_ROOT / "artifacts" / "rf_runs").glob("*/gps_l1ca_s8_iq.bin"))
    if not candidates:
        raise FileNotFoundError("No generated gps_l1ca_s8_iq.bin was found under artifacts/rf_runs")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GPS L1 C/A IQ as waveform, I/Q density, spectrum, and spectrogram")
    parser.add_argument("iq_file", nargs="?", type=Path, help="s8 interleaved IQ file; defaults to the newest generated run")
    parser.add_argument("--sample-rate", type=float, default=2_600_000.0, help="complex sample rate in Hz")
    parser.add_argument("--max-samples", type=int, default=2_600_000, help="maximum complex samples to load")
    parser.add_argument("--output", type=Path, help="output PNG; defaults to iq_dashboard.png beside the IQ file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    iq_path = (args.iq_file or latest_iq_file()).expanduser().resolve()
    output = (args.output or iq_path.with_name("iq_dashboard.png")).expanduser().resolve()
    iq = load_s8_iq(iq_path, max_complex_samples=args.max_samples)
    render_iq_dashboard(iq, sample_rate_hz=args.sample_rate, output_path=output)
    summary = summarize_iq(iq, sample_rate_hz=args.sample_rate)
    summary.update({"iq_file": str(iq_path), "output_png": str(output)})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
