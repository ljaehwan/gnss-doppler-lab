#!/usr/bin/env python3
"""Separate real 9-tap GCMR-PI ablation; never replaces primary E/P/L 3-tap."""
from train_gcmr_peak_innovation import main
if __name__ == "__main__":
    raise SystemExit(main(["--tap-count", "9", "--output-dir", "artifacts/gcmr_peak_innovation_9tap", "--open-attacks"]))
