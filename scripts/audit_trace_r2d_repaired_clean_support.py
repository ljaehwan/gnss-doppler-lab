#!/usr/bin/env python3
"""Validate repaired OAKBAT clean support before frozen Phase-B fitting."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_native_1ms import load_native_trace_pairs, sha256_file

ARTIFACT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
DUMP = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2d-oakbat-clean-support-repair/dumps/phase_b/"
    "oakbat_cleanstatic/rep1"
)


def main() -> int:
    pairs = load_native_trace_pairs(DUMP, cn0_min_db_hz=28.0, lock_min=0.85, prompt_epsilon=1e-12)
    common = pairs.valid_support[:, np.arange(1, 8)].all(axis=1)
    finite = (
        np.isfinite(pairs.current[:, 1:8].real).all(axis=1)
        & np.isfinite(pairs.current[:, 1:8].imag).all(axis=1)
        & np.isfinite(pairs.target[:, 1:8].real).all(axis=1)
        & np.isfinite(pairs.target[:, 1:8].imag).all(axis=1)
    )
    selected = pairs.take(common & finite)
    start, end = float(selected.time_s.min()), float(selected.time_s.max())
    duration = end - start
    b1, b2, b3 = start + 0.45 * duration, start + 0.65 * duration, start + 0.80 * duration
    guard = 5.0
    masks = {
        "train": selected.time_s < b1 - guard,
        "covariance_validation": (selected.time_s >= b1 + guard) & (selected.time_s < b2 - guard),
        "calibration": (selected.time_s >= b2 + guard) & (selected.time_s < b3 - guard),
        "holdout": selected.time_s >= b3 + guard,
    }
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    manifest = json.loads((DUMP / "manifest.json").read_text())
    passed = all(value > 0 for value in counts.values()) and len(set(map(int, selected.prn))) >= 4
    prior = json.loads((ARTIFACT / "oakbat_clean_support_audit.json").read_text())
    payload = {
        "schema": "gnss-doppler-lab.trace-r2d-oakbat-clean-support-audit.v2",
        "status": "PASS" if passed else "FAIL",
        "failure_label_if_any": None if passed else "OAKBAT_CLEAN_SPLIT_EMPTY_AFTER_REPAIR",
        "parent_failure_audit": prior,
        "repair": {
            "mapping": "cleanStatic-specific normal-only target-aligned handoff at zero source skip",
            "handoff_path": manifest["frozen_handoff_path"],
            "handoff_sha256": manifest["frozen_handoff_sha256"],
            "receiver_config_sha256": manifest["receiver_config_sha256"],
            "receiver_executable_sha256": manifest["receiver_executable"]["sha256"],
            "receiver_manifest_path": str(DUMP / "manifest.json"),
            "receiver_manifest_sha256": sha256_file(DUMP / "manifest.json"),
        },
        "chronological_clean_support": {
            "selected_pair_count": int(len(selected.time_s)),
            "unique_prn_count": len(set(map(int, selected.prn))),
            "unique_prns": sorted(set(map(int, selected.prn))),
            "time_start_s": start,
            "time_end_s": end,
            "duration_s": duration,
            "boundaries_s": [b1, b2, b3],
            "guard_s": guard,
            "role_pair_counts": counts,
        },
        "frozen_contract": {
            "split_ratios": [0.45, 0.20, 0.15, 0.20],
            "guard_s": 5.0,
            "minimum_prns": 4,
            "cn0_min_db_hz": 28.0,
            "lock_min": 0.85,
        },
        "attack_data_read_or_scored": False,
    }
    (ARTIFACT / "oakbat_clean_support_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
