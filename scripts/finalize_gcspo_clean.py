#!/usr/bin/env python3
"""Finalize the aggregate clean-only gate after a byte-identical rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.gcspo_artifacts import canonical_write_json, sha256_file

SCIENTIFIC = (
    "normal_model_summary.json", "thresholds.json", "clean_ablation_report.json",
    "clean_a5_report.json", "clean_b0_report.json", "physical_controls.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/gcspo_stage0_static")
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    observed = {name: sha256_file(args.artifact_dir / name) for name in SCIENTIFIC}
    baseline = json.loads(args.baseline.read_text())
    if observed != baseline:
        raise RuntimeError(f"deterministic clean rerun mismatch: baseline={baseline} observed={observed}")
    ablation_path = args.artifact_dir / "clean_ablation_report.json"
    ablations = json.loads(ablation_path.read_text())
    ablations.update({"run_status": "CLEAN_ABLATIONS_PASS", "remaining": [],
                      "A5_report": "clean_a5_report.json", "A0_B0_report": "clean_b0_report.json"})
    canonical_write_json(ablation_path, ablations)
    clean_path = args.artifact_dir / "clean_only_report.json"
    clean = json.loads(clean_path.read_text())
    clean.pop("started_utc", None); clean.pop("finished_utc", None)
    clean.update({"all_methods": ["A0", "A1", "A2", "A3", "A4", "A5", "Full"],
                  "deterministic_rerun": "PASS", "deterministic_pre_finalize_hashes": observed,
                  "related_regressions": "PENDING", "clean_gate_status": "PASS"})
    canonical_write_json(clean_path, clean)
    print("CLEAN_DETERMINISTIC_RERUN_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
