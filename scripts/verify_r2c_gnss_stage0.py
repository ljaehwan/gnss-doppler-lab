#!/usr/bin/env python3
"""Independent fail-closed verifier for an R2C-GNSS Stage-0 artifact."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.r2c_gnss import artifact_hashes, write_json  # noqa: E402

REQUIRED = {
    "README.md", "config.json", "provenance.json", "input_validity.json", "training_summary.json",
    "thresholds.json", "scenario_metrics.csv", "ablation_metrics.csv", "per_epoch_scores.csv",
    "gain_invariance.json", "phase_invariance.json", "noise_control.json", "multipath_control.json",
    "second_source_injection.json", "relation_destruction.json", "decision.json", "verification.json",
    "hashes.json", "plots/relation_control.png", "plots/relation_control_source.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts/r2c_gnss_stage0")
    parser.add_argument("--write-result", action="store_true"); args = parser.parse_args(); root = args.artifact.resolve()
    errors: list[str] = []
    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    if missing: errors.append(f"missing files: {missing}")
    if not missing:
        config = json.loads((root / "config.json").read_text()); provenance = json.loads((root / "provenance.json").read_text())
        validity = json.loads((root / "input_validity.json").read_text()); training = json.loads((root / "training_summary.json").read_text())
        thresholds = json.loads((root / "thresholds.json").read_text()); decision = json.loads((root / "decision.json").read_text())
        stored = json.loads((root / "hashes.json").read_text())["files"]; actual = artifact_hashes(root)
        if stored != actual: errors.append("artifact hashes changed or file set mismatched")
        if not config.get("decision_criteria_frozen"): errors.append("decision criteria not frozen")
        if not validity.get("frozen_before_attack_evaluation") or validity.get("attack_outcomes_inspected_for_tuning"):
            errors.append("invalid input-gate provenance")
        if training.get("fit_roles"): errors.append("DATA_INVALID artifact must not claim fitted roles")
        if thresholds.get("values"): errors.append("DATA_INVALID artifact must not contain fitted thresholds")
        if decision.get("verdict") != validity.get("decision") or decision.get("verdict") != "DATA_INVALID":
            errors.append("decision inconsistent with validity gate")
        if provenance.get("frozen_base_commit") != "461eb4dc7bb794e719295daf028f6811658ba37f": errors.append("wrong frozen base")
        if provenance.get("branch") != "research/r2c-gnss-stage0": errors.append("wrong branch")
        current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if provenance.get("source_commit_at_generation") not in {current, "461eb4dc7bb794e719295daf028f6811658ba37f"}:
            errors.append("source commit is not current or frozen base")
        for table in ("scenario_metrics.csv", "ablation_metrics.csv", "per_epoch_scores.csv"):
            with (root / table).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows: errors.append(f"empty table: {table}")
            for row in rows:
                if any(str(value).lower() in {"nan", "inf", "-inf"} for value in row.values()): errors.append(f"non-finite value: {table}")
    status = "PASS" if not errors else "FAIL"
    result = {"status": status, "errors": errors, "checked_required_files": len(REQUIRED),
              "hash_policy": "verification.json and hashes.json excluded to avoid self-reference"}
    if args.write_result:
        write_json(root / "verification.json", result)
        write_json(root / "hashes.json", {"algorithm": "sha256", "files": artifact_hashes(root)})
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
