#!/usr/bin/env python3
"""Independent checksum and scientific-contract verifier for MCTD Stage-0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = ["README.md", "config.json", "preregistration.json", "source_commit.json", "prior_evidence.json",
            "loop_configuration_freeze.json", "raw_source_binding.json", "receiver_inventory.json",
            "phase_a_reproducibility.json", "clean_split_audit.json", "normal_model_summary.json", "thresholds.json",
            "scenario_metrics.csv", "ablation_metrics.csv", "per_block_scores.csv.gz", "per_prn_divergence.csv.gz",
            "external_static_fpr.csv", "configuration_collapse_metrics.json", "pairing_destruction_metrics.json",
            "physical_controls.json", "bootstrap_intervals.csv", "final_verdict.json", "artifact_manifest_sha256.json",
            "runner_runs.json", "test_results.json", "plots"]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact-root", type=Path, required=True); args = parser.parse_args()
    root = args.artifact_root; failures = []
    for name in REQUIRED:
        if not (root / name).exists(): failures.append(f"missing:{name}")
    if (root / "artifact_manifest_sha256.json").exists():
        manifest = json.loads((root / "artifact_manifest_sha256.json").read_text())
        for name, expected in manifest.get("files", {}).items():
            path = root / name
            if not path.is_file(): failures.append(f"manifest_missing:{name}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected: failures.append(f"hash:{name}")
    for name in ("preregistration.json", "phase_a_reproducibility.json", "final_verdict.json"):
        if (root / name).exists():
            value = json.loads((root / name).read_text())
            if name == "preregistration.json" and value.get("attack_data_accessed_by_mctd") is not False: failures.append("preregistration_not_clean_only")
            if name == "phase_a_reproducibility.json" and not value.get("phase_a_passed"): failures.append("phase_a_not_passed")
            if name == "final_verdict.json" and value.get("verdict") not in {"GO_FOR_MCTD_NEURAL_STAGE1", "NO_GO_MCTD_PHYSICAL_HYPOTHESIS", "NO_GO_RECEIVER_DIFFERENTIAL_INVALID", "INCONCLUSIVE_INPUT_OR_RECEIVER"}: failures.append("invalid_verdict")
    output = {"schema": "gnss-doppler-lab.mctd-verification.v1", "status": "PASS" if not failures else "FAIL",
              "ok": not failures, "failures": failures}
    print(json.dumps(output, indent=2, sort_keys=True)); return 0 if not failures else 2


if __name__ == "__main__": raise SystemExit(main())
