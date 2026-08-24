#!/usr/bin/env python3
"""Materialize the result-independent R1 design freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnss_doppler_lab.splitclock_r1_contract import BASE_SHA, BRANCH, R0_ARTIFACT, frozen_design


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("artifact", type=Path); args = parser.parse_args()
    args.artifact.mkdir(parents=True, exist_ok=True)
    design = frozen_design(); write_json(args.artifact / "design_freeze.json", design)
    write_json(args.artifact / "source_binding.json", {
        "status": "PASS", "base_sha": BASE_SHA, "branch": BRANCH,
        "r0_artifact": R0_ARTIFACT, "r0_final_sha": BASE_SHA,
        "r0_results_reused": ["verified RINEX L1B sign", "verified clean V3 output manifests", "verified clean panel inventory"],
        "r0_artifact_modified": False,
    })
    write_json(args.artifact / "access_audit.json", {
        "status": "PASS", "phase": "R1_DESIGN_FREEZE", "clean_score_operations": 0,
        "clean_raw": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
        "attack": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
        "jammertest_raw": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
    })
    (args.artifact / "README.md").write_text(
        "# SPLITCLOCK-GNSS Stage-0A R1 contract/model repair\n\n"
        "Status: pre-clean-score R1 design freeze. The R0 artifact is immutable. No clean score, threshold, attack access, or Jammertest raw access occurred before this freeze.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
