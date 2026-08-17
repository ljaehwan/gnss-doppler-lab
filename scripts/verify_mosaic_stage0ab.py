#!/usr/bin/env python3
"""Verify MOSAIC Stage-0A/0B fail-closed artifact bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mosaic_stage0ab_foundation"
ALLOWED = {
    "GO_FOR_MOSAIC_NEURAL_STAGE1",
    "NO_GO_MOSAIC_INJECTOR_PHYSICS",
    "INCONCLUSIVE_INPUT_OR_ALIGNMENT",
    "INCONCLUSIVE_INPUT_OR_RECEIVER_UNAVAILABLE",
    "INCONCLUSIVE_TRACKER_RAW_ALIGNMENT",
    "INCONCLUSIVE_NAVIGATION_BIT_PROVENANCE",
}
REQUIRED = [
    "README.md", "config.json", "preregistration.json", "source_commit.json", "environment_inventory.json",
    "data_inventory.json", "receiver_field_contract.json", "raw_source_binding.json", "clean_split_audit.json",
    "stage0a_alignment_metrics.json", "stage0a_per_epoch_sample.csv.gz", "injection_design.json",
    "injection_design_sha256.json", "injection_physics_metrics.csv", "parameter_recovery_metrics.csv",
    "residual_caf_metrics.csv", "physical_controls.json", "resource_profile.json", "final_verdict.json",
    "artifact_manifest_sha256.json",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default=str(ART))
    args = parser.parse_args()
    art = Path(args.artifact_root)
    errors: list[str] = []
    for name in REQUIRED:
        if not (art / name).exists():
            errors.append(f"missing {name}")
    if not (art / "plots").is_dir():
        errors.append("missing plots/")
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 2
    final = json.loads((art / "final_verdict.json").read_text())
    if final.get("verdict") not in ALLOWED:
        errors.append("final verdict not in allowed label set")
    if final.get("attack_scores_computed") or final.get("neural_training"):
        errors.append("prohibited attack scores or neural training flag set")
    design = json.loads((art / "injection_design.json").read_text())
    design_sha = json.loads((art / "injection_design_sha256.json").read_text())
    payload = json.dumps(design, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(payload).hexdigest() != design_sha.get("sha256"):
        errors.append("injection design checksum mismatch")
    manifest = json.loads((art / "artifact_manifest_sha256.json").read_text())
    for rel, expected in manifest.items():
        p = art / rel
        if not p.exists() or sha(p) != expected:
            errors.append(f"artifact checksum mismatch: {rel}")
    forbidden = [p for p in art.rglob("*") if p.is_file() and p.suffix.lower() in {".bin", ".mat", ".npz"}]
    if forbidden:
        errors.append("large/raw binary artifact committed under MOSAIC root")
    status = "PASS" if not errors else "FAIL"
    result = {"status": status, "artifact_root": str(art), "verdict": final.get("verdict"), "checked_files": len(manifest), "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
