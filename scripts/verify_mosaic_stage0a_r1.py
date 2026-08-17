#!/usr/bin/env python3
"""Verify the self-contained MOSAIC Stage-0A R1 artifact bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mosaic_stage0a_r1_raw_recorrelation"
REQUIRED = {
    "README.md", "config.json", "source_commit.json", "execution_environment.json",
    "receiver_provenance.json", "raw_source_binding.json", "navbit_requirement_audit.md",
    "navbit_invariance_test.json", "selected_epoch_inventory.csv", "raw_recorrelation_metrics.csv",
    "per_prn_alignment_metrics.csv", "rejected_rows.csv", "alignment_summary.json",
    "final_verdict.json", "artifact_manifest_sha256.json",
}
ALLOWED_VERDICTS = {
    "STAGE0A_RAW_ALIGNMENT_PASS", "NAV_BIT_OR_SYMBOL_ALIGNMENT_REQUIRED",
    "RECEIVER_CONVENTION_UNRESOLVED", "SOURCE_BINDING_MISMATCH", "STAGE0A_RAW_ALIGNMENT_FAIL",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest_sha256.json"
    }


def verify(root: Path = ART) -> dict[str, object]:
    files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    missing = sorted(REQUIRED - files)
    manifest_path = root / "artifact_manifest_sha256.json"
    recorded = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    actual = build_manifest(root) if root.exists() else {}
    verdict_path = root / "final_verdict.json"
    verdict = json.loads(verdict_path.read_text()).get("verdict") if verdict_path.exists() else None
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    attack_free = config.get("attack_data_used") is False and config.get("stage0b_run") is False
    status = "PASS" if not missing and recorded == actual and verdict in ALLOWED_VERDICTS and attack_free else "FAIL"
    return {
        "status": status,
        "missing_required_files": missing,
        "manifest_matches": recorded == actual,
        "verdict_allowed": verdict in ALLOWED_VERDICTS,
        "attack_free": attack_free,
        "file_count": len(actual),
    }


def main() -> int:
    if "--write-manifest" in sys.argv:
        (ART / "artifact_manifest_sha256.json").write_text(json.dumps(build_manifest(ART), indent=2, sort_keys=True) + "\n")

    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
