#!/usr/bin/env python3
"""Independent verifier for Stage-1 R1 continuous tracker artifacts."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path


def verify_artifact(root: Path) -> tuple[bool, list[str], dict[str, object]]:
    fixed_cmd = ("PYTHONPATH=src python3 scripts/build_acaf_nf_continuous_tracker.py "
                 "--checkpoint A --source-binding configs/acaf_nf_stage1_source_binding.json "
                 "--output artifacts/acaf_nf_stage1_r1_continuous_tracker "
                 "--scenario cleanStatic --scenario ds3 --scenario ds4 --scenario ds7 --scenario ds8")
    errors: list[str] = []
    required = [
        "tracker_cadence_audit.json",
        "tracker_cadence_by_channel.csv",
        "checksums.json",
        "execution_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        errors.append("missing_files: " + ",".join(missing))

    audit = json.loads((root / "tracker_cadence_audit.json").read_text(encoding="utf-8")) if (root / "tracker_cadence_audit.json").exists() else {}
    if not isinstance(audit, dict):
        errors.append("tracker_cadence_audit_not_dict")
    schema = audit.get("schema")
    if schema != "acaf_nf_stage1_continuous_tracker_cadence.v1":
        errors.append("unexpected_schema")

    checksums_path = root / "checksums.json"
    if checksums_path.exists():
        checksum_doc = json.loads(checksums_path.read_text(encoding="utf-8"))
        file_checks = checksum_doc.get("files", {})
        for entry in ("tracker_cadence_audit.json", "tracker_cadence_by_channel.csv"):
            expected = file_checks.get(entry, {}).get("sha256")
            if not expected:
                errors.append(f"checksum_missing:{entry}")
                continue
            actual = hashlib.sha256((root / entry).read_bytes()).hexdigest()
            if actual != expected:
                errors.append(f"checksum_mismatch:{entry}")
    else:
        errors.append("missing_file:checksums.json")

    manifest_path = root / "execution_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("command") != fixed_cmd:
            errors.append("execution_manifest_command_mismatch")
    else:
        errors.append("missing_file:execution_manifest.json")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "required_files": required,
        "artifact_path": str(root),
        "artifact_schema": schema,
    }
    return report["status"] == "PASS", errors, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    ok, errors, report = verify_artifact(args.artifact)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
