#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_successor_freeze import (
    ARTIFACT_RELATIVE,
    CONFIG_SHA256,
    HANDOFF_SCHEMA,
    INVALID_EVIDENCE_COMMIT,
    INVOCATION_ID,
    INVOCATION_NONCE,
    LATER_GCSPO_PATHS,
    PREDECESSOR_FREEZE_COMMIT,
    PREREGISTRATION_SHA256,
    REJECTED_TARGET_COMMIT,
    REJECTED_WRAPPER_COMMIT,
    SECOND_REJECTED_TARGET_COMMIT,
    SECOND_REJECTED_WRAPPER_COMMIT,
    REQUIRED_INTERNAL_DEPENDENCY_PATHS,
    STALE_CROSS_GENERATION_PATHS,
    build_successor_manifest,
    strict_json_bytes,
    verify_control_protected_state,
    verify_handoff_protected_state,
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                          capture_output=True).stdout.strip()


def _write(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-commit", required=True)
    args = parser.parse_args()
    target = args.target_commit
    if _git("rev-parse", "HEAD") != target:
        raise SystemExit("builder HEAD must equal approved successor target")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise SystemExit("builder requires a clean target worktree")
    artifact = ROOT / ARTIFACT_RELATIVE
    control = strict_json_bytes((artifact / "successor_control.json").read_bytes(), "control")
    verify_control_protected_state(control)
    if (control.get("invocation", {}).get("invocation_id") != INVOCATION_ID or
            control.get("invocation", {}).get("nonce") != INVOCATION_NONCE or
            control.get("invocation", {}).get("same_invocation_retry") is not False):
        raise SystemExit("immutable invocation control mismatch")
    marker = artifact.parent / f".{artifact.name}.protected_run_started.json"
    old_marker = ROOT / "artifacts/.gcspo_stage0_static_rerun.protected_run_started.json"
    if marker.exists() or old_marker.exists():
        raise SystemExit("protected marker exists")
    if (artifact / "access_ledger.jsonl").stat().st_size != 0:
        raise SystemExit("successor protected ledger is not empty")
    manifest = build_successor_manifest(
        ROOT, target_commit=target, invocation_id=INVOCATION_ID, nonce=INVOCATION_NONCE,
        predecessor_freeze_commit=PREDECESSOR_FREEZE_COMMIT,
        invalid_evidence_commit=INVALID_EVIDENCE_COMMIT,
        config_sha256=CONFIG_SHA256, preregistration_sha256=PREREGISTRATION_SHA256,
    )
    required = set(STALE_CROSS_GENERATION_PATHS) | set(LATER_GCSPO_PATHS)
    if not required.issubset(manifest["implementation_paths"]):
        raise SystemExit("stale/later GCSPO path coverage is incomplete")
    if not set(REQUIRED_INTERNAL_DEPENDENCY_PATHS).issubset(
            manifest["internal_import_closure_paths"]):
        raise SystemExit("required direct internal dependency closure is incomplete")
    red = strict_json_bytes((artifact / "red_report.json").read_bytes(), "RED report")
    green = strict_json_bytes((artifact / "green_report.json").read_bytes(), "GREEN report")
    handoff = {
        "schema": HANDOFF_SCHEMA,
        "state": "READY_FOR_FRESH_INDEPENDENT_READ_ONLY_REVIEW",
        "wrapper_commit_binding": "COMMIT_CONTAINING_THIS_DOCUMENT",
        "wrapper_parent_sha": target,
        "target_commit": target,
        "invocation_id": INVOCATION_ID,
        "nonce": INVOCATION_NONCE,
        "same_invocation_retry": False,
        "predecessor_freeze_commit": PREDECESSOR_FREEZE_COMMIT,
        "invalid_evidence_commit": INVALID_EVIDENCE_COMMIT,
        "repair_scope": "PYTHON_IMPORT_RESOLUTION_EXACT_DOCUMENT_SCHEMA_AND_CANONICAL_ARTIFACT_ROOT_ONLY",
        "config_sha256": CONFIG_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "red_green": {"red": red, "green": green},
        "manifest_coverage": {
            "total_rows": len(manifest["files"]),
            "stale_rows_required_and_present": list(STALE_CROSS_GENERATION_PATHS),
            "later_gcspo_rows_required_and_present": list(LATER_GCSPO_PATHS),
            "required_direct_internal_dependencies": list(REQUIRED_INTERNAL_DEPENDENCY_PATHS),
            "internal_import_closure_paths": manifest["internal_import_closure_paths"],
            "missing_rows": 0, "extra_rows": 0,
        },
        "protected": {"access_count": 0, "marker_present": False,
                         "ledger_size_bytes": 0, "authorized": False},
        "prior_independent_rejection": {
            "wrapper_commit": REJECTED_WRAPPER_COMMIT,
            "target_commit": REJECTED_TARGET_COMMIT,
            "verdict": "REJECT",
            "blocking_findings": [
                "TRANSITIVE_INTERNAL_IMPORT_CLOSURE_MISSING",
                "PROTECTED_STATE_SCHEMA_TYPE_VALUE_NOT_STRICT",
            ],
        },
        "latest_independent_rejection": {
            "wrapper_commit": SECOND_REJECTED_WRAPPER_COMMIT,
            "target_commit": SECOND_REJECTED_TARGET_COMMIT,
            "verdict": "REJECT",
            "blocking_findings": [
                "PACKAGE_INIT_RELATIVE_IMPORT_RESOLUTION_FAIL_OPEN",
                "DOCUMENT_SCHEMA_AND_PRIMITIVE_TYPE_FAIL_OPEN",
                "ARTIFACT_ROOT_CANONICAL_PATH_FAIL_OPEN",
            ],
        },
        "invalid_artifact_root_preserved": True,
        "push_performed": False,
        "independent_review_command":
            "python3 scripts/verify_gcspo_successor_freeze.py --expected-wrapper-commit $(git rev-parse HEAD)",
    }
    verify_handoff_protected_state(handoff)
    _write(artifact / "implementation_manifest.json", manifest)
    _write(artifact / "review_handoff.json", handoff)
    print(json.dumps({"status": "BUILT", "target_commit": target,
                      "manifest_rows": len(manifest["files"]),
                      "stale_rows": len(STALE_CROSS_GENERATION_PATHS),
                      "later_rows": len(LATER_GCSPO_PATHS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
