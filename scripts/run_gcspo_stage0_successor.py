#!/usr/bin/env python3
"""One fixed, native protected entrypoint for the reviewed GCSPO successor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_artifacts import (  # noqa: E402
    FROZEN_HASHES, canonical_write_json, quarantine_failed_final_verdict,
)
from gnss_doppler_lab.gcspo_capabilities import validate_preaccess_capabilities  # noqa: E402
from gnss_doppler_lab.gcspo_core import AccessGate  # noqa: E402
from gnss_doppler_lab.gcspo_evaluate import (  # noqa: E402
    run_one_shot, validate_clean_contrast_preaccess,
)
from gnss_doppler_lab.gcspo_freeze import validate_protected_manifest_inventory  # noqa: E402
from gnss_doppler_lab.gcspo_successor_launch import (  # noqa: E402
    ARTIFACT_RELATIVE, AUTHORIZATION_RELATIVE, CONTROL_RELATIVE, INVOCATION_ID,
    INVOCATION_NONCE, MARKER_RELATIVE, PROTECTED_PROVENANCE_SCHEMA,
    RECEIPT_RELATIVE, prepare_successor_valid_artifact_manifest, verify_preclaim,
)

ARTIFACT_DIR = ROOT / ARTIFACT_RELATIVE
CONFIG = ARTIFACT_DIR / "config.json"
CONTROL = ROOT / CONTROL_RELATIVE
MARKER = ROOT / MARKER_RELATIVE
LEDGER = ARTIFACT_DIR / "access_ledger.jsonl"


def _clean_identities(control: dict) -> list[dict]:
    return [{"path": str(ROOT / row["destination"]), "sha256": row["sha256"],
             "size_bytes": row["size_bytes"]}
            for row in control["clean_bundle"]["files"]]


def preflight() -> dict:
    """Complete all Git, review, clean-data, and protected-metadata checks."""
    result = verify_preclaim(ROOT, check_remote=True)
    control = result["control"]
    inventory = json.loads((ARTIFACT_DIR / "data_inventory.json").read_text())
    capabilities = validate_preaccess_capabilities(
        json.loads((ARTIFACT_DIR / "protected_capabilities.json").read_text()))
    manifest_identities = validate_protected_manifest_inventory(
        inventory, required=tuple(capabilities["available"]))
    for scenario, capability in capabilities["available"].items():
        if manifest_identities[scenario] != capability["manifest_identity"]:
            raise ValueError(f"{scenario} inventory/capability identity mismatch")
    clean_identities = _clean_identities(control)
    validate_clean_contrast_preaccess(ARTIFACT_DIR, clean_identities)
    return {**result, "inventory": inventory, "capabilities": capabilities,
            "manifest_identities": manifest_identities,
            "clean_identities": clean_identities}


def claim(wrapper_commit: str, target_commit: str) -> dict:
    """Claim the fixed marker once with O_EXCL; caller must finish preflight first."""
    document = {
        "schema": "gnss-doppler-lab.gcspo-stage0.successor-protected-run-start.v1",
        "protected_run_count": 1, "authorization_wrapper_commit": wrapper_commit,
        "target_commit": target_commit, "invocation_id": INVOCATION_ID,
        "nonce": INVOCATION_NONCE,
    }
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    descriptor = os.open(MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(MARKER.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return document


def protected() -> int:
    checked = preflight()
    wrapper = checked["freeze_sha"]
    target = checked["target_commit"]
    claim(wrapper, target)
    canonical_write_json(ARTIFACT_DIR / "protected_run_provenance.json", {
        "schema": PROTECTED_PROVENANCE_SCHEMA,
        "authorization_wrapper_commit": wrapper, "target_commit": target,
        "invocation_id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
        "receipt_path": RECEIPT_RELATIVE,
        "execution_freeze_path": AUTHORIZATION_RELATIVE,
    })
    gate = AccessGate(LEDGER)
    gate.set_preflight(clean_only_pass=True, reviews_pass=True, freeze_sha=wrapper,
                       frozen_hashes=FROZEN_HASHES)
    gate.set_remote_sync(local_sha=wrapper, remote_sha=wrapper, ahead=0, behind=0,
                         clean=True)
    verdict = run_one_shot(
        artifact_dir=ARTIFACT_DIR, repo_root=ROOT, inventory=checked["inventory"],
        gate=gate, manifest_identities=checked["manifest_identities"],
        clean_identities=checked["clean_identities"],
        capabilities=checked["capabilities"],
    )
    prepare_successor_valid_artifact_manifest(ARTIFACT_DIR)
    print(f"PROTECTED_ONE_SHOT_PASS verdict={verdict}", flush=True)
    return 0


def main() -> int:
    claimed = False
    try:
        checked = preflight()
        wrapper, target = checked["freeze_sha"], checked["target_commit"]
        claim(wrapper, target)
        claimed = True
        canonical_write_json(ARTIFACT_DIR / "protected_run_provenance.json", {
            "schema": PROTECTED_PROVENANCE_SCHEMA,
            "authorization_wrapper_commit": wrapper, "target_commit": target,
            "invocation_id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
            "receipt_path": RECEIPT_RELATIVE,
            "execution_freeze_path": AUTHORIZATION_RELATIVE,
        })
        gate = AccessGate(LEDGER)
        gate.set_preflight(clean_only_pass=True, reviews_pass=True, freeze_sha=wrapper,
                           frozen_hashes=FROZEN_HASHES)
        gate.set_remote_sync(local_sha=wrapper, remote_sha=wrapper, ahead=0, behind=0,
                             clean=True)
        verdict = run_one_shot(
            artifact_dir=ARTIFACT_DIR, repo_root=ROOT, inventory=checked["inventory"],
            gate=gate, manifest_identities=checked["manifest_identities"],
            clean_identities=checked["clean_identities"],
            capabilities=checked["capabilities"],
        )
        prepare_successor_valid_artifact_manifest(ARTIFACT_DIR)
        print(f"PROTECTED_ONE_SHOT_PASS verdict={verdict}", flush=True)
        return 0
    except Exception as exc:
        if claimed:
            quarantine_failed_final_verdict(ARTIFACT_DIR)
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
