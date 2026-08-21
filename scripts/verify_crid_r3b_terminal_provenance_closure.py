#!/usr/bin/env python3
"""Fail-closed verifier for the CRID R3b terminal provenance closure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ART = ROOT / "artifacts/crid_stage0_r3b_terminal_provenance_closure"
R3A_ART = ROOT / "artifacts/crid_stage0_r3a_independent_reference_estimand_repair"
TARGET_COMMIT = "aa3833fb73ae572521e3a3ac8f2b865d3aac0307"
R3A_VERIFIER_SHA256 = "6609e918df671aab5db9894533c0a6b0da5d435d3370ca48bcb89abb3eb02a98"
R3A_MANIFEST_SHA256 = "5fd07fda8b4fe01ee34defe240ed2a024f1faf425aaa0a6463eade82130cfd7b"
STALE_LOG_SHA256 = "948dc92c57453cc4744d4017d29caaa5b0bba8edd7ded072948e1608d7e7a1c0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(payload)
    return digest.hexdigest()


def verify_manifest(artifact: Path) -> dict:
    """Verify all files bound by an artifact manifest without trusting paths."""
    artifact = Path(artifact)
    try:
        manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
        entries = manifest["files"]
        checks = []
        seen = set()
        for entry in entries:
            relative = Path(entry["path"])
            safe = not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() not in seen
            seen.add(relative.as_posix())
            path = artifact / relative
            match = (
                safe
                and path.is_file()
                and path.stat().st_size == int(entry["size_bytes"])
                and sha256_file(path) == entry["sha256"]
            )
            checks.append({"path": relative.as_posix(), "match": match})
        passed = (
            manifest.get("schema") == "gnss-doppler-lab.crid-r3b-artifact-manifest.v1"
            and manifest.get("status") == "PASS"
            and manifest.get("file_count") == len(entries)
            and len(seen) == len(entries)
            and all(row["match"] for row in checks)
        )
        return {"status": "PASS" if passed else "FAIL", "checks": checks, "manifest": manifest}
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "checks": [], "error": f"{type(exc).__name__}: {exc}"}


def verify_artifact(artifact: Path) -> dict:
    artifact = Path(artifact)
    manifest_result = verify_manifest(artifact)
    checks = {"manifest": manifest_result["status"] == "PASS"}
    try:
        manifest = manifest_result["manifest"]
        listed = {entry["path"] for entry in manifest["files"]}
        required = {
            "README.md",
            "terminal_attestation.json",
            "historical_evidence_classification.json",
            "final_verdict.json",
            "logs/r3a_verifier_stdout.txt",
        }
        checks["required_files"] = required.issubset(listed)

        attestation = json.loads((artifact / "terminal_attestation.json").read_text())
        history = json.loads((artifact / "historical_evidence_classification.json").read_text())
        verdict = json.loads((artifact / "final_verdict.json").read_text())

        checkout = attestation["source_checkout"]
        bindings = attestation["bindings"]
        execution = attestation["execution"]
        parsed_stdout = json.loads(execution["stdout"])
        stdout_log = (artifact / "logs/r3a_verifier_stdout.txt").read_text()
        checks["target_commit"] = checkout["target_commit_sha"] == TARGET_COMMIT and checkout["fresh_clone"] is True
        checks["r3a_bindings"] = (
            bindings["verifier_sha256"] == R3A_VERIFIER_SHA256
            and bindings["artifact_manifest_sha256"] == R3A_MANIFEST_SHA256
            and sha256_file(ROOT / bindings["verifier_path"]) == R3A_VERIFIER_SHA256
            and sha256_file(ROOT / bindings["artifact_manifest_path"]) == R3A_MANIFEST_SHA256
        )
        checks["execution"] = (
            execution["argv"] == ["python3", "scripts/verify_crid_r3a_estimand_repair.py"]
            and execution["exit_code"] == 0
            and execution["stderr"] == ""
            and execution["stdout"] == stdout_log
            and parsed_stdout["status"] == "PASS"
            and parsed_stdout["verdict"] == "INDEPENDENT_REFERENCE_ESTIMAND_REPAIR_PASS"
            and execution["parsed_result"] == {
                "status": "PASS",
                "verdict": "INDEPENDENT_REFERENCE_ESTIMAND_REPAIR_PASS",
            }
        )

        stale = history["preserved_evidence"][0]
        stale_path = ROOT / stale["path"]
        stale_payload = json.loads(stale_path.read_text())
        checks["historical_evidence"] = (
            history["classification"] == "HISTORICAL_STALE_PRE_FINALIZATION_EVIDENCE"
            and history["preservation"] == "PRESERVED_UNMODIFIED"
            and history["superseding_attestation"] == "terminal_attestation.json"
            and stale["sha256"] == STALE_LOG_SHA256
            and stale["size_bytes"] == 285
            and sha256_file(stale_path) == STALE_LOG_SHA256
            and stale_payload["status"] == "PASS"
            and stale_payload["verdict"] == "INCONCLUSIVE_REFERENCE_PROVENANCE"
        )

        guards = attestation["scope_guards"]
        forbidden = (
            "phase_a_executed",
            "crid_score_computed",
            "c1_c2_c3_replay_executed",
            "attack_data_accessed",
            "generator_modified",
            "estimator_modified",
            "controls_modified",
            "thresholds_modified",
            "r3_scientific_artifact_modified",
            "r3a_scientific_artifact_modified",
            "existing_verifier_modified",
        )
        checks["scope_guards"] = all(guards[name] is False for name in forbidden)
        checks["terminal_verdict"] = (
            verdict["status"] == "PASS"
            and verdict["verdict"] == "TERMINAL_PROVENANCE_CLOSURE_PASS"
            and verdict["next_state"] == "READY_TO_REPEAT_CRID_PHASE_A"
            and verdict["phase_a_executed"] is False
            and verdict["historical_evidence_preserved"] is True
        )
    except (IndexError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        checks["semantic_contract"] = False
        error = f"{type(exc).__name__}: {exc}"
    else:
        error = None

    passed = bool(checks) and all(checks.values())
    result = {
        "schema": "gnss-doppler-lab.crid-r3b-terminal-provenance-verifier.v1",
        "checks": checks,
        "status": "PASS" if passed else "FAIL",
        "verdict": "TERMINAL_PROVENANCE_CLOSURE_PASS" if passed else "TERMINAL_PROVENANCE_CLOSURE_FAIL",
    }
    if error is not None:
        result["error"] = error
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ART)
    args = parser.parse_args()
    result = verify_artifact(args.artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
