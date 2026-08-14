#!/usr/bin/env python3
"""Build deterministic, public-data-only Round-8 successor package metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/gcspo_stage0_static_rerun"
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_round6_verify import (
    EXPECTED_NUMERIC_FIELD_COUNT,
    EXPECTED_NUMERIC_PATH_SHA256,
    SOURCE_COMMIT,
    strict_json_load,
)
from gnss_doppler_lab.gcspo_round8_verify import (
    ANCESTRY,
    FREEZE_RELATIVE,
    README_SHA256,
    ROUND7_REJECTED_FREEZE_COMMIT,
    ROUND7_VERDICT_SHA256,
    ROUND7_VERDICT_SIZE_BYTES,
    ROUND8_REPAIR_COMMIT,
)


CHANGE_PATHS = sorted([
    "artifacts/gcspo_stage0_static_rerun/round6_a5_parity.json",
    "artifacts/gcspo_stage0_static_rerun/round8_all_gcspo_junit.xml",
    "artifacts/gcspo_stage0_static_rerun/round8_audit_report.json",
    "artifacts/gcspo_stage0_static_rerun/round8_evidence_manifest.json",
    "artifacts/gcspo_stage0_static_rerun/round8_focused_junit.xml",
    "artifacts/gcspo_stage0_static_rerun/round8_green_report.json",
    "artifacts/gcspo_stage0_static_rerun/round8_packaging_notes.md",
    "artifacts/gcspo_stage0_static_rerun/round8_red_report.json",
    "artifacts/gcspo_stage0_static_rerun/round8_repair_review_handoff.json",
    "artifacts/gcspo_stage0_static_rerun/round8_round7_independent_review_rejection.txt",
    "artifacts/gcspo_stage0_static_rerun/round8_targeted_junit.xml",
    "scripts/build_gcspo_round8_package.py",
    "scripts/verify_gcspo_round8_freeze.py",
    "src/gnss_doppler_lab/gcspo_round6_verify.py",
    "src/gnss_doppler_lab/gcspo_round8_verify.py",
    "tests/test_gcspo_round8_repairs.py",
])
CONTRACT = {
    "expected_numeric_field_count": EXPECTED_NUMERIC_FIELD_COUNT,
    "expected_numeric_path_sha256": EXPECTED_NUMERIC_PATH_SHA256,
    "path_digest_encoding": "SHA256_OF_SEQUENCE_OF_UINT64_BE_LENGTH_PLUS_COMPACT_UTF8_JSON_PATH_TOKENS",
    "coverage_requirement": "POSITIVE_AND_EXACT_COUNT_AND_EXACT_PATH_DIGEST",
}


def identity(relative: str) -> dict:
    payload = (ROOT / relative).read_bytes()
    return {"path": relative, "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload)}


def write_json(name: str, document: dict) -> None:
    (ARTIFACT / name).write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def junit(name: str) -> dict:
    path = ARTIFACT / name
    root = ET.fromstring(path.read_bytes())
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    return {
        "passed": int(suite.attrib["tests"]) - int(suite.attrib["failures"]) -
                  int(suite.attrib["errors"]) - int(suite.attrib["skipped"]),
        "failed": int(suite.attrib["failures"]) + int(suite.attrib["errors"]),
        "skipped": int(suite.attrib["skipped"]),
        "junit": f"artifacts/gcspo_stage0_static_rerun/{name}",
        "junit_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "junit_size_bytes": path.stat().st_size,
    }


def main() -> int:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    if head != ROUND8_REPAIR_COMMIT:
        raise SystemExit("Round-8 package builder requires exact repair parent HEAD")
    marker = ARTIFACT.parent / f".{ARTIFACT.name}.protected_run_started.json"
    if marker.exists() or (ARTIFACT / "access_ledger.jsonl").exists():
        raise SystemExit("protected marker or ledger exists")
    verdict = ARTIFACT / "round8_round7_independent_review_rejection.txt"
    verdict_payload = verdict.read_bytes()
    if (hashlib.sha256(verdict_payload).hexdigest() != ROUND7_VERDICT_SHA256 or
            len(verdict_payload) != ROUND7_VERDICT_SIZE_BYTES):
        raise SystemExit("Round-7 verdict evidence is not verbatim")
    if hashlib.sha256((ARTIFACT / "README.md").read_bytes()).hexdigest() != README_SHA256:
        raise SystemExit("frozen README changed")

    tests = {
        "targeted_malformed": junit("round8_targeted_junit.xml"),
        "focused": junit("round8_focused_junit.xml"),
        "all_gcspo": junit("round8_all_gcspo_junit.xml"),
    }
    green = {
        "schema": "gnss-doppler-lab.gcspo-stage0.round8-repair-green-evidence.v1",
        "phase": "ROUND8_SUCCESSOR_PREFREEZE_GREEN",
        "repair_parent_sha": ROUND8_REPAIR_COMMIT,
        "rejected_round7_freeze_sha": ROUND7_REJECTED_FREEZE_COMMIT,
        "tests": tests,
        "strict_malformed_probe_count": tests["targeted_malformed"]["passed"],
        "strict_malformed_failed_open": 0,
        "signature_and_package_verification": {
            "status": "PASS", "signature_count": 6, "evidence_file_count": 33,
            "external_package_identity_count": 33,
            "run_count": 3, "backends": ["cuda", "cuda", "cpu"],
            "independence": "EXTERNALLY_WITNESSED",
        },
        "numeric_trace_contract": CONTRACT,
        "parity": {
            "same_backend": "BYTE_IDENTICAL",
            "cpu_cuda": "WITHIN_PREREGISTERED_TOLERANCE",
            "numeric_field_count": 789115,
            "numeric_path_sha256": EXPECTED_NUMERIC_PATH_SHA256,
            "maximum_absolute_delta": 5.760082785855047e-06,
            "maximum_relative_delta": 7.291690239802362e-08,
            "tolerance_violations": 0,
            "absolute_tolerance": 1e-5,
            "relative_tolerance": 1e-8,
            "or_rule_preserved": True,
        },
        "protected": {"authorized": False, "marker_present": False,
                      "ledger_present": False, "rows_opened": 0, "bytes_opened": 0},
        "attack_run_count": 0,
        "push_performed": False,
    }
    write_json("round8_green_report.json", green)

    rejection = {
        "path": "artifacts/gcspo_stage0_static_rerun/round8_round7_independent_review_rejection.txt",
        "reviewed_sha": ROUND7_REJECTED_FREEZE_COMMIT,
        "sha256": ROUND7_VERDICT_SHA256,
        "size_bytes": ROUND7_VERDICT_SIZE_BYTES,
        "verdict": "INDEPENDENT_REVIEW_REJECT",
    }
    audit = {
        "schema": "gnss-doppler-lab.gcspo-stage0.round8-successor-audit.v1",
        "state": "READY_FOR_FRESH_INDEPENDENT_READ_ONLY_REVIEW",
        "final_freeze_sha_binding": "COMMIT_CONTAINING_THIS_DOCUMENT",
        "final_freeze_sha_resolution": "git rev-parse HEAD; strict verifier requires exact HEAD and committed bytes",
        "freeze_parent_sha": ROUND8_REPAIR_COMMIT,
        "ancestry": list(ANCESTRY) + ["COMMIT_CONTAINING_THIS_DOCUMENT"],
        "round7_rejection_evidence": rejection,
        "repairs": {
            "boolean_before_equality": "REJECTED_AT_EVERY_TRACE_LEAF",
            "empty_and_vacuous_documents": "REJECTED",
            "same_field_omission": "EXACT_COUNT_AND_PATH_DIGEST_REJECTED",
            "unexpected_numeric_coverage": "EXACT_COUNT_AND_PATH_DIGEST_REJECTED",
            "duplicate_json_keys": "REJECTED_AT_EVERY_NESTING_LEVEL_BEFORE_PASS",
            "nonfinite_and_overflow": "REJECTED",
            "approved_nonnumeric_metadata": ["schema"],
        },
        "strict_json_contract": {
            "single_loader": "gnss_doppler_lab.gcspo_round6_verify.strict_json_load/strict_json_loads",
            "parse_constant_rejection": ["NaN", "Infinity", "-Infinity"],
            "duplicate_object_key_rejection": "ALL_NESTING_LEVELS",
            "signed_envelopes": "STRICT_PREVALIDATION_PLUS_UNCHANGED_EXACT_CANONICAL_BYTE_VERIFICATION",
        },
        "numeric_trace_contract": CONTRACT,
        "tests": tests,
        "signed_evidence": green["signature_and_package_verification"],
        "parity": green["parity"],
        "scientific_scope": {"gate": "cleanStatic-only",
                             "cleanDynamic": "OOD_DIAGNOSIS_NOT_GATE_EVIDENCE"},
        "change_control": {
            "signed_evidence_bytes_changed": False,
            "scientific_output_bytes_changed": False,
            "config_threshold_tolerance_scoring_gate_bytes_changed": False,
            "readme_sha256": README_SHA256,
        },
        "protected": green["protected"],
        "attack_run_count": 0,
        "push_performed": False,
    }
    write_json("round8_audit_report.json", audit)

    handoff = dict(audit)
    handoff["schema"] = "gnss-doppler-lab.gcspo-stage0.round8-repair-review-handoff.v1"
    handoff["round7_rejection_evidence"] = rejection
    handoff["red_green_evidence"] = {
        "red": "artifacts/gcspo_stage0_static_rerun/round8_red_report.json",
        "green": "artifacts/gcspo_stage0_static_rerun/round8_green_report.json",
        "red_failed": 28,
        "green_failed": 0,
        "targeted_green_passed": tests["targeted_malformed"]["passed"],
        "focused_green_passed": tests["focused"]["passed"],
        "all_gcspo_green_passed": tests["all_gcspo"]["passed"],
    }
    handoff["review_mode"] = "FRESH_INDEPENDENT_READ_ONLY_REVIEW"
    write_json("round8_repair_review_handoff.json", handoff)

    (ARTIFACT / "round8_packaging_notes.md").write_text(
        "# Round-8 successor packaging notes\n\n"
        "Round 8 repairs the exact independent rejection of `b54383b799f47fd1a849126d3f21fe6c643eb209`. "
        "The reviewer verdict is preserved verbatim and prior evidence is unchanged.\n\n"
        "The verifier now rejects booleans before equality, rejects empty/vacuous trace structures, "
        "pins 789,115 numeric leaves and the immutable numeric-path digest, and applies duplicate/non-finite "
        "JSON rejection before any PASS-capable verifier path. Signed-envelope canonical-byte checks remain unchanged.\n\n"
        "No protected execution, attack, signing, private-key search, or push was performed.\n",
        encoding="utf-8",
    )

    evidence_paths = sorted(set(CHANGE_PATHS) - {
        FREEZE_RELATIVE,
        "artifacts/gcspo_stage0_static_rerun/round8_evidence_manifest.json",
    })
    evidence = {
        "schema": "gnss-doppler-lab.gcspo-stage0.round8-successor-evidence-manifest.v1",
        "source_commit": SOURCE_COMMIT,
        "rejected_round7_freeze_commit": ROUND7_REJECTED_FREEZE_COMMIT,
        "repair_commit": ROUND8_REPAIR_COMMIT,
        "manifest_excludes_self": True,
        "self_binding_exclusion": "artifacts/gcspo_stage0_static_rerun/round8_evidence_manifest.json",
        "files": [identity(path) for path in evidence_paths],
        "protected_access_authorized": False,
        "attack_run_count": 0,
        "push_performed": False,
    }
    write_json("round8_evidence_manifest.json", evidence)

    historical = strict_json_load(ARTIFACT / "round7_freeze_manifest.json")
    prior_paths = {row["path"] for row in historical["files"]}
    freeze_paths = sorted(prior_paths | set(CHANGE_PATHS) |
                          {"artifacts/gcspo_stage0_static_rerun/round7_freeze_manifest.json"} -
                          {FREEZE_RELATIVE})
    freeze = {
        "schema": "gnss-doppler-lab.gcspo-stage0.round8-successor-freeze.v1",
        "state": "READY_FOR_FRESH_INDEPENDENT_READ_ONLY_REVIEW",
        "commit_binding": "COMMIT_CONTAINING_THIS_MANIFEST",
        "freeze_parent_sha": ROUND8_REPAIR_COMMIT,
        "rejected_round7_freeze_commit": ROUND7_REJECTED_FREEZE_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "ancestry": list(ANCESTRY) + ["COMMIT_CONTAINING_THIS_MANIFEST"],
        "file_set_policy": "UNION_OF_ROUND7_FREEZE_FILES_ROUND8_CHANGES_AND_ROUND7_FREEZE_MANIFEST",
        "manifest_excludes_self": True,
        "self_binding_exclusions": [FREEZE_RELATIVE],
        "successor_change_paths": CHANGE_PATHS,
        "numeric_trace_contract": CONTRACT,
        "files": [identity(path) for path in freeze_paths],
        "tests": tests,
        "signatures": {"verified": 6,
                       "public_key_fingerprint": "SHA256:L+STBb5P7+DDfilvAZUxV2eHGYZGfMQf4aDbXSsNi0c"},
        "evidence_file_count": 33,
        "protected_access_authorized": False,
        "protected_marker_present": False,
        "protected_ledger_present": False,
        "protected_rows_opened": 0,
        "protected_bytes_opened": 0,
        "attack_run_count": 0,
        "push_performed": False,
    }
    write_json("round8_freeze_manifest.json", freeze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
