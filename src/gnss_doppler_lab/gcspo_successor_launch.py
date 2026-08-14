"""Fail-closed GCSPO successor launch contracts.

This module only verifies metadata, Git objects, and pre-access clean artifacts.  It
contains no protected-data reader and deliberately does not use the legacy
``verify_freeze_record`` path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

from .gcspo_successor_freeze import (
    implementation_paths_at_commit,
    verify_successor_manifest,
)

HISTORICAL_WRAPPER_COMMIT = "f8b7785bafc05c3fcc4839c0e7c0973ecbf8e46e"
HISTORICAL_TARGET_COMMIT = "13945ba1b53d40ad70ccdbbf318ec898b66644e2"
PREDECESSOR_FREEZE_COMMIT = "0ab94567938234ca925f0bb8fbaece41e7d5e4a3"
INVALID_EVIDENCE_COMMIT = "58943724ea278b754d388ec2dca0f3666ed6c8a2"
CONFIG_SHA256 = "0db816116b95b41db8b7af7379cd7411cc52d43b6428ae00ab02d6ccac19f4ad"
PREREGISTRATION_SHA256 = "715e11965854f785487e9d2c747718747c1d31cdd8603696ea7af126a45a70da"
BRANCH = "research/gcspo-stage0-launch-adapter"
REMOTE = "origin"
INVOCATION_ID = (
    "gcspo-stage0-successor-launch-"
    "77e586dfdb50a008ed2f0b31052e33bb700e191641e7ffbb4845860df15cf48e"
)
INVOCATION_NONCE = "023a99334eaf7c08d8f74a14fda7da363e77d0c54b605582e9a976437ce623a3"
ARTIFACT_RELATIVE = f"artifacts/gcspo_stage0_successor_launch/{INVOCATION_ID}"
CONTROL_RELATIVE = f"config/gcspo_successor_launch/{INVOCATION_ID}/launch_control.json"
RECEIPT_RELATIVE = f"config/gcspo_successor_launch/{INVOCATION_ID}/independent_review_receipt.json"
AUTHORIZATION_RELATIVE = f"config/gcspo_successor_launch/{INVOCATION_ID}/execution_freeze.json"
MARKER_RELATIVE = f"artifacts/gcspo_stage0_successor_launch/.{INVOCATION_ID}.protected_run_started.json"
LEDGER_RELATIVE = f"{ARTIFACT_RELATIVE}/access_ledger.jsonl"
LAUNCH_PROVENANCE_RELATIVE = f"{ARTIFACT_RELATIVE}/launch_provenance.json"

HISTORICAL_ARTIFACT_RELATIVE = (
    "artifacts/gcspo_stage0_static_rerun_successor/"
    "gcspo-stage0-successor-7be48c4411644ff3a9ec41c7701dfa01"
)
HISTORICAL_MANIFEST_RELATIVE = f"{HISTORICAL_ARTIFACT_RELATIVE}/implementation_manifest.json"
HISTORICAL_HANDOFF_RELATIVE = f"{HISTORICAL_ARTIFACT_RELATIVE}/review_handoff.json"
PREDECESSOR_ARTIFACT_RELATIVE = "artifacts/gcspo_stage0_static_rerun"
PREDECESSOR_MANIFEST_RELATIVE = f"{PREDECESSOR_ARTIFACT_RELATIVE}/implementation_manifest.json"
ARCHIVE_SOURCE = (
    "/home/ubuntu/projects/gnss-doppler-checkpoints/"
    "gcspo-stage0-static-rerun-invalid-0ab9456/preaccess_package_quarantine"
)

CLEAN_BUNDLE_RELATIVE_PATHS = (
    "README.md",
    "clean_a5_report.json",
    "clean_ablation_report.json",
    "clean_b0_report.json",
    "clean_only_report.json",
    "clean_reproduction_evidence.json",
    "config.json",
    "data_inventory.json",
    "implementation_resolution.json",
    "normal_model_summary.json",
    "physical_controls.json",
    "preregistration.json",
    "protected_capabilities.json",
    "reproduction_run_1.json",
    "reproduction_run_2.json",
    "reproductions/a5-round3-run-1-b341801/clean_a5_report.json",
    "reproductions/a5-round3-run-1-b341801/thresholds.json",
    "reproductions/a5-round3-run-2-b341801/clean_a5_report.json",
    "reproductions/a5-round3-run-2-b341801/thresholds.json",
    "source_commit.json",
    "thresholds.json",
)

DYNAMIC_RUNTIME_PATHS = (
    "artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt",
    "artifacts/ai_morph_gru_cleanStatic_q70_frame/validation_prn_node_scores.csv",
    "scripts/eval_btail_support_gate.py",
    "scripts/score_texbat_prn_node_gru.py",
    "scripts/train_prn_node_gru.py",
)

SUCCESSOR_VALID_ADDITIONS = frozenset({
    "reproductions", "launch_provenance.json", "protected_run_provenance.json",
    "protected_control_status.json",
})

_ARTIFACT_TARGET_PATHS = tuple(
    f"{ARTIFACT_RELATIVE}/{relative}" for relative in CLEAN_BUNDLE_RELATIVE_PATHS
) + (LEDGER_RELATIVE, LAUNCH_PROVENANCE_RELATIVE)
TARGET_DIFF_ALLOWLIST = tuple(sorted((
    *_ARTIFACT_TARGET_PATHS,
    CONTROL_RELATIVE,
    "scripts/finalize_gcspo_successor_authorization.py",
    "scripts/run_gcspo_stage0_successor.py",
    "scripts/verify_gcspo_successor_launch.py",
    "src/gnss_doppler_lab/gcspo_evaluate.py",
    "src/gnss_doppler_lab/gcspo_successor_launch.py",
    "src/gnss_doppler_lab/gcspo_verify.py",
    "tests/test_gcspo_successor_launch.py",
    "tests/test_gcspo_verifier.py",
)))

CONTROL_SCHEMA = "gnss-doppler-lab.gcspo-stage0.successor-launch-control.v1"
RECEIPT_SCHEMA = "gnss-doppler-lab.gcspo-stage0.successor-independent-review-receipt.v1"
AUTHORIZATION_SCHEMA = "gnss-doppler-lab.gcspo-stage0.successor-execution-freeze.v1"
LAUNCH_PROVENANCE_SCHEMA = "gnss-doppler-lab.gcspo-stage0.successor-launch-provenance.v1"
PROTECTED_PROVENANCE_SCHEMA = "gnss-doppler-lab.gcspo-stage0.successor-protected-run-provenance.v1"

_CONTROL_KEYS = {
    "schema", "state", "commit_binding", "invocation", "historical_successor",
    "predecessor", "scientific_inputs", "namespace", "archive_provenance",
    "clean_bundle", "runtime_closure", "target_diff_allowlist", "protected_state",
    "authorization", "prohibitions",
}
_RECEIPT_KEYS = {"schema", "state", "reviewer", "target_commit", "invocation_id", "nonce", "evidence"}
_AUTHORIZATION_KEYS = {
    "schema", "state", "target_commit", "wrapper_commit_binding", "invocation_id",
    "nonce", "control_path", "receipt_path", "runtime_files",
    "runtime_manifest_sha256", "clean_bundle_manifest_sha256", "access_gate_identity",
}


def prepare_successor_valid_artifact_manifest(artifact_dir: str | Path) -> dict:
    """Package final artifacts while preserving successor-only evidence.

    The Round-5 artifact writer is byte-frozen, so the successor extends its
    packaging policy only for the duration of this call and restores the exact
    original module binding even when packaging fails.
    """
    from . import gcspo_artifacts

    original_additions = gcspo_artifacts.VALID_ADDITIONS
    gcspo_artifacts.VALID_ADDITIONS = set(original_additions) | set(SUCCESSOR_VALID_ADDITIONS)
    try:
        return gcspo_artifacts.prepare_valid_artifact_manifest(artifact_dir)
    finally:
        gcspo_artifacts.VALID_ADDITIONS = original_additions


def _run(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, text=True,
                            capture_output=True)
    return result.stdout.strip()


def _git_bytes(repo: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=repo,
                            capture_output=True)
    if result.returncode:
        raise ValueError(f"required Git object path is absent: {commit}:{relative}")
    return result.stdout


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes, label: str) -> dict:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def identity(path: str, payload: bytes) -> dict:
    return {"path": path, "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload)}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _exact_dict(value: object, keys: Iterable[str], label: str) -> dict:
    expected = set(keys)
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} exact schema/key set mismatch")
    return value


def _exact_value(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{label} exact type/value mismatch")


def _sha(value: object, label: str, length: int) -> str:
    if type(value) is not str or len(value) != length or any(
            char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase {length}-character hexadecimal string")
    return value


def _identity_rows(rows: object, label: str) -> list[dict]:
    if type(rows) is not list or not rows:
        raise ValueError(f"{label} must be a nonempty list")
    paths = []
    for index, row in enumerate(rows):
        record = _exact_dict(row, {"path", "sha256", "size_bytes"}, f"{label}[{index}]")
        if type(record["path"]) is not str or not record["path"]:
            raise ValueError(f"{label}[{index}] path type/value mismatch")
        _sha(record["sha256"], f"{label}[{index}] sha256", 64)
        if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
            raise ValueError(f"{label}[{index}] size exact type/value mismatch")
        paths.append(record["path"])
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{label} paths are not sorted and unique")
    return rows


def verify_historical_pair(repo: str | Path) -> dict:
    """Authenticate the frozen f8/13945 pair solely through committed Git objects."""
    root = Path(repo).resolve(strict=True)
    parent = _run(root, "rev-parse", f"{HISTORICAL_WRAPPER_COMMIT}^")
    if parent != HISTORICAL_TARGET_COMMIT:
        raise ValueError("historical wrapper immediate parent mismatch")
    changes = _run(root, "diff-tree", "--no-commit-id", "--name-only", "-r",
                   HISTORICAL_WRAPPER_COMMIT).splitlines()
    if sorted(changes) != sorted((HISTORICAL_MANIFEST_RELATIVE, HISTORICAL_HANDOFF_RELATIVE)):
        raise ValueError("historical wrapper exact diff mismatch")
    manifest = strict_json_bytes(
        _git_bytes(root, HISTORICAL_WRAPPER_COMMIT, HISTORICAL_MANIFEST_RELATIVE),
        "historical successor manifest",
    )
    verify_successor_manifest(
        manifest, root, expected_target_commit=HISTORICAL_TARGET_COMMIT,
        wrapper_commit=HISTORICAL_WRAPPER_COMMIT, require_repository_coverage=True,
    )
    rows = manifest.get("files")
    if type(rows) is not list or len(rows) != 69:
        raise ValueError("historical successor manifest must contain exactly 69 rows")
    if manifest.get("protected_access_authorized") is not False:
        raise ValueError("historical successor must remain explicitly unauthorized")
    handoff = strict_json_bytes(
        _git_bytes(root, HISTORICAL_WRAPPER_COMMIT, HISTORICAL_HANDOFF_RELATIVE),
        "historical successor handoff",
    )
    protected = handoff.get("protected")
    if protected != {"access_count": 0, "authorized": False,
                      "ledger_size_bytes": 0, "marker_present": False}:
        raise ValueError("historical handoff protected state mismatch")
    return {"wrapper_commit": HISTORICAL_WRAPPER_COMMIT,
            "target_commit": HISTORICAL_TARGET_COMMIT, "manifest_rows": 69,
            "protected_access_authorized": False}


def verify_launch_control(control: dict) -> bool:
    record = _exact_dict(control, _CONTROL_KEYS, "launch control")
    fixed = {
        "schema": CONTROL_SCHEMA,
        "state": "AWAITING_INDEPENDENT_REVIEW",
        "commit_binding": "COMMIT_CONTAINING_THIS_DOCUMENT",
        "invocation": {"id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
                       "same_invocation_retry": False},
        "historical_successor": {
            "implementation_target_commit": HISTORICAL_TARGET_COMMIT,
            "wrapper_commit": HISTORICAL_WRAPPER_COMMIT, "manifest_rows": 69,
            "structurally_approved": True, "protected_access_authorized": False,
        },
        "predecessor": {"freeze_commit": PREDECESSOR_FREEZE_COMMIT,
                        "invalid_evidence_commit": INVALID_EVIDENCE_COMMIT},
        "scientific_inputs": {"config_sha256": CONFIG_SHA256,
                              "preregistration_sha256": PREREGISTRATION_SHA256},
        "namespace": {"artifact_root": ARTIFACT_RELATIVE,
                      "control_path": CONTROL_RELATIVE,
                      "ledger_path": LEDGER_RELATIVE,
                      "marker_path": MARKER_RELATIVE},
        "archive_provenance": {
            "archive_root": ARCHIVE_SOURCE,
            "copy_policy": "EXACT_ALLOWLIST_ONLY_NO_BULK_COPY",
            "source_commit_for_authentication": PREDECESSOR_FREEZE_COMMIT,
        },
        "runtime_closure": {
            "historical_manifest_rows": 69,
            "policy": "TARGET_GCSPO_SCOPE_PLUS_TRANSITIVE_INTERNAL_IMPORTS_PLUS_DYNAMIC_B0",
            "required_dynamic_paths": list(DYNAMIC_RUNTIME_PATHS),
        },
        "target_diff_allowlist": list(TARGET_DIFF_ALLOWLIST),
        "protected_state": {"access_count": 0, "authorized": False,
                            "final_verdict_present": False, "ledger_size_bytes": 0,
                            "marker_present": False},
        "authorization": {"authorized": False, "execution_freeze_present": False,
                          "independent_review_receipt_present": False},
        "prohibitions": {"authorization_in_target": True, "marker_claim": True,
                         "protected_evaluation": True, "push": True},
    }
    for key, expected in fixed.items():
        _exact_value(record[key], expected, f"launch control {key}")
    bundle = _exact_dict(record["clean_bundle"], {"policy", "files"},
                         "launch control clean bundle")
    _exact_value(bundle["policy"],
                 "EXACT_DESTINATION_BOUND_PREDECESSOR_GIT_AND_MANIFEST_AUTHENTICATION",
                 "launch control clean bundle policy")
    rows = bundle["files"]
    if type(rows) is not list or len(rows) != len(CLEAN_BUNDLE_RELATIVE_PATHS):
        raise ValueError("launch control clean bundle row count mismatch")
    destinations = []
    for index, row in enumerate(rows):
        item = _exact_dict(row, {"destination", "predecessor_manifest_authenticated",
                                 "sha256", "size_bytes", "source_commit", "source_path"},
                           f"launch control clean bundle row {index}")
        expected_relative = CLEAN_BUNDLE_RELATIVE_PATHS[index]
        _exact_value(item["destination"], f"{ARTIFACT_RELATIVE}/{expected_relative}",
                     f"clean bundle row {index} destination")
        _exact_value(item["source_path"], f"{PREDECESSOR_ARTIFACT_RELATIVE}/{expected_relative}",
                     f"clean bundle row {index} source")
        _exact_value(item["source_commit"], PREDECESSOR_FREEZE_COMMIT,
                     f"clean bundle row {index} source commit")
        if type(item["predecessor_manifest_authenticated"]) is not bool:
            raise ValueError("clean bundle predecessor manifest flag exact type mismatch")
        _sha(item["sha256"], "clean bundle sha256", 64)
        if type(item["size_bytes"]) is not int or item["size_bytes"] < 0:
            raise ValueError("clean bundle size exact type/value mismatch")
        destinations.append(item["destination"])
    if destinations != sorted(destinations) or len(destinations) != len(set(destinations)):
        raise ValueError("clean bundle destinations are not sorted and unique")
    return True


def runtime_rows_at_commit(repo: str | Path, target_commit: str) -> list[dict]:
    root = Path(repo).resolve(strict=True)
    paths = sorted(set(implementation_paths_at_commit(root, target_commit)) |
                   set(DYNAMIC_RUNTIME_PATHS))
    rows = []
    for relative in paths:
        mode_line = _run(root, "ls-tree", target_commit, "--", relative)
        if not mode_line or not mode_line.startswith("100644 ") and not mode_line.startswith("100755 "):
            raise ValueError(f"runtime closure member is absent or not a regular Git file: {relative}")
        rows.append(identity(relative, _git_bytes(root, target_commit, relative)))
    missing = sorted(set(DYNAMIC_RUNTIME_PATHS) - {row["path"] for row in rows})
    if missing:
        raise ValueError(f"dynamic B0 runtime closure is incomplete: {missing}")
    return rows


def _predecessor_manifest_identities(root: Path) -> dict[str, tuple[str, int]]:
    manifest = strict_json_bytes(
        _git_bytes(root, PREDECESSOR_FREEZE_COMMIT, PREDECESSOR_MANIFEST_RELATIVE),
        "predecessor implementation manifest",
    )
    result = {}
    for row in manifest.get("clean_scientific_artifacts", []):
        if type(row) is not dict:
            raise ValueError("predecessor clean manifest row is malformed")
        path = row.get("path")
        marker = f"/{PREDECESSOR_ARTIFACT_RELATIVE}/"
        if type(path) is not str or marker not in path:
            continue
        relative = path.split(marker, 1)[1]
        result[relative] = (row.get("sha256"), row.get("size_bytes"))
    return result


def verify_clean_bundle_at_target(repo: str | Path, target_commit: str,
                                  control: dict) -> bool:
    root = Path(repo).resolve(strict=True)
    verify_launch_control(control)
    predecessor = _predecessor_manifest_identities(root)
    for row in control["clean_bundle"]["files"]:
        source = _git_bytes(root, PREDECESSOR_FREEZE_COMMIT, row["source_path"])
        destination = _git_bytes(root, target_commit, row["destination"])
        observed = identity(row["destination"], destination)
        if destination != source or observed["sha256"] != row["sha256"] or \
                observed["size_bytes"] != row["size_bytes"]:
            raise ValueError(f"clean bundle destination/source Git identity mismatch: {row['destination']}")
        relative = row["source_path"].removeprefix(PREDECESSOR_ARTIFACT_RELATIVE + "/")
        expected_manifest = predecessor.get(relative)
        authenticated = expected_manifest == (row["sha256"], row["size_bytes"])
        if row["predecessor_manifest_authenticated"] is not authenticated:
            raise ValueError(f"clean bundle predecessor manifest authentication mismatch: {relative}")
        if not authenticated and relative != "README.md":
            raise ValueError(f"required predecessor implementation manifest row absent: {relative}")
    return True


def _no_symlink_components(root: Path, path: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes bundle root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")


def verify_destination_bundle(root: str | Path, rows: list[dict], *,
                              expected_paths: Iterable[str]) -> bool:
    bundle = Path(root)
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError("bundle root must be a regular directory without symlink aliasing")
    expected = list(expected_paths)
    destinations = [row.get("destination") for row in rows]
    if destinations != expected or expected != sorted(expected) or len(expected) != len(set(expected)):
        raise ValueError("bundle destination path set is missing, extra, unsorted, or duplicated")
    observed_paths = sorted(path.relative_to(bundle).as_posix() for path in bundle.rglob("*")
                            if path.is_file() or path.is_symlink())
    if observed_paths != expected:
        raise ValueError(f"bundle exact-set mismatch: expected={expected} observed={observed_paths}")
    for row in rows:
        relative = row["destination"]
        path = bundle / relative
        _no_symlink_components(bundle, path, f"bundle member {relative}")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"bundle member is not a regular file: {relative}")
        payload = path.read_bytes()
        if type(row.get("sha256")) is not str or type(row.get("size_bytes")) is not int:
            raise ValueError(f"bundle identity type mismatch: {relative}")
        if hashlib.sha256(payload).hexdigest() != row["sha256"] or len(payload) != row["size_bytes"]:
            raise ValueError(f"bundle identity hash/size mismatch: {relative}")
    return True


def verify_preaccess_worktree_bundle(repo: str | Path, control: dict) -> bool:
    root = Path(repo).resolve(strict=True)
    artifact = root / ARTIFACT_RELATIVE
    rows = [{"destination": row["destination"].removeprefix(ARTIFACT_RELATIVE + "/"),
             "sha256": row["sha256"], "size_bytes": row["size_bytes"]}
            for row in control["clean_bundle"]["files"]]
    ledger_row = {"destination": "access_ledger.jsonl",
                  "sha256": hashlib.sha256(b"").hexdigest(), "size_bytes": 0}
    provenance_payload = (artifact / "launch_provenance.json").read_bytes()
    provenance_row = {"destination": "launch_provenance.json",
                      "sha256": hashlib.sha256(provenance_payload).hexdigest(),
                      "size_bytes": len(provenance_payload)}
    complete = sorted([*rows, ledger_row, provenance_row], key=lambda row: row["destination"])
    return verify_destination_bundle(
        artifact, complete, expected_paths=[row["destination"] for row in complete])


def build_review_authorization_documents(
        control: dict, *, target_commit: str, reviewer: str, review_command: str,
        passed: int, findings: list[str], evidence_sha256: str,
        repo: str | Path | None = None, runtime_rows: list[dict] | None = None,
) -> tuple[dict, dict]:
    """Build (but never write) the two documents for a future independent wrapper."""
    verify_launch_control(control)
    _sha(target_commit, "review target commit", 40)
    _sha(evidence_sha256, "review evidence sha256", 64)
    if type(reviewer) is not str or not reviewer or type(review_command) is not str or not review_command:
        raise ValueError("independent reviewer identity/command is incomplete")
    if type(passed) is not int or passed <= 0 or type(findings) is not list or any(
            type(item) is not str for item in findings):
        raise ValueError("independent review evidence types are invalid")
    if findings:
        raise ValueError("authorization cannot be built with independent review findings")
    if runtime_rows is None:
        if repo is None:
            raise ValueError("repository is required to generate exact runtime rows")
        runtime_rows = runtime_rows_at_commit(repo, target_commit)
    _identity_rows(runtime_rows, "execution runtime files")
    receipt = {
        "schema": RECEIPT_SCHEMA, "state": "APPROVED", "reviewer": reviewer,
        "target_commit": target_commit, "invocation_id": INVOCATION_ID,
        "nonce": INVOCATION_NONCE,
        "evidence": {"command": review_command, "passed": passed,
                     "findings": findings, "sha256": evidence_sha256},
    }
    clean_rows = control["clean_bundle"]["files"]
    execution = {
        "schema": AUTHORIZATION_SCHEMA, "state": "VALID_FOR_PROTECTED_ACCESS",
        "target_commit": target_commit,
        "wrapper_commit_binding": "COMMIT_CONTAINING_THIS_DOCUMENT",
        "invocation_id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
        "control_path": CONTROL_RELATIVE, "receipt_path": RECEIPT_RELATIVE,
        "runtime_files": runtime_rows,
        "runtime_manifest_sha256": _canonical_hash(runtime_rows),
        "clean_bundle_manifest_sha256": _canonical_hash(clean_rows),
        "access_gate_identity": "COMMIT_CONTAINING_THIS_DOCUMENT",
    }
    return receipt, execution


def verify_wrapper_documents(control: dict, receipt: dict, execution: dict, *,
                             target_commit: str) -> bool:
    verify_launch_control(control)
    review = _exact_dict(receipt, _RECEIPT_KEYS, "independent review receipt")
    expected_review = {
        "schema": RECEIPT_SCHEMA, "state": "APPROVED", "target_commit": target_commit,
        "invocation_id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
    }
    for key, expected in expected_review.items():
        _exact_value(review[key], expected, f"independent review receipt {key}")
    if type(review["reviewer"]) is not str or not review["reviewer"]:
        raise ValueError("independent review receipt reviewer is absent")
    evidence = _exact_dict(review["evidence"], {"command", "passed", "findings", "sha256"},
                           "independent review receipt evidence")
    if type(evidence["command"]) is not str or not evidence["command"]:
        raise ValueError("independent review receipt command is absent")
    if type(evidence["passed"]) is not int or evidence["passed"] <= 0:
        raise ValueError("independent review receipt passed exact type/value mismatch")
    if type(evidence["findings"]) is not list or evidence["findings"]:
        raise PermissionError("independent review receipt contains blocking findings")
    _sha(evidence["sha256"], "independent review evidence sha256", 64)

    freeze = _exact_dict(execution, _AUTHORIZATION_KEYS, "execution freeze")
    fixed = {
        "schema": AUTHORIZATION_SCHEMA, "state": "VALID_FOR_PROTECTED_ACCESS",
        "target_commit": target_commit,
        "wrapper_commit_binding": "COMMIT_CONTAINING_THIS_DOCUMENT",
        "invocation_id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
        "control_path": CONTROL_RELATIVE, "receipt_path": RECEIPT_RELATIVE,
        "clean_bundle_manifest_sha256": _canonical_hash(control["clean_bundle"]["files"]),
        "access_gate_identity": "COMMIT_CONTAINING_THIS_DOCUMENT",
    }
    for key, expected in fixed.items():
        _exact_value(freeze[key], expected, f"execution freeze {key}")
    rows = _identity_rows(freeze["runtime_files"], "execution runtime files")
    _exact_value(freeze["runtime_manifest_sha256"], _canonical_hash(rows),
                 "execution freeze runtime manifest hash")
    return True


def authorize_launch(
        control: dict, receipt: dict | None, execution: dict | None, *,
        target_commit: str, wrapper_commit: str, wrapper_parent: str,
        wrapper_changed_paths: Iterable[str], local_sha: str, remote_sha: str,
        branch: str, clean: bool, marker_present: bool, ledger_size: int,
        verdict_present: bool,
) -> dict:
    """Pure fail-closed authorization decision; it performs no I/O or marker claim."""
    verify_launch_control(control)
    if receipt is None or execution is None:
        raise PermissionError("independent review receipt and execution authorization wrapper are required")
    verify_wrapper_documents(control, receipt, execution, target_commit=target_commit)
    if wrapper_parent != target_commit:
        raise PermissionError("authorization wrapper immediate parent is not the reviewed target")
    if sorted(wrapper_changed_paths) != sorted((RECEIPT_RELATIVE, AUTHORIZATION_RELATIVE)):
        raise PermissionError("authorization wrapper must contain receipt and execution freeze only")
    if local_sha != wrapper_commit or remote_sha != wrapper_commit:
        raise PermissionError("local/live remote authorization wrapper exact sync is absent")
    if branch != BRANCH:
        raise PermissionError("protected launch branch mismatch")
    if clean is not True:
        raise PermissionError("protected launch worktree is not clean")
    if marker_present is not False or type(ledger_size) is not int or ledger_size != 0 or verdict_present is not False:
        raise PermissionError("protected marker/ledger/verdict state is not zero before authorization")
    return {"authorized": True, "freeze_sha": wrapper_commit,
            "run_identity": wrapper_commit, "target_commit": target_commit,
            "invocation_id": INVOCATION_ID}


def verify_wrapper_commit(repo: str | Path, target_commit: str,
                          wrapper_commit: str) -> tuple[dict, dict]:
    root = Path(repo).resolve(strict=True)
    if _run(root, "rev-parse", f"{wrapper_commit}^") != target_commit:
        raise PermissionError("authorization wrapper immediate parent is not target")
    changes = _run(root, "diff-tree", "--no-commit-id", "--name-only", "-r",
                   wrapper_commit).splitlines()
    if sorted(changes) != sorted((RECEIPT_RELATIVE, AUTHORIZATION_RELATIVE)):
        raise PermissionError("authorization wrapper is not receipt/execution-freeze only")
    receipt = strict_json_bytes(_git_bytes(root, wrapper_commit, RECEIPT_RELATIVE),
                                "independent review receipt")
    execution = strict_json_bytes(_git_bytes(root, wrapper_commit, AUTHORIZATION_RELATIVE),
                                  "execution freeze")
    control = strict_json_bytes(_git_bytes(root, target_commit, CONTROL_RELATIVE),
                                "launch control")
    verify_wrapper_documents(control, receipt, execution, target_commit=target_commit)
    expected_runtime = runtime_rows_at_commit(root, target_commit)
    if execution["runtime_files"] != expected_runtime:
        raise ValueError("execution freeze does not bind the exact target runtime closure")
    return receipt, execution


def verify_launch_target(repo: str | Path, target_commit: str) -> dict:
    root = Path(repo).resolve(strict=True)
    verify_historical_pair(root)
    if _run(root, "rev-parse", f"{target_commit}^") != HISTORICAL_WRAPPER_COMMIT:
        raise ValueError("launch adapter target parent is not the historical f8 wrapper")
    changes = _run(root, "diff-tree", "--no-commit-id", "--name-only", "-r",
                   target_commit).splitlines()
    if sorted(changes) != list(TARGET_DIFF_ALLOWLIST):
        raise ValueError(f"launch adapter target exact diff allowlist mismatch: {sorted(changes)}")
    if RECEIPT_RELATIVE in changes or AUTHORIZATION_RELATIVE in changes:
        raise ValueError("target must not contain approval or execution authorization")
    control = strict_json_bytes(_git_bytes(root, target_commit, CONTROL_RELATIVE), "launch control")
    verify_launch_control(control)
    verify_clean_bundle_at_target(root, target_commit, control)
    runtime = runtime_rows_at_commit(root, target_commit)
    provenance = strict_json_bytes(
        _git_bytes(root, target_commit, LAUNCH_PROVENANCE_RELATIVE), "launch provenance")
    verify_launch_provenance(provenance)
    return {"status": "AWAITING_INDEPENDENT_REVIEW", "target_commit": target_commit,
            "runtime_rows": len(runtime), "clean_bundle_rows": len(CLEAN_BUNDLE_RELATIVE_PATHS),
            "protected_access_authorized": False}


def verify_launch_provenance(provenance: dict) -> bool:
    expected = {
        "schema": LAUNCH_PROVENANCE_SCHEMA,
        "state": "PREACCESS_CLEAN_BUNDLE_MATERIALIZED",
        "target_commit_binding": "COMMIT_CONTAINING_THIS_DOCUMENT",
        "invocation_id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
        "control_path": CONTROL_RELATIVE,
        "predecessor_freeze_commit": PREDECESSOR_FREEZE_COMMIT,
        "archive_root": ARCHIVE_SOURCE,
        "clean_bundle_paths": list(CLEAN_BUNDLE_RELATIVE_PATHS),
        "protected_access_count": 0, "marker_present": False,
        "ledger_size_bytes": 0, "authorized": False,
    }
    _exact_dict(provenance, set(expected), "launch provenance")
    for key, value in expected.items():
        _exact_value(provenance[key], value, f"launch provenance {key}")
    return True


def verify_sync_snapshot(snapshot: dict, *, expected_sha: str) -> bool:
    required = {"branch", "local_sha", "remote_sha", "ahead", "behind", "clean"}
    _exact_dict(snapshot, required, "sync snapshot")
    if snapshot["branch"] != BRANCH:
        raise PermissionError("protected launch branch mismatch")
    if snapshot["local_sha"] != expected_sha or snapshot["remote_sha"] != expected_sha:
        raise PermissionError("local/live remote exact sync mismatch")
    if type(snapshot["ahead"]) is not int or type(snapshot["behind"]) is not int or \
            snapshot["ahead"] != 0 or snapshot["behind"] != 0:
        raise PermissionError("local/live remote ahead/behind sync mismatch")
    if snapshot["clean"] is not True:
        raise PermissionError("protected launch worktree is not clean")
    return True


def live_sync_snapshot(repo: str | Path, expected_sha: str) -> dict:
    root = Path(repo).resolve(strict=True)
    lines = _run(root, "ls-remote", REMOTE, f"refs/heads/{BRANCH}").splitlines()
    if len(lines) != 1 or len(lines[0].split()) != 2:
        raise PermissionError("live remote launch branch is absent or ambiguous")
    remote_sha = lines[0].split()[0]
    local_sha = _run(root, "rev-parse", "HEAD")
    branch = _run(root, "branch", "--show-current")
    counts = _run(root, "rev-list", "--left-right", "--count",
                  f"{local_sha}...{remote_sha}").split()
    if len(counts) != 2:
        raise PermissionError("live remote ahead/behind count is malformed")
    status = _run(root, "status", "--porcelain=v1", "--untracked-files=all")
    snapshot = {"branch": branch, "local_sha": local_sha, "remote_sha": remote_sha,
                "ahead": int(counts[0]), "behind": int(counts[1]), "clean": not status}
    verify_sync_snapshot(snapshot, expected_sha=expected_sha)
    return snapshot


def verify_zero_protected_state(repo: str | Path, *, markers: Iterable[str] | None = None,
                                ledgers: Iterable[str] | None = None,
                                verdicts: Iterable[str] | None = None) -> bool:
    root = Path(repo).resolve(strict=True)
    marker_paths = list(markers or (
        "artifacts/.gcspo_stage0_static_rerun.protected_run_started.json",
        ("artifacts/gcspo_stage0_static_rerun_successor/."
         "gcspo-stage0-successor-7be48c4411644ff3a9ec41c7701dfa01.protected_run_started.json"),
        MARKER_RELATIVE,
    ))
    ledger_paths = list(ledgers or (
        "artifacts/gcspo_stage0_static_rerun/access_ledger.jsonl",
        f"{HISTORICAL_ARTIFACT_RELATIVE}/access_ledger.jsonl",
        LEDGER_RELATIVE,
    ))
    verdict_paths = list(verdicts or (
        "artifacts/gcspo_stage0_static_rerun/final_verdict.json",
        f"{HISTORICAL_ARTIFACT_RELATIVE}/final_verdict.json",
        f"{ARTIFACT_RELATIVE}/final_verdict.json",
    ))
    for relative in marker_paths:
        if (root / relative).exists():
            raise PermissionError(f"protected marker is already present: {relative}")
    for relative in ledger_paths:
        path = root / relative
        _no_symlink_components(root, path, f"protected ledger {relative}")
        if not path.is_file() or path.is_symlink() or path.stat().st_size != 0:
            raise PermissionError(f"protected ledger is absent, aliased, or nonzero: {relative}")
    for relative in verdict_paths:
        if (root / relative).exists():
            raise PermissionError(f"protected final verdict is already present: {relative}")
    return True


def verify_preclaim(repo: str | Path, *, check_remote: bool = True) -> dict:
    """Finish every adapter authorization check without claiming a marker."""
    root = Path(repo).resolve(strict=True)
    wrapper = _run(root, "rev-parse", "HEAD")
    try:
        target = _run(root, "rev-parse", f"{wrapper}^")
        target_report = verify_launch_target(root, target)
        receipt, execution = verify_wrapper_commit(root, target, wrapper)
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise PermissionError("independent authorization wrapper is absent or invalid") from exc
    control = strict_json_bytes(_git_bytes(root, target, CONTROL_RELATIVE), "launch control")
    verify_preaccess_worktree_bundle(root, control)
    verify_zero_protected_state(root)
    snapshot = live_sync_snapshot(root, wrapper) if check_remote else {
        "branch": BRANCH, "local_sha": wrapper, "remote_sha": wrapper,
        "ahead": 0, "behind": 0, "clean": True,
    }
    authorization = authorize_launch(
        control, receipt, execution, target_commit=target, wrapper_commit=wrapper,
        wrapper_parent=target, wrapper_changed_paths=[RECEIPT_RELATIVE, AUTHORIZATION_RELATIVE],
        local_sha=snapshot["local_sha"], remote_sha=snapshot["remote_sha"],
        branch=snapshot["branch"], clean=snapshot["clean"], marker_present=False,
        ledger_size=0, verdict_present=False,
    )
    return {**authorization, "target_report": target_report, "sync": snapshot,
            "control": control, "execution": execution}


def verify_final_authorization_binding(repo: str | Path, artifact_root: str | Path,
                                       records: list[dict]) -> bool:
    """Bind successor final artifacts and every access record to the reviewed wrapper."""
    root = Path(repo).resolve(strict=True)
    artifact = Path(artifact_root).resolve(strict=True)
    launch = strict_json_bytes((artifact / "launch_provenance.json").read_bytes(),
                               "launch provenance")
    verify_launch_provenance(launch)
    protected = strict_json_bytes((artifact / "protected_run_provenance.json").read_bytes(),
                                  "protected run provenance")
    expected_keys = {"schema", "authorization_wrapper_commit", "target_commit",
                     "invocation_id", "nonce", "receipt_path", "execution_freeze_path"}
    _exact_dict(protected, expected_keys, "protected run provenance")
    _exact_value(protected["schema"], PROTECTED_PROVENANCE_SCHEMA,
                 "protected run provenance schema")
    _exact_value(protected["invocation_id"], INVOCATION_ID,
                 "protected run provenance invocation")
    _exact_value(protected["nonce"], INVOCATION_NONCE,
                 "protected run provenance nonce")
    _exact_value(protected["receipt_path"], RECEIPT_RELATIVE,
                 "protected run provenance receipt path")
    _exact_value(protected["execution_freeze_path"], AUTHORIZATION_RELATIVE,
                 "protected run provenance execution path")
    target = _sha(protected["target_commit"], "protected run target", 40)
    wrapper = _sha(protected["authorization_wrapper_commit"], "protected run wrapper", 40)
    verify_launch_target(root, target)
    verify_wrapper_commit(root, target, wrapper)
    if not records or any(row.get("authorization_sha") != wrapper or
                          row.get("run_identity") != wrapper for row in records):
        raise ValueError("final access evidence is not bound to authorization wrapper")
    return True
