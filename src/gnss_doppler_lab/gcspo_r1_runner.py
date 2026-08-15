"""Fail-closed R1 protected runner for the preregistered frozen completion."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile

from .gcspo_artifacts import canonical_write_json
from .gcspo_capabilities import validate_preaccess_capabilities
from .gcspo_core import AccessGate
from .gcspo_evaluate import validate_clean_contrast_preaccess
from .gcspo_freeze import validate_protected_manifest_inventory
from .gcspo_successor_freeze import implementation_paths_at_commit
from .gcspo_r1_support import (
    exact_b0_full_contrast_r1,
    integrate_protected_b0_r1,
    r1_support_adapter_scope,
    validate_protected_method_support_r1,
)

INVOCATION_ID = "gcspo-stage0-r1-frozen-completion-0acca83b5245429b"
INVOCATION_NONCE = "97109aa102565cb9a5b7c8864ea766c4780b525c1be190dc56129b69f92a1535"
ARTIFACT_RELATIVE = "artifacts/gcspo_stage0_r1_frozen_completion"
CONFIG_RELATIVE = f"config/gcspo_r1_completion/{INVOCATION_ID}"
FREEZE_RELATIVE = f"{CONFIG_RELATIVE}/execution_freeze.json"
RECEIPT_RELATIVE = f"{CONFIG_RELATIVE}/independent_review_receipt.json"
MARKER_RELATIVE = f"{ARTIFACT_RELATIVE}/.protected_run_started.json"
BRANCH = "research/gcspo-stage0-r1-frozen-completion-preclaim-repair"

BASE_COMMIT = "57271e526e9e346c8d4d7626b006c5a88166f1be"
SCIENCE_MANIFEST_RELATIVE = f"{ARTIFACT_RELATIVE}/frozen_science_hashes.json"
PREREG_RELATIVE = f"{ARTIFACT_RELATIVE}/completion_preregistration.json"
AMENDMENT_RELATIVE = f"{ARTIFACT_RELATIVE}/completion_preregistration_amendment.json"
AUDIT_RECEIPT_RELATIVE = f"{ARTIFACT_RELATIVE}/physical_controls_audit_receipt.json"
FINAL_MANIFEST_NAME = "artifact_manifest_sha256.json"
FROZEN_SCIENCE_PATHS = (
    "artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt",
    "artifacts/ai_morph_gru_cleanStatic_q70_frame/validation_prn_node_scores.csv",
    "scripts/eval_btail_support_gate.py",
    "scripts/score_texbat_prn_node_gru.py",
    "src/gnss_doppler_lab/gcspo_a5.py",
    "src/gnss_doppler_lab/gcspo_ablations.py",
    "src/gnss_doppler_lab/gcspo_b0.py",
    "src/gnss_doppler_lab/gcspo_clean.py",
    "src/gnss_doppler_lab/gcspo_core.py",
    "src/gnss_doppler_lab/gcspo_full.py",
    "src/gnss_doppler_lab/gcspo_protected.py",
    "src/gnss_doppler_lab/gcspo_statistics.py",
)
PIPELINE_BASELINE_PATHS = (
    "scripts/run_gcspo_stage0_successor.py",
    "src/gnss_doppler_lab/gcspo_evaluate.py",
    "src/gnss_doppler_lab/gcspo_successor_launch.py",
)
DYNAMIC_RUNTIME_PATHS = (
    "artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt",
    "artifacts/ai_morph_gru_cleanStatic_q70_frame/validation_prn_node_scores.csv",
    "scripts/eval_btail_support_gate.py",
    "scripts/score_texbat_prn_node_gru.py",
    "scripts/train_prn_node_gru.py",
    PREREG_RELATIVE,
    AMENDMENT_RELATIVE,
    SCIENCE_MANIFEST_RELATIVE,
    AUDIT_RECEIPT_RELATIVE,
)
CLEAN_FILE_PATHS = (
    f"{ARTIFACT_RELATIVE}/clean_a5_report.json",
    f"{ARTIFACT_RELATIVE}/clean_ablation_report.json",
    f"{ARTIFACT_RELATIVE}/clean_only_report.json",
    f"{ARTIFACT_RELATIVE}/clean_reproduction_evidence.json",
)
_BASE_JOIN_KEYS = (
    "scenario", "phase", "window_start_s", "availability_s", "epoch_ids",
    "epoch_prn_support",
)
JOIN_KEYS = (
    "scenario", "phase", "window_start_s", "availability_s", "prns",
    "epoch_ids", "epoch_prn_support",
)
_FREEZE_KEYS = {
    "schema", "state", "target_commit", "wrapper_commit_binding",
    "invocation_id", "nonce", "runtime_files", "clean_files",
}
_RECEIPT_KEYS = {
    "schema", "state", "passed", "target_commit", "invocation_id", "nonce",
    "findings",
}
_PREREG_KEYS = {
    "allowed_engineering_repairs", "base_commit", "created_at_utc",
    "decision_contract", "forbidden", "invocation", "namespace",
    "predecessor_failure", "scenarios", "schema", "scientific_freeze",
    "status", "support_contract", "time_budget_hours",
}
_SUPPORT_KEYS = {
    "b0_common_support_is_exact_intersection",
    "diagnostic_nearest_time_delta_may_be_reported_but_not_used_for_join",
    "full_standalone_support_preserved", "join_keys",
    "timestamp_tolerance_s", "warmup_context_not_scored",
}
_SCIENCE_KEYS = {
    "audited_artifacts", "base_commit", "frozen_scientific_code",
    "original_preregistration", "pipeline_baseline_for_scoped_change_audit",
    "policy", "r1_preregistration", "recorded_at_utc", "schema",
    "scientific_configuration",
}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_UTC_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?\+00:00"
)
_AUDIT_PATH = f"{ARTIFACT_RELATIVE}/physical_controls_audit.json"
_AUDIT_SHA256 = "12250c1be04128e96eaeb97301768535730bda097d8670ae388fc8cc898412e9"
_AUDIT_SIZE = 19744
_CONTROL_PATH = f"{ARTIFACT_RELATIVE}/physical_controls.json"
_CONTROL_SHA256 = "e1e14890e06151ccac41df29c6d77b8d1b41ee1394864e8e5fc0ade6dc504681"
_BASE_PREREG_SHA256 = "fb810c3e1548f0375c853805e1451666690b5dff5775698d000f324e0dfb63e7"
_BASE_PREREG_CANONICAL_SHA256 = "529da473751edd43f8bf7899b16a8337a7f0d5798b2ec525c6d24c171ddbe07e"
_AMENDMENT_SHA256 = "34bfb0e81b3d3b4c99bc234bb6d0f0afbeaf8578b1d4fc3096ba8e050b02b27e"
_AMENDMENT_CANONICAL_SHA256 = "6b0629eff48da5a7028ea8c302f364008e887d446266742233c9ed255614399c"
_AUDIT_RECEIPT_SHA256 = "13db129bce8b6ec0e13f9f82607b595810a078def06b3a0eb43441af4cf91725"
_AUDIT_RECEIPT_SIZE = 765
_AUDIT_RECEIPT_CANONICAL_SHA256 = "8c268d5c556b43dda01b161afcf003a089b735bfe9377cfbea956850583288f8"
_FINAL_SCENARIOS = ("DS3", "DS7")
_B0_GENERATED_FILES = tuple(sorted(
    f"b0_protected_recomputed/{scenario}/{name}"
    for scenario in _FINAL_SCENARIOS
    for name in (
        "scheduled_node_windows.csv",
        f"gcspo_b0_{scenario}_prn_local_scores.csv",
        f"gcspo_b0_{scenario}_prn_local_event_scores.csv",
        f"gcspo_b0_{scenario}_prn_local_onset_summary.json",
        f"gcspo_b0_{scenario}_prn_local_score_vs_time.png",
    )
))
_ARTIFACT_INPUT_FILES = (
    "README.md", "clean_a5_report.json", "clean_ablation_report.json",
    "clean_only_report.json", "clean_reproduction_check.json",
    "clean_reproduction_evidence.json", "completion_preregistration.json",
    "completion_preregistration_amendment.json", "config.json", "data_inventory.json",
    "frozen_science_hashes.json", "normal_model_summary.json",
    "original_preregistration.json", "physical_controls.json",
    "physical_controls_audit.json", "physical_controls_audit_receipt.json",
    "protected_capabilities.json", "source_commit.json", "thresholds.json",
)
_FINAL_GENERATED_FILES = (
    ".protected_run_started.json", "ablation_metrics.csv", "access_ledger.jsonl",
    "bootstrap_intervals.csv", "external_static_fpr.csv", "final_verdict.json",
    "implementation_manifest.json", "per_epoch_scores.csv",
    "plots/full_score_numeric_sidecar.csv",
    "protected_control_status.json", "protected_run_provenance.json",
    "relation_destruction_metrics.json", "scenario_metrics.csv",
    "shared_state_estimates.csv", *_B0_GENERATED_FILES,
)
_POST_ATTESTATION_FILES = (
    "fresh_clone_verifier_report.json",
    "verifier_report.json",
)
_FINAL_EXPECTED_FILES = tuple(sorted({*_ARTIFACT_INPUT_FILES, *_FINAL_GENERATED_FILES,
                                      FINAL_MANIFEST_NAME}))
_FINAL_EXPECTED_DIRECTORIES = tuple(sorted({
    "plots", "b0_protected_recomputed",
    *(f"b0_protected_recomputed/{scenario}" for scenario in _FINAL_SCENARIOS),
}))

def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["/usr/bin/git", *args], cwd=root, text=True,
                            capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_bytes(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root,
                            capture_output=True)
    if result.returncode:
        raise ValueError(f"required Git object absent: {commit}:{relative}")
    return result.stdout


def _strict_object(payload: bytes, label: str) -> dict:
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f"nonfinite JSON in {label}: {token}")))
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return _sha(payload)


def _valid_utc_timestamp(value: object) -> bool:
    if type(value) is not str or _UTC_TIMESTAMP.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc


def _lstat(path: Path):
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _require_regular(path: Path, label: str):
    info = _lstat(path)
    if (info is None or stat.S_ISLNK(info.st_mode) or
            not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise ValueError(
            f"{label} must be an existing single-link regular non-symlink file: {path}")
    return info


def _stable_file_identity(info) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_nlink,
            info.st_mtime_ns, info.st_ctime_ns)


def _read_regular_bytes(path: Path, label: str) -> bytes:
    """Read one stable single-link regular-file snapshot without following symlinks."""
    path = Path(path)
    before = _require_regular(path, label)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or
                _stable_file_identity(opened) != _stable_file_identity(before)):
            raise ValueError(f"{label} changed before descriptor bind: {path}")
        chunks = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = _require_regular(path, label)
    identity = _stable_file_identity(before)
    if (_stable_file_identity(after_descriptor) != identity or
            _stable_file_identity(after_path) != identity or
            len(payload) != before.st_size):
        raise ValueError(f"{label} changed while read: {path}")
    return payload


def install_and_verify_adapter():
    """Return the scoped R1 adapter contract; callers must use it with ``with``."""
    return r1_support_adapter_scope()


def _verify_adapter_bindings() -> None:
    from . import gcspo_evaluate, gcspo_verify_artifacts

    bindings = (
        (gcspo_evaluate.integrate_protected_b0, integrate_protected_b0_r1),
        (gcspo_evaluate.validate_protected_method_support,
         validate_protected_method_support_r1),
        (gcspo_evaluate.exact_b0_full_contrast, exact_b0_full_contrast_r1),
        (gcspo_verify_artifacts.exact_b0_full_contrast, exact_b0_full_contrast_r1),
    )
    if any(binding is not expected for binding, expected in bindings):
        raise ValueError("R1 evaluator/verifier adapter identity mismatch")


def verify_zero_access_state(artifact_dir: str | Path, marker: str | Path) -> dict:
    artifact = Path(artifact_dir)
    marker_path = Path(marker)
    root_info = _lstat(artifact)
    if (root_info is None or stat.S_ISLNK(root_info.st_mode) or
            not stat.S_ISDIR(root_info.st_mode)):
        raise ValueError("R1 artifact root must be a real directory")
    if marker_path.parent != artifact:
        raise ValueError("R1 marker must be an immediate artifact-root child")
    if _lstat(marker_path) is not None:
        raise ValueError("protected marker already exists or is nonregular")
    ledger = artifact / "access_ledger.jsonl"
    ledger_info = _lstat(ledger)
    if ledger_info is not None:
        if (stat.S_ISLNK(ledger_info.st_mode) or not stat.S_ISREG(ledger_info.st_mode) or
                ledger_info.st_nlink != 1):
            raise ValueError("protected access ledger is not a single-link regular file")
        if ledger_info.st_size:
            raise ValueError("protected access ledger is nonempty")
    if _lstat(artifact / "final_verdict.json") is not None:
        raise ValueError("protected final verdict already exists or is nonregular")
    return {"dev": root_info.st_dev, "ino": root_info.st_ino}


def claim_once(marker: str | Path, *, wrapper_commit: str, target_commit: str,
               parent_binding: dict, keep_parent_fd: bool = False):
    document = {
        "schema": "gnss-doppler-lab.gcspo-stage0.r1-protected-run-start.v1",
        "protected_run_count": 1,
        "invocation_id": INVOCATION_ID,
        "nonce": INVOCATION_NONCE,
        "authorization_wrapper_commit": wrapper_commit,
        "target_commit": target_commit,
    }
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    path = Path(marker)
    if (type(parent_binding) is not dict or set(parent_binding) != {"dev", "ino"} or
            any(isinstance(parent_binding[key], bool) or
                type(parent_binding[key]) is not int or parent_binding[key] < 0
                for key in ("dev", "ino"))):
        raise ValueError("R1 claim parent binding is malformed")
    directory = os.open(
        path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
        getattr(os, "O_NOFOLLOW", 0))
    try:
        opened_parent = os.fstat(directory)
        if (not stat.S_ISDIR(opened_parent.st_mode) or
                (opened_parent.st_dev, opened_parent.st_ino) !=
                (parent_binding["dev"], parent_binding["ino"])):
            raise ValueError("R1 artifact directory changed before exact-once claim")
        descriptor = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o600, dir_fd=directory)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory)
    except Exception:
        os.close(directory)
        raise
    if keep_parent_fd:
        return document, directory
    os.close(directory)
    return document


def claim_repository_anchor(root: str | Path, *, wrapper_commit: str, target_commit: str) -> Path:
    root = Path(root).resolve(strict=True)
    git_common = Path(_git(root, "rev-parse", "--git-common-dir"))
    if not git_common.is_absolute():
        git_common = root / git_common
    git_common = git_common.resolve(strict=True)
    common_info = os.lstat(git_common)
    if not stat.S_ISDIR(common_info.st_mode):
        raise ValueError("Git common directory must be a real directory")
    directory = os.open(git_common, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                        getattr(os, "O_NOFOLLOW", 0))
    opened_info = os.fstat(directory)
    if (opened_info.st_dev, opened_info.st_ino) != (common_info.st_dev, common_info.st_ino):
        os.close(directory)
        raise ValueError("Git common directory changed before repository claim")
    anchor = git_common / f"gcspo-r1-{INVOCATION_ID}.claimed.json"
    payload = (json.dumps({
        "schema": "gnss-doppler-lab.gcspo-stage0.r1-repository-claim.v1",
        "invocation_id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
        "authorization_wrapper_commit": wrapper_commit,
        "target_commit": target_commit, "protected_run_count": 1,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    try:
        descriptor = os.open(anchor.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                             getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory)
    finally:
        os.close(directory)
    return anchor


def _canonical_target_path(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value or any(
            ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{label} path is malformed")
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value or
            any(part in {"", ".", ".."} for part in path.parts)):
        raise ValueError(f"{label} path is noncanonical")
    return value


def _git_regular_blob(root: Path, target: str, relative: str, label: str) -> bytes:
    line = _git(root, "ls-tree", target, "--", relative)
    if not line:
        raise ValueError(f"{label} target-tree blob is absent: {relative}")
    metadata, separator, observed_path = line.partition("\t")
    fields = metadata.split()
    if (separator != "\t" or observed_path != relative or len(fields) != 3 or
            fields[0] not in {"100644", "100755"} or fields[1] != "blob"):
        raise ValueError(f"{label} target-tree member is not a regular file: {relative}")
    return _git_bytes(root, target, relative)


def expected_runtime_paths(root: str | Path, target: str) -> tuple[str, ...]:
    root = Path(root).resolve(strict=True)
    paths = set(implementation_paths_at_commit(root, target)) | set(DYNAMIC_RUNTIME_PATHS)
    result = tuple(sorted(paths))
    if not result or len(result) != len(set(result)):
        raise ValueError("runtime expected inventory is invalid")
    for relative in result:
        _canonical_target_path(relative, "runtime expected inventory")
    return result


def _verify_identity_rows(
        root: Path, target: str, rows: object, label: str, *,
        expected_paths: tuple[str, ...],
) -> list[dict]:
    if type(rows) is not list:
        raise ValueError(f"{label} must be a list")
    if tuple(expected_paths) != tuple(sorted(set(expected_paths))):
        raise ValueError(f"{label} expected inventory is invalid")
    paths = []
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"{label} row schema mismatch")
        relative = _canonical_target_path(row["path"], f"{label} row {index}")
        if (type(row["sha256"]) is not str or
                _HEX64.fullmatch(row["sha256"]) is None or
                isinstance(row["size_bytes"], bool) or
                type(row["size_bytes"]) is not int or row["size_bytes"] < 0):
            raise ValueError(f"{label} row identity type mismatch")
        paths.append(relative)
    if paths != list(expected_paths):
        raise ValueError(f"{label} exact expected-set mismatch")
    for row in rows:
        payload = _git_regular_blob(root, target, row["path"], label)
        if _sha(payload) != row["sha256"] or len(payload) != row["size_bytes"]:
            raise ValueError(f"{label} identity mismatch: {row['path']}")
    return rows


def _verify_audit_git_transition(root: Path, target: str) -> None:
    """Prove the audit/receipt were introduced only by the reviewed target commit."""
    ancestry = _git(root, "rev-list", "--parents", "-n", "1", target).split()
    if len(ancestry) != 2 or ancestry[0] != target:
        raise ValueError("R1 repair target must have exactly one parent")
    parent = ancestry[1]
    for relative in (_AUDIT_PATH, AUDIT_RECEIPT_RELATIVE):
        if _git(root, "ls-tree", parent, "--", relative):
            raise ValueError(f"R1 audit provenance path unexpectedly exists in parent: {relative}")
    parent_control = _git_regular_blob(
        root, parent, _CONTROL_PATH, "physical controls parent input")
    if _sha(parent_control) != _CONTROL_SHA256:
        raise ValueError("physical controls parent input identity mismatch")


def _verify_frozen_science_manifest(root: Path, target: str) -> dict:
    science = _strict_object(
        _git_regular_blob(root, target, SCIENCE_MANIFEST_RELATIVE, "frozen science manifest"),
        "frozen science hashes",
    )
    if set(science) != _SCIENCE_KEYS:
        raise ValueError("frozen science manifest exact schema mismatch")
    if (science["schema"] !=
            "gnss-doppler-lab.gcspo-stage0-r1-frozen-completion.science-hashes.v1" or
            science["base_commit"] != BASE_COMMIT or
            science["recorded_at_utc"] != "2026-08-14T16:22:31.256859+00:00"):
        raise ValueError("frozen science manifest identity mismatch")

    frozen = science["frozen_scientific_code"]
    if type(frozen) is not dict or tuple(sorted(frozen)) != FROZEN_SCIENCE_PATHS:
        raise ValueError("frozen scientific code exact path inventory mismatch")
    for relative in FROZEN_SCIENCE_PATHS:
        expected = frozen[relative]
        if type(expected) is not str or _HEX64.fullmatch(expected) is None:
            raise ValueError(f"frozen science hash is malformed: {relative}")
        if (_sha(_git_regular_blob(root, BASE_COMMIT, relative, "frozen science baseline")) != expected or
                _sha(_git_regular_blob(root, target, relative, "frozen science target")) != expected):
            raise ValueError(f"frozen science hash changed: {relative}")

    pipeline = science["pipeline_baseline_for_scoped_change_audit"]
    if type(pipeline) is not dict or tuple(sorted(pipeline)) != PIPELINE_BASELINE_PATHS:
        raise ValueError("pipeline baseline exact path inventory mismatch")
    for relative in PIPELINE_BASELINE_PATHS:
        expected = pipeline[relative]
        baseline_payload = _git_regular_blob(
            root, BASE_COMMIT, relative, "pipeline baseline")
        target_payload = _git_regular_blob(root, target, relative, "pipeline target")
        if (type(expected) is not str or _HEX64.fullmatch(expected) is None or
                _sha(baseline_payload) != expected or
                (relative != "src/gnss_doppler_lab/gcspo_evaluate.py" and
                 _sha(target_payload) != expected)):
            raise ValueError(f"pipeline baseline identity mismatch: {relative}")

    original = science["original_preregistration"]
    expected_original = {
        "path": ("artifacts/gcspo_stage0_successor_launch/"
                 "gcspo-stage0-successor-launch-"
                 "77e586dfdb50a008ed2f0b31052e33bb700e191641e7ffbb4845860df15cf48e/"
                 "preregistration.json"),
        "sha256": "715e11965854f785487e9d2c747718747c1d31cdd8603696ea7af126a45a70da",
    }
    config = science["scientific_configuration"]
    expected_config = {
        "path": ("artifacts/gcspo_stage0_successor_launch/"
                 "gcspo-stage0-successor-launch-"
                 "77e586dfdb50a008ed2f0b31052e33bb700e191641e7ffbb4845860df15cf48e/"
                 "config.json"),
        "sha256": "0db816116b95b41db8b7af7379cd7411cc52d43b6428ae00ab02d6ccac19f4ad",
    }
    if original != expected_original or config != expected_config:
        raise ValueError("frozen source provenance identity mismatch")
    if (_sha(_git_regular_blob(
            root, target, f"{ARTIFACT_RELATIVE}/original_preregistration.json",
            "copied original preregistration")) != original["sha256"] or
            _sha(_git_regular_blob(
                root, target, f"{ARTIFACT_RELATIVE}/config.json",
                "copied scientific configuration")) != config["sha256"]):
        raise ValueError("frozen source copy target-tree identity mismatch")

    expected_r1_preregistration = {
        "amendment": {"path": AMENDMENT_RELATIVE, "sha256": _AMENDMENT_SHA256},
        "base": {"path": PREREG_RELATIVE, "sha256": _BASE_PREREG_SHA256},
        "effective_status": "BASE_PREREGISTRATION_PLUS_POST_IMPLEMENTATION_AMENDMENT",
    }
    if science["r1_preregistration"] != expected_r1_preregistration:
        raise ValueError("R1 preregistration closure mismatch")
    for label, identity in (("base", expected_r1_preregistration["base"]),
                            ("amendment", expected_r1_preregistration["amendment"])):
        _canonical_target_path(identity["path"], f"R1 preregistration {label}")
        if _sha(_git_regular_blob(root, target, identity["path"],
                                  f"R1 preregistration {label}")) != identity["sha256"]:
            raise ValueError(f"R1 preregistration {label} target-tree identity mismatch")

    expected_policy = {
        "pipeline_changes_allowed_only_for_exact_support_and_control_aggregation": True,
        "scientific_code_must_remain_byte_identical": True,
        "thresholds_features_models_lambdas_ridges_timelines_labels_unchanged": True,
    }
    if science["policy"] != expected_policy:
        raise ValueError("frozen science policy mismatch")
    expected_audit = {
        "physical_controls_audit": {
            "input_path": _CONTROL_PATH,
            "input_sha256": _CONTROL_SHA256,
            "path": _AUDIT_PATH,
            "provenance": (
                "EXACT_WORKTREE_BYTES_INTENTIONALLY_COMMITTED_DURING_R1_REPAIR_"
                "NOT_RECOMPUTED"
            ),
            "sha256": _AUDIT_SHA256,
            "size_bytes": _AUDIT_SIZE,
        },
        "physical_controls_audit_receipt": {
            "path": AUDIT_RECEIPT_RELATIVE,
            "sha256": _AUDIT_RECEIPT_SHA256,
            "size_bytes": _AUDIT_RECEIPT_SIZE,
        },
    }
    if _canonical_json_sha(science["audited_artifacts"]) != _canonical_json_sha(expected_audit):
        raise ValueError("audited artifact manifest identity/provenance mismatch")
    _verify_audit_git_transition(root, target)
    audit_payload = _git_regular_blob(root, target, _AUDIT_PATH, "physical controls audit")
    control_payload = _git_regular_blob(root, target, _CONTROL_PATH, "physical controls input")
    receipt_payload = _git_regular_blob(
        root, target, AUDIT_RECEIPT_RELATIVE, "physical controls audit receipt")
    if (len(audit_payload) != _AUDIT_SIZE or _sha(audit_payload) != _AUDIT_SHA256 or
            _sha(control_payload) != _CONTROL_SHA256 or
            len(receipt_payload) != _AUDIT_RECEIPT_SIZE or
            _sha(receipt_payload) != _AUDIT_RECEIPT_SHA256):
        raise ValueError("physical controls audit target-tree identity mismatch")
    audit = _strict_object(audit_payload, "physical controls audit")
    if (audit.get("schema") !=
            "gnss-doppler-lab.gcspo-stage0-r1.physical-controls-audit.v1" or
            audit.get("input_sha256") != _CONTROL_SHA256 or
            audit.get("threshold_or_score_changed") is not False or
            audit.get("code_change_required") is not False):
        raise ValueError("physical controls audit provenance mismatch")
    receipt_document = _strict_object(receipt_payload, "physical controls audit receipt")
    if _canonical_json_sha(receipt_document) != _AUDIT_RECEIPT_CANONICAL_SHA256:
        raise ValueError("physical controls audit receipt exact schema/value mismatch")
    if not _valid_utc_timestamp(receipt_document.get("recorded_at_utc")):
        raise ValueError("physical controls audit receipt timestamp is malformed")
    return science


def _verify_directory_root(root: Path, label: str) -> None:
    info = _lstat(root)
    if info is not None and stat.S_ISDIR(info.st_mode):
        return
    match = re.fullmatch(r"/proc/self/fd/(\d+)", str(root))
    if info is None or not stat.S_ISLNK(info.st_mode) or match is None:
        raise ValueError(f"{label} root must be a real directory or held directory descriptor")
    descriptor_info = os.fstat(int(match.group(1)))
    followed = os.stat(root)
    if (not stat.S_ISDIR(descriptor_info.st_mode) or
            (descriptor_info.st_dev, descriptor_info.st_ino) != (followed.st_dev, followed.st_ino)):
        raise ValueError(f"{label} held directory descriptor binding mismatch")


def _scan_tree(root: Path, label: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return regular files/directories without following any directory entry."""
    root = Path(root)
    _verify_directory_root(root, label)
    files, directories = [], []

    def visit(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            try:
                entry_info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"{label} unreadable entry: {relative}") from exc
            mode = entry_info.st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"{label} rejects symlink or broken symlink: {relative}")
            if stat.S_ISDIR(mode):
                directories.append(relative)
                visit(Path(entry.path), relative)
            elif stat.S_ISREG(mode):
                files.append(relative)
            else:
                raise ValueError(f"{label} rejects nonregular entry: {relative}")

    visit(root, "")
    return tuple(files), tuple(directories)


def _verify_preaccess_input_tree(root: Path, runtime_paths: tuple[str, ...]) -> None:
    files, directories = _scan_tree(root / ARTIFACT_RELATIVE, "R1 preaccess artifact")
    if files != tuple(sorted(_ARTIFACT_INPUT_FILES)) or directories:
        raise ValueError("R1 preaccess artifact exact inventory mismatch")
    for relative in runtime_paths:
        _require_regular(root / relative, "R1 runtime input")


def _verify_no_runtime_shadows(root: Path, runtime_paths: tuple[str, ...]) -> None:
    package_files, _package_directories = _scan_tree(
        root / "src/gnss_doppler_lab", "R1 runtime Python package")
    script_files, _script_directories = _scan_tree(
        root / "scripts", "R1 runtime scripts")
    pathspecs = sorted({ARTIFACT_RELATIVE, *runtime_paths})
    status = _git(root, "status", "--porcelain=v1", "--ignored",
                  "--untracked-files=all", "--", *pathspecs)
    shadows = []
    for line in status.splitlines():
        if line[:2] in {"??", "!!"}:
            shadows.append(line[3:])
    if shadows:
        raise ValueError(f"ignored/untracked R1 runtime shadow: {sorted(shadows)}")

    runtime_stems = {PurePosixPath(path).stem for path in runtime_paths
                     if PurePosixPath(path).suffix == ".py"}
    runtime_path_set = set(runtime_paths)
    scanned = (
        *((f"src/gnss_doppler_lab/{relative}", relative) for relative in package_files),
        *((f"scripts/{relative}", relative) for relative in script_files),
    )
    for repository_path, relative in scanned:
        name = PurePosixPath(relative).name
        module_stem = name.split(".cpython-", 1)[0].split(".abi3", 1)[0]
        for suffix in (".pyc", ".pyo", ".py", ".so"):
            if module_stem.endswith(suffix):
                module_stem = module_stem[:-len(suffix)]
                break
        if module_stem in runtime_stems and repository_path not in runtime_path_set:
            raise ValueError(f"ignored/untracked R1 import shadow: {repository_path}")

    cache_status = _git(root, "status", "--porcelain=v1", "--ignored",
                        "--untracked-files=all", "--", "src", "scripts")
    importable_suffixes = (".py", ".pyc", ".pyo", ".so")
    for line in cache_status.splitlines():
        if line[:2] not in {"??", "!!"}:
            continue
        relative = line[3:]
        name = PurePosixPath(relative).name
        if (name.endswith(importable_suffixes) or ".cpython-" in name or
                "__pycache__" in PurePosixPath(relative).parts):
            raise ValueError(f"ignored/untracked R1 import shadow: {relative}")


def _verify_authorization_documents(freeze: dict, receipt: dict, target: str) -> None:
    if set(freeze) != _FREEZE_KEYS:
        raise ValueError("R1 execution freeze exact schema mismatch")
    if set(receipt) != _RECEIPT_KEYS:
        raise ValueError("R1 review receipt exact schema mismatch")
    if (freeze["schema"] != "gnss-doppler-lab.gcspo-stage0.r1-execution-freeze.v1" or
            freeze["invocation_id"] != INVOCATION_ID or freeze["nonce"] != INVOCATION_NONCE or
            freeze["state"] != "VALID_FOR_PROTECTED_ACCESS" or
            freeze["target_commit"] != target or
            freeze["wrapper_commit_binding"] != "COMMIT_CONTAINING_THIS_DOCUMENT" or
            type(freeze["runtime_files"]) is not list or
            type(freeze["clean_files"]) is not list):
        raise ValueError("R1 execution freeze exact type/value mismatch")
    if (receipt["schema"] != "gnss-doppler-lab.gcspo-stage0.r1-independent-review.v1" or
            receipt["state"] != "APPROVED" or receipt["passed"] is not True or
            receipt["target_commit"] != target or receipt["invocation_id"] != INVOCATION_ID or
            receipt["nonce"] != INVOCATION_NONCE or type(receipt["findings"]) is not list or
            receipt["findings"] != []):
        raise ValueError("R1 independent review receipt exact type/value mismatch")


def verify_execution_freeze(root: str | Path, *, check_remote: bool = True) -> dict:
    root = Path(root).resolve(strict=True)
    if _git(root, "status", "--porcelain"):
        raise ValueError("target worktree is not clean")
    wrapper = _git(root, "rev-parse", "HEAD")
    wrapper_parents = _git(root, "rev-list", "--parents", "-n", "1", wrapper).split()
    if len(wrapper_parents) != 2 or wrapper_parents[0] != wrapper:
        raise ValueError("authorization wrapper must have exactly one parent")
    target = wrapper_parents[1]
    wrapper_diff = _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", wrapper)
    expected_wrapper_paths = {FREEZE_RELATIVE, RECEIPT_RELATIVE}
    if set(wrapper_diff.splitlines()) != expected_wrapper_paths:
        raise ValueError("authorization wrapper must contain exactly two JSON blobs")
    for relative in sorted(expected_wrapper_paths):
        _git_regular_blob(root, wrapper, relative, "authorization wrapper JSON blob")
    raw_diff = _git(root, "diff-tree", "--no-commit-id", "--raw", "-r", wrapper)
    wrapper_modes = []
    for line in raw_diff.splitlines():
        metadata, path = line.split("\t", 1)
        fields = metadata.split()
        old_mode = fields[0][1:]
        new_mode = fields[1]
        status = fields[4]
        if path not in expected_wrapper_paths or new_mode != "100644" or status not in {"A", "M"}:
            raise ValueError("authorization wrapper blobs must use allowed regular JSON modes")
        if status == "M" and old_mode != "100644":
            raise ValueError("modified authorization blob has disallowed prior mode")
        if status == "A" and old_mode != "000000":
            raise ValueError("added authorization blob has disallowed prior mode")
        wrapper_modes.append(path)
    if set(wrapper_modes) != expected_wrapper_paths or len(wrapper_modes) != 2:
        raise ValueError("authorization wrapper raw diff must contain exactly two JSON blobs")

    freeze = _strict_object(_git_bytes(root, wrapper, FREEZE_RELATIVE), "R1 execution freeze")
    receipt = _strict_object(_git_bytes(root, wrapper, RECEIPT_RELATIVE), "R1 review receipt")
    _verify_authorization_documents(freeze, receipt, target)

    runtime_paths = expected_runtime_paths(root, target)
    runtime_rows = _verify_identity_rows(
        root, target, freeze["runtime_files"], "runtime freeze",
        expected_paths=runtime_paths,
    )
    r1_identity_map = {row["path"]: row["sha256"] for row in runtime_rows}
    if (len(r1_identity_map) != len(runtime_rows) or not r1_identity_map or
            any(_HEX64.fullmatch(value) is None for value in r1_identity_map.values())):
        raise ValueError("verified R1 runtime identity map is invalid")
    clean_rows = _verify_identity_rows(
        root, target, freeze["clean_files"], "clean input freeze",
        expected_paths=CLEAN_FILE_PATHS,
    )
    science = _verify_frozen_science_manifest(root, target)

    if check_remote:
        branch = _git(root, "branch", "--show-current")
        if branch != BRANCH:
            raise ValueError("R1 branch mismatch")
        subprocess.run(["git", "fetch", "origin", BRANCH], cwd=root, check=True,
                       capture_output=True, text=True)
        remote = _git(root, "rev-parse", f"origin/{BRANCH}")
        counts = _git(root, "rev-list", "--left-right", "--count",
                      f"HEAD...origin/{BRANCH}").split()
        if remote != wrapper or counts != ["0", "0"]:
            raise ValueError("R1 remote synchronization mismatch")
    return {"wrapper_commit": wrapper, "target_commit": target,
            "freeze": freeze, "receipt": receipt, "runtime_rows": runtime_rows,
            "r1_identity_map": r1_identity_map, "clean_rows": clean_rows,
            "science": science}


def _verify_preregistration(prereg: dict) -> None:
    """Validate the immutable original preregistration, including every nested value."""
    if set(prereg) != _PREREG_KEYS:
        raise ValueError("R1 preregistration exact schema mismatch")
    if _canonical_json_sha(prereg) != _BASE_PREREG_CANONICAL_SHA256:
        raise ValueError("R1 preregistration exact nested type/value mismatch")
    if not _valid_utc_timestamp(prereg.get("created_at_utc")):
        raise ValueError("R1 preregistration timestamp is malformed")
    predecessor = prereg.get("predecessor_failure")
    science = prereg.get("scientific_freeze")
    namespace = prereg.get("namespace")
    for value, label in (
            (predecessor.get("path") if type(predecessor) is dict else None,
             "predecessor failure"),
            (science.get("hash_manifest") if type(science) is dict else None,
             "scientific freeze"),
            (namespace.get("artifact_root") if type(namespace) is dict else None,
             "artifact root"),
            (namespace.get("ledger_path") if type(namespace) is dict else None,
             "ledger"),
            (namespace.get("marker_path") if type(namespace) is dict else None,
             "marker"),
    ):
        _canonical_target_path(value, label)
    if prereg["support_contract"]["join_keys"] != list(_BASE_JOIN_KEYS):
        raise ValueError("base preregistration join identity mismatch")


def _verify_amendment(amendment: dict) -> None:
    expected_keys = {
        "amendment_id", "correction", "created_at_utc", "original_preregistration",
        "previous_amendment_sha256", "reason", "schema", "status",
    }
    if set(amendment) != expected_keys:
        raise ValueError("R1 preregistration amendment exact schema mismatch")
    if _canonical_json_sha(amendment) != _AMENDMENT_CANONICAL_SHA256:
        raise ValueError("R1 preregistration amendment exact nested type/value mismatch")
    if not _valid_utc_timestamp(amendment.get("created_at_utc")):
        raise ValueError("R1 preregistration amendment timestamp is malformed")
    original = amendment["original_preregistration"]
    if (type(original) is not dict or set(original) != {"path", "sha256"} or
            _canonical_target_path(original["path"], "amendment original") != PREREG_RELATIVE or
            type(original["sha256"]) is not str or
            _HEX64.fullmatch(original["sha256"]) is None or
            original["sha256"] != _BASE_PREREG_SHA256):
        raise ValueError("R1 amendment original preregistration identity mismatch")
    if (type(amendment["previous_amendment_sha256"]) is not str or
            amendment["previous_amendment_sha256"] != "0" * 64):
        raise ValueError("R1 amendment hash-chain predecessor mismatch")
    correction = amendment["correction"]
    if (type(correction) is not dict or
            set(correction) != {"amended_value", "original_value", "path", "rationale"} or
            correction["path"] != "support_contract.join_keys" or
            correction["original_value"] != list(_BASE_JOIN_KEYS) or
            correction["amended_value"] != list(JOIN_KEYS)):
        raise ValueError("R1 preregistration amendment correction mismatch")
    if amendment["status"] == "PREREGISTERED_BEFORE_IMPLEMENTATION":
        raise ValueError("amended content must not claim preregistration before implementation")


def verify_effective_preregistration(artifact: str | Path) -> dict:
    """Authenticate base plus append-only amendment and return their effective contract."""
    artifact = Path(artifact)
    base_path = artifact / PurePosixPath(PREREG_RELATIVE).name
    amendment_path = artifact / PurePosixPath(AMENDMENT_RELATIVE).name
    _require_regular(base_path, "R1 base preregistration")
    _require_regular(amendment_path, "R1 preregistration amendment")
    base_payload = _read_regular_bytes(base_path, "R1 base preregistration")
    amendment_payload = _read_regular_bytes(
        amendment_path, "R1 preregistration amendment")
    if _sha(base_payload) != _BASE_PREREG_SHA256:
        raise ValueError("R1 original preregistration byte identity mismatch")
    if _sha(amendment_payload) != _AMENDMENT_SHA256:
        raise ValueError("R1 preregistration amendment byte identity mismatch")
    prereg = _strict_object(base_payload, "R1 preregistration")
    amendment = _strict_object(amendment_payload, "R1 preregistration amendment")
    _verify_preregistration(prereg)
    _verify_amendment(amendment)
    effective = dict(prereg)
    effective["support_contract"] = dict(prereg["support_contract"])
    effective["support_contract"]["join_keys"] = list(JOIN_KEYS)
    effective["status"] = "BASE_PREREGISTRATION_PLUS_POST_IMPLEMENTATION_AMENDMENT"
    effective["governance_amendment_sha256"] = _AMENDMENT_SHA256
    return effective


def _bind_scenario_manifest_paths(inventory: dict, capabilities: dict,
                                  identities: dict) -> dict:
    available = capabilities["available"]
    if set(identities) != set(available):
        raise ValueError("R1 scenario manifest identity set mismatch")
    rows = inventory.get("scenario_inventory") if type(inventory) is dict else None
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise ValueError("R1 scenario inventory rows are malformed")
    by_scenario = {}
    for row in rows:
        scenario = row.get("id")
        if scenario in available:
            if scenario in by_scenario:
                raise ValueError(f"duplicate R1 scenario inventory row: {scenario}")
            by_scenario[scenario] = row
    if set(by_scenario) != set(available):
        raise ValueError("R1 scenario inventory exact available set mismatch")

    result, canonical_paths, pinned_objects = {}, set(), set()
    for scenario in sorted(available):
        identity = identities[scenario]
        if type(identity) is not dict or set(identity) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"{scenario} R1 manifest identity schema mismatch")
        path = identity["path"]
        if (type(path) is not str or not path or "\\" in path or
                any(ord(character) < 32 or ord(character) == 127 for character in path)):
            raise ValueError(f"{scenario} protected manifest path is malformed")
        candidate = PurePosixPath(path)
        if (not candidate.is_absolute() or candidate.as_posix() != path or
                any(part in {"", ".", ".."} for part in candidate.parts)):
            raise ValueError(f"{scenario} protected manifest path is noncanonical")
        if (type(identity["sha256"]) is not str or
                _HEX64.fullmatch(identity["sha256"]) is None or
                isinstance(identity["size_bytes"], bool) or
                type(identity["size_bytes"]) is not int or identity["size_bytes"] <= 0):
            raise ValueError(f"{scenario} protected manifest identity is malformed")
        capability_identity = available[scenario]["manifest_identity"]
        if _canonical_json_sha(identity) != _canonical_json_sha(capability_identity):
            raise ValueError(f"{scenario} capability manifest identity mismatch")
        binding = available[scenario]["manifest_binding"]
        if (type(binding) is not dict or set(binding) != {"dev", "ino"} or
                any(isinstance(binding[key], bool) or type(binding[key]) is not int or
                    binding[key] < 0 for key in ("dev", "ino"))):
            raise ValueError(f"{scenario} protected manifest binding is malformed")
        pinned = (binding["dev"], binding["ino"])
        if path in canonical_paths or pinned in pinned_objects:
            raise ValueError("canonical protected manifest path/object reused across scenarios")
        canonical_paths.add(path)
        pinned_objects.add(pinned)
        result[scenario] = {"scenario": scenario, **identity,
                            "binding": {"dev": binding["dev"], "ino": binding["ino"]}}
    return result


def _build_scenario_access_plan(capabilities: dict, identities: dict) -> tuple[dict, ...]:
    """Bind every protected root/child path and object to exactly one scenario."""
    available = capabilities.get("available") if type(capabilities) is dict else None
    if type(available) is not dict or set(available) != set(identities):
        raise ValueError("R1 protected access-plan scenario set mismatch")
    rows, seen_paths, seen_objects = [], set(), set()

    def append(*, scenario: str, path: object, sha256: object, size_bytes: object,
               dev: object, ino: object, kind: str) -> None:
        if (type(path) is not str or not path or "\\" in path or
                any(ord(character) < 32 or ord(character) == 127 for character in path)):
            raise ValueError(f"{scenario} R1 protected access-plan path is malformed")
        candidate = PurePosixPath(path)
        if (not candidate.is_absolute() or candidate.as_posix() != path or
                any(part in {"", ".", ".."} for part in candidate.parts)):
            raise ValueError(f"{scenario} R1 protected access-plan path is noncanonical")
        if (type(sha256) is not str or _HEX64.fullmatch(sha256) is None or
                isinstance(size_bytes, bool) or type(size_bytes) is not int or size_bytes <= 0 or
                isinstance(dev, bool) or type(dev) is not int or dev < 0 or
                isinstance(ino, bool) or type(ino) is not int or ino < 0):
            raise ValueError(f"{scenario} R1 protected access-plan identity is malformed")
        object_identity = (dev, ino)
        if path in seen_paths or object_identity in seen_objects:
            raise ValueError("R1 protected path/object reused across scenarios or roles")
        info = _require_regular(Path(path), f"{scenario} R1 protected {kind}")
        if ((info.st_dev, info.st_ino, info.st_size) != (dev, ino, size_bytes)):
            raise ValueError(f"{scenario} R1 protected access-plan live identity mismatch")
        seen_paths.add(path)
        seen_objects.add(object_identity)
        rows.append({"scenario": scenario, "path": path, "sha256": sha256,
                     "size_bytes": size_bytes, "dev": dev, "ino": ino,
                     "kind": kind})

    for scenario in sorted(available):
        identity = identities[scenario]
        binding = identity["binding"]
        append(scenario=scenario, path=identity["path"], sha256=identity["sha256"],
               size_bytes=identity["size_bytes"], dev=binding["dev"],
               ino=binding["ino"], kind="RECEIVER_MANIFEST")
        sidecar = available[scenario]["sidecar"]
        if sidecar.get("scenario") != scenario or type(sidecar.get("children")) is not list:
            raise ValueError(f"{scenario} R1 sidecar scenario/children mismatch")
        for child in sidecar["children"]:
            if type(child) is not dict or child.get("scenario") != scenario:
                raise ValueError(f"{scenario} R1 child scenario mismatch")
            child_path = child.get("canonical_path")
            suffix = PurePosixPath(child_path).suffix if type(child_path) is str else ""
            append(scenario=scenario, path=child_path, sha256=child.get("sha256"),
                   size_bytes=child.get("size_bytes"), dev=child.get("_preclaim_dev"),
                   ino=child.get("_preclaim_ino"),
                   kind=suffix.upper().lstrip(".") or "FILE")
    return tuple(rows)


class R1ScenarioAccessGate(AccessGate):
    """R1-local scenario-bound facade over the frozen generic AccessGate."""

    def __init__(self, ledger_path: str | Path, access_plan: tuple[dict, ...]):
        super().__init__(ledger_path)
        self._r1_plan = {row["path"]: dict(row) for row in access_plan}
        if len(self._r1_plan) != len(access_plan) or not self._r1_plan:
            raise ValueError("R1 scenario-bound access plan is empty or duplicated")

    def _r1_row(self, path: str | Path) -> dict:
        candidate = str(Path(path).resolve(strict=True))
        row = self._r1_plan.get(candidate)
        if row is None:
            raise PermissionError("path absent from R1 scenario-bound access plan")
        return row

    def register_pinned(self, path, *, expected_sha256, expected_size, kind,
                        preclaim_dev=None, preclaim_ino=None):
        row = self._r1_row(path)
        if ({"sha256": expected_sha256, "size_bytes": expected_size, "kind": kind,
             "dev": preclaim_dev, "ino": preclaim_ino} !=
                {key: row[key] for key in ("sha256", "size_bytes", "kind", "dev", "ino")}):
            raise ValueError("R1 root registration differs from scenario-bound access plan")
        canonical = super().register_pinned(
            path, expected_sha256=expected_sha256, expected_size=expected_size,
            kind=kind, preclaim_dev=preclaim_dev, preclaim_ino=preclaim_ino)
        self._allow[str(canonical)]["scenario"] = row["scenario"]
        return canonical

    def register_sidecar_children(self, root_manifest, children):
        root_row = self._r1_row(root_manifest)
        if root_row["kind"] != "RECEIVER_MANIFEST":
            raise ValueError("R1 sidecar root role mismatch")
        observed_paths = []
        for child in children:
            row = self._r1_row(child.get("canonical_path") if type(child) is dict else "")
            expected = {
                "canonical_path": row["path"], "sha256": row["sha256"],
                "size_bytes": row["size_bytes"], "scenario": row["scenario"],
                "_preclaim_dev": row["dev"], "_preclaim_ino": row["ino"],
            }
            if (row["scenario"] != root_row["scenario"] or
                    any(child.get(key) != value for key, value in expected.items())):
                raise ValueError("R1 sidecar child differs from scenario-bound access plan")
            observed_paths.append(row["path"])
        expected_paths = [row["path"] for row in self._r1_plan.values()
                          if row["scenario"] == root_row["scenario"] and
                          row["kind"] != "RECEIVER_MANIFEST"]
        if sorted(observed_paths) != sorted(expected_paths):
            raise ValueError("R1 sidecar exact child-set mismatch")
        parsed = super().register_sidecar_children(root_manifest, children)
        for child in parsed:
            self._allow[str(child)]["scenario"] = root_row["scenario"]
        return parsed

    def consume(self, path, *, scenario, phase, purpose, consumer, operation="READ"):
        row = self._r1_row(path)
        if row["scenario"] != scenario:
            self._deny(path, "R1_SCENARIO_BINDING_MISMATCH", scenario=scenario,
                       phase=phase, purpose=purpose)
            raise PermissionError("R1 protected path is bound to a different scenario")
        return super().consume(path, scenario=scenario, phase=phase, purpose=purpose,
                               consumer=consumer, operation=operation)


def preflight(root: str | Path, *, check_remote: bool = True) -> dict:
    root = Path(root).resolve(strict=True)
    with install_and_verify_adapter():
        _verify_adapter_bindings()
    checked = verify_execution_freeze(root, check_remote=check_remote)
    artifact = root / ARTIFACT_RELATIVE
    marker = root / MARKER_RELATIVE
    runtime_paths = tuple(row["path"] for row in checked["runtime_rows"])
    _verify_preaccess_input_tree(root, runtime_paths)
    _verify_no_runtime_shadows(root, runtime_paths)
    artifact_binding = verify_zero_access_state(artifact, marker)

    effective_preregistration = verify_effective_preregistration(artifact)
    inventory = _strict_object(
        _read_regular_bytes(artifact / "data_inventory.json", "inventory"), "inventory")
    capabilities = validate_preaccess_capabilities(
        _strict_object(_read_regular_bytes(
            artifact / "protected_capabilities.json", "capabilities"), "capabilities"))
    manifest_identities = validate_protected_manifest_inventory(
        inventory, required=tuple(capabilities["available"]))
    scenario_path_identities = _bind_scenario_manifest_paths(
        inventory, capabilities, manifest_identities)
    scenario_access_plan = _build_scenario_access_plan(
        capabilities, scenario_path_identities)
    gate_identity_map = dict(checked["r1_identity_map"])
    for row in scenario_access_plan:
        key = f"protected::{row['scenario']}::{row['kind']}::{row['path']}"
        if key in gate_identity_map:
            raise ValueError("R1 gate identity key collision")
        gate_identity_map[key] = row["sha256"]

    clean_identities = []
    for row in checked["clean_rows"]:
        clean_identities.append({"path": str(root / row["path"]),
                                 "sha256": row["sha256"],
                                 "size_bytes": row["size_bytes"]})
    clean_runtime_root = Path(
        os.environ.get("GCSPO_R1_RUNTIME_SNAPSHOT", root)
    ).resolve(strict=True)
    clean_runtime_identities = [
        {
            **row,
            "path": str(clean_runtime_root / Path(row["path"]).relative_to(root)),
        }
        for row in clean_identities
    ]
    validate_clean_contrast_preaccess(
        clean_runtime_root / ARTIFACT_RELATIVE, clean_runtime_identities)
    return {**checked, "artifact": artifact, "marker": marker,
            "artifact_binding": artifact_binding,
            "effective_preregistration": effective_preregistration,
            "inventory": inventory, "capabilities": capabilities,
            "manifest_identities": scenario_path_identities,
            "scenario_path_identities": scenario_path_identities,
            "scenario_access_plan": scenario_access_plan,
            "gate_identity_map": gate_identity_map,
            "clean_identities": clean_identities}


def _final_manifest_rows(artifact: Path, expected_paths: tuple[str, ...]) -> list[dict]:
    rows = []
    for relative in expected_paths:
        path = artifact / relative
        payload = _read_regular_bytes(path, "R1 final artifact")
        rows.append({"path": relative, "sha256": _sha(payload),
                     "size_bytes": len(payload)})
    return rows


def _verify_post_attestation_report(artifact: Path, name: str, target_commit: str) -> None:
    path = artifact / name
    document = json.loads(_read_regular_bytes(path, f"R1 {name}"))
    expected_schema = {
        "verifier_report.json": "gnss-doppler-lab.gcspo-stage0.r1-verifier-report.v1",
        "fresh_clone_verifier_report.json": "gnss-doppler-lab.gcspo-stage0.fresh-clone-verifier-report.v2",
    }[name]
    manifest_sha = hashlib.sha256(_read_regular_bytes(
        artifact / "artifact_manifest_sha256.json", "R1 artifact manifest")).hexdigest()
    checks = document.get("checks")
    if (document.get("schema") != expected_schema or
            document.get("target_commit") != target_commit or
            document.get("artifact_manifest_sha256") != manifest_sha or
            document.get("overall_status") != "PASS" or
            document.get("verified_run_status") != "VALID_SCIENCE" or
            document.get("exit_code") != 0 or
            not isinstance(document.get("evidence_commit"), str) or
            not re.fullmatch(r"[0-9a-f]{40}", document["evidence_commit"]) or
            not isinstance(checks, list) or not checks or
            any(not isinstance(row, dict) or row.get("status") != "PASS" for row in checks) or
            not str(document.get("attestation_scope", "")).startswith("post-evidence")):
        raise ValueError(f"R1 post-attestation report binding mismatch: {name}")


def verify_r1_final_manifest(artifact: str | Path) -> dict:
    """Standalone-safe exact-inventory verification of the R1 final manifest."""
    artifact = Path(artifact)
    files, directories = _scan_tree(artifact, "R1 final artifact")
    observed_files = set(files)
    scientific_files = set(_FINAL_EXPECTED_FILES)
    allowed_files = scientific_files | set(_POST_ATTESTATION_FILES)
    if (not scientific_files.issubset(observed_files) or not observed_files.issubset(allowed_files) or
            tuple(sorted(directories)) != _FINAL_EXPECTED_DIRECTORIES):
        raise ValueError("R1 final artifact exact file/directory inventory mismatch")
    manifest_path = artifact / FINAL_MANIFEST_NAME
    _require_regular(manifest_path, "R1 final manifest")
    document = _strict_object(
        _read_regular_bytes(manifest_path, "R1 final manifest"), "R1 final manifest")
    if set(document) != {"schema", "invocation_id", "files"}:
        raise ValueError("R1 final manifest exact schema mismatch")
    if (document["schema"] !=
            "gnss-doppler-lab.gcspo-stage0.r1-artifact-manifest.v1" or
            document["invocation_id"] != INVOCATION_ID or
            type(document["files"]) is not list):
        raise ValueError("R1 final manifest identity/type mismatch")
    expected_members = tuple(path for path in _FINAL_EXPECTED_FILES
                             if path != FINAL_MANIFEST_NAME)
    observed_paths = []
    for index, row in enumerate(document["files"]):
        if type(row) is not dict or set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"R1 final manifest row schema mismatch: {index}")
        relative = _canonical_target_path(row["path"], f"R1 final manifest row {index}")
        if (type(row["sha256"]) is not str or _HEX64.fullmatch(row["sha256"]) is None or
                isinstance(row["size_bytes"], bool) or
                type(row["size_bytes"]) is not int or row["size_bytes"] < 0):
            raise ValueError(f"R1 final manifest row type mismatch: {index}")
        observed_paths.append(relative)
    if tuple(observed_paths) != expected_members:
        raise ValueError("R1 final manifest exact member inventory mismatch")
    if document["files"] != _final_manifest_rows(artifact, expected_members):
        raise ValueError("R1 final manifest file identity mismatch")
    implementation = _strict_object(_read_regular_bytes(
        artifact / "implementation_manifest.json", "R1 implementation manifest"),
        "R1 implementation manifest")
    target_commit = implementation.get("target_commit")
    if not isinstance(target_commit, str) or _HEX40.fullmatch(target_commit) is None:
        raise ValueError("R1 implementation target binding mismatch")
    for name in _POST_ATTESTATION_FILES:
        if name in observed_files:
            _verify_post_attestation_report(artifact, name, target_commit)
    return document


def verify_r1_final(artifact: str | Path) -> dict:
    """Independently verify the exact package and reconstruct final R1 evidence."""
    artifact = Path(artifact)
    manifest = verify_r1_final_manifest(artifact)
    verify_effective_preregistration(artifact)
    verdict = _strict_object(_read_regular_bytes(
        artifact / "final_verdict.json", "R1 final verdict"), "R1 final verdict")
    expected_verdict_keys = {
        "schema", "verdict", "scientific_status", "protected_run_count",
        "gates", "evidence", "unavailable",
    }
    if (set(verdict) != expected_verdict_keys or
            verdict["schema"] != "gnss-doppler-lab.gcspo-stage0.final-verdict.v2" or
            verdict["verdict"] not in {"GO_FOR_NEURAL_STAGE1", "NO_GO_PHYSICAL_HYPOTHESIS"} or
            verdict["scientific_status"] != "VALID_SCIENCE" or
            isinstance(verdict["protected_run_count"], bool) or
            type(verdict["protected_run_count"]) is not int or
            verdict["protected_run_count"] != 1 or type(verdict["gates"]) is not list or
            type(verdict["evidence"]) is not dict or type(verdict["unavailable"]) is not list):
        raise ValueError("R1 final verdict exact schema/type/value mismatch")
    from .gcspo_verify_artifacts import reconstruct_final_evidence
    from .gcspo_verify_reconstruct import validate_access_ledger, verify_evidence_document

    records = []
    ledger_payload = _read_regular_bytes(
        artifact / "access_ledger.jsonl", "R1 access ledger")
    for index, line in enumerate(ledger_payload.splitlines()):
        if line.strip():
            records.append(_strict_object(line, f"R1 access ledger row {index}"))
    validate_access_ledger(records)
    with install_and_verify_adapter():
        _verify_adapter_bindings()
        reconstructed = reconstruct_final_evidence(artifact)
    if _canonical_json_sha(verdict["evidence"]) != _canonical_json_sha(reconstructed):
        raise ValueError("R1 reported evidence differs from standalone reconstruction")
    verify_evidence_document(verdict)
    return {"status": "PASS", "manifest": manifest,
            "protected_run_count": 1, "verdict": verdict["verdict"]}


def _write_full_manifest(artifact: Path) -> None:
    artifact = Path(artifact)
    _verify_directory_root(artifact, "R1 artifact")
    path = artifact / FINAL_MANIFEST_NAME
    if _lstat(path) is not None:
        raise ValueError("R1 final manifest already exists or is nonregular")
    files, directories = _scan_tree(artifact, "R1 final artifact before manifest")
    expected_members = tuple(name for name in _FINAL_EXPECTED_FILES
                             if name != FINAL_MANIFEST_NAME)
    if (tuple(sorted(files)) != expected_members or
            tuple(sorted(directories)) != _FINAL_EXPECTED_DIRECTORIES):
        raise ValueError("R1 final artifact exact inventory mismatch before manifest")
    canonical_write_json(path, {
        "schema": "gnss-doppler-lab.gcspo-stage0.r1-artifact-manifest.v1",
        "invocation_id": INVOCATION_ID,
        "files": _final_manifest_rows(artifact, expected_members),
    })
    verify_r1_final_manifest(artifact)


_PRECLAIM_SNAPSHOT_KEYS = (
    "wrapper_commit", "target_commit", "runtime_rows", "clean_rows",
    "r1_identity_map", "artifact_binding", "effective_preregistration",
    "inventory", "capabilities", "scenario_path_identities",
    "scenario_access_plan", "gate_identity_map", "clean_identities",
)


def _preclaim_snapshot_sha(checked: dict) -> str:
    missing = set(_PRECLAIM_SNAPSHOT_KEYS) - set(checked)
    if missing:
        raise ValueError(f"R1 preclaim snapshot keys absent: {sorted(missing)}")
    return _canonical_json_sha({key: checked[key] for key in _PRECLAIM_SNAPSHOT_KEYS})


def _write_r1_implementation_manifest(artifact: Path, checked: dict) -> None:
    artifact = Path(artifact).resolve(strict=True)
    path = artifact / "implementation_manifest.json"
    if _lstat(path) is not None:
        raise ValueError("R1 implementation compatibility manifest already exists")
    portable_clean = []
    for row in checked["clean_identities"]:
        source = Path(row["path"]).resolve(strict=True)
        try:
            relative = source.relative_to(artifact).as_posix()
        except ValueError as exc:
            raise ValueError("R1 clean identity escapes artifact root") from exc
        portable_clean.append({**row, "path": relative})
    canonical_write_json(path, {
        "schema": "gnss-doppler-lab.gcspo-stage0.r1-reconstruction-manifest.v1",
        "target_commit": checked["target_commit"],
        "clean_scientific_artifacts": portable_clean,
        "manifest_excludes_self": True,
    })
    document = _strict_object(
        _read_regular_bytes(path, "R1 implementation compatibility manifest"),
        "R1 implementation compatibility manifest")
    if (set(document) != {"schema", "target_commit", "clean_scientific_artifacts",
                          "manifest_excludes_self"} or
            document["schema"] !=
            "gnss-doppler-lab.gcspo-stage0.r1-reconstruction-manifest.v1" or
            document["target_commit"] != checked["target_commit"] or
            document["clean_scientific_artifacts"] != portable_clean or
            document["manifest_excludes_self"] is not True):
        raise ValueError("R1 implementation compatibility manifest mismatch")


def _quarantine_failed_final_verdict_bound(artifact: Path):
    verdict = Path(artifact) / "final_verdict.json"
    if not verdict.is_file():
        return None
    quarantine = Path(tempfile.mkdtemp(prefix=".gcspo-r1-poststart-quarantine-"))
    target = quarantine / verdict.name
    payload = verdict.read_bytes()
    target.write_bytes(payload)
    with target.open("rb") as handle:
        os.fsync(handle.fileno())
    verdict.unlink()
    return target


def protected(root: str | Path, *, check_remote: bool = True) -> int:
    root = Path(root).resolve(strict=True)
    claimed = False
    artifact_fd = None
    artifact = root / ARTIFACT_RELATIVE
    try:
        checked = preflight(root, check_remote=check_remote)
        first_snapshot = _preclaim_snapshot_sha(checked)
        rechecked = preflight(root, check_remote=check_remote)
        if _preclaim_snapshot_sha(rechecked) != first_snapshot:
            raise ValueError("R1 full preflight snapshot changed before exact-once claim")
        checked = rechecked
        snapshot_root = Path(os.environ.get("GCSPO_R1_RUNTIME_SNAPSHOT", "")).resolve(strict=True)
        if os.environ.get("GCSPO_R1_SNAPSHOT_ACTIVE") != checked["target_commit"]:
            raise ValueError("R1 reviewed runtime snapshot target mismatch")
        snapshot_artifact = snapshot_root / ARTIFACT_RELATIVE
        snapshot_clean = [{**row, "path": str(snapshot_root / Path(row["path"]).relative_to(root))}
                          for row in checked["clean_identities"]]
        validate_clean_contrast_preaccess(snapshot_artifact, snapshot_clean)
        claim_repository_anchor(root, wrapper_commit=checked["wrapper_commit"],
                                target_commit=checked["target_commit"])
        _claim, artifact_fd = claim_once(
            checked["marker"], wrapper_commit=checked["wrapper_commit"],
            target_commit=checked["target_commit"],
            parent_binding=checked["artifact_binding"], keep_parent_fd=True)
        claimed = True
        artifact = Path(f"/proc/self/fd/{artifact_fd}")
        canonical_write_json(artifact / "protected_run_provenance.json", {
            "schema": "gnss-doppler-lab.gcspo-stage0.r1-protected-run-provenance.v1",
            "invocation_id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
            "authorization_wrapper_commit": checked["wrapper_commit"],
            "target_commit": checked["target_commit"],
            "receipt_path": RECEIPT_RELATIVE, "execution_freeze_path": FREEZE_RELATIVE,
        })
        _write_r1_implementation_manifest(artifact, checked)
        gate = R1ScenarioAccessGate(
            artifact / "access_ledger.jsonl", checked["scenario_access_plan"])
        gate.set_preflight(
            clean_only_pass=True, reviews_pass=True,
            freeze_sha=checked["wrapper_commit"],
            frozen_hashes=checked["gate_identity_map"],
        )
        gate.set_remote_sync(local_sha=checked["wrapper_commit"],
                             remote_sha=checked["wrapper_commit"], ahead=0, behind=0,
                             clean=True)
        from . import gcspo_evaluate
        with install_and_verify_adapter():
            _verify_adapter_bindings()
            verdict = gcspo_evaluate.run_one_shot(
                artifact_dir=artifact, input_artifact_dir=snapshot_artifact,
                repo_root=snapshot_root, inventory=checked["inventory"],
                gate=gate, manifest_identities=checked["scenario_path_identities"],
                clean_identities=snapshot_clean,
                capabilities=checked["capabilities"],
            )
        _write_full_manifest(artifact)
        os.close(artifact_fd); artifact_fd = None
        print(f"PROTECTED_ONE_SHOT_PASS verdict={verdict}", flush=True)
        return 0
    except Exception as exc:
        if claimed:
            _quarantine_failed_final_verdict_bound(artifact)
        if artifact_fd is not None:
            os.close(artifact_fd)
        print(str(exc), file=sys.stderr)
        return 2


def main(root: str | Path | None = None) -> int:
    if root is None:
        root = os.environ.get("GCSPO_R1_REPO_ROOT") or Path(__file__).resolve().parents[2]
    return protected(root)
