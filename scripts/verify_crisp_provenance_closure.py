#!/usr/bin/env python3
"""Verify the administrative CRISP Stage-0 provenance closure.

PASS means provenance metadata is closed with frozen science. It does not mean
that CRISP passed its physical hypothesis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/crisp_stage0_static"
CLOSURE_PATH = ARTIFACT / "result_provenance_closure.json"
MANIFEST_PATH = ARTIFACT / "artifact_manifest_sha256.json"
MAIN_SHA = "461eb4dc7bb794e719295daf028f6811658ba37f"
PREREG_SHA = "9a3119fc77ad830f438d723a225274f19c43be90"
SCIENCE_SHA = "cd36f1d2a3ab3ca64c85747db1e1463198a966ee"
NO_GO = "NO_GO_CRISP_PHYSICAL_HYPOTHESIS"
RECOMMENDATION = "Terminate CRISP as a neural Stage-1 path; do not retune this result."
ADMIN_VERDICT_FIELDS = {
    "experiment_rerun", "provenance_closure_status", "science_metrics_modified",
    "scientific_result_sha",
}
ALLOWED_CHANGES = {
    "artifacts/crisp_stage0_static/README.md",
    "artifacts/crisp_stage0_static/artifact_manifest_sha256.json",
    "artifacts/crisp_stage0_static/final_verdict.json",
    "artifacts/crisp_stage0_static/result_provenance_closure.json",
    "scripts/verify_crisp_provenance_closure.py",
    "tests/test_crisp_provenance_closure.py",
}
PROTECTED_CODE = {
    "src/gnss_doppler_lab/crisp.py",
    "src/gnss_doppler_lab/crisp_data.py",
    "scripts/run_crisp_stage0.py",
}
RERUN_TRACE_PATHS = {
    "artifacts/crisp_stage0_static/execution_status.json",
    "artifacts/crisp_stage0_static/reproduction_validation.json",
    "artifacts/crisp_stage0_static/runner/evaluate.exit_code",
    "artifacts/crisp_stage0_static/runner/evaluate.stderr.log",
    "artifacts/crisp_stage0_static/runner/evaluate.stdout.log",
    "artifacts/crisp_stage0_static/runtime_summary.json",
}


class ClosureError(RuntimeError):
    pass


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if check and result.returncode:
        raise ClosureError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob(ref: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise ClosureError(f"missing Git blob {ref}:{path}")
    return result.stdout


def git_json(ref: str, path: str) -> dict[str, Any]:
    return json.loads(git_blob(ref, path))


def load_closure() -> dict[str, Any]:
    return json.loads(CLOSURE_PATH.read_text())


def ensure_allowed_changes(changes: set[str], allowed: set[str] = ALLOWED_CHANGES) -> None:
    forbidden = changes - allowed
    if forbidden:
        raise ClosureError(f"forbidden closure changes: {sorted(forbidden)}")


def validate_document(doc: dict[str, Any]) -> None:
    required = {
        "schema", "version", "repository", "branch", "main_base_sha",
        "preregistration_sha", "execution_source_sha", "scientific_result_sha",
        "scientific_result_parent_verified", "preregistration_is_ancestor",
        "scientific_result_is_descendant_of_preregistration", "closure_parent_sha",
        "provenance_closure_parent_sha", "result_commit_pending",
        "experiment_rerun", "detector_code_modified", "thresholds_modified",
        "metrics_modified", "plots_modified", "allowed_administrative_changes",
        "original_scientific_file_hashes", "final_scientific_file_hashes",
        "original_final_hash_equality", "closure_verdict",
    }
    missing = required - set(doc)
    if missing:
        raise ClosureError(f"missing closure fields: {sorted(missing)}")
    if "provenance_closure_commit_sha" in doc:
        raise ClosureError("self-referential closure SHA is forbidden")
    if "PENDING_COMMIT" in json.dumps(doc, sort_keys=True):
        raise ClosureError("fake pending closure SHA is forbidden")
    expected = {
        "main_base_sha": MAIN_SHA,
        "preregistration_sha": PREREG_SHA,
        "execution_source_sha": PREREG_SHA,
        "scientific_result_sha": SCIENCE_SHA,
        "closure_parent_sha": SCIENCE_SHA,
        "provenance_closure_parent_sha": SCIENCE_SHA,
        "closure_verdict": "CRISP_PROVENANCE_CLOSED_WITH_SCIENCE_UNCHANGED",
    }
    for key, value in expected.items():
        if doc.get(key) != value:
            raise ClosureError(f"{key} binding mismatch")
    false_fields = (
        "result_commit_pending", "experiment_rerun", "detector_code_modified",
        "thresholds_modified", "metrics_modified", "plots_modified",
    )
    if any(doc.get(key) is not False for key in false_fields):
        raise ClosureError("closure mutation/rerun flags are not all false")
    true_fields = (
        "scientific_result_parent_verified", "preregistration_is_ancestor",
        "scientific_result_is_descendant_of_preregistration",
        "original_final_hash_equality",
    )
    if any(doc.get(key) is not True for key in true_fields):
        raise ClosureError("closure verification/equality flags are not all true")
    if set(doc["allowed_administrative_changes"]) != ALLOWED_CHANGES:
        raise ClosureError("administrative allowlist mismatch")
    original = doc["original_scientific_file_hashes"]
    final = doc["final_scientific_file_hashes"]
    if not original or original != final:
        raise ClosureError("original/final scientific hash maps differ")
    if any(len(value) != 64 for value in original.values()):
        raise ClosureError("invalid scientific SHA-256 entry")


def verify_hash_bindings(doc: dict[str, Any]) -> None:
    original = doc["original_scientific_file_hashes"]
    final = doc["final_scientific_file_hashes"]
    for path, expected in original.items():
        historical = sha256_bytes(git_blob(SCIENCE_SHA, path))
        current_path = ROOT / path
        if not current_path.is_file():
            raise ClosureError(f"missing scientific file: {path}")
        current = sha256_file(current_path)
        if historical != expected or current != final[path] or historical != current:
            raise ClosureError(f"scientific hash mutation: {path}")


def build_manifest() -> None:
    rows = []
    for path in sorted(ARTIFACT.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"artifact_manifest_sha256.json", "execution_status.json"}:
            continue
        rows.append({
            "path": path.relative_to(ARTIFACT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    value = {"schema": "gnss-doppler-lab.crisp-stage0-manifest.v1", "files": rows}
    MANIFEST_PATH.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_manifest(root: Path = ARTIFACT, manifest: dict[str, Any] | None = None) -> None:
    value = manifest if manifest is not None else json.loads(MANIFEST_PATH.read_text())
    paths = set()
    for row in value.get("files", []):
        rel = row["path"]
        if rel == "artifact_manifest_sha256.json":
            raise ClosureError("manifest contains its own hash")
        path = root / rel
        if rel in paths or not path.is_file():
            raise ClosureError(f"manifest missing/duplicate file: {rel}")
        paths.add(rel)
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ClosureError(f"manifest tamper: {rel}")
    required = {
        "final_verdict.json", "README.md", "result_provenance_closure.json",
        *[Path(path).relative_to("artifacts/crisp_stage0_static").as_posix()
          for path in load_closure()["original_scientific_file_hashes"]],
    }
    if not required <= paths:
        raise ClosureError(f"manifest coverage missing: {sorted(required - paths)}")


def validate_verdict() -> None:
    original = git_json(SCIENCE_SHA, "artifacts/crisp_stage0_static/final_verdict.json")
    current = json.loads((ARTIFACT / "final_verdict.json").read_text())
    if original["verdict"] != NO_GO or current["verdict"] != NO_GO:
        raise ClosureError("NO-GO verdict was not preserved")
    if original["go_checks"].get("core_detection") is not False:
        raise ClosureError("original core_detection is not false")
    if current["recommended_next_action"] != RECOMMENDATION:
        raise ClosureError("Stage-1 termination recommendation was not preserved")
    expected = dict(original)
    expected["result_commit_pending"] = False
    for key in ADMIN_VERDICT_FIELDS:
        expected[key] = current.get(key)
    if current != expected:
        raise ClosureError("non-administrative final verdict content changed")
    if current["scientific_result_sha"] != SCIENCE_SHA:
        raise ClosureError("missing scientific-result SHA")
    if current["experiment_rerun"] or current["science_metrics_modified"]:
        raise ClosureError("verdict claims science rerun/mutation")


def verify_git(precommit: bool = False) -> None:
    if git("rev-parse", "origin/main") != MAIN_SHA:
        raise ClosureError("origin/main base mismatch")
    if git("rev-parse", "origin/research/crisp-stage0-static") != SCIENCE_SHA:
        raise ClosureError("recorded scientific result is not baseline branch tip")
    for sha in (MAIN_SHA, PREREG_SHA, SCIENCE_SHA):
        git("cat-file", "-e", f"{sha}^{{commit}}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PREREG_SHA, SCIENCE_SHA], cwd=ROOT).returncode:
        raise ClosureError("preregistration is not scientific-result ancestor")
    head = git("rev-parse", "HEAD")
    if precommit:
        if head != SCIENCE_SHA:
            raise ClosureError("precommit closure must start at scientific result")
        changes = set(filter(None, git("diff", "--name-only", SCIENCE_SHA).splitlines()))
        changes |= set(filter(None, git("diff", "--cached", "--name-only", SCIENCE_SHA).splitlines()))
    else:
        parents = git("show", "-s", "--format=%P", head).split()
        if not parents or parents[0] != SCIENCE_SHA:
            raise ClosureError("scientific result is not closure commit parent")
        changes = set(filter(None, git("diff", "--name-only", SCIENCE_SHA, head).splitlines()))
    ensure_allowed_changes(changes)
    code_diff = set(filter(None, git("diff", "--name-only", PREREG_SHA, SCIENCE_SHA, "--", *sorted(PROTECTED_CODE)).splitlines()))
    if code_diff:
        raise ClosureError(f"execution detector/runner drift: {sorted(code_diff)}")
    closure_code_diff = set(filter(None, git("diff", "--name-only", SCIENCE_SHA, "HEAD", "--", *sorted(PROTECTED_CODE)).splitlines()))
    if not precommit and closure_code_diff:
        raise ClosureError(f"detector code modified by closure: {sorted(closure_code_diff)}")
    rerun_diff = set(filter(None, git("diff", "--name-only", SCIENCE_SHA, "HEAD", "--", *sorted(RERUN_TRACE_PATHS)).splitlines()))
    if not precommit and rerun_diff:
        raise ClosureError(f"experiment rerun trace changed: {sorted(rerun_diff)}")


def verify_evidence() -> None:
    reproduction = json.loads((ARTIFACT / "reproduction_validation.json").read_text())
    if reproduction["preregistration_sha"] != PREREG_SHA or reproduction["runs"] != 2:
        raise ClosureError("execution source/reproduction evidence mismatch")
    if (ARTIFACT / "runner/evaluate.exit_code").read_text().strip() != "0":
        raise ClosureError("first evaluation did not exit successfully")


def pass_message() -> str:
    return "PROVENANCE_CLOSURE_VERIFIED"


def verify(precommit: bool = False) -> None:
    doc = load_closure()
    validate_document(doc)
    verify_git(precommit=precommit)
    verify_hash_bindings(doc)
    validate_verdict()
    verify_evidence()
    verify_manifest()
    print(pass_message())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--precommit", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        build_manifest()
    verify(precommit=args.precommit)


if __name__ == "__main__":
    try:
        main()
    except ClosureError as exc:
        raise SystemExit(f"FAIL: {exc}")
