import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/crid_stage0_r3b_terminal_provenance_closure"
R3A = ROOT / "artifacts/crid_stage0_r3a_independent_reference_estimand_repair"
TARGET = "aa3833fb73ae572521e3a3ac8f2b865d3aac0307"
SCRIPT = ROOT / "scripts/verify_crid_r3b_terminal_provenance_closure.py"
SPEC = importlib.util.spec_from_file_location("verify_crid_r3b", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_commit_r3a_bindings_and_fresh_execution_pass():
    attestation = json.loads((ART / "terminal_attestation.json").read_text())
    assert attestation["source_checkout"]["fresh_clone"] is True
    assert attestation["source_checkout"]["target_commit_sha"] == TARGET
    assert attestation["bindings"]["verifier_sha256"] == _sha256(
        ROOT / "scripts/verify_crid_r3a_estimand_repair.py"
    )
    assert attestation["bindings"]["artifact_manifest_sha256"] == _sha256(
        R3A / "artifact_manifest_sha256.json"
    )
    execution = attestation["execution"]
    assert execution["exit_code"] == 0
    assert execution["stderr"] == ""
    assert execution["stdout"] == (ART / "logs/r3a_verifier_stdout.txt").read_text()
    parsed = json.loads(execution["stdout"])
    assert parsed["status"] == "PASS"
    assert parsed["verdict"] == "INDEPENDENT_REFERENCE_ESTIMAND_REPAIR_PASS"


def test_stale_pre_finalization_log_is_preserved_and_superseded_only_by_attestation():
    classification = json.loads((ART / "historical_evidence_classification.json").read_text())
    stale = classification["preserved_evidence"][0]
    stale_path = ROOT / stale["path"]
    stale_payload = json.loads(stale_path.read_text())
    assert classification["classification"] == "HISTORICAL_STALE_PRE_FINALIZATION_EVIDENCE"
    assert classification["preservation"] == "PRESERVED_UNMODIFIED"
    assert classification["superseding_attestation"] == "terminal_attestation.json"
    assert stale["sha256"] == _sha256(stale_path)
    assert stale_payload["status"] == "PASS"
    assert stale_payload["verdict"] == "INCONCLUSIVE_REFERENCE_PROVENANCE"


def test_forbidden_scope_guards_are_all_false():
    attestation = json.loads((ART / "terminal_attestation.json").read_text())
    assert attestation["attestation_scope"] == "TERMINAL_PROVENANCE_CLOSURE_ONLY"
    assert all(value is False for value in attestation["scope_guards"].values())


def test_manifest_helper_detects_payload_and_manifest_hash_tamper(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = artifact / "payload.txt"
    payload.write_text("bound\n")
    manifest = {
        "schema": "gnss-doppler-lab.crid-r3b-artifact-manifest.v1",
        "file_count": 1,
        "files": [{"path": "payload.txt", "sha256": _sha256(payload), "size_bytes": payload.stat().st_size}],
        "status": "PASS",
    }
    manifest_path = artifact / "artifact_manifest_sha256.json"
    manifest_path.write_text(json.dumps(manifest))
    assert VERIFY.verify_manifest(artifact)["status"] == "PASS"

    payload.write_text("tampered\n")
    assert VERIFY.verify_manifest(artifact)["status"] == "FAIL"
    payload.write_text("bound\n")
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    assert VERIFY.verify_manifest(artifact)["status"] == "FAIL"


def test_committed_r3b_artifact_and_manifest_tamper_fail_closed(tmp_path):
    baseline = VERIFY.verify_artifact(ART)
    assert baseline["status"] == "PASS"
    assert baseline["verdict"] == "TERMINAL_PROVENANCE_CLOSURE_PASS"

    copied = tmp_path / "r3b"
    shutil.copytree(ART, copied)
    attestation = copied / "terminal_attestation.json"
    attestation.write_bytes(attestation.read_bytes() + b" ")
    result = subprocess.run(
        ["python3", str(SCRIPT), "--artifact", str(copied)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert '"status": "FAIL"' in result.stdout

    shutil.rmtree(copied)
    shutil.copytree(ART, copied)
    manifest_path = copied / "artifact_manifest_sha256.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = subprocess.run(
        ["python3", str(SCRIPT), "--artifact", str(copied)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert '"status": "FAIL"' in result.stdout


def test_only_new_r3b_closure_paths_differ_from_target_commit():
    result = subprocess.run(
        ["git", "diff", "--name-only", TARGET, "--"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = {line for line in result.stdout.splitlines() if line}
    assert changed
    assert all(
        path.startswith("artifacts/crid_stage0_r3b_terminal_provenance_closure/")
        or path == "scripts/verify_crid_r3b_terminal_provenance_closure.py"
        or path == "tests/test_crid_r3b_terminal_provenance_closure.py"
        for path in changed
    )
