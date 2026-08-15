import importlib.util
import json
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_b0_cs_stage0.py"
    spec = importlib.util.spec_from_file_location("verify_b0_cs_test", path)
    result = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(result)
    return result


def test_manifest_is_deterministic_excludes_itself_and_verifies_hashes(tmp_path):
    mod = module()
    (tmp_path / "x.txt").write_text("x")
    first = mod.manifest_document(tmp_path)
    mod.write_manifest(tmp_path)
    second = json.loads((tmp_path / "artifact_manifest_sha256.json").read_text())
    assert first == second == mod.manifest_document(tmp_path)
    (tmp_path / "x.txt").write_text("changed")
    assert second != mod.manifest_document(tmp_path)


def test_allowed_verdict_vocabulary_is_exact():
    mod = module()
    assert mod.ALLOWED_VERDICTS == {
        "GO_WCL_B0_CS", "B0_ONLY_STRONG_BUT_METHOD_WEAK",
        "PIVOT_TO_PROVENANCE_EVALUATION_PAPER", "NO_PAPER_READY_EVIDENCE",
    }
