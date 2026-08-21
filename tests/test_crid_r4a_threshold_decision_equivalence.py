import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import gnss_doppler_lab.crid_threshold_equivalence as repair


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/crid_stage0_r4a_threshold_decision_equivalence_repair"


def test_threshold_numeric_and_decision_equivalence(monkeypatch):
    threshold = repair.COMMITTED["OAK"]["q99"]
    scores = np.array([threshold - 1.0, threshold + 1.0, threshold - 2.0, threshold - 3.0])
    monkeypatch.setitem(repair.COMMITTED["OAK"], "fpr", 0.25)
    numeric, alarms = repair.evaluate_threshold_equivalence("OAK", threshold + 1e-13, scores)
    assert numeric["status"] == "PASS"
    assert alarms["status"] == "PASS"
    assert alarms["alarm_vectors_byte_identical"]


def test_threshold_byte_flip_cannot_pass_numeric_gate():
    scores = np.array([-30.0, -20.0])
    numeric, _ = repair.evaluate_threshold_equivalence("OAK", repair.COMMITTED["OAK"]["q99"] + 1e-6, scores)
    assert numeric["status"] == "FAIL"


def test_clean_score_flip_is_detected_as_alarm_mismatch(monkeypatch):
    threshold = repair.COMMITTED["OAK"]["q99"]
    scores = np.array([threshold + 5e-14, threshold - 1.0])
    monkeypatch.setitem(repair.COMMITTED["OAK"], "fpr", 0.5)
    _, alarms = repair.evaluate_threshold_equivalence("OAK", threshold + 1e-13, scores)
    assert alarms["status"] == "FAIL"
    assert not alarms["alarm_vectors_byte_identical"]


def test_quantile_method_mutation_fails():
    threshold = repair.COMMITTED["TEX"]["q99"]
    numeric, _ = repair.evaluate_threshold_equivalence("TEX", threshold, np.array([threshold - 1.0]), quantile_method="linear")
    assert numeric["status"] == "FAIL"


def test_split_id_byte_flip_fails_identity(monkeypatch):
    candidate = {name: np.array([1, 2, 3], dtype=np.int64) for name in repair.EXPECTED_SPLIT["OAK"]}
    reference = {name: value.copy() for name, value in candidate.items()}
    monkeypatch.setitem(repair.EXPECTED_SPLIT, "OAK", {name: (1, 3, 3) for name in candidate})
    assert repair.split_identity("OAK", candidate, reference)["status"] == "PASS"
    reference["holdout"][1] = 99
    assert repair.split_identity("OAK", candidate, reference)["status"] == "FAIL"


def test_source_hash_byte_flip_fails_binding(tmp_path):
    path = tmp_path / "clean.bin"; path.write_bytes(b"clean")
    expected = repair.sha256_file(path); repair.require_file_binding(path, 5, expected)
    path.write_bytes(b"Clean")
    with pytest.raises(repair.ThresholdProvenanceError, match="SHA-256"):
        repair.require_file_binding(path, 5, expected)


def test_control_and_attack_paths_rejected_before_access(tmp_path):
    root = tmp_path / "clean"; exact = tmp_path / "receiver"
    assert repair.require_clean_path(root / "trace.bin", (root,)) == root / "trace.bin"
    with pytest.raises(repair.ThresholdProvenanceError, match="forbidden"):
        repair.require_clean_path(tmp_path / "controls" / "score.csv", (root,), (exact,))
    with pytest.raises(repair.ThresholdProvenanceError):
        repair.require_clean_path(Path("/data/texbat/ds3.bin"), (root,), (exact,))


def test_committed_artifact_contract():
    final = json.loads((ART / "final_verdict.json").read_text())
    source = json.loads((ART / "source_binding.json").read_text())
    numeric = json.loads((ART / "threshold_numeric_comparison.json").read_text())
    alarms = json.loads((ART / "holdout_alarm_equivalence.json").read_text())
    audit = json.loads((ART / "attack_and_control_access_audit.json").read_text())
    assert final["verdict"] == "THRESHOLD_DECISION_EQUIVALENCE_REPAIR_PASS"
    assert source["r4_artifact_unchanged"]["status"] == "PASS"
    assert numeric["status"] == alarms["status"] == "PASS"
    assert all(alarms["domains"][domain]["alarm_vectors_byte_identical"] for domain in ("OAK", "TEX"))
    assert audit["control_scores_read"] == audit["attack_bytes_read"] == 0


def test_artifact_byte_flip_fails_compact_verifier(tmp_path):
    copied = tmp_path / "artifact"; shutil.copytree(ART, copied)
    target = copied / "threshold_numeric_comparison.json"
    target.write_bytes(target.read_bytes() + b"X")
    result = subprocess.run(["python3", str(ROOT / "scripts/verify_crid_r4a_threshold_equivalence.py"), "--artifact", str(copied)], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert '"status": "FAIL"' in result.stdout
