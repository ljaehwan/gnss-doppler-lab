from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/cinder_stage0a_clean_emitter_identifiability"


def load_runner():
    spec = importlib.util.spec_from_file_location("cinder_runner", ROOT / "scripts/run_cinder_stage0a.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_clean_only_paths_and_no_prn_feature_input() -> None:
    module = load_runner()
    assert module.ATTACK_DATA_USED is False
    assert module.NEURAL_MODEL_USED is False
    assert set(module.DATASETS) == {"OAKBAT.cleanStatic", "TEXBAT.cleanStatic"}
    for value in module.DATASETS.values():
        assert "clean" in value["raw"].lower()
        assert "clean" in value["slug"].lower()
    contract = json.loads((ART / "feature_contract.json").read_text())
    assert "PRN" in contract["forbidden_feature_inputs"]
    assert contract["primary_family"] == "C4 fourth-order cyclic cumulant only"


def test_synthetic_emitter_relation_and_label_permutation() -> None:
    rng = np.random.default_rng(2026081901)
    emitters = rng.normal(size=(5, 12))
    features = np.vstack([emitters + 0.05 * rng.normal(size=emitters.shape) for _ in range(8)])
    prns = np.tile(np.arange(5), 8)
    blocks = np.repeat(np.arange(8), 5)
    labels = []; scores = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if blocks[i] == blocks[j]:
                continue
            labels.append(int(prns[i] == prns[j])); scores.append(-float(np.linalg.norm(features[i] - features[j])))
    assert roc_auc_score(labels, scores) > 0.99
    null = [roc_auc_score(rng.permutation(labels), scores) for _ in range(100)]
    assert 0.45 <= np.median(null) <= 0.55


def test_final_artifact_checksum_and_zero_bounds() -> None:
    manifest = json.loads((ART / "artifact_manifest_sha256.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((ART / name).read_bytes()).hexdigest() == expected
    summary = json.loads((ART / "cyclic_feature_summary.json").read_text())
    assert summary["resampling_audit_summary"] == {
        "audited_windows": 300,
        "maximum_fractional_source_error": 0.0,
        "maximum_read_extension_samples": 2,
        "status": "PASS",
        "total_out_of_bounds_queries": 0,
    }


def test_final_verdict_is_honest_negative() -> None:
    verdict = json.loads((ART / "final_verdict.json").read_text())
    assert verdict["verdict"] == "NO_GO_CINDER_CLEAN_IDENTIFIABILITY"
    assert verdict["attack_data_used"] is False
    assert verdict["neural_model_used"] is False
    assert all(values["median_auc"] < 0.70 for values in verdict["dataset_primary"].values())
