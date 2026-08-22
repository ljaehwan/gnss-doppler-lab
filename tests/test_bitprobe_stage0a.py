from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from gnss_doppler_lab.bitprobe_stage0a import (
    ARTIFACT_REL, AccessGuard, BindingError, compact_manifest,
    evaluate_synthetic, load_preregistration, normalized_similarity,
    phase_aligned_distance,
)

def prereg() -> dict:
    repo = Path(__file__).resolve().parents[1]
    return load_preregistration(repo / ARTIFACT_REL)

def test_complex_similarity_is_global_phase_and_gain_invariant() -> None:
    rng = np.random.default_rng(8)
    left = rng.normal(size=99) + 1j * rng.normal(size=99)
    transformed = 2.5 * np.exp(1j * 0.72) * left
    assert normalized_similarity(left, transformed) == pytest.approx(1.0)
    assert phase_aligned_distance(left, transformed) == pytest.approx(0.0, abs=1e-12)

def test_forbidden_guard_rejects_before_any_operation() -> None:
    guard = AccessGuard.create(["/tmp/cleanStatic.bin"])
    with pytest.raises(BindingError):
        guard.stat(Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds3.bin"))
    assert guard.audit()["forbidden_attack_inputs"] == {
        "stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0
    }

def test_synthetic_controls_are_deterministic_and_honest() -> None:
    first_rows, first, first_confound = evaluate_synthetic(prereg())
    second_rows, second, second_confound = evaluate_synthetic(prereg())
    assert first_rows == second_rows
    assert first == second
    assert first_confound == second_confound
    assert first["common_vs_separate_auc"] >= 0.80
    assert first["weak_levels_same_direction"] >= 2
    assert first["null_false_positive_rate"] <= 0.05
    assert first_confound["receiver_linear_not_misclassified"] is True
    assert first_confound["receiver_nonlinear_indistinguishable_from_transmitter_common_nonlinearity"] is True
    assert first_confound["source_localization_available"] is False

def test_manifest_detects_compact_file_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "one.json").write_text(json.dumps({"a": 1}) + "\n")
    frozen = compact_manifest(artifact)
    (artifact / "one.json").write_text(json.dumps({"a": 2}) + "\n")
    assert frozen != compact_manifest(artifact)
