import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_b0_m1_cross_layer.py"
    spec = importlib.util.spec_from_file_location("clif_inventory", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alignment_gate_rejects_relative_time_when_m1_raw_identity_is_missing():
    module = load_module()
    result = module.assess_pair(
        scenario="os2",
        b0={"exists": True, "sha_matches_manifest": True, "timestamp_grid_s": 0.5, "raw_iq_sha256": "b0raw"},
        m1={"exists": True, "sha_matches_manifest": True, "timestamp_grid_s": 0.5, "raw_iq_sha256": None},
    )
    assert result["relative_time_alignment_available"] is True
    assert result["same_recording_proven"] is False
    assert result["phase1_cross_layer_permitted"] is False
    assert any("raw_iq_sha256" in blocker for blocker in result["blockers"])


def test_alignment_gate_accepts_matching_time_and_raw_identity():
    module = load_module()
    result = module.assess_pair(
        scenario="cleanStatic",
        b0={"exists": True, "sha_matches_manifest": True, "timestamp_grid_s": 0.5, "raw_iq_sha256": "same"},
        m1={"exists": True, "sha_matches_manifest": True, "timestamp_grid_s": 0.5, "raw_iq_sha256": "same"},
    )
    assert result["phase1_cross_layer_permitted"] is True
    assert result["blockers"] == []
