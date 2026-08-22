from __future__ import annotations
import copy
from pathlib import Path
import pytest
from gnss_doppler_lab import qset_stage0a_r2a as Q

def test_variant_matrix_is_finite_and_frozen() -> None:
    assert list(Q.VARIANTS) == ["V0", "V1", "V2", "V3"]
    assert Q.VARIANTS["V1"]["in_acquisition"] == 12
    assert Q.VARIANTS["V2"]["pfa"] == "0.0001"
    assert Q.VARIANTS["V3"]["coherent_ms"] == 8

def test_config_changes_only_preregistered_acquisition_fields(tmp_path: Path) -> None:
    target = tmp_path / "receiver"; target.mkdir()
    text = Q.render_variant_config("V3", target)
    assert "Channels.in_acquisition=12" in text
    assert "Acquisition_1B.coherent_integration_time_ms=8" in text
    assert "Tracking_1B.pll_bw_hz=20.0" in text

def test_manifest_tamper_detection(tmp_path: Path) -> None:
    path = tmp_path / "evidence"; path.write_bytes(b"locked")
    manifest = Q.output_manifest(tmp_path); Q.verify_output_manifest(tmp_path, manifest)
    path.write_bytes(b"tamper")
    with pytest.raises(Q.AuditError): Q.verify_output_manifest(tmp_path, manifest)

def test_source_has_no_attack_score_or_raw_resolver() -> None:
    source = (Q.ROOT / "src/gnss_doppler_lab/qset_stage0a_r2a.py").read_text()
    for token in ("evaluate_attack", "binary_auc", "scenario_path(" , "md5_file("):
        assert token not in source

def test_preregistered_success_gate_is_not_relaxed() -> None:
    prereg = Q.make_preregistration()
    assert prereg["success_gate"]["m_ge_5_windows_minimum"] == 60
    assert prereg["success_gate"]["prn9_stable_windows_minimum"] == 60
    assert "minimum-five relaxation" in prereg["prohibited"]


def test_cadence_contract_matches_frozen_r2_tolerance() -> None:
    source = (Q.ROOT / "src/gnss_doppler_lab/qset_stage0a_r2a.py").read_text()
    assert "dt < 0.0035" in source and "dt > 0.0045" in source
    assert "!= 16000" not in source


def test_clean_decoder_manifest_keys_are_explicit() -> None:
    source = (Q.ROOT / "src/gnss_doppler_lab/qset_stage0a_r2a.py").read_text()
    assert 'scenario == "SS-1"' in source
    assert 'decoder_binding["output_sha256"]' in source


def test_final_verifier_binds_no_ss1_score() -> None:
    source = (Q.ROOT / "scripts/verify_qset_gnss_stage0a_r2a.py").read_text()
    assert 'ss1_spoofing_score_computed' in source
    assert 'RECEIVER_REPAIR_FAILED_CLEAN_REGRESSION' in source
