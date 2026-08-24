from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np

from gnss_doppler_lab.splitclock_observable_audit import panel_support, sign_unit_audit, verify_artifact
from gnss_doppler_lab.splitclock_stage0a import GALILEO_E1_WAVELENGTH_M


def synthetic_rows(sign: float = 1.0):
    rows = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for epoch in range(65):
        for prn in range(1, 7):
            rate = 1.0 + prn / 10
            rows.append({"epoch": start + timedelta(seconds=epoch), "receiver_epoch": epoch, "prn": prn,
                         "pseudorange_m": 20e6 + rate * epoch, "carrier_cycles": sign * rate * epoch / GALILEO_E1_WAVELENGTH_M,
                         "doppler_hz": -rate / GALILEO_E1_WAVELENGTH_M, "cn0_db_hz": 40.0, "carrier_indicator": "7"})
    return rows


def test_complete_panel_support():
    result = panel_support(synthetic_rows())
    assert result["status"] == "PASS"
    assert result["longest_continuous_m_ge_5_seconds"] == 65


def test_frozen_negative_sign_fails_for_native_rinex():
    result = sign_unit_audit(synthetic_rows(sign=1.0))
    assert result["rinex_native_sign_status"] == "PASS"
    assert result["frozen_contract_sign_status"] == "FAIL"


def test_frozen_sign_would_pass_for_opposite_phase_representation():
    result = sign_unit_audit(synthetic_rows(sign=-1.0))
    assert result["frozen_contract_sign_status"] == "PASS"


def test_artifact_byte_flip_detected(tmp_path):
    source = __import__("pathlib").Path("artifacts/splitclock_stage0a_clean_identifiability")
    if not (source / "artifact_manifest_sha256.json").exists():
        return
    target = tmp_path / "artifact"; shutil.copytree(source, target)
    path = target / "final_verdict.json"; path.write_bytes(path.read_bytes() + b" ")
    assert "final_verdict.json" in verify_artifact(target)
