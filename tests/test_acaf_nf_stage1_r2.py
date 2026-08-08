from pathlib import Path

from gnss_doppler_lab.acaf_nf_stage1_r2 import R1_REQUIRED, SCENARIOS


def test_r2_contract_names_are_complete():
    assert {"cleanStatic", "ds3", "ds4", "ds7", "ds8"} == set(SCENARIOS)
    assert "go_no_go.json" in R1_REQUIRED
    assert "per_window_scores.csv" in R1_REQUIRED
    assert Path("artifacts/acaf_nf_stage1_r2_full_normal").name == "acaf_nf_stage1_r2_full_normal"
