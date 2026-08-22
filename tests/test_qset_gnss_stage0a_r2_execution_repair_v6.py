from __future__ import annotations

from gnss_doppler_lab import qset_stage0a_r2_execution_repair_v6 as R


def test_v6_patch_keeps_nine_trace_taps_and_separate_veml_loop_taps() -> None:
    text = R.V6_PATCH.read_text(encoding="utf-8")
    assert "d_n_correlator_taps = d_veml ? 13 : 9" in text
    assert "for (int32_t n = 0; n < 9; ++n)" in text
    assert "d_Very_Early = d_veml ? &d_correlator_outs[9] : nullptr" in text
    assert "d_Early = d_veml ? &d_correlator_outs[10] : &d_correlator_outs[3]" in text
    assert "d_Prompt = &d_correlator_outs[4]" in text
    assert "d_Late = d_veml ? &d_correlator_outs[11] : &d_correlator_outs[5]" in text
    assert "d_Very_Late = d_veml ? &d_correlator_outs[12] : nullptr" in text


def test_v6_patch_preserves_wide_and_narrow_receiver_loop_spacing() -> None:
    text = R.V6_PATCH.read_text(encoding="utf-8")
    assert "very_early_late_space_chips" in text
    assert "early_late_space_chips" in text
    assert "very_early_late_space_narrow_chips" in text
    assert "early_late_space_narrow_chips" in text
    assert "d_veml ? d_trk_parameters.early_late_space_chips : d_trk_parameters.tap_spacing_chips" in text
    assert "d_veml ? d_trk_parameters.early_late_space_narrow_chips : d_trk_parameters.tap_spacing_chips" in text


def test_v6_repair_declares_scientific_invariants() -> None:
    assert R.V6_PATCH_SHA256 == R.sha256_file(R.V6_PATCH)
    source = R.build_receiver_repaired_v6.__doc__ or ""
    assert "score" not in source.lower()
    assert R.V6_REPAIR_PATHS[-1].endswith("veml_preservation_repair.patch")
