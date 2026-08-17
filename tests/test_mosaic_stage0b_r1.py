import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.mosaic_iq_injector import sampled_prn_replica
from gnss_doppler_lab.mosaic_iq_injector_int16 import decode_interleaved_int16, encode_interleaved_int16, inject_payload
from gnss_doppler_lab.mosaic_receiver_in_loop import (
    NavBitTimeline, ReceiverNcoEpoch, ReplicaState, StatefulReplica, assign_case_targets,
    estimate_authentic_amplitude, iter_stateful_replica_epochs, raised_cosine_envelope,
    realized_scer_db, spoof_amplitude, verify_sha256,
)
from gnss_doppler_lab.mosaic_residual_caf import h0_residual, residual_caf
from gnss_doppler_lab.mosaic_stage0b_metrics import (
    bic, caf_grids, classify_case, delta_bic_h1_support, preregistered_verdict, strong_resolvable,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mosaic_stage0b_r1_receiver_in_loop"


def test_int16_iq_round_trip_and_iq_ordering():
    iq = np.array([1 - 2j, -32768 + 32767j, 1234 + 5678j])
    payload, metrics = encode_interleaved_int16(iq)
    assert np.array_equal(decode_interleaved_int16(payload), iq)
    assert np.frombuffer(payload[:4], dtype="<i2").tolist() == [1, -2]
    assert metrics["clipped_sample_count"] == 0


def test_zero_amplitude_is_byte_identical_and_count_preserved():
    payload, _ = encode_interleaved_int16(np.array([5 + 6j, -7 + 8j]))
    output, metrics = inject_payload(payload, np.zeros(2, complex))
    assert output == payload and metrics["byte_identity"]
    assert len(output) == len(payload)


def test_int16_clipping_detection():
    _, metrics = encode_interleaved_int16(np.array([40000 + 0j, 0 - 50000j, 1 + 2j]))
    assert metrics["clipped_sample_count"] == 2
    assert metrics["clipped_component_count"] == 2


def test_per_prn_complex_ls_amplitude_and_scer_scaling():
    replica = np.exp(1j * np.arange(128) / 11)
    alpha = 2 - 3j
    clean = alpha * replica + 0.1 * sampled_prn_replica(2, 1_023_000, 128)
    estimated = estimate_authentic_amplitude(clean, replica)
    assert abs(estimated - alpha) < .02
    spoof = spoof_amplitude(estimated, -6)
    assert realized_scer_db(spoof, estimated) == pytest.approx(-6)


def test_code_and_carrier_phase_are_continuous_across_chunks():
    signs = np.r_[np.ones(53), -np.ones(47)]
    whole = StatefulReplica(3, 1_023_000, ReplicaState(1000, .2, .3))
    expected = whole.render(100, code_rate_chips_s=1_023_100, carrier_doppler_hz=120,
                            delta_f_hz=25, phase_offset_rad=.4, nav_signs=signs)
    chunked = StatefulReplica(3, 1_023_000, ReplicaState(1000, .2, .3))
    observed = np.r_[chunked.render(37, code_rate_chips_s=1_023_100, carrier_doppler_hz=120,
                                    delta_f_hz=25, phase_offset_rad=.4, nav_signs=signs[:37]),
                     chunked.render(63, code_rate_chips_s=1_023_100, carrier_doppler_hz=120,
                                    delta_f_hz=25, phase_offset_rad=.4, nav_signs=signs[37:])]
    assert np.allclose(observed, expected)
    assert chunked.state.absolute_sample_index == 1100


def test_nav_boundary_and_receiver_nco_epoch_continuity():
    mapping = [
        {"prn": "3", "corrected_raw_start_sample": "100", "corrected_raw_end_sample_exclusive": "110", "bit_value_pm1": "1"},
        {"prn": "3", "corrected_raw_start_sample": "110", "corrected_raw_end_sample_exclusive": "120", "bit_value_pm1": "-1"},
    ]
    nav = NavBitTimeline(mapping, prn=3)
    replica = StatefulReplica(3, 1_023_000, ReplicaState(100, 0, 0))
    epochs = [ReceiverNcoEpoch(100, 110, 1_023_000, 10), ReceiverNcoEpoch(110, 120, 1_023_001, 11)]
    rendered = list(iter_stateful_replica_epochs(replica, epochs, nav, delay_chips=0, delta_f_hz=0, phase_offset_rad=0))
    assert [start for start, _ in rendered] == [100, 110]
    assert replica.state.absolute_sample_index == 120
    assert np.all(nav.signs(100, 10) == 1) and np.all(nav.signs(110, 10) == -1)


def test_raised_cosine_has_no_step_at_registered_boundaries():
    t = np.array([0, 2, 3, 4, 9.999, 10, 11, 12])
    e = raised_cosine_envelope(t, 12)
    assert e[[0, 1, 3, 5, 7]].tolist() == [0, 0, 1, 1, 0]
    assert 0 < e[2] < 1 and 0 < e[6] < 1


def test_r0c_sample_bounds_and_int16_binding():
    common = json.loads((ART / "r0c_input_binding.json").read_text())["common_intervals"]
    formats = json.loads((ART / "sample_format_validation.json").read_text())["datasets"]
    for dataset, item in common.items():
        assert 0 <= item["common_raw_start_sample"] < item["common_raw_end_sample_exclusive"] <= formats[dataset]["complex_sample_count"]


def test_source_hash_mismatch_is_rejected(tmp_path):
    source = tmp_path / "source.bin"; source.write_bytes(b"abc")
    with pytest.raises(ValueError, match="source hash mismatch"):
        verify_sha256(source, "0" * 64)


def test_prn_assignment_is_deterministic_and_matches_rule():
    design = json.loads((ART / "frozen_injection_design.json").read_text())
    prns = {"OAKBAT.cleanStatic": [10, 11, 21, 24, 27], "TEXBAT.cleanStatic": [3, 13, 16, 19, 30]}
    first = assign_case_targets(design, prns); second = assign_case_targets(design, prns)
    assert first == second and len(first) == 72
    assert first[0]["target_prns"] == [3]
    tex_four = next(row for row in first if row["mode"].startswith("four") and row["dataset"].startswith("TEX"))
    assert tex_four["excluded_prn"] == 3 and len(tex_four["target_prns"]) == 4


def test_residual_caf_positive_delay_and_doppler_sign_convention():
    fs = 10_230_000
    authentic = sampled_prn_replica(4, fs, 10230)
    spoof = sampled_prn_replica(4, fs, 10230, code_phase_chips=.1, doppler_hz=25)
    residual, _ = h0_residual(authentic + .5 * spoof, authentic)
    result = residual_caf(residual, 4, fs, np.array([-.1, 0, .1]), np.array([-25, 0, 25]))
    assert result.peak_delay_chips == .1 and result.peak_doppler_hz == 25


def test_bic_applies_h1_complexity_correction():
    raw_improvement = 100 - 90
    score = delta_bic_h1_support(100, 90, 1000, h0_parameters=2, h1_parameters=6)
    assert raw_improvement > 0
    assert score < 1000 * np.log(100 / 90)
    assert bic(100, 1000, 6) > bic(100, 1000, 2)


def test_collapsed_control_and_strong_subset():
    assert classify_case(0, 0) == "COLLAPSED_SINGLE_SOURCE_CONTROL"
    assert classify_case(.1, 0) == "IDENTIFIABLE_SECOND_SOURCE"
    assert strong_resolvable(-6, .1, 0) and strong_resolvable(-3, 0, 25)
    assert not strong_resolvable(-10, .25, 50)


def test_preregistered_verdict_labels_receiver_vs_physics():
    assert preregistered_verdict({}) == "INCONCLUSIVE_RECEIVER_IN_LOOP"
    base = {"identity_receiver_replay": True, "zero_amplitude_byte_identity": True, "actual_int16_format": True,
        "realized_scer_median_error_db": 0, "oak_observability": 1, "tex_observability": 1,
        "delay_sign_accuracy": 1, "delay_median_absolute_error_chips": 0,
        "doppler_sign_accuracy": 1, "doppler_median_absolute_error_hz": 0,
        "four_prn_three_of_four_fraction": 1, "bic_control_separation": True,
        "target_over_nontarget": True, "not_total_iq_rms_shortcut": True, "prn_permutation_invariance": True}
    assert preregistered_verdict(base) == "GO_FOR_MOSAIC_NEURAL_STAGE1"
    base["delay_sign_accuracy"] = .5
    assert preregistered_verdict(base) == "NO_GO_MOSAIC_INJECTOR_PHYSICS"


def test_artifact_checksum_and_execution_status():
    spec = importlib.util.spec_from_file_location("r1_verify_test", ROOT / "scripts/verify_mosaic_stage0b_r1.py")
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    module.main()
    status = json.loads((ART / "execution_status.json").read_text())
    assert status["status"] == "READY_FOR_R1_EXECUTION"
    assert not status["injection_executed"] and not status["attack_data_accessed"] and not status["results_viewed"]
