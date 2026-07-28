import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_btail_support_gate.py"
    spec = importlib.util.spec_from_file_location("eval_btail_support_gate_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scores(rows):
    frame = pd.DataFrame(rows, columns=["run_id", "prn", "window_mid_s", "prn_node_rmse"])
    frame["window_start_s"] = frame["window_mid_s"]
    frame["window_bin_s"] = (frame["window_mid_s"] * 2.0).round() / 2.0
    return frame


def test_binomial_tail_surprise_matches_exact_four_of_four_coin_tail():
    mod = _load_module()

    score = mod.binomial_tail_surprise(k=4, n=4, exceedance_probability=0.5)

    assert score == pytest.approx(-math.log(0.5**4))


def test_build_event_scores_counts_current_prns_and_applies_causal_ewma():
    mod = _load_module()
    prn = _scores([
        ("r1", "G01", 0.49, 0.2),
        ("r1", "G02", 0.51, 0.8),
        ("r1", "G01", 1.01, 0.9),
        ("r1", "G02", 1.02, 0.95),
    ])
    thresholds = {"q50": 0.5, "q70": 0.7, "q80": 0.8}

    out = mod.build_event_scores(prn, thresholds, alpha=0.75)

    assert out["tracked_prn_count"].tolist() == [2, 2]
    assert out["k_q50"].tolist() == [1, 2]
    raw = out["btail_max_507080"].tolist()
    assert out["btail_max_507080_ewma075"].tolist() == pytest.approx([
        0.25 * raw[0], 0.75 * (0.25 * raw[0]) + 0.25 * raw[1]
    ])


def test_clean_only_calibration_produces_node_thresholds_and_q99_event_threshold():
    mod = _load_module()
    clean_static = _scores([
        ("static-a", "G01", 0.5, 0.1), ("static-a", "G02", 0.5, 0.2),
        ("static-a", "G01", 1.0, 0.3), ("static-a", "G02", 1.0, 0.4),
    ])
    clean_dynamic = _scores([
        ("dynamic-a", "G01", 0.5, 0.2), ("dynamic-a", "G02", 0.5, 0.3),
        ("dynamic-a", "G01", 1.0, 0.4), ("dynamic-a", "G02", 1.0, 0.5),
    ])

    calibration = mod.calibrate_clean_gate(clean_static, clean_dynamic)

    combined = pd.concat([clean_static, clean_dynamic], ignore_index=True)
    assert calibration.node_thresholds["q50"] == pytest.approx(combined.prn_node_rmse.quantile(0.50))
    assert calibration.node_thresholds["q70"] == pytest.approx(combined.prn_node_rmse.quantile(0.70))
    assert calibration.node_thresholds["q80"] == pytest.approx(combined.prn_node_rmse.quantile(0.80))
    expected_q99 = pd.concat(
        [calibration.clean_static_events, calibration.clean_dynamic_events], ignore_index=True
    )[mod.FINAL_SCORE].quantile(0.99)
    assert calibration.event_q99_threshold == pytest.approx(expected_q99)


@pytest.mark.parametrize("missing", ["run_id", "prn", "window_bin_s"])
def test_build_event_scores_requires_frozen_grouping_contract(missing):
    mod = _load_module()
    prn = _scores([("r1", "G01", 0.5, 0.2)]).drop(columns=[missing])

    with pytest.raises(ValueError, match=missing):
        mod.build_event_scores(prn, {"q50": 0.5, "q70": 0.7, "q80": 0.8})


@pytest.mark.parametrize("column", ["window_bin_s", "window_start_s", "window_mid_s", "prn_node_rmse"])
def test_build_event_scores_rejects_non_finite_numeric_inputs(column):
    mod = _load_module()
    prn = _scores([("r1", "G01", 0.5, 0.2)])
    prn.loc[0, column] = float("nan")

    with pytest.raises(ValueError, match=f"non-finite {column}"):
        mod.build_event_scores(prn, {"q50": 0.5, "q70": 0.7, "q80": 0.8})


@pytest.mark.parametrize("column", ["run_id", "prn"])
def test_build_event_scores_rejects_null_identifiers(column):
    mod = _load_module()
    prn = _scores([("r1", "G01", 0.5, 0.2)])
    prn.loc[0, column] = None

    with pytest.raises(ValueError, match=f"null or empty {column}"):
        mod.build_event_scores(prn, {"q50": 0.5, "q70": 0.7, "q80": 0.8})


def test_build_event_scores_rejects_duplicate_run_event_prn():
    mod = _load_module()
    prn = _scores([
        ("r1", "G01", 0.49, 0.2),
        ("r1", "G01", 0.51, 0.8),
    ])

    with pytest.raises(ValueError, match="duplicate.*run_id.*window_bin_s.*prn"):
        mod.build_event_scores(prn, {"q50": 0.5, "q70": 0.7, "q80": 0.8})


def test_build_event_scores_keeps_runs_separate_and_resets_ewma():
    mod = _load_module()
    thresholds = {"q50": 0.5, "q70": 0.7, "q80": 0.8}
    prn = _scores([
        ("r1", "G01", 0.5, 0.9),
        ("r1", "G01", 1.0, 0.1),
        ("r2", "G01", 0.5, 0.1),
        ("r2", "G01", 1.0, 0.9),
    ])

    out = mod.build_event_scores(prn, thresholds)

    assert list(zip(out.run_id, out.window_mid_s)) == [
        ("r1", 0.5), ("r1", 1.0), ("r2", 0.5), ("r2", 1.0)
    ]
    r2 = out[out.run_id == "r2"].reset_index(drop=True)
    assert r2.loc[0, mod.FINAL_SCORE] == pytest.approx(0.25 * r2.loc[0, "btail_max_507080"])


def test_detector_rejects_alpha_that_disagrees_with_ewma075_name():
    mod = _load_module()
    prn = _scores([("r1", "G01", 0.5, 0.2)])

    with pytest.raises(ValueError, match="0.75"):
        mod.build_event_scores(prn, {"q50": 0.5, "q70": 0.7, "q80": 0.8}, alpha=0.5)


@pytest.mark.parametrize("times", [[100.0, 101.0], [98.0, 99.0]])
def test_evaluate_scenario_requires_both_pre_and_post_windows(times):
    mod = _load_module()
    events = pd.DataFrame({
        "window_start_s": times,
        "window_mid_s": times,
        mod.FINAL_SCORE: [0.0, 1.0],
    })

    with pytest.raises(ValueError, match="pre.*post|post.*pre"):
        mod.evaluate_scenario(events, threshold=0.5, onset_s=100.0)


def test_ds7_ds8_phase_metrics_preserve_overall_onset_and_report_each_phase():
    mod = _load_module()
    events = pd.DataFrame({
        "window_start_s": [99.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0],
        "window_mid_s": [99.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0],
        mod.FINAL_SCORE: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    })

    result = mod.evaluate_scenario(
        events, threshold=0.5, onset_s=110.0, phase_windows=mod.DS7_DS8_PHASE_WINDOWS
    )

    assert result["onset_s"] == 110.0
    assert result["first_detection_s"] == 120.0
    assert result["phases"]["110_130"]["detection_rate"] == pytest.approx(0.5)
    assert result["phases"]["110_130"]["first_delay_s"] == 10.0
    assert result["phases"]["130_150"]["detection_rate"] == pytest.approx(0.5)
    assert result["phases"]["130_150"]["first_delay_s"] == 10.0
    assert result["phases"]["150_end"]["detection_rate"] == pytest.approx(0.5)
    assert result["phases"]["150_end"]["first_delay_s"] == 10.0


def test_frozen_champion_checkpoint_hash_and_feature_contract():
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "artifacts" / "ai_morph_gru_cleanStatic_q70_frame" / "prn_local_gru_predictor.pt"
    expected_sha256 = "f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b"
    expected_features = [
        "tap_E4_rel_prompt_mean",
        "tap_E3_rel_prompt_mean",
        "tap_E2_rel_prompt_mean",
        "tap_E_rel_prompt_mean",
        "tap_P_rel_prompt_mean",
        "tap_L_rel_prompt_mean",
        "tap_L2_rel_prompt_mean",
        "tap_L3_rel_prompt_mean",
        "tap_L4_rel_prompt_mean",
    ]

    assert checkpoint.is_file()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected_sha256
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["node_feature_columns"] == expected_features
    assert "model_state_dict" in payload
    assert "standardizer" in payload


def test_tracked_frozen_calibration_has_exact_champion_contract():
    root = Path(__file__).resolve().parents[1]
    doc = __import__("json").loads((root / "configs/detectors/texbat_btail_gate_v1.json").read_text())
    assert doc["node_thresholds"] == {"q50": 0.0914354398846626, "q70": 0.12956311106681812, "q80": 0.1630456149578094}
    assert doc["event_q99_threshold"] == 4.169877716047041
    assert doc["surprise_log_base"] == "e"
    assert doc["ewma_previous_state_weight"] == 0.75
    assert doc["checkpoint_sha256"] == "f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b"
    assert doc["source_score_sha256"] == {"cleanStatic": "9a6bc537bd8f1bc16a17257a5f7ae2e47f327c10e215c63d7ebd82ca0b80c36a", "cleanDynamic": "855c5ad2b2ea355136f027c49cc22e7234fab2147a6b812f832213b0c7ab082c"}


def test_frozen_calibration_validation_fails_closed_on_mismatch(tmp_path):
    mod = _load_module()
    good = {"schema": "gnss-doppler-lab.btail-gate-calibration.v1", "detector": mod.FINAL_SCORE, "node_thresholds": {"q50": .1, "q70": .2, "q80": .3}, "event_q99_threshold": 4.0, "surprise_log_base": "e", "ewma_previous_state_weight": .75, "checkpoint_sha256": "a" * 64, "source_score_sha256": {"cleanStatic": "b" * 64, "cleanDynamic": "c" * 64}}
    path = tmp_path / "cal.json"
    for field, bad in [("event_q99_threshold", None), ("surprise_log_base", "10"), ("ewma_previous_state_weight", .5), ("detector", "other")]:
        doc = dict(good); doc[field] = bad; path.write_text(__import__("json").dumps(doc))
        with pytest.raises(ValueError, match="calibration"):
            mod.load_frozen_calibration(path)


def test_generic_score_prefix_and_filename(tmp_path):
    mod = _load_module()
    scenario_dir = tmp_path / "os1"; scenario_dir.mkdir()
    expected = _scores([("oak", "G01", 1.0, .2)])
    expected.to_csv(scenario_dir / "oakbat_os1_scores.csv", index=False)
    got = mod.read_prn_scores(tmp_path, "os1", score_prefix="oakbat", score_filename="{prefix}_{scenario}_scores.csv")
    pd.testing.assert_frame_equal(got, expected)


@pytest.mark.parametrize("mutation", [
    ("node_thresholds", {"q50": 0.09143543988466261, "q70": 0.12956311106681812, "q80": 0.1630456149578094}),
    ("event_q99_threshold", 4.169877716047042),
    ("checkpoint_sha256", "a" * 64),
    ("source_score_sha256", {"cleanStatic": "b" * 64, "cleanDynamic": "c" * 64}),
])
def test_frozen_loader_rejects_well_formed_but_noncanonical_contract(tmp_path, mutation):
    mod = _load_module()
    root = Path(__file__).resolve().parents[1]
    doc = __import__("json").loads((root / "configs/detectors/texbat_btail_gate_v1.json").read_text())
    doc[mutation[0]] = mutation[1]
    path = tmp_path / "cal.json"; path.write_text(__import__("json").dumps(doc))
    with pytest.raises(ValueError, match="canonical|frozen.*contract"):
        mod.load_frozen_calibration(path)


def test_clean_only_evaluation_is_onset_free_and_reports_false_positive_metrics():
    mod = _load_module()
    events = pd.DataFrame({"window_start_s": [0.0, 0.5, 1.0], mod.FINAL_SCORE: [0.1, 0.8, 0.9]})
    result = mod.evaluate_clean_scenario(events, threshold=0.5)
    assert result == {"q99_threshold": 0.5, "windows": 3, "false_positive_flags": 2,
                      "false_positive_exceedance_rate": pytest.approx(2 / 3), "any_false_positive": True}
    assert "onset" not in __import__("json").dumps(result).lower()


def test_gate_timing_contract_documents_frozen_start_and_availability_offset():
    mod = _load_module()
    assert mod.TIMING_CONTRACT['score_time_field'] == 'window_start_s'
    assert mod.TIMING_CONTRACT['window_availability_offset_s'] == 1.0
