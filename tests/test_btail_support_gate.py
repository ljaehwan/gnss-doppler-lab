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
