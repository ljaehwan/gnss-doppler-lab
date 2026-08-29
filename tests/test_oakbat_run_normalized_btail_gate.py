import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_oakbat_run_normalized_btail_gate.py"
    )
    spec = importlib.util.spec_from_file_location("oakbat_run_normalized_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scores(*, attack=False):
    rows = []
    times = np.arange(0.0, 180.0, 0.5)
    for prn_index in range(8):
        for time_index, start in enumerate(times):
            value = 1.0 + 0.01 * ((time_index % 5) - 2)
            if attack and start >= 120.0:
                value += 3.0
            rows.append(
                {
                    "run_id": "run",
                    "prn": f"G{prn_index + 1:02d}",
                    "window_bin_s": start + 0.5,
                    "window_start_s": start,
                    "window_mid_s": start + 0.5,
                    "prn_node_rmse": value,
                }
            )
    return pd.DataFrame(rows)


def _write_scenario(mod, root, scenario, *, attack=False):
    directory = root / scenario
    directory.mkdir(parents=True)
    _scores(attack=attack).to_csv(
        directory / f"oakbat_{scenario}_prn_local_scores.csv", index=False
    )
    summary = {
        "checkpoint_provenance": {"checkpoint_sha256": mod.CHECKPOINT_SHA256}
    }
    (directory / f"oakbat_{scenario}_prn_local_onset_summary.json").write_text(
        json.dumps(summary)
    )


def test_run_normalization_is_causal_and_uses_scale_floor():
    mod = _load_module()
    scores = _scores()
    scores["prn_node_rmse"] = 1.0
    normalized, baselines = mod.normalize_run_scores(
        scores,
        warmup_end_s=60.0,
        minimum_baseline_rows=40,
        scale_floor=0.001,
    )
    assert normalized["window_start_s"].min() == 60.0
    assert "prn_node_rmse_raw" in normalized
    assert normalized["prn_node_rmse"].eq(0.0).all()
    assert baselines["eligible"].all()
    assert baselines["applied_scale"].eq(0.001).all()


def test_calibration_and_attack_evaluation_are_separate_and_reproducible(tmp_path):
    mod = _load_module()
    root = tmp_path / "scores"
    _write_scenario(mod, root, "cleanStatic")
    _write_scenario(mod, root, "cleanDynamic")
    _write_scenario(mod, root, "os2", attack=True)

    calibration_path = mod.calibrate(
        score_root=root,
        out_dir=tmp_path / "calibration",
        score_prefix="oakbat",
        warmup_end_s=60.0,
        minimum_baseline_rows=40,
        scale_floor=0.001,
        min_event_prns=8,
        event_quantile=0.99,
        ewma_previous_weight=0.75,
    )
    calibration = json.loads(calibration_path.read_text())
    assert set(calibration["clean_score_sources"]) == {"cleanStatic", "cleanDynamic"}
    assert "os2" not in calibration_path.read_text()

    summary_path = mod.evaluate(
        score_root=root,
        out_dir=tmp_path / "evaluation",
        calibration_json=calibration_path,
        scenarios=("os2",),
        onsets={"os2": 120.0},
        onset_buffer_s=10.0,
    )
    summary = json.loads(summary_path.read_text())
    metrics = summary["scenarios"]["os2"]["metrics"]
    assert metrics["pre_false_positive_rate"] == 0.0
    assert metrics["post_detection_rate"] == 1.0
    assert metrics["post_detection_wilson95"][0] < 1.0
    assert metrics["pre_false_positive_wilson95"][1] > 0.0
    assert metrics["first_operational_delay_s"] == pytest.approx(
        metrics["first_delay_s"] + 1.0
    )
    assert summary["scenarios"]["os2"]["support"]["minimum_required_prns"] == 8
    assert summary["scenarios"]["os2"]["physical_contrast"]["sustained_post_q80_prns"] == 8
    assert summary["attack_file_preattack_negative_control"]["false_positive_rate"] == 0.0


def test_evaluation_rejects_clean_scenarios_and_tampered_calibration(tmp_path):
    mod = _load_module()
    calibration = {
        "schema": mod.CALIBRATION_SCHEMA,
        "checkpoint_sha256": mod.CHECKPOINT_SHA256,
        "clean_score_sources": {"cleanStatic": {}, "cleanDynamic": {}},
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="forbids clean"):
        mod.evaluate(
            score_root=tmp_path,
            out_dir=tmp_path / "out",
            calibration_json=path,
            scenarios=("cleanStatic",),
            onsets={"cleanStatic": 120.0},
            onset_buffer_s=10.0,
        )
    calibration["clean_score_sources"]["os2"] = {}
    path.write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="exactly"):
        mod._load_calibration(path)
