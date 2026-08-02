from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_amcf_lite_texbat.py"
    spec = importlib.util.spec_from_file_location("amcf_lite_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metrics_include_auc_and_first_sustained_delay():
    r = _runner()
    rows = []
    for time, score in [(30.0, 0.0), (30.5, 0.1), (100.0, 2.0), (100.5, 3.0), (101.0, 4.0)]:
        rows.append({"recording_id": "DS1", "decision_time_s": time, "tracked_prn_count": 4,
                     "score": score, "model": "complex adaptive K5", "selected_tap_histogram_json": "{}"})
    thresholds = {"complex adaptive K5": {"q99": 1.0, "q995": 1.5}}
    got = r.metrics({"DS1": rows}, thresholds)
    q99 = next(x for x in got if x["operating_point"] == "q99")
    assert q99["roc_auc"] == 1.0 and q99["pr_auc"] == 1.0
    assert q99["first_sustained_alarm_delay_s"] == 0.0
    assert q99["persistent_alarm_ratio"] == q99["persistent_detection_rate"]


def test_tap_histogram_expands_epoch_json_by_scenario_model_and_tap():
    r = _runner()
    rows = {"DS1": [
        {"model": "complex adaptive K5", "selected_tap_histogram_json": json.dumps({"3": 2, "4": 2, "6": 1})},
        {"model": "complex adaptive K5", "selected_tap_histogram_json": json.dumps({"3": 1, "7": 3})},
    ]}
    got = r.tap_histogram(rows)
    by_tap = {x["tap_index"]: x["query_count"] for x in got}
    assert by_tap == {3: 3, 4: 2, 6: 1, 7: 3}
    assert all(x["scenario"] == "DS1" and x["tap_name"] for x in got)
