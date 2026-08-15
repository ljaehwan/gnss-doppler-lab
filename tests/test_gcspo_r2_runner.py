import json

import h5py
import numpy as np

from gnss_doppler_lab.gcspo_r2_runner import (
    SCORE_FIELDS,
    _new_access_gate,
    _support_identity,
    _support_mismatch_events,
    _write_csv,
)
from gnss_doppler_lab.gcspo_artifacts import sha256_file


def test_support_identity_includes_complete_epoch_prn_window_support():
    row = {
        "phase": "transition", "window_start_s": 118.0, "availability_s": 119.0,
        "epoch_ids": (5900, 5901), "prns": (3, 7, 10, 13),
        "epoch_prn_support": ((5900, (3, 7, 10, 13)), (5901, (3, 7, 10, 13))),
    }
    same = dict(row)
    changed = dict(row, epoch_prn_support=((5900, (3, 7, 10, 13)), (5901, (3, 7, 10, 16))))
    assert _support_identity(row) == _support_identity(same)
    assert _support_identity(row) != _support_identity(changed)


def test_parse_score_rows_restores_authenticated_support():
    from gnss_doppler_lab.gcspo_r2_runner import _parse_score_rows

    row = {
        "scenario": "DS3", "phase": "transition", "method": "Full",
        "window_start_s": "118.0", "availability_s": "119.0", "score": "2.5",
        "phase_start_s": "118.9", "phase_end_s": "195.0", "label": "True",
        "prns_json": json.dumps([3, 7, 10, 13]),
        "epoch_ids_json": json.dumps([5900, 5901]),
        "epoch_prn_support_json": json.dumps([[5900, [3, 7, 10, 13]], [5901, [3, 7, 10, 13]]]),
    }
    parsed = _parse_score_rows([row])[0]
    assert parsed["prns"] == (3, 7, 10, 13)
    assert parsed["epoch_ids"] == (5900, 5901)
    assert parsed["epoch_prn_support"][1] == (5901, (3, 7, 10, 13))
    assert parsed["label"] is True


def test_support_mismatch_events_records_contracted_timing_and_support(tmp_path):
    scenario_dir = tmp_path / "phase_outputs/ds3-evaluation"
    full = {
        "scenario": "DS3", "family": "DS3", "phase": "transition", "method": "Full",
        "window_start_s": 118.0, "availability_s": 119.0, "score": 2.5,
        "threshold_q99": 1.0, "threshold_q995": 1.2, "alarm_q99": True,
        "alarm_q995": True, "tracked_n": 4, "prns_json": json.dumps([3, 7, 10, 13]),
        "epoch_ids_json": json.dumps([5900]),
        "epoch_prn_support_json": json.dumps([[5900, [3, 7, 10, 13]]]),
        "label": True, "phase_start_s": 118.0, "phase_end_s": 195.0,
    }
    _write_csv(scenario_dir / "per_epoch_scores.csv", [full], SCORE_FIELDS)
    nodes = [
        {"run_id": "DS3-transition", "phase": "transition", "window_start_s": 118.0, "window_end_s": 119.0,
         "prn": f"G{prn:02d}", "epoch_ids_json": json.dumps([5900])}
        for prn in (3, 7, 10, 16)
    ]
    _write_csv(tmp_path / "b0_recomputed/DS3/scheduled_node_windows.csv", nodes,
               ["run_id", "phase", "window_start_s", "window_end_s", "prn", "epoch_ids_json"])
    scored = [{"run_id": row["run_id"], "prn": row["prn"],
               "window_start_s": row["window_start_s"]} for row in nodes]
    _write_csv(tmp_path / "b0_recomputed/DS3/gcspo_r2_b0_DS3_prn_local_scores.csv", scored,
               ["run_id", "prn", "window_start_s"])
    event = _support_mismatch_events(tmp_path, "DS3")[0]
    assert event["cause"] == "EPOCH_PRN_SUPPORT_MISMATCH"
    assert event["absolute_sample_index"] == 2_950_000_000
    assert event["full_prn_set"] == [3, 7, 10, 13]
    assert event["b0_prn_set"] == [3, 7, 10, 16]


def test_new_access_gate_registers_preflight_hashed_tracking_file(tmp_path):
    tracking = tmp_path / "tracking.mat"
    with h5py.File(tracking, "w") as handle:
        handle.create_dataset("PRN", data=np.asarray([3, 7]))
    rows = [{"path": str(tracking), "sha256": sha256_file(tracking),
             "size_bytes": tracking.stat().st_size}]
    gate = _new_access_gate(tmp_path, "phase", rows)
    values = gate.read_h5(tracking, datasets=("PRN",), scenario="DS3",
                          phase="transition", purpose="test authenticated tracking read")
    assert values["PRN"].tolist() == [3, 7]
