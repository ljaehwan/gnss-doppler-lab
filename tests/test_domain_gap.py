import csv
import json

import h5py
import numpy as np

from gnss_doppler_lab.domain_gap import (
    assign_gate_status,
    compare_feature_distributions,
    deterministic_group_cap,
    domain_classifier_audit,
    select_rows,
    worst_gate_status,
)
from gnss_doppler_lab.tracking_feature_windows import export_receiver_run_tracking_feature_csv


def _row(value, *, source, prn, index):
    return {
        "f1": str(value),
        "f2": str(value * 0.5),
        "scenario_name": source,
        "domain_source": source,
        "paired_group_id": source,
        "prn": prn,
        "window_mid_s": str(index * 0.5),
        "window_index": str(index),
    }


def test_select_rows_matches_exact_union_without_duplicates():
    rows = [
        {"scenario": "steady", "state": "normal", "id": "1"},
        {"scenario": "recovery", "state": "post", "id": "2"},
        {"scenario": "spoof", "state": "normal", "id": "3"},
    ]
    selected = select_rows(rows, [
        {"scenario": "steady", "state": "normal"},
        {"scenario": "recovery", "state": "post"},
        {"state": "normal"},
    ])
    assert [row["id"] for row in selected] == ["1", "2", "3"]


def test_distribution_metrics_and_gate_status_are_explicit():
    simulation = [_row(value, source="steady", prn="G01", index=index) for index, value in enumerate([0, 1, 2, 3])]
    real = [_row(value, source="clean", prn="G02", index=index) for index, value in enumerate([10, 11, 12, 13])]
    metrics, summary = compare_feature_distributions(simulation, real, ["f1", "f2"])
    assert len(metrics) == 2
    assert summary["median_ks_statistic"] == 1.0
    assert summary["median_robust_median_shift"] > 5.0

    thresholds = {
        "pass": {"domain_auc_max": 0.7, "median_ks_max": 0.2, "median_robust_shift_max": 0.75},
        "conditional": {"domain_auc_max": 0.85, "median_ks_max": 0.35, "median_robust_shift_max": 1.5},
    }
    status, reasons = assign_gate_status(summary, {"pooled_separability_auc": 1.0}, thresholds)
    assert status == "stop"
    assert len(reasons) == 3
    assert worst_gate_status(["pass", "stop", "conditional"]) == "stop"


def test_group_cap_and_grouped_domain_classifier_are_deterministic():
    rows = [_row(index, source="source", prn="G01", index=index) for index in range(10)]
    retained = deterministic_group_cap(rows, max_rows_per_group=4, group_columns=("scenario_name", "prn"))
    assert [row["window_index"] for row in retained] == ["0", "3", "6", "9"]

    simulation = []
    real = []
    for group_index in range(6):
        for index in range(8):
            simulation.append(_row(-3.0 + index * 0.01, source=f"sim{group_index}", prn=f"G{group_index + 1:02d}", index=index))
            real.append(_row(3.0 + index * 0.01, source=f"real{group_index}", prn=f"G{group_index + 20:02d}", index=index))
    result = domain_classifier_audit(
        simulation,
        real,
        ["f1", "f2"],
        n_splits=3,
        max_rows_per_group=5,
        random_state=17,
    )
    assert result["class_groups"] == {"simulation": 6, "real": 6}
    assert result["sampled_rows"] == 60
    assert result["pooled_separability_auc"] == 1.0
    assert all(fold["test_groups"] == 4 for fold in result["folds"])


def test_three_tap_export_can_read_epl_from_nine_tap_receiver_run(tmp_path):
    run = tmp_path / "receiver"
    raw = run / "raw"
    raw.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "source": {"sample_rate_hz": 1000},
        "tracking": {"raw_directory": "raw", "tap_count": 9},
    }))
    count = 200
    indices = np.arange(count, dtype=np.float64)
    with h5py.File(raw / "epl_tracking_ch_0.mat", "w") as handle:
        handle["PRN"] = np.full(count, 3.0)
        handle["PRN_start_sample_count"] = indices
        handle["abs_E"] = 2.0 + 0.01 * indices
        handle["abs_P"] = 4.0 + 0.01 * indices
        handle["abs_L"] = 1.5 + 0.01 * indices
        handle["carrier_doppler_hz"] = 100.0 + indices
        handle["CN0_SNV_dB_Hz"] = 40.0 + 0.01 * indices
        handle["Prompt_I"] = 3.0 + 0.01 * indices
        handle["Prompt_Q"] = 1.0 + 0.005 * indices
        handle["code_error_chips"] = 0.01 * np.sin(indices)
        handle["code_freq_chips"] = np.full(count, 1.023e6)
    output = tmp_path / "features.csv"
    export_receiver_run_tracking_feature_csv(
        run,
        output_path=output,
        tap_count=3,
        window_s=0.05,
        stride_s=0.05,
        min_epochs=4,
    )
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert rows[0]["prn"] == "G03"
    assert "near_sym_mean" in rows[0]
