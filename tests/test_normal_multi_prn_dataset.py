from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gnss_doppler_lab.tracking_feature_windows import TrackingWindowFeatureRecord
from gnss_doppler_lab.normal_multi_prn_dataset import (
    GRAPH_COLUMNS,
    NODE_COLUMNS,
    export_normal_multi_prn_dataset,
)


def _record(run_id: str, prn: str, window_index: int, *, label: str = "normal") -> dict[str, object]:
    values = {}
    for index, name in enumerate(TrackingWindowFeatureRecord.__dataclass_fields__):
        if name == "run_id":
            values[name] = run_id
        elif name == "source_fingerprint":
            values[name] = f"fingerprint-{run_id}"
        elif name == "label":
            values[name] = label
        elif name == "prn":
            values[name] = prn
        elif name == "channel":
            values[name] = int(prn[1:])
        elif name == "sample_rate_hz":
            values[name] = 10
        elif name == "segment_index":
            values[name] = 0
        elif name == "window_index":
            values[name] = window_index
        elif name == "epoch_count":
            values[name] = 5
        elif name == "window_start_s":
            values[name] = window_index * 0.5
        elif name == "window_end_s":
            values[name] = window_index * 0.5 + 1.0
        elif name == "window_mid_s":
            # Slight PRN-dependent offsets emulate real channel timing jitter; the
            # graph builder must bin these into the same receiver window.
            values[name] = window_index * 0.5 + 0.5 + int(prn[1:]) * 1e-4
        else:
            values[name] = float(index * 0.01 + window_index * 0.1 + int(prn[1:]) * 0.001)
    return TrackingWindowFeatureRecord(**values).to_row()


def _dataset(path: Path) -> None:
    fields = list(TrackingWindowFeatureRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run_id in ("normal-a", "normal-b"):
            for window_index in range(3):
                for prn in ("G03", "G08", "G14"):
                    writer.writerow(_record(run_id, prn, window_index))


def test_export_normal_multi_prn_dataset_writes_node_and_graph_tables(tmp_path: Path) -> None:
    dataset = tmp_path / "tracking_feature_windows.csv"
    _dataset(dataset)

    node_csv, graph_csv, manifest_json = export_normal_multi_prn_dataset(dataset, output_dir=tmp_path / "multi", stride_s=0.5, min_prns_per_graph=2)

    node_rows = list(csv.DictReader(node_csv.open(newline="", encoding="utf-8")))
    graph_rows = list(csv.DictReader(graph_csv.open(newline="", encoding="utf-8")))
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

    assert list(node_rows[0]) == NODE_COLUMNS
    assert list(graph_rows[0]) == GRAPH_COLUMNS
    assert len(node_rows) == 18
    assert len(graph_rows) == 6
    assert {row["tracked_prn_count"] for row in graph_rows} == {"3"}
    assert all(row["tracked_prns"] == "G03 G08 G14" for row in graph_rows)
    assert manifest["node_table"]["row_count"] == 18
    assert manifest["graph_table"]["row_count"] == 6
    assert "S_total" in manifest["score_decomposition"]

    first_node = node_csv.read_bytes()
    first_graph = graph_csv.read_bytes()
    first_manifest = manifest_json.read_bytes()
    export_normal_multi_prn_dataset(dataset, output_dir=tmp_path / "multi", stride_s=0.5, min_prns_per_graph=2)
    assert node_csv.read_bytes() == first_node
    assert graph_csv.read_bytes() == first_graph
    assert manifest_json.read_bytes() == first_manifest


def test_export_normal_multi_prn_dataset_rejects_bad_inputs(tmp_path: Path) -> None:
    dataset = tmp_path / "tracking_feature_windows.csv"
    _dataset(dataset)

    with pytest.raises(ValueError, match="stride_s"):
        export_normal_multi_prn_dataset(dataset, output_dir=tmp_path / "bad", stride_s=0.0)
    with pytest.raises(ValueError, match="min_prns"):
        export_normal_multi_prn_dataset(dataset, output_dir=tmp_path / "bad", min_prns_per_graph=0)
    with pytest.raises(ValueError, match="zero receiver graph"):
        export_normal_multi_prn_dataset(dataset, output_dir=tmp_path / "bad", min_prns_per_graph=4)

    rows = list(csv.DictReader(dataset.open(newline="", encoding="utf-8")))
    rows[0]["near_sym_mean"] = "nan"
    with dataset.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="finite"):
        export_normal_multi_prn_dataset(dataset, output_dir=tmp_path / "bad")


def test_export_tap_multi_prn_dataset_accepts_real_nine_tap_feature_csv(tmp_path):
    from gnss_doppler_lab.normal_multi_prn_dataset import export_tap_multi_prn_dataset
    import csv, json
    feature_csv = tmp_path / "tap9_features.csv"
    fields = [
        "run_id", "source_fingerprint", "label", "prn", "channel", "sample_rate_hz",
        "segment_index", "window_index", "window_start_s", "window_end_s", "window_mid_s",
        "epoch_count", "tap_count", "tap_layout", "left_right_imbalance_mean",
        "peak_width_mean", "doppler_std", "tap_E4_mean", "tap_P_mean", "tap_L4_mean",
    ]
    rows = []
    for wi in range(14):
        for prn_i, prn in enumerate(["G05", "G07", "G09"]):
            rows.append({
                "run_id": "run-a", "source_fingerprint": "abc", "label": "normal", "prn": prn,
                "channel": prn_i, "sample_rate_hz": 25000000, "segment_index": 0, "window_index": wi,
                "window_start_s": wi * 0.5, "window_end_s": wi * 0.5 + 1.0, "window_mid_s": wi * 0.5 + 0.5,
                "epoch_count": 1000, "tap_count": 9, "tap_layout": "E4,E3,E2,E,P,L,L2,L3,L4",
                "left_right_imbalance_mean": 0.01 * prn_i, "peak_width_mean": 1.5 + wi * 0.01,
                "doppler_std": 2.0 + prn_i, "tap_E4_mean": 1.0 + wi, "tap_P_mean": 10.0 + wi, "tap_L4_mean": 1.2 + wi,
            })
    with feature_csv.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    node_csv, graph_csv, manifest_json = export_tap_multi_prn_dataset(feature_csv, output_dir=tmp_path / "out", stride_s=0.5, min_prns_per_graph=2)
    assert node_csv.exists() and graph_csv.exists() and manifest_json.exists()
    manifest = json.loads(manifest_json.read_text())
    assert manifest["tap_count"] == 9
    assert "tap_E4_mean" in manifest["node_table"]["feature_columns"]
    assert "tap_P_mean_mean_across_prn" in manifest["graph_table"]["feature_columns"]


def test_export_tap_multi_prn_dataset_rejects_three_tap_claim(tmp_path):
    from gnss_doppler_lab.normal_multi_prn_dataset import export_tap_multi_prn_dataset
    import csv, pytest
    feature_csv = tmp_path / "tap3_features.csv"
    fields = ["run_id", "label", "prn", "window_start_s", "window_end_s", "window_mid_s", "tap_count", "tap_layout", "tap_E_mean"]
    with feature_csv.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader(); writer.writerow({"run_id":"r", "label":"normal", "prn":"G05", "window_start_s":0, "window_end_s":1, "window_mid_s":0.5, "tap_count":3, "tap_layout":"E,P,L", "tap_E_mean":1.0})
    with pytest.raises(ValueError, match="expects real 9-tap"):
        export_tap_multi_prn_dataset(feature_csv, output_dir=tmp_path / "out")



def test_export_tap_multi_prn_dataset_adds_receiver_relative_relation_contrast_features(tmp_path):
    from gnss_doppler_lab.normal_multi_prn_dataset import export_tap_multi_prn_dataset
    feature_csv = tmp_path / "tap9_relation_features.csv"
    fields = [
        "run_id", "source_fingerprint", "label", "prn", "channel", "sample_rate_hz",
        "segment_index", "window_index", "window_start_s", "window_end_s", "window_mid_s",
        "epoch_count", "tap_count", "tap_layout",
        "dmcpd_pair1_abs_asym_mean", "dmcpd_pair2_abs_asym_mean",
        "dmcpd_centroid_shift_mean", "dmcpd_width_variance_mean",
        "dmcpd_max_side_to_prompt_mean", "doppler_slope", "code_err_abs_mean",
        "cn0_std", "prompt_mag_cv",
    ]
    base = {
        "run_id": "run-a", "source_fingerprint": "abc", "label": "normal",
        "sample_rate_hz": 25000000, "segment_index": 0, "window_index": 0,
        "window_start_s": 0.0, "window_end_s": 1.0, "window_mid_s": 0.5,
        "epoch_count": 1000, "tap_count": 9, "tap_layout": "E4,E3,E2,E,P,L,L2,L3,L4",
        "dmcpd_pair1_abs_asym_mean": 0.02, "dmcpd_pair2_abs_asym_mean": 0.02,
        "dmcpd_centroid_shift_mean": 0.0, "dmcpd_width_variance_mean": 1.0,
        "dmcpd_max_side_to_prompt_mean": 0.25, "doppler_slope": 0.1,
        "code_err_abs_mean": 0.01, "cn0_std": 0.1, "prompt_mag_cv": 0.01,
    }
    rows = []
    for channel, prn in enumerate(["G03", "G08", "G14"]):
        row = dict(base, prn=prn, channel=channel)
        if prn == "G14":
            row.update({
                "dmcpd_pair1_abs_asym_mean": 0.60,
                "dmcpd_pair2_abs_asym_mean": 0.45,
                "dmcpd_centroid_shift_mean": 0.35,
                "dmcpd_width_variance_mean": 2.40,
                "dmcpd_max_side_to_prompt_mean": 0.70,
                "doppler_slope": 5.0,
                "code_err_abs_mean": 0.30,
            })
        rows.append(row)
    with feature_csv.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    node_csv, graph_csv, manifest_json = export_tap_multi_prn_dataset(
        feature_csv,
        output_dir=tmp_path / "out",
        stride_s=0.5,
        min_prns_per_graph=2,
        feature_mode="normalized_dmcpd",
    )

    node_rows = list(csv.DictReader(node_csv.open(newline="", encoding="utf-8")))
    graph_rows = list(csv.DictReader(graph_csv.open(newline="", encoding="utf-8")))
    manifest = json.loads(manifest_json.read_text())

    assert "receiver_relative_morph_l2" in node_rows[0]
    assert "receiver_relative_doppler_code_l2" in node_rows[0]
    assert "morph_doppler_coupling" in node_rows[0]
    g14 = next(row for row in node_rows if row["prn"] == "G14")
    g03 = next(row for row in node_rows if row["prn"] == "G03")
    assert float(g14["receiver_relative_morph_l2"]) > float(g03["receiver_relative_morph_l2"])
    assert float(g14["morph_doppler_coupling"]) > 0.0

    graph = graph_rows[0]
    assert float(graph["receiver_relative_morph_l2_top3_mean"]) > 0.0
    assert float(graph["morph_doppler_coupling_top3_mean"]) > 0.0
    assert float(graph["relation_contrast_score_seed"]) > 0.0
    assert "relation_contrast_score_seed" in manifest["graph_table"]["feature_columns"]




def test_export_tap_multi_prn_dataset_adds_recent_temporal_relation_delta(tmp_path):
    from gnss_doppler_lab.normal_multi_prn_dataset import export_tap_multi_prn_dataset

    feature_csv = tmp_path / "tap9_temporal_relation_features.csv"
    fields = [
        "run_id", "source_fingerprint", "label", "prn", "channel", "sample_rate_hz",
        "segment_index", "window_index", "window_start_s", "window_end_s", "window_mid_s",
        "epoch_count", "tap_count", "tap_layout",
        "dmcpd_pair1_abs_asym_mean", "dmcpd_pair2_abs_asym_mean",
        "dmcpd_centroid_shift_mean", "dmcpd_width_variance_mean",
        "dmcpd_max_side_to_prompt_mean", "doppler_slope", "code_err_abs_mean",
        "cn0_std", "prompt_mag_cv",
    ]
    rows = []
    for wi in range(24):
        for channel, prn in enumerate(["G03", "G08", "G14"]):
            row = {
                "run_id": "ds4-like", "source_fingerprint": "abc", "label": "attack",
                "prn": prn, "channel": channel, "sample_rate_hz": 25000000,
                "segment_index": 0, "window_index": wi,
                "window_start_s": wi * 0.5, "window_end_s": wi * 0.5 + 1.0,
                "window_mid_s": wi * 0.5 + 0.5,
                "epoch_count": 1000, "tap_count": 9,
                "tap_layout": "E4,E3,E2,E,P,L,L2,L3,L4",
                "dmcpd_pair1_abs_asym_mean": 0.02,
                "dmcpd_pair2_abs_asym_mean": 0.02,
                "dmcpd_centroid_shift_mean": 0.0,
                "dmcpd_width_variance_mean": 1.0,
                "dmcpd_max_side_to_prompt_mean": 0.25,
                "doppler_slope": 0.1,
                "code_err_abs_mean": 0.01,
                "cn0_std": 0.1,
                "prompt_mag_cv": 0.01,
            }
            if wi >= 20 and prn == "G14":
                row.update({
                    "dmcpd_pair1_abs_asym_mean": 0.55,
                    "dmcpd_pair2_abs_asym_mean": 0.45,
                    "dmcpd_centroid_shift_mean": 0.30,
                    "dmcpd_width_variance_mean": 2.20,
                    "dmcpd_max_side_to_prompt_mean": 0.65,
                    "doppler_slope": 4.5,
                    "code_err_abs_mean": 0.25,
                })
            rows.append(row)
    with feature_csv.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    _, graph_csv, manifest_json = export_tap_multi_prn_dataset(
        feature_csv,
        output_dir=tmp_path / "out",
        stride_s=0.5,
        min_prns_per_graph=2,
        feature_mode="normalized_dmcpd",
    )

    graph_rows = list(csv.DictReader(graph_csv.open(newline="", encoding="utf-8")))
    manifest = json.loads(manifest_json.read_text())
    early = next(row for row in graph_rows if float(row["window_bin_s"]) == 4.5)
    onset = next(row for row in graph_rows if float(row["window_bin_s"]) == 10.5)

    assert "relation_contrast_delta_recent" in onset
    assert float(early["relation_contrast_delta_positive"]) == 0.0
    assert float(onset["relation_contrast_baseline_recent_median"]) == pytest.approx(0.0)
    assert float(onset["relation_contrast_delta_recent"]) > 1.0
    assert float(onset["relation_contrast_temporal_score"]) >= float(onset["relation_contrast_delta_positive"])
    assert "relation_contrast_delta_recent" in manifest["graph_table"]["feature_columns"]
