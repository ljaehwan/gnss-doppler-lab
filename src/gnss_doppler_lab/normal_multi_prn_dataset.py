"""Receiver-level multi-PRN dataset builder for normal-only morphology models.

This module converts the existing per-PRN E/P/L tracking-window dataset into two
coordinated tables:

* PRN node windows: one row per run / time bin / PRN.
* Receiver graph windows: one row per run / time bin aggregating tracked PRNs.

The goal is to support a single integrated anomaly model that jointly considers
E/P/L morphology, Doppler/code dynamics, and inter-PRN relational consistency.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .isolation_forest_baseline import DYNAMICS_FEATURE_COLUMNS, MORPHOLOGY_FEATURE_COLUMNS
from .tracking_feature_windows import TrackingWindowFeatureRecord

SCHEMA_VERSION = 1
NODE_SCHEMA = "gnss-doppler-lab.normal-prn-node-windows"
GRAPH_SCHEMA = "gnss-doppler-lab.normal-receiver-graph-windows"

NODE_FEATURE_COLUMNS = MORPHOLOGY_FEATURE_COLUMNS + DYNAMICS_FEATURE_COLUMNS
MORPHOLOGY_PRIMARY_COLUMNS = [
    "near_sym_mean",
    "near_sym_std",
    "sharp_narrow_mean",
    "sharp_narrow_std",
    "sharp_narrow_slope",
]
DYNAMICS_PRIMARY_COLUMNS = [
    "doppler_std",
    "doppler_slope",
    "cn0_std",
    "code_err_abs_mean",
    "code_err_std",
    "prompt_mag_cv",
]
NODE_COLUMNS = [
    "run_id",
    "source_fingerprint",
    "label",
    "window_bin_s",
    "window_start_s",
    "window_end_s",
    "window_mid_s",
    "prn",
    "channel",
    "segment_index",
    "window_index",
    "epoch_count",
    *NODE_FEATURE_COLUMNS,
]
GRAPH_COLUMNS = [
    "run_id",
    "label",
    "window_bin_s",
    "window_start_s_min",
    "window_end_s_max",
    "tracked_prn_count",
    "tracked_prns",
    "morph_l2_median",
    "morph_l2_top3_mean",
    "morph_l2_std_across_prn",
    "near_sym_std_across_prn",
    "sharp_narrow_std_across_prn",
    "doppler_slope_common_removed_std",
    "code_err_abs_common_removed_std",
    "prompt_mag_cv_std_across_prn",
    "fraction_prn_morph_above_median",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _temp(path: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    return Path(name)


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _read_feature_rows(dataset_path: Path) -> list[dict[str, str]]:
    expected = list(TrackingWindowFeatureRecord.__dataclass_fields__)
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            raise ValueError(f"tracking feature dataset schema mismatch: expected {expected}, got {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("tracking feature dataset has zero rows")
    for row in rows:
        for column in NODE_FEATURE_COLUMNS:
            try:
                value = float(row[column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"feature column {column!r} must be numeric") from exc
            if not np.isfinite(value):
                raise ValueError(f"feature column {column!r} must be finite")
    return rows


def _bin_time(mid_s: float, stride_s: float) -> float:
    if stride_s <= 0:
        raise ValueError("stride_s must be positive")
    return round(round(float(mid_s) / float(stride_s)) * float(stride_s), 10)


def _node_row(row: dict[str, str], *, stride_s: float) -> dict[str, object]:
    out = {column: row[column] for column in NODE_COLUMNS if column in row}
    out["window_bin_s"] = _format_float(_bin_time(float(row["window_mid_s"]), stride_s))
    for column in ("window_start_s", "window_end_s", "window_mid_s", *NODE_FEATURE_COLUMNS):
        out[column] = _format_float(float(row[column]))
    for column in ("channel", "segment_index", "window_index", "epoch_count"):
        out[column] = int(row[column])
    return out


def _l2(values: dict[str, float], columns: Sequence[str]) -> float:
    return float(np.linalg.norm([values[column] for column in columns]))


def _std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.std(array, ddof=0)) if array.size else 0.0


def _topk_mean(values: Sequence[float], k: int = 3) -> float:
    if not values:
        return 0.0
    top = sorted(values, reverse=True)[: min(k, len(values))]
    return float(np.mean(top))


def _graph_row(group: list[dict[str, object]]) -> dict[str, object]:
    if not group:
        raise ValueError("graph group must not be empty")
    run_ids = {str(row["run_id"]) for row in group}
    labels = {str(row["label"]) for row in group}
    bins = {str(row["window_bin_s"]) for row in group}
    if len(run_ids) != 1 or len(labels) != 1 or len(bins) != 1:
        raise ValueError("graph group must contain exactly one run_id, label, and window_bin_s")

    numeric = [{column: float(row[column]) for column in NODE_FEATURE_COLUMNS} for row in group]
    morph_l2 = [_l2(row, MORPHOLOGY_PRIMARY_COLUMNS) for row in numeric]
    threshold = float(np.median(morph_l2)) if morph_l2 else 0.0
    prns = sorted({str(row["prn"]) for row in group})
    doppler_slopes = [row["doppler_slope"] for row in numeric]
    code_errors = [row["code_err_abs_mean"] for row in numeric]

    return {
        "run_id": next(iter(run_ids)),
        "label": next(iter(labels)),
        "window_bin_s": next(iter(bins)),
        "window_start_s_min": _format_float(min(float(row["window_start_s"]) for row in group)),
        "window_end_s_max": _format_float(max(float(row["window_end_s"]) for row in group)),
        "tracked_prn_count": len(prns),
        "tracked_prns": " ".join(prns),
        "morph_l2_median": _format_float(float(np.median(morph_l2))),
        "morph_l2_top3_mean": _format_float(_topk_mean(morph_l2, 3)),
        "morph_l2_std_across_prn": _format_float(_std(morph_l2)),
        "near_sym_std_across_prn": _format_float(_std(row["near_sym_mean"] for row in numeric)),
        "sharp_narrow_std_across_prn": _format_float(_std(row["sharp_narrow_mean"] for row in numeric)),
        "doppler_slope_common_removed_std": _format_float(_std(v - float(np.median(doppler_slopes)) for v in doppler_slopes)),
        "code_err_abs_common_removed_std": _format_float(_std(v - float(np.median(code_errors)) for v in code_errors)),
        "prompt_mag_cv_std_across_prn": _format_float(_std(row["prompt_mag_cv"] for row in numeric)),
        "fraction_prn_morph_above_median": _format_float(float(np.mean([value > threshold for value in morph_l2]))),
    }


def export_normal_multi_prn_dataset(
    tracking_feature_dataset_path: str | Path,
    *,
    output_dir: str | Path,
    stride_s: float = 0.5,
    min_prns_per_graph: int = 2,
) -> tuple[Path, Path, Path]:
    """Export node and receiver-graph windows from the current E/P/L dataset.

    Returns ``(node_csv, graph_csv, manifest_json)``.
    """
    if min_prns_per_graph < 1:
        raise ValueError("min_prns_per_graph must be at least 1")
    dataset_path = Path(tracking_feature_dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"tracking feature dataset does not exist: {dataset_path}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    node_csv = out / "normal_prn_node_windows.csv"
    graph_csv = out / "normal_receiver_graph_windows.csv"
    manifest_json = out / "manifest.json"
    input_sha = _sha(dataset_path)

    rows = _read_feature_rows(dataset_path)
    node_rows = [_node_row(row, stride_s=stride_s) for row in rows]
    node_rows.sort(key=lambda r: (str(r["run_id"]), float(r["window_bin_s"]), str(r["prn"]), int(r["segment_index"]), int(r["window_index"])))

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in node_rows:
        grouped.setdefault((str(row["run_id"]), str(row["label"]), str(row["window_bin_s"])), []).append(row)
    graph_rows = [_graph_row(group) for _, group in sorted(grouped.items()) if len({str(r["prn"]) for r in group}) >= min_prns_per_graph]
    if not graph_rows:
        raise ValueError("zero receiver graph rows generated; lower min_prns_per_graph or provide multi-PRN data")

    tn, tg, tm = _temp(node_csv), _temp(graph_csv), _temp(manifest_json)
    try:
        with tn.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=NODE_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(node_rows)
        with tg.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=GRAPH_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(graph_rows)
        if _sha(dataset_path) != input_sha:
            raise ValueError("tracking feature dataset changed during export")
        manifest = {
            "schema": "gnss-doppler-lab.normal-multi-prn-morphology-dynamics-dataset",
            "schema_version": SCHEMA_VERSION,
            "inputs": {"tracking_feature_dataset_path": str(dataset_path), "sha256": input_sha},
            "parameters": {"stride_s": stride_s, "min_prns_per_graph": min_prns_per_graph},
            "node_table": {"schema": NODE_SCHEMA, "path": str(node_csv), "row_count": len(node_rows), "columns": NODE_COLUMNS, "feature_columns": NODE_FEATURE_COLUMNS, "sha256": _sha(tn)},
            "graph_table": {"schema": GRAPH_SCHEMA, "path": str(graph_csv), "row_count": len(graph_rows), "columns": GRAPH_COLUMNS, "sha256": _sha(tg)},
            "score_decomposition": "S_total = alpha*S_morph + beta*S_dopp + gamma*S_rel; this dataset supplies node features for S_morph/S_dopp and graph features for S_rel.",
        }
        tm.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tn, node_csv)
        os.replace(tg, graph_csv)
        os.replace(tm, manifest_json)
    finally:
        tn.unlink(missing_ok=True)
        tg.unlink(missing_ok=True)
        tm.unlink(missing_ok=True)
    return node_csv, graph_csv, manifest_json


TAP_META_COLUMNS = {
    "run_id", "source_fingerprint", "split", "label", "prn", "channel", "sample_rate_hz",
    "segment_index", "window_index", "window_start_s", "window_end_s", "window_mid_s",
    "window_bin_s", "epoch_count", "tap_count", "tap_layout",
}
TAP_NODE_BASE_COLUMNS = ["run_id", "source_fingerprint", "label", "window_bin_s", "window_start_s", "window_end_s", "window_mid_s", "prn", "channel", "segment_index", "window_index", "epoch_count", "tap_count", "tap_layout"]
TAP_GRAPH_BASE_COLUMNS = ["run_id", "label", "window_bin_s", "window_start_s_min", "window_end_s_max", "tracked_prn_count", "tracked_prns", "tap_count", "tap_layout"]
TAP_RELATION_NODE_COLUMNS = [
    "receiver_relative_morph_l2",
    "receiver_relative_doppler_code_l2",
    "morph_doppler_coupling",
    "receiver_relative_centroid_abs",
]
TAP_RELATION_GRAPH_COLUMNS = [
    "receiver_relative_morph_l2_median",
    "receiver_relative_morph_l2_top3_mean",
    "receiver_relative_morph_l2_std_across_prn",
    "receiver_relative_doppler_code_l2_top3_mean",
    "morph_doppler_coupling_top3_mean",
    "fraction_prn_receiver_relative_morph_above_median",
    "signed_centroid_consistency_abs_mean",
    "relation_contrast_score_seed",
    "relation_contrast_baseline_recent_median",
    "relation_contrast_baseline_recent_mad",
    "relation_contrast_delta_recent",
    "relation_contrast_delta_positive",
    "relation_contrast_delta_z_recent",
    "relation_contrast_temporal_score",
]

def _read_tap_feature_rows(dataset_path: Path, *, feature_mode: str = "all") -> tuple[list[dict[str, str]], list[str]]:
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle); rows = list(reader); fieldnames = reader.fieldnames or []
    if not rows: raise ValueError("tap feature dataset has zero rows")
    required = {"run_id", "label", "prn", "window_mid_s", "window_start_s", "window_end_s", "tap_count", "tap_layout"}
    missing = sorted(required - set(fieldnames))
    if missing: raise ValueError(f"tap feature dataset missing required columns: {missing}")
    if feature_mode not in {"all", "normalized_dmcpd"}:
        raise ValueError(f"unknown tap feature_mode={feature_mode!r}")
    def allowed(column: str) -> bool:
        if feature_mode == "all":
            return True
        # Avoid raw absolute tap magnitude/scale features. Keep dynamics, normalized
        # ratios/CV, and DMCPD/SQM-inspired morphology statistics.
        if column.startswith("tap_"):
            return column.endswith("_cv") or column.endswith("_rel_prompt_mean") or column.endswith("_rel_sum_mean")
        return (
            column.startswith("dmcpd_")
            or column in {
                "left_right_imbalance_mean", "left_right_imbalance_std",
                "peak_index_mean", "peak_index_std",
                "peak_width_mean", "peak_width_std",
                "peak_sharpness_mean", "peak_sharpness_std",
                "doppler_std", "doppler_slope", "cn0_std",
                "code_err_abs_mean", "code_err_std", "prompt_mag_cv",
            }
        )
    feature_columns: list[str] = []
    for column in fieldnames:
        if column in TAP_META_COLUMNS or not allowed(column): continue
        try:
            vals = [float(row[column]) for row in rows]
        except (TypeError, ValueError):
            continue
        if np.isfinite(np.asarray(vals, dtype=float)).all(): feature_columns.append(column)
    if not feature_columns: raise ValueError("tap feature dataset has no finite numeric model feature columns")
    tap_counts = {str(row["tap_count"]) for row in rows}; layouts = {str(row["tap_layout"]) for row in rows}
    if len(tap_counts) != 1 or len(layouts) != 1: raise ValueError("tap feature dataset must contain exactly one tap_count and tap_layout")
    if tap_counts != {"9"}: raise ValueError(f"this Method-A model dataset expects real 9-tap features, got tap_count={sorted(tap_counts)}")
    return rows, feature_columns

def _tap_node_row(row: dict[str, str], *, stride_s: float, feature_columns: Sequence[str]) -> dict[str, object]:
    out = {column: row[column] for column in TAP_NODE_BASE_COLUMNS if column in row}
    out["window_bin_s"] = _format_float(_bin_time(float(row["window_mid_s"]), stride_s))
    for column in ("window_start_s", "window_end_s", "window_mid_s", *feature_columns): out[column] = _format_float(float(row[column]))
    for column in ("channel", "segment_index", "window_index", "epoch_count", "tap_count"):
        if column in out: out[column] = int(float(out[column]))
    return out

def _tap_morphology_columns(feature_columns: Sequence[str]) -> list[str]:
    return [
        c for c in feature_columns
        if c.startswith("dmcpd_")
        or c.startswith("left_right_imbalance")
        or c.startswith("peak_index")
        or c.startswith("peak_width")
        or c.startswith("peak_sharpness")
        or (c.startswith("tap_") and (c.endswith("_rel_prompt_mean") or c.endswith("_rel_sum_mean")))
    ]


def _tap_doppler_code_columns(feature_columns: Sequence[str]) -> list[str]:
    return [
        c for c in feature_columns
        if c.startswith("doppler_")
        or c.startswith("code_err")
        or c in {"cn0_std", "prompt_mag_cv"}
    ]


def _augment_tap_relation_node_rows(group: list[dict[str, object]], feature_columns: Sequence[str]) -> list[dict[str, object]]:
    """Add receiver-relative PRN-local imbalance features for a single run/time PRN set.

    The added values preserve each PRN's local 9-tap morphology residual after
    removing receiver-common mode within the currently tracked PRN set. This is
    the feature layer used to contrast local peak imbalance against Doppler/code
    motion and cross-PRN simultaneity.
    """
    morph_columns = _tap_morphology_columns(feature_columns)
    doppler_code_columns = _tap_doppler_code_columns(feature_columns)
    medians = {
        column: float(np.median([float(row[column]) for row in group]))
        for column in set(morph_columns + doppler_code_columns)
    }
    augmented: list[dict[str, object]] = []
    for row in group:
        out = dict(row)
        morph_residual = np.asarray([float(row[c]) - medians[c] for c in morph_columns], dtype=float)
        doppler_residual = np.asarray([float(row[c]) - medians[c] for c in doppler_code_columns], dtype=float)
        morph_l2 = float(np.linalg.norm(morph_residual)) if morph_residual.size else 0.0
        doppler_l2 = float(np.linalg.norm(doppler_residual)) if doppler_residual.size else 0.0
        centroid_column = "dmcpd_centroid_shift_mean"
        centroid_abs = abs(float(row[centroid_column]) - medians.get(centroid_column, 0.0)) if centroid_column in row else 0.0
        out["receiver_relative_morph_l2"] = _format_float(morph_l2)
        out["receiver_relative_doppler_code_l2"] = _format_float(doppler_l2)
        out["morph_doppler_coupling"] = _format_float(morph_l2 * doppler_l2)
        out["receiver_relative_centroid_abs"] = _format_float(centroid_abs)
        augmented.append(out)
    return augmented


def _tap_graph_row(group: list[dict[str, object]], feature_columns: Sequence[str]) -> dict[str, object]:
    run_ids={str(r["run_id"]) for r in group}; labels={str(r["label"]) for r in group}; bins={str(r["window_bin_s"]) for r in group}; tap_counts={str(r["tap_count"]) for r in group}; layouts={str(r["tap_layout"]) for r in group}
    if len(run_ids)!=1 or len(labels)!=1 or len(bins)!=1 or len(tap_counts)!=1 or len(layouts)!=1: raise ValueError("tap graph group must contain exactly one run_id, label, bin, tap_count, and tap_layout")
    prns=sorted({str(r["prn"]) for r in group})
    out={"run_id":next(iter(run_ids)),"label":next(iter(labels)),"window_bin_s":next(iter(bins)),"window_start_s_min":_format_float(min(float(r["window_start_s"]) for r in group)),"window_end_s_max":_format_float(max(float(r["window_end_s"]) for r in group)),"tracked_prn_count":len(prns),"tracked_prns":" ".join(prns),"tap_count":int(float(next(iter(tap_counts)))),"tap_layout":next(iter(layouts))}
    for column in feature_columns:
        values=np.asarray([float(r[column]) for r in group], dtype=float)
        out[f"{column}_mean_across_prn"]=_format_float(float(np.mean(values)))
        out[f"{column}_std_across_prn"]=_format_float(float(np.std(values, ddof=0)))
    if all(column in group[0] for column in TAP_RELATION_NODE_COLUMNS):
        morph = [float(r["receiver_relative_morph_l2"]) for r in group]
        doppler = [float(r["receiver_relative_doppler_code_l2"]) for r in group]
        coupling = [float(r["morph_doppler_coupling"]) for r in group]
        centroid_signed = [float(r.get("dmcpd_centroid_shift_mean", 0.0)) for r in group]
        morph_median = float(np.median(morph)) if morph else 0.0
        morph_top3 = _topk_mean(morph, 3)
        coupling_top3 = _topk_mean(coupling, 3)
        out["receiver_relative_morph_l2_median"] = _format_float(morph_median)
        out["receiver_relative_morph_l2_top3_mean"] = _format_float(morph_top3)
        out["receiver_relative_morph_l2_std_across_prn"] = _format_float(_std(morph))
        out["receiver_relative_doppler_code_l2_top3_mean"] = _format_float(_topk_mean(doppler, 3))
        out["morph_doppler_coupling_top3_mean"] = _format_float(coupling_top3)
        out["fraction_prn_receiver_relative_morph_above_median"] = _format_float(float(np.mean([value > morph_median for value in morph])) if morph else 0.0)
        out["signed_centroid_consistency_abs_mean"] = _format_float(abs(float(np.mean(np.sign(centroid_signed)))) if centroid_signed else 0.0)
        out["relation_contrast_score_seed"] = _format_float(morph_top3 + 0.5 * coupling_top3 + float(out["fraction_prn_receiver_relative_morph_above_median"]))
    return out

def _augment_tap_temporal_graph_rows(
    graph_rows: list[dict[str, object]],
    *,
    baseline_lookback_s: float = 30.0,
    baseline_gap_s: float = 5.0,
) -> list[dict[str, object]]:
    """Add recent-baseline temporal deltas to receiver relation-contrast graph rows.

    DS4-style carry-off can be weak in absolute relation contrast because early
    receiver/artifact motion may dominate the absolute score. These features
    therefore compare S(t) against the same run's recent history, using the
    median over [t-lookback, t-gap] as a robust local baseline.
    """
    if baseline_lookback_s <= 0:
        raise ValueError("baseline_lookback_s must be positive")
    if baseline_gap_s < 0:
        raise ValueError("baseline_gap_s must be non-negative")
    if baseline_gap_s > baseline_lookback_s:
        raise ValueError("baseline_gap_s must be <= baseline_lookback_s")

    by_run: dict[str, list[dict[str, object]]] = {}
    for row in graph_rows:
        by_run.setdefault(str(row["run_id"]), []).append(row)

    augmented_by_identity: dict[tuple[str, str], dict[str, object]] = {}
    for run_id, rows in by_run.items():
        ordered = sorted(rows, key=lambda r: float(r["window_bin_s"]))
        for row in ordered:
            out = dict(row)
            t = float(row["window_bin_s"])
            score = float(row.get("relation_contrast_score_seed", 0.0))
            coupling = float(row.get("morph_doppler_coupling_top3_mean", 0.0))
            candidates = [
                prior for prior in ordered
                if t - baseline_lookback_s <= float(prior["window_bin_s"]) <= t - baseline_gap_s
            ]
            if candidates:
                baseline_scores = np.asarray([float(r.get("relation_contrast_score_seed", 0.0)) for r in candidates], dtype=float)
                baseline_couplings = np.asarray([float(r.get("morph_doppler_coupling_top3_mean", 0.0)) for r in candidates], dtype=float)
                baseline = float(np.median(baseline_scores))
                baseline_coupling = float(np.median(baseline_couplings))
                mad = float(np.median(np.abs(baseline_scores - baseline)))
            else:
                baseline = score
                baseline_coupling = coupling
                mad = 0.0
            delta = score - baseline
            positive_delta = max(delta, 0.0)
            coupling_delta = max(coupling - baseline_coupling, 0.0)
            robust_scale = 1.4826 * mad + 1e-6
            out["relation_contrast_baseline_recent_median"] = _format_float(baseline)
            out["relation_contrast_baseline_recent_mad"] = _format_float(mad)
            out["relation_contrast_delta_recent"] = _format_float(delta)
            out["relation_contrast_delta_positive"] = _format_float(positive_delta)
            out["relation_contrast_delta_z_recent"] = _format_float(delta / robust_scale if mad > 0.0 else (positive_delta / robust_scale if positive_delta > 0.0 else 0.0))
            out["relation_contrast_temporal_score"] = _format_float(positive_delta + 0.5 * coupling_delta)
            augmented_by_identity[(run_id, str(row["window_bin_s"]))] = out
    return [augmented_by_identity[(str(row["run_id"]), str(row["window_bin_s"]))] for row in graph_rows]


def export_tap_multi_prn_dataset(tap_feature_dataset_path: str | Path, *, output_dir: str | Path, stride_s: float = 0.5, min_prns_per_graph: int = 2, feature_mode: str = "all", temporal_baseline_lookback_s: float = 30.0, temporal_baseline_gap_s: float = 5.0) -> tuple[Path, Path, Path]:
    """Export Method-A 9-tap PRN-node and receiver-graph model CSVs."""
    if min_prns_per_graph < 1: raise ValueError("min_prns_per_graph must be at least 1")
    dataset_path=Path(tap_feature_dataset_path)
    if not dataset_path.exists(): raise ValueError(f"tap feature dataset does not exist: {dataset_path}")
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    node_csv=out/"normal_prn_node_windows.csv"; graph_csv=out/"normal_receiver_graph_windows.csv"; manifest_json=out/"manifest.json"
    input_sha=_sha(dataset_path); rows, feature_columns=_read_tap_feature_rows(dataset_path, feature_mode=feature_mode)
    base_node_rows=[_tap_node_row(row, stride_s=stride_s, feature_columns=feature_columns) for row in rows]
    base_node_rows.sort(key=lambda r:(str(r["run_id"]), float(r["window_bin_s"]), str(r["prn"]), int(r["segment_index"]), int(r["window_index"])))
    grouped: dict[tuple[str,str,str], list[dict[str,object]]] = {}
    for row in base_node_rows: grouped.setdefault((str(row["run_id"]), str(row["label"]), str(row["window_bin_s"])), []).append(row)
    augmented_groups=[_augment_tap_relation_node_rows(g, feature_columns) for _,g in sorted(grouped.items())]
    node_rows=[row for group in augmented_groups for row in group]
    graph_feature_input_columns=list(feature_columns)+TAP_RELATION_NODE_COLUMNS
    graph_rows=[_tap_graph_row(g, graph_feature_input_columns) for g in augmented_groups if len({str(r["prn"]) for r in g}) >= min_prns_per_graph]
    graph_rows=_augment_tap_temporal_graph_rows(graph_rows, baseline_lookback_s=temporal_baseline_lookback_s, baseline_gap_s=temporal_baseline_gap_s)
    if not graph_rows: raise ValueError("zero receiver graph rows generated; lower min_prns_per_graph or provide multi-PRN data")
    node_columns=TAP_NODE_BASE_COLUMNS+feature_columns+TAP_RELATION_NODE_COLUMNS
    graph_feature_columns=[c for c in graph_rows[0].keys() if c not in TAP_GRAPH_BASE_COLUMNS]
    graph_columns=TAP_GRAPH_BASE_COLUMNS+graph_feature_columns
    tn,tg,tm=_temp(node_csv),_temp(graph_csv),_temp(manifest_json)
    try:
        with tn.open("w", newline="", encoding="utf-8") as h: w=csv.DictWriter(h, fieldnames=node_columns, lineterminator="\n"); w.writeheader(); w.writerows(node_rows)
        with tg.open("w", newline="", encoding="utf-8") as h: w=csv.DictWriter(h, fieldnames=graph_columns, lineterminator="\n"); w.writeheader(); w.writerows(graph_rows)
        if _sha(dataset_path) != input_sha: raise ValueError("tap feature dataset changed during export")
        manifest={"schema":"gnss-doppler-lab.method-a-9tap-multi-prn-dataset","schema_version":1,"inputs":{"tap_feature_dataset_path":str(dataset_path),"sha256":input_sha},"parameters":{"stride_s":stride_s,"min_prns_per_graph":min_prns_per_graph,"feature_mode":feature_mode,"temporal_baseline_lookback_s":temporal_baseline_lookback_s,"temporal_baseline_gap_s":temporal_baseline_gap_s},"node_table":{"path":str(node_csv),"row_count":len(node_rows),"columns":node_columns,"feature_columns":feature_columns+TAP_RELATION_NODE_COLUMNS,"sha256":_sha(tn)},"graph_table":{"path":str(graph_csv),"row_count":len(graph_rows),"columns":graph_columns,"feature_columns":graph_feature_columns,"sha256":_sha(tg)},"tap_count":9,"tap_layout":rows[0]["tap_layout"],"note":"Built only from real validated 9-tap Method-A feature rows; no zero-filled stock GNSS-SDR placeholders accepted."}
        tm.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        os.replace(tn,node_csv); os.replace(tg,graph_csv); os.replace(tm,manifest_json)
    finally:
        for path in (tn,tg,tm):
            if path.exists(): path.unlink()
    return node_csv, graph_csv, manifest_json
