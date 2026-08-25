#!/usr/bin/env python3
"""Audit simulation-v4 normal features against TEXBAT clean recordings only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab.domain_gap import (  # noqa: E402
    assign_gate_status,
    compare_feature_distributions,
    domain_classifier_audit,
    select_rows,
    worst_gate_status,
)
from gnss_doppler_lab.tracking_feature_windows import (  # noqa: E402
    export_receiver_run_tracking_feature_csv,
)

DEFAULT_CONFIG = Path("configs/experiments/simulation_v4_domain_gap_gate_v1.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + extras, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _in_intervals(time_s: np.ndarray, intervals: list[list[float | None]]) -> np.ndarray:
    selected = np.zeros(time_s.shape, dtype=bool)
    for start, end in intervals:
        interval = np.ones(time_s.shape, dtype=bool)
        if start is not None:
            interval &= time_s >= float(start)
        if end is not None:
            interval &= time_s < float(end)
        selected |= interval
    return selected


def _receiver_state_summary(
    receiver_dir: Path,
    intervals: list[list[float | None]] | None = None,
) -> dict[str, Any]:
    manifest_path = receiver_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_rate = float(manifest["source"]["sample_rate_hz"])
    raw_dir = receiver_dir / str(manifest.get("tracking", {}).get("raw_directory", "raw"))
    mat_paths = sorted(raw_dir.glob("epl_tracking_ch_*.mat"))
    if not mat_paths:
        raise ValueError(f"receiver has no tracking MAT files: {receiver_dir}")
    active_intervals = intervals or [[None, None]]
    locks: list[np.ndarray] = []
    cn0s: list[np.ndarray] = []
    times: list[np.ndarray] = []
    prns: set[str] = set()
    for path in mat_paths:
        with h5py.File(path, "r") as handle:
            raw_prn = np.asarray(handle["PRN"]).reshape(-1)
            time_s = np.asarray(handle["PRN_start_sample_count"]).reshape(-1).astype(np.float64) / sample_rate
            lock = np.asarray(handle["carrier_lock_test"]).reshape(-1).astype(np.float64)
            cn0 = np.asarray(handle["CN0_SNV_dB_Hz"]).reshape(-1).astype(np.float64)
        valid_prn = np.isfinite(raw_prn) & (raw_prn >= 1) & (raw_prn <= 32)
        selected = valid_prn & np.isfinite(time_s) & np.isfinite(lock) & np.isfinite(cn0)
        selected &= _in_intervals(time_s, active_intervals)
        if selected.any():
            locks.append(lock[selected])
            cn0s.append(cn0[selected])
            times.append(time_s[selected])
            prns.update(f"G{int(value):02d}" for value in raw_prn[selected])
    if not locks:
        raise ValueError(f"no valid receiver-state epochs: {receiver_dir}")
    all_locks = np.concatenate(locks)
    all_cn0 = np.concatenate(cn0s)
    all_times = np.concatenate(times)
    return {
        "receiver_run_id": receiver_dir.name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_dataset": manifest.get("source", {}).get("dataset"),
        "sample_rate_hz": int(sample_rate),
        "tracking_tap_count": int(manifest.get("tracking", {}).get("tap_count", 3)),
        "tap_spacing_chips": float(manifest.get("tracking", {}).get("tap_spacing_chips", float("nan"))),
        "intervals_s": active_intervals,
        "raw_mat_file_count": len(mat_paths),
        "epoch_count": int(all_locks.size),
        "tracked_prn_count": len(prns),
        "tracked_prns": sorted(prns),
        "first_time_s": float(np.min(all_times)),
        "last_time_s": float(np.max(all_times)),
        "carrier_lock_median": float(np.median(all_locks)),
        "carrier_lock_iqr": float(np.subtract(*np.quantile(all_locks, [0.75, 0.25]))),
        "carrier_lock_above_0_5_fraction": float(np.mean(all_locks > 0.5)),
        "cn0_db_hz_median": float(np.median(all_cn0)),
        "cn0_db_hz_iqr": float(np.subtract(*np.quantile(all_cn0, [0.75, 0.25]))),
    }


def _validate_boundary(config: dict[str, Any]) -> None:
    allowed = set(config["data_boundary"]["allowed_texbat_recordings"])
    configured = set(config["real_clean"])
    if configured != allowed:
        raise ValueError(f"configured real-clean inputs {sorted(configured)} != allowed {sorted(allowed)}")
    forbidden = {name.lower() for name in config["data_boundary"]["forbidden_texbat_recordings"]}
    for name, item in config["real_clean"].items():
        text = f"{name} {item['receiver_dir']}".lower()
        if any(f"/{blocked}/" in text or f"-{blocked}-" in text for blocked in forbidden):
            raise ValueError(f"forbidden TEXBAT scenario configured: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)

    config_path = _repo_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("unsupported domain-gap audit config version")
    _validate_boundary(config)
    output_root = _repo_path(args.output_root or config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    simulation_csv = _repo_path(config["simulation"]["feature_csv"])
    observed_simulation_sha = _sha256(simulation_csv)
    expected_simulation_sha = config["simulation"]["expected_feature_sha256"]
    if observed_simulation_sha != expected_simulation_sha:
        raise ValueError("simulation feature CSV SHA-256 does not match frozen config")
    simulation_all = _read_csv(simulation_csv)
    simulation_rows = select_rows(simulation_all, config["simulation"]["normal_selectors"])
    if not simulation_rows:
        raise ValueError("simulation normal selection produced zero rows")
    if any(row.get("is_spoofing") != "0" or row.get("label") != "normal" for row in simulation_rows):
        raise ValueError("simulation normal selection contains spoofing rows")
    _write_csv(output_root / "simulation_normal_selected.csv", simulation_rows)

    extraction = config["extraction"]
    real_by_source: dict[str, list[dict[str, str]]] = {}
    real_feature_provenance: dict[str, Any] = {}
    for source_name, source in config["real_clean"].items():
        receiver_dir = _repo_path(source["receiver_dir"])
        feature_path = output_root / f"texbat_{source_name}_3tap_features.csv"
        export_receiver_run_tracking_feature_csv(
            receiver_dir,
            output_path=feature_path,
            tap_count=int(extraction["tap_count"]),
            window_s=float(extraction["window_s"]),
            stride_s=float(extraction["stride_s"]),
            min_epochs=int(extraction["min_epochs"]),
            label="normal",
        )
        rows = _read_csv(feature_path)
        for row in rows:
            row["domain_source"] = source_name
        real_by_source[source_name] = rows
        manifest_path = receiver_dir / "manifest.json"
        real_feature_provenance[source_name] = {
            "receiver_dir": str(receiver_dir),
            "receiver_manifest_sha256": _sha256(manifest_path),
            "extracted_feature_csv": str(feature_path),
            "extracted_feature_sha256": _sha256(feature_path),
            "row_count": len(rows),
        }

    combined_real = [row for name in sorted(real_by_source) for row in real_by_source[name]]
    comparisons = dict(real_by_source)
    comparisons["cleanCombined"] = combined_real
    feature_columns = config["feature_columns"]
    classifier_config = config["classifier"]
    per_feature_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    result_by_comparison: dict[str, Any] = {}
    for comparison_name, real_rows in comparisons.items():
        feature_metrics, distribution_summary = compare_feature_distributions(
            simulation_rows,
            real_rows,
            feature_columns,
        )
        classifier_summary = domain_classifier_audit(
            simulation_rows,
            real_rows,
            feature_columns,
            simulation_source_column="paired_group_id",
            max_rows_per_group=int(classifier_config["max_windows_per_source_prn_group"]),
            n_splits=int(classifier_config["n_splits"]),
            random_state=int(classifier_config["random_state"]),
        )
        status, reasons = assign_gate_status(
            distribution_summary,
            classifier_summary,
            config["gate"],
        )
        per_feature_rows.extend({"comparison": comparison_name, **row} for row in feature_metrics)
        fold_rows.extend({"comparison": comparison_name, **row} for row in classifier_summary["folds"])
        result_by_comparison[comparison_name] = {
            "gate_status": status,
            "stop_reasons": reasons,
            "distribution": distribution_summary,
            "domain_classifier": classifier_summary,
        }

    _write_csv(output_root / "per_feature_metrics.csv", per_feature_rows)
    _write_csv(output_root / "domain_classifier_folds.csv", fold_rows)

    receiver_states: dict[str, Any] = {}
    for name, source in config["simulation"]["receiver_state_sources"].items():
        receiver_states[name] = _receiver_state_summary(
            _repo_path(source["receiver_dir"]),
            source.get("intervals_s"),
        )
    for name, source in config["real_clean"].items():
        receiver_states[f"texbat_{name}"] = _receiver_state_summary(_repo_path(source["receiver_dir"]))
    expected_spacing = float(extraction["tap_spacing_chips"])
    mismatched_spacing = {
        name: state["tap_spacing_chips"]
        for name, state in receiver_states.items()
        if not np.isclose(state["tap_spacing_chips"], expected_spacing)
    }
    if mismatched_spacing:
        raise ValueError(f"receiver tap spacing does not match {expected_spacing}: {mismatched_spacing}")

    overall_status = worst_gate_status(
        result["gate_status"] for result in result_by_comparison.values()
    )
    summary = {
        "audit_name": config["audit_name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "data_boundary": {
            **config["data_boundary"],
            "forbidden_scenarios_accessed": False,
        },
        "feature_schema": {
            "tap_count": int(extraction["tap_count"]),
            "tap_spacing_chips": expected_spacing,
            "window_s": float(extraction["window_s"]),
            "stride_s": float(extraction["stride_s"]),
            "columns": feature_columns,
        },
        "simulation": {
            "source_csv": str(simulation_csv),
            "source_sha256": observed_simulation_sha,
            "input_rows": len(simulation_all),
            "selected_normal_rows": len(simulation_rows),
            "selectors": config["simulation"]["normal_selectors"],
            "selected_state_counts": {
                state: sum(row["event_state"] == state for row in simulation_rows)
                for state in sorted({row["event_state"] for row in simulation_rows})
            },
        },
        "real_clean": real_feature_provenance,
        "receiver_state": receiver_states,
        "gate": config["gate"],
        "comparisons": result_by_comparison,
        "overall_gate_status": overall_status,
        "decision": (
            "do_not_scale_simulation_v4; anchor or normalize against real clean and rerun the gate"
            if overall_status == "stop"
            else "scale only under the restrictions associated with the reported gate status"
        ),
        "limitations": [
            "This is a domain-fidelity engineering screen, not detector accuracy or WCL evidence.",
            "Only one independent simulation-v4 paired campaign exists, so cross-campaign simulation generalization is not estimable.",
            "Group CV holds out paired-campaign/recording plus PRN groups, but source-wide receiver signatures can still make separability optimistic.",
            "TEXBAT spoofing scenarios ds1-ds8 remain untouched for later frozen detector evaluation.",
        ],
    }
    summary_path = output_root / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps({
        "summary": str(summary_path),
        "overall_gate_status": overall_status,
        "comparisons": {
            name: {
                "gate_status": result["gate_status"],
                "domain_auc": result["domain_classifier"]["pooled_separability_auc"],
                "median_ks": result["distribution"]["median_ks_statistic"],
                "median_robust_shift": result["distribution"]["median_robust_median_shift"],
            }
            for name, result in result_by_comparison.items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
