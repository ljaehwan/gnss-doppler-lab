#!/usr/bin/env python3
"""Audit the dual-order 9-tap peak-deformation law on train pairs only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import plan_simulation_v4_paired_split as splitter  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import (  # noqa: E402
    displacement_envelope_proxy,
    first_persistent_crossing,
    robust_center_scale,
)
from gnss_doppler_lab.tracking_feature_windows import (  # noqa: E402
    export_receiver_run_tap_feature_csv,
)

DEFAULT_CONFIG = Path("configs/experiments/simulation_v4_peak_mixture_law_v1.json")
WIDTH_FEATURE = "dmcpd_width_variance_mean"
OUTER_FEATURE = "dmcpd_pair4_abs_asym_mean"
SCORE_COLUMNS = ("width_z", "outer_z")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False, lineterminator="\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise



def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return strict-JSON records, mapping pandas missing values to null."""
    return [
        {
            str(key): None if pd.isna(value) else value
            for key, value in record.items()
        }
        for record in frame.to_dict(orient="records")
    ]


def _load_pinned_json(source: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    path = _repo_path(source["path"])
    observed = _sha256(path)
    if observed != source["sha256"]:
        raise ValueError(f"{name} SHA-256 mismatch: {observed}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def validate_config(
    config: dict[str, Any], *, verify_source_artifacts: bool = False
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    if config.get("version") != 1:
        raise ValueError("unsupported peak-mixture audit config version")
    experiment = config.get("experiment", {})
    splitter._safe_name(str(experiment.get("name", "")), "experiment.name")
    if experiment.get("role") != "exploratory train-only physical hypothesis generation":
        raise ValueError("experiment role must remain exploratory and train-only")

    record_path, record = _load_pinned_json(config["source_record"], "source record")
    split_path, split = _load_pinned_json(config["split_config"], "split config")
    splitter.validate_config(split)
    if record.get("decision", {}).get("test_status") != "locked":
        raise ValueError("source record test status must remain locked")
    if record.get("decision", {}).get("model_trained") is not False:
        raise ValueError("source record must precede model training")

    train_ids = [
        str(pair["paired_group_id"]) for pair in split["pairs"] if pair["split"] == "train"
    ]
    boundary = config.get("data_boundary", {})
    if boundary.get("allowed_partition") != "train":
        raise ValueError("only the train partition may be audited")
    if boundary.get("allowed_pair_ids") != train_ids or len(train_ids) != 6:
        raise ValueError("allowed pair roster differs from the frozen train split")
    if boundary.get("validation_pairs_accessed") is not False:
        raise ValueError("validation access must remain false")
    if boundary.get("test_pairs_accessed") is not False:
        raise ValueError("test access must remain false")
    if boundary.get("texbat_recordings_accessed") != []:
        raise ValueError("this physical audit may not access TEXBAT")
    if record.get("data_boundary", {}).get("generated_pair_ids") != train_ids:
        raise ValueError("source record train roster mismatch")

    tracking = config["tracking_contract"]
    if tracking != {
        "tap_count": 9,
        "tap_spacing_chips": 0.125,
        "tap_layout": "E4,E3,E2,E,P,L,L2,L3,L4",
    }:
        raise ValueError("physical audit requires the exact frozen nine-tap contract")
    extraction = config["feature_extraction"]
    if (
        float(extraction["window_s"]) != 0.25
        or float(extraction["stride_s"]) != 0.125
        or int(extraction["min_epochs"]) < 4
        or int(extraction["minimum_prns_per_event"]) < 2
    ):
        raise ValueError("invalid causal short-window extraction contract")
    baseline = config["baseline"]
    earliest_onset = min(float(pair["spoofing"]["start_seconds"]) for pair in split["pairs"])
    if not (
        0 <= float(baseline["start_s"]) < float(baseline["end_s"]) < earliest_onset
        and int(baseline["minimum_windows_per_prn"]) >= 5
        and float(baseline["mad_scale_floor"]) > 0
    ):
        raise ValueError("baseline must be a sufficiently populated pre-onset interval")
    physics = config["physics"]
    if (
        physics.get("width_feature") != WIDTH_FEATURE
        or physics.get("early_warning_feature") != OUTER_FEATURE
        or float(physics["code_rate_hz"]) <= 0
        or float(physics["speed_of_light_m_s"]) <= 0
    ):
        raise ValueError("invalid peak-mixture physical contract")
    diagnostic = config["detection_diagnostic"]
    quantile = float(diagnostic["normal_counterpart_quantile"])
    if (
        diagnostic.get("event_aggregation") != "median_across_prns"
        or not 0.5 < quantile < 1.0
        or int(diagnostic["persistence_windows"]) < 1
        or diagnostic.get("score_available_at") != "window_end_s"
    ):
        raise ValueError("invalid detection diagnostic contract")
    support = config["exploratory_support_rule"]
    if support.get("requires_confirmation_on_validation") is not True:
        raise ValueError("train-generated hypothesis must require validation confirmation")

    if verify_source_artifacts:
        _, source = _load_pinned_json(
            config["source_artifact_summary"], "source artifact summary"
        )
        if source.get("paired_prefix_all_byte_identical") is not True:
            raise ValueError("source pairs do not satisfy the identical-prefix contract")
        if set(source.get("artifacts", {})) != set(train_ids):
            raise ValueError("source artifact roster differs from train")
        if source.get("data_boundary", {}).get("validation_pair_ids_accessed") != []:
            raise ValueError("source artifact summary accessed validation")
        if source.get("data_boundary", {}).get("test_pair_ids_accessed") != []:
            raise ValueError("source artifact summary accessed test")
    return record_path, record, split_path, split


def _feature_cache(
    config: dict[str, Any],
    config_sha256: str,
    output_root: Path,
    pair_id: str,
    member: str,
    source: dict[str, Any],
    *,
    resume: bool,
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    receiver_manifest = Path(source["receiver_manifest"]).resolve()
    expected_receiver_sha = str(source["receiver_manifest_sha256"])
    if _sha256(receiver_manifest) != expected_receiver_sha:
        raise ValueError(f"receiver manifest integrity failure: {pair_id}/{member}")
    receiver_document = json.loads(receiver_manifest.read_text(encoding="utf-8"))
    tracking = receiver_document.get("tracking", {})
    expected = config["tracking_contract"]
    if (
        int(tracking.get("tap_count", 0)) != int(expected["tap_count"])
        or float(tracking.get("tap_spacing_chips", 0.0))
        != float(expected["tap_spacing_chips"])
    ):
        raise ValueError(f"receiver tap contract mismatch: {pair_id}/{member}")
    feature_path = output_root / "features" / f"{pair_id}-{member}_9tap_w025.csv"
    manifest_path = feature_path.with_suffix(".manifest.json")
    if feature_path.is_file() and manifest_path.is_file():
        if not resume:
            raise FileExistsError(feature_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("config_sha256") != config_sha256
            or manifest.get("receiver_manifest_sha256") != expected_receiver_sha
            or manifest.get("feature_csv_sha256") != _sha256(feature_path)
        ):
            raise ValueError(f"feature cache provenance mismatch: {pair_id}/{member}")
    else:
        if feature_path.exists() or manifest_path.exists():
            raise FileExistsError(f"partial feature cache: {pair_id}/{member}")
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = feature_path.with_suffix(".tmp.csv")
        extraction = config["feature_extraction"]
        export_receiver_run_tap_feature_csv(
            receiver_manifest.parent,
            output_path=temporary,
            tap_count=int(expected["tap_count"]),
            window_s=float(extraction["window_s"]),
            stride_s=float(extraction["stride_s"]),
            min_epochs=int(extraction["min_epochs"]),
            label=member,
        )
        os.replace(temporary, feature_path)
        frame = pd.read_csv(feature_path)
        manifest = {
            "schema": "gnss-doppler-lab.simulation-v4-peak-mixture-feature-cache",
            "schema_version": 1,
            "config_sha256": config_sha256,
            "pair_id": pair_id,
            "member": member,
            "receiver_manifest": str(receiver_manifest),
            "receiver_manifest_sha256": expected_receiver_sha,
            "extraction": extraction,
            "tracking_contract": expected,
            "feature_csv": str(feature_path.resolve()),
            "feature_csv_sha256": _sha256(feature_path),
            "row_count": int(len(frame)),
        }
        _atomic_json(manifest_path, manifest)
    frame = pd.read_csv(feature_path)
    required = {
        "run_id",
        "source_fingerprint",
        "prn",
        "window_mid_s",
        "window_end_s",
        "tap_count",
        "tap_layout",
        WIDTH_FEATURE,
        OUTER_FEATURE,
    }
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"invalid nine-tap feature cache: {pair_id}/{member}")
    if not (pd.to_numeric(frame["tap_count"], errors="raise") == 9).all():
        raise ValueError(f"non-nine-tap feature row: {pair_id}/{member}")
    if not (frame["tap_layout"] == expected["tap_layout"]).all():
        raise ValueError(f"tap layout mismatch: {pair_id}/{member}")
    frame["pair_id"] = pair_id
    frame["member"] = member
    return feature_path, frame, manifest


def _robust_scores(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baseline = config["baseline"]
    mask = (
        (frame["window_mid_s"] >= float(baseline["start_s"]))
        & (frame["window_end_s"] <= float(baseline["end_s"]))
    )
    records: list[dict[str, Any]] = []
    for identity, group in frame.loc[mask].groupby(["pair_id", "member", "prn"]):
        if len(group) < int(baseline["minimum_windows_per_prn"]):
            continue
        record = {"pair_id": identity[0], "member": identity[1], "prn": identity[2]}
        record["baseline_window_count"] = int(len(group))
        for source_name, short_name in (
            (WIDTH_FEATURE, "width"),
            (OUTER_FEATURE, "outer"),
        ):
            center, scale = robust_center_scale(
                group[source_name].to_numpy(dtype=np.float64),
                scale_floor=float(baseline["mad_scale_floor"]),
            )
            record[f"{short_name}_center"] = center
            record[f"{short_name}_scale"] = scale
        records.append(record)
    baselines = pd.DataFrame(records)
    if baselines.empty:
        raise ValueError("no PRN has a valid pre-onset baseline")
    scored = frame.merge(baselines, on=["pair_id", "member", "prn"], how="left")
    eligible = scored["baseline_window_count"].notna()
    excluded_rows = int((~eligible).sum())
    scored = scored.loc[eligible].copy()
    scored["width_excess"] = scored[WIDTH_FEATURE] - scored["width_center"]
    scored["outer_excess"] = scored[OUTER_FEATURE] - scored["outer_center"]
    scored["width_z"] = scored["width_excess"] / scored["width_scale"]
    scored["outer_z"] = scored["outer_excess"] / scored["outer_scale"]
    if not np.isfinite(scored[[*SCORE_COLUMNS, "width_excess", "outer_excess"]]).all().all():
        raise ValueError("nonfinite robust physical score")
    return scored, baselines, {
        "input_rows": int(len(frame)),
        "eligible_rows": int(len(scored)),
        "excluded_without_pre_onset_baseline": excluded_rows,
        "eligible_prn_run_count": int(len(baselines)),
    }


def _event_state(pair: dict[str, Any], member: str, available_time_s: float) -> str:
    if member == "normal":
        return "steady_normal"
    event = pair["spoofing"]
    if available_time_s < float(event["start_seconds"]):
        return "pre_event_normal"
    if available_time_s < float(event["start_seconds"]) + float(event["transition_seconds"]):
        return "carryoff_transition"
    return "carryoff_final"


def _event_scores(
    scored: pd.DataFrame, config: dict[str, Any], pairs: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    stride = float(config["feature_extraction"]["stride_s"])
    scored = scored.copy()
    scored["available_time_s"] = (
        np.floor(scored["window_end_s"] / stride + 1e-8) * stride
    )
    grouped = scored.groupby(["pair_id", "member", "available_time_s"], sort=True)
    events = grouped.agg(
        prn_count=("prn", "nunique"),
        width_excess=("width_excess", "median"),
        outer_excess=("outer_excess", "median"),
        width_z=("width_z", "median"),
        outer_z=("outer_z", "median"),
    ).reset_index()
    events = events[
        events["prn_count"] >= int(config["feature_extraction"]["minimum_prns_per_event"])
    ].copy()
    chip_length = float(config["physics"]["speed_of_light_m_s"]) / float(
        config["physics"]["code_rate_hz"]
    )
    states, proxies = [], []
    for row in events.itertuples(index=False):
        pair = pairs[row.pair_id]
        states.append(_event_state(pair, row.member, float(row.available_time_s)))
        proxies.append(
            0.0
            if row.member == "normal"
            else displacement_envelope_proxy(
                pair["spoofing"], float(row.available_time_s), chip_length_m=chip_length
            )
        )
    events["event_state"] = states
    events["mixture_variance_envelope_proxy"] = proxies
    return events.sort_values(["pair_id", "member", "available_time_s"], kind="stable")


def _spearman(frame: pd.DataFrame) -> dict[str, Any]:
    finite = frame[["mixture_variance_envelope_proxy", "width_excess"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(finite) < 3:
        raise ValueError("insufficient finite rows for physical correlation")
    result = spearmanr(
        finite["mixture_variance_envelope_proxy"], finite["width_excess"]
    )
    return {
        "row_count": int(len(finite)),
        "spearman_rho": float(result.statistic),
        "two_sided_p_value": float(result.pvalue),
    }


def _diagnostics(
    events: pd.DataFrame, config: dict[str, Any], pairs: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    diagnostic = config["detection_diagnostic"]
    quantile = float(diagnostic["normal_counterpart_quantile"])
    persistence = int(diagnostic["persistence_windows"])
    stride = float(config["feature_extraction"]["stride_s"])
    pair_rows: list[dict[str, Any]] = []
    correlations: dict[str, Any] = {}
    for pair_id, pair in pairs.items():
        onset = float(pair["spoofing"]["start_seconds"])
        transition_end = onset + float(pair["spoofing"]["transition_seconds"])
        attack_transition = events[
            (events["pair_id"] == pair_id)
            & (events["member"] == "spoof")
            & (events["available_time_s"] >= onset)
            & (events["available_time_s"] <= transition_end)
        ]
        correlations[pair_id] = _spearman(attack_transition)
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "domain": pair["domain"],
            "motion_kind": splitter._motion_kind(pair),
            "width_proxy_spearman_rho": correlations[pair_id]["spearman_rho"],
            "width_proxy_p_value": correlations[pair_id]["two_sided_p_value"],
        }
        pre = events[
            (events["pair_id"] == pair_id)
            & (events["member"] == "spoof")
            & (events["available_time_s"] < onset)
        ]
        row["pre_onset_width_z_median"] = float(pre["width_z"].median())
        for score in SCORE_COLUMNS:
            normal = events[
                (events["pair_id"] == pair_id) & (events["member"] == "normal")
            ]
            spoof = events[
                (events["pair_id"] == pair_id) & (events["member"] == "spoof")
            ]
            threshold = float(normal[score].quantile(quantile))
            crossing = first_persistent_crossing(
                spoof["available_time_s"],
                spoof[score],
                threshold=threshold,
                onset_s=onset,
                persistence=persistence,
                expected_step_s=stride,
            )
            prefix = "width" if score == "width_z" else "outer_warning"
            row[f"{prefix}_normal_q99_threshold"] = threshold
            row[f"{prefix}_normal_point_exceedance_rate"] = float(
                (normal[score] > threshold).mean()
            )
            row[f"{prefix}_first_persistent_time_s"] = crossing
            row[f"{prefix}_availability_delay_s"] = (
                None if crossing is None else crossing - onset
            )
        pair_rows.append(row)
    pair_frame = pd.DataFrame(pair_rows)
    pooled_parts = []
    for pair_id, pair in pairs.items():
        onset = float(pair["spoofing"]["start_seconds"])
        transition_end = onset + float(pair["spoofing"]["transition_seconds"])
        pooled_parts.append(
            events[
                (events["pair_id"] == pair_id)
                & (events["member"] == "spoof")
                & (events["available_time_s"] >= onset)
                & (events["available_time_s"] <= transition_end)
            ]
        )
    pooled = _spearman(pd.concat(pooled_parts, ignore_index=True))
    labels = np.asarray(
        [
            int(
                row.member == "spoof"
                and row.available_time_s
                >= float(pairs[row.pair_id]["spoofing"]["start_seconds"])
            )
            for row in events.itertuples(index=False)
        ],
        dtype=np.int8,
    )
    auc = {
        score: float(roc_auc_score(labels, events[score].to_numpy(dtype=np.float64)))
        for score in SCORE_COLUMNS
    }
    support = config["exploratory_support_rule"]
    pre_median = float(
        events[events["event_state"] == "pre_event_normal"]["width_z"].median()
    )
    supported = (
        pooled["spearman_rho"]
        >= float(support["pooled_width_proxy_spearman_min"])
        and (pair_frame["width_proxy_spearman_rho"] >= float(
            support["each_pair_width_proxy_spearman_min"]
        )).all()
        and abs(pre_median)
        <= float(support["absolute_pre_onset_width_z_median_max"])
    )
    return pair_frame, {
        "pooled_width_proxy_correlation": pooled,
        "pair_width_proxy_correlations": correlations,
        "event_auc_diagnostic": auc,
        "pre_event_normal_width_z_median": pre_median,
        "exploratory_support_rule": support,
        "exploratory_status": "supported_on_train_requires_validation"
        if supported
        else "not_supported_on_train",
        "support_rule_passed": bool(supported),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    config_path = _repo_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    record_path, record, split_path, split = validate_config(
        config, verify_source_artifacts=True
    )
    source_path, source = _load_pinned_json(
        config["source_artifact_summary"], "source artifact summary"
    )
    config_sha = _sha256(config_path)
    runner_sha = _sha256(Path(__file__))
    output_root = _repo_path(config["output_root"])
    if output_root.exists() and not args.resume:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = {
        str(pair["paired_group_id"]): pair
        for pair in split["pairs"]
        if pair["split"] == "train"
    }

    frames, caches = [], {}
    for pair_id in config["data_boundary"]["allowed_pair_ids"]:
        caches[pair_id] = {}
        for member in ("normal", "spoof"):
            print(f"[9tap] {pair_id}/{member}", flush=True)
            path, frame, manifest = _feature_cache(
                config,
                config_sha,
                output_root,
                pair_id,
                member,
                source["artifacts"][pair_id]["members"][member],
                resume=args.resume,
            )
            frames.append(frame)
            caches[pair_id][member] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "row_count": int(len(frame)),
                "manifest_sha256": _sha256(path.with_suffix(".manifest.json")),
            }
    all_features = pd.concat(frames, ignore_index=True)
    numeric = ["window_mid_s", "window_end_s", WIDTH_FEATURE, OUTER_FEATURE]
    for column in numeric:
        all_features[column] = pd.to_numeric(all_features[column], errors="raise")
    scored, baselines, eligibility = _robust_scores(all_features, config)
    events = _event_scores(scored, config, pairs)
    pair_results, diagnostic = _diagnostics(events, config, pairs)

    prn_columns = [
        "pair_id",
        "member",
        "run_id",
        "source_fingerprint",
        "prn",
        "channel",
        "segment_index",
        "window_index",
        "window_mid_s",
        "window_end_s",
        "epoch_count",
        WIDTH_FEATURE,
        OUTER_FEATURE,
        "width_center",
        "width_scale",
        "outer_center",
        "outer_scale",
        "width_excess",
        "outer_excess",
        "width_z",
        "outer_z",
    ]
    prn_path = output_root / "prn_window_scores.csv"
    baseline_path = output_root / "prn_baselines.csv"
    event_path = output_root / "event_scores.csv"
    pair_path = output_root / "pair_results.csv"
    _atomic_frame(prn_path, scored[prn_columns])
    _atomic_frame(baseline_path, baselines)
    _atomic_frame(event_path, events)
    _atomic_frame(pair_path, pair_results)

    delays = {}
    for prefix in ("outer_warning", "width"):
        values = pair_results[f"{prefix}_availability_delay_s"].dropna().to_numpy()
        delays[prefix] = {
            "detected_pairs": int(len(values)),
            "median_delay_s": float(np.median(values)) if len(values) else None,
            "maximum_delay_s": float(np.max(values)) if len(values) else None,
        }
    summary = {
        "schema": "gnss-doppler-lab.simulation-v4-dual-order-peak-deformation-audit",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "config": {"path": str(config_path), "sha256": config_sha},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": runner_sha},
        "source_record": {"path": str(record_path), "sha256": _sha256(record_path)},
        "source_artifact_summary": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
        "split_config": {"path": str(split_path), "sha256": _sha256(split_path)},
        "data_boundary": {
            **config["data_boundary"],
            "observed_pair_ids": list(pairs),
            "validation_pair_ids_accessed": [],
            "test_pair_ids_accessed": [],
        },
        "physical_hypothesis": {
            "model": "two equal-shape correlation profiles with relative voltage rho and code-delay separation delta",
            "variance_identity": "Delta V = rho/(1+rho)^2 * delta^2",
            "first_order_warning": "absolute outer E4/L4 shoulder asymmetry",
            "second_order_confirmation": "baseline-normalized nine-tap width variance excess",
            "proxy_scope": config["physics"]["proxy_scope"],
        },
        "tracking_contract": config["tracking_contract"],
        "feature_extraction": config["feature_extraction"],
        "baseline": config["baseline"],
        "feature_caches": caches,
        "eligibility": eligibility,
        "event_count": int(len(events)),
        "event_state_counts": {
            str(key): int(value) for key, value in events["event_state"].value_counts().items()
        },
        "physical_diagnostic": diagnostic,
        "detection_diagnostic": {
            "contract": config["detection_diagnostic"],
            "delays": delays,
            "pair_results": _json_records(pair_results),
            "interpretation": "train-only paired-counterpart diagnostic; thresholds are not deployable or frozen",
        },
        "outputs": {
            "prn_window_scores": {"path": str(prn_path), "sha256": _sha256(prn_path)},
            "prn_baselines": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
            "event_scores": {"path": str(event_path), "sha256": _sha256(event_path)},
            "pair_results": {"path": str(pair_path), "sha256": _sha256(pair_path)},
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "limitations": [
            "The hypothesis and descriptive support rule were generated using train data and are not confirmatory evidence.",
            "The displacement proxy uses ENU offset magnitude and does not compute per-satellite LOS code delay.",
            "The paired normal counterpart supplies diagnostic thresholds; a deployable threshold remains unfrozen.",
            "Correlation distortion can also be caused by multipath and must be challenged on independent normal data.",
            "Validation and test pairs were not generated or accessed.",
            "No TEXBAT recording was accessed in this audit.",
        ],
        "next_gate": "freeze this statistic now, then confirm its physical correlation and delays on validation pairs 007-009 without changing the formula",
    }
    summary_path = output_root / "summary.json"
    _atomic_json(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "status": diagnostic["exploratory_status"],
        "pooled_width_proxy_spearman": diagnostic["pooled_width_proxy_correlation"],
        "event_auc": diagnostic["event_auc_diagnostic"],
        "delays": delays,
        "data_boundary": summary["data_boundary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
