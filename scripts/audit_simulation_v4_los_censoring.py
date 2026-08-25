#!/usr/bin/env python3
"""Audit LOS peak spreading and tracking-attrition censoring on train only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
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
    los_displacement_proxy,
    parse_gps_sdr_sim_los_table,
)

DEFAULT_CONFIG = Path("configs/experiments/simulation_v4_los_censoring_audit_v1.json")
NAV_SHA_PATTERN = re.compile(r"'nav_sha256':\s*'([0-9a-f]{64})'")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_strict_json(document), stream, indent=2, sort_keys=True, allow_nan=False)
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


def _load_pinned(source: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    path = _repo_path(source["path"])
    observed = _sha256(path)
    if observed != source["sha256"]:
        raise ValueError(f"{name} SHA-256 mismatch: {observed}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def validate_config(
    config: dict[str, Any], *, verify_outputs: bool = False
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    if config.get("version") != 1:
        raise ValueError("unsupported LOS-censoring audit config version")
    experiment = config.get("experiment", {})
    splitter._safe_name(str(experiment.get("name", "")), "experiment.name")
    if experiment.get("role") != "exploratory train-only mechanism and improvement generation":
        raise ValueError("experiment role must remain exploratory and train-only")
    if "post-hoc" not in str(experiment.get("analysis_origin", "")):
        raise ValueError("the train-generated follow-up must remain labeled post-hoc")

    source_path, source = _load_pinned(config["source_audit"], "source audit")
    source_status = source.get("physical_diagnostic", {}).get("exploratory_status")
    if source_status != config["source_audit"]["required_status"]:
        raise ValueError("source v1 failure status changed")
    artifact_path, artifact = _load_pinned(
        config["source_artifact_summary"], "source artifact summary"
    )
    split_path, split = _load_pinned(config["split_config"], "split config")
    splitter.validate_config(split)
    train_ids = [
        str(pair["paired_group_id"])
        for pair in split["pairs"]
        if pair["split"] == "train"
    ]
    boundary = config.get("data_boundary", {})
    if boundary.get("allowed_partition") != "train":
        raise ValueError("only train may enter the LOS-censoring audit")
    if boundary.get("allowed_pair_ids") != train_ids or len(train_ids) != 6:
        raise ValueError("allowed roster differs from the frozen train split")
    if boundary.get("validation_pairs_accessed") is not False:
        raise ValueError("validation access must remain false")
    if boundary.get("test_pairs_accessed") is not False:
        raise ValueError("test access must remain false")
    if boundary.get("texbat_recordings_accessed") != []:
        raise ValueError("TEXBAT may not enter this train-only audit")
    if source.get("data_boundary", {}).get("observed_pair_ids") != train_ids:
        raise ValueError("source audit pair boundary mismatch")
    if set(artifact.get("artifacts", {})) != set(train_ids):
        raise ValueError("source artifact pair boundary mismatch")

    physics = config.get("physics", {})
    if (
        float(physics.get("code_rate_hz", 0.0)) <= 0
        or float(physics.get("speed_of_light_m_s", 0.0)) <= 0
        or physics.get("observed_width_column") != "width_excess"
        or not re.fullmatch(r"[0-9a-f]{64}", str(physics.get("expected_navigation_sha256", "")))
    ):
        raise ValueError("invalid LOS physical contract")
    analysis = config.get("analysis", {})
    if (
        analysis.get("physical_interval") != "carryoff_transition"
        or analysis.get("correlation_measure") != "spearman"
        or analysis.get("score_column") != "width_z"
        or analysis.get("event_aggregation") != "median_across_available_prns"
        or float(analysis.get("event_grid_s", 0.0)) <= 0
        or not 0.5 < float(analysis.get("normal_counterpart_quantile", 0.0)) < 1.0
        or int(analysis.get("persistence_windows", 0)) < 1
        or int(analysis.get("strict_minimum_prns", 0)) != 4
        or int(analysis.get("censoring_aware_minimum_prns", 0)) != 1
        or not float(analysis.get("support_baseline_start_s", -1))
        < float(analysis.get("support_baseline_end_s", -1))
        < 10.0
    ):
        raise ValueError("invalid frozen censoring analysis contract")
    confirmation = config.get("confirmation", {})
    if (
        confirmation.get("formula_and_policies_must_freeze_before_validation") is not True
        or confirmation.get("test_remains_locked") is not True
    ):
        raise ValueError("confirmation boundary must remain frozen")

    if verify_outputs:
        scores = source.get("outputs", {}).get("prn_window_scores", {})
        score_path = Path(str(scores.get("path", ""))).resolve()
        if not score_path.is_file() or _sha256(score_path) != scores.get("sha256"):
            raise ValueError("source PRN score artifact integrity failure")
    return source_path, source, artifact_path, artifact, split_path, split


def _navigation_sha(text: str) -> str:
    matches = NAV_SHA_PATTERN.findall(text)
    if len(matches) != 1:
        raise ValueError("simulator log must contain exactly one navigation SHA-256")
    return matches[0]


def _los_sources(
    pair_id: str,
    artifact: dict[str, Any],
    expected_nav_sha: str,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, Any]]:
    pair_source = artifact["artifacts"][pair_id]
    pair_manifest_path = Path(pair_source["pair_manifest"]).resolve()
    if _sha256(pair_manifest_path) != pair_source["pair_manifest_sha256"]:
        raise ValueError(f"pair manifest integrity failure: {pair_id}")
    pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    component_path = Path(pair_manifest["component_manifest"]).resolve()
    if _sha256(component_path) != pair_manifest["component_manifest_sha256"]:
        raise ValueError(f"component manifest integrity failure: {pair_id}")
    logs, tables = {}, {}
    for member in ("authentic", "counterfeit"):
        path = component_path.parent / f"{member}-gps-sdr-sim.log"
        text = path.read_text(encoding="utf-8")
        nav_sha = _navigation_sha(text)
        if nav_sha != expected_nav_sha:
            raise ValueError(f"navigation provenance mismatch: {pair_id}/{member}")
        tables[member] = parse_gps_sdr_sim_los_table(text)
        logs[member] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "navigation_sha256": nav_sha,
            "prn_count": len(tables[member]),
        }
    if tables["authentic"] != tables["counterfeit"]:
        raise ValueError(f"initial authentic/counterfeit LOS mismatch: {pair_id}")
    return tables["authentic"], {
        "pair_manifest": {"path": str(pair_manifest_path), "sha256": _sha256(pair_manifest_path)},
        "component_manifest": {"path": str(component_path), "sha256": _sha256(component_path)},
        "logs": logs,
    }


def _state(pair: dict[str, Any], member: str, time_s: float) -> str:
    if member == "normal":
        return "steady_normal"
    onset = float(pair["spoofing"]["start_seconds"])
    end = onset + float(pair["spoofing"]["transition_seconds"])
    if time_s < onset:
        return "pre_event_normal"
    if time_s <= end:
        return "carryoff_transition"
    return "carryoff_final"


def _enrich_scores(
    scores: pd.DataFrame,
    pairs: dict[str, dict[str, Any]],
    artifact: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "pair_id", "member", "prn", "window_end_s", "width_excess", "width_z"
    }
    if not required.issubset(scores.columns) or scores.empty:
        raise ValueError("invalid source PRN score table")
    if set(scores["pair_id"].astype(str)) != set(pairs):
        raise ValueError("source PRN score pair boundary mismatch")
    if set(scores["member"].astype(str)) != {"normal", "spoof"}:
        raise ValueError("unexpected source score member")
    for column in ("window_end_s", "width_excess", "width_z"):
        scores[column] = pd.to_numeric(scores[column], errors="raise")
    if not np.isfinite(scores[["window_end_s", "width_excess", "width_z"]]).all().all():
        raise ValueError("nonfinite source PRN score")

    chip_length = float(config["physics"]["speed_of_light_m_s"]) / float(
        config["physics"]["code_rate_hz"]
    )
    enriched, provenance = [], {}
    for pair_id, pair in pairs.items():
        los, provenance[pair_id] = _los_sources(
            pair_id, artifact, str(config["physics"]["expected_navigation_sha256"])
        )
        frame = scores[scores["pair_id"] == pair_id].copy()
        missing = sorted(set(frame["prn"].astype(str)) - set(los))
        if missing:
            raise ValueError(f"tracked PRNs missing from LOS table: {pair_id}: {missing}")
        los_proxy, enu_proxy, states = [], [], []
        for row in frame.itertuples(index=False):
            time_s = float(row.window_end_s)
            states.append(_state(pair, str(row.member), time_s))
            if row.member == "normal":
                los_proxy.append(0.0)
                enu_proxy.append(0.0)
            else:
                los_proxy.append(
                    los_displacement_proxy(
                        pair["spoofing"], time_s, los[str(row.prn)], chip_length_m=chip_length
                    )
                )
                enu_proxy.append(
                    displacement_envelope_proxy(
                        pair["spoofing"], time_s, chip_length_m=chip_length
                    )
                )
        frame["los_variance_proxy"] = los_proxy
        frame["enu_variance_envelope_proxy"] = enu_proxy
        frame["physical_state"] = states
        enriched.append(frame)
    result = pd.concat(enriched, ignore_index=True)
    return result.sort_values(
        ["pair_id", "member", "prn", "window_end_s"], kind="stable"
    ), provenance


def _spearman(frame: pd.DataFrame, predictor: str) -> dict[str, Any]:
    values = frame[[predictor, "width_excess"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3 or values[predictor].nunique() < 2:
        raise ValueError(f"insufficient variation for Spearman correlation: {predictor}")
    result = spearmanr(values[predictor], values["width_excess"])
    return {
        "row_count": int(len(values)),
        "spearman_rho": float(result.statistic),
        "two_sided_p_value": float(result.pvalue),
    }


def _physical_diagnostic(
    scores: pd.DataFrame, pairs: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    transition = scores[
        (scores["member"] == "spoof")
        & (scores["physical_state"] == "carryoff_transition")
    ].copy()
    rows = []
    for pair_id in pairs:
        group = transition[transition["pair_id"] == pair_id]
        enu = _spearman(group, "enu_variance_envelope_proxy")
        los = _spearman(group, "los_variance_proxy")
        rows.append(
            {
                "pair_id": pair_id,
                "transition_prn_window_count": int(len(group)),
                "enu_proxy_spearman_rho": enu["spearman_rho"],
                "enu_proxy_p_value": enu["two_sided_p_value"],
                "los_proxy_spearman_rho": los["spearman_rho"],
                "los_proxy_p_value": los["two_sided_p_value"],
            }
        )
    pair_frame = pd.DataFrame(rows)
    within = []
    for (pair_id, prn), group in transition.groupby(["pair_id", "prn"]):
        if len(group) < 8 or group["los_variance_proxy"].nunique() < 2:
            continue
        result = spearmanr(group["los_variance_proxy"], group["width_excess"])
        within.append(
            {"pair_id": pair_id, "prn": prn, "row_count": int(len(group)), "spearman_rho": float(result.statistic)}
        )
    within_rho = np.asarray([row["spearman_rho"] for row in within], dtype=np.float64)
    pooled_enu = _spearman(transition, "enu_variance_envelope_proxy")
    pooled_los = _spearman(transition, "los_variance_proxy")
    return pair_frame, {
        "interval": "spoof carry-off transition; PRN-window level",
        "pooled_enu_envelope": pooled_enu,
        "pooled_los_corrected": pooled_los,
        "los_minus_enu_spearman_rho": pooled_los["spearman_rho"] - pooled_enu["spearman_rho"],
        "pair_los_rho_positive_count": int((pair_frame["los_proxy_spearman_rho"] > 0).sum()),
        "pair_count": int(len(pair_frame)),
        "within_prn": {
            "eligible_prn_count": int(len(within)),
            "median_spearman_rho": float(np.median(within_rho)),
            "positive_count": int((within_rho > 0).sum()),
            "rho_at_least_0_5_count": int((within_rho >= 0.5).sum()),
        },
        "interpretation": "direction-consistent but moderate support; the exact untruncated equal-shape mixture law is not an exact receiver-output predictor",
    }


def _event_scores(
    scores: pd.DataFrame,
    pairs: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    grid = float(config["analysis"]["event_grid_s"])
    scored = scores.copy()
    scored["available_time_s"] = np.floor(scored["window_end_s"] / grid + 1e-8) * grid
    events = scored.groupby(["pair_id", "member", "available_time_s"], sort=True).agg(
        prn_count=("prn", "nunique"),
        width_z=("width_z", "median"),
        width_excess=("width_excess", "median"),
    ).reset_index()
    start = float(config["analysis"]["support_baseline_start_s"])
    end = float(config["analysis"]["support_baseline_end_s"])
    baseline_counts = {}
    for (pair_id, member), group in events.groupby(["pair_id", "member"]):
        baseline = group[(group["available_time_s"] >= start) & (group["available_time_s"] <= end)]
        if baseline.empty:
            raise ValueError(f"empty support baseline: {pair_id}/{member}")
        baseline_counts[(pair_id, member)] = float(baseline["prn_count"].median())
    events["baseline_prn_count"] = [
        baseline_counts[(row.pair_id, row.member)] for row in events.itertuples(index=False)
    ]
    events["support_fraction"] = events["prn_count"] / events["baseline_prn_count"]
    states = []
    for row in events.itertuples(index=False):
        states.append(_state(pairs[row.pair_id], row.member, float(row.available_time_s)))
    events["physical_state"] = states
    labels = np.asarray(
        [int(row.member == "spoof" and row.physical_state in {"carryoff_transition", "carryoff_final"}) for row in events.itertuples(index=False)],
        dtype=np.int8,
    )
    attack = events[labels == 1]
    reference = events[labels == 0]
    support = {
        "attack_support_fraction_median": float(attack["support_fraction"].median()),
        "reference_support_fraction_median": float(reference["support_fraction"].median()),
        "support_deficit_auc_diagnostic": float(roc_auc_score(labels, -events["support_fraction"])),
        "attack_event_count": int(len(attack)),
        "reference_event_count": int(len(reference)),
    }
    return events.sort_values(["pair_id", "member", "available_time_s"], kind="stable"), support


def _policy_diagnostic(
    events: pd.DataFrame,
    pairs: dict[str, dict[str, Any]],
    config: dict[str, Any],
    *,
    minimum_prns: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    quantile = float(config["analysis"]["normal_counterpart_quantile"])
    persistence = int(config["analysis"]["persistence_windows"])
    step = float(config["analysis"]["event_grid_s"])
    rows = []
    for pair_id, pair in pairs.items():
        onset = float(pair["spoofing"]["start_seconds"])
        normal = events[
            (events["pair_id"] == pair_id)
            & (events["member"] == "normal")
            & (events["prn_count"] >= minimum_prns)
        ]
        spoof = events[
            (events["pair_id"] == pair_id)
            & (events["member"] == "spoof")
            & (events["prn_count"] >= minimum_prns)
        ]
        if normal.empty or spoof.empty:
            raise ValueError(f"policy has no eligible events: {pair_id}/{minimum_prns}")
        threshold = float(normal["width_z"].quantile(quantile))
        attack_crossing = first_persistent_crossing(
            spoof["available_time_s"], spoof["width_z"], threshold=threshold,
            onset_s=onset, persistence=persistence, expected_step_s=step,
        )
        normal_crossing = first_persistent_crossing(
            normal["available_time_s"], normal["width_z"], threshold=threshold,
            onset_s=onset, persistence=persistence, expected_step_s=step,
        )
        crossing_support = []
        if attack_crossing is not None:
            block = spoof[
                (spoof["available_time_s"] >= attack_crossing)
                & (spoof["available_time_s"] <= attack_crossing + (persistence - 1) * step + 1e-9)
            ]
            crossing_support = [int(value) for value in block["prn_count"].tolist()]
        rows.append(
            {
                "pair_id": pair_id,
                "minimum_prns": int(minimum_prns),
                "normal_q99_threshold": threshold,
                "normal_point_exceedance_rate": float((normal["width_z"] > threshold).mean()),
                "attack_first_persistent_time_s": attack_crossing,
                "attack_availability_delay_s": None if attack_crossing is None else attack_crossing - onset,
                "normal_first_persistent_time_s": normal_crossing,
                "crossing_prn_counts": ",".join(str(value) for value in crossing_support),
            }
        )
    frame = pd.DataFrame(rows)
    delays = frame["attack_availability_delay_s"].dropna().to_numpy(dtype=np.float64)
    normal_crossings = frame["normal_first_persistent_time_s"].dropna()
    return frame, {
        "minimum_prns": int(minimum_prns),
        "detected_pair_count": int(len(delays)),
        "pair_count": int(len(frame)),
        "median_availability_delay_s": float(np.median(delays)) if len(delays) else None,
        "maximum_availability_delay_s": float(np.max(delays)) if len(delays) else None,
        "paired_normal_persistent_warning_count": int(len(normal_crossings)),
        "threshold_interpretation": "train-only paired-normal q99 diagnostic; not a deployable threshold",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    config_path = _repo_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path, source, artifact_path, artifact, split_path, split = validate_config(
        config, verify_outputs=True
    )
    output_root = _repo_path(config["output_root"])
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = {
        str(pair["paired_group_id"]): pair
        for pair in split["pairs"]
        if pair["split"] == "train"
    }
    score_source = source["outputs"]["prn_window_scores"]
    scores = pd.read_csv(Path(score_source["path"]).resolve())
    enriched, los_provenance = _enrich_scores(scores, pairs, artifact, config)
    physical_pairs, physical = _physical_diagnostic(enriched, pairs)
    events, support = _event_scores(enriched, pairs, config)
    strict_pairs, strict = _policy_diagnostic(
        events, pairs, config, minimum_prns=int(config["analysis"]["strict_minimum_prns"])
    )
    aware_pairs, aware = _policy_diagnostic(
        events, pairs, config, minimum_prns=int(config["analysis"]["censoring_aware_minimum_prns"])
    )
    pair_results = physical_pairs.merge(
        strict_pairs.add_prefix("strict_").rename(columns={"strict_pair_id": "pair_id"}), on="pair_id"
    ).merge(
        aware_pairs.add_prefix("aware_").rename(columns={"aware_pair_id": "pair_id"}), on="pair_id"
    )
    common = pair_results.dropna(
        subset=["strict_attack_availability_delay_s", "aware_attack_availability_delay_s"]
    )
    paired_improvement = (
        common["strict_attack_availability_delay_s"]
        - common["aware_attack_availability_delay_s"]
    )
    relaxed_attack = events[
        (events["member"] == "spoof")
        & events["physical_state"].isin(["carryoff_transition", "carryoff_final"])
    ]
    censored_fraction = float(
        (relaxed_attack["prn_count"] < int(config["analysis"]["strict_minimum_prns"])).mean()
    )

    prn_path = output_root / "prn_physical_scores.csv"
    event_path = output_root / "event_scores.csv"
    pair_path = output_root / "pair_results.csv"
    _atomic_frame(prn_path, enriched)
    _atomic_frame(event_path, events)
    _atomic_frame(pair_path, pair_results)
    summary = {
        "schema": "gnss-doppler-lab.simulation-v4-los-censoring-audit",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        "physics_module": {
            "path": str((REPO_ROOT / "src/gnss_doppler_lab/peak_mixture_law.py").resolve()),
            "sha256": _sha256(REPO_ROOT / "src/gnss_doppler_lab/peak_mixture_law.py"),
        },
        "source_audit": {"path": str(source_path), "sha256": _sha256(source_path), "status": source["physical_diagnostic"]["exploratory_status"]},
        "source_artifact_summary": {"path": str(artifact_path), "sha256": _sha256(artifact_path)},
        "split_config": {"path": str(split_path), "sha256": _sha256(split_path)},
        "data_boundary": {
            **config["data_boundary"],
            "observed_pair_ids": list(pairs),
            "validation_pair_ids_accessed": [],
            "test_pair_ids_accessed": [],
        },
        "physical_hypothesis": {
            "variance_law": config["physics"]["variance_law"],
            "mechanism": "carry-off peak spreading and tracking loss make surviving correlation observations missing-not-at-random",
            "los_source": config["physics"]["los_source"],
            "static_los_approximation": config["physics"]["static_los_approximation"],
        },
        "los_provenance": los_provenance,
        "physical_diagnostic": physical,
        "support_attrition": {**support, "strict_attack_event_censored_fraction": censored_fraction},
        "detector_comparison": {
            "contract": config["analysis"],
            "strict_four_prn": strict,
            "censoring_aware_one_or_more_prns": aware,
            "additional_detected_pairs": aware["detected_pair_count"] - strict["detected_pair_count"],
            "detected_set_median_delay_reduction_s": strict["median_availability_delay_s"] - aware["median_availability_delay_s"],
            "common_detected_pairs": int(len(common)),
            "common_pair_median_delay_reduction_s": float(np.median(paired_improvement)) if len(paired_improvement) else None,
            "pair_results": _strict_json(pair_results.to_dict(orient="records")),
        },
        "candidate_status": "train_generated_censoring_mechanism_and_fix_require_frozen_validation",
        "claim_boundary": [
            "The frozen v1 strong per-pair envelope support rule failed on train and remains a reported negative result.",
            "LOS correction made all six pair-level transition correlations positive, but pooled association remained moderate rather than exact.",
            "The one-PRN policy is a train-generated censoring-aware candidate, not a validated deployable detector.",
            "Normal q99 thresholds use paired train counterparts and cannot be deployed as written.",
            "No validation pair, test pair, or TEXBAT recording was accessed.",
        ],
        "next_gate": "freeze the LOS formula, strict and censoring-aware policies, then run once on validation pairs 007-009 without tuning",
        "outputs": {
            "prn_physical_scores": {"path": str(prn_path), "sha256": _sha256(prn_path)},
            "event_scores": {"path": str(event_path), "sha256": _sha256(event_path)},
            "pair_results": {"path": str(pair_path), "sha256": _sha256(pair_path)},
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = output_root / "summary.json"
    _atomic_json(summary_path, summary)
    print(json.dumps(_strict_json({
        "summary": str(summary_path.resolve()),
        "candidate_status": summary["candidate_status"],
        "pooled_physics": physical,
        "support_attrition": summary["support_attrition"],
        "detector_comparison": summary["detector_comparison"],
        "data_boundary": summary["data_boundary"],
    }), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
