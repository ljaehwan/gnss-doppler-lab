#!/usr/bin/env python3
"""Audit frozen CGC scores on normal train RF and derive a train-only freeze."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    sys.path.insert(0, str(search_root))

import run_cgc_rf_challenge_pilot as pilot  # noqa: E402
import run_cgc_rf_train_replication as replication  # noqa: E402
from gnss_doppler_lab.correlator_geometry import EPSILON  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import (  # noqa: E402
    parse_gps_sdr_sim_los_table,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/experiments/cgc_normal_detector_freeze_audit_v1.json"
)
EXPECTED_PAIR_IDS = [f"pv1-pair-{index:03d}" for index in range(2, 7)]
EXPECTED_GATES = {
    "required_pair_count": 5,
    "minimum_finite_score_fraction": 1.0,
    "minimum_full_rank_fraction": 1.0,
    "maximum_lopo_normal_persistent_alarm_pair_count": 0,
    "maximum_lopo_multipath_persistent_alarm_pair_count": 1,
    "minimum_lopo_spoof_persistent_detection_pair_count": 4,
    "minimum_lopo_macro_balanced_accuracy": 0.80,
}
SCENARIOS = ("normal", "independent_multipath", "carryoff_spoof")
BENIGN_SCENARIOS = ("normal", "independent_multipath")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    return (REPO_ROOT / value).resolve()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fieldnames = list(dict.fromkeys(
        field
        for row in rows
        for field in row
    ))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_pinned(record: dict[str, Any], label: str) -> tuple[Path, Any]:
    path = _repo_path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"{label} hash mismatch")
    if path.suffix == ".json":
        return path, json.loads(path.read_text(encoding="utf-8"))
    return path, None


def _normal_iq_record(manifest: dict[str, Any]) -> tuple[Path, str, int]:
    iq = manifest["iq"]
    path = Path(iq["path"]).resolve()
    return path, str(iq["sha256"]), int(iq["actual_bytes"])


def validate_config(
    config: dict[str, Any], *, verify_iq: bool = False
) -> dict[str, Any]:
    if config.get("schema") != (
        "gnss-doppler-lab.cgc-normal-detector-freeze-audit-config"
    ):
        raise ValueError("unsupported normal-audit config schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported normal-audit config version")

    experiment = config.get("experiment", {})
    if experiment.get("candidate_commit") != (
        "60280575737ff70618c5619ff7522c516bbdc67a"
    ):
        raise ValueError("candidate commit drifted")
    if experiment.get("execution_policy") != (
        "one deterministic execution on train pairs 002--006; no score, "
        "window, threshold-selection, persistence, or gate changes after "
        "normal nine-tap outcome inspection"
    ):
        raise ValueError("execution policy drifted")
    if experiment.get("runner_path") != (
        "scripts/run_cgc_normal_detector_freeze_audit.py"
    ):
        raise ValueError("runner path drifted")

    frozen = config["frozen_candidate"]
    pinned: dict[str, Path] = {}
    documents: dict[str, Any] = {}
    for name in (
        "replication_config",
        "replication_result",
        "replication_artifact_summary",
        "replication_geometry_scores",
        "replication_runner",
        "receiver_runner",
        "clock_centered_module",
        "controlled_template_config",
    ):
        path, document = _load_pinned(frozen[name], name)
        pinned[name] = path
        documents[name] = document

    if frozen.get("residual") != "SSE_LOS+clock / SSE_clock-only":
        raise ValueError("residual drifted")
    if frozen.get("detection_score") != "negative clock-centered residual":
        raise ValueError("score direction drifted")
    if float(frozen.get("clock_energy_epsilon")) != EPSILON:
        raise ValueError("clock energy epsilon drifted")
    if frozen.get("degenerate_clock_energy_policy") != (
        "residual=1 and score=-1; retain as benign-valued finite observation"
    ):
        raise ValueError("degenerate clock policy drifted")

    split_path, split = _load_pinned(config["split_config"], "split config")
    pairs = [
        pair for pair in split["pairs"]
        if pair["paired_group_id"] in EXPECTED_PAIR_IDS
    ]
    if [pair["paired_group_id"] for pair in pairs] != EXPECTED_PAIR_IDS:
        raise ValueError("authorized pair roster drifted")
    if any(pair["split"] != "train" for pair in pairs):
        raise ValueError("non-train pair crossed the data boundary")
    boundary = config["data_boundary"]
    if boundary.get("authorized_partition") != "train":
        raise ValueError("authorized partition drifted")
    if boundary.get("allowed_pair_ids") != EXPECTED_PAIR_IDS:
        raise ValueError("allowed pair IDs drifted")
    if (
        boundary.get("validation_pairs_accessed") is not False
        or boundary.get("test_pairs_accessed") is not False
        or boundary.get("texbat_attack_recordings_accessed") != []
    ):
        raise ValueError("forbidden data-access declaration")

    replication_result = documents["replication_result"]
    if (
        replication_result["data_boundary"]["allowed_pair_ids"]
        != EXPECTED_PAIR_IDS
        or replication_result["primary_pair_block_evaluation"][
            "all_support_gates_passed"
        ] is not True
    ):
        raise ValueError("frozen replication result is not supported")
    if documents["replication_artifact_summary"] != replication_result:
        raise ValueError("tracked and artifact replication summaries differ")

    sources = config["normal_sources"]
    if list(sources) != EXPECTED_PAIR_IDS:
        raise ValueError("normal source roster drifted")
    source_records: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        pair_id = pair["paired_group_id"]
        record = sources[pair_id]
        manifest_path = _repo_path(record["manifest_path"])
        los_path = _repo_path(record["los_log_path"])
        if _sha256(manifest_path) != record["manifest_sha256"]:
            raise ValueError(f"normal manifest hash mismatch: {pair_id}")
        if _sha256(los_path) != record["los_log_sha256"]:
            raise ValueError(f"LOS log hash mismatch: {pair_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenario = manifest["scenario"]
        if {
            "paired_group_id": scenario.get("paired_group_id"),
            "split": scenario.get("split"),
            "class": scenario.get("class"),
            "is_spoofing": scenario.get("is_spoofing"),
        } != {
            "paired_group_id": pair_id,
            "split": "train",
            "class": "normal",
            "is_spoofing": False,
        }:
            raise ValueError(f"normal manifest role mismatch: {pair_id}")
        iq_path, iq_sha256, iq_bytes = _normal_iq_record(manifest)
        if (
            iq_sha256 != record["iq_sha256"]
            or iq_bytes != int(record["iq_bytes"])
            or not iq_path.is_file()
            or iq_path.stat().st_size != iq_bytes
        ):
            raise ValueError(f"normal IQ contract mismatch: {pair_id}")
        if verify_iq and _sha256(iq_path) != iq_sha256:
            raise ValueError(f"normal IQ content mismatch: {pair_id}")
        source_records[pair_id] = {
            "manifest_path": manifest_path,
            "manifest": manifest,
            "iq_path": iq_path,
            "los_path": los_path,
        }

    receiver = config["gnss_sdr"]
    executable = _repo_path(receiver["executable"]["path"])
    patch = _repo_path(receiver["patch"]["path"])
    if _sha256(executable) != receiver["executable"]["sha256"]:
        raise ValueError("GNSS-SDR executable hash mismatch")
    if _sha256(patch) != receiver["patch"]["sha256"]:
        raise ValueError("GNSS-SDR patch hash mismatch")
    if {
        "channel_count": receiver.get("channel_count"),
        "tracking_tap_count": receiver.get("tracking_tap_count"),
        "tracking_tap_spacing_chips": receiver.get(
            "tracking_tap_spacing_chips"
        ),
    } != {
        "channel_count": 11,
        "tracking_tap_count": 9,
        "tracking_tap_spacing_chips": 0.125,
    }:
        raise ValueError("receiver configuration drifted")

    analysis = config["analysis"]
    if {
        "bin_seconds": analysis.get("bin_seconds"),
        "minimum_prns": analysis.get("minimum_prns"),
        "normal_warmup_seconds": analysis.get("normal_warmup_seconds"),
        "minimum_normal_bins_per_pair": analysis.get(
            "minimum_normal_bins_per_pair"
        ),
        "minimum_multipath_and_spoof_bins_per_pair": analysis.get(
            "minimum_multipath_and_spoof_bins_per_pair"
        ),
        "required_fit_rank": analysis.get("required_fit_rank"),
    } != {
        "bin_seconds": 1.0,
        "minimum_prns": 8,
        "normal_warmup_seconds": 1.0,
        "minimum_normal_bins_per_pair": 20,
        "minimum_multipath_and_spoof_bins_per_pair": 5,
        "required_fit_rank": 4,
    }:
        raise ValueError("analysis contract drifted")

    threshold = config["threshold_selection"]
    if (
        threshold.get("benign_scenarios") != list(BENIGN_SCENARIOS)
        or threshold.get("positive_scenario") != "carryoff_spoof"
        or threshold.get("persistence_consecutive_bins") != 2
        or threshold.get("final_threshold_rule") != (
            "median of the five leave-one-pair-out training thresholds"
        )
        or threshold.get("refitting_after_audit") is not False
    ):
        raise ValueError("threshold-selection contract drifted")
    if config.get("support_gates") != EXPECTED_GATES:
        raise ValueError("support gates drifted")

    output_root = _repo_path(config["output_root"])
    if output_root.parent != REPO_ROOT / "artifacts":
        raise ValueError("normal audit output must stay under artifacts")
    return {
        "split_path": split_path,
        "pairs": pairs,
        "sources": source_records,
        "pinned": pinned,
        "documents": documents,
        "receiver_executable": executable,
        "output_root": output_root,
    }


def _receiver_config(config: dict[str, Any]) -> dict[str, Any]:
    receiver = config["gnss_sdr"]
    return {
        "executable": receiver["executable"]["path"],
        "channel_count": receiver["channel_count"],
        "timeout_seconds": receiver["timeout_seconds"],
        "tracking_tap_spacing_chips": receiver[
            "tracking_tap_spacing_chips"
        ],
    }


def _runtime_path(output_root: Path, pair_id: str) -> Path:
    return output_root / "pairs" / pair_id / "runtime_manifest.json"


def _load_runtime(
    config_path: Path,
    output_root: Path,
    pair: dict[str, Any],
) -> dict[str, Any]:
    pair_id = pair["paired_group_id"]
    path = _runtime_path(output_root, pair_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("config_sha256") != _sha256(config_path)
        or document.get("pair") != pair
    ):
        raise ValueError(f"runtime contract mismatch: {pair_id}")
    for name in ("normal_manifest", "normal_iq", "los_log", "receiver_manifest"):
        record = document[name]
        source = Path(record["path"])
        if (
            not source.is_file()
            or source.stat().st_size != int(record["bytes"])
            or _sha256(source) != record["sha256"]
        ):
            raise ValueError(f"runtime provenance mismatch: {pair_id} {name}")
    return document


def process_pair(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
    pair: dict[str, Any],
    *,
    resume: bool,
) -> Path:
    pair_id = pair["paired_group_id"]
    runtime_path = _runtime_path(context["output_root"], pair_id)
    if runtime_path.is_file():
        if not resume:
            raise FileExistsError(runtime_path)
        runtime = _load_runtime(
            config_path, context["output_root"], pair
        )
        return Path(runtime["receiver_manifest"]["path"])

    source = context["sources"][pair_id]
    record = config["normal_sources"][pair_id]
    iq_path = source["iq_path"]
    if _sha256(iq_path) != record["iq_sha256"]:
        raise ValueError(f"normal IQ content mismatch: {pair_id}")

    receiver_manifest = pilot._ensure_receiver(
        source["manifest_path"],
        context["output_root"] / "receiver",
        _receiver_config(config),
        resume=resume,
    )
    receiver_document = json.loads(
        receiver_manifest.read_text(encoding="utf-8")
    )
    if (
        receiver_document["source"]["rf_manifest_sha256"]
        != record["manifest_sha256"]
        or receiver_document["receiver"]["executable_sha256"]
        != config["gnss_sdr"]["executable"]["sha256"]
    ):
        raise ValueError(f"normal receiver provenance mismatch: {pair_id}")

    def file_record(path: Path) -> dict[str, Any]:
        return {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    runtime = {
        "schema": "gnss-doppler-lab.cgc-normal-audit-runtime",
        "schema_version": 1,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "pair": pair,
        "normal_manifest": file_record(source["manifest_path"]),
        "normal_iq": file_record(iq_path),
        "los_log": file_record(source["los_path"]),
        "receiver_manifest": file_record(receiver_manifest),
    }
    _write_json(runtime_path, runtime)
    return receiver_manifest


def _augment_normal_geometry(
    delay_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    *,
    duration_seconds: float,
    warmup_seconds: float,
    epsilon: float,
) -> None:
    by_bin: dict[int, list[float]] = {}
    for row in delay_rows:
        by_bin.setdefault(int(row["bin_index"]), []).append(
            float(row["estimated_delay_chips"])
        )
    for row in geometry_rows:
        values = np.asarray(
            by_bin[int(row["bin_index"])], dtype=np.float64
        )
        centered = values - float(np.mean(values))
        energy = float(np.sum(centered * centered))
        start = float(row["bin_start_s"])
        row["clock_centered_energy_chips2"] = energy
        row["clock_energy_degenerate"] = int(energy <= epsilon)
        row["normal_eligible"] = int(
            start >= warmup_seconds and start + 1.0 <= duration_seconds
        )
        row["score"] = -float(row["clock_centered_geometry_residual"])


def _read_replication_rows(
    path: Path,
    pairs: list[dict[str, Any]],
    *,
    minimum_bins: int,
) -> list[dict[str, Any]]:
    pair_by_id = {pair["paired_group_id"]: pair for pair in pairs}
    selected: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pair_id = row["pair_id"]
            if pair_id not in pair_by_id:
                raise ValueError(f"unexpected replication pair: {pair_id}")
            if row["scenario"] not in (
                "independent_multipath", "carryoff_spoof"
            ):
                raise ValueError("unexpected replication scenario")
            start = replication.comparison_start_seconds(
                pair_by_id[pair_id]
            )
            if float(row["bin_start_s"]) < start:
                continue
            converted = dict(row)
            converted["bin_index"] = int(row["bin_index"])
            converted["bin_start_s"] = float(row["bin_start_s"])
            converted["prn_count"] = int(row["prn_count"])
            converted["fit_rank"] = int(row["fit_rank"])
            converted["clock_centered_geometry_residual"] = float(
                row["clock_centered_geometry_residual"]
            )
            converted["score"] = -converted[
                "clock_centered_geometry_residual"
            ]
            converted["source"] = "frozen_train_replication"
            selected.append(converted)
    for pair_id in EXPECTED_PAIR_IDS:
        for scenario in ("independent_multipath", "carryoff_spoof"):
            count = sum(
                row["pair_id"] == pair_id and row["scenario"] == scenario
                for row in selected
            )
            if count < minimum_bins:
                raise ValueError(
                    f"too few frozen replication bins: {pair_id} {scenario}"
                )
    return selected


def _stream_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    streams: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["pair_id"]), str(row["scenario"]))
        streams.setdefault(key, []).append(row)
    for stream in streams.values():
        stream.sort(key=lambda row: int(row["bin_index"]))
    return streams


def persistent_alarm(
    rows: Iterable[dict[str, Any]],
    threshold: float,
    consecutive: int,
) -> bool:
    run = 0
    previous: int | None = None
    for row in sorted(rows, key=lambda item: int(item["bin_index"])):
        index = int(row["bin_index"])
        alarm = float(row["score"]) > threshold
        if alarm and previous is not None and index == previous + 1:
            run += 1
        elif alarm:
            run = 1
        else:
            run = 0
        previous = index
        if run >= consecutive:
            return True
    return False


def threshold_metrics(
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    streams = _stream_rows(rows)
    expected = {
        (pair_id, scenario)
        for pair_id in sorted({str(row["pair_id"]) for row in rows})
        for scenario in SCENARIOS
    }
    if set(streams) != expected:
        raise ValueError("threshold rows do not contain three complete streams")
    rates: dict[str, float] = {}
    benign_rates: list[float] = []
    spoof_rates: list[float] = []
    for (pair_id, scenario), stream in sorted(streams.items()):
        rate = float(np.mean([
            float(row["score"]) > threshold for row in stream
        ]))
        rates[f"{pair_id}:{scenario}"] = rate
        if scenario in BENIGN_SCENARIOS:
            benign_rates.append(rate)
        else:
            spoof_rates.append(rate)
    benign_fpr = float(np.mean(benign_rates))
    spoof_tpr = float(np.mean(spoof_rates))
    return {
        "threshold": float(threshold),
        "macro_benign_false_positive_rate": benign_fpr,
        "macro_spoof_true_positive_rate": spoof_tpr,
        "macro_balanced_accuracy": float(
            ((1.0 - benign_fpr) + spoof_tpr) / 2.0
        ),
        "stream_alarm_rates": rates,
    }


def threshold_candidates(rows: list[dict[str, Any]]) -> list[float]:
    values = sorted({float(row["score"]) for row in rows})
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("threshold scores must be finite")
    candidates = [float(np.nextafter(values[0], -np.inf))]
    candidates.extend(
        float((left + right) / 2.0)
        for left, right in zip(values, values[1:])
    )
    candidates.append(float(np.nextafter(values[-1], np.inf)))
    return candidates


def fit_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [
        threshold_metrics(rows, threshold)
        for threshold in threshold_candidates(rows)
    ]
    best = max(
        evaluated,
        key=lambda result: (
            result["macro_balanced_accuracy"],
            -result["macro_benign_false_positive_rate"],
            result["threshold"],
        ),
    )
    return {
        **best,
        "candidate_count": len(evaluated),
        "selection_objective": (
            "max macro balanced accuracy; min benign FPR; max threshold"
        ),
    }


def lopo_evaluate(
    rows: list[dict[str, Any]],
    *,
    pair_ids: list[str],
    consecutive: int,
    gates: dict[str, Any],
    finite_fraction: float,
    full_rank_fraction: float,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    all_heldout_rows: list[dict[str, Any]] = []
    for heldout in pair_ids:
        training = [row for row in rows if row["pair_id"] != heldout]
        heldout_rows = [row for row in rows if row["pair_id"] == heldout]
        fitted = fit_threshold(training)
        metrics = threshold_metrics(heldout_rows, fitted["threshold"])
        streams = _stream_rows(heldout_rows)
        persistent = {
            scenario: persistent_alarm(
                streams[(heldout, scenario)],
                fitted["threshold"],
                consecutive,
            )
            for scenario in SCENARIOS
        }
        fold = {
            "heldout_pair_id": heldout,
            "training_pair_ids": [
                pair_id for pair_id in pair_ids if pair_id != heldout
            ],
            "training_fit": fitted,
            "heldout_metrics": metrics,
            "heldout_persistent_alarm": persistent,
        }
        folds.append(fold)
        for row in heldout_rows:
            scored = dict(row)
            scored["fold_threshold"] = fitted["threshold"]
            scored["fold_alarm"] = int(
                float(row["score"]) > fitted["threshold"]
            )
            all_heldout_rows.append(scored)

    normal_persistent = sum(
        fold["heldout_persistent_alarm"]["normal"] for fold in folds
    )
    multipath_persistent = sum(
        fold["heldout_persistent_alarm"]["independent_multipath"]
        for fold in folds
    )
    spoof_persistent = sum(
        fold["heldout_persistent_alarm"]["carryoff_spoof"]
        for fold in folds
    )
    normal_rates = [
        fold["heldout_metrics"]["stream_alarm_rates"][
            f"{fold['heldout_pair_id']}:normal"
        ]
        for fold in folds
    ]
    multipath_rates = [
        fold["heldout_metrics"]["stream_alarm_rates"][
            f"{fold['heldout_pair_id']}:independent_multipath"
        ]
        for fold in folds
    ]
    spoof_rates = [
        fold["heldout_metrics"]["stream_alarm_rates"][
            f"{fold['heldout_pair_id']}:carryoff_spoof"
        ]
        for fold in folds
    ]
    benign_fpr = float(np.mean(normal_rates + multipath_rates))
    spoof_tpr = float(np.mean(spoof_rates))
    balanced_accuracy = float(
        ((1.0 - benign_fpr) + spoof_tpr) / 2.0
    )
    final_threshold = float(np.median([
        fold["training_fit"]["threshold"] for fold in folds
    ]))
    final_descriptive = threshold_metrics(rows, final_threshold)

    gate_records = {
        "required_pair_count": {
            "observed": len(pair_ids),
            "required": gates["required_pair_count"],
            "passed": len(pair_ids) == gates["required_pair_count"],
        },
        "minimum_finite_score_fraction": {
            "observed": finite_fraction,
            "required": gates["minimum_finite_score_fraction"],
            "passed": finite_fraction
            >= gates["minimum_finite_score_fraction"],
        },
        "minimum_full_rank_fraction": {
            "observed": full_rank_fraction,
            "required": gates["minimum_full_rank_fraction"],
            "passed": full_rank_fraction
            >= gates["minimum_full_rank_fraction"],
        },
        "maximum_lopo_normal_persistent_alarm_pair_count": {
            "observed": normal_persistent,
            "required": gates[
                "maximum_lopo_normal_persistent_alarm_pair_count"
            ],
            "passed": normal_persistent
            <= gates["maximum_lopo_normal_persistent_alarm_pair_count"],
        },
        "maximum_lopo_multipath_persistent_alarm_pair_count": {
            "observed": multipath_persistent,
            "required": gates[
                "maximum_lopo_multipath_persistent_alarm_pair_count"
            ],
            "passed": multipath_persistent
            <= gates[
                "maximum_lopo_multipath_persistent_alarm_pair_count"
            ],
        },
        "minimum_lopo_spoof_persistent_detection_pair_count": {
            "observed": spoof_persistent,
            "required": gates[
                "minimum_lopo_spoof_persistent_detection_pair_count"
            ],
            "passed": spoof_persistent
            >= gates[
                "minimum_lopo_spoof_persistent_detection_pair_count"
            ],
        },
        "minimum_lopo_macro_balanced_accuracy": {
            "observed": balanced_accuracy,
            "required": gates["minimum_lopo_macro_balanced_accuracy"],
            "passed": balanced_accuracy
            >= gates["minimum_lopo_macro_balanced_accuracy"],
        },
    }
    passed = all(record["passed"] for record in gate_records.values())
    return {
        "folds": folds,
        "cross_validated_macro_normal_false_positive_rate": float(
            np.mean(normal_rates)
        ),
        "cross_validated_macro_multipath_false_positive_rate": float(
            np.mean(multipath_rates)
        ),
        "cross_validated_macro_benign_false_positive_rate": benign_fpr,
        "cross_validated_macro_spoof_true_positive_rate": spoof_tpr,
        "cross_validated_macro_balanced_accuracy": balanced_accuracy,
        "normal_persistent_alarm_pair_count": normal_persistent,
        "multipath_persistent_alarm_pair_count": multipath_persistent,
        "spoof_persistent_detection_pair_count": spoof_persistent,
        "final_threshold": final_threshold,
        "final_train_descriptive_metrics": final_descriptive,
        "gates": gate_records,
        "all_support_gates_passed": passed,
        "status": (
            "train_detector_freeze_supported_requires_locked_test"
            if passed
            else "train_detector_freeze_not_supported"
        ),
        "heldout_scored_rows": all_heldout_rows,
    }


def analyze(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    output_root = context["output_root"]
    analysis_root = output_root / "analysis"
    if (output_root / "summary.json").exists():
        raise FileExistsError(output_root / "summary.json")

    estimator = pilot._estimator(
        context["documents"]["controlled_template_config"]
    )
    normal_delay_rows: list[dict[str, Any]] = []
    normal_geometry_rows: list[dict[str, Any]] = []
    normal_pair_summaries: list[dict[str, Any]] = []
    analysis_config = config["analysis"]

    for pair in context["pairs"]:
        pair_id = pair["paired_group_id"]
        runtime = _load_runtime(config_path, output_root, pair)
        receiver_manifest = Path(runtime["receiver_manifest"]["path"])
        source = context["sources"][pair_id]
        los = parse_gps_sdr_sim_los_table(
            source["los_path"].read_text(encoding="utf-8")
        )
        delays, geometry = pilot._scenario_geometry(
            "normal",
            receiver_manifest,
            estimator,
            los,
            bin_seconds=float(analysis_config["bin_seconds"]),
            minimum_prns=int(analysis_config["minimum_prns"]),
        )
        duration = float(source["manifest"]["iq"]["actual_duration_seconds"])
        _augment_normal_geometry(
            delays,
            geometry,
            duration_seconds=duration,
            warmup_seconds=float(
                analysis_config["normal_warmup_seconds"]
            ),
            epsilon=float(
                config["frozen_candidate"]["clock_energy_epsilon"]
            ),
        )
        for row in delays:
            row["pair_id"] = pair_id
            row["domain"] = pair["domain"]
        for row in geometry:
            row["pair_id"] = pair_id
            row["domain"] = pair["domain"]
            row["source"] = "normal_nine_tap_audit"
        eligible = [
            row for row in geometry if int(row["normal_eligible"]) == 1
        ]
        if len(eligible) < int(
            analysis_config["minimum_normal_bins_per_pair"]
        ):
            raise ValueError(f"too few eligible normal bins: {pair_id}")
        normal_delay_rows.extend(delays)
        normal_geometry_rows.extend(geometry)
        energies = np.asarray([
            row["clock_centered_energy_chips2"] for row in eligible
        ], dtype=np.float64)
        residuals = np.asarray([
            row["clock_centered_geometry_residual"] for row in eligible
        ], dtype=np.float64)
        normal_pair_summaries.append({
            "pair_id": pair_id,
            "domain": pair["domain"],
            "eligible_bin_count": len(eligible),
            "median_clock_centered_residual": float(np.median(residuals)),
            "median_score": float(np.median(-residuals)),
            "minimum_clock_centered_energy_chips2": float(np.min(energies)),
            "median_clock_centered_energy_chips2": float(np.median(energies)),
            "degenerate_clock_energy_bin_count": int(np.count_nonzero(
                energies <= float(
                    config["frozen_candidate"]["clock_energy_epsilon"]
                )
            )),
            "minimum_prn_count": min(
                int(row["prn_count"]) for row in eligible
            ),
            "full_rank_bin_count": sum(
                int(row["fit_rank"])
                == int(analysis_config["required_fit_rank"])
                for row in eligible
            ),
            "runtime_manifest_sha256": _sha256(
                _runtime_path(output_root, pair_id)
            ),
        })

    replication_rows = _read_replication_rows(
        context["pinned"]["replication_geometry_scores"],
        context["pairs"],
        minimum_bins=int(
            analysis_config["minimum_multipath_and_spoof_bins_per_pair"]
        ),
    )
    selected_normal = [
        row for row in normal_geometry_rows
        if int(row["normal_eligible"]) == 1
    ]
    threshold_rows = selected_normal + replication_rows
    expected_streams = len(EXPECTED_PAIR_IDS) * len(SCENARIOS)
    if len(_stream_rows(threshold_rows)) != expected_streams:
        raise ValueError("threshold stream roster is incomplete")
    scores = np.asarray([
        float(row["score"]) for row in threshold_rows
    ], dtype=np.float64)
    finite_fraction = float(np.mean(np.isfinite(scores)))
    required_rank = int(analysis_config["required_fit_rank"])
    full_rank_fraction = float(np.mean([
        int(row["fit_rank"]) == required_rank for row in threshold_rows
    ]))
    evaluation = lopo_evaluate(
        threshold_rows,
        pair_ids=EXPECTED_PAIR_IDS,
        consecutive=int(
            config["threshold_selection"]["persistence_consecutive_bins"]
        ),
        gates=config["support_gates"],
        finite_fraction=finite_fraction,
        full_rank_fraction=full_rank_fraction,
    )
    heldout_rows = evaluation.pop("heldout_scored_rows")

    normal_delay_path = analysis_root / "normal_delay_estimates.csv"
    normal_geometry_path = analysis_root / "normal_geometry_scores.csv"
    threshold_rows_path = analysis_root / "threshold_input_rows.csv"
    lopo_rows_path = analysis_root / "lopo_scored_rows.csv"
    pair_summary_path = analysis_root / "normal_pair_summary.csv"
    _write_csv(normal_delay_path, normal_delay_rows)
    _write_csv(normal_geometry_path, normal_geometry_rows)
    _write_csv(threshold_rows_path, threshold_rows)
    _write_csv(lopo_rows_path, heldout_rows)
    _write_csv(pair_summary_path, normal_pair_summaries)

    threshold_artifact = {
        "schema": "gnss-doppler-lab.cgc-detector-threshold-freeze",
        "schema_version": 1,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "score": "negative clock-centered residual",
        "alarm_comparison": "score > threshold",
        "threshold": evaluation["final_threshold"],
        "persistence_consecutive_bins": int(
            config["threshold_selection"]["persistence_consecutive_bins"]
        ),
        "training_pair_ids": EXPECTED_PAIR_IDS,
        "selection": config["threshold_selection"],
        "lopo_support_status": evaluation["status"],
        "usable_for_locked_test": evaluation[
            "all_support_gates_passed"
        ],
        "refitting_permitted": False,
        "claim_boundary": config["claim_boundary"],
    }
    threshold_path = output_root / "threshold_freeze.json"
    _write_json(threshold_path, threshold_artifact)

    def artifact(path: Path, row_count: int | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        if row_count is not None:
            record["row_count"] = row_count
        return record

    summary = {
        "schema": "gnss-doppler-lab.cgc-normal-detector-freeze-audit-result",
        "schema_version": 1,
        "role": config["experiment"]["role"],
        "claim_boundary": config["claim_boundary"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "data_boundary": config["data_boundary"],
        "frozen_candidate": config["frozen_candidate"],
        "normal_pairs": normal_pair_summaries,
        "normal_clock_energy": {
            "total_eligible_bin_count": len(selected_normal),
            "degenerate_bin_count": sum(
                int(row["clock_energy_degenerate"])
                for row in selected_normal
            ),
            "finite_score_fraction": float(np.mean(np.isfinite([
                float(row["score"]) for row in selected_normal
            ]))),
        },
        "threshold_cross_validation": evaluation,
        "threshold_freeze": artifact(threshold_path),
        "detector_freeze_supported": evaluation[
            "all_support_gates_passed"
        ],
        "next_gate": (
            "if supported, preregister test pairs 010--012 with this exact "
            "threshold and persistence before any test signal access"
        ),
        "artifacts": {
            "normal_delay_estimates": artifact(
                normal_delay_path, len(normal_delay_rows)
            ),
            "normal_geometry_scores": artifact(
                normal_geometry_path, len(normal_geometry_rows)
            ),
            "threshold_input_rows": artifact(
                threshold_rows_path, len(threshold_rows)
            ),
            "lopo_scored_rows": artifact(
                lopo_rows_path, len(heldout_rows)
            ),
            "normal_pair_summary": artifact(
                pair_summary_path, len(normal_pair_summaries)
            ),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pair-id", choices=EXPECTED_PAIR_IDS)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if sum((args.generate_only, args.analysis_only, args.verify_only)) > 1:
        parser.error("choose at most one mode")
    if args.pair_id and not args.generate_only:
        parser.error("--pair-id requires --generate-only")

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    context = validate_config(config, verify_iq=args.verify_only)
    if args.verify_only:
        print("normal-audit config and all authorized inputs verified")
        return 0

    selected = context["pairs"]
    if args.pair_id:
        selected = [
            pair for pair in selected
            if pair["paired_group_id"] == args.pair_id
        ]
    if not args.analysis_only:
        for pair in selected:
            print(f"[receive] {pair['paired_group_id']} normal RF", flush=True)
            process_pair(
                config, config_path, context, pair, resume=args.resume
            )
    if args.generate_only:
        return 0

    print("[analyze] normal stability and LOPO detector freeze", flush=True)
    summary = analyze(config, config_path, context)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
