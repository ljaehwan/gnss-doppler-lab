#!/usr/bin/env python3
"""Release and analyze the sealed CGC receiver-RF test exactly once."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import plan_simulation_v4_paired_split as splitter  # noqa: E402
import run_cgc_rf_challenge_pilot as pilot  # noqa: E402
import run_cgc_rf_train_replication as replication  # noqa: E402
import run_simulation_v4_paired_train_generation as source  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import (  # noqa: E402
    parse_gps_sdr_sim_los_table,
)


DEFAULT_CONFIG = (
    REPO_ROOT / "configs/experiments/cgc_rf_locked_test_v1.json"
)
PROTOCOL_PATH = (
    REPO_ROOT / "docs/results/cgc_rf_locked_test_protocol_v1.md"
)
EXPECTED_CONFIG_SHA256 = (
    "f3792b00993d40b028218ae0024d855a294f6e71cc0f12caab5c268bd5404c95"
)
EXPECTED_PROTOCOL_SHA256 = (
    "be93580ecd60d90f8758ea833be0cff2ca4a11aeb609d678b5388976cd3ae655"
)
EXPECTED_PAIR_IDS = [
    "pv1-pair-010",
    "pv1-pair-011",
    "pv1-pair-012",
]
EXPECTED_GATES = {
    "required_pair_count": 3,
    "positive_clock_centered_separation_pair_count": 3,
    "minimum_pair_block_auc": 0.80,
    "minimum_clock_centered_improvement_over_legacy_pair_count": 2,
    "minimum_comparison_bins_per_scenario_per_pair": 5,
    "minimum_startup_los_prns_per_pair": 8,
}
RELEASE_TOKEN = "RELEASE-CGC-RF-LOCKED-TEST-V1"
SOURCE_CAMPAIGN = "simulation-v4-paired-spoof-locked-test-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(
            handle, fieldnames=fields + extras, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def _load_pinned(
    record: dict[str, str], label: str
) -> tuple[Path, dict[str, Any]]:
    path = _repo_path(record["path"])
    observed = _sha256(path)
    if observed != record["sha256"]:
        raise ValueError(
            f"{label} hash mismatch: expected {record['sha256']}, observed {observed}"
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _verify_all_path_hash_records(document: Any) -> int:
    count = 0
    if isinstance(document, dict):
        if isinstance(document.get("path"), str) and isinstance(
            document.get("sha256"), str
        ):
            path = _repo_path(document["path"])
            if _sha256(path) != document["sha256"]:
                raise ValueError(f"pinned path hash mismatch: {path}")
            count += 1
        for value in document.values():
            count += _verify_all_path_hash_records(value)
    elif isinstance(document, list):
        for value in document:
            count += _verify_all_path_hash_records(value)
    return count


def validate_config(
    config: dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
    enforce_config_hash: bool = True,
) -> dict[str, Any]:
    if enforce_config_hash and _sha256(config_path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("sealed locked-test config hash drifted")
    if _sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("sealed locked-test protocol hash drifted")
    if config.get("schema") != "gnss-doppler-lab.cgc-rf-locked-test-config":
        raise ValueError("unsupported locked-test config schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported locked-test config version")

    experiment = config.get("experiment", {})
    if experiment.get("name") != "cgc-rf-locked-test-v1":
        raise ValueError("locked-test experiment name drifted")
    if experiment.get("candidate_commit") != (
        "60280575737ff70618c5619ff7522c516bbdc67a"
    ):
        raise ValueError("candidate commit drifted")
    if experiment.get("runner_path") != "scripts/run_cgc_rf_locked_test.py":
        raise ValueError("locked-test runner path drifted")
    if "one deterministic release and one analysis" not in str(
        experiment.get("execution_policy")
    ):
        raise ValueError("single-release execution policy drifted")

    frozen = config.get("frozen_candidate", {})
    if frozen.get("residual") != (
        "SSE_full / sum(weight * (delay - weighted_mean(delay))^2)"
    ):
        raise ValueError("clock-centered residual law drifted")
    if frozen.get("detection_score") != "negative residual":
        raise ValueError("score direction drifted")
    if frozen.get("absolute_threshold_applied") is not False:
        raise ValueError("absolute threshold is forbidden")
    if frozen.get("threshold_or_calibration_fitting") is not False:
        raise ValueError("test calibration fitting is forbidden")

    boundary = config.get("data_boundary", {})
    if boundary.get("authorized_partition") != "test":
        raise ValueError("locked release must remain test-only")
    if boundary.get("allowed_pair_ids") != EXPECTED_PAIR_IDS:
        raise ValueError("authorized test pair roster drifted")
    if boundary.get("pre_freeze_test_signal_artifacts_accessed") is not False:
        raise ValueError("pre-freeze access declaration drifted")
    if boundary.get("test_refit_or_adaptation_forbidden") is not True:
        raise ValueError("test adaptation must remain forbidden")
    if boundary.get("texbat_recordings_accessed") != []:
        raise ValueError("TEXBAT access is outside this locked test")

    pinned_count = _verify_all_path_hash_records(config)
    if pinned_count != 15:
        raise ValueError(f"expected 15 pinned path hashes, observed {pinned_count}")

    split_path, split = _load_pinned(
        config["source_generation"]["split_config"], "split config"
    )
    split_record_path, split_record = _load_pinned(
        config["source_generation"]["split_record"], "split record"
    )
    normal_path, normal_profile = _load_pinned(
        config["source_generation"]["normal_profile"], "normal profile"
    )
    _, controlled = _load_pinned(
        config["analysis"]["controlled_template_config"],
        "controlled template config",
    )
    splitter.validate_config(split)
    if split_record.get("config", {}).get("sha256") != _sha256(split_path):
        raise ValueError("split record does not pin the selected split")
    if split_record.get("source_split_manifest", {}).get("sha256") != (
        config["source_generation"]["canonical_split_manifest_sha256"]
    ):
        raise ValueError("canonical split manifest pin drifted")
    if split_record.get("test_release", {}).get("status") != "locked":
        raise ValueError("source split record did not leave test locked")

    pairs = [
        pair
        for pair in split["pairs"]
        if pair["paired_group_id"] in EXPECTED_PAIR_IDS
    ]
    if [pair["paired_group_id"] for pair in pairs] != EXPECTED_PAIR_IDS:
        raise ValueError("split config does not contain the frozen test order")
    if any(pair.get("split") != "test" for pair in pairs):
        raise ValueError("non-test pair entered the locked roster")
    if [
        "static" if pair["domain"] == "static" else pair["motion"]["kind"]
        for pair in pairs
    ] != ["static", "straight", "parallel-sweep"]:
        raise ValueError("locked motion roster drifted")

    profile = split["fixed_rf_profile"]
    if {
        "rf_sample_rate_hz": int(
            normal_profile["rf_profile"]["rf_sample_rate_hz"]
        ),
        "frontend_cutoff_hz": float(
            normal_profile["rf_profile"]["frontend_cutoff_hz"]
        ),
        "frontend_order": int(normal_profile["receiver"]["frontend_order"]),
        "normal_target_rms": float(normal_profile["normal_target_rms"]),
        "tracking_tap_count": int(
            normal_profile["gnss_sdr"]["tracking_tap_count"]
        ),
        "tracking_tap_spacing_chips": float(
            normal_profile["gnss_sdr"]["tracking_tap_spacing_chips"]
        ),
    } != {
        "rf_sample_rate_hz": int(profile["rf_sample_rate_hz"]),
        "frontend_cutoff_hz": float(profile["frontend_cutoff_hz"]),
        "frontend_order": int(profile["frontend_order"]),
        "normal_target_rms": float(profile["normal_target_rms"]),
        "tracking_tap_count": int(profile["tracking_tap_count"]),
        "tracking_tap_spacing_chips": float(
            profile["tracking_tap_spacing_chips"]
        ),
    }:
        raise ValueError("source normal profile differs from frozen RF profile")

    multipath = config.get("multipath", {})
    if list(multipath.get("seed_by_pair", {})) != EXPECTED_PAIR_IDS:
        raise ValueError("multipath seed roster drifted")
    if multipath.get("seed_by_pair") != {
        "pv1-pair-010": 2026091310,
        "pv1-pair-011": 2026091311,
        "pv1-pair-012": 2026091312,
    }:
        raise ValueError("multipath seeds drifted")
    if multipath.get("delay_chips_range") != [0.12, 0.45]:
        raise ValueError("multipath delay range drifted")
    if multipath.get("amplitude_range") != [0.20, 0.70]:
        raise ValueError("multipath amplitude range drifted")

    receiver = config.get("gnss_sdr", {})
    if {
        "channel_count": receiver.get("channel_count"),
        "timeout_seconds": receiver.get("timeout_seconds"),
        "tracking_tap_count": receiver.get("tracking_tap_count"),
        "tracking_tap_spacing_chips": receiver.get(
            "tracking_tap_spacing_chips"
        ),
    } != {
        "channel_count": 11,
        "timeout_seconds": 1200,
        "tracking_tap_count": 9,
        "tracking_tap_spacing_chips": 0.125,
    }:
        raise ValueError("GNSS-SDR contract drifted")

    analysis = config.get("analysis", {})
    if analysis.get("bin_seconds") != 1.0:
        raise ValueError("analysis bin width drifted")
    if analysis.get("minimum_prns") != 8:
        raise ValueError("analysis minimum PRNs drifted")
    if analysis.get("comparison_start_policy") != (
        "spoof start + max(transition, power ramp) + 1 second"
    ):
        raise ValueError("comparison boundary drifted")

    evaluation = config.get("evaluation", {})
    if evaluation.get("support_gates") != EXPECTED_GATES:
        raise ValueError("support gates drifted")
    if evaluation.get("bootstrap_seed") != 2026091399:
        raise ValueError("bootstrap seed drifted")
    if evaluation.get("bootstrap_repetitions") != 10000:
        raise ValueError("bootstrap repetitions drifted")
    if evaluation.get("threshold_fitting") is not False:
        raise ValueError("threshold fitting is forbidden")
    if evaluation.get("post_release_tuning_or_retest") is not False:
        raise ValueError("post-release tuning/retest must remain forbidden")

    source_root = _repo_path(config["source_generation"]["output_root"])
    output_root = _repo_path(config["output_root"])
    if source_root != (
        REPO_ROOT / "artifacts/simulation_v4_paired_test_generation_v1"
    ):
        raise ValueError("source output root drifted")
    if output_root != REPO_ROOT / "artifacts/cgc_rf_locked_test_v1":
        raise ValueError("analysis output root drifted")

    return {
        "split_path": split_path,
        "split_record_path": split_record_path,
        "split": split,
        "pairs": pairs,
        "normal_path": normal_path,
        "normal_profile": normal_profile,
        "controlled": controlled,
        "source_root": source_root,
        "output_root": output_root,
        "pinned_path_hash_count": pinned_count,
    }


def _source_paths(context: dict[str, Any], pair_id: str) -> dict[str, Path]:
    root = context["source_root"] / "pairs" / pair_id
    return {
        "pair_root": root,
        "normal_manifest": root / "rf/normal/manifest.json",
        "spoof_manifest": root / "rf/spoof/manifest.json",
        "component_manifest": root / "components/manifest.json",
        "los_log": root / "components/authentic-gps-sdr-sim.log",
        "pair_manifest": root / "pair_manifest.json",
    }


def _runtime_config(
    config: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    return {
        **config,
        "source_generation": {
            **config["source_generation"],
            "pair_root_template": str(
                context["source_root"] / "pairs/{pair_id}"
            ),
            "normal_manifest_relative": "rf/normal/manifest.json",
            "spoof_manifest_relative": "rf/spoof/manifest.json",
            "component_manifest_relative": "components/manifest.json",
            "los_log_relative": "components/authentic-gps-sdr-sim.log",
        },
    }


def _source_run_id(pair: dict[str, Any], member: str) -> str:
    number = str(pair["paired_group_id"]).rsplit("-", 1)[-1]
    suffix = "n" if member == "normal" else "s"
    timestamp = source.normal._run_datetime(pair).strftime("%Y%m%dT%H%M%SZ")
    return f"simv4px-p{number}-{suffix}_{timestamp}"


def _relabel_source_as_test(
    config: dict[str, Any],
    pair: dict[str, Any],
    component_path: Path,
    paired: dict[str, Any],
) -> dict[str, Any]:
    runner_record = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__).resolve()),
    }
    component = json.loads(component_path.read_text(encoding="utf-8"))
    component["scope"] = (
        "offline locked-test paired source generation only; no RF transmission"
    )
    component["orchestrator_runner"] = runner_record
    _write_json(component_path, component)

    for member, manifest_path in paired["rf_manifests"].items():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        document["run_id"] = _source_run_id(pair, member)
        document["scenario"]["split"] = "test"
        document["simulation_v4"]["scope"] = (
            "offline locked-test baseband only; no RF transmission"
        )
        document["generation"]["reference_paired_generator"] = config[
            "source_generation"
        ]["reference_paired_generator"]
        document["generation"]["orchestrator_runner"] = runner_record
        _write_json(manifest_path, document)

    pair_path = paired["pair_manifest"]
    pair_document = json.loads(pair_path.read_text(encoding="utf-8"))
    pair_document["schema"] = (
        "gnss-doppler-lab.simulation-v4-paired-locked-test-pair"
    )
    pair_document["component_manifest_sha256"] = _sha256(component_path)
    pair_document["rf_manifests"] = {
        member: {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
        for member, path in paired["rf_manifests"].items()
    }
    pair_document["orchestrator_runner"] = runner_record
    _write_json(pair_path, pair_document)
    paired["document"] = pair_document
    return paired


def _ensure_source_pair(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
    pair: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    generation_view = {"campaign": {"name": SOURCE_CAMPAIGN}}
    simulator = source.GpsSdrSimRunner(
        str(_repo_path(context["normal_profile"]["simulator"]["executable"]))
    )
    config_sha = _sha256(config_path)
    split_sha = _sha256(context["split_path"])
    canonical_sha = config["source_generation"][
        "canonical_split_manifest_sha256"
    ]
    authentic, counterfeit, component_path, component = (
        source._component_pair(
            generation_view,
            config_sha,
            split_sha,
            canonical_sha,
            context["normal_profile"],
            context["source_root"],
            pair,
            simulator,
            resume=resume,
        )
    )
    component["scope"] = (
        "offline locked-test paired source generation only; no RF transmission"
    )
    component["orchestrator_runner"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__).resolve()),
    }
    _write_json(component_path, component)
    paired = source._compose_pair(
        generation_view,
        config_sha,
        split_sha,
        canonical_sha,
        context["normal_profile"],
        context["source_root"],
        pair,
        authentic,
        counterfeit,
        component_path,
        component,
        resume=resume,
    )
    return _relabel_source_as_test(
        config, pair, component_path, paired
    )


def _write_source_summary(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
) -> Path:
    artifacts: dict[str, Any] = {}
    for pair in context["pairs"]:
        pair_id = pair["paired_group_id"]
        paths = _source_paths(context, pair_id)
        artifacts[pair_id] = {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
            if name != "pair_root"
        }
    document = {
        "schema": "gnss-doppler-lab.simulation-v4-paired-locked-test-source",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "partition": "test",
        "pair_ids": EXPECTED_PAIR_IDS,
        "pair_count": 3,
        "artifacts": artifacts,
        "normal_score_use": "forbidden",
        "scope": "source-generation provenance only; contains no CGC outcome",
    }
    path = context["source_root"] / "summary.json"
    _write_json(path, document)
    return path


def _multipath_run_id(pair: dict[str, Any]) -> str:
    number = str(pair["paired_group_id"]).rsplit("-", 1)[-1]
    utc = datetime.fromisoformat(str(pair["utc"]).replace("Z", "+00:00"))
    return (
        f"cgc-rf-test-mp-p{number}_"
        f"{utc.strftime('%Y%m%dT%H%M%SZ')}"
    )


def _ensure_test_pair(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
    pair: dict[str, Any],
    *,
    resume: bool,
) -> Path:
    pair_id = pair["paired_group_id"]
    pair_root = context["output_root"] / "pairs" / pair_id
    paths = _source_paths(context, pair_id)
    for name in (
        "normal_manifest",
        "spoof_manifest",
        "component_manifest",
        "los_log",
        "pair_manifest",
    ):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])

    runtime_config = _runtime_config(config, context)
    component, component_log, component_manifest, component_document = (
        replication._ensure_component(
            pair_root,
            runtime_config,
            config_path,
            pair,
            context["normal_profile"],
            resume=resume,
        )
    )
    component_document["schema"] = (
        "gnss-doppler-lab.cgc-rf-locked-test-component"
    )
    component_document["scope"] = (
        "held-out simulated receiver-RF test only; no RF transmission"
    )
    component_document["runner"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__).resolve()),
    }
    _write_json(component_manifest, component_document)

    rf_config = {
        "source_normal_rf_manifest": str(
            paths["normal_manifest"].resolve()
        )
    }
    _, multipath_rf_manifest, multipath_rf = pilot._ensure_rf(
        pair_root,
        rf_config,
        pair,
        context["normal_profile"],
        component,
        component_document,
        resume=resume,
    )
    multipath_rf["run_id"] = _multipath_run_id(pair)
    multipath_rf["scenario"]["split"] = "test"
    multipath_rf["scenario"]["motion"] = pair.get("motion")
    multipath_rf["simulation_v4"]["scope"] = (
        "held-out simulated receiver-RF test only; no RF transmission"
    )
    multipath_rf["generation"] = {
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }
    _write_json(multipath_rf_manifest, multipath_rf)
    replication._validate_rf_resume(
        multipath_rf_manifest,
        component_document,
        paths["normal_manifest"],
    )

    receiver_config = {
        "executable": config["gnss_sdr"]["executable"]["path"],
        "channel_count": config["gnss_sdr"]["channel_count"],
        "timeout_seconds": config["gnss_sdr"]["timeout_seconds"],
        "tracking_tap_spacing_chips": config["gnss_sdr"][
            "tracking_tap_spacing_chips"
        ],
    }
    receiver_root = pair_root / "receiver"
    multipath_receiver = pilot._ensure_receiver(
        multipath_rf_manifest,
        receiver_root,
        receiver_config,
        resume=resume,
    )
    spoof_receiver = pilot._ensure_receiver(
        paths["spoof_manifest"],
        receiver_root,
        receiver_config,
        resume=resume,
    )
    runtime = {
        "schema": "gnss-doppler-lab.cgc-rf-locked-test-runtime",
        "schema_version": 1,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "pair": pair,
        "sources": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
            if name not in {"pair_root", "pair_manifest"}
        },
        "source_pair_manifest": {
            "path": str(paths["pair_manifest"].resolve()),
            "sha256": _sha256(paths["pair_manifest"]),
        },
        "component_manifest": {
            "path": str(component_manifest.resolve()),
            "sha256": _sha256(component_manifest),
        },
        "component_log": {
            "path": str(component_log.resolve()),
            "sha256": _sha256(component_log),
        },
        "multipath_rf_manifest": {
            "path": str(multipath_rf_manifest.resolve()),
            "sha256": _sha256(multipath_rf_manifest),
        },
        "multipath_receiver_manifest": {
            "path": str(multipath_receiver.resolve()),
            "sha256": _sha256(multipath_receiver),
        },
        "spoof_receiver_manifest": {
            "path": str(spoof_receiver.resolve()),
            "sha256": _sha256(spoof_receiver),
        },
        "data_boundary": config["data_boundary"],
    }
    runtime_path = pair_root / "runtime_manifest.json"
    _write_json(runtime_path, runtime)
    return runtime_path


def comparison_start_seconds(pair: dict[str, Any]) -> float:
    return replication.comparison_start_seconds(pair)


def _analyze_pair_in_memory(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
    pair: dict[str, Any],
    estimator: Any,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    pair_id = pair["paired_group_id"]
    pair_root = context["output_root"] / "pairs" / pair_id
    runtime = replication._load_runtime(config_path, pair_root, pair)
    source_los = parse_gps_sdr_sim_los_table(
        Path(runtime["sources"]["los_log"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    component_los = parse_gps_sdr_sim_los_table(
        Path(runtime["component_log"]["path"]).read_text(encoding="utf-8")
    )
    if set(source_los) != set(component_los) or any(
        not np.allclose(
            source_los[prn], component_los[prn], rtol=0.0, atol=1e-12
        )
        for prn in source_los
    ):
        raise ValueError(f"analysis startup LOS mismatch: {pair_id}")

    receiver_records = (
        (
            "independent_multipath",
            Path(runtime["multipath_receiver_manifest"]["path"]),
        ),
        (
            "carryoff_spoof",
            Path(runtime["spoof_receiver_manifest"]["path"]),
        ),
    )
    delay_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for scenario, receiver_manifest in receiver_records:
        delays, geometry = pilot._scenario_geometry(
            scenario,
            receiver_manifest,
            estimator,
            source_los,
            bin_seconds=float(config["analysis"]["bin_seconds"]),
            minimum_prns=int(config["analysis"]["minimum_prns"]),
        )
        delay_rows.extend(
            {"pair_id": pair_id, "domain": pair["domain"], **row}
            for row in delays
        )
        geometry_rows.extend(
            {"pair_id": pair_id, "domain": pair["domain"], **row}
            for row in geometry
        )

    start = comparison_start_seconds(pair)
    for row in geometry_rows:
        row["comparison_eligible"] = int(
            float(row["bin_start_s"]) >= start
        )
    comparison = [row for row in geometry_rows if row["comparison_eligible"]]
    scenario_rows = {
        scenario: [
            row for row in comparison if row["scenario"] == scenario
        ]
        for scenario, _ in receiver_records
    }
    if any(not rows for rows in scenario_rows.values()):
        raise ValueError(f"empty comparison scenario: {pair_id}")

    def median(scenario: str, field: str) -> float:
        return float(
            np.median([float(row[field]) for row in scenario_rows[scenario]])
        )

    legacy_mp = median(
        "independent_multipath", "complex_geometry_residual"
    )
    legacy_sp = median("carryoff_spoof", "complex_geometry_residual")
    centered_mp = median(
        "independent_multipath", "clock_centered_geometry_residual"
    )
    centered_sp = median(
        "carryoff_spoof", "clock_centered_geometry_residual"
    )
    summary = {
        "pair_id": pair_id,
        "domain": pair["domain"],
        "motion_kind": (
            "static" if pair["domain"] == "static" else pair["motion"]["kind"]
        ),
        "comparison_start_seconds": start,
        "startup_los_prn_count": len(source_los),
        "multipath_comparison_bin_count": len(
            scenario_rows["independent_multipath"]
        ),
        "spoof_comparison_bin_count": len(
            scenario_rows["carryoff_spoof"]
        ),
        "legacy_multipath_median_residual": legacy_mp,
        "legacy_spoof_median_residual": legacy_sp,
        "legacy_separation": legacy_mp - legacy_sp,
        "clock_centered_multipath_median_residual": centered_mp,
        "clock_centered_spoof_median_residual": centered_sp,
        "clock_centered_separation": centered_mp - centered_sp,
        "clock_centered_improvement_over_legacy": (
            centered_mp - centered_sp - (legacy_mp - legacy_sp)
        ),
        "direction_supported": centered_mp > centered_sp,
        "runtime_manifest_sha256": _sha256(
            pair_root / "runtime_manifest.json"
        ),
    }
    pair_scenarios = [
        {
            "pair_id": pair_id,
            "domain": pair["domain"],
            "scenario": scenario,
            "label_spoof": int(scenario == "carryoff_spoof"),
            "legacy_median_residual": median(
                scenario, "complex_geometry_residual"
            ),
            "clock_centered_median_residual": median(
                scenario, "clock_centered_geometry_residual"
            ),
            "comparison_bin_count": len(scenario_rows[scenario]),
        }
        for scenario, _ in receiver_records
    ]
    return summary, delay_rows, geometry_rows, pair_scenarios


def evaluate_pair_summaries(
    pair_summaries: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_repetitions: int,
    gates: dict[str, Any],
) -> dict[str, Any]:
    multipath = np.asarray(
        [
            row["clock_centered_multipath_median_residual"]
            for row in pair_summaries
        ],
        dtype=np.float64,
    )
    spoof = np.asarray(
        [
            row["clock_centered_spoof_median_residual"]
            for row in pair_summaries
        ],
        dtype=np.float64,
    )
    separation = multipath - spoof
    legacy = np.asarray(
        [row["legacy_separation"] for row in pair_summaries],
        dtype=np.float64,
    )
    auc = replication._pair_block_auc(multipath, spoof)
    rng = np.random.default_rng(bootstrap_seed)
    boot_separation = np.empty(bootstrap_repetitions, dtype=np.float64)
    boot_auc = np.empty(bootstrap_repetitions, dtype=np.float64)
    count = len(pair_summaries)
    for repetition in range(bootstrap_repetitions):
        indices = rng.integers(0, count, size=count)
        boot_separation[repetition] = float(
            np.median(separation[indices])
        )
        boot_auc[repetition] = replication._pair_block_auc(
            multipath[indices], spoof[indices]
        )

    positive_count = int(np.count_nonzero(separation > 0.0))
    improvement_count = int(np.count_nonzero(separation > legacy))
    minimum_bins = min(
        min(
            int(row["multipath_comparison_bin_count"]),
            int(row["spoof_comparison_bin_count"]),
        )
        for row in pair_summaries
    )
    minimum_los = min(
        int(row["startup_los_prn_count"]) for row in pair_summaries
    )
    records = {
        "required_pair_count": {
            "observed": count,
            "required": int(gates["required_pair_count"]),
            "passed": count == int(gates["required_pair_count"]),
        },
        "positive_clock_centered_separation_pair_count": {
            "observed": positive_count,
            "required": int(
                gates["positive_clock_centered_separation_pair_count"]
            ),
            "passed": positive_count >= int(
                gates["positive_clock_centered_separation_pair_count"]
            ),
        },
        "minimum_pair_block_auc": {
            "observed": auc,
            "required": float(gates["minimum_pair_block_auc"]),
            "passed": auc >= float(gates["minimum_pair_block_auc"]),
        },
        "minimum_clock_centered_improvement_over_legacy_pair_count": {
            "observed": improvement_count,
            "required": int(
                gates[
                    "minimum_clock_centered_improvement_over_legacy_pair_count"
                ]
            ),
            "passed": improvement_count >= int(
                gates[
                    "minimum_clock_centered_improvement_over_legacy_pair_count"
                ]
            ),
        },
        "minimum_comparison_bins_per_scenario_per_pair": {
            "observed": minimum_bins,
            "required": int(
                gates["minimum_comparison_bins_per_scenario_per_pair"]
            ),
            "passed": minimum_bins >= int(
                gates["minimum_comparison_bins_per_scenario_per_pair"]
            ),
        },
        "minimum_startup_los_prns_per_pair": {
            "observed": minimum_los,
            "required": int(gates["minimum_startup_los_prns_per_pair"]),
            "passed": minimum_los >= int(
                gates["minimum_startup_los_prns_per_pair"]
            ),
        },
    }
    all_passed = all(record["passed"] for record in records.values())
    return {
        "pair_count": count,
        "pair_block_auc": auc,
        "pair_level_clock_centered_separations": separation.tolist(),
        "median_pair_separation": float(np.median(separation)),
        "median_pair_separation_bootstrap_95_percentile_interval": [
            float(value)
            for value in np.percentile(boot_separation, [2.5, 97.5])
        ],
        "pair_block_auc_bootstrap_95_percentile_interval": [
            float(value) for value in np.percentile(boot_auc, [2.5, 97.5])
        ],
        "positive_separation_pair_count": positive_count,
        "clock_centered_improvement_over_legacy_pair_count": (
            improvement_count
        ),
        "gates": records,
        "all_support_gates_passed": all_passed,
        "status": "SUPPORTED" if all_passed else "NOT_SUPPORTED",
    }


def _analyze_all_in_memory(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    estimator = pilot._estimator(context["controlled"])
    pair_summaries: list[dict[str, Any]] = []
    all_delays: list[dict[str, Any]] = []
    all_geometry: list[dict[str, Any]] = []
    pair_scenarios: list[dict[str, Any]] = []
    for pair in context["pairs"]:
        summary, delays, geometry, scenarios = _analyze_pair_in_memory(
            config, config_path, context, pair, estimator
        )
        pair_summaries.append(summary)
        all_delays.extend(delays)
        all_geometry.extend(geometry)
        pair_scenarios.extend(scenarios)

    eligible = [row for row in all_geometry if row["comparison_eligible"]]
    labels = np.asarray(
        [
            int(row["scenario"] == "carryoff_spoof")
            for row in eligible
        ]
    )
    secondary = {
        "legacy_zero_referenced": float(
            roc_auc_score(
                labels,
                -np.asarray(
                    [
                        float(row["complex_geometry_residual"])
                        for row in eligible
                    ]
                ),
            )
        ),
        "clock_centered_directional": float(
            roc_auc_score(
                labels,
                -np.asarray(
                    [
                        float(row["clock_centered_geometry_residual"])
                        for row in eligible
                    ]
                ),
            )
        ),
    }
    evaluation = config["evaluation"]
    primary = evaluate_pair_summaries(
        pair_summaries,
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
        bootstrap_repetitions=int(evaluation["bootstrap_repetitions"]),
        gates=evaluation["support_gates"],
    )
    return {
        "pairs": pair_summaries,
        "delays": all_delays,
        "geometry": all_geometry,
        "pair_scenarios": pair_scenarios,
        "primary": primary,
        "secondary": secondary,
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _committed_release_record() -> dict[str, Any]:
    relative_paths = [
        "scripts/run_cgc_rf_locked_test.py",
        "configs/experiments/cgc_rf_locked_test_v1.json",
        "docs/results/cgc_rf_locked_test_protocol_v1.md",
    ]
    for relative in relative_paths:
        _git("ls-files", "--error-unmatch", relative)
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=REPO_ROOT,
        )
        if dirty.returncode != 0:
            raise ValueError(
                f"release input is not committed and clean: {relative}"
            )
    candidate = "60280575737ff70618c5619ff7522c516bbdc67a"
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", candidate, "HEAD"],
        cwd=REPO_ROOT,
    )
    if ancestor.returncode != 0:
        raise ValueError("frozen candidate commit is not an ancestor of HEAD")
    return {
        "head_commit": _git("rev-parse", "HEAD"),
        "candidate_commit": candidate,
        "config_commit": _git(
            "log",
            "-1",
            "--format=%H",
            "--",
            "configs/experiments/cgc_rf_locked_test_v1.json",
        ),
        "protocol_commit": _git(
            "log",
            "-1",
            "--format=%H",
            "--",
            "docs/results/cgc_rf_locked_test_protocol_v1.md",
        ),
        "runner_commit": _git(
            "log",
            "-1",
            "--format=%H",
            "--",
            "scripts/run_cgc_rf_locked_test.py",
        ),
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }


def _start_or_resume_release(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
    *,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    state_path = context["output_root"] / "release_state.json"
    summary_path = context["output_root"] / "summary.json"
    commit_record = _committed_release_record()
    if not resume:
        if context["source_root"].exists():
            raise FileExistsError(context["source_root"])
        if context["output_root"].exists():
            raise FileExistsError(context["output_root"])
        state = {
            "schema": "gnss-doppler-lab.cgc-rf-locked-test-release-state",
            "schema_version": 1,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "released_before_source_generation",
            "config": {
                "path": str(config_path),
                "sha256": _sha256(config_path),
            },
            "protocol": {
                "path": str(PROTOCOL_PATH),
                "sha256": _sha256(PROTOCOL_PATH),
            },
            "commits": commit_record,
            "authorized_pair_ids": EXPECTED_PAIR_IDS,
            "post_release_tuning_or_retest": False,
            "metrics_emitted": False,
        }
        _write_json(state_path, state)
        return state_path, state

    if not state_path.is_file():
        raise FileNotFoundError(
            "resume requested without release_state.json"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("config", {}).get("sha256") != _sha256(config_path):
        raise ValueError("resume config hash mismatch")
    if state.get("commits", {}).get("runner_sha256") != (
        commit_record["runner_sha256"]
    ):
        raise ValueError("resume runner hash mismatch")
    if state.get("metrics_emitted") is not False:
        raise ValueError("metrics were emitted; rerun is forbidden")
    if summary_path.exists():
        raise ValueError(
            "locked-test summary already exists; rerun is forbidden"
        )
    return state_path, state


def _update_phase(
    state_path: Path, state: dict[str, Any], phase: str
) -> None:
    state["phase"] = phase
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(state_path, state)


def _emit_result(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
    state_path: Path,
    state: dict[str, Any],
    analysis: dict[str, Any],
    source_summary: Path,
) -> dict[str, Any]:
    analysis_root = context["output_root"] / "analysis"
    delay_path = analysis_root / "delay_estimates.csv"
    geometry_path = analysis_root / "geometry_scores.csv"
    pair_path = analysis_root / "pair_summary.csv"
    scenario_path = analysis_root / "pair_scenario_medians.csv"

    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(state_path, state)

    _write_csv(delay_path, analysis["delays"])
    _write_csv(geometry_path, analysis["geometry"])
    _write_csv(pair_path, analysis["pairs"])
    _write_csv(scenario_path, analysis["pair_scenarios"])
    result = {
        "schema": "gnss-doppler-lab.cgc-rf-locked-test-result",
        "schema_version": 1,
        "role": config["experiment"]["role"],
        "execution_policy": config["experiment"]["execution_policy"],
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": _sha256(PROTOCOL_PATH),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
            "commit": state["commits"]["runner_commit"],
        },
        "release_state": {
            "path": str(state_path.resolve()),
            "sha256": _sha256(state_path),
        },
        "source_generation_summary": {
            "path": str(source_summary.resolve()),
            "sha256": _sha256(source_summary),
        },
        "frozen_candidate": config["frozen_candidate"],
        "pairs": analysis["pairs"],
        "primary_pair_block_evaluation": analysis["primary"],
        "secondary_serial_bin_auc": analysis["secondary"],
        "artifacts": {
            "delay_estimates": {
                "path": str(delay_path.resolve()),
                "sha256": _sha256(delay_path),
                "row_count": len(analysis["delays"]),
            },
            "geometry_scores": {
                "path": str(geometry_path.resolve()),
                "sha256": _sha256(geometry_path),
                "row_count": len(analysis["geometry"]),
            },
            "pair_summary": {
                "path": str(pair_path.resolve()),
                "sha256": _sha256(pair_path),
                "row_count": len(analysis["pairs"]),
            },
            "pair_scenario_medians": {
                "path": str(scenario_path.resolve()),
                "sha256": _sha256(scenario_path),
                "row_count": len(analysis["pair_scenarios"]),
            },
        },
        "data_boundary": config["data_boundary"],
        "threshold_fitted": False,
        "absolute_threshold_applied": False,
        "post_release_tuning_or_retest": False,
        "claim_boundary": config["claim_boundary"],
    }
    summary_path = context["output_root"] / "summary.json"
    _write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release-token", choices=[RELEASE_TOKEN])
    mode.add_argument("--resume-before-metrics", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("only the sealed default config path is authorized")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    context = validate_config(config, config_path=config_path)
    if args.validate_only:
        print(
            "sealed config/protocol and 15 pinned inputs verified; "
            "test signals were not accessed"
        )
        return 0

    resume = bool(args.resume_before_metrics)
    state_path, state = _start_or_resume_release(
        config, config_path, context, resume=resume
    )
    _update_phase(state_path, state, "source_generation")
    for pair in context["pairs"]:
        print(f"[test-source] {pair['paired_group_id']}", flush=True)
        _ensure_source_pair(
            config, config_path, context, pair, resume=resume
        )
    source_summary = _write_source_summary(config, config_path, context)
    _update_phase(state_path, state, "receiver_processing")
    for pair in context["pairs"]:
        print(f"[test-receiver] {pair['paired_group_id']}", flush=True)
        _ensure_test_pair(
            config, config_path, context, pair, resume=resume
        )
    _update_phase(state_path, state, "analysis_in_memory")
    analysis = _analyze_all_in_memory(config, config_path, context)
    _emit_result(
        config,
        config_path,
        context,
        state_path,
        state,
        analysis,
        source_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
