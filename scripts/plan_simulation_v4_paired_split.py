#!/usr/bin/env python3
"""Plan and enforce leakage-safe train/validation/test splits for paired simulation-v4."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/experiments/simulation_v4_paired_split_v1.json")
DEFAULT_OUTPUT_ROOT = Path("artifacts/simulation_v4_paired_split_v1")
PARTITIONS = ("train", "validation", "test")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _safe_name(value: Any, name: str) -> str:
    result = str(value)
    if not SAFE_NAME.fullmatch(result):
        raise ValueError(f"{name} must be a safe identifier")
    return result


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return value


def _utc(value: Any, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.microsecond:
        raise ValueError(f"{name} must be timezone-aware with integer seconds")
    return parsed.astimezone(timezone.utc)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _motion_kind(pair: dict[str, Any]) -> str:
    return "static" if pair["domain"] == "static" else str(pair["motion"]["kind"])


def _base_run_document(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "utc": pair["utc"],
        "duration_seconds": pair["duration_seconds"],
        "domain": pair["domain"],
        "position": pair["position"],
        "motion": pair.get("motion"),
        "receiver_seed": pair["receiver_seed"],
        "target_composite_cn0_db_hz": pair["target_composite_cn0_db_hz"],
    }


def _validate_position(position: Any, name: str) -> None:
    doc = _exact_keys(
        position,
        {"latitude_deg", "longitude_deg", "altitude_m"},
        name,
    )
    latitude = _finite(doc["latitude_deg"], f"{name}.latitude_deg")
    longitude = _finite(doc["longitude_deg"], f"{name}.longitude_deg")
    _finite(doc["altitude_m"], f"{name}.altitude_m")
    if not -89.9 <= latitude <= 89.9 or not -180 <= longitude <= 180:
        raise ValueError(f"{name} is outside the trajectory generator bounds")


def _validate_motion(pair: dict[str, Any], name: str) -> str:
    domain = str(pair["domain"])
    motion = pair.get("motion")
    if domain == "static":
        if motion is not None:
            raise ValueError(f"{name}: static pair must not define motion")
        return "static"
    if domain != "dynamic" or not isinstance(motion, dict):
        raise ValueError(f"{name}: dynamic pair must define motion")
    kind = str(motion.get("kind"))
    common = {"kind", "speed_mps", "heading_deg"}
    expected = {
        "straight": common,
        "circle": common | {"radius_m"},
        "parallel-sweep": common | {"leg_length_m", "lane_spacing_m"},
    }
    if kind not in expected:
        raise ValueError(f"{name}: unsupported motion kind {kind!r}")
    _exact_keys(motion, expected[kind], f"{name}.motion")
    if _finite(motion["speed_mps"], f"{name}.motion.speed_mps") <= 0:
        raise ValueError(f"{name}: speed_mps must be positive")
    _finite(motion["heading_deg"], f"{name}.motion.heading_deg")
    for key in ("radius_m", "leg_length_m", "lane_spacing_m"):
        if key in motion and _finite(motion[key], f"{name}.motion.{key}") <= 0:
            raise ValueError(f"{name}: {key} must be positive")
    return kind


def _validate_spoof(spoof: Any, duration: int, name: str) -> None:
    keys = {
        "start_seconds",
        "transition_seconds",
        "target_offset_enu_m",
        "initial_advantage_db",
        "final_advantage_db",
        "power_ramp_seconds",
    }
    doc = _exact_keys(spoof, keys, name)
    start = _finite(doc["start_seconds"], f"{name}.start_seconds")
    transition = _finite(doc["transition_seconds"], f"{name}.transition_seconds")
    ramp = _finite(doc["power_ramp_seconds"], f"{name}.power_ramp_seconds")
    if not 0 < start < duration:
        raise ValueError(f"{name}: start must occur inside the run")
    if transition <= 0 or start + transition >= duration:
        raise ValueError(f"{name}: transition must finish inside the run")
    if ramp < 0 or start + ramp >= duration:
        raise ValueError(f"{name}: power ramp must finish inside the run")
    offsets = doc["target_offset_enu_m"]
    if not isinstance(offsets, list) or len(offsets) != 3:
        raise ValueError(f"{name}.target_offset_enu_m must contain three values")
    vector = tuple(_finite(value, f"{name}.target_offset_enu_m") for value in offsets)
    if vector == (0.0, 0.0, 0.0):
        raise ValueError(f"{name}: target offset must be non-zero")
    advantages = (
        _finite(doc["initial_advantage_db"], f"{name}.initial_advantage_db"),
        _finite(doc["final_advantage_db"], f"{name}.final_advantage_db"),
    )
    if any(not -40 <= value <= 6 for value in advantages):
        raise ValueError(f"{name}: spoof advantages must be in [-40, 6] dB")


def validate_config(
    config: dict[str, Any],
    *,
    verify_prior_source: bool = True,
) -> dict[str, str]:
    """Validate the frozen split contract and return base fingerprints by pair."""
    top_keys = {
        "version",
        "campaign",
        "normal_profile_source",
        "calibration_exclusions",
        "split_policy",
        "fixed_rf_profile",
        "pair_contract",
        "data_boundary",
        "pairs",
    }
    _exact_keys(config, top_keys, "configuration")
    if config["version"] != 1:
        raise ValueError("unsupported paired split config version")
    campaign = _exact_keys(config["campaign"], {"name", "purpose"}, "campaign")
    _safe_name(campaign["name"], "campaign.name")

    policy = config["split_policy"]
    expected_counts = policy["expected_pair_counts"]
    if set(expected_counts) != set(PARTITIONS):
        raise ValueError("expected_pair_counts must define train, validation, and test")
    if policy["unit"] != "paired_group_id":
        raise ValueError("split unit must be paired_group_id")
    for key in (
        "pair_members_atomic",
        "prns_atomic_with_pair",
        "windows_atomic_with_pair",
        "receiver_seed_unique_across_pairs",
        "base_run_fingerprint_unique_across_pairs",
        "random_window_split_forbidden",
        "test_refit_or_adaptation_forbidden",
    ):
        if policy.get(key) is not True:
            raise ValueError(f"split_policy.{key} must be true")
    if policy.get("normalization_fit_partition") != "train":
        raise ValueError("normalization must be fit on train only")
    if policy.get("threshold_selection_partition") != "validation":
        raise ValueError("threshold selection must use validation only")
    if "locked" not in str(policy.get("test_access", "")).lower():
        raise ValueError("test_access must be locked before model freeze")

    profile = config["fixed_rf_profile"]
    expected_profile = {
        "rf_sample_rate_hz": 25_000_000,
        "frontend_cutoff_hz": 8_000_000.0,
        "frontend_order": 5,
        "normal_target_rms": 22.0,
        "tracking_tap_count": 9,
        "tracking_tap_spacing_chips": 0.125,
        "feature_tap_count": 3,
    }
    if profile != expected_profile:
        raise ValueError("fixed_rf_profile drifted from the selected Method-A profile")

    contract = config["pair_contract"]
    if contract.get("members") != ["steady_normal", "carryoff_spoof"]:
        raise ValueError("each pair must contain steady_normal and carryoff_spoof")
    if contract.get("pre_onset_iq_must_be_byte_identical") is not True:
        raise ValueError("paired prefix identity must be required")

    exclusions = config["calibration_exclusions"]
    excluded = exclusions.get("paired_group_ids")
    if not isinstance(excluded, list) or not excluded:
        raise ValueError("calibration exclusions must list prior inspected runs")
    if len(excluded) != len(set(excluded)):
        raise ValueError("calibration exclusions contain duplicates")
    for value in excluded:
        _safe_name(value, "calibration exclusion")

    if verify_prior_source:
        source = config["normal_profile_source"]
        source_path = _repo_path(source["path"])
        if _sha256(source_path) != source["sha256"]:
            raise ValueError("normal profile source hash mismatch")
        source_doc = json.loads(source_path.read_text(encoding="utf-8"))
        if source_doc.get("decision", {}).get("overall_gate_status") != source["decision"]:
            raise ValueError("normal profile source decision mismatch")
        if set(source_doc.get("runs", {})) != set(excluded):
            raise ValueError("calibration exclusions do not match inspected source runs")

    boundary = config["data_boundary"]
    if set(boundary["texbat_clean_used_only_for_prior_normal_calibration"]) != {
        "cleanStatic",
        "cleanDynamic",
    }:
        raise ValueError("only TEXBAT clean recordings may be prior calibration inputs")
    if set(boundary["forbidden_texbat_recordings"]) != {
        f"ds{index}" for index in range(1, 9)
    }:
        raise ValueError("TEXBAT spoof recording boundary must remain ds1-ds8")
    if boundary.get("test_partition_is_simulation_only") is not True:
        raise ValueError("this split plan must remain simulation-only")

    pairs = config["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pairs must be a non-empty list")
    pair_ids: list[str] = []
    seeds: list[int] = []
    timestamps: list[datetime] = []
    fingerprints: dict[str, str] = {}
    motions_by_split = {partition: set() for partition in PARTITIONS}
    domains_by_split = {partition: set() for partition in PARTITIONS}
    counts = {partition: 0 for partition in PARTITIONS}
    for index, pair in enumerate(pairs):
        name = f"pairs[{index}]"
        required = {
            "paired_group_id",
            "split",
            "utc",
            "duration_seconds",
            "domain",
            "position",
            "receiver_seed",
            "target_composite_cn0_db_hz",
            "spoofing",
        }
        allowed = required | {"motion"}
        if not isinstance(pair, dict) or set(pair) - allowed or required - set(pair):
            raise ValueError(f"{name} has missing or unknown keys")
        pair_id = _safe_name(pair["paired_group_id"], f"{name}.paired_group_id")
        split = str(pair["split"])
        if split not in PARTITIONS:
            raise ValueError(f"{name}.split must be one of {PARTITIONS}")
        if pair_id in excluded:
            raise ValueError(f"{pair_id} reuses an inspected calibration run")
        duration = _integer(pair["duration_seconds"], f"{name}.duration_seconds")
        if not 20 <= duration <= 300:
            raise ValueError(f"{name}.duration_seconds must be in [20, 300]")
        timestamp = _utc(pair["utc"], f"{name}.utc")
        _validate_position(pair["position"], f"{name}.position")
        motion = _validate_motion(pair, name)
        seed = _integer(pair["receiver_seed"], f"{name}.receiver_seed")
        if not 0 <= seed < 2**64:
            raise ValueError(f"{name}.receiver_seed must be unsigned 64-bit")
        cn0 = _finite(
            pair["target_composite_cn0_db_hz"],
            f"{name}.target_composite_cn0_db_hz",
        )
        if cn0 not in {59.5, 60.0, 60.5}:
            raise ValueError(f"{name}: C/N0 must stay in the frozen 59.5-60.5 grid")
        _validate_spoof(pair["spoofing"], duration, f"{name}.spoofing")
        fingerprint = _canonical_sha256(_base_run_document(pair))
        pair_ids.append(pair_id)
        seeds.append(seed)
        timestamps.append(timestamp)
        fingerprints[pair_id] = fingerprint
        counts[split] += 1
        motions_by_split[split].add(motion)
        domains_by_split[split].add(str(pair["domain"]))

    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("paired_group_id values must be unique")
    if len(seeds) != len(set(seeds)):
        raise ValueError("receiver seeds must be unique across every pair")
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("UTC epochs must be unique across every pair")
    if len(fingerprints) != len(set(fingerprints.values())):
        raise ValueError("base run fingerprints must be unique across every pair")
    normalized_expected = {key: int(value) for key, value in expected_counts.items()}
    if counts != normalized_expected:
        raise ValueError(f"pair counts differ from the frozen split: {counts}")
    required_motion = policy["required_motion_coverage"]
    if set(required_motion) != set(PARTITIONS):
        raise ValueError("required_motion_coverage must define every partition")
    for partition in PARTITIONS:
        expected = set(required_motion[partition])
        if motions_by_split[partition] != expected:
            raise ValueError(
                f"{partition} motion coverage mismatch: {sorted(motions_by_split[partition])}"
            )
        if domains_by_split[partition] != {"static", "dynamic"}:
            raise ValueError(f"{partition} must contain static and dynamic pairs")
    return fingerprints


def _catalog_rows(
    config: dict[str, Any], fingerprints: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    for pair in config["pairs"]:
        pair_id = str(pair["paired_group_id"])
        split = str(pair["split"])
        normal_id = f"{pair_id}-normal"
        spoof_id = f"{pair_id}-spoof"
        pairs.append({
            "paired_group_id": pair_id,
            "split": split,
            "base_run_fingerprint": fingerprints[pair_id],
            "domain": pair["domain"],
            "motion_kind": _motion_kind(pair),
            "utc": pair["utc"],
            "duration_seconds": pair["duration_seconds"],
            "latitude_deg": pair["position"]["latitude_deg"],
            "longitude_deg": pair["position"]["longitude_deg"],
            "altitude_m": pair["position"]["altitude_m"],
            "motion_json": json.dumps(pair.get("motion"), sort_keys=True, separators=(",", ":")),
            "receiver_seed": pair["receiver_seed"],
            "target_composite_cn0_db_hz": pair["target_composite_cn0_db_hz"],
            "normal_scenario_id": normal_id,
            "spoof_scenario_id": spoof_id,
            "test_access": "locked" if split == "test" else "available_after_generation",
        })
        scenarios.extend((
            {
                "scenario_id": normal_id,
                "paired_group_id": pair_id,
                "split": split,
                "scenario_kind": "steady_normal",
                "class": "normal",
                "spoofing_json": "null",
            },
            {
                "scenario_id": spoof_id,
                "paired_group_id": pair_id,
                "split": split,
                "scenario_kind": "carryoff_spoof",
                "class": "spoofing",
                "spoofing_json": json.dumps(pair["spoofing"], sort_keys=True, separators=(",", ":")),
            },
        ))
    return pairs, scenarios


def create_plan(config_path: Path, output_root: Path) -> Path:
    config_path = _repo_path(config_path)
    output_root = _repo_path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fingerprints = validate_config(config)
    pair_rows, scenario_rows = _catalog_rows(config, fingerprints)
    output_root.mkdir(parents=True)
    pair_fields = list(pair_rows[0])
    scenario_fields = list(scenario_rows[0])
    pair_catalog = output_root / "pair_catalog.csv"
    scenario_catalog = output_root / "scenario_catalog.csv"
    _atomic_csv(pair_catalog, pair_rows, pair_fields)
    _atomic_csv(scenario_catalog, scenario_rows, scenario_fields)

    partition_files: dict[str, dict[str, Any]] = {}
    for partition in PARTITIONS:
        suffix = ".locked" if partition == "test" else ""
        path = output_root / "partitions" / f"{partition}_groups{suffix}.txt"
        group_ids = sorted(
            row["paired_group_id"] for row in pair_rows if row["split"] == partition
        )
        _atomic_bytes(path, ("\n".join(group_ids) + "\n").encode("utf-8"))
        partition_files[partition] = {
            "path": str(path),
            "sha256": _sha256(path),
            "paired_group_ids": group_ids,
            "pair_count": len(group_ids),
            "scenario_count": 2 * len(group_ids),
            "access": "locked" if partition == "test" else "planned",
        }

    partitions = {
        partition: set(partition_files[partition]["paired_group_ids"])
        for partition in PARTITIONS
    }
    overlap = {
        f"{left}_vs_{right}": sorted(partitions[left] & partitions[right])
        for index, left in enumerate(PARTITIONS)
        for right in PARTITIONS[index + 1 :]
    }
    manifest = {
        "schema": "gnss-doppler-lab.simulation-v4-paired-split-plan",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": config["campaign"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        "split_policy": config["split_policy"],
        "pair_contract": config["pair_contract"],
        "normal_profile_source": config["normal_profile_source"],
        "calibration_exclusions": config["calibration_exclusions"],
        "partitions": partition_files,
        "catalogs": {
            "pairs": {"path": str(pair_catalog), "sha256": _sha256(pair_catalog), "rows": len(pair_rows)},
            "scenarios": {"path": str(scenario_catalog), "sha256": _sha256(scenario_catalog), "rows": len(scenario_rows)},
        },
        "leakage_checks": {
            "partition_intersections": overlap,
            "all_partition_intersections_empty": all(not values for values in overlap.values()),
            "unique_receiver_seed_per_pair": True,
            "unique_base_run_fingerprint_per_pair": True,
            "calibration_runs_excluded": True,
            "paired_members_never_split": True,
            "prns_and_windows_inherit_pair_partition": True,
        },
        "test_release": {
            "status": "locked",
            "required_before_release": [
                "model artifact and SHA-256",
                "preprocessing artifact and SHA-256",
                "decision-threshold artifact and SHA-256",
                "frozen split-plan SHA-256",
            ],
            "refit_or_adaptation_after_release": "forbidden",
        },
        "data_boundary": config["data_boundary"],
        "interpretation": "split assignment is frozen before paired IQ generation; no detector was trained and no test features were accessed",
    }
    manifest_path = output_root / "split_manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _read_dataset(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "run_id",
        "source_fingerprint",
        "label",
        "paired_group_id",
        "scenario_kind",
        "is_spoofing",
    }
    if not required.issubset(fields):
        raise ValueError(f"labeled dataset is missing columns: {sorted(required - set(fields))}")
    if not rows:
        raise ValueError("labeled dataset is empty")
    return fields, rows


def validate_dataset_rows(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate a completed labeled dataset without allowing pair-level leakage."""
    assignment = {str(pair["paired_group_id"]): str(pair["split"]) for pair in config["pairs"]}
    expected_groups = set(assignment)
    observed_groups = {row["paired_group_id"] for row in rows}
    if observed_groups != expected_groups:
        raise ValueError(
            f"dataset paired groups differ from plan; missing={sorted(expected_groups-observed_groups)}, "
            f"unknown={sorted(observed_groups-expected_groups)}"
        )
    run_to_group: dict[str, str] = {}
    source_to_group: dict[str, str] = {}
    group_kinds = {group: set() for group in expected_groups}
    group_labels = {group: set() for group in expected_groups}
    group_flags = {group: set() for group in expected_groups}
    row_counts = {partition: 0 for partition in PARTITIONS}
    for row in rows:
        group = row["paired_group_id"]
        split = assignment[group]
        run_id = row["run_id"]
        source = row["source_fingerprint"]
        if not run_id or not source:
            raise ValueError("run_id and source_fingerprint must be non-empty")
        previous_group = run_to_group.setdefault(run_id, group)
        if previous_group != group:
            raise ValueError(f"run_id leaks across paired groups: {run_id}")
        previous_source_group = source_to_group.setdefault(source, group)
        if previous_source_group != group:
            raise ValueError(f"source fingerprint leaks across paired groups: {source}")
        group_kinds[group].add(row["scenario_kind"])
        group_labels[group].add(row["label"])
        group_flags[group].add(row["is_spoofing"])
        row_counts[split] += 1
    for group in sorted(expected_groups):
        if group_kinds[group] != {"steady_normal", "carryoff_spoof"}:
            raise ValueError(f"{group} does not contain both planned scenario kinds")
        if not {"normal", "spoofing"}.issubset(group_labels[group]):
            raise ValueError(f"{group} does not contain both normal and spoofing labels")
        if not {"0", "1"}.issubset(group_flags[group]):
            raise ValueError(f"{group} does not contain both is_spoofing states")
    source_partitions: dict[str, set[str]] = {}
    for source, group in source_to_group.items():
        source_partitions.setdefault(source, set()).add(assignment[group])
    leaking_sources = sorted(
        source for source, partitions in source_partitions.items() if len(partitions) > 1
    )
    if leaking_sources:
        raise ValueError(f"source fingerprint leaks across partitions: {leaking_sources}")
    return {
        "row_counts": row_counts,
        "paired_group_count": len(expected_groups),
        "run_id_count": len(run_to_group),
        "source_fingerprint_count": len(source_to_group),
        "pair_atomic": True,
        "source_fingerprint_partition_overlap": [],
    }


def _verify_freeze_manifest(
    freeze_path: Path,
    *,
    split_config_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    document = json.loads(freeze_path.read_text(encoding="utf-8"))
    if document.get("schema") != "gnss-doppler-lab.simulation-v4-model-freeze":
        raise ValueError("invalid model freeze schema")
    if document.get("split_config_sha256") != split_config_sha256:
        raise ValueError("model freeze split config hash mismatch")
    if document.get("split_manifest_sha256") != split_manifest_sha256:
        raise ValueError("model freeze split manifest hash mismatch")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "model",
        "preprocessing",
        "thresholds",
    }:
        raise ValueError("model freeze must pin model, preprocessing, and thresholds")
    for name, artifact in artifacts.items():
        path = _repo_path(artifact["path"])
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"frozen {name} artifact hash mismatch")
    return document


def materialize_dataset(
    dataset_path: Path,
    config_path: Path,
    plan_root: Path,
    *,
    release_test: bool = False,
    freeze_manifest: Path | None = None,
) -> Path:
    """Materialize pair-atomic partitions; test remains locked without a model freeze."""
    dataset_path = _repo_path(dataset_path)
    config_path = _repo_path(config_path)
    plan_root = _repo_path(plan_root)
    split_manifest_path = plan_root / "split_manifest.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(split_manifest_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("config", {}).get("sha256") != _sha256(config_path):
        raise ValueError("split plan/config hash mismatch")
    fields, rows = _read_dataset(dataset_path)
    checks = validate_dataset_rows(rows, config)
    assignment = {str(pair["paired_group_id"]): str(pair["split"]) for pair in config["pairs"]}
    release_record = None
    if release_test:
        if freeze_manifest is None:
            raise ValueError("--release-test requires --freeze-manifest")
        freeze_path = _repo_path(freeze_manifest)
        release_record = _verify_freeze_manifest(
            freeze_path,
            split_config_sha256=_sha256(config_path),
            split_manifest_sha256=_sha256(split_manifest_path),
        )

    partition_root = plan_root / "dataset_partitions"
    if partition_root.exists():
        raise FileExistsError(partition_root)
    partition_root.mkdir(parents=True)

    outputs: dict[str, Any] = {}
    for partition in PARTITIONS:
        selected = [row for row in rows if assignment[row["paired_group_id"]] == partition]
        if partition == "test" and not release_test:
            outputs[partition] = {
                "status": "locked",
                "row_count": len(selected),
                "path": None,
                "sha256": None,
            }
            continue
        path = partition_root / f"{partition}.csv"
        _atomic_csv(path, selected, fields)
        outputs[partition] = {
            "status": "released" if partition == "test" else "available",
            "row_count": len(selected),
            "path": str(path),
            "sha256": _sha256(path),
        }
    document = {
        "schema": "gnss-doppler-lab.simulation-v4-paired-dataset-split",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(dataset_path), "sha256": _sha256(dataset_path), "rows": len(rows)},
        "split_config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "split_plan": {"path": str(split_manifest_path), "sha256": _sha256(split_manifest_path)},
        "validation": checks,
        "partitions": outputs,
        "test_release": {
            "released": release_test,
            "freeze_manifest": None if not release_record else {
                "path": str(_repo_path(freeze_manifest)),
                "sha256": _sha256(_repo_path(freeze_manifest)),
            },
            "refit_or_adaptation_after_release": "forbidden",
        },
    }
    output_manifest = partition_root / "manifest.json"
    _atomic_json(output_manifest, document)
    return output_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--resume-plan", action="store_true")
    parser.add_argument("--release-test", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    args = parser.parse_args(argv)
    config_path = _repo_path(args.config)
    output_root = _repo_path(args.output_root)
    if args.release_test and args.dataset is None:
        parser.error("--release-test requires --dataset")
    if args.resume_plan:
        plan = output_root / "split_manifest.json"
        if not plan.is_file():
            raise FileNotFoundError(plan)
        document = json.loads(plan.read_text(encoding="utf-8"))
        if document.get("config", {}).get("sha256") != _sha256(config_path):
            raise ValueError("existing split plan/config hash mismatch")
    else:
        plan = create_plan(config_path, output_root)
    if args.dataset is None:
        print(plan)
        return 0
    result = materialize_dataset(
        args.dataset,
        config_path,
        output_root,
        release_test=args.release_test,
        freeze_manifest=args.freeze_manifest,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
