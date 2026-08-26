#!/usr/bin/env python3
"""Execute the locked CGC candidate once on validation geometries 007--009."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import audit_correlator_geometry_identifiability as train_audit  # noqa: E402
import plan_simulation_v4_paired_split as splitter  # noqa: E402
from gnss_doppler_lab.gps_sdr_sim import GpsSdrSimRunner  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.rf_config import (  # noqa: E402
    InputConfig,
    OutputConfig,
    RFGenerationConfig,
    Scenario,
    SimulatorConfig,
    StaticPosition,
)
from gnss_doppler_lab.rf_impairments import clean_impairments  # noqa: E402

DEFAULT_CONFIG = Path("configs/experiments/correlator_geometry_identifiability_validation_v1.json")
EXPECTED_VALIDATION_IDS = ["pv1-pair-007", "pv1-pair-008", "pv1-pair-009"]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_pinned_json(source: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    path = _repo_path(source["path"])
    observed = _sha256(path)
    if observed != source["sha256"]:
        raise ValueError(f"{name} SHA-256 mismatch: {observed}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_config(
    config: dict[str, Any], *, verify_inputs: bool = False
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    if config.get("version") != 1:
        raise ValueError("unsupported CGC validation config version")
    experiment = config.get("experiment", {})
    splitter._safe_name(str(experiment.get("name", "")), "experiment.name")
    if experiment.get("role") != "locked held-out geometry validation of the train-generated CGC candidate":
        raise ValueError("validation role must remain locked and held-out")
    if experiment.get("execution_policy") != "one deterministic validation execution; no tuning or rerun based on outcome":
        raise ValueError("validation execution policy drifted")

    frozen = config.get("frozen_candidate", {})
    train_config_path, train_config = _load_pinned_json(frozen["train_config"], "train config")
    train_record_path, train_record = _load_pinned_json(frozen["train_result_record"], "train result record")
    train_artifact_path, train_artifact = _load_pinned_json(frozen["train_artifact_summary"], "train artifact summary")
    split_path, split = _load_pinned_json(config["split_config"], "split config")
    train_audit.validate_config(train_config, verify_source_artifacts=True)
    splitter.validate_config(split)

    if frozen.get("git_commit") != "15f59ab70de6269845fd88b8188106cb97cdd736":
        raise ValueError("frozen candidate commit drifted")
    if _sha256(REPO_ROOT / "scripts/audit_correlator_geometry_identifiability.py") != frozen.get("train_runner_sha256"):
        raise ValueError("frozen train runner changed before validation")
    if _sha256(REPO_ROOT / "src/gnss_doppler_lab/correlator_geometry.py") != frozen.get("physics_module_sha256"):
        raise ValueError("frozen CGC physics module changed before validation")
    if train_record.get("status") != "supported_on_train_requires_validation":
        raise ValueError("train candidate was not eligible for validation")
    if train_record.get("decision", {}).get("all_exploratory_support_gates_passed") is not True:
        raise ValueError("train support gates did not pass")
    if train_artifact.get("candidate_status") != "supported_on_train_requires_validation":
        raise ValueError("train artifact candidate status mismatch")
    if train_artifact.get("config", {}).get("sha256") != frozen["train_config"]["sha256"]:
        raise ValueError("train artifact does not pin the frozen config")
    if train_artifact.get("runner", {}).get("sha256") != frozen["train_runner_sha256"]:
        raise ValueError("train artifact runner provenance mismatch")
    if train_artifact.get("physics_module", {}).get("sha256") != frozen["physics_module_sha256"]:
        raise ValueError("train artifact physics provenance mismatch")

    validation_ids = [
        str(pair["paired_group_id"])
        for pair in split["pairs"]
        if pair["split"] == "validation"
    ]
    if validation_ids != EXPECTED_VALIDATION_IDS:
        raise ValueError("canonical validation pair roster changed")
    boundary = config.get("data_boundary", {})
    if boundary.get("authorized_partition") != "validation" or boundary.get("allowed_pair_ids") != validation_ids:
        raise ValueError("only validation pairs 007--009 are authorized")
    if boundary.get("train_access") != "frozen candidate records only":
        raise ValueError("train access must remain frozen-record only")
    if boundary.get("test_pairs_accessed") is not False:
        raise ValueError("test access must remain false")
    if boundary.get("texbat_recordings_accessed") != []:
        raise ValueError("TEXBAT access is forbidden in this validation")

    randomness = config.get("validation_randomness", {})
    if randomness != {
        "generator_seed": 2026082701,
        "bootstrap_seed": 2026082702,
        "event_count_per_geometry": 300,
        "bootstrap_repetitions": 1000,
    }:
        raise ValueError("validation randomness or sample count drifted")
    if int(randomness["event_count_per_geometry"]) != int(train_config["generator"]["event_count_per_geometry"]):
        raise ValueError("validation event count differs from train")
    if int(randomness["bootstrap_repetitions"]) != int(train_config["evaluation"]["bootstrap_repetitions"]):
        raise ValueError("validation bootstrap count differs from train")
    if config.get("frozen_support_rule") != train_config.get("exploratory_support_rule"):
        raise ValueError("validation support thresholds differ from train")

    los = config.get("los_generation", {})
    evidence = los.get("train_equivalence_evidence", {})
    if (
        int(los.get("duration_seconds", 0)) != 1
        or int(los.get("rf_sample_rate_hz", 0)) != 1_000_000
        or los.get("sample_format") != "s8_iq"
        or evidence.get("dynamic_pair_ids") != [f"pv1-pair-{index:03d}" for index in range(3, 7)]
        or evidence.get("prn_roster_exact_match") is not True
        or float(evidence.get("maximum_los_component_absolute_difference", math.inf)) != 0.0
    ):
        raise ValueError("startup LOS generation contract drifted")
    claim = config.get("claim_boundary", {})
    if claim.get("held_out_satellite_geometry_validation") is not True:
        raise ValueError("validation must remain limited to held-out geometry")
    if any(claim.get(key) is not False for key in (
        "actual_multipath_rf_generated",
        "actual_receiver_tracking_evaluated",
        "actual_receiver_complex_taps_evaluated",
    )):
        raise ValueError("claim boundary asserts an unperformed RF or receiver validation")

    if verify_inputs:
        for key, name in (("rinex_nav", "RINEX NAV"), ("simulator", "gps-sdr-sim")):
            source = los[key]
            path = _repo_path(source["path"])
            if _sha256(path) != source["sha256"]:
                raise ValueError(f"{name} integrity failure")
            if key == "simulator" and not path.is_file():
                raise ValueError("gps-sdr-sim executable is missing")
    return (
        train_config_path,
        train_config,
        train_record_path,
        train_record,
        train_artifact_path,
        train_artifact,
        split_path,
        split,
    )


def build_analysis_config(
    validation: dict[str, Any], train_config: dict[str, Any]
) -> dict[str, Any]:
    """Copy the frozen candidate, changing only validation seeds and pair boundary."""
    result = copy.deepcopy(train_config)
    randomness = validation["validation_randomness"]
    result["generator"]["seed"] = int(randomness["generator_seed"])
    result["evaluation"]["bootstrap_seed"] = int(randomness["bootstrap_seed"])
    result["data_boundary"] = {
        "allowed_partition": "validation",
        "allowed_pair_ids": list(validation["data_boundary"]["allowed_pair_ids"]),
        "validation_pairs_accessed": True,
        "test_pairs_accessed": False,
        "texbat_recordings_accessed": [],
    }
    result["los_sources"] = {}
    result["output_root"] = validation["output_root"]
    return result


def _generate_validation_los(
    config: dict[str, Any], split: dict[str, Any], output_root: Path
) -> tuple[dict[str, tuple[list[str], np.ndarray]], dict[str, Any]]:
    los_config = config["los_generation"]
    nav = _repo_path(los_config["rinex_nav"]["path"])
    executable = _repo_path(los_config["simulator"]["path"])
    runner = GpsSdrSimRunner(str(executable))
    pairs = {
        str(pair["paired_group_id"]): pair
        for pair in split["pairs"]
        if pair["split"] == "validation"
    }
    los_sets: dict[str, tuple[list[str], np.ndarray]] = {}
    provenance: dict[str, Any] = {}
    los_root = output_root / "los"
    los_root.mkdir(parents=True, exist_ok=False)
    for pair_id in config["data_boundary"]["allowed_pair_ids"]:
        pair = pairs[pair_id]
        position = pair["position"]
        scenario = Scenario(
            f"{pair_id}-startup-los",
            "GPS",
            "L1CA",
            datetime.fromisoformat(str(pair["utc"]).replace("Z", "+00:00")),
            int(los_config["duration_seconds"]),
            StaticPosition(
                float(position["latitude_deg"]),
                float(position["longitude_deg"]),
                float(position["altitude_m"]),
            ),
        )
        rf_config = RFGenerationConfig(
            version=1,
            scenario=scenario,
            input=InputConfig(nav),
            output=OutputConfig(
                los_root,
                int(los_config["rf_sample_rate_hz"]),
                str(los_config["sample_format"]),
            ),
            simulator=SimulatorConfig(str(executable)),
            impairments=clean_impairments(),
        )
        iq_path = los_root / f"{pair_id}-startup-s8-iq.bin"
        log_path = los_root / f"{pair_id}-gps-sdr-sim.log"
        run = runner.run(rf_config, iq_path, log_path)
        expected_bytes = runner.expected_output_bytes(rf_config)
        if iq_path.stat().st_size != expected_bytes:
            raise ValueError(f"startup IQ byte contract failed: {pair_id}")
        table = parse_gps_sdr_sim_los_table(log_path.read_text(encoding="utf-8"))
        prns = sorted(table)
        los = np.asarray([table[prn] for prn in prns], dtype=np.float64)
        rank = int(np.linalg.matrix_rank(np.column_stack((-los, np.ones(len(los))))))
        if len(prns) < int(train_audit.EXPECTED_TAPS.size - 1) or rank != 4:
            raise ValueError(f"validation LOS geometry is insufficient: {pair_id}")
        los_sets[pair_id] = (prns, los)
        provenance[pair_id] = {
            "pair_definition_sha256": _canonical_sha256(pair),
            "position": position,
            "original_domain": pair["domain"],
            "startup_position_approximation": pair["domain"] == "dynamic",
            "prns": prns,
            "prn_count": len(prns),
            "design_rank": rank,
            "iq": {"path": str(iq_path), "sha256": _sha256(iq_path), "bytes": iq_path.stat().st_size},
            "log": {"path": str(log_path), "sha256": _sha256(log_path)},
            "command": run["command"],
            "simulator_time": run["time"],
        }
    return los_sets, provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    started = time.time()
    config_path = _repo_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    (
        train_config_path,
        train_config,
        train_record_path,
        train_record,
        train_artifact_path,
        train_artifact,
        split_path,
        split,
    ) = validate_config(config, verify_inputs=True)
    output_root = _repo_path(config["output_root"])
    if output_root.exists():
        raise FileExistsError(
            f"locked validation output already exists; rerun is forbidden: {output_root}"
        )
    output_root.mkdir(parents=True)

    los_sets, los_provenance = _generate_validation_los(config, split, output_root)
    analysis_config = build_analysis_config(config, train_config)
    events, delays, assignments, template_manifest = train_audit._simulate(
        analysis_config, los_sets
    )
    diagnostic = train_audit._summarize(
        analysis_config, events, delays, assignments, template_manifest
    )
    confirmed = bool(diagnostic["all_exploratory_support_gates_passed"])

    event_path = output_root / "event_scores.csv"
    delay_path = output_root / "delay_estimates.csv"
    assignment_path = output_root / "profile_assignments.csv"
    template_path = output_root / "template_manifest.json"
    train_audit._atomic_frame(event_path, events)
    train_audit._atomic_frame(delay_path, delays)
    train_audit._atomic_frame(assignment_path, assignments)
    train_audit._atomic_json(template_path, template_manifest)

    train_complex_auc = float(train_record["metrics"]["complex_geometry_auc"]["estimate"])
    validation_complex_auc = float(diagnostic["geometry_auc"]["complex"]["estimate"])
    summary = {
        "schema": "gnss-doppler-lab.correlator-geometry-identifiability-validation",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "validation_status": (
            "confirmed_on_heldout_geometry" if confirmed else "not_confirmed_on_heldout_geometry"
        ),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        "frozen_candidate": {
            **config["frozen_candidate"],
            "train_config_path": str(train_config_path),
            "train_result_record_path": str(train_record_path),
            "train_artifact_summary_path": str(train_artifact_path),
        },
        "split_config": {"path": str(split_path), "sha256": _sha256(split_path)},
        "data_boundary": {
            **config["data_boundary"],
            "validation_pair_ids_accessed": list(los_sets),
            "test_pair_ids_accessed": [],
        },
        "los_generation": {
            "contract": config["los_generation"],
            "provenance": los_provenance,
        },
        "diagnostic": diagnostic,
        "generalization": {
            "train_complex_geometry_auc": train_complex_auc,
            "validation_complex_geometry_auc": validation_complex_auc,
            "validation_minus_train_complex_auc": validation_complex_auc - train_complex_auc,
            "validation_pair_complex_auc_min": min(
                item["complex_geometry_auc"]
                for item in diagnostic["per_geometry"].values()
            ),
            "validation_pair_complex_auc_max": max(
                item["complex_geometry_auc"]
                for item in diagnostic["per_geometry"].values()
            ),
        },
        "confirmation": {
            "frozen_support_rule": config["frozen_support_rule"],
            "all_frozen_gates_passed": confirmed,
            "decision": (
                "controlled CGC identifiability confirmed across held-out satellite geometries"
                if confirmed
                else "controlled CGC identifiability not confirmed on held-out satellite geometries"
            ),
        },
        "claim_boundary": [
            "This is held-out satellite-geometry validation of the controlled correlator-domain mechanism.",
            "The startup LOS table uses each dynamic pair's exact initial LLH; train dynamic pairs established exact startup-table equivalence to the full trajectory logs.",
            "No RF multipath was generated and no receiver tracking output was evaluated.",
            "Validation pairs 007--009 were accessed once; test pairs 010--012 and TEXBAT were not accessed.",
            "The result cannot establish field multipath false-alarm performance or a deployable threshold.",
        ],
        "next_gate": "implement and verify complex I/Q export for all nine GNSS-SDR taps, then challenge CGC with satellite-specific RF multipath before unlocking test",
        "outputs": {
            "event_scores": {"path": str(event_path), "sha256": _sha256(event_path)},
            "delay_estimates": {"path": str(delay_path), "sha256": _sha256(delay_path)},
            "profile_assignments": {"path": str(assignment_path), "sha256": _sha256(assignment_path)},
            "template_manifest": {"path": str(template_path), "sha256": _sha256(template_path)},
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = output_root / "summary.json"
    train_audit._atomic_json(summary_path, summary)
    print(json.dumps(train_audit._jsonable({
        "summary": str(summary_path),
        "validation_status": summary["validation_status"],
        "diagnostic": diagnostic,
        "generalization": summary["generalization"],
        "data_boundary": summary["data_boundary"],
    }), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
