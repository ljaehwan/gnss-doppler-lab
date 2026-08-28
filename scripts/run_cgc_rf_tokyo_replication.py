#!/usr/bin/env python3
"""Replicate the Tokyo 240 m CGC high-power reversal over five new RF seeds."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_rf_geometry_aperture_validation as ga  # noqa: E402
import run_cgc_rf_observability_anchors as anchor  # noqa: E402
import run_cgc_rf_transfer_sweep as transfer  # noqa: E402
import run_simulation_v4_normal_independent_validation as normal  # noqa: E402
from gnss_doppler_lab.simulation_v4 import SimulationScenario, compose_paired_iq  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_rf_tokyo_replication_v1.json"
PROTOCOL_PATH = REPO_ROOT / "docs/results/cgc_rf_tokyo_replication_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-CGC-RF-TOKYO-REPLICATION-V1"
RELEASE_INPUTS = (
    "configs/experiments/cgc_rf_tokyo_replication_v1.json",
    "docs/results/cgc_rf_tokyo_replication_protocol_v1.md",
    "scripts/run_cgc_rf_tokyo_replication.py",
)
REPLICA_IDS = ("r01", "r02", "r03", "r04", "r05")
RECEIVER_SEEDS = (20268101, 20268102, 20268103, 20268104, 20268105)
MULTIPATH_SEEDS = (20269101, 20269102, 20269103, 20269104, 20269105)
POWERS_DB = (-6.0, 3.0)


def sha256(path: str | Path) -> str:
    return ga.sha256(path)


def repo_path(path: str | Path) -> Path:
    return ga.repo_path(path)


def write_json(path: Path, document: dict[str, Any]) -> None:
    ga.write_json(path, document)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ga.write_csv(path, rows)


def verify_record(record: dict[str, Any], label: str) -> Path:
    path = repo_path(record["path"])
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"pinned input mismatch: {label}")
    return path


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-rf-tokyo-replication-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported Tokyo replication config")
    if config.get("experiment", {}).get("name") != "cgc-rf-tokyo-replication-v1":
        raise ValueError("experiment identity drifted")
    base_paths = {key: verify_record(record, f"base_campaign.{key}") for key, record in config["base_campaign"].items()}
    base_config = json.loads(base_paths["config"].read_text(encoding="utf-8"))
    base = ga.validate_config(base_config)
    if config.get("geometry_id") != "tokyo-straight" or float(config.get("distance_m")) != 240.0:
        raise ValueError("Tokyo geometry or distance drifted")
    if tuple(float(value) for value in config.get("final_advantages_db", [])) != POWERS_DB:
        raise ValueError("power roster drifted")
    replicas = config.get("replicas", [])
    if tuple(row.get("replica_id") for row in replicas) != REPLICA_IDS:
        raise ValueError("replica roster drifted")
    if tuple(int(row.get("receiver_seed")) for row in replicas) != RECEIVER_SEEDS:
        raise ValueError("receiver seed roster drifted")
    if tuple(int(row.get("multipath_seed")) for row in replicas) != MULTIPATH_SEEDS:
        raise ValueError("multipath seed roster drifted")
    analysis = config.get("analysis", {})
    expected_analysis = {
        "primary_aperture_taps": 9,
        "auc_threshold": 0.8,
        "direction_threshold": 0.85,
        "minimum_comparison_bins_per_stream": 8,
        "minimum_prns": 8,
        "systematic_min_replica_count": 4,
        "systematic_min_median_paired_auc_drop": 0.2,
    }
    if analysis != expected_analysis:
        raise ValueError("analysis contract drifted")
    output_root = repo_path(config["output_root"])
    if output_root != REPO_ROOT / "artifacts/cgc_rf_tokyo_replication_v1":
        raise ValueError("output root drifted")
    if config["retention"].get("shared_source_inputs_removed") is not False:
        raise ValueError("shared inputs may not be removed")
    return {
        "config": config,
        "base_config": base_config,
        "base": base,
        "base_paths": base_paths,
        "output_root": output_root,
    }


def condition_spec(power_db: float) -> dict[str, Any]:
    distance_m = 240.0
    return {
        "condition_id": f"tokyo-straight-{transfer.condition_id(power_db, distance_m)}",
        "geometry_id": "tokyo-straight",
        "distance_m": distance_m,
        "final_advantage_db": float(power_db),
        "target_offset_enu_m": [192.0, 144.0, 0.0],
        "transition_seconds": 12.0,
    }


def support_pass(row: dict[str, Any], analysis: dict[str, Any]) -> bool:
    minimum_bins = int(analysis["minimum_comparison_bins_per_stream"])
    minimum_prns = int(analysis["minimum_prns"])
    return (
        int(row["spoof_bin_count"]) >= minimum_bins
        and int(row["multipath_bin_count"]) >= minimum_bins
        and int(row["minimum_spoof_prn_count"]) >= minimum_prns
        and int(row["minimum_multipath_prn_count"]) >= minimum_prns
    )


def evaluate_replication(rows: list[dict[str, Any]], analysis: dict[str, Any]) -> dict[str, Any]:
    expected = {(replica_id, power) for replica_id in REPLICA_IDS for power in POWERS_DB}
    observed = {(str(row["replica_id"]), float(row["final_advantage_db"])) for row in rows}
    if len(rows) != 10 or observed != expected:
        raise ValueError("replication evaluation requires the complete five-by-two grid")
    auc_threshold = float(analysis["auc_threshold"])
    direction_threshold = float(analysis["direction_threshold"])
    per_replica: dict[str, dict[str, Any]] = {}
    for replica_id in REPLICA_IDS:
        by_power = {float(row["final_advantage_db"]): row for row in rows if row["replica_id"] == replica_id}
        low, high = by_power[-6.0], by_power[3.0]
        per_replica[replica_id] = {
            "support_pass": support_pass(low, analysis) and support_pass(high, analysis),
            "minus6_auc": float(low["serial_bin_auc"]),
            "plus3_auc": float(high["serial_bin_auc"]),
            "plus3_direction": float(high["median_absolute_direction_cosine"]),
            "paired_auc_drop_minus6_minus_plus3": float(low["serial_bin_auc"]) - float(high["serial_bin_auc"]),
        }
    support_count = sum(int(row["support_pass"]) for row in per_replica.values())
    minus6_pass_count = sum(int(row["minus6_auc"] >= auc_threshold) for row in per_replica.values())
    plus3_pass_count = sum(int(row["plus3_auc"] >= auc_threshold) for row in per_replica.values())
    plus3_fail_count = len(REPLICA_IDS) - plus3_pass_count
    plus3_direction_pass_count = sum(int(row["plus3_direction"] >= direction_threshold) for row in per_replica.values())
    median_drop = float(np.median([row["paired_auc_drop_minus6_minus_plus3"] for row in per_replica.values()]))
    minimum_count = int(analysis["systematic_min_replica_count"])
    systematic = (
        support_count == len(REPLICA_IDS)
        and minus6_pass_count >= minimum_count
        and plus3_fail_count >= minimum_count
        and plus3_direction_pass_count >= minimum_count
        and median_drop >= float(analysis["systematic_min_median_paired_auc_drop"])
    )
    single_exception = (
        support_count == len(REPLICA_IDS)
        and minus6_pass_count >= minimum_count
        and plus3_pass_count >= minimum_count
        and plus3_fail_count <= 1
        and plus3_direction_pass_count >= minimum_count
    )
    decision = "SYSTEMATIC_HIGH_POWER_BLIND_SPOT" if systematic else "SINGLE_REALIZATION_EXCEPTION" if single_exception else "MIXED_OR_UNRESOLVED"
    return {
        "decision": decision,
        "per_replica": per_replica,
        "support_replica_count": support_count,
        "minus6_auc_pass_count": minus6_pass_count,
        "plus3_auc_pass_count": plus3_pass_count,
        "plus3_auc_fail_count": plus3_fail_count,
        "plus3_direction_pass_count": plus3_direction_pass_count,
        "median_paired_auc_drop_minus6_minus_plus3": median_drop,
        "original_observed_tokyo_cell_in_primary_counts": False,
    }


def ensure_seed_normal_rf(root: Path, context: dict[str, Any], replica_id: str) -> tuple[Path, Path, dict[str, Any]]:
    directory = root / "seed_normal"
    iq_path = directory / "gps_l1ca_s8_iq.bin"
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not iq_path.is_file() or sha256(iq_path) != document["iq"]["sha256"]:
            raise ValueError("seed normal RF resume integrity failure")
        return iq_path, manifest_path, document
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"partial seed normal RF: {directory}")
    run, profile = context["run"], context["normal_profile"]
    receiver = normal._run_impairment(profile, run)
    scenario = SimulationScenario(f"tokyo-replication-{replica_id}-normal", "steady_normal")
    composition = compose_paired_iq(
        context["paths"]["authentic_component"],
        context["paths"]["authentic_component"],
        {scenario.name: iq_path},
        (scenario,),
        sample_rate_hz=int(profile["rf_profile"]["rf_sample_rate_hz"]),
        receiver=receiver,
        normal_target_rms=float(profile["normal_target_rms"]),
        reference_override=context["source_normal_manifest"]["validation"]["composition_reference"],
    )
    report = composition["scenarios"][scenario.name]
    document = {
        "schema": "gnss-doppler-lab.cgc-rf-tokyo-seed-normal",
        "schema_version": 1,
        "run_id": f"tr-{replica_id}-normal",
        "replica_id": replica_id,
        "run": run,
        "iq": {
            "path": iq_path.name,
            "sha256": report["sha256"],
            "actual_bytes": report["bytes"],
            "complex_samples": report["complex_samples"],
            "actual_duration_seconds": report["actual_duration_seconds"],
            "rf_sample_rate_hz": int(profile["rf_profile"]["rf_sample_rate_hz"]),
            "sample_format": "s8_iq",
            "channels": 2,
        },
        "validation": {
            "run": run,
            "composition_reference": composition["reference"],
            "processing": composition["processing"],
        },
        "source_authentic_component": {
            "path": str(context["paths"]["authentic_component"].resolve()),
            "sha256": sha256(context["paths"]["authentic_component"]),
        },
    }
    write_json(manifest_path, document)
    return iq_path, manifest_path, document


def replica_context(validated: dict[str, Any], replica: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base_context = validated["base"]["contexts"]["tokyo-straight"]
    run = deepcopy(base_context["run"])
    run["receiver_seed"] = int(replica["receiver_seed"])
    definition = deepcopy(base_context["definition"])
    definition["multipath_seed"] = int(replica["multipath_seed"])
    definition["run"] = run
    context = {
        **base_context,
        "definition": definition,
        "run": run,
        "normal_profile": validated["base"]["normal_profile"],
        "source_normal_manifest": base_context["normal_manifest"],
        "output_root": root,
    }
    seed_iq, seed_manifest_path, seed_manifest = ensure_seed_normal_rf(root, context, replica["replica_id"])
    context["normal_manifest"] = seed_manifest
    context["paths"] = {**context["paths"], "normal_rf_iq": seed_iq, "normal_rf_manifest": seed_manifest_path}
    return context, {"path": str(seed_iq.resolve()), "sha256": seed_manifest["iq"]["sha256"], "bytes": seed_manifest["iq"]["actual_bytes"]}


def remove_seed_normal(identity: dict[str, Any], path: Path, retention_path: Path) -> None:
    if retention_path.is_file():
        return
    if not path.is_file() or sha256(path) != identity["sha256"] or path.stat().st_size != int(identity["bytes"]):
        raise ValueError("refusing to remove changed seed normal IQ")
    path.unlink()
    write_json(retention_path, {
        "schema": "gnss-doppler-lab.cgc-rf-tokyo-seed-normal-retention",
        "schema_version": 1,
        "removed_intermediate": {**identity, "removed": True},
        "shared_source_inputs_removed": False,
        "receiver_outputs_retained": True,
    })


def run_replica(validated: dict[str, Any], config: dict[str, Any], replica: dict[str, Any]) -> list[dict[str, Any]]:
    replica_id = replica["replica_id"]
    root = validated["output_root"] / "replicas" / replica_id
    context, seed_identity = replica_context(validated, replica, root)
    estimators = {taps: ga._estimator(validated["base"]["controlled"], taps) for taps in ga.APERTURE_TAPS}
    multipath, _ = ga.prepare_multipath_control(root / "common_multipath", validated["base_config"], context, estimators)
    pair = ga.component_pair(context["run"], "tokyo-straight", 240.0, validated["base_config"])
    dpaths = transfer.distance_paths(root, 240.0)
    counterfeit, counterfeit_manifest_path, counterfeit_manifest = anchor.ensure_counterfeit(
        dpaths["root"], pair, context["normal_profile"], repo_path(validated["base_config"]["rf_tools"]["simulator"]["path"])
    )
    results: list[dict[str, Any]] = []
    for power_db in POWERS_DB:
        result = ga.run_condition(
            condition_spec(power_db), validated["base_config"], context, estimators,
            multipath, counterfeit, counterfeit_manifest_path, counterfeit_manifest,
        )
        results.append(result)
    counterfeit_identity = {
        "path": str(counterfeit.resolve()),
        "sha256": counterfeit_manifest["counterfeit"]["sha256"],
        "bytes": counterfeit_manifest["counterfeit"]["bytes"],
    }
    transfer.remove_single_iq(counterfeit, counterfeit_identity, dpaths["retention"], "shared_distance_counterfeit_iq")
    remove_seed_normal(seed_identity, Path(seed_identity["path"]), root / "seed_normal_retention.json")
    return results


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        git("ls-files", "--error-unmatch", relative)
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT).returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "head_commit": git("rev-parse", "HEAD"),
        "input_commits": {relative: git("log", "-1", "--format=%H", "--", relative) for relative in RELEASE_INPUTS},
        "runner_sha256": sha256(Path(__file__).resolve()),
    }


def start_release(config_path: Path, output_root: Path, resume: bool) -> tuple[Path, dict[str, Any]]:
    state_path = output_root / "release_state.json"
    commits = committed_release()
    if not resume:
        if output_root.exists():
            raise FileExistsError(output_root)
        output_root.mkdir(parents=True)
        state = {
            "schema": "gnss-doppler-lab.cgc-rf-tokyo-replication-release-state",
            "schema_version": 1,
            "phase": "released_before_new_seed_outcomes",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
            "commits": commits,
            "replica_ids": list(REPLICA_IDS),
            "powers_db": list(POWERS_DB),
            "metrics_emitted": False,
        }
        write_json(state_path, state)
        return state_path, state
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["config"]["sha256"] != sha256(config_path) or state["commits"]["runner_sha256"] != commits["runner_sha256"] or state.get("metrics_emitted") is not False:
        raise ValueError("resume release provenance mismatch")
    return state_path, state


def run(config_path: Path, *, resume: bool) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validated = validate_config(config)
    state_path, state = start_release(config_path, validated["output_root"], resume)
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for replica in config["replicas"]:
        state["phase"] = f"replica:{replica['replica_id']}"
        write_json(state_path, state)
        replica_results = run_replica(validated, config, replica)
        results.extend(replica_results)
        for result in replica_results:
            summary = result["summary_by_aperture"][str(config["analysis"]["primary_aperture_taps"])]
            rows.append({"replica_id": replica["replica_id"], "receiver_seed": replica["receiver_seed"], "multipath_seed": replica["multipath_seed"], **summary})
    evaluation = evaluate_replication(rows, config["analysis"])
    table_path = validated["output_root"] / "condition_summary_9tap.csv"
    write_csv(table_path, rows)
    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    summary = {
        "schema": "gnss-doppler-lab.cgc-rf-tokyo-replication-result",
        "schema_version": 1,
        "decision": evaluation["decision"],
        "evaluation": evaluation,
        "condition_count": len(results),
        "primary_row_count": len(rows),
        "full_grid_reported": len(results) == 10 and len(rows) == 10,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "release_state": {"path": str(state_path.resolve()), "sha256": sha256(state_path)},
        "artifact": {"path": str(table_path.resolve()), "sha256": sha256(table_path), "row_count": len(rows)},
        "retention": config["retention"],
        "claim_boundary": config["claim_boundary"],
        "post_release_seed_power_estimator_or_gate_substitution": False,
    }
    summary_path = validated["output_root"] / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "decision": evaluation["decision"], "full_grid_reported": summary["full_grid_reported"]}, indent=2, sort_keys=True))
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-token", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.release_token != RELEASE_TOKEN:
        raise ValueError("explicit release token is required")
    run(args.config.resolve(), resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
