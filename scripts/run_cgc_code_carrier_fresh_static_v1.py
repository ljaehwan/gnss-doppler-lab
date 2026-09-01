#!/usr/bin/env python3
"""Release the five-geometry fresh static Doppler-locked CGC test once."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_code_carrier_decoupling_pilot as pilot  # noqa: E402
import run_cgc_rf_challenge_pilot as challenge  # noqa: E402
import run_cgc_rf_geometry_aperture_validation as geometry  # noqa: E402
import run_simulation_v4_normal_independent_validation as normal  # noqa: E402
from gnss_doppler_lab.code_carrier_sim import (  # noqa: E402
    CodeCarrierGpsSdrSimRunner,
    DecoupledSimulationRequest,
    sha256,
    summarize_truth_triplet,
)
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.simulation_v4 import (  # noqa: E402
    SimulationScenario,
    SpoofEvent,
    build_carryoff_rows,
    compare_prefix,
    compose_paired_iq,
)


CONFIG = ROOT / "configs/experiments/cgc_code_carrier_fresh_static_v1.json"
PROTOCOL = ROOT / "docs/results/cgc_code_carrier_fresh_static_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-CGC-CODE-CARRIER-FRESH-STATIC-V1"
EXPECTED_IDS = [
    "ccfs-s1-a-tokyo",
    "ccfs-s2-a-london",
    "ccfs-s3-a-sao-paulo",
    "ccfs-s4-a-sydney",
    "ccfs-s5-a-nairobi",
]
RELEASE_INPUTS = (
    "configs/experiments/cgc_code_carrier_fresh_static_v1.json",
    "docs/results/cgc_code_carrier_fresh_static_protocol_v1.md",
    "docs/results/cgc_code_carrier_fresh_static_preflight_v1_summary.json",
    "scripts/run_cgc_code_carrier_fresh_static_v1.py",
    "scripts/run_cgc_code_carrier_decoupling_pilot.py",
    "src/gnss_doppler_lab/code_carrier_sim.py",
    "patches/gps-sdr-sim-code-carrier-decoupling-v1.patch",
)


def repo_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_pinned(record: dict[str, Any], label: str) -> tuple[Path, Any]:
    source = repo_path(record["path"])
    if not source.is_file() or sha256(source) != record["sha256"]:
        raise ValueError(f"pinned input mismatch: {label}")
    return source, json.loads(source.read_text(encoding="utf-8"))


def validate(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-code-carrier-fresh-static-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported fresh-static config")
    if config["experiment"].get("post_release_pair_substitution") is not False or config["experiment"].get("post_release_tuning_or_retest") is not False:
        raise ValueError("release mutation is forbidden")
    pool_path, pool = load_pinned(config["selection"]["candidate_pool"], "candidate pool")
    record_path, record = load_pinned(config["selection"]["preflight_record"], "preflight record")
    if record.get("score_accessed") is not False or record.get("probe_iq_retained") is not False or record.get("selected_candidate_ids") != EXPECTED_IDS:
        raise ValueError("support-only selection boundary drifted")
    pairs = config.get("pairs", [])
    if [row.get("candidate_id") for row in pairs] != EXPECTED_IDS or len(pairs) != 5:
        raise ValueError("selected pair roster drifted")
    pool_by_id = {row["candidate_id"]: row for row in pool["candidates"]}
    if any(row != pool_by_id[row["candidate_id"]] for row in pairs):
        raise ValueError("selected pair differs from frozen candidate pool")
    counts = config["selection"]["selected_startup_los_counts"]
    if counts != record["selected_startup_los_counts"] or min(counts.values()) < 10:
        raise ValueError("startup support proof drifted")
    for group in (config["inputs"], config["tools"]):
        for key, item in group.items():
            if key == "receiver":
                source = repo_path(item["path"])
                expected = item["sha256"]
            else:
                source = repo_path(item["path"])
                expected = item["sha256"]
            if not source.is_file() or sha256(source) != expected:
                raise ValueError(f"input mismatch: {key}")
    carryoff = config["carryoff"]
    if {key: carryoff[key] for key in ("duration_seconds", "start_seconds", "transition_seconds", "hold_start_seconds")} != {
        "duration_seconds": 30, "start_seconds": 5.0, "transition_seconds": 5.0, "hold_start_seconds": 12.0,
    }:
        raise ValueError("carry-off timing drifted")
    analysis = config["analysis"]
    if analysis["minimum_prns"] != 8 or analysis["partial_f_p_alarm_threshold"] != 0.06028418845288192 or analysis["persistence_rule"] != "3 of the latest 5 available one-second bins":
        raise ValueError("frozen detector drifted")
    if config["rf"] != {
        "sample_rate_hz": 25_000_000,
        "target_composite_cn0_db_hz": 60.5,
        "normal_target_rms": 22.0,
        "paired_frontend_noise_between_conditions": True,
    }:
        raise ValueError("RF contract drifted")
    expected_gates = {
        "required_pair_count": 5,
        "required_truth_invariant_pair_count": 5,
        "required_carrier_coupled_persistent_detection_count": 5,
        "minimum_doppler_locked_persistent_detection_count": 4,
        "maximum_total_pre_attack_persistent_alarm_count": 0,
        "minimum_median_carrier_coupled_hold_raw_alarm_rate": 0.9,
        "minimum_median_doppler_locked_hold_raw_alarm_rate": 0.5,
        "maximum_median_doppler_locked_latency_seconds": 10.0,
    }
    if config["evaluation"] != expected_gates:
        raise ValueError("evaluation gates drifted")
    if Path(config["output_root"]).resolve() != Path("/home/ubuntu/hdd_data/cgc_code_carrier_fresh_static_v1"):
        raise ValueError("output root drifted")
    normal_path, normal_profile = load_pinned(config["inputs"]["normal_profile"], "normal profile")
    controlled_path, controlled = load_pinned(config["inputs"]["controlled_template"], "controlled template")
    return {
        "pool_path": pool_path,
        "record_path": record_path,
        "normal_path": normal_path,
        "normal_profile": normal_profile,
        "controlled_path": controlled_path,
        "controlled": controlled,
        "output_root": Path(config["output_root"]).resolve(),
    }


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        git("ls-files", "--error-unmatch", relative)
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT).returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "head_commit": git("rev-parse", "HEAD"),
        "input_commits": {relative: git("log", "-1", "--format=%H", "--", relative) for relative in RELEASE_INPUTS},
    }


def start_release(config: dict[str, Any], context: dict[str, Any], resume: bool) -> tuple[Path, dict[str, Any]]:
    root = context["output_root"]
    state_path = root / "release_state.json"
    if not resume:
        if root.exists():
            raise FileExistsError(root)
        release = committed_release()
        state = {
            "schema": "gnss-doppler-lab.cgc-code-carrier-fresh-static-release",
            "schema_version": 1,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "released_before_25mhz_generation",
            "metrics_emitted": False,
            "config": {"path": str(CONFIG.resolve()), "sha256": sha256(CONFIG)},
            "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
            "release": release,
            "authorized_pair_ids": EXPECTED_IDS,
        }
        write_json(state_path, state)
        return state_path, state
    if not state_path.is_file():
        raise FileNotFoundError("resume requested without release state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["config"]["sha256"] != sha256(CONFIG) or state.get("metrics_emitted") is not False:
        raise ValueError("resume provenance mismatch or metrics already emitted")
    return state_path, state


def phase(state_path: Path, state: dict[str, Any], value: str) -> None:
    state["phase"] = value
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)


def spoof_event(config: dict[str, Any], pair: dict[str, Any]) -> SpoofEvent:
    item = config["carryoff"]
    return SpoofEvent(
        float(item["start_seconds"]), float(item["transition_seconds"]),
        tuple(float(value) for value in pair["target_offset_enu_m"]),
        float(item["initial_advantage_db"]), float(item["final_advantage_db"]),
        float(item["power_ramp_seconds"]),
    )


def ensure_trajectories(pair_root: Path, config: dict[str, Any], pair: dict[str, Any]) -> tuple[Path, Path]:
    directory = pair_root / "trajectories"; directory.mkdir(parents=True, exist_ok=True)
    authentic, false = directory / "authentic.csv", directory / "false_code.csv"
    position = pair["position"]
    rows = tuple(
        (index / 10.0, float(position["latitude_deg"]), float(position["longitude_deg"]), float(position["altitude_m"]))
        for index in range(int(config["carryoff"]["duration_seconds"]) * 10)
    )
    false_rows = build_carryoff_rows(rows, spoof_event(config, pair))
    for destination, values in ((authentic, rows), (false, false_rows)):
        payload = "".join(f"{time_s:.1f},{lat:.9f},{lon:.9f},{height:.4f}\n" for time_s, lat, lon, height in values)
        if destination.is_file() and destination.read_text(encoding="ascii") != payload:
            raise ValueError(f"trajectory resume mismatch: {destination}")
        if not destination.exists():
            destination.write_text(payload, encoding="ascii")
    write_json(directory / "manifest.json", {
        "pair": pair,
        "carryoff": asdict(spoof_event(config, pair)),
        "authentic": {"path": str(authentic.resolve()), "sha256": sha256(authentic)},
        "false_code": {"path": str(false.resolve()), "sha256": sha256(false)},
    })
    return authentic, false


def request(config: dict[str, Any], pair: dict[str, Any], code: Path, carrier: Path | None, mode: str) -> DecoupledSimulationRequest:
    return DecoupledSimulationRequest(
        repo_path(config["inputs"]["rinex_nav"]["path"]), code, carrier,
        datetime.fromisoformat(pair["utc"].replace("Z", "+00:00")),
        int(config["carryoff"]["duration_seconds"]), int(config["rf"]["sample_rate_hz"]), mode,
    )


def ensure_component(pair_root: Path, config: dict[str, Any], pair: dict[str, Any], authentic_path: Path, false_path: Path, name: str, resume: bool) -> tuple[Path, Path, Path, dict[str, Any]]:
    directory = pair_root / "components" / name
    iq, truth, log, manifest = directory / "gps_l1ca_s8_iq.bin", directory / "truth.csv", directory / "simulator.log", directory / "manifest.json"
    if manifest.is_file():
        if not resume:
            raise FileExistsError(manifest)
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if not iq.is_file() or sha256(iq) != document["iq"]["sha256"] or sha256(truth) != document["truth"]["sha256"]:
            raise ValueError(f"component integrity failure: {pair['candidate_id']} {name}")
        return iq, truth, log, document
    simulator = CodeCarrierGpsSdrSimRunner(
        repo_path(config["tools"]["decoupled_simulator"]["path"]),
        repo_path(config["tools"]["simulator_patch"]["path"]),
    )
    if name == "authentic":
        item = request(config, pair, authentic_path, None, "coupled")
    elif name == "carrier-coupled":
        item = request(config, pair, false_path, None, "coupled")
    elif name == "doppler-locked":
        item = request(config, pair, false_path, authentic_path, "doppler_locked")
    else:
        raise ValueError(name)
    record = simulator.run(item, iq, truth, log)
    document = {
        "schema": "gnss-doppler-lab.cgc-code-carrier-fresh-static-component",
        "schema_version": 1,
        "pair": pair,
        "condition": name,
        "config_sha256": sha256(CONFIG),
        **record,
        "log": {"path": str(log.resolve()), "sha256": sha256(log)},
    }
    write_json(manifest, document)
    return iq, truth, log, document


def truth_audit(pair_root: Path, config: dict[str, Any], truths: dict[str, Path]) -> tuple[dict[str, Any], bool]:
    metrics = summarize_truth_triplet(
        truths["authentic"], truths["carrier-coupled"], truths["doppler-locked"],
        hold_start_seconds=float(config["carryoff"]["hold_start_seconds"]),
    )
    gates = {
        "identical_code_range": metrics["locked_vs_coupled_code_range_max_abs_m"] <= 1e-6,
        "identical_code_rate": metrics["locked_vs_coupled_code_rate_max_abs_mps"] <= 1e-6,
        "locked_carrier_matches_authentic_range": metrics["locked_vs_authentic_carrier_range_max_abs_m"] <= 1e-6,
        "locked_carrier_matches_authentic_rate": metrics["locked_vs_authentic_carrier_rate_max_abs_mps"] <= 1e-6,
        "nonzero_code_carryoff": metrics["locked_code_vs_carrier_hold_max_abs_m"] >= 20.0,
    }
    document = {"metrics": metrics, "gates": gates, "passed": all(gates.values())}
    write_json(pair_root / "truth_audit.json", document)
    return document, bool(document["passed"])


def ensure_rf_pair(pair_root: Path, config: dict[str, Any], pair: dict[str, Any], authentic: Path, counterfeits: dict[str, Path], component_documents: dict[str, dict[str, Any]], resume: bool) -> dict[str, Path]:
    manifests = {mode: pair_root / "rf" / mode / "manifest.json" for mode in ("carrier-coupled", "doppler-locked")}
    iq_paths = {mode: manifest.parent / "gps_l1ca_s8_iq.bin" for mode, manifest in manifests.items()}
    profile = json.loads(repo_path(config["inputs"]["normal_profile"]["path"]).read_text(encoding="utf-8"))
    run = {**pair, "name": pair["candidate_id"], "duration_seconds": config["carryoff"]["duration_seconds"], "target_composite_cn0_db_hz": config["rf"]["target_composite_cn0_db_hz"]}
    receiver = normal._run_impairment(profile, run)
    event = spoof_event(config, pair)
    reference = None
    for mode in ("carrier-coupled", "doppler-locked"):
        if manifests[mode].exists():
            if not resume:
                raise FileExistsError(manifests[mode])
            document = json.loads(manifests[mode].read_text(encoding="utf-8"))
            if not iq_paths[mode].is_file() or sha256(iq_paths[mode]) != document["iq"]["sha256"]:
                raise ValueError(f"RF integrity failure: {pair['candidate_id']} {mode}")
            observed_reference = document["simulation_v4"]["receiver"]["reference"]
            if reference is None:
                reference = observed_reference
            elif reference != observed_reference:
                raise ValueError(f"paired RF reference mismatch: {pair['candidate_id']}")
            continue
        scenario = SimulationScenario(mode, "carryoff_spoof", spoofing=event)
        composition = compose_paired_iq(
            authentic, counterfeits[mode], {mode: iq_paths[mode]}, (scenario,),
            sample_rate_hz=int(config["rf"]["sample_rate_hz"]), receiver=receiver,
            normal_target_rms=float(config["rf"]["normal_target_rms"]), reference_override=reference,
        )
        if reference is None:
            reference = composition["reference"]
        report = composition["scenarios"][mode]
        document = {
            "schema_version": 4,
            "run_id": f"cgc-cc-fresh-{pair['candidate_id']}-{mode}",
            "scenario": {"name": mode, "campaign": config["experiment"]["name"], "class": "spoofing", "event": "carryoff", "is_spoofing": True, "domain": "static", "split": "fresh_static", **pair, "duration_seconds": config["carryoff"]["duration_seconds"]},
            "iq": {"path": iq_paths[mode].name, "sha256": report["sha256"], "actual_bytes": report["bytes"], "complex_samples": report["complex_samples"], "actual_duration_seconds": report["actual_duration_seconds"], "rf_sample_rate_hz": int(config["rf"]["sample_rate_hz"]), "sample_format": "s8_iq", "channels": 2},
            "simulation_v4": {"truth": {"condition": mode, "carryoff": asdict(event)}, "receiver": {"requested": receiver.manifest(), "reference": composition["reference"], "processing": composition["processing"]}, "measurements": report},
            "generation": {"config_sha256": sha256(CONFIG), "component": {"condition": mode, "sha256": component_documents[mode]["iq"]["sha256"]}},
            "claim_boundary": config["claim_boundary"],
        }
        write_json(manifests[mode], document)
    if reference is None:
        raise RuntimeError(f"missing authentic composition reference: {pair['candidate_id']}")
    prefix = compare_prefix(iq_paths["carrier-coupled"], iq_paths["doppler-locked"], int(float(config["carryoff"]["start_seconds"]) * int(config["rf"]["sample_rate_hz"])))
    if prefix["byte_identical"] is not True:
        raise RuntimeError(f"paired pre-onset RF mismatch: {pair['candidate_id']}")
    write_json(pair_root / "rf" / "paired_manifest.json", {
        "pair": pair,
        "prefix": prefix,
        "authentic_reference": reference,
        "condition_manifests": {mode: {"path": str(manifests[mode].resolve()), "sha256": sha256(manifests[mode])} for mode in manifests},
    })
    return manifests


def ensure_receivers(pair_root: Path, config: dict[str, Any], manifests: dict[str, Path], resume: bool) -> dict[str, Path]:
    receiver = config["tools"]["receiver"]
    receiver_config = {
        "executable": str(repo_path(receiver["path"])),
        "channel_count": int(receiver["channel_count"]),
        "timeout_seconds": int(receiver["timeout_seconds"]),
        "tracking_tap_spacing_chips": float(receiver["tracking_tap_spacing_chips"]),
    }
    outputs = {}
    for mode, manifest in manifests.items():
        expected_run_id = json.loads(manifest.read_text(encoding="utf-8"))["run_id"]
        expected = pair_root / "receiver" / expected_run_id / "manifest.json"
        outputs[mode] = challenge._ensure_receiver(manifest, pair_root / "receiver", receiver_config, resume=resume and expected.is_file())
    return outputs


def tracking_diagnostics(receiver_manifest: Path) -> dict[str, Any]:
    source = receiver_manifest.parent / "tracking.csv"
    selected = []
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if 8.3 <= float(row["time_s"]) < 8.7:
                selected.append(row)
    if not selected:
        return {"rows": 0, "median_carrier_lock_test": None, "median_cn0_db_hz": None}
    return {
        "rows": len(selected),
        "median_carrier_lock_test": float(np.median([float(row["carrier_lock_test"]) for row in selected])),
        "median_cn0_db_hz": float(np.median([float(row["CN0_SNV_dB_Hz"]) for row in selected])),
    }


def condition_metrics(scored: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    threshold = float(config["analysis"]["partial_f_p_alarm_threshold"])
    first_any, annotated = pilot.persistence(scored, threshold)
    attack_start = float(config["carryoff"]["start_seconds"])
    hold_start = float(config["analysis"]["primary_interval_start_seconds"])
    pre = [row for row in annotated if float(row["bin_start_s"]) < attack_start]
    hold = [row for row in annotated if float(row["bin_start_s"]) >= hold_start]
    first_after = next((float(row["bin_start_s"]) for row in annotated if float(row["bin_start_s"]) >= attack_start and row["persistent_spoof_alarm"]), None)
    return {
        "geometry_bin_count": len(scored),
        "hold_bin_count": len(hold),
        "minimum_hold_prns": min((int(row["prn_count"]) for row in hold), default=0),
        "hold_raw_alarm_rate": float(np.mean([bool(row["raw_spoof_alarm"]) for row in hold])) if hold else 0.0,
        "hold_persistent_alarm_rate": float(np.mean([bool(row["persistent_spoof_alarm"]) for row in hold])) if hold else 0.0,
        "median_hold_partial_f_p_value": float(np.median([float(row["partial_f_p_value"]) for row in hold])) if hold else None,
        "pre_attack_raw_alarm_count": sum(bool(row["raw_spoof_alarm"]) for row in pre),
        "pre_attack_persistent_alarm_count": sum(bool(row["persistent_spoof_alarm"]) for row in pre),
        "first_persistent_alarm_s": first_any,
        "first_persistent_alarm_at_or_after_onset_s": first_after,
        "detection_latency_from_onset_s": None if first_after is None else first_after - attack_start,
        "persistent_detection": first_after is not None,
    }, annotated


def analyze(config: dict[str, Any], context: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    estimator = challenge._estimator(context["controlled"])
    pair_results = []
    delay_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for pair in config["pairs"]:
        pair_id = pair["candidate_id"]
        record = runtime[pair_id]
        los = parse_gps_sdr_sim_los_table(record["authentic_log"].read_text(encoding="utf-8"))
        conditions = {}
        for mode in ("carrier-coupled", "doppler-locked"):
            delays, rows = geometry.analyze_stream(f"{pair_id}:{mode}", record["receivers"][mode], estimator, los, config, 9)
            scored = pilot.score_rows(rows)
            metrics, annotated = condition_metrics(scored, config)
            for row in delays:
                delay_rows.append({"pair_id": pair_id, "condition": mode, **row})
            for row in annotated:
                geometry_rows.append({"pair_id": pair_id, "condition": mode, **row})
            conditions[mode] = {
                "metrics": metrics,
                "tracking": tracking_diagnostics(record["receivers"][mode]),
                "receiver_manifest": {"path": str(record["receivers"][mode].resolve()), "sha256": sha256(record["receivers"][mode])},
            }
        pair_results.append({
            "pair_id": pair_id,
            "slot": pair["slot"],
            "position": pair["position"],
            "target_offset_enu_m": pair["target_offset_enu_m"],
            "startup_los_prn_count": config["selection"]["selected_startup_los_counts"][pair_id],
            "truth": record["truth_audit"],
            "conditions": conditions,
        })
    coupled = [row["conditions"]["carrier-coupled"]["metrics"] for row in pair_results]
    locked = [row["conditions"]["doppler-locked"]["metrics"] for row in pair_results]
    locked_latencies = [row["detection_latency_from_onset_s"] for row in locked if row["detection_latency_from_onset_s"] is not None]
    evaluation = config["evaluation"]
    aggregates = {
        "pair_count": len(pair_results),
        "truth_invariant_pair_count": sum(bool(row["truth"]["passed"]) for row in pair_results),
        "carrier_coupled_persistent_detection_count": sum(bool(row["persistent_detection"]) for row in coupled),
        "doppler_locked_persistent_detection_count": sum(bool(row["persistent_detection"]) for row in locked),
        "total_pre_attack_persistent_alarm_count": sum(int(row["pre_attack_persistent_alarm_count"]) for row in coupled + locked),
        "median_carrier_coupled_hold_raw_alarm_rate": float(np.median([row["hold_raw_alarm_rate"] for row in coupled])),
        "median_doppler_locked_hold_raw_alarm_rate": float(np.median([row["hold_raw_alarm_rate"] for row in locked])),
        "median_doppler_locked_latency_seconds": float(np.median(locked_latencies)) if locked_latencies else None,
    }
    gates = {
        "required_pair_count": aggregates["pair_count"] == evaluation["required_pair_count"],
        "truth_invariants": aggregates["truth_invariant_pair_count"] == evaluation["required_truth_invariant_pair_count"],
        "carrier_coupled_detection": aggregates["carrier_coupled_persistent_detection_count"] == evaluation["required_carrier_coupled_persistent_detection_count"],
        "doppler_locked_detection": aggregates["doppler_locked_persistent_detection_count"] >= evaluation["minimum_doppler_locked_persistent_detection_count"],
        "pre_attack_persistent_alarm": aggregates["total_pre_attack_persistent_alarm_count"] <= evaluation["maximum_total_pre_attack_persistent_alarm_count"],
        "carrier_coupled_hold_raw_alarm_rate": aggregates["median_carrier_coupled_hold_raw_alarm_rate"] >= evaluation["minimum_median_carrier_coupled_hold_raw_alarm_rate"],
        "doppler_locked_hold_raw_alarm_rate": aggregates["median_doppler_locked_hold_raw_alarm_rate"] >= evaluation["minimum_median_doppler_locked_hold_raw_alarm_rate"],
        "doppler_locked_latency": aggregates["median_doppler_locked_latency_seconds"] is not None and aggregates["median_doppler_locked_latency_seconds"] <= evaluation["maximum_median_doppler_locked_latency_seconds"],
    }
    return {
        "pairs": pair_results,
        "aggregates": aggregates,
        "gates": gates,
        "decision": "SUPPORTED" if all(gates.values()) else "NOT_SUPPORTED",
        "delay_rows": delay_rows,
        "geometry_rows": geometry_rows,
    }


def run(config: dict[str, Any], context: dict[str, Any], resume: bool, state_path: Path, state: dict[str, Any]) -> Path:
    runtime: dict[str, Any] = {}
    phase(state_path, state, "25mhz_component_generation")
    for pair in config["pairs"]:
        pair_id = pair["candidate_id"]
        print(f"[component] {pair_id}", flush=True)
        pair_root = context["output_root"] / "pairs" / pair_id
        authentic_path, false_path = ensure_trajectories(pair_root, config, pair)
        components = {}; truths = {}; documents = {}; logs = {}
        for name in ("authentic", "carrier-coupled", "doppler-locked"):
            iq, truth, log, document = ensure_component(pair_root, config, pair, authentic_path, false_path, name, resume)
            components[name], truths[name], logs[name], documents[name] = iq, truth, log, document
        audit, passed = truth_audit(pair_root, config, truths)
        if not passed:
            raise RuntimeError(f"truth invariant failure: {pair_id}")
        runtime[pair_id] = {"pair_root": pair_root, "components": components, "truth_audit": audit, "authentic_log": logs["authentic"], "documents": documents}
    phase(state_path, state, "paired_rf_composition")
    for pair in config["pairs"]:
        pair_id = pair["candidate_id"]
        print(f"[rf] {pair_id}", flush=True)
        record = runtime[pair_id]
        record["rf_manifests"] = ensure_rf_pair(record["pair_root"], config, pair, record["components"]["authentic"], {mode: record["components"][mode] for mode in ("carrier-coupled", "doppler-locked")}, record["documents"], resume)
    phase(state_path, state, "gnss_sdr_receiver")
    for pair in config["pairs"]:
        pair_id = pair["candidate_id"]
        print(f"[receiver] {pair_id}", flush=True)
        record = runtime[pair_id]
        record["receivers"] = ensure_receivers(record["pair_root"], config, record["rf_manifests"], resume)
    phase(state_path, state, "analysis_in_memory")
    outcome = analyze(config, context, runtime)
    analysis_root = context["output_root"] / "analysis"
    write_csv(analysis_root / "delay_estimates.csv", outcome.pop("delay_rows"))
    write_csv(analysis_root / "geometry_scores.csv", outcome.pop("geometry_rows"))
    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    summary = {
        "schema": "gnss-doppler-lab.cgc-code-carrier-fresh-static-result",
        "schema_version": 1,
        "role": config["experiment"]["role"],
        "config": {"path": str(CONFIG.resolve()), "sha256": sha256(CONFIG)},
        "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
        "release_state": {"path": str(state_path.resolve()), "sha256": sha256(state_path)},
        "selection": config["selection"],
        **outcome,
        "artifacts": {
            "delay_estimates": {"path": str((analysis_root / "delay_estimates.csv").resolve()), "sha256": sha256(analysis_root / "delay_estimates.csv")},
            "geometry_scores": {"path": str((analysis_root / "geometry_scores.csv").resolve()), "sha256": sha256(analysis_root / "geometry_scores.csv")},
        },
        "post_release_pair_substitution": False,
        "post_release_tuning_or_retest": False,
        "claim_boundary": config["claim_boundary"],
    }
    destination = context["output_root"] / "summary.json"
    write_json(destination, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--release-token", choices=[RELEASE_TOKEN])
    group.add_argument("--resume-before-metrics", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    context = validate(config)
    if args.validate_only:
        print("fresh static final config verified; no 25 MHz RF or CGC score accessed")
        return 0
    resume = bool(args.resume_before_metrics)
    state_path, state = start_release(config, context, resume)
    run(config, context, resume, state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
