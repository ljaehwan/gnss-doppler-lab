#!/usr/bin/env python3
"""Run the preregistered untouched static temporal-CGC receiver-RF release."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import evaluate_cgc_temporal_stabilization_dev as temporal  # noqa: E402
import run_cgc_code_carrier_fresh_static_v1 as base  # noqa: E402
import run_cgc_rf_challenge_pilot as challenge  # noqa: E402
import run_cgc_rf_geometry_aperture_validation as geometry  # noqa: E402
import run_simulation_v4_normal_independent_validation as normal  # noqa: E402
from gnss_doppler_lab.code_carrier_sim import (  # noqa: E402
    CodeCarrierGpsSdrSimRunner, DecoupledSimulationRequest, sha256,
)
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.rf_config import (  # noqa: E402
    InputConfig, OutputConfig, RFGenerationConfig, Scenario, SimulatorConfig,
    TrajectoryPosition,
)
from gnss_doppler_lab.rf_impairments import ImpairmentConfig  # noqa: E402
from gnss_doppler_lab.satellite_multipath import (  # noqa: E402
    PrnMultipathGpsSdrSimRunner, independent_echoes,
)
from gnss_doppler_lab.simulation_v4 import (  # noqa: E402
    SimulationScenario, compare_prefix, compose_paired_iq,
)


CONFIG = ROOT / "configs/experiments/cgc_temporal_final_static_v1.json"
PROTOCOL = ROOT / "docs/results/cgc_temporal_final_static_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-CGC-TEMPORAL-FINAL-STATIC-V1"
EXPECTED_IDS = [
    "ctfs-s1-a-fairbanks",
    "ctfs-s2-a-punta-arenas",
    "ctfs-s3-a-casablanca",
    "ctfs-s4-a-sapporo",
    "ctfs-s5-a-prince-george",
]
CONDITIONS = ("normal", "multipath", "carrier-coupled", "doppler-locked")
RELEASE_INPUTS = (
    "configs/experiments/cgc_temporal_final_static_v1.json",
    "docs/results/cgc_temporal_final_static_protocol_v1.md",
    "docs/results/cgc_temporal_final_static_preflight_v1_summary.json",
    "scripts/run_cgc_temporal_final_static_v1.py",
    "scripts/evaluate_cgc_temporal_stabilization_dev.py",
    "src/gnss_doppler_lab/code_carrier_sim.py",
    "src/gnss_doppler_lab/temporal_cgc.py",
    "patches/gps-sdr-sim-code-carrier-decoupling-v1.patch",
    "patches/gps-sdr-sim-code-carrier-prn-phase-v1.patch",
    "patches/gps-sdr-sim-prn-multipath-v1.patch",
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
        writer.writeheader()
        writer.writerows(rows)


def load_pinned(record: dict[str, Any], label: str) -> tuple[Path, Any]:
    source = repo_path(record["path"])
    if not source.is_file() or sha256(source) != record["sha256"]:
        raise ValueError(f"pinned input mismatch: {label}")
    return source, json.loads(source.read_text(encoding="utf-8"))


def validate(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-temporal-final-static-config":
        raise ValueError("unsupported final-static config schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported final-static config version")
    experiment = config["experiment"]
    if experiment.get("post_release_pair_substitution") is not False:
        raise ValueError("pair substitution must be forbidden")
    if experiment.get("post_release_tuning_or_retest") is not False:
        raise ValueError("post-release tuning must be forbidden")
    pool_path, pool = load_pinned(config["selection"]["candidate_pool"], "candidate pool")
    record_path, record = load_pinned(config["selection"]["preflight_record"], "preflight record")
    if record.get("score_accessed") is not False or record.get("probe_iq_retained") is not False:
        raise ValueError("preflight boundary drifted")
    if record.get("selected_candidate_ids") != EXPECTED_IDS:
        raise ValueError("preflight selection drifted")
    pairs = config["pairs"]
    if [row["candidate_id"] for row in pairs] != EXPECTED_IDS or len(pairs) != 5:
        raise ValueError("final pair roster drifted")
    pool_by_id = {row["candidate_id"]: row for row in pool["candidates"]}
    for pair in pairs:
        expected = pool_by_id[pair["candidate_id"]]
        for key in (
            "candidate_id", "slot", "utc", "position", "target_offset_enu_m",
            "receiver_seed", "counterfeit_phase_seed",
        ):
            if pair[key] != expected[key]:
                raise ValueError(f"selected pair field drifted: {pair['candidate_id']} {key}")
    counts = config["selection"]["selected_startup_los_counts"]
    if counts != record["selected_startup_los_counts"] or min(counts.values()) < 10:
        raise ValueError("startup support proof drifted")
    for group in (config["inputs"], config["tools"]):
        for key, item in group.items():
            source = repo_path(item["path"])
            if not source.is_file() or sha256(source) != item["sha256"]:
                raise ValueError(f"input mismatch: {key}")
    carryoff = config["carryoff"]
    timing = tuple(carryoff[key] for key in (
        "duration_seconds", "start_seconds", "transition_seconds", "hold_start_seconds",
    ))
    if timing != (30, 5.0, 5.0, 12.0):
        raise ValueError("carry-off timing drifted")
    analysis = config["analysis"]
    expected_analysis = {
        "minimum_prns": 8,
        "tap_count": 9,
        "causal_prn_median_window_bins": 5,
        "centered_delay_rms_observable_threshold_chips": 0.10,
        "partial_f_p_alarm_threshold": 0.06028418845288192,
    }
    if any(analysis[key] != value for key, value in expected_analysis.items()):
        raise ValueError("frozen detector drifted")
    if temporal.SELECTED_WINDOW != 5:
        raise ValueError("temporal implementation window drifted")
    if temporal.DIAGNOSTIC_RMS_THRESHOLD_CHIPS != 0.10:
        raise ValueError("temporal observability implementation drifted")
    if temporal.P_THRESHOLD != 0.06028418845288192:
        raise ValueError("Partial-F implementation threshold drifted")
    if config["multipath"] != {
        "delay_chips_range": [0.12, 0.45],
        "amplitude_range": [0.20, 0.70],
        "echo_prns": "all startup LOS PRNs",
        "independent_per_prn_delay_amplitude_phase": True,
    }:
        raise ValueError("multipath contract drifted")
    if Path(config["output_root"]).resolve() != Path(
        "/home/ubuntu/hdd_data/cgc_temporal_final_static_v1"
    ):
        raise ValueError("output root drifted")
    normal_path, normal_profile = load_pinned(config["inputs"]["normal_profile"], "normal profile")
    controlled_path, controlled = load_pinned(
        config["inputs"]["controlled_template"], "controlled template",
    )
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
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        git("ls-files", "--error-unmatch", relative)
        if subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT,
        ).returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "head_commit": git("rev-parse", "HEAD"),
        "input_commits": {
            relative: git("log", "-1", "--format=%H", "--", relative)
            for relative in RELEASE_INPUTS
        },
    }


def start_release(
    config: dict[str, Any], context: dict[str, Any], resume: bool,
) -> tuple[Path, dict[str, Any]]:
    root = context["output_root"]
    state_path = root / "release_state.json"
    if not resume:
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        state = {
            "schema": "gnss-doppler-lab.cgc-temporal-final-static-release",
            "schema_version": 1,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "released_before_selected_25mhz_generation",
            "metrics_emitted": False,
            "config": {"path": str(CONFIG.resolve()), "sha256": sha256(CONFIG)},
            "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
            "release": committed_release(),
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


def set_phase(state_path: Path, state: dict[str, Any], value: str) -> None:
    state["phase"] = value
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)


def request(
    config: dict[str, Any], pair: dict[str, Any], code: Path,
    carrier: Path | None, mode: str, phase_seed: int | None,
) -> DecoupledSimulationRequest:
    return DecoupledSimulationRequest(
        repo_path(config["inputs"]["rinex_nav"]["path"]),
        code,
        carrier,
        datetime.fromisoformat(pair["utc"].replace("Z", "+00:00")),
        int(config["carryoff"]["duration_seconds"]),
        int(config["rf"]["sample_rate_hz"]),
        mode,
        carrier_phase_seed=phase_seed,
    )


def ensure_decoupled_component(
    pair_root: Path, config: dict[str, Any], pair: dict[str, Any],
    authentic_path: Path, false_path: Path, name: str, resume: bool,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    directory = pair_root / "components" / name
    iq = directory / "gps_l1ca_s8_iq.bin"
    truth = directory / "truth.csv"
    log = directory / "simulator.log"
    manifest = directory / "manifest.json"
    if manifest.is_file():
        if not resume:
            raise FileExistsError(manifest)
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if not iq.is_file() or sha256(iq) != document["iq"]["sha256"]:
            raise ValueError(f"component integrity failure: {pair['candidate_id']} {name}")
        if not truth.is_file() or sha256(truth) != document["truth"]["sha256"]:
            raise ValueError(f"truth integrity failure: {pair['candidate_id']} {name}")
        return iq, truth, log, document
    simulator = CodeCarrierGpsSdrSimRunner(
        repo_path(config["tools"]["decoupled_phase_simulator"]["path"]),
        repo_path(config["tools"]["phase_patch"]["path"]),
    )
    phase_seed = None if name == "authentic" else int(pair["counterfeit_phase_seed"])
    if name == "authentic":
        item = request(config, pair, authentic_path, None, "coupled", phase_seed)
    elif name == "carrier-coupled":
        item = request(config, pair, false_path, None, "coupled", phase_seed)
    elif name == "doppler-locked":
        item = request(config, pair, false_path, authentic_path, "doppler_locked", phase_seed)
    else:
        raise ValueError(name)
    record = simulator.run(item, iq, truth, log)
    document = {
        "schema": "gnss-doppler-lab.cgc-temporal-final-component",
        "schema_version": 1,
        "pair": pair,
        "condition": name,
        "config_sha256": sha256(CONFIG),
        **record,
        "decoupling_patch": config["tools"]["decoupling_patch"],
        "phase_patch": config["tools"]["phase_patch"],
        "log": {"path": str(log.resolve()), "sha256": sha256(log)},
    }
    write_json(manifest, document)
    return iq, truth, log, document


def motion_rows(path: Path) -> tuple[tuple[float, float, float, float], ...]:
    rows = []
    with path.open(newline="", encoding="ascii") as stream:
        for row in csv.reader(stream):
            rows.append(tuple(float(value) for value in row))
    return tuple(rows)


def ensure_multipath_component(
    pair_root: Path, config: dict[str, Any], pair: dict[str, Any],
    authentic_path: Path, authentic_log: Path, resume: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    directory = pair_root / "components" / "multipath"
    iq = directory / "gps_l1ca_s8_iq.bin"
    log = directory / "simulator.log"
    manifest = directory / "manifest.json"
    if manifest.is_file():
        if not resume:
            raise FileExistsError(manifest)
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if not iq.is_file() or sha256(iq) != document["iq"]["sha256"]:
            raise ValueError(f"multipath component integrity failure: {pair['candidate_id']}")
        return iq, log, document
    los = parse_gps_sdr_sim_los_table(authentic_log.read_text(encoding="utf-8"))
    prns = [int(prn[1:]) for prn in sorted(los)]
    specification = config["multipath"]
    echoes = independent_echoes(
        prns,
        seed=int(pair["multipath_seed"]),
        delay_chips_range=tuple(float(x) for x in specification["delay_chips_range"]),
        amplitude_range=tuple(float(x) for x in specification["amplitude_range"]),
    )
    trajectory = TrajectoryPosition(
        path=authentic_path.resolve(),
        coordinate_system="llh",
        rows=motion_rows(authentic_path),
        csv_sha256=sha256(authentic_path),
        metadata_path=None,
        metadata_sha256=None,
    )
    rf_config = RFGenerationConfig(
        version=1,
        scenario=Scenario(
            f"{pair['candidate_id']}-multipath-component", "GPS", "L1CA",
            datetime.fromisoformat(pair["utc"].replace("Z", "+00:00")),
            int(config["carryoff"]["duration_seconds"]), trajectory,
        ),
        input=InputConfig(repo_path(config["inputs"]["rinex_nav"]["path"])),
        output=OutputConfig(directory, int(config["rf"]["sample_rate_hz"]), "s8_iq"),
        simulator=SimulatorConfig(str(repo_path(config["tools"]["multipath_simulator"]["path"]))),
        impairments=ImpairmentConfig(),
    )
    runner = PrnMultipathGpsSdrSimRunner(rf_config.simulator.executable, echoes)
    result = runner.run(rf_config, iq, log)
    if int(result["actual_bytes"]) != runner.expected_output_bytes(rf_config):
        raise RuntimeError("multipath simulator byte contract failed")
    document = {
        "schema": "gnss-doppler-lab.cgc-temporal-final-multipath-component",
        "schema_version": 1,
        "pair": pair,
        "config_sha256": sha256(CONFIG),
        "simulator": config["tools"]["multipath_simulator"],
        "patch": config["tools"]["multipath_patch"],
        "command": result["command"],
        "multipath": result["multipath"],
        "iq": {
            "path": str(iq.resolve()), "sha256": sha256(iq),
            "bytes": iq.stat().st_size,
        },
        "log": {"path": str(log.resolve()), "sha256": sha256(log)},
    }
    write_json(manifest, document)
    return iq, log, document


def rf_manifest(
    config: dict[str, Any], pair: dict[str, Any], condition: str, path: Path,
    report: dict[str, Any], receiver: Any, reference: dict[str, Any],
    processing: dict[str, Any], truth: dict[str, Any], source: dict[str, Any],
) -> Path:
    manifest = path.parent / "manifest.json"
    document = {
        "schema_version": 4,
        "run_id": f"cgc-temporal-final-{pair['candidate_id']}-{condition}",
        "scenario": {
            "name": condition,
            "campaign": config["experiment"]["name"],
            "class": "spoofing" if condition in {"carrier-coupled", "doppler-locked"} else condition,
            "event": "carryoff" if condition in {"carrier-coupled", "doppler-locked"} else "steady",
            "is_spoofing": condition in {"carrier-coupled", "doppler-locked"},
            "domain": "static",
            "split": "untouched_final_static",
            "paired_group_id": pair["candidate_id"],
            "utc": pair["utc"],
            "duration_seconds": config["carryoff"]["duration_seconds"],
            "position": pair["position"],
        },
        "iq": {
            "path": path.name,
            "sha256": report["sha256"],
            "actual_bytes": report["bytes"],
            "complex_samples": report["complex_samples"],
            "actual_duration_seconds": report["actual_duration_seconds"],
            "rf_sample_rate_hz": int(config["rf"]["sample_rate_hz"]),
            "sample_format": "s8_iq",
            "channels": 2,
        },
        "simulation_v4": {
            "truth": truth,
            "receiver": {
                "requested": receiver.manifest(),
                "reference": reference,
                "processing": processing,
            },
            "measurements": report,
            "source": source,
            "scope": config["claim_boundary"],
        },
        "generation": {"config_sha256": sha256(CONFIG)},
    }
    write_json(manifest, document)
    return manifest


def ensure_rf_conditions(
    pair_root: Path, config: dict[str, Any], context: dict[str, Any],
    pair: dict[str, Any], components: dict[str, Path],
    documents: dict[str, dict[str, Any]], resume: bool,
) -> dict[str, Path]:
    paths = {
        condition: pair_root / "rf" / condition / "gps_l1ca_s8_iq.bin"
        for condition in CONDITIONS
    }
    manifests = {condition: path.parent / "manifest.json" for condition, path in paths.items()}
    if all(manifest.is_file() for manifest in manifests.values()):
        if not resume:
            raise FileExistsError(pair_root / "rf")
        for condition in CONDITIONS:
            document = json.loads(manifests[condition].read_text(encoding="utf-8"))
            if not paths[condition].is_file() or sha256(paths[condition]) != document["iq"]["sha256"]:
                raise ValueError(f"RF integrity failure: {pair['candidate_id']} {condition}")
        return manifests
    if any(manifest.exists() or path.exists() for manifest, path in zip(manifests.values(), paths.values())):
        raise FileExistsError(f"partial RF condition set: {pair['candidate_id']}")
    run = {
        **pair,
        "name": pair["candidate_id"],
        "duration_seconds": config["carryoff"]["duration_seconds"],
        "target_composite_cn0_db_hz": config["rf"]["target_composite_cn0_db_hz"],
    }
    receiver = normal._run_impairment(context["normal_profile"], run)
    event = base.spoof_event(config, pair)
    normal_scenario = SimulationScenario("normal", "steady_normal")
    coupled_scenario = SimulationScenario("carrier-coupled", "carryoff_spoof", spoofing=event)
    first = compose_paired_iq(
        components["authentic"], components["carrier-coupled"],
        {"normal": paths["normal"], "carrier-coupled": paths["carrier-coupled"]},
        (normal_scenario, coupled_scenario),
        sample_rate_hz=int(config["rf"]["sample_rate_hz"]),
        receiver=receiver,
        normal_target_rms=float(config["rf"]["normal_target_rms"]),
    )
    locked_scenario = SimulationScenario("doppler-locked", "carryoff_spoof", spoofing=event)
    locked = compose_paired_iq(
        components["authentic"], components["doppler-locked"],
        {"doppler-locked": paths["doppler-locked"]}, (locked_scenario,),
        sample_rate_hz=int(config["rf"]["sample_rate_hz"]),
        receiver=receiver,
        normal_target_rms=float(config["rf"]["normal_target_rms"]),
        reference_override=first["reference"],
    )
    multipath_scenario = SimulationScenario("multipath", "steady_normal")
    multipath = compose_paired_iq(
        components["multipath"], components["multipath"],
        {"multipath": paths["multipath"]}, (multipath_scenario,),
        sample_rate_hz=int(config["rf"]["sample_rate_hz"]),
        receiver=receiver,
        normal_target_rms=float(config["rf"]["normal_target_rms"]),
        reference_override=first["reference"],
    )
    payloads = {
        "normal": (first, {"class": "normal", "is_spoofing": False}),
        "carrier-coupled": (
            first, {"class": "spoofing", "is_spoofing": True,
                    "carryoff": base.asdict(event), "carrier_mode": "coupled"},
        ),
        "doppler-locked": (
            locked, {"class": "spoofing", "is_spoofing": True,
                     "carryoff": base.asdict(event), "carrier_mode": "authentic-doppler-locked"},
        ),
        "multipath": (
            multipath, {"class": "multipath", "is_spoofing": False,
                        "echoes": documents["multipath"]["multipath"]["echoes"]},
        ),
    }
    for condition, (composition, truth) in payloads.items():
        source = (
            documents["multipath"]["iq"] if condition == "multipath"
            else documents[condition if condition != "normal" else "authentic"]["iq"]
        )
        manifests[condition] = rf_manifest(
            config, pair, condition, paths[condition], composition["scenarios"][condition],
            receiver, composition["reference"], composition["processing"], truth, source,
        )
    onset_samples = int(config["carryoff"]["start_seconds"] * config["rf"]["sample_rate_hz"])
    prefixes = {
        condition: compare_prefix(paths["normal"], paths[condition], onset_samples)
        for condition in ("carrier-coupled", "doppler-locked")
    }
    if not all(record["byte_identical"] for record in prefixes.values()):
        raise RuntimeError(f"spoof pre-onset RF mismatch: {pair['candidate_id']}")
    write_json(pair_root / "rf" / "paired_manifest.json", {
        "pair": pair,
        "pre_onset_prefixes": prefixes,
        "frontend_reference": first["reference"],
        "condition_manifests": {
            condition: {"path": str(path.resolve()), "sha256": sha256(path)}
            for condition, path in manifests.items()
        },
    })
    return manifests


def ensure_receivers(
    pair_root: Path, config: dict[str, Any], manifests: dict[str, Path], resume: bool,
) -> dict[str, Path]:
    receiver = config["tools"]["receiver"]
    receiver_config = {
        "executable": str(repo_path(receiver["path"])),
        "channel_count": int(receiver["channel_count"]),
        "timeout_seconds": int(receiver["timeout_seconds"]),
        "tracking_tap_spacing_chips": float(receiver["tracking_tap_spacing_chips"]),
    }
    outputs = {}
    for condition, manifest in manifests.items():
        run_id = json.loads(manifest.read_text(encoding="utf-8"))["run_id"]
        expected = pair_root / "receiver" / run_id / "manifest.json"
        outputs[condition] = challenge._ensure_receiver(
            manifest, pair_root / "receiver", receiver_config,
            resume=resume and expected.is_file(),
        )
    return outputs


def remove_iq(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "path": str(path.resolve()),
        "sha256": expected["sha256"],
        "bytes": int(expected.get("bytes", expected.get("actual_bytes"))),
    }
    if path.is_file():
        if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
            raise ValueError(f"refusing to remove changed IQ: {path}")
        path.unlink()
    return {**identity, "removed_after_receiver_success": True}


def process_pair(
    config: dict[str, Any], context: dict[str, Any], pair: dict[str, Any], resume: bool,
) -> dict[str, Any]:
    pair_root = context["output_root"] / "pairs" / pair["candidate_id"]
    complete = pair_root / "pair_complete.json"
    if complete.is_file():
        if not resume:
            raise FileExistsError(complete)
        document = json.loads(complete.read_text(encoding="utf-8"))
        for record in document["receivers"].values():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(f"completed receiver provenance failure: {path}")
        return document
    authentic_path, false_path = base.ensure_trajectories(pair_root, config, pair)
    components: dict[str, Path] = {}
    truths: dict[str, Path] = {}
    documents: dict[str, dict[str, Any]] = {}
    logs: dict[str, Path] = {}
    for name in ("authentic", "carrier-coupled", "doppler-locked"):
        iq, truth, log, document = ensure_decoupled_component(
            pair_root, config, pair, authentic_path, false_path, name, resume,
        )
        components[name], truths[name], logs[name], documents[name] = iq, truth, log, document
    multipath_iq, multipath_log, multipath_document = ensure_multipath_component(
        pair_root, config, pair, authentic_path, logs["authentic"], resume,
    )
    components["multipath"] = multipath_iq
    logs["multipath"] = multipath_log
    documents["multipath"] = multipath_document
    truth_audit, passed = base.truth_audit(pair_root, config, truths)
    if not passed:
        raise RuntimeError(f"truth invariant failure: {pair['candidate_id']}")
    manifests = ensure_rf_conditions(
        pair_root, config, context, pair, components, documents, resume,
    )
    receivers = ensure_receivers(pair_root, config, manifests, resume)
    document = {
        "pair": pair,
        "truth_audit": truth_audit,
        "authentic_los_log": {
            "path": str(logs["authentic"].resolve()), "sha256": sha256(logs["authentic"]),
        },
        "receivers": {
            condition: {"path": str(path.resolve()), "sha256": sha256(path)}
            for condition, path in receivers.items()
        },
    }
    write_json(complete, document)
    retention = []
    for name, path in components.items():
        retention.append(remove_iq(path, documents[name]["iq"]))
    for condition, manifest in manifests.items():
        rf_document = json.loads(manifest.read_text(encoding="utf-8"))
        retention.append(remove_iq(manifest.parent / rf_document["iq"]["path"], rf_document["iq"]))
    write_json(pair_root / "retention.json", {"removed_iq": retention})
    return document


def condition_metrics(
    rows: list[dict[str, Any]], condition: str, config: dict[str, Any],
) -> dict[str, Any]:
    analysis = config["analysis"]
    if condition in {"normal", "multipath"}:
        selected = [
            row for row in rows
            if int(row["bin_index"]) >= int(analysis["benign_evaluation_start_bin"])
        ]
    else:
        selected = [
            row for row in rows
            if float(row["bin_start_s"]) >= float(analysis["spoof_hold_interval_start_seconds"])
        ]
    pre = [row for row in rows if float(row["bin_start_s"]) < 5.0]
    after_onset = [row for row in rows if float(row["bin_start_s"]) >= 5.0]
    first = next(
        (float(row["bin_start_s"]) for row in after_onset if row["persistent_spoof_alarm"]),
        None,
    )
    return {
        "scored_bin_count": len(rows),
        "evaluation_bin_count": len(selected),
        "minimum_evaluation_prns": min((int(row["prn_count"]) for row in selected), default=0),
        "supported": bool(selected) and min(int(row["prn_count"]) for row in selected) >= 8,
        "observable_bin_rate": float(np.mean([row["observable"] for row in selected])) if selected else 0.0,
        "raw_alarm_rate": float(np.mean([row["raw_spoof_alarm"] for row in selected])) if selected else 0.0,
        "persistent_alarm_rate": float(np.mean([row["persistent_spoof_alarm"] for row in selected])) if selected else 0.0,
        "persistent_alarm_any": any(row["persistent_spoof_alarm"] for row in selected),
        "median_partial_f_p_value": float(np.median([row["partial_f_p_value"] for row in selected])) if selected else None,
        "median_centered_delay_rms_chips": float(np.median([row["centered_delay_rms_chips"] for row in selected])) if selected else None,
        "pre_attack_persistent_alarm_count": int(sum(row["persistent_spoof_alarm"] for row in pre)),
        "first_persistent_alarm_at_or_after_onset_s": first,
        "detection_latency_from_onset_s": None if first is None else first - 5.0,
    }


def analyze(
    config: dict[str, Any], context: dict[str, Any], runtime: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    estimator = challenge._estimator(context["controlled"])
    delay_rows: list[dict[str, Any]] = []
    los_by_pair: dict[str, dict[str, tuple[float, float, float]]] = {}
    for pair in config["pairs"]:
        pair_id = pair["candidate_id"]
        record = runtime[pair_id]
        log = Path(record["authentic_los_log"]["path"])
        if sha256(log) != record["authentic_los_log"]["sha256"]:
            raise ValueError(f"LOS provenance mismatch: {pair_id}")
        los_by_pair[pair_id] = parse_gps_sdr_sim_los_table(log.read_text(encoding="utf-8"))
        for condition in CONDITIONS:
            receiver = Path(record["receivers"][condition]["path"])
            delays, _ = geometry.analyze_stream(
                f"{pair_id}:{condition}", receiver, estimator,
                los_by_pair[pair_id], config, 9,
            )
            delay_rows.extend(
                {**row, "pair_id": pair_id, "condition": condition} for row in delays
            )
    stabilized, scored = temporal.score_delays(delay_rows, los_by_pair, window_bins=5)
    prepared = []
    for row in scored:
        observable = float(row["centered_delay_rms_chips"]) >= 0.10
        prepared.append({
            **row,
            "partial_f_only_alarm": bool(row["raw_spoof_alarm"]),
            "observable": observable,
            "raw_spoof_alarm": bool(observable and row["partial_f_p_value"] <= 0.06028418845288192),
            "observable_gated_partial_f_score": (
                float(row["partial_f_score"]) if observable else 0.0
            ),
        })
    scored = temporal.add_persistence(
        prepared, "raw_spoof_alarm", "persistent_spoof_alarm",
    )
    pair_results = []
    for pair in config["pairs"]:
        pair_id = pair["candidate_id"]
        conditions = {}
        for condition in CONDITIONS:
            selected = [
                row for row in scored
                if row["pair_id"] == pair_id and row["condition"] == condition
            ]
            conditions[condition] = condition_metrics(selected, condition, config)
        pair_results.append({
            "pair_id": pair_id,
            "slot": pair["slot"],
            "position": pair["position"],
            "target_offset_enu_m": pair["target_offset_enu_m"],
            "startup_los_prn_count": config["selection"]["selected_startup_los_counts"][pair_id],
            "truth": runtime[pair_id]["truth_audit"],
            "conditions": conditions,
        })
    condition_rows = [item for pair in pair_results for item in pair["conditions"].values()]
    coupled = [pair["conditions"]["carrier-coupled"] for pair in pair_results]
    locked = [pair["conditions"]["doppler-locked"] for pair in pair_results]
    normal_rows = [pair["conditions"]["normal"] for pair in pair_results]
    multipath_rows = [pair["conditions"]["multipath"] for pair in pair_results]
    locked_latencies = [
        row["detection_latency_from_onset_s"] for row in locked
        if row["detection_latency_from_onset_s"] is not None
    ]
    auc_rows = [
        row for row in scored
        if (
            row["condition"] == "doppler-locked" and float(row["bin_start_s"]) >= 12.0
        ) or (
            row["condition"] == "multipath" and int(row["bin_index"]) >= 4
        )
    ]
    auc_labels = np.asarray([row["condition"] == "doppler-locked" for row in auc_rows])
    gated_scores = np.asarray([row["observable_gated_partial_f_score"] for row in auc_rows])
    baseline_scores = np.asarray([row["partial_f_score"] for row in auc_rows])
    aggregates = {
        "pair_count": len(pair_results),
        "truth_invariant_pair_count": sum(pair["truth"]["passed"] for pair in pair_results),
        "condition_count": len(condition_rows),
        "supported_condition_count": sum(row["supported"] for row in condition_rows),
        "carrier_coupled_persistent_detection_count": sum(row["persistent_alarm_any"] for row in coupled),
        "doppler_locked_persistent_detection_count": sum(row["persistent_alarm_any"] for row in locked),
        "normal_persistent_alarm_pair_count": sum(row["persistent_alarm_any"] for row in normal_rows),
        "multipath_persistent_alarm_pair_count": sum(row["persistent_alarm_any"] for row in multipath_rows),
        "total_spoof_pre_attack_persistent_alarm_count": sum(
            row["pre_attack_persistent_alarm_count"] for row in coupled + locked
        ),
        "median_doppler_locked_hold_raw_alarm_rate": float(np.median([row["raw_alarm_rate"] for row in locked])),
        "median_doppler_locked_latency_seconds": float(np.median(locked_latencies)) if locked_latencies else None,
        "doppler_locked_vs_multipath_auc": float(roc_auc_score(auc_labels, gated_scores)),
        "partial_f_only_doppler_locked_vs_multipath_auc": float(roc_auc_score(auc_labels, baseline_scores)),
    }
    gates_config = config["evaluation"]
    gates = {
        "pair_count": aggregates["pair_count"] == gates_config["required_pair_count"],
        "truth_invariants": aggregates["truth_invariant_pair_count"] == gates_config["required_truth_invariant_pair_count"],
        "condition_support": aggregates["condition_count"] == gates_config["required_condition_count"] and aggregates["supported_condition_count"] == gates_config["required_supported_condition_count"],
        "carrier_coupled_detection": aggregates["carrier_coupled_persistent_detection_count"] >= gates_config["minimum_carrier_coupled_persistent_detection_count"],
        "doppler_locked_detection": aggregates["doppler_locked_persistent_detection_count"] >= gates_config["minimum_doppler_locked_persistent_detection_count"],
        "normal_false_alarm": aggregates["normal_persistent_alarm_pair_count"] <= gates_config["maximum_normal_persistent_alarm_pair_count"],
        "multipath_false_alarm": aggregates["multipath_persistent_alarm_pair_count"] <= gates_config["maximum_multipath_persistent_alarm_pair_count"],
        "spoof_pre_attack_false_alarm": aggregates["total_spoof_pre_attack_persistent_alarm_count"] <= gates_config["maximum_total_spoof_pre_attack_persistent_alarm_count"],
        "locked_hold_raw_alarm_rate": aggregates["median_doppler_locked_hold_raw_alarm_rate"] >= gates_config["minimum_median_doppler_locked_hold_raw_alarm_rate"],
        "locked_latency": aggregates["median_doppler_locked_latency_seconds"] is not None and aggregates["median_doppler_locked_latency_seconds"] <= gates_config["maximum_median_doppler_locked_latency_seconds"],
        "locked_vs_multipath_auc": aggregates["doppler_locked_vs_multipath_auc"] >= gates_config["minimum_doppler_locked_vs_multipath_auc"],
    }
    return {
        "pairs": pair_results,
        "aggregates": aggregates,
        "gates": gates,
        "decision": "SUPPORTED" if all(gates.values()) else "NOT_SUPPORTED",
        "delay_rows": delay_rows,
        "stabilized_rows": stabilized,
        "score_rows": scored,
    }


def run(
    config: dict[str, Any], context: dict[str, Any], resume: bool,
    state_path: Path, state: dict[str, Any],
) -> Path:
    runtime = {}
    for index, pair in enumerate(config["pairs"], 1):
        pair_id = pair["candidate_id"]
        set_phase(state_path, state, f"pair_{index}_of_5:{pair_id}")
        print(f"[pair {index}/5] {pair_id}", flush=True)
        runtime[pair_id] = process_pair(config, context, pair, resume)
    set_phase(state_path, state, "analysis_in_memory")
    outcome = analyze(config, context, runtime)
    analysis_root = context["output_root"] / "analysis"
    write_csv(analysis_root / "delay_estimates.csv", outcome.pop("delay_rows"))
    write_csv(analysis_root / "stabilized_delay_estimates.csv", outcome.pop("stabilized_rows"))
    write_csv(analysis_root / "geometry_scores.csv", outcome.pop("score_rows"))
    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    summary = {
        "schema": "gnss-doppler-lab.cgc-temporal-final-static-result",
        "schema_version": 1,
        "role": config["experiment"]["role"],
        "config": {"path": str(CONFIG.resolve()), "sha256": sha256(CONFIG)},
        "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
        "release_state": {"path": str(state_path.resolve()), "sha256": sha256(state_path)},
        "selection": config["selection"],
        "detector": config["analysis"],
        **outcome,
        "artifacts": {
            name: {
                "path": str((analysis_root / filename).resolve()),
                "sha256": sha256(analysis_root / filename),
            }
            for name, filename in {
                "delay_estimates": "delay_estimates.csv",
                "stabilized_delay_estimates": "stabilized_delay_estimates.csv",
                "geometry_scores": "geometry_scores.csv",
            }.items()
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
        print("temporal final-static config verified; no selected 25 MHz RF or score accessed")
        return 0
    resume = bool(args.resume_before_metrics)
    state_path, state = start_release(config, context, resume)
    run(config, context, resume, state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
