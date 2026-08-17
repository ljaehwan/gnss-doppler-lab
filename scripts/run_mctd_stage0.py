#!/usr/bin/env python3
"""MCTD receiver replay driver and clean-only freeze initializer."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gnss_doppler_lab.mctd import sha256_file
import run_trace_stage0_r2e as r2e

ARTIFACT = ROOT / "artifacts/mctd_stage0_static"
SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mctd-stage0-static")
RECEIVER = r2e.driver.RECEIVER
R2E_ARTIFACT = ROOT / "artifacts/trace_stage0_r2e_attack_support_repair"
LOOPS = {
    "slow": {"dll_bw_hz": 0.5, "pll_bw_hz": 10.0},
    "fast": {"dll_bw_hz": 2.0, "pll_bw_hz": 25.0},
    "identical_left": {"dll_bw_hz": 0.5, "pll_bw_hz": 10.0},
    "identical_right": {"dll_bw_hz": 0.5, "pll_bw_hz": 10.0},
}
SCENARIOS = r2e.driver.PHASE_B_SCENARIOS


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stat(path: Path) -> dict[str, object]:
    value = path.stat()
    return {"path": str(path), "size_bytes": value.st_size, "device": value.st_dev,
            "inode": value.st_ino, "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def handoff_path(name: str) -> Path:
    return Path(r2e.driver.HANDOFF_ROOT) / SCENARIOS[name]["phase_b_handoff"]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def config_text(name: str, loop: str, phase: str) -> str:
    spec = SCENARIOS[name]
    handoff = handoff_path(name)
    mapping = {f"Channel{row['channel']}.satellite": row["prn"] for row in rows(handoff)}
    skip_s = 0.0 if name.endswith("cleanStatic") else 90.0
    duration_s = 45.0 if phase == "phase_a" else None
    fs = int(spec["fs"])
    source_items = str(int(round(duration_s * fs * 2))) if duration_s else "0"
    values = {
        "SignalSource.filename": str(spec["raw"]), "SignalSource.seconds_to_skip": str(skip_s),
        "SignalSource.samples": source_items, "Channels_1C.count": str(len(mapping)),
        "Channels.in_acquisition": str(len(mapping)), "Tracking_1C.dump": "false",
        "Tracking_1C.dump_mat": "false", "Tracking_1C.tap_count": "9",
        "Tracking_1C.tap_spacing_chips": "0.125", "Tracking_1C.extend_correlation_symbols": "1",
        "Tracking_1C.trace_dump": "true", "Tracking_1C.trace_dump_filename": "trace_native_1ms_ch_",
        "Tracking_1C.trace_scenario_id": f"MCTD.{name}.{loop}.{phase}",
        "Tracking_1C.trace_raw_sample_offset": str(int(round(skip_s * fs))),
        "Tracking_1C.trace_handoff_filename": str(handoff.resolve()), "Observables.dump": "false",
        "SignalSource.enable_terminal_drain": "true",
        "Tracking_1C.dll_bw_hz": str(LOOPS[loop]["dll_bw_hz"]),
        "Tracking_1C.pll_bw_hz": str(LOOPS[loop]["pll_bw_hz"]), **mapping,
    }
    return r2e.driver.set_config_values(spec["base_config"].read_text(), values)


def initialize() -> int:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    prior_b0 = Path("/home/ubuntu/projects/gnss-doppler-b0-cs-stage0-static/artifacts/b0_cs_stage0_static/ablation_metrics.csv")
    prior_trace = R2E_ARTIFACT / "final_verdict.json"
    dump_json(ARTIFACT / "prior_evidence.json", {
        "schema": "gnss-doppler-lab.mctd-prior-evidence.v1", "used_as_mctd_results": False,
        "B0": {"status": "AVAILABLE", "artifact_commit": "f3de06040ae5f2f70a267e035c882756e397ba92",
               "source": str(prior_b0), "source_sha256": sha256_file(prior_b0),
               "paper_b0_normalized_pauc_fpr_le_0_05": {"DS1": 0.997032640949555, "DS2": 0.9985611510791367, "DS3": 0.9970501474926253},
               "interpretation": "Signed fixed complex nine-tap morphology carries attack information; the historical multi-PRN binomial aggregation added only a small benefit."},
        "TRACE_R2e": {"status": "AVAILABLE", "artifact_commit": "d6609fb8c708e19c4ebd1381357283efb0f3e7f8",
               "source": str(prior_trace), "source_sha256": sha256_file(prior_trace),
               "texbat_clean_holdout_fpr": 0.32748538011695905, "DS3_pre_onset_fpr": 0.5625,
               "DS7_pre_onset_fpr": 0.10,
               "interpretation": "OS3/OS4 actual-action, no-action, shuffled-action, and fixed9 results were nearly identical; action shift/shuffle did not reduce attack score, so action-next-peak relation was not supported."}
    })
    receiver_manifest = json.loads((R2E_ARTIFACT / "receiver_build_manifest.json").read_text())
    raw_binding = json.loads((R2E_ARTIFACT / "raw_source_binding.json").read_text())
    dump_json(ARTIFACT / "receiver_inventory.json", {
        "schema": "gnss-doppler-lab.mctd-receiver-inventory.v1", "status": "PASS",
        "receiver": receiver_manifest, "required_native_fields": "AVAILABLE_IN_TRC1MS02",
        "loop_bandwidth_configuration": "SUPPORTED_BY_TRACKING_1C.dll_bw_hz/pll_bw_hz",
        "inherited_code_modified": False,
    })
    dump_json(ARTIFACT / "raw_source_binding.json", {**raw_binding, "schema": "gnss-doppler-lab.mctd-raw-source-binding.v1", "inherited_read_only": True})
    loop_freeze = {
        "schema": "gnss-doppler-lab.mctd-loop-freeze.v1", "status": "PREREGISTERED_CLEAN_ONLY",
        "requested": {"slow": LOOPS["slow"], "fast": LOOPS["fast"]},
        "final": {"slow": LOOPS["slow"], "fast": LOOPS["fast"]},
        "identical_control": {"left": LOOPS["identical_left"], "right": LOOPS["identical_right"]},
        "common": {"coherent_integration_ms": 1, "tap_count": 9, "tap_spacing_chips": 0.125,
                   "tracking_order": 3, "warmup_s": 10.0, "minimum_cn0_db_hz": 20.0,
                   "same_discriminators": True, "same_nav_bit_handling": True},
        "fll": {"status": "NOT_APPLICABLE_SEPARATE_CONFIG", "reason": "Receiver preserves the same DLL/PLL tracking implementation; no independent FLL bandwidth property is exposed by this configuration."},
        "support_reason": "Requested values are directly supported receiver properties; stable support remains a Phase-A gate.",
        "attack_data_used": False, "freeze_commit": "TO_BE_FILLED_BY_COMMIT",
    }
    dump_json(ARTIFACT / "loop_configuration_freeze.json", loop_freeze)
    config = {
        "schema": "gnss-doppler-lab.mctd-config.v1", "attack_data_used": False,
        "minimum_common_prns": 4, "minimum_common_epochs": 1000, "block_ms": 100,
        "alarm_consecutive_blocks": 3, "primary_quantile": 0.99,
        "sensitivity_quantiles": [0.995], "empirical_fpr_target": 0.01,
        "prompt_epsilon": 1e-9, "minimum_prompt_magnitude": 1e-6,
        "minimum_cn0_db_hz": 20.0, "guard_s": 5.0, "bootstrap_width_s": 10.0,
        "bootstrap_resamples": 999, "random_seed": 20260817,
        "core_scenarios": {"TEXBAT.DS3": {"onset_s": 118.9, "pull_off_s": 195.0},
                           "TEXBAT.DS7": {"onset_s": 110.0, "pull_off_s": 150.0},
                           "OAKBAT.OS3": {"onset_s": 120.0}, "OAKBAT.OS4": {"onset_s": 120.0}},
    }
    dump_json(ARTIFACT / "config.json", config)
    dump_json(ARTIFACT / "preregistration.json", {
        "schema": "gnss-doppler-lab.mctd-preregistration.v1", "status": "SEALED_BEFORE_ATTACK_EVALUATION",
        "attack_data_accessed_by_mctd": False, "loops": loop_freeze["final"], "feature_sets": ["A0", "A1", "A2", "A3", "A4", "A5", "Full"],
        "model": "robust median center plus Ledoit-Wolf shrinkage covariance",
        "threshold": "dataset-local clean calibration q99", "pooling": "PRN median then non-overlap 100 ms median",
        "alarm": "three true consecutive blocks; reset on gaps", "go_criteria": "TASK.md section 12 unchanged",
        "phase_b_authorized": False, "freeze_commit": "TO_BE_FILLED_BY_COMMIT",
    })
    dump_json(ARTIFACT / "source_commit.json", {
        "schema": "gnss-doppler-lab.mctd-source.v1", "branch": "research/mctd-stage0-static",
        "required_base": "d6609fb8c708e19c4ebd1381357283efb0f3e7f8", "preregistration_commit": "TO_BE_FILLED_BY_COMMIT",
    })
    freeze_dir = ARTIFACT / "frozen_configs"
    for name in SCENARIOS:
        for loop in LOOPS:
            for phase in ("phase_a", "full"):
                path = freeze_dir / phase / f"{name.lower().replace('.', '_')}__{loop}.conf"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(config_text(name, loop, phase))
    print(json.dumps({"status": "INITIALIZED", "artifact": str(ARTIFACT)}, indent=2))
    return 0


def run_receiver(name: str, loop: str, phase: str, repetition: int) -> int:
    if phase == "attack" and not json.loads((ARTIFACT / "preregistration.json").read_text()).get("phase_b_authorized"):
        raise RuntimeError("Phase B NOT_AUTHORIZED")
    config_phase = "phase_a" if phase == "phase_a" else "full"
    slug = name.lower().replace(".", "_")
    out = SSD / "dumps" / phase / slug / loop / f"rep{repetition}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    frozen = ARTIFACT / "frozen_configs" / config_phase / f"{slug}__{loop}.conf"
    expected = config_text(name, loop, config_phase)
    if frozen.read_text() != expected:
        raise ValueError("frozen config mismatch")
    receiver_conf = out / "receiver.conf"
    receiver_conf.write_bytes(frozen.read_bytes())
    raw = Path(SCENARIOS[name]["raw"]); handoff = handoff_path(name)
    before = {"raw": stat(raw), "receiver": stat(RECEIVER)}
    command = [str(RECEIVER), f"--config_file={receiver_conf}", "--keyboard=false"]
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=out, check=False)
    ended = datetime.now(timezone.utc).isoformat()
    dumps = sorted(out.glob("trace_native_1ms_ch_*.bin"))
    manifest = {
        "schema": "gnss-doppler-lab.mctd-replay.v1", "status": "PASS" if result.returncode == 0 and len(dumps) >= 4 else "FAIL",
        "scenario": name, "loop": loop, "loop_settings": LOOPS[loop], "phase": phase, "repetition": repetition,
        "started_at": started, "ended_at": ended, "command": command, "exit_code": result.returncode,
        "raw_iq": {**before["raw"], "sha256": SCENARIOS[name]["sha256"]},
        "raw_stable": before["raw"] == stat(raw), "receiver": {**before["receiver"], "sha256": sha256_file(RECEIVER)},
        "receiver_stable": before["receiver"] == stat(RECEIVER), "config_sha256": sha256_file(receiver_conf),
        "handoff": {"path": str(handoff), "sha256": sha256_file(handoff)},
        "dump_files": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in dumps],
    }
    dump_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("initialize")
    run = sub.add_parser("run-receiver")
    run.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    run.add_argument("--loop", choices=tuple(LOOPS), required=True)
    run.add_argument("--phase", choices=("phase_a", "clean", "attack"), required=True)
    run.add_argument("--repetition", type=int, default=1)
    args = parser.parse_args()
    if args.command == "initialize":
        return initialize()
    return run_receiver(args.scenario, args.loop, args.phase, args.repetition)


if __name__ == "__main__":
    raise SystemExit(main())

