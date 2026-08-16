#!/usr/bin/env python3
"""Materialize preregistered attack handoffs and freeze the R2e run inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gnss_doppler_lab.trace_native_1ms import ACTION_VALUE_FIELDS, read_records, sha256_file
import run_trace_stage0_r2e as r2e

ARTIFACT = r2e.driver.ARTIFACT
SSD = r2e.driver.SSD_ROOT
PARENT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
PREREGISTRATION_COMMIT = "9eb2308675a98cd2342c36066553077a57fc089e"
FIELDS = [
    "channel", "source_channel", "prn", "first_raw_interval_start_sample",
    "source_raw_interval_start_sample", *ACTION_VALUE_FIELDS, "interval_length_samples",
]
SPECS = {
    "TEXBAT.DS7": {"slug": "texbat_ds7", "filename": "texbat_ds7.csv"},
    "OAKBAT.OS4": {"slug": "oakbat_os4", "filename": "oakbat_os4.csv"},
}


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def write(name: str, payload: object) -> None:
    path = ARTIFACT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def make_handoff(name: str, spec: dict, prereg: dict, committed: Path) -> dict:
    repair = prereg["scenario_repairs"][name]
    source = SSD / "support_acquisition" / spec["slug"] / "rep1"
    acquisition = json.loads((source / "manifest.json").read_text())
    if acquisition["status"] != "PASS" or not acquisition["pre_onset_only"]:
        raise ValueError(f"{name}: pre-onset support acquisition did not pass")
    fs = int(r2e.driver.PHASE_B_SCENARIOS[name]["fs"])
    raw_offset = int(round(float(repair["source_skip_s"]) * fs))
    threshold = int(round(float(repair["selection_time_s"]) * fs))
    onset_sample = int(round(float(repair["onset_s"]) * fs))
    rows, sources = [], []
    for path in sorted(source.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path)
        eligible = records[
            (records["raw_interval_start_sample"] >= threshold)
            & (records["raw_interval_start_sample"] < onset_sample)
        ]
        if not len(eligible):
            sources.append({"path": str(path), "sha256": sha256_file(path), "status": "EXCLUDED_NO_ELIGIBLE_PRE_ONSET_ROW"})
            continue
        selected = eligible[0]
        absolute = int(selected["raw_interval_start_sample"])
        output = {
            "channel": len(rows),
            "source_channel": int(selected["channel"]),
            "prn": int(selected["prn"]),
            "first_raw_interval_start_sample": absolute - raw_offset,
            "source_raw_interval_start_sample": absolute,
        }
        for field in ACTION_VALUE_FIELDS:
            output[field] = scalar(selected[f"action_used_{field}"])
        output["interval_length_samples"] = int(selected["action_used_interval_length_samples"])
        rows.append(output)
        sources.append(
            {
                "path": str(path), "sha256": sha256_file(path), "status": "SELECTED",
                "source_channel": int(selected["channel"]), "prn": int(selected["prn"]),
                "selected_absolute_raw_sample": absolute,
            }
        )
    if len(rows) < 4 or [row["channel"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"{name}: fewer than four selected channels or non-contiguous map")
    output_path = committed / spec["filename"]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "handoff_path": str(output_path.relative_to(ROOT)),
        "handoff_sha256": sha256_file(output_path),
        "selection_rule": prereg["support_selection_rule"],
        "selection_absolute_sample": threshold,
        "onset_absolute_sample": onset_sample,
        "raw_offset_samples": raw_offset,
        "channel_count": len(rows),
        "fixed_channel_prn_map": {str(row["channel"]): row["prn"] for row in rows},
        "source_channel_map": {str(row["channel"]): row["source_channel"] for row in rows},
        "source_files": sources,
        "scenario_specific": True,
        "pre_onset_only": True,
        "trace_scores_used": False,
        "attack_outcomes_used": False,
    }


def main() -> int:
    prereg = json.loads((ARTIFACT / "preregistration.json").read_text())
    committed = ARTIFACT / "handoffs"
    if committed.exists():
        raise FileExistsError(f"refusing to overwrite {committed}")
    shutil.copytree(PARENT / "handoffs", committed)
    manifest = json.loads((committed / "manifest.json").read_text())
    manifest["schema"] = "gnss-doppler-lab.trace-r2e-handoff-manifest.v1"
    for name, spec in SPECS.items():
        manifest["scenarios"][name] = make_handoff(name, spec, prereg, committed)
    (committed / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write("handoff_manifest.json", manifest)

    runtime = r2e.driver.HANDOFF_ROOT
    runtime.mkdir(parents=True, exist_ok=True)
    mirror_files = {}
    for source in sorted(committed.glob("*.csv")):
        destination = runtime / source.name
        shutil.copy2(source, destination)
        if sha256_file(source) != sha256_file(destination):
            raise ValueError(f"runtime handoff mirror differs: {source.name}")
        mirror_files[source.name] = {
            "committed_path": str(source), "runtime_path": str(destination),
            "byte_size": source.stat().st_size, "sha256": sha256_file(source),
        }
    shutil.copy2(committed / "manifest.json", runtime / "manifest.json")
    mirror = {
        "schema": "gnss-doppler-lab.trace-r2e-handoff-path-mirror.v1",
        "status": "PASS", "runtime_root": str(runtime), "files": mirror_files,
        "sha256_identity": True,
    }
    write("handoff_path_mirror_manifest.json", mirror)

    frozen = ARTIFACT / "frozen_configs"
    frozen.mkdir(parents=True)
    phase_a = {}
    for name, spec in r2e.driver.SCENARIOS.items():
        path = frozen / f"{spec['slug']}.conf"
        path.write_text(r2e.frozen_config_text(name))
        phase_a[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    phase_b_dir = frozen / "phase_b"
    phase_b_dir.mkdir()
    phase_b = {}
    for name, spec in r2e.driver.PHASE_B_SCENARIOS.items():
        path = phase_b_dir / f"{spec['slug']}.conf"
        path.write_text(r2e.frozen_phase_b_config_text(name))
        phase_b[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}

    for filename in ("receiver_repair.diff", "semantic_reproduction_contract.json"):
        shutil.copy2(PARENT / filename, ARTIFACT / filename)
    parent_build = json.loads((PARENT / "receiver_build_manifest.json").read_text())
    executable_hash = sha256_file(r2e.driver.RECEIVER)
    patch_hash = sha256_file(ARTIFACT / "receiver_repair.diff")
    if executable_hash != parent_build["receiver_executable"]["sha256"] or patch_hash != parent_build["receiver_repair_diff"]["sha256"]:
        raise ValueError("R2c/R2d receiver executable or terminal-drain patch changed")
    write(
        "receiver_build_manifest.json",
        {
            **parent_build,
            "schema": "gnss-doppler-lab.trace-r2e-receiver-build-manifest.v1",
            "status": "PASS_REUSED_BYTE_IDENTICAL_R2C_R2D",
            "terminal_drain_semantics": "unchanged byte-identical R2c/R2d executable and patch",
            "build_step": "No rebuild; verified byte-identical reuse.",
        },
    )
    parent_config = json.loads((PARENT / "config.json").read_text())
    write(
        "config.json",
        {
            "schema": "gnss-doppler-lab.trace-r2e-config.v1",
            "frozen_trace_r2_score_policy": parent_config["frozen_trace_r2_score_policy"],
            "semantic_tolerances": parent_config["semantic_tolerances"],
            "native_dump_schema": parent_config["native_dump_schema"],
            "finite_source_terminal_policy": parent_config["finite_source_terminal_policy"],
            "frozen_receiver_configs": {"phase_a": phase_a, "phase_b": phase_b},
            "only_support_mapping_changes": {
                "TEXBAT.DS7": {"before": "texbat_ds3.csv", "after": "texbat_ds7.csv"},
                "OAKBAT.OS4": {"before": "oakbat_os3.csv", "after": "oakbat_os4.csv"},
            },
            "runtime_handoff_path_mirror": mirror,
        },
    )
    prereg["status"] = "FROZEN_AFTER_PREREGISTERED_PRE_ONSET_HANDOFF_REPAIR"
    prereg["source_hashes"] = {
        "receiver_executable_sha256": executable_hash,
        "receiver_patch_sha256": patch_hash,
        "receiver_configs": {"phase_a": phase_a, "phase_b": phase_b},
        "handoffs": {name: item["handoff_sha256"] for name, item in manifest["scenarios"].items()},
    }
    write("preregistration.json", prereg)
    source = json.loads((ARTIFACT / "source_commit.json").read_text())
    source["preregistration_commit"] = PREREGISTRATION_COMMIT
    write("source_commit.json", source)
    print(json.dumps({"status": "PASS", "handoffs": {name: manifest["scenarios"][name]["fixed_channel_prn_map"] for name in SPECS}, "receiver_executable_sha256": executable_hash}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
