"""ACAF-NF Stage-1 R2 full-normal scientific reevaluation.

Checkpoint 1 is deliberately provenance-heavy: it preserves the R1 scientific
record, summarizes excluded tracker exports, and resolves exporter row semantics
using cleanStatic only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from scipy.stats import spearmanr

from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import code_replica, carrier_wipeoff
from gnss_doppler_lab.acaf_nf_stage1_continuous_tracker import (
    ContinuousTrackerRow,
    build_continuous_tracker_rows,
    save_continuous_tracker_csv,
    validate_cleanstatic_reconstruction,
)

FS_HZ = 25_000_000
SUPPORT = 25_000
DELAYS = np.arange(-1.0, 1.0001, 0.125)
DOPPLERS = np.arange(-250.0, 250.0001, 50.0)
R1_REQUIRED = (
    "README.md", "go_no_go.json", "scenario_metrics.csv", "phase_metrics.csv",
    "baseline_metrics.csv", "control_metrics.csv", "per_window_scores.csv",
    "secondary_component_metrics.csv", "thresholds.json", "normal_model_summary.json",
    "bootstrap_results.json", "execution_validity.json", "verification_report.json",
    "checkpoint_c_verification_report.json", "checksums.json", "test_report.txt",
    "config.json", "cleanstatic_validation.json", "cleanstatic_validation_epochs.csv",
    "cleanstatic_l20_windows.json", "cleanstatic_caf_surfaces.npz",
    "continuous_tracker_manifest.json", "attack_tracker_manifest.json",
    "tracker_cadence_audit.json", "tracker_cadence_by_channel.csv",
    "scenario_timeline.json", "source_binding.json", "execution_manifest.json",
    "execution_manifest_checkpoint_b.json", "execution_manifest_checkpoint_c.json",
    "plots/cleanstatic_prompt_reproduction.svg", "plots/stage1_r1_scores.png",
    "receiver_exporter_build/build_provenance.json",
)
SCENARIOS = ("cleanStatic", "ds3", "ds4", "ds7", "ds8")
FRESH_REPLAY_RELATIVE = Path("receiver_replays/cleanStatic")
INVALID_CHECKPOINT2_SHA = "b5d53fa"


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_r1(r1: Path, output: Path) -> list[str]:
    destination = output / "r1_preserved"
    copied: list[str] = []
    for relative in R1_REQUIRED:
        source = r1 / relative
        if not source.is_file():
            raise FileNotFoundError(f"required R1 artifact missing: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative)
    for scenario in SCENARIOS:
        for name in ("manifest.json", "receiver.conf"):
            source = r1 / "receiver_replays" / scenario / name
            if source.is_file():
                target = destination / "receiver_replays" / scenario / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(target.relative_to(destination).as_posix())
    return copied


def _tracker_summary(r1: Path, output: Path, checksums: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    inventory: dict[str, Any] = {}
    phase_manifest = json.loads((r1 / "attack_tracker_manifest.json").read_text(encoding="utf-8"))
    clean_manifest = json.loads((r1 / "continuous_tracker_manifest.json").read_text(encoding="utf-8"))
    for scenario in SCENARIOS:
        path = r1 / f"continuous_tracker_{scenario}.csv"
        seconds: dict[int, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "pairs": set(), "prns": set(), "channels": set()})
        total = 0
        pairs: set[tuple[int, int]] = set()
        prns: set[int] = set()
        channels: set[int] = set()
        first = None
        last = None
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total += 1
                start = int(row["raw_start_sample"])
                second = start // FS_HZ
                channel = int(row["channel"])
                prn = int(row["prn"])
                pair = (channel, prn)
                bucket = seconds[second]
                bucket["rows"] += 1
                bucket["pairs"].add(pair)
                bucket["prns"].add(prn)
                bucket["channels"].add(channel)
                pairs.add(pair); prns.add(prn); channels.add(channel)
                first = start if first is None else min(first, start)
                last = start + SUPPORT if last is None else max(last, start + SUPPORT)
        for second, bucket in sorted(seconds.items()):
            rows_out.append({
                "scenario": scenario, "receiver_second": second, "rows": bucket["rows"],
                "channel_prn_pairs": len(bucket["pairs"]), "prns": len(bucket["prns"]),
                "channels": len(bucket["channels"]),
            })
        manifest_validation = clean_manifest["validation"] if scenario == "cleanStatic" else phase_manifest["scenarios"][scenario]["tracker_validation"]
        phase_coverage = {"all": {"rows": total}} if scenario == "cleanStatic" else phase_manifest["scenarios"][scenario]["phase_coverage"]
        expected = checksums["files"][path.name]
        inventory[scenario] = {
            "path": str(path), "excluded_from_git": True,
            "sha256": expected["sha256"], "size_bytes": expected["size_bytes"],
            "rows": total, "rows_match_manifest": total == int(manifest_validation["rows"]),
            "receiver_seconds": len(seconds), "first_support_start_sample": first,
            "last_support_end_sample": last, "prns": sorted(prns), "channels": sorted(channels),
            "channel_prn_pairs": [list(pair) for pair in sorted(pairs)],
            "phase_coverage": phase_coverage,
        }
    fields = ("scenario", "receiver_second", "rows", "channel_prn_pairs", "prns", "channels")
    with (output / "r1_tracker_receiver_second_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows_out)
    return rows_out, inventory


def _load_clean_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _alignment_sample(rows: list[dict[str, str]], target: int = 192) -> list[dict[str, str]]:
    by_pair: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_pair[(int(row["channel"]), int(row["prn"]))].append(row)
    chosen: list[dict[str, str]] = []
    pairs = sorted(by_pair)
    per_pair = max(12, target // max(len(pairs), 1))
    for pair in pairs:
        values = by_pair[pair]
        indices = np.linspace(1, len(values) - 2, min(per_pair, max(0, len(values) - 2)), dtype=int)
        chosen.extend(values[int(i)] for i in sorted(set(indices.tolist())))
    return sorted(chosen, key=lambda row: (int(row["raw_start_sample"]), int(row["channel"])))[:target]


def _vector(handle: h5py.File, name: str) -> np.ndarray:
    return np.asarray(handle[name]).reshape(-1)


def _alignment_audit(r1: Path, output: Path) -> dict[str, Any]:
    rows = _alignment_sample(_load_clean_rows(r1 / "continuous_tracker_cleanStatic.csv"))
    raw_path = Path(json.loads((r1 / "source_binding.json").read_text(encoding="utf-8"))["scenarios"]["cleanStatic"]["raw_path"])
    raw = np.memmap(raw_path, dtype="<i2", mode="r")
    mats: dict[str, dict[str, np.ndarray]] = {}
    evidence: dict[int, list[dict[str, Any]]] = {shift: [] for shift in (-1, 0, 1)}
    for row in rows:
        mat_path = row["source_mat"]
        if mat_path not in mats:
            with h5py.File(mat_path, "r") as handle:
                mats[mat_path] = {name: _vector(handle, name) for name in (
                    "PRN_start_sample_count", "PRN", "carrier_doppler_hz", "code_freq_chips",
                    "aux1", "Prompt_I", "Prompt_Q")}
        values = mats[mat_path]
        current = int(row["mat_row"])
        prompt = complex(float(values["Prompt_I"][current]), float(values["Prompt_Q"][current]))
        for shift in (-1, 0, 1):
            state = current - 1 + shift
            if state < 0 or state >= len(values["PRN"]):
                continue
            start = int(values["PRN_start_sample_count"][state])
            if start < 0 or 2 * (start + SUPPORT) > raw.size or int(values["PRN"][state]) != int(row["prn"]):
                continue
            packed = np.asarray(raw[2 * start:2 * (start + SUPPORT)]).reshape(-1, 2)
            iq = packed[:, 0].astype(np.float64) + 1j * packed[:, 1].astype(np.float64)
            code = code_replica(int(row["prn"]), SUPPORT, FS_HZ, float(values["code_freq_chips"][state]),
                                float(values["aux1"][state]), -1, 0.0, replica_direction=1)[0]
            wipe = carrier_wipeoff(SUPPORT, FS_HZ, float(values["carrier_doppler_hz"][state]), 0.0, -1)[0]
            center = (wipe * iq) @ code
            evidence[shift].append({
                "shift": shift, "channel": int(row["channel"]), "prn": int(row["prn"]),
                "tracker_row": current, "state_row": state, "support_start_sample": start,
                "mat_prompt_magnitude": abs(prompt), "reconstructed_center_magnitude": abs(center),
                "relative_error": abs(abs(center) / max(abs(prompt), 1e-15) - 1.0),
            })
    metrics: dict[str, Any] = {}
    for shift, values in evidence.items():
        center = np.asarray([x["reconstructed_center_magnitude"] for x in values])
        prompt = np.asarray([x["mat_prompt_magnitude"] for x in values])
        rel = np.asarray([x["relative_error"] for x in values])
        by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, item in enumerate(values): by_pair[(item["channel"], item["prn"])].append(i)
        pair_rho = [float(spearmanr(center[idx], prompt[idx]).statistic) for idx in by_pair.values() if len(idx) >= 3]
        metrics[str(shift)] = {
            "n": len(values), "pooled_spearman": float(spearmanr(center, prompt).statistic),
            "median_pair_spearman": float(np.nanmedian(pair_rho)),
            "median_relative_error": float(np.median(rel)), "p99_relative_error": float(np.quantile(rel, .99)),
            "max_relative_error": float(np.max(rel)), "clean_only": True,
        }
    selected = max((-1, 0, 1), key=lambda shift: (metrics[str(shift)]["pooled_spearman"], -metrics[str(shift)]["p99_relative_error"]))
    csv_fields = tuple(evidence[0][0])
    with (output / "clean_alignment_shift_evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields); writer.writeheader()
        for shift in (-1, 0, 1): writer.writerows(evidence[shift])
    source = Path("/home/ubuntu/build-gnss-sdr-complex9/src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc")
    source_text = source.read_text(encoding="utf-8")
    return {
        "schema": "acaf_nf_stage1_r2_alignment_audit.v1", "status": "PASS" if selected == 0 else "FAIL",
        "selection_data": "cleanStatic_only", "attacks_examined_for_alignment": False,
        "candidate_row_shifts": [-1, 0, 1], "selected_row_shift": selected,
        "selected_mapping": {
            "prompt": "current MAT row k", "nco_aux": "previous MAT row k-1",
            "support": "[stamp(k-1), stamp(k-1)+25000)", "tracker_csv_state_mat_row": "k-1",
        },
        "metrics": metrics,
        "exporter_semantics": {
            "log_data_call_order": "do_correlation_step; save_correlation_results; run_dll_pll; update_tracking_vars; check_carrier_phase_coherent_initialization; log_data",
            "prompt_at_log": "d_Prompt from the correlation just completed (current row k)",
            "nco_aux_at_log": "loop/remnant state updated for the next call; row k state applies to Prompt row k+1",
            "stamp_at_log": "nitems_read(0)+d_current_prn_length_samples; boundary for the next call",
            "carrier_code_aux_alignment": "Prompt(k) uses NCO/code/aux and raw support identified by row k-1",
            "nav_bit_gate_effect": "R1 patch moves log_data outside d_current_data_symbol==0 only in narrow tracking; dump cadence becomes each valid 1 ms DLL/PLL cycle while telemetry/nav-bit gating is unchanged",
            "source_git_head": "1ddd4562723040fd66cb334b578a5b69455625f4",
            "source_sha256_current": _sha256(source),
            "source_contains_unpatched_gate": "if (d_current_data_symbol == 0)" in source_text[source_text.find("case 4:"):],
            "r1_exporter_patch_sha256": _sha256(Path("patches/gnss-sdr-valid-1ms-tracking-dump.patch")),
        },
        "r14_common_mapping": {
            "identity": ["channel", "prn", "tracker_row", "support_start_sample"],
            "exact_r1_mapping": "current Prompt k + previous NCO/aux k-1 + support stamp k-1",
            "r1_common_epochs": json.loads((r1 / "cleanstatic_validation.json").read_text(encoding="utf-8"))["r14_common_epochs"],
        },
    }


def _refresh_checksums(output: Path) -> None:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"checksums.json", "verification_report.json"}:
            files[path.relative_to(output).as_posix()] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    _dump(output / "checksums.json", {"schema": "acaf_nf_stage1_r2_checksums.v1", "files": files})


def checkpoint1(output: Path, r1: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    copied = _copy_r1(r1, output)
    r1_checksums = json.loads((r1 / "checksums.json").read_text(encoding="utf-8"))
    summaries, trackers = _tracker_summary(r1, output, r1_checksums)
    audit = _alignment_audit(r1, output)
    _dump(output / "continuous_tracker_alignment_audit.json", audit)
    source_binding = json.loads((r1 / "source_binding.json").read_text(encoding="utf-8"))
    manifests = {}
    for scenario in SCENARIOS:
        binding = source_binding["scenarios"][scenario]
        manifests[scenario] = {
            "raw_path": binding["raw_path"], "raw_sha256": binding["raw_sha256"],
            "raw_size_bytes": binding["raw_size_bytes"],
            "receiver_manifest_path": binding["receiver_manifest_path"],
            "receiver_manifest_sha256": binding["receiver_manifest_sha256"],
            "receiver_config_path": binding["receiver_config_path"],
            "receiver_config_sha256": binding["receiver_config_sha256"],
            "mat_inventory": binding["tracker_mat_inventory"],
        }
    _dump(output / "r1_artifact_inventory.json", {
        "schema": "acaf_nf_stage1_r2_r1_inventory.v1", "authoritative_git_sha": "7693d5ec13934b54f9ab5fa68d04f60d437dd060",
        "r1_artifact": str(r1), "preserved_under": "r1_preserved", "preserved_files": copied,
        "excluded_large_tracker_exports": trackers,
        "excluded_receiver_binary": r1_checksums["files"]["receiver_exporter_build/gnss-sdr-continuous-1ms"],
        "receiver_and_raw_bindings": manifests,
        "receiver_second_summary_file": "r1_tracker_receiver_second_summary.csv",
        "receiver_second_summary_rows": len(summaries),
    })
    _dump(output / "config.json", {
        "schema": "acaf_nf_stage1_r2_config.v1", "checkpoint": 1,
        "authoritative_base_sha": "7693d5ec13934b54f9ab5fa68d04f60d437dd060",
        "r1_artifact": str(r1), "output": str(output), "alignment_row_shift": audit["selected_row_shift"],
        "alignment_selection_source": "cleanStatic_only", "attacks_used_for_alignment": False,
    })
    (output / "README.md").write_text(
        "# ACAF-NF Stage-1 R2 full-normal\n\n"
        "Checkpoint 1 preserves the R1 scientific record, inventories excluded large exports, and freezes "
        f"cleanStatic-only exporter alignment shift `{audit['selected_row_shift']}`. No attack data selected alignment.\n",
        encoding="utf-8",
    )
    _refresh_checksums(output)
    return output


def _csv_tracker_rows(path: Path) -> list[ContinuousTrackerRow]:
    result: list[ContinuousTrackerRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result.append(ContinuousTrackerRow(
                scenario=row["scenario"], channel=int(row["channel"]), prn=int(row["prn"]),
                tracker_row=int(row["tracker_row"]), mat_row=int(row["mat_row"]),
                state_mat_row=int(row["state_mat_row"]), raw_start_sample=int(row["raw_start_sample"]),
                raw_end_sample=int(row["raw_end_sample"]), sample_count=int(row["sample_count"]),
                code_freq_chips=float(row["code_freq_chips"]), carrier_doppler_hz=float(row["carrier_doppler_hz"]),
                aux1=float(row["aux1"]), prompt_i=float(row["prompt_i"]), prompt_q=float(row["prompt_q"]),
                cn0_db_hz=float(row["cn0_db_hz"]), carrier_lock_test=float(row["carrier_lock_test"]),
                quality_min_cn0_db_hz=float(row["quality_min_cn0_db_hz"]),
                quality_min_carrier_lock=float(row["quality_min_carrier_lock"]), source_mat=row["source_mat"],
                source_dat_row_match=row["source_dat_row_match"] == "True",
                source_dat_rows=int(row["source_dat_rows"]), source_dat_record_bytes=int(row["source_dat_record_bytes"]),
                source_dat_sample_stamp_match=row["source_dat_sample_stamp_match"] == "True",
            ))
    return result


def _first_rows_semantics(path: Path, limit: int = 32) -> dict[str, Any]:
    checked = 0
    errors: list[str] = []
    mats: dict[str, dict[str, np.ndarray]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mat_path = row["source_mat"]
            if mat_path not in mats:
                with h5py.File(mat_path, "r") as source:
                    mats[mat_path] = {name: _vector(source, name) for name in (
                        "PRN_start_sample_count", "PRN", "carrier_doppler_hz", "code_freq_chips", "aux1",
                        "Prompt_I", "Prompt_Q")}
            values = mats[mat_path]; current = int(row["mat_row"]); state = int(row["state_mat_row"])
            if state != current - 1: errors.append("state_not_previous")
            if int(row["raw_start_sample"]) != int(values["PRN_start_sample_count"][state]): errors.append("support_stamp")
            if int(row["raw_end_sample"]) - int(row["raw_start_sample"]) != SUPPORT: errors.append("support_length")
            for name, index, csv_name in (
                ("carrier_doppler_hz", state, "carrier_doppler_hz"), ("code_freq_chips", state, "code_freq_chips"),
                ("aux1", state, "aux1"), ("Prompt_I", current, "prompt_i"), ("Prompt_Q", current, "prompt_q"),
            ):
                if float(row[csv_name]) != float(values[name][index]): errors.append(f"value:{csv_name}")
            checked += 1
            if checked >= limit: break
    return {"status": "PASS" if not errors and checked == limit else "FAIL", "checked_rows": checked,
            "errors": sorted(set(errors)), "mapping": "current Prompt k; previous NCO/aux k-1; support stamp k-1"}


def _fresh_replay_inventory(replay: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = replay / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_dir = replay / "raw"
    files: dict[str, Any] = {}
    all_hashes_match = True
    all_mat_dat_rows_match = True
    total_exporter_rows = 0
    mat_rows: dict[str, int] = {}
    for mat_name, expected in sorted(manifest["tracking"]["mat_inventory"].items()):
        mat_path = raw_dir / mat_name
        actual = _sha256(mat_path)
        with h5py.File(mat_path, "r") as handle:
            lengths = {name: int(np.asarray(handle[name]).size) for name in (
                "PRN_start_sample_count", "PRN", "carrier_doppler_hz", "code_freq_chips",
                "aux1", "Prompt_I", "Prompt_Q", "CN0_SNV_dB_Hz", "carrier_lock_test",
            )}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"fresh replay MAT dataset length mismatch: {mat_path}")
        rows = next(iter(lengths.values()))
        mat_rows[mat_name] = rows
        total_exporter_rows += rows
        matched = actual == expected
        all_hashes_match = all_hashes_match and matched
        files[f"raw/{mat_name}"] = {
            "sha256": actual, "manifest_sha256": expected, "sha256_match": matched,
            "size_bytes": mat_path.stat().st_size, "exporter_rows": rows,
        }
    for dat_name, expected in sorted(manifest["tracking"]["dat_inventory"].items()):
        dat_path = raw_dir / dat_name
        actual = _sha256(dat_path)
        rows = dat_path.stat().st_size // 148
        mat_name = f"{dat_path.stem}.mat"
        row_match = dat_path.stat().st_size % 148 == 0 and rows == mat_rows[mat_name]
        matched = actual == expected
        all_hashes_match = all_hashes_match and matched
        all_mat_dat_rows_match = all_mat_dat_rows_match and row_match
        files[f"raw/{dat_name}"] = {
            "sha256": actual, "manifest_sha256": expected, "sha256_match": matched,
            "size_bytes": dat_path.stat().st_size, "exporter_rows": rows,
            "mat_rows": mat_rows[mat_name], "mat_dat_rows_match": row_match,
        }
    source_path = Path(manifest["source"]["path"])
    report = {
        "schema": "acaf_nf_stage1_r2_fresh_replay_binding.v1",
        "status": "PASS" if all_hashes_match and all_mat_dat_rows_match else "FAIL",
        "scenario": "cleanStatic", "replay_path": str(replay),
        "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        "source_path": str(source_path), "source_sha256": manifest["source"]["sha256"],
        "source_size_bytes": manifest["source"]["size_bytes"],
        "source_sample_rate_hz": manifest["source"]["sample_rate_hz"],
        "source_sample_format": manifest["source"]["sample_format"],
        "receiver_executable": manifest["receiver"]["executable"],
        "receiver_executable_sha256": manifest["receiver"]["executable_sha256"],
        "receiver_config": manifest["receiver"]["config"],
        "receiver_config_sha256": manifest["receiver"]["config_sha256"],
        "exporter_patch_sha256": manifest["receiver"]["exporter_patch_sha256"],
        "dump_contract": manifest["tracking"]["dump_contract"],
        "extend_correlation_symbols": manifest["tracking"]["extend_correlation_symbols"],
        "exporter_rows": total_exporter_rows, "all_file_hashes_match_manifest": all_hashes_match,
        "all_mat_dat_rows_match": all_mat_dat_rows_match, "files": files,
    }
    return manifest, report


def _fresh_alignment_audit(
    rows: list[ContinuousTrackerRow], raw_path: Path, output: Path, binding: dict[str, Any], target: int = 192,
) -> dict[str, Any]:
    chosen = _alignment_sample([
        {
            "channel": str(row.channel), "prn": str(row.prn),
            "raw_start_sample": str(row.raw_start_sample), "_row": row,
        }
        for row in rows
    ], target=target)
    selected = [entry["_row"] for entry in chosen]
    raw = np.memmap(raw_path, dtype="<i2", mode="r")
    mats: dict[str, dict[str, np.ndarray]] = {}
    evidence: list[dict[str, Any]] = []
    for row in selected:
        if row.source_mat not in mats:
            with h5py.File(row.source_mat, "r") as handle:
                mats[row.source_mat] = {name: _vector(handle, name) for name in (
                    "PRN_start_sample_count", "PRN", "carrier_doppler_hz", "code_freq_chips",
                    "aux1", "Prompt_I", "Prompt_Q",
                )}
        values = mats[row.source_mat]
        current = row.mat_row
        for prompt_semantics, prompt_offset in (("current", 0), ("next", 1)):
            prompt_row = current + prompt_offset
            if prompt_row >= len(values["PRN"]) or int(values["PRN"][prompt_row]) != row.prn:
                continue
            prompt = abs(complex(float(values["Prompt_I"][prompt_row]), float(values["Prompt_Q"][prompt_row])))
            for shift in (-1, 0, 1):
                state = current - 1 + shift
                if state < 0 or state >= len(values["PRN"]) or int(values["PRN"][state]) != row.prn:
                    continue
                start = int(values["PRN_start_sample_count"][state])
                if start < 0 or 2 * (start + SUPPORT) > raw.size:
                    continue
                packed = np.asarray(raw[2 * start:2 * (start + SUPPORT)]).reshape(-1, 2)
                iq = packed[:, 0].astype(np.float64) + 1j * packed[:, 1].astype(np.float64)
                code = code_replica(row.prn, SUPPORT, FS_HZ, float(values["code_freq_chips"][state]),
                                    float(values["aux1"][state]), -1, 0.0, replica_direction=1)[0]
                wipe = carrier_wipeoff(SUPPORT, FS_HZ, float(values["carrier_doppler_hz"][state]), 0.0, -1)[0]
                center = abs((wipe * iq) @ code)
                evidence.append({
                    "prompt_semantics": prompt_semantics, "prompt_offset": prompt_offset,
                    "state_shift": shift, "channel": row.channel, "prn": row.prn,
                    "prompt_mat_row": prompt_row, "state_mat_row": state,
                    "support_start_sample": start, "mat_prompt_magnitude": prompt,
                    "reconstructed_center_magnitude": center,
                    "relative_error": abs(center / max(prompt, 1e-15) - 1.0),
                })
    metrics: dict[str, Any] = {}
    for prompt_semantics in ("current", "next"):
        for shift in (-1, 0, 1):
            values = [item for item in evidence if item["prompt_semantics"] == prompt_semantics and item["state_shift"] == shift]
            center = np.asarray([item["reconstructed_center_magnitude"] for item in values])
            prompt = np.asarray([item["mat_prompt_magnitude"] for item in values])
            relative = np.asarray([item["relative_error"] for item in values])
            by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
            for index, item in enumerate(values):
                by_pair[(item["channel"], item["prn"])].append(index)
            pair_rho = [float(spearmanr(center[index], prompt[index]).statistic) for index in by_pair.values() if len(index) >= 3]
            key = f"prompt_{prompt_semantics}_state_shift_{shift:+d}"
            metrics[key] = {
                "n": len(values), "pooled_spearman": float(spearmanr(center, prompt).statistic),
                "median_pair_spearman": float(np.nanmedian(pair_rho)),
                "median_relative_error": float(np.median(relative)),
                "p99_relative_error": float(np.quantile(relative, 0.99)),
                "max_relative_error": float(np.max(relative)), "clean_only": True,
            }
    selected_key = max(metrics, key=lambda key: (metrics[key]["pooled_spearman"], -metrics[key]["p99_relative_error"]))
    fields = list(evidence[0])
    with (output / "fresh_clean_alignment_evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(evidence)
    canonical = "prompt_current_state_shift_+0"
    canonical_metrics = metrics[canonical]
    gates = {
        "fresh_manifest_authenticated": binding["status"] == "PASS",
        "canonical_mapping_selected": selected_key == canonical,
        "pooled_spearman_ge_0_999": canonical_metrics["pooled_spearman"] >= 0.999,
        "prompt_p99_relative_error_le_0_01": canonical_metrics["p99_relative_error"] <= 0.01,
    }
    return {
        "schema": "acaf_nf_stage1_r2_fresh_alignment_audit.v2",
        "status": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
        "selection_data": "fresh_cleanStatic_replay_only", "attacks_examined_for_alignment": False,
        "candidate_state_shifts": [-1, 0, 1], "candidate_prompt_semantics": ["current", "next"],
        "selected_candidate": selected_key, "selected_row_shift": 0,
        "selected_mapping": {
            "prompt": "current MAT row k", "nco_aux": "previous MAT row k-1",
            "support": "[stamp(k-1), stamp(k-1)+25000)", "tracker_csv_state_mat_row": "k-1",
            "equivalent_reindexing": "next Prompt k+1 with current state/stamp k",
        },
        "metrics": metrics, "probe_rows": len(selected),
        "fresh_replay_manifest_path": binding["manifest_path"],
        "fresh_replay_manifest_sha256": binding["manifest_sha256"],
        "supersedes": {
            "commit": INVALID_CHECKPOINT2_SHA,
            "reason": "checkpoint2 consumed the authenticated original tracker CSV rather than this fresh exporter replay",
        },
    }


def checkpoint2(output: Path, r1: Path) -> Path:
    replay = r1 / FRESH_REPLAY_RELATIVE
    receiver_manifest, binding = _fresh_replay_inventory(replay)
    _dump(output / "fresh_replay_binding.json", binding)
    raw_path = Path(receiver_manifest["source"]["path"])
    rows, tracker_report = build_continuous_tracker_rows(
        "cleanStatic", replay / "raw", raw_sample_count=raw_path.stat().st_size // 4,
        mat_inventory=receiver_manifest["tracking"]["mat_inventory"],
    )
    tracker_csv = output / "continuous_tracker_cleanStatic.csv"
    save_continuous_tracker_csv(rows, tracker_csv)
    alignment = _fresh_alignment_audit(rows, raw_path, output, binding)
    _dump(output / "continuous_tracker_alignment_audit.json", alignment)
    temporary = output / "checkpoint2_reconstruction"
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    report = validate_cleanstatic_reconstruction(
        rows, raw_path, temporary, Path("artifacts/acaf_nf_stage0_static_r14_doppler_validation"),
        r14_match_mode="raw_support",
    )
    mapping = {
        "cleanstatic_validation.json": "full_cleanstatic_validation.json",
        "cleanstatic_validation_epochs.csv": "full_cleanstatic_validation_epochs.csv",
        "cleanstatic_caf_surfaces.npz": "full_cleanstatic_caf_surfaces.npz",
        "cleanstatic_l20_windows.json": "full_cleanstatic_l20_windows.json",
    }
    for old, new in mapping.items(): shutil.move(temporary / old, output / new)
    source_plot = temporary / "plots" / "cleanstatic_prompt_reproduction.svg"
    plot_target = output / "plots" / "full_cleanstatic_prompt_reproduction.svg"
    plot_target.parent.mkdir(exist_ok=True); shutil.move(source_plot, plot_target)
    shutil.rmtree(temporary)
    report["alignment"] = alignment["selected_mapping"]
    report["alignment_row_shift"] = 0
    report["alignment_selected_from"] = "fresh_cleanStatic_replay_only"
    report["attack_alignment_selection_rows"] = 0
    report["fresh_tracker_cadence"] = tracker_report
    report["fresh_replay_binding_status"] = binding["status"]
    report["scenario_semantics_checks"] = {
        "cleanStatic": {"status": alignment["status"], "mapping": alignment["selected_mapping"]},
        "attacks": {"status": "NOT_EVALUATED", "reason": "foundation gate evaluated before attack replay"},
    }
    report["producer_status"] = "PASS" if (
        report["status"] == "CONTINUOUS_TRACKER_VALID" and tracker_report["status"] == "CONTINUOUS_TRACKER_VALID"
        and binding["status"] == "PASS" and alignment["status"] == "PASS"
    ) else "FAIL"
    if report["producer_status"] != "PASS": report["status"] = "CONTINUOUS_TRACKER_INVALID"
    _dump(output / "full_cleanstatic_validation.json", report)
    tracker_manifest = {
        "schema": "acaf_nf_stage1_r2_fresh_continuous_tracker.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "scenario": "cleanStatic",
        "status": report["status"], "rows": len(rows), "csv_path": str(tracker_csv),
        "csv_sha256": _sha256(tracker_csv), "csv_size_bytes": tracker_csv.stat().st_size,
        "fresh_replay_binding": "fresh_replay_binding.json",
        "fresh_replay_manifest_path": binding["manifest_path"],
        "fresh_replay_manifest_sha256": binding["manifest_sha256"],
        "tracker_path": str(replay / "raw"), "exporter_rows": binding["exporter_rows"],
        "accepted_rows": len(rows), "alignment": alignment["selected_mapping"],
        "validation": report, "supersedes_invalid_checkpoint2_commit": INVALID_CHECKPOINT2_SHA,
    }
    _dump(output / "fresh_continuous_tracker_manifest.json", tracker_manifest)
    foundation = {
        "schema": "acaf_nf_stage1_r2_foundation.v1",
        "status": "FOUNDATION_PASS" if report["producer_status"] == "PASS" else "FOUNDATION_INVALID",
        "continuous_tracker_status": report["status"],
        "failed_clean_gates": sorted(key for key, value in report["gates"].items() if not value),
        "checkpoint3_physics_authorized": report["producer_status"] == "PASS",
        "attack_iq_scoring_performed": False,
        "reason": None if report["producer_status"] == "PASS" else "fresh replay failed one or more independent clean foundation gates",
        "supersedes_invalid_checkpoint2_commit": INVALID_CHECKPOINT2_SHA,
    }
    _dump(output / "foundation_status.json", foundation)
    _dump(output / "checkpoint2_correction.json", {
        "schema": "acaf_nf_stage1_r2_checkpoint2_correction.v1", "supersedes_commit": INVALID_CHECKPOINT2_SHA,
        "invalid_source": str(r1 / "continuous_tracker_cleanStatic.csv"),
        "invalid_source_disposition": "retained only in git history/R1 preservation; not relabeled or used by corrected checkpoint2",
        "correct_source": str(replay / "raw"), "correct_manifest": binding["manifest_path"],
        "correct_manifest_sha256": binding["manifest_sha256"], "attack_data_used_for_selection": False,
        "result": foundation["status"],
    })
    config = json.loads((output / "config.json").read_text(encoding="utf-8")); config["checkpoint"] = 2
    config["checkpoint2_correction_supersedes"] = INVALID_CHECKPOINT2_SHA
    config["fresh_replay_manifest"] = binding["manifest_path"]
    config["fresh_replay_manifest_sha256"] = binding["manifest_sha256"]
    config["frozen_alignment"] = alignment["selected_mapping"]; config["full_cleanstatic_validation_epochs"] = report["selected_epochs"]
    _dump(output / "config.json", config)
    (output / "README.md").write_text(
        "# ACAF-NF Stage-1 R2 full-normal\n\n"
        f"Corrective Checkpoint 2 explicitly supersedes `{INVALID_CHECKPOINT2_SHA}`: that commit incorrectly read the "
        "authenticated original R1 tracker CSV instead of the fresh exporter replay. The original CSV is not relabeled "
        "or used here. Fresh cleanStatic MAT/DAT files are authenticated against their replay manifest and reconstructed "
        f"with clean-only alignment. Corrected status: `{report['status']}` / `{foundation['status']}`. "
        "Attack IQ was not scored and Checkpoint 3 physics is not authorized when foundation is invalid.\n",
        encoding="utf-8",
    )
    _refresh_checksums(output)
    return output
