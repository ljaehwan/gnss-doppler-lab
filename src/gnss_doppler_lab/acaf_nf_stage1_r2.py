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
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from scipy.stats import spearmanr

from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import code_replica, carrier_wipeoff
from gnss_doppler_lab.acaf_nf_stage1_continuous_tracker import ContinuousTrackerRow

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
