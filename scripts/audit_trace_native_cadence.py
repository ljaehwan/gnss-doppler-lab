#!/usr/bin/env python3
"""Audit TRACE-R1 cadence and validate the receiver action mapping.

This script deliberately performs no attack scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_action_warp import prompt_normalize, warp_complex_taps
from gnss_doppler_lab.trace_native_cadence import (
    CADENCE_1MS,
    CADENCE_20MS,
    ScenarioSpec,
    block_support,
    cadence_counts,
    gap_distribution,
    load_consecutive_pairs,
    valid_native_mask,
)

ARTIFACT = ROOT / "artifacts/trace_stage0_r1_native_cadence"
RECEIVER_SOURCE = Path("/home/ubuntu/build-gnss-sdr-complex9")
TRACKING_SOURCE = RECEIVER_SOURCE / "src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc"

SPECS = (
    ScenarioSpec("TEXBAT", "cleanStatic", Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9"), 25_000_000),
    ScenarioSpec("TEXBAT", "DS1", Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds1-complex9"), 25_000_000, 125.0),
    ScenarioSpec("TEXBAT", "DS3", Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds3-complex9"), 25_000_000, 118.9),
    ScenarioSpec("TEXBAT", "DS7", Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/receiver/ds7-complex9"), 25_000_000, 110.0),
    ScenarioSpec("OAKBAT", "cleanStatic", Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/cleanstatic/receiver/cleanstatic-complex9"), 5_000_000),
    ScenarioSpec("OAKBAT", "OS1", Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/os1/receiver/os1-complex9"), 5_000_000, 120.0),
    ScenarioSpec("OAKBAT", "OS3", Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/os3/receiver/os3-complex9"), 5_000_000, 120.0),
    ScenarioSpec("OAKBAT", "OS4", Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/os4/receiver/os4-complex9"), 5_000_000, 120.0),
)

TAP_FIELDS = [f"{part}_{tap}" for tap in ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4") for part in ("I", "Q")]
CONTRACT_FIELDS = TAP_FIELDS + [
    "Prompt_I", "Prompt_Q", "code_error_chips", "code_error_filt_chips",
    "carr_error_hz", "carr_error_filt_hz", "carrier_doppler_hz",
    "code_freq_chips", "aux1", "PRN_start_sample_count", "PRN",
    "CN0_SNV_dB_Hz", "carrier_lock_test",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def manifest_source(manifest: dict) -> tuple[str | None, str | None]:
    source = manifest.get("source", {})
    auth = manifest.get("authenticated_inputs", {})
    iq = source.get("iq") or auth.get("iq_before_receiver", {}).get("path")
    iq_sha = source.get("iq_sha256") or auth.get("iq_before_receiver", {}).get("sha256")
    return iq, iq_sha


def clean_reconstruction(spec: ScenarioSpec, limit: int = 50_000) -> dict[str, object]:
    """Clean-only diagnostic; it never selects the physical mapping."""
    error_no_change: list[float] = []
    error_code_1ms: list[float] = []
    error_code_actual_dt: list[float] = []
    count = 0
    for path in sorted((spec.receiver_root / "raw").glob("epl_tracking_ch_*.mat")):
        with h5py.File(path, "r") as handle:
            samples = np.asarray(handle["PRN_start_sample_count"]).reshape(-1).astype(np.int64)
            prn = np.asarray(handle["PRN"]).reshape(-1).astype(np.int16)
            code_freq = np.asarray(handle["code_freq_chips"]).reshape(-1).astype(float)
            doppler = np.asarray(handle["carrier_doppler_hz"]).reshape(-1).astype(float)
            taps = np.column_stack([
                np.asarray(handle[f"I_{tap}"]).reshape(-1) + 1j * np.asarray(handle[f"Q_{tap}"]).reshape(-1)
                for tap in ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
            ])
        normalized, prompt_valid = prompt_normalize(taps)
        dt = np.diff(samples).astype(float) / spec.sample_rate_hz
        candidates = np.flatnonzero(
            (prn[:-1] == prn[1:]) & (dt >= .019) & (dt <= .021)
            & prompt_valid[:-1] & prompt_valid[1:]
        )
        for row in candidates:
            aided_nominal = 1_023_000.0 * (1.0 + doppler[row] / 1_575_420_000.0)
            rate_residual = code_freq[row] - aided_nominal
            warped_1ms, support_1ms = warp_complex_taps(normalized[row], rate_residual * .001, 0.0)
            warped_actual, support_actual = warp_complex_taps(normalized[row], rate_residual * dt[row], 0.0)
            common = np.zeros(9, dtype=bool); common[1:8] = True
            support_1ms &= common; support_actual &= common
            target = normalized[row + 1]
            error_no_change.append(float(np.mean(np.abs(target[common] - normalized[row, common]) ** 2)))
            error_code_1ms.append(float(np.mean(np.abs(target[support_1ms] - warped_1ms[support_1ms]) ** 2)))
            error_code_actual_dt.append(float(np.mean(np.abs(target[support_actual] - warped_actual[support_actual]) ** 2)))
            count += 1
            if count >= limit:
                break
        if count >= limit:
            break
    return {
        "role": "clean_only_diagnostic_not_mapping_selection",
        "sampled_20ms_spaced_pairs": count,
        "prompt_referenced": True,
        "carrier_global_phase_applied": False,
        "mean_complex_mse": {
            "no_change": float(np.mean(error_no_change)),
            "row_action_for_next_1ms_code_only": float(np.mean(error_code_1ms)),
            "invalid_row_action_times_20ms_code_only": float(np.mean(error_code_actual_dt)),
        },
        "interpretation": "The row action governs only the next 1 ms buffer. Neither diagnostic bridges the 19 unobserved loop updates to the next retained row.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    support_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    contract: dict[str, object] = {
        "schema": "gnss-doppler-lab.trace-native-cadence.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": {"one_ms_s": [0.0009, 0.0011], "twenty_ms_s": [0.019, 0.021], "gap_s": ">0.021"},
        "transition_exclusion": "exclude both adjacent pairs at every 1ms<->20ms cadence change",
        "scenarios": {},
    }
    inventory: dict[str, object] = {"schema": "gnss-doppler-lab.trace-r1-input-inventory.v1", "datasets": {}}
    binding: dict[str, object] = {"schema": "gnss-doppler-lab.trace-r1-source-binding.v1", "datasets": {}}
    for spec in SPECS:
        rows = load_consecutive_pairs(spec)
        pre = np.ones(len(rows.prn), dtype=bool) if spec.onset_s is None else rows.time_s < spec.onset_s
        post = np.zeros(len(rows.prn), dtype=bool) if spec.onset_s is None else rows.time_s >= spec.onset_s
        blocks = block_support(rows)
        post_blocks = [block for block in blocks if spec.onset_s is not None and block["block_end_s"] > spec.onset_s]
        scenario_key = f"{spec.dataset}.{spec.scenario}"
        transitions = np.flatnonzero(rows.transition_excluded)
        contract["scenarios"][scenario_key] = {
            "sample_rate_hz": spec.sample_rate_hz,
            "onset_s": spec.onset_s,
            "pair_counts": cadence_counts(rows),
            "pre_onset_pair_counts": cadence_counts(rows, pre) if spec.onset_s is not None else None,
            "post_onset_pair_counts": cadence_counts(rows, post) if spec.onset_s is not None else None,
            "transition_excluded_pair_count": int(len(transitions)),
            "cadence_transition_times_s": [float(x) for x in rows.time_s[transitions]],
            "gap_distribution": gap_distribution(rows),
            "post_sync_cadence": "approximately_20_ms_retained_row_spacing",
            "post_sync_receiver_behavior": "1_ms_NCO_update_with_20_ms_dump_decimation",
            "valid_20ms_blocks": len(blocks),
            "post_onset_valid_20ms_blocks_ge4_prns": int(sum(block["valid_prn_count"] >= 4 for block in post_blocks)),
            "post_onset_max_valid_prns_per_0p5s_block": max((block["valid_prn_count"] for block in post_blocks), default=None),
            "post_onset_ge4_prn_support": bool(any(block["valid_prn_count"] >= 4 for block in post_blocks)),
        }
        for prn in np.unique(rows.prn):
            selected = rows.prn == prn
            counts = cadence_counts(rows, selected)
            support_rows.append({
                "row_type": "prn_summary", "dataset": spec.dataset, "scenario": spec.scenario,
                "prn": int(prn), "block_start_s": "", "block_end_s": "", "valid_prn_count": "",
                "pair_count": int(selected.sum()), "pairs_1ms": counts[CADENCE_1MS], "pairs_20ms": counts[CADENCE_20MS],
                "pre_onset_pairs_1ms": cadence_counts(rows, selected & pre)[CADENCE_1MS] if spec.onset_s is not None else "",
                "pre_onset_pairs_20ms": cadence_counts(rows, selected & pre)[CADENCE_20MS] if spec.onset_s is not None else "",
                "post_onset_pairs_1ms": cadence_counts(rows, selected & post)[CADENCE_1MS] if spec.onset_s is not None else "",
                "post_onset_pairs_20ms": cadence_counts(rows, selected & post)[CADENCE_20MS] if spec.onset_s is not None else "",
            })
        for block in blocks:
            support_rows.append({
                "row_type": "block_support", "dataset": spec.dataset, "scenario": spec.scenario,
                "prn": "", **block, "pairs_1ms": "", "pairs_20ms": "", "pre_onset_pairs_1ms": "",
                "pre_onset_pairs_20ms": "", "post_onset_pairs_1ms": "", "post_onset_pairs_20ms": "",
            })
        for index in transitions:
            transition_rows.append({
                "dataset": spec.dataset, "scenario": spec.scenario, "channel": int(rows.channel[index]),
                "prn": int(rows.prn[index]), "time_s": float(rows.time_s[index]), "dt_s": float(rows.dt_s[index]),
                "cadence": str(rows.cadence[index]), "reason": "adjacent_to_1ms_20ms_transition",
            })
        manifest_path = spec.receiver_root / "manifest.json"
        config_path = spec.receiver_root / "receiver.conf"
        manifest = json.loads(manifest_path.read_text())
        mats = sorted((spec.receiver_root / "raw").glob("epl_tracking_ch_*.mat"))
        with h5py.File(mats[0], "r") as handle:
            fields = sorted(handle.keys())
        inventory["datasets"][scenario_key] = {
            "receiver_root": str(spec.receiver_root), "mat_count": len(mats), "fields": fields,
            "missing_contract_fields": sorted(set(CONTRACT_FIELDS) - set(fields)),
            "full_complex_9tap": all(field in fields for field in TAP_FIELDS), "sample_rate_hz": spec.sample_rate_hz,
        }
        iq_path, iq_sha = manifest_source(manifest)
        binding["datasets"][scenario_key] = {
            "raw_iq_path": iq_path, "raw_iq_sha256": iq_sha,
            "receiver_manifest_path": str(manifest_path), "receiver_manifest_sha256": sha256(manifest_path),
            "receiver_config_path": str(config_path), "receiver_config_sha256": sha256(config_path),
            "receiver_executable_sha256": manifest.get("receiver", {}).get("executable_sha256"),
            "source_mat_count": len(mats), "source_mat_sha256": {path.name: sha256(path) for path in mats},
        }
    with (args.out_dir / "cadence_support.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(support_rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(support_rows)
    with (args.out_dir / "cadence_transition_by_prn.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "scenario", "channel", "prn", "time_s", "dt_s", "cadence", "reason"), lineterminator="\n"); writer.writeheader(); writer.writerows(transition_rows)
    dump_json(args.out_dir / "cadence_contract.json", contract)
    dump_json(args.out_dir / "input_inventory.json", inventory)
    receiver_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RECEIVER_SOURCE, text=True).strip()
    receiver_diff = subprocess.check_output(["git", "diff", "--binary"], cwd=RECEIVER_SOURCE)
    source_text = TRACKING_SOURCE.read_text()
    evidence_tokens = ["do_correlation_step(in);", "run_dll_pll();", "update_tracking_vars();", "log_data();", "if (d_current_data_symbol == 0)"]
    mapping = {
        "schema": "gnss-doppler-lab.trace-action-mapping-validation.v1",
        "status": "INVALID_FOR_RETAINED_20MS_ROW_TO_NEXT_ROW",
        "route": "B",
        "source_authenticated": all(token in source_text for token in evidence_tokens),
        "receiver_source_root": str(RECEIVER_SOURCE), "receiver_source_commit": receiver_head,
        "receiver_source_diff_sha256": hashlib.sha256(receiver_diff).hexdigest(),
        "sequence": [
            "correlate the current 1 ms interval",
            "accumulate/save correlator and compute DLL/PLL discriminator at the loop boundary",
            "update tracking filter and NCO",
            "store code/carrier state for the next 1 ms input buffer",
            "emit a dump row only when current_data_symbol wraps to zero",
            "consume one current_prn_length_samples buffer and repeat",
        ],
        "field_semantics": {
            "code_error_chips": "current completed loop-boundary discriminator output",
            "code_error_filt_chips": "current completed loop-filter output",
            "carr_error_hz": "current completed carrier phase discriminator in cycles, despite field suffix",
            "carr_error_filt_hz": "current completed carrier loop-filter/NCO output in Hz",
            "code_freq_chips": "new code NCO rate applied to the next 1 ms input buffer",
            "carrier_doppler_hz": "new carrier NCO Doppler applied to the next 1 ms input buffer",
        },
        "sample_stamp": "intended current interval end: nitems_read(0)+new current_prn_length_samples; its +/-sample variation reflects next-buffer length, so use consecutive stamps only for observed row spacing",
        "retained_row_t_to_t_plus_1": False,
        "why_invalid": "At default extend_correlation_symbols=1, state 4 runs DLL/PLL and updates the NCO every 1 ms, but log_data is gated by current_data_symbol==0, producing about 20 ms row spacing. Nineteen intervening correlations/actions are not retained.",
        "transition_exclusion": "exclude both adjacent pairs at 1 ms<->20 ms transitions; exclude all gaps, reacquisitions, PRN changes, and non-native outliers",
        "nco_update_cadence": "1_ms_with_20_ms_dump_decimation",
        "carrier_handling": {
            "prompt_reference_removes_global_phase": True,
            "full_doppler_phase_rotation_allowed": False,
            "reason": "Applying 2*pi*carrier_doppler_hz*dt after Prompt-referenced normalization double-applies a removed common phase and cannot reconstruct 19 unobserved updates.",
            "allowed_context": ["carr_error_hz", "carr_error_filt_hz", "signed Prompt phase innovation if independently verified"],
        },
        "known_vector_tests": ["tests/test_trace_native_cadence.py"],
        "clean_reconstruction": {
            "TEXBAT.cleanStatic": clean_reconstruction(SPECS[0]),
            "OAKBAT.cleanStatic": clean_reconstruction(SPECS[4]),
        },
        "route_b_feasibility": {
            "feasible_with_current_retained_dumps": False,
            "raw_iq_available": True,
            "requires_receiver_source_change": True,
            "performed": False,
            "blocking_constraint": "Task scope permits work only in this project worktree; the receiver implementation is a separate dirty detached worktree and cannot be modified or rebuilt in-scope.",
            "required_patch": "emit per-channel complex nine taps and the receiver-applied next-interval actions every native 1 ms, plus loop-boundary/integration flags",
        },
    }
    binding["receiver_source"] = {
        "path": str(RECEIVER_SOURCE), "base_commit": receiver_head,
        "diff_sha256": hashlib.sha256(receiver_diff).hexdigest(), "tracking_source_sha256": sha256(TRACKING_SOURCE),
    }
    dump_json(args.out_dir / "action_mapping_validation.json", mapping)
    dump_json(args.out_dir / "source_binding.json", binding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
