#!/usr/bin/env python3
"""Generate and receive the paired simulation-v4 pilot dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab.gnss_sdr import run_receiver  # noqa: E402
from gnss_doppler_lab.simulation_v4 import (  # noqa: E402
    SimulationScenario,
    generate_simulation_campaign,
    load_simulation_campaign,
)
from gnss_doppler_lab.tracking_feature_dataset import export_tracking_feature_dataset  # noqa: E402
from gnss_doppler_lab.tracking_peaks import (  # noqa: E402
    available_tracking_prns,
    load_receiver_tracking_peak_series_segments,
)

DEFAULT_CONFIG = Path("configs/experiments/simulation_v4_pilot.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_label(scenario: SimulationScenario, time_s: float) -> tuple[str, str, int]:
    if scenario.kind == "steady_normal":
        return "normal", "steady_normal", 0
    if scenario.kind == "recovery_normal":
        event = scenario.outage
        assert event is not None
        if time_s < event.start_seconds:
            return "normal", "pre_event_normal", 0
        if time_s < event.end_seconds:
            return "normal", "normal_outage", 0
        if time_s < event.end_seconds + event.recovery_ramp_seconds:
            return "normal", "normal_recovery_ramp", 0
        return "normal", "post_recovery_normal", 0
    event = scenario.spoofing
    assert event is not None
    if time_s < event.start_seconds:
        return "normal", "pre_event_normal", 0
    if time_s < event.start_seconds + event.transition_seconds:
        return "spoofing", "carryoff_transition", 1
    return "spoofing", "carryoff_final", 1


def _combine_labeled_features(
    feature_paths: dict[str, Path],
    scenarios: tuple[SimulationScenario, ...],
    paired_group_id: str,
    output: Path,
) -> tuple[int, dict[str, int]]:
    by_name = {scenario.name: scenario for scenario in scenarios}
    extra = ["scenario_name", "scenario_kind", "paired_group_id", "event_state", "is_spoofing", "dataset_role"]
    fields: list[str] | None = None
    count = 0
    state_counts: dict[str, int] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as out:
        writer = None
        for name in sorted(feature_paths):
            scenario = by_name[name]
            with feature_paths[name].open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                source_fields = list(reader.fieldnames or [])
                if fields is None:
                    fields = source_fields + extra
                    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                elif source_fields + extra != fields:
                    raise RuntimeError(f"feature schema mismatch: {feature_paths[name]}")
                assert writer is not None
                for row in reader:
                    label, state, is_spoofing = _event_label(scenario, float(row["window_mid_s"]))
                    row.update({
                        "label": label,
                        "scenario_name": name,
                        "scenario_kind": scenario.kind,
                        "paired_group_id": paired_group_id,
                        "event_state": state,
                        "is_spoofing": is_spoofing,
                        "dataset_role": "simulation_only_pilot",
                    })
                    writer.writerow(row)
                    count += 1
                    state_counts[state] = state_counts.get(state, 0) + 1
    temporary.replace(output)
    return count, state_counts


def _lock_metrics(receiver_dir: Path, start_s: float, end_s: float) -> dict[str, float | int]:
    manifest = json.loads((receiver_dir / "manifest.json").read_text(encoding="utf-8"))
    sample_rate = float(manifest["source"]["sample_rate_hz"])
    raw_dir = receiver_dir / str(manifest.get("tracking", {}).get("raw_directory", "raw"))
    lock_values: list[np.ndarray] = []
    cn0_values: list[np.ndarray] = []
    for path in sorted(raw_dir.glob("epl_tracking_ch_*.mat")):
        with h5py.File(path, "r") as handle:
            prn = np.asarray(handle["PRN"]).reshape(-1)
            time_s = np.asarray(handle["PRN_start_sample_count"]).reshape(-1) / sample_rate
            lock = np.asarray(handle["carrier_lock_test"]).reshape(-1)
            cn0 = np.asarray(handle["CN0_SNV_dB_Hz"]).reshape(-1)
        selected = (
            (prn >= 1) & (prn <= 32) & (time_s >= start_s) & (time_s < end_s)
            & np.isfinite(lock) & np.isfinite(cn0)
        )
        lock_values.append(lock[selected].astype(np.float64))
        cn0_values.append(cn0[selected].astype(np.float64))
    locks = np.concatenate(lock_values) if lock_values else np.empty(0)
    cn0s = np.concatenate(cn0_values) if cn0_values else np.empty(0)
    if not locks.size:
        raise RuntimeError(f"no receiver lock epochs in [{start_s}, {end_s})")
    return {
        "start_time_s": start_s,
        "end_time_s": end_s,
        "epoch_count": int(locks.size),
        "carrier_lock_median": float(np.median(locks)),
        "carrier_lock_q10": float(np.quantile(locks, 0.1)),
        "carrier_lock_q90": float(np.quantile(locks, 0.9)),
        "carrier_lock_above_0_5_fraction": float(np.mean(locks > 0.5)),
        "cn0_db_hz_median": float(np.median(cn0s)),
    }



def _receiver_timeline(receiver_dir: Path, scenario: SimulationScenario) -> dict[str, object]:
    prns = available_tracking_prns(receiver_dir)
    segments: list[dict[str, object]] = []
    for prn in prns:
        for series in load_receiver_tracking_peak_series_segments(receiver_dir, prn):
            segments.append({
                "prn": prn,
                "channel": series.channel,
                "segment_index": series.segment_index,
                "start_time_s": float(series.time_s[0]),
                "end_time_s": float(series.time_s[-1]),
                "epoch_count": int(series.time_s.size),
                "times": series.time_s,
            })
    result: dict[str, object] = {
        "tracked_prns": prns,
        "tracked_prn_count": len(prns),
        "segment_count": len(segments),
        "segments": [{key: value for key, value in segment.items() if key != "times"} for segment in segments],
    }
    if scenario.kind == "steady_normal":
        result["acceptance"] = {"tracked_prns_at_least_4": len(prns) >= 4}
        return result
    onset = scenario.outage.start_seconds if scenario.outage else scenario.spoofing.start_seconds
    post_start = scenario.outage.end_seconds if scenario.outage else onset + scenario.spoofing.transition_seconds
    pre = {segment["prn"] for segment in segments if segment["start_time_s"] < onset - 1.0}
    post = {segment["prn"] for segment in segments if segment["end_time_s"] > post_start + 1.0}
    result.update({
        "event_onset_seconds": onset,
        "post_event_reference_seconds": post_start,
        "pre_event_prns": sorted(pre),
        "post_event_prns": sorted(post),
        "pre_post_common_prns": sorted(pre & post),
    })
    if scenario.kind == "recovery_normal":
        event = scenario.outage
        assert event is not None
        interior_start = event.start_seconds + min(1.0, (event.end_seconds - event.start_seconds) / 4.0)
        interior_end = event.end_seconds - min(1.0, (event.end_seconds - event.start_seconds) / 4.0)
        outage_epochs = sum(
            int(((segment["times"] >= interior_start) & (segment["times"] < interior_end)).sum())
            for segment in segments
        )
        reacquisition = [
            segment for segment in segments
            if event.end_seconds <= float(segment["start_time_s"]) < event.end_seconds + 8.0
        ]
        lock_windows = {
            "pre_event": _lock_metrics(receiver_dir, max(0.0, event.start_seconds - 8.0), event.start_seconds - 1.0),
            "outage_interior": _lock_metrics(receiver_dir, interior_start, interior_end),
            "post_recovery": _lock_metrics(
                receiver_dir,
                event.end_seconds + event.recovery_ramp_seconds + 1.0,
                event.end_seconds + event.recovery_ramp_seconds + 9.0,
            ),
        }
        pre_lock = lock_windows["pre_event"]
        outage_lock = lock_windows["outage_interior"]
        post_lock = lock_windows["post_recovery"]
        loss_and_recovery = (
            pre_lock["carrier_lock_median"] > 0.5
            and outage_lock["carrier_lock_median"] < 0.2
            and outage_lock["carrier_lock_above_0_5_fraction"] < 0.05
            and post_lock["carrier_lock_median"] > 0.5
        )
        result.update({
            "outage_interior": [interior_start, interior_end],
            "outage_tracking_epoch_count": outage_epochs,
            "lock_windows": lock_windows,
            "loss_and_recovery_observed": loss_and_recovery,
            "reacquisition_segments": [
                {key: value for key, value in segment.items() if key != "times"} for segment in reacquisition
            ],
            "reacquisition_segment_count": len(reacquisition),
            "acceptance": {
                "pre_event_prns_at_least_4": len(pre) >= 4,
                "post_event_prns_at_least_4": len(post) >= 4,
                "common_pre_post_prns_at_least_2": len(pre & post) >= 2,
                "at_least_one_post_restore_segment": bool(reacquisition),
                "carrier_lock_loss_and_recovery_observed": loss_and_recovery,
            },
        })
    else:
        result["acceptance"] = {
            "pre_event_prns_at_least_4": len(pre) >= 4,
            "post_transition_prns_at_least_4": len(post) >= 4,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gnss-sdr", default="gnss-sdr")
    parser.add_argument("--receiver-timeout", type=int, default=1200)
    parser.add_argument("--channel-count", type=int, default=11)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--reuse-generated", action="store_true")
    parser.add_argument("--reuse-receiver", action="store_true")
    args = parser.parse_args(argv)

    if args.reuse_receiver and not args.reuse_generated:
        raise RuntimeError("--reuse-receiver requires --reuse-generated")
    started = time.time()
    campaign = load_simulation_campaign(args.config)
    if args.reuse_generated:
        campaign_manifest_path = campaign.output_root / "campaign_manifest.json"
        if not campaign_manifest_path.is_file():
            raise RuntimeError("--reuse-generated requires an existing campaign_manifest.json")
        reused = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
        expected_config_sha = reused.get("campaign", {}).get("source_config_sha256")
        if expected_config_sha != _sha256(campaign.source_config_path):
            raise RuntimeError("generated campaign config hash does not match the requested config")
    else:
        campaign_manifest_path = generate_simulation_campaign(campaign)
    if args.generate_only:
        print(campaign_manifest_path)
        return 0
    campaign_manifest = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
    receiver_root = campaign.output_root / "receiver"
    feature_root = campaign.output_root / "features"
    receiver_root.mkdir(exist_ok=args.reuse_receiver)
    feature_root.mkdir(exist_ok=args.reuse_receiver)
    scenario_by_name = {scenario.name: scenario for scenario in campaign.scenarios}
    receiver_manifests: dict[str, str] = {}
    feature_paths: dict[str, Path] = {}
    validations: dict[str, object] = {}
    for name in sorted(campaign_manifest["rf_manifests"]):
        rf_manifest = Path(campaign_manifest["rf_manifests"][name])
        print(f"[receiver] {name}", flush=True)
        rf_document = json.loads(rf_manifest.read_text(encoding="utf-8"))
        existing_receiver = receiver_root / rf_document["run_id"] / "manifest.json"
        feature_path = feature_root / f"{name}_tracking_features.csv"
        if args.reuse_receiver:
            if not existing_receiver.is_file() or not feature_path.is_file():
                raise RuntimeError(f"--reuse-receiver missing outputs for {name}")
            receiver_document = json.loads(existing_receiver.read_text(encoding="utf-8"))
            if receiver_document.get("source", {}).get("rf_manifest_sha256") != _sha256(rf_manifest):
                raise RuntimeError(f"receiver RF manifest hash mismatch for {name}")
            receiver_manifest = existing_receiver
        else:
            receiver_manifest = run_receiver(
                rf_manifest,
                receiver_root,
                executable=args.gnss_sdr,
                channel_count=args.channel_count,
                timeout_seconds=args.receiver_timeout,
            )
            export_tracking_feature_dataset(
                [receiver_manifest.parent],
                output_path=feature_path,
                manifest_output_path=feature_path.with_suffix(".manifest.json"),
                window_s=1.0,
                stride_s=0.5,
                min_epochs=4,
                label=scenario_by_name[name].kind,
            )
        receiver_manifests[name] = str(receiver_manifest)
        feature_paths[name] = feature_path
        validations[name] = _receiver_timeline(receiver_manifest.parent, scenario_by_name[name])
    dataset = campaign.output_root / "simulation_v4_tracking_features_labeled.csv"
    row_count, state_counts = _combine_labeled_features(feature_paths, campaign.scenarios, campaign.name, dataset)
    summary = {
        "schema": "gnss-doppler-lab.simulation-v4-pipeline",
        "schema_version": 1,
        "campaign_manifest": str(campaign_manifest_path),
        "campaign_manifest_sha256": _sha256(campaign_manifest_path),
        "receiver_manifests": receiver_manifests,
        "receiver_validation": validations,
        "dataset": {
            "path": str(dataset),
            "sha256": _sha256(dataset),
            "row_count": row_count,
            "paired_group_id": campaign.name,
            "event_state_counts": state_counts,
            "role": "simulation-only pilot for pipeline qualification; TEXBAT remains external validation",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = campaign.output_root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
