#!/usr/bin/env python3
"""Run the preregistered TUNI GPS partial-spoofer external validation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
from scipy.stats import f as f_distribution


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry  # noqa: E402
from gnss_doppler_lab.correlator_geometry import (  # noqa: E402
    TemplateDelayEstimator,
    build_complex_template_bank,
    complex_profile_features,
)
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    parse_gnss_sdr_gps_ephemeris_xml,
    satellite_observation,
)
from gnss_doppler_lab.tracking_peaks import (  # noqa: E402
    available_tracking_prns,
    load_receiver_tracking_peak_series_segments,
)
from gnss_doppler_lab.trajectory import llh_to_ecef  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/experiments/tuni_gps_partial_spoof_external_v1.json"
PROTOCOL = ROOT / "docs/results/tuni_gps_partial_spoof_external_protocol_v1.md"
EXPECTED_IDS = ("C-5", "SS-17", "SS-18", "SS-20")


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def valid_nmea_sentence(line: str) -> list[str] | None:
    text = line.strip()
    if not text.startswith("$") or "*" not in text:
        return None
    body, raw = text[1:].rsplit("*", 1)
    try:
        expected = int(raw[:2], 16)
    except ValueError:
        return None
    checksum = 0
    for character in body:
        checksum ^= ord(character)
    return body.split(",") if checksum == expected else None


def nmea_hms(value: str) -> tuple[int, int, float]:
    if len(value) < 6:
        raise ValueError("invalid NMEA time")
    return int(value[:2]), int(value[2:4]), float(value[4:])


def nmea_degree(value: str, hemisphere: str) -> float:
    width = 2 if hemisphere in ("N", "S") else 3
    result = float(value[:width]) + float(value[width:]) / 60.0
    return -result if hemisphere in ("S", "W") else result


def gps_week_and_tow(date: Any, hms: tuple[int, int, float]) -> tuple[int, float]:
    hour, minute, second = hms
    timestamp = datetime(
        date.year, date.month, date.day, hour, minute, int(second), tzinfo=timezone.utc
    )
    seconds = (
        (timestamp - datetime(1980, 1, 6, tzinfo=timezone.utc)).total_seconds()
        + 18.0 + (second - int(second))
    )
    return int(seconds // 604800.0), seconds % 604800.0


def verify(path: str | Path, expected: str, label: str) -> Path:
    target = resolve(path)
    if not target.is_file():
        raise FileNotFoundError(f"{label} missing: {target}")
    observed = file_hash(target)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return target


def git_clean_commit() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if status:
        raise RuntimeError("release requires a clean git worktree")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def ishort_source_item_count(duration_s: float, sample_rate_hz: int) -> int:
    if duration_s <= 0 or sample_rate_hz <= 0:
        raise ValueError("duration and sample rate must be positive")
    return round(duration_s * sample_rate_hz * 2)


def render_receiver_config(
    *, iq_path: Path, output_dir: Path, duration_s: float,
    input_rate_hz: int = 50_000_000, internal_rate_hz: int = 5_000_000,
    channel_count: int = 31, acquisition_pfa: float = 0.01,
    acquisition_max_dwells: int = 5,
) -> str:
    if not 60.0 <= duration_s <= 150.0:
        raise ValueError("external run duration must be in [60,150] seconds")
    if not 8 <= channel_count <= 31:
        raise ValueError("channel_count must be in [8,31]")
    if input_rate_hz % internal_rate_hz:
        raise ValueError("input/internal sample-rate ratio must be integral")
    raw = output_dir / "raw"
    nmea = output_dir / "nmea_pvt.nmea"
    return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={internal_rate_hz}

SignalSource.implementation=File_Signal_Source
SignalSource.filename={iq_path.resolve()}
SignalSource.item_type=ishort
SignalSource.sampling_frequency={input_rate_hz}
SignalSource.samples={ishort_source_item_count(duration_s, input_rate_hz)}
SignalSource.repeat=false
SignalSource.dump=false
SignalSource.enable_throttle_control=false

SignalConditioner.implementation=Signal_Conditioner
DataTypeAdapter.implementation=Ishort_To_Complex
DataTypeAdapter.swap_endian=true
InputFilter.implementation=Pass_Through
InputFilter.input_item_type=gr_complex
InputFilter.output_item_type=gr_complex
Resampler.implementation=Direct_Resampler
Resampler.sample_freq_in={input_rate_hz}
Resampler.sample_freq_out={internal_rate_hz}
Resampler.item_type=gr_complex

Channels_1C.count={channel_count}
Channels.in_acquisition={channel_count}
Channel.signal=1C

Acquisition_1C.implementation=GPS_L1_CA_PCPS_Acquisition
Acquisition_1C.item_type=gr_complex
Acquisition_1C.coherent_integration_time_ms=1
Acquisition_1C.pfa={acquisition_pfa}
Acquisition_1C.max_dwells={acquisition_max_dwells}
Acquisition_1C.doppler_max=6000
Acquisition_1C.doppler_step=125
Acquisition_1C.blocking=true
Acquisition_1C.dump=false

Tracking_1C.implementation=GPS_L1_CA_DLL_PLL_Tracking
Tracking_1C.item_type=gr_complex
Tracking_1C.pll_bw_hz=20.0
Tracking_1C.dll_bw_hz=1.5
Tracking_1C.order=3
Tracking_1C.early_late_space_chips=0.125
Tracking_1C.early_late_space_narrow_chips=0.125
Tracking_1C.tap_count=9
Tracking_1C.tap_spacing_chips=0.125
Tracking_1C.dump=true
Tracking_1C.dump_filename={raw / 'epl_tracking_ch_'}

TelemetryDecoder_1C.implementation=GPS_L1_CA_Telemetry_Decoder
TelemetryDecoder_1C.dump=false

Observables.implementation=Hybrid_Observables
Observables.dump=true
Observables.dump_filename={raw / 'observables.dat'}

PVT.implementation=RTKLIB_PVT
PVT.positioning_mode=Single
PVT.output_rate_ms=100
PVT.display_rate_ms=1000
PVT.nmea_output_enabled=true
PVT.nmea_dump_filename={nmea}
PVT.flag_nmea_tty_port=false
PVT.flag_rtcm_server=false
PVT.flag_rtcm_tty_port=false
PVT.kml_output_enabled=false
PVT.gpx_output_enabled=false
PVT.geojson_output_enabled=false
PVT.dump=false
"""


def tracking_support(paths: list[Path]) -> tuple[list[int], int]:
    prns: set[int] = set()
    epochs = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "PRN" not in handle:
                continue
            values = np.asarray(handle["PRN"]).reshape(-1)
        if values.shape == (2,) and np.array_equal(values, np.asarray([1, 0])):
            continue
        valid = [int(value) for value in values if 1 <= int(value) <= 32]
        prns.update(valid)
        epochs += len(valid)
    return sorted(prns), epochs


def axis(low: float, high: float, step: float) -> np.ndarray:
    count = int(round((high - low) / step)) + 1
    return np.linspace(low, high, count, dtype=np.float64)


def build_estimator(config: dict[str, Any]) -> TemplateDelayEstimator:
    correlator = config["correlator"]
    template = config["template_estimator"]
    phase_axis = np.linspace(
        -np.pi, np.pi, int(template["phase_count"]), endpoint=False,
        dtype=np.float64,
    )
    bank = build_complex_template_bank(
        correlator["tap_offsets_chips"],
        prompt_index=int(correlator["prompt_index"]),
        delays_chips=axis(
            float(template["delay_min_chips"]),
            float(template["delay_max_chips"]),
            float(template["delay_step_chips"]),
        ),
        centers_chips=axis(
            float(template["center_min_chips"]),
            float(template["center_max_chips"]),
            float(template["center_step_chips"]),
        ),
        amplitude_ratios=axis(
            float(template["amplitude_min"]),
            float(template["amplitude_max"]),
            float(template["amplitude_step"]),
        ),
        phases_rad=phase_axis,
    )
    return TemplateDelayEstimator(bank)


def consolidate_profile_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["bin_index"]), int(row["prn"])), []).append(row)
    combined: list[dict[str, Any]] = []
    for (bin_index, prn), group in sorted(grouped.items()):
        combined.append({
            "bin_index": bin_index,
            "bin_start_s": float(group[0]["bin_start_s"]),
            "prn": prn,
            "prn_name": str(group[0]["prn_name"]),
            "epoch_count": max(int(row["epoch_count"]) for row in group),
            "estimated_delay_chips": float(np.median([
                row["estimated_delay_chips"] for row in group
            ])),
            "median_template_distance": float(np.median([
                row["median_template_distance"] for row in group
            ])),
            "median_cn0_db_hz": float(np.median([
                row["median_cn0_db_hz"] for row in group
            ])),
        })
    return combined


def profile_rows(
    run_dir: Path, estimator: TemplateDelayEstimator, *, bin_seconds: float,
    minimum_epochs: int, start_s: float, end_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prn_name in available_tracking_prns(run_dir):
        prn = int(prn_name[1:])
        segments = load_receiver_tracking_peak_series_segments(
            run_dir, prn_name, tap_count=9, require_complex_taps=True
        )
        for segment in segments:
            selected_time = (segment.time_s >= start_s) & (segment.time_s < end_s)
            if not np.any(selected_time):
                continue
            times = segment.time_s[selected_time]
            taps = segment.complex_taps[selected_time]
            features = complex_profile_features(taps, prompt_index=4)
            estimates, distances, _ = estimator.estimate(features)
            cn0 = segment.cn0_db_hz[selected_time]
            bins = np.floor(times / bin_seconds).astype(np.int64)
            for bin_index in np.unique(bins):
                selected_bin = bins == bin_index
                count = int(np.count_nonzero(selected_bin))
                if count < minimum_epochs:
                    continue
                rows.append({
                    "bin_index": int(bin_index),
                    "bin_start_s": float(bin_index * bin_seconds),
                    "prn": prn, "prn_name": prn_name, "epoch_count": count,
                    "estimated_delay_chips": float(np.median(estimates[selected_bin])),
                    "median_template_distance": float(np.median(distances[selected_bin])),
                    "median_cn0_db_hz": float(np.median(cn0[selected_bin])),
                })
    return consolidate_profile_rows(rows)


def partial_f_p_value(residual: float, prn_count: int) -> float:
    value = float(residual)
    count = int(prn_count)
    if not np.isfinite(value) or value < 0.0 or value > 1.0 + 1e-9:
        raise ValueError("geometry residual must be in [0,1]")
    if count <= 4:
        raise ValueError("partial-F requires more than four PRNs")
    value = min(max(value, np.finfo(float).tiny), 1.0)
    statistic = (1.0 - value) * (count - 4) / (3.0 * value)
    return float(f_distribution.sf(statistic, 3, count - 4))


def persistent_alarm(
    raw: np.ndarray, bins: np.ndarray, *, window: int, required: int,
) -> np.ndarray:
    alarms = np.asarray(raw, dtype=bool)
    indices = np.asarray(bins, dtype=np.int64)
    if alarms.ndim != 1 or indices.shape != alarms.shape:
        raise ValueError("raw alarms and bins must be matching vectors")
    if window < 1 or required < 1 or required > window:
        raise ValueError("invalid persistence rule")
    result = np.zeros(len(alarms), dtype=bool)
    for end in range(window - 1, len(alarms)):
        start = end - window + 1
        if np.array_equal(
            indices[start : end + 1], np.arange(indices[start], indices[start] + window)
        ):
            result[end] = int(alarms[start : end + 1].sum()) >= required
    return result


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.tuni-gps-partial-spoof-external-config":
        raise ValueError("unsupported config schema")
    experiment = config["experiment"]
    if experiment.get("threshold_refitting") is not False:
        raise ValueError("threshold refitting must be disabled")
    if experiment.get("post_attack_tuning_or_retest") is not False:
        raise ValueError("post-attack tuning must be disabled")
    scenarios = config["dataset"]["scenarios"]
    if tuple(row["id"] for row in scenarios) != EXPECTED_IDS:
        raise ValueError("scenario order or membership drifted")
    if scenarios[0]["role"] != "clean_negative" or any(
        row["role"] != "attack" for row in scenarios[1:]
    ):
        raise ValueError("scenario roles drifted")
    expected_targets = {"C-5": [], "SS-17": [1], "SS-18": [1, 2], "SS-20": [1, 2, 21, 32]}
    if {row["id"]: row["spoofed_prns"] for row in scenarios} != expected_targets:
        raise ValueError("documented target PRNs drifted")
    analysis = config["analysis"]
    if int(analysis["minimum_primary_prns"]) != 8:
        raise ValueError("primary support must remain N>=8")
    if int(analysis["secondary_boundary_prns"]) != 7:
        raise ValueError("secondary support boundary must remain N=7")


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    dataset = config["dataset"]
    manifest_path = verify(
        dataset["download_manifest"], dataset["download_manifest_sha256"],
        "download manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("scientific_access") != "download_and_checksum_only_no_raw_analysis":
        raise ValueError("download manifest does not preserve the sealed boundary")
    payloads = {row["scenario"]: row for row in manifest["payloads"]}
    for scenario in dataset["scenarios"]:
        entry = payloads.get(scenario["id"])
        if entry is None or entry["md5"] != scenario["md5"]:
            raise ValueError(f"download manifest mismatch for {scenario['id']}")
        if int(entry["size"]) != int(dataset["expected_bytes_each"]):
            raise ValueError(f"download size mismatch for {scenario['id']}")
        verify(scenario["readme"], scenario["readme_sha256"], f"{scenario['id']} README")
    frozen = config["frozen"]
    clean_manifest_path = verify(
        frozen["clean_preflight_manifest"]["path"],
        frozen["clean_preflight_manifest"]["sha256"], "clean preflight manifest",
    )
    clean_manifest = json.loads(clean_manifest_path.read_text(encoding="utf-8"))
    if clean_manifest.get("compatible") is not True or clean_manifest["source"]["scenario"] != "C-5":
        raise ValueError("pinned C-5 clean preflight is not compatible")
    receiver = verify(
        frozen["receiver"]["path"], frozen["receiver"]["sha256"], "receiver executable"
    )
    template = verify(
        frozen["delay_template"]["path"], frozen["delay_template"]["sha256"],
        "delay template",
    )
    verify(
        frozen["threshold_source"]["path"], frozen["threshold_source"]["sha256"],
        "threshold source",
    )
    verify(
        frozen["support_audit"]["config"], frozen["support_audit"]["config_sha256"],
        "support audit config",
    )
    for item in frozen["implementation"]:
        verify(item["path"], item["sha256"], f"implementation {item['path']}")
    return {"receiver": receiver, "template": template, "download_manifest": manifest_path}


def run_receiver(
    scenario: dict[str, Any], config: dict[str, Any], output_dir: Path,
    executable: Path,
) -> dict[str, Any]:
    dataset = config["dataset"]
    receiver = config["frozen"]["receiver"]
    iq = Path(scenario["iq"]).resolve()
    if not iq.is_file() or iq.stat().st_size != int(dataset["expected_bytes_each"]):
        raise ValueError(f"{scenario['id']} IQ byte count mismatch")
    print(f"[tuni-gps] hashing {scenario['id']} sealed IQ", flush=True)
    observed_md5 = file_hash(iq, "md5")
    if observed_md5 != scenario["md5"]:
        raise ValueError(f"{scenario['id']} IQ MD5 mismatch")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    (output_dir / "raw").mkdir(parents=True)
    config_path = output_dir / "receiver.conf"
    config_path.write_text(
        render_receiver_config(
            iq_path=iq,
            output_dir=output_dir,
            duration_s=float(receiver["duration_seconds"]),
            input_rate_hz=int(dataset["input_sample_rate_hz"]),
            internal_rate_hz=int(receiver["internal_sample_rate_hz"]),
            channel_count=int(receiver["channel_count"]),
            acquisition_pfa=float(receiver["acquisition_pfa"]),
            acquisition_max_dwells=int(receiver["acquisition_max_dwells"]),
        ),
        encoding="utf-8",
    )
    command = [str(executable), f"--config_file={config_path}", "--keyboard=false"]
    print(f"[tuni-gps] receiver {scenario['id']} started", flush=True)
    completed = subprocess.run(
        command, cwd=output_dir, capture_output=True, text=True,
        timeout=7200, check=False,
    )
    (output_dir / "receiver.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    mats = sorted((output_dir / "raw").glob("epl_tracking_ch_*.mat"))
    prns, epochs = tracking_support(mats)
    ephemeris = output_dir / "gps_ephemeris.xml"
    nmea = output_dir / "nmea_pvt.nmea"
    observables = output_dir / "raw" / "observables.mat"
    manifest = {
        "schema": "gnss-doppler-lab.tuni-gps-partial-spoof-receiver.v1",
        "scenario": scenario["id"],
        "source": {
            "iq": str(iq), "iq_bytes": iq.stat().st_size, "iq_md5": observed_md5,
            "input_sample_rate_hz": int(dataset["input_sample_rate_hz"]),
            "sample_rate_hz": int(receiver["internal_sample_rate_hz"]),
            "requested_duration_s": float(receiver["duration_seconds"]),
            "sample_format": dataset["sample_format"],
        },
        "receiver": {
            "executable": str(executable), "executable_sha256": file_hash(executable),
            "config": config_path.name, "config_sha256": file_hash(config_path),
            "return_code": completed.returncode,
        },
        "acquisition": {"channel_count": int(receiver["channel_count"]), "tracked_prns": prns},
        "tracking": {
            "raw_directory": "raw", "tap_count": 9, "tap_spacing_chips": 0.125,
            "mat_file_count": len(mats), "valid_prns": prns,
            "valid_prn_count": len(prns), "valid_epoch_count": epochs,
        },
        "navigation": {
            "ephemeris_present": ephemeris.is_file(), "nmea_present": nmea.is_file(),
            "observables_present": observables.is_file(),
        },
        "compatible": bool(
            completed.returncode == 0 and prns and epochs > 0 and ephemeris.is_file()
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{scenario['id']} receiver exited {completed.returncode}")
    print(
        f"[tuni-gps] receiver {scenario['id']} complete: {len(prns)} PRNs, {epochs} epochs",
        flush=True,
    )
    return manifest


def observables_tow0(path: Path) -> float:
    with h5py.File(path, "r") as handle:
        if "RX_time" not in handle:
            raise ValueError("observables MAT is missing RX_time")
        values = np.asarray(handle["RX_time"]).reshape(-1).astype(np.float64)
    usable = values[np.isfinite(values) & (values > 0.0)]
    if not len(usable):
        raise ValueError("observables RX_time has no positive value")
    return float(np.min(usable))


def clean_static_position(nmea_path: Path, tow0_s: float) -> dict[str, Any]:
    current_date = None
    points: list[tuple[float, float, float, float]] = []
    for line in nmea_path.read_text(errors="replace").splitlines():
        fields = valid_nmea_sentence(line)
        if not fields:
            continue
        kind = fields[0][-3:]
        try:
            if kind == "RMC" and len(fields) > 9 and fields[2] == "A":
                current_date = datetime.strptime(fields[9], "%d%m%y").date()
            elif (
                kind == "GGA" and current_date is not None and len(fields) > 11
                and int(fields[6]) > 0 and fields[9] and fields[10] == "M"
            ):
                _, tow = gps_week_and_tow(current_date, nmea_hms(fields[1]))
                relative = (tow - tow0_s + 302400.0) % 604800.0 - 302400.0
                if 60.0 <= relative < 140.0:
                    points.append((
                        relative, nmea_degree(fields[2], fields[3]),
                        nmea_degree(fields[4], fields[5]), float(fields[9]),
                    ))
        except (IndexError, ValueError):
            continue
    if not points:
        raise ValueError("C-5 has no valid NMEA GGA position in [60,140) s")
    values = np.asarray(points, dtype=np.float64)
    llh = tuple(float(np.median(values[:, index])) for index in (1, 2, 3))
    ecef = tuple(float(value) for value in llh_to_ecef(*llh))
    return {
        "llh": llh, "ecef": ecef, "sample_count": len(points),
        "relative_time_range_s": [float(values[:, 0].min()), float(values[:, 0].max())],
        "source": "clean C-5 checksum-valid NMEA GGA only",
    }


def static_los_map(ephemeris_path: Path, receiver_ecef: np.ndarray) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(ephemeris_path)
    los: dict[int, np.ndarray] = {}
    exclusions: dict[int, str] = {}
    snapshot_tow: dict[int, float] = {}
    for prn, ephemeris in sorted(ephemerides.items()):
        if ephemeris.SV_health != 0:
            exclusions[prn] = f"SV_health={ephemeris.SV_health}"
            continue
        if ephemeris.decoded_tow is None:
            exclusions[prn] = "decoded_tow unavailable"
            continue
        tow = float(ephemeris.decoded_tow)
        los[prn] = np.asarray(
            satellite_observation(receiver_ecef, ephemeris, tow).los_ecef,
            dtype=np.float64,
        )
        snapshot_tow[prn] = tow
    return los, {
        "decoded_ephemeris_prns": sorted(ephemerides),
        "usable_snapshot_prns": sorted(los),
        "snapshot_tow_s_by_prn": snapshot_tow,
        "excluded_prns": exclusions,
    }


def apply_persistence(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    analysis = config["analysis"]
    bins = np.asarray([row["bin_index"] for row in rows], dtype=np.int64)
    primary_raw = np.asarray([
        bool(row["primary_eligible"] and row["raw_spoof_alarm"]) for row in rows
    ])
    legacy_raw = np.asarray([
        bool(row["primary_eligible"] and row["legacy_raw_spoof_alarm"]) for row in rows
    ])
    persistent = persistent_alarm(
        primary_raw, bins, window=int(analysis["persistence_window_bins"]),
        required=int(analysis["persistence_required_bins"]),
    )
    legacy_persistent = persistent_alarm(
        legacy_raw, bins, window=int(analysis["persistence_window_bins"]),
        required=int(analysis["persistence_required_bins"]),
    )
    for row, alarm, legacy_alarm in zip(rows, persistent, legacy_persistent):
        row["persistent_spoof_alarm"] = bool(alarm)
        row["legacy_persistent_spoof_alarm"] = bool(legacy_alarm)


def analyze_scenario(
    scenario: dict[str, Any], receiver_dir: Path, estimator: Any,
    receiver_ecef: np.ndarray, config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    analysis = config["analysis"]
    start_s, end_s = map(float, analysis["analysis_interval_seconds"])
    delays = profile_rows(
        receiver_dir, estimator, bin_seconds=float(analysis["bin_seconds"]),
        minimum_epochs=int(analysis["minimum_epochs_per_prn_bin"]),
        start_s=start_s, end_s=end_s,
    )
    los_by_prn, ephemeris_report = static_los_map(
        receiver_dir / "gps_ephemeris.xml", receiver_ecef
    )
    by_bin: dict[int, list[dict[str, Any]]] = {}
    for row in delays:
        if int(row["prn"]) in los_by_prn:
            by_bin.setdefault(int(row["bin_index"]), []).append(row)
    geometries: list[dict[str, Any]] = []
    for bin_index, entries in sorted(by_bin.items()):
        if len(entries) < 5:
            continue
        los = np.asarray([los_by_prn[int(row["prn"])] for row in entries])
        delay = np.asarray([float(row["estimated_delay_chips"]) for row in entries])
        fit = fit_clock_centered_geometry(los, delay)
        residual = float(fit.clock_centered_normalized_residual)
        p_value = partial_f_p_value(residual, len(entries))
        primary = len(entries) >= int(analysis["minimum_primary_prns"])
        geometries.append({
            "scenario": scenario["id"], "role": scenario["role"],
            "bin_index": bin_index, "bin_start_s": float(bin_index),
            "bin_end_s": float(bin_index + 1), "prn_count": len(entries),
            "prns": " ".join(f"G{int(row['prn']):02d}" for row in entries),
            "fit_rank": int(fit.rank),
            "clock_centered_geometry_residual": residual,
            "directional_geometry_coherence": float(fit.directional_coherence),
            "partial_f_p_value": p_value,
            "partial_f_score": float(-np.log10(max(p_value, np.finfo(float).tiny))),
            "primary_eligible": primary,
            "seven_prn_secondary": len(entries) == int(analysis["secondary_boundary_prns"]),
            "raw_spoof_alarm": p_value <= float(analysis["partial_f_p_alarm_threshold"]),
            "legacy_raw_spoof_alarm": residual <= float(analysis["legacy_residual_alarm_threshold"]),
        })
    if geometries:
        apply_persistence(geometries, config)
    target = set(map(int, scenario["spoofed_prns"]))
    observed_delay_prns = {int(row["prn"]) for row in delays}
    target_covered = sorted(target.intersection(observed_delay_prns))
    primary_rows = [row for row in geometries if row["primary_eligible"]]
    persistent = [row for row in primary_rows if row["persistent_spoof_alarm"]]
    legacy = [row for row in primary_rows if row["legacy_persistent_spoof_alarm"]]
    summary = {
        "scenario": scenario["id"], "role": scenario["role"],
        "documented_spoofed_prns": sorted(target),
        "tracked_target_prns": target_covered,
        "all_documented_spoof_prns_tracked": target_covered == sorted(target),
        "delay_prns": sorted(observed_delay_prns), "delay_bin_prn_rows": len(delays),
        "geometry_bin_count": len(geometries), "primary_bin_count": len(primary_rows),
        "seven_prn_secondary_bin_count": sum(row["seven_prn_secondary"] for row in geometries),
        "minimum_prn_count": min((row["prn_count"] for row in geometries), default=None),
        "maximum_prn_count": max((row["prn_count"] for row in geometries), default=None),
        "persistent_alarm_count": len(persistent),
        "persistent_alarm_rate": len(persistent) / len(primary_rows) if primary_rows else None,
        "detected": bool(persistent),
        "first_persistent_alarm_time_s": persistent[0]["bin_end_s"] if persistent else None,
        "median_partial_f_p_value": float(np.median([row["partial_f_p_value"] for row in primary_rows])) if primary_rows else None,
        "median_geometry_residual": float(np.median([row["clock_centered_geometry_residual"] for row in primary_rows])) if primary_rows else None,
        "legacy_persistent_alarm_count": len(legacy),
        "legacy_persistent_alarm_rate": len(legacy) / len(primary_rows) if primary_rows else None,
        "ephemeris": ephemeris_report,
    }
    return delays, geometries, summary


def serial_auc(negative: list[float], positive: list[float]) -> float | None:
    if not negative or not positive:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return float(wins / (len(positive) * len(negative)))


def terminal_decision(summaries: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    analysis = config["analysis"]
    minimum_bins = int(analysis["minimum_primary_bins_per_recording"])
    support_by_scenario: dict[str, bool] = {}
    for scenario_id, summary in summaries.items():
        target_support = (
            summary["all_documented_spoof_prns_tracked"]
            if analysis["require_all_documented_spoof_prns_tracked"] else True
        )
        support_by_scenario[scenario_id] = bool(
            summary["primary_bin_count"] >= minimum_bins and target_support
        )
    support = all(support_by_scenario.values())
    clean_rate = summaries["C-5"]["persistent_alarm_rate"]
    specificity = bool(
        clean_rate is not None
        and clean_rate <= float(analysis["maximum_clean_persistent_alarm_rate"])
    )
    attacks = [summaries[key] for key in EXPECTED_IDS[1:]]
    detected = sum(bool(row["detected"]) for row in attacks)
    sensitivity = bool(
        detected >= int(analysis["minimum_detected_attack_recordings"])
        and (summaries["SS-20"]["detected"] if analysis["require_ss20_detection"] else True)
    )
    if not support:
        decision = "INSUFFICIENT_SUPPORT"
    elif specificity and sensitivity:
        decision = "REAL_PARTIAL_SPOOF_TRANSFER_SUPPORTED"
    elif specificity:
        decision = "SPECIFICITY_ONLY_DETECTION_NOT_SUPPORTED"
    else:
        decision = "REAL_PARTIAL_SPOOF_TRANSFER_NOT_SUPPORTED"
    return {
        "decision": decision, "support_sufficient": support,
        "support_by_scenario": support_by_scenario,
        "clean_specificity_pass": specificity,
        "attack_sensitivity_pass": sensitivity,
        "detected_attack_recordings": detected,
        "required_detected_attack_recordings": int(analysis["minimum_detected_attack_recordings"]),
        "ss20_detection_required": bool(analysis["require_ss20_detection"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-token", required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if args.release_token != config["experiment"]["release_token"]:
        raise ValueError("release token mismatch")
    release_commit = git_clean_commit()
    frozen = verify_frozen_inputs(config)
    output = Path(config["output_root"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    state_path = output / "release_state.json"
    state = {
        "schema": "gnss-doppler-lab.tuni-gps-partial-spoof-release-state.v1",
        "release_commit": release_commit, "config_sha256": file_hash(config_path),
        "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
        "phase": "released_before_attack_access", "metrics_emitted": False,
        "post_release_tuning_or_retest": False,
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    scenarios = config["dataset"]["scenarios"]
    receiver_manifests: dict[str, Any] = {}
    receiver_root = output / "receiver"
    clean_scenario = scenarios[0]
    state["phase"] = "receiver:C-5-clean-support"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    clean_dir = receiver_root / "c-5"
    receiver_manifests["C-5"] = run_receiver(
        clean_scenario, config, clean_dir, frozen["receiver"]
    )
    tow0 = observables_tow0(clean_dir / "raw" / "observables.mat")
    position = clean_static_position(clean_dir / "nmea_pvt.nmea", tow0)
    receiver_ecef = np.asarray(position["ecef"], dtype=np.float64)
    clean_los, clean_ephemeris_report = static_los_map(
        clean_dir / "gps_ephemeris.xml", receiver_ecef
    )
    if len(clean_los) < int(config["analysis"]["minimum_primary_prns"]):
        raise RuntimeError(
            f"C-5 clean support has only {len(clean_los)} usable ephemerides; "
            "attack payloads remain unopened"
        )

    for scenario in scenarios[1:]:
        state["phase"] = f"receiver:{scenario['id']}"
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run_dir = receiver_root / scenario["id"].lower()
        receiver_manifests[scenario["id"]] = run_receiver(
            scenario, config, run_dir, frozen["receiver"]
        )

    state["phase"] = "analysis"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    estimator = build_estimator(json.loads(frozen["template"].read_text(encoding="utf-8")))

    all_delays: list[dict[str, Any]] = []
    all_geometry: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        run_dir = receiver_root / scenario["id"].lower()
        delays, geometry, summary = analyze_scenario(
            scenario, run_dir, estimator, receiver_ecef, config
        )
        all_delays.extend({"scenario": scenario["id"], **row} for row in delays)
        all_geometry.extend(geometry)
        summaries[scenario["id"]] = summary

    clean_scores = [
        row["partial_f_score"] for row in all_geometry
        if row["scenario"] == "C-5" and row["primary_eligible"]
    ]
    for scenario_id in EXPECTED_IDS[1:]:
        attack_scores = [
            row["partial_f_score"] for row in all_geometry
            if row["scenario"] == scenario_id and row["primary_eligible"]
        ]
        summaries[scenario_id]["serial_bin_auc_vs_c5"] = serial_auc(clean_scores, attack_scores)

    decision = terminal_decision(summaries, config)
    delay_path, geometry_path = output / "delay_estimates.csv", output / "geometry_scores.csv"
    write_csv(delay_path, all_delays)
    write_csv(geometry_path, all_geometry)
    summary = {
        "schema": "gnss-doppler-lab.tuni-gps-partial-spoof-external-result.v1",
        "release_commit": release_commit, "decision": decision,
        "scenario_summaries": summaries, "clean_position": position,
        "clean_observables_tow0_s": tow0, "clean_ephemeris": clean_ephemeris_report,
        "receiver_manifests": receiver_manifests,
        "frozen_rules": config["analysis"],
        "claim_boundary": "controlled real static partial-PRN true-position no-multipath RF; attack active from recording start; offline per-PRN ephemeris snapshot LOS",
        "post_release_tuning_or_retest": False,
        "artifacts": {
            "delay_estimates": {"path": str(delay_path), "sha256": file_hash(delay_path), "row_count": len(all_delays)},
            "geometry_scores": {"path": str(geometry_path), "sha256": file_hash(geometry_path), "row_count": len(all_geometry)},
        },
        "config": {"path": str(config_path), "sha256": file_hash(config_path)},
        "protocol": {"path": str(PROTOCOL), "sha256": file_hash(PROTOCOL)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": file_hash(Path(__file__))},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state.update({
        "phase": "complete", "metrics_emitted": True,
        "summary_sha256": file_hash(summary_path),
    })
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **decision}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
