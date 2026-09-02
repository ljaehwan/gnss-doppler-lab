#!/usr/bin/env python3
"""Run a one-geometry receiver-in-loop phase sweep for exact Doppler lock.

The released five-pair campaign remains immutable.  This development-only
experiment reuses its authentic and locked-code components, changes only one
global counterfeit carrier phase, and reuses the frozen frontend realization.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import audit_cgc_locked_phase_root_cause as cause  # noqa: E402
import run_cgc_code_carrier_decoupling_pilot as pilot  # noqa: E402
import run_cgc_rf_challenge_pilot as challenge  # noqa: E402
import run_cgc_rf_geometry_aperture_validation as geometry  # noqa: E402
from gnss_doppler_lab.gnss_sdr import run_receiver  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.rf_impairments import (  # noqa: E402
    CompositeChannelProcessor,
    ImpairmentConfig,
    MultipathTap,
    apply_iq_imbalance,
)
from gnss_doppler_lab.simulation_v4 import SpoofEvent, spoof_power_envelope  # noqa: E402


CAMPAIGN_ROOT = Path("/home/ubuntu/hdd_data/cgc_code_carrier_fresh_static_v1")
RF_OUTPUT_ROOT = Path("/home/ubuntu/hdd_data/cgc_locked_phase_sweep_dev_v1")
ANALYSIS_OUTPUT_ROOT = ROOT / "artifacts/cgc_locked_phase_sweep_dev_v1"
CONFIG = ROOT / "configs/experiments/cgc_code_carrier_fresh_static_v1.json"
PAIR_ID = "ccfs-s1-a-tokyo"


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receiver_from_manifest(document: dict[str, Any]) -> ImpairmentConfig:
    requested = dict(document["simulation_v4"]["receiver"]["requested"])
    requested["multipath"] = tuple(MultipathTap(**item) for item in requested.get("multipath", []))
    return ImpairmentConfig(**requested)


def event_from_config(config: dict[str, Any], pair: dict[str, Any]) -> SpoofEvent:
    item = config["carryoff"]
    return SpoofEvent(
        start_seconds=float(item["start_seconds"]),
        transition_seconds=float(item["transition_seconds"]),
        target_offset_enu_m=tuple(float(value) for value in pair["target_offset_enu_m"]),
        initial_advantage_db=float(item["initial_advantage_db"]),
        final_advantage_db=float(item["final_advantage_db"]),
        power_ramp_seconds=float(item["power_ramp_seconds"]),
    )


def compose_phase_shifted_locked_iq(
    authentic_path: Path,
    counterfeit_path: Path,
    destination: Path,
    *,
    phase_offset_rad: float,
    sample_rate_hz: int,
    receiver: ImpairmentConfig,
    reference: dict[str, float],
    event: SpoofEvent,
) -> dict[str, Any]:
    """Mirror the frozen composer while rotating only the counterfeit source."""
    if authentic_path.stat().st_size != counterfeit_path.stat().st_size:
        raise ValueError("component sizes differ")
    if destination.exists():
        raise FileExistsError(destination)
    phase = float(phase_offset_rad)
    if not math.isfinite(phase):
        raise ValueError("phase_offset_rad must be finite")
    expected_samples = authentic_path.stat().st_size // 2
    if int(reference["complex_samples"]) != expected_samples:
        raise ValueError("composition reference sample count differs")

    fixed_gain = np.float32(reference["fixed_receiver_gain"])
    noise_std = np.float32(math.sqrt(reference["frontend_output_awgn_complex_variance"] / 2.0))
    phasor = np.complex64(np.exp(1j * phase))
    channel = CompositeChannelProcessor(sample_rate_hz, receiver)
    noise_rng = np.random.default_rng(np.random.SeedSequence(receiver.seed).spawn(2)[1])
    dc = np.complex64(receiver.dc_i + 1j * receiver.dc_q)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    complex_samples = 0
    output_power_sum = 0.0
    signal_power_sum = 0.0
    clipped_complex_samples = 0
    clipped_components = 0
    peak_magnitude = 0.0
    sample_index = 0
    try:
        with os.fdopen(fd, "wb") as output, authentic_path.open("rb") as authentic_stream, counterfeit_path.open("rb") as counterfeit_stream:
            while True:
                raw_authentic = authentic_stream.read(receiver.chunk_samples * 2)
                raw_counterfeit = counterfeit_stream.read(receiver.chunk_samples * 2)
                if not raw_authentic and not raw_counterfeit:
                    break
                if len(raw_authentic) != len(raw_counterfeit) or len(raw_authentic) % 2:
                    raise ValueError("component IQ became misaligned")
                a = np.frombuffer(raw_authentic, dtype=np.int8).astype(np.float32)
                c = np.frombuffer(raw_counterfeit, dtype=np.int8).astype(np.float32)
                authentic = (a[0::2] + 1j * a[1::2]).astype(np.complex64)
                counterfeit = (c[0::2] + 1j * c[1::2]).astype(np.complex64) * phasor
                envelope = spoof_power_envelope(
                    authentic.size, sample_rate_hz, event, start_sample=sample_index
                )
                source = authentic + counterfeit * envelope
                signal = channel.process_complex(source)
                draws = noise_rng.standard_normal(authentic.size * 2).reshape(-1, 2).astype(np.float32)
                noise = (noise_std * (draws[:, 0] + 1j * draws[:, 1])).astype(np.complex64)
                post = (apply_iq_imbalance(signal, receiver) + apply_iq_imbalance(noise, receiver) + dc) * fixed_gain
                over_i = np.abs(post.real) > receiver.clip_level
                over_q = np.abs(post.imag) > receiver.clip_level
                clipped_complex_samples += int(np.count_nonzero(over_i | over_q))
                clipped_components += int(np.count_nonzero(over_i) + np.count_nonzero(over_q))
                i = np.rint(np.clip(post.real, -receiver.clip_level, receiver.clip_level)).astype(np.int8)
                q = np.rint(np.clip(post.imag, -receiver.clip_level, receiver.clip_level)).astype(np.int8)
                interleaved = np.empty(signal.size * 2, dtype=np.int8)
                interleaved[0::2], interleaved[1::2] = i, q
                payload = interleaved.tobytes()
                output.write(payload)
                digest.update(payload)
                complex_samples += signal.size
                signal_power_sum += float(np.vdot(signal, signal).real)
                output_power_sum += float(np.sum(i.astype(np.float32) ** 2 + q.astype(np.float32) ** 2))
                if signal.size:
                    peak_magnitude = max(peak_magnitude, float(np.max(np.hypot(i.astype(np.float32), q.astype(np.float32)))))
                sample_index += authentic.size
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
    return {
        "path": destination.name,
        "sha256": digest.hexdigest(),
        "bytes": destination.stat().st_size,
        "complex_samples": complex_samples,
        "actual_duration_seconds": complex_samples / sample_rate_hz,
        "mean_quantized_complex_power": output_power_sum / complex_samples,
        "mean_analog_signal_power": signal_power_sum / complex_samples,
        "peak_quantized_magnitude": peak_magnitude,
        "clipped_complex_samples": clipped_complex_samples,
        "clipped_components": clipped_components,
        "clipping_fraction": clipped_complex_samples / complex_samples,
    }


def phase_tag(degrees: float) -> str:
    normalized = degrees % 360.0
    return f"phase-{normalized:06.1f}deg".replace(".", "p")


def ensure_phase_receiver(
    phase_deg: float, config: dict[str, Any], pair: dict[str, Any]
) -> Path:
    pair_root = CAMPAIGN_ROOT / "pairs" / PAIR_ID
    if math.isclose(phase_deg % 360.0, 0.0, abs_tol=1e-12):
        return pair_root / "receiver" / f"cgc-cc-fresh-{PAIR_ID}-doppler-locked"

    frozen_manifest_path = pair_root / "rf/doppler-locked/manifest.json"
    frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    tag = phase_tag(phase_deg)
    run_id = f"cgc-locked-phase-dev-{PAIR_ID}-{tag}"
    rf_root = RF_OUTPUT_ROOT / "rf" / tag
    iq_path = rf_root / "gps_l1ca_s8_iq.bin"
    manifest_path = rf_root / "manifest.json"
    receiver_root = RF_OUTPUT_ROOT / "receiver"
    receiver_manifest = receiver_root / run_id / "manifest.json"
    if receiver_manifest.is_file():
        return receiver_manifest.parent

    if not manifest_path.is_file():
        receiver = receiver_from_manifest(frozen)
        reference = frozen["simulation_v4"]["receiver"]["reference"]
        report = compose_phase_shifted_locked_iq(
            pair_root / "components/authentic/gps_l1ca_s8_iq.bin",
            pair_root / "components/doppler-locked/gps_l1ca_s8_iq.bin",
            iq_path,
            phase_offset_rad=math.radians(phase_deg),
            sample_rate_hz=int(config["rf"]["sample_rate_hz"]),
            receiver=receiver,
            reference=reference,
            event=event_from_config(config, pair),
        )
        write_json(
            manifest_path,
            {
                "schema_version": 4,
                "run_id": run_id,
                "scenario": {
                    "name": tag, "campaign": "cgc-locked-phase-sweep-dev-v1",
                    "class": "spoofing", "event": "carryoff", "is_spoofing": True,
                    "domain": "static", **pair,
                    "duration_seconds": config["carryoff"]["duration_seconds"],
                },
                "iq": {
                    "path": iq_path.name, "sha256": report["sha256"],
                    "actual_bytes": report["bytes"], "complex_samples": report["complex_samples"],
                    "actual_duration_seconds": report["actual_duration_seconds"],
                    "rf_sample_rate_hz": int(config["rf"]["sample_rate_hz"]),
                    "sample_format": "s8_iq", "channels": 2,
                },
                "phase_sweep": {
                    "counterfeit_global_phase_offset_deg": phase_deg,
                    "counterfeit_global_phase_offset_rad": math.radians(phase_deg),
                    "frozen_source_manifest": str(frozen_manifest_path),
                    "frozen_source_manifest_sha256": sha256(frozen_manifest_path),
                    "frontend_reference": reference,
                    "frontend_requested": receiver.manifest(),
                    "measurement": report,
                    "scope": "development-only single-geometry causal phase intervention",
                },
            },
        )
    receiver_config = config["tools"]["receiver"]
    result = run_receiver(
        manifest_path,
        receiver_root,
        executable=ROOT / receiver_config["path"],
        channel_count=int(receiver_config["channel_count"]),
        timeout_seconds=int(receiver_config["timeout_seconds"]),
        tracking_tap_count=9,
        tracking_tap_spacing_chips=float(receiver_config["tracking_tap_spacing_chips"]),
    )
    return result.parent


def truth_agreement_rows(
    delay_rows: list[dict[str, Any]], pair_root: Path
) -> list[dict[str, float]]:
    authentic = cause.truth_by_time_prn(pair_root / "components/authentic/truth.csv")
    spoof = cause.truth_by_time_prn(pair_root / "components/doppler-locked/truth.csv")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in delay_rows:
        grouped.setdefault(int(row["bin_index"]), []).append(row)
    output: list[dict[str, float]] = []
    for bin_index, rows in sorted(grouped.items()):
        if bin_index < cause.HOLD_START_BIN:
            continue
        time_s = round(bin_index + 0.5, 1)
        estimates, truth = [], []
        for row in rows:
            key = (time_s, str(row["prn"]))
            if key not in authentic or key not in spoof:
                continue
            estimates.append(float(row["estimated_delay_chips"]))
            truth.append(
                (spoof[key]["code_range_m"] - authentic[key]["code_range_m"])
                / cause.CHIP_LENGTH_M
            )
        output.append({"bin_index": float(bin_index), **cause.centered_truth_agreement(np.asarray(estimates), np.asarray(truth))})
    return output


def receiver_phase_metrics(run_dir: Path) -> dict[str, float]:
    from gnss_doppler_lab.tracking_peaks import available_tracking_prns

    quadrature, cn0 = [], []
    for prn in available_tracking_prns(run_dir):
        for segment in load_segments(run_dir, prn):
            selected = segment.time_s >= cause.HOLD_START_BIN
            if not np.any(selected):
                continue
            quadrature.extend(cause.quadrature_fraction(segment.complex_taps[selected]).tolist())
            cn0.extend(segment.cn0_db_hz[selected].tolist())
    return {
        "median_quadrature_fraction": float(np.median(quadrature)),
        "median_cn0_db_hz": float(np.median(cn0)),
    }


def load_segments(run_dir: Path, prn: str) -> list[Any]:
    from gnss_doppler_lab.tracking_peaks import load_receiver_tracking_peak_series_segments

    return load_receiver_tracking_peak_series_segments(
        run_dir, prn, epoch_step=20, tap_count=9, require_complex_taps=True
    )


def analyze_phase(
    phase_deg: float,
    run_dir: Path,
    config: dict[str, Any],
    estimator: Any,
    los: dict[str, tuple[float, float, float]],
) -> dict[str, Any]:
    delays, geometry_rows = geometry.analyze_stream(
        phase_tag(phase_deg), run_dir / "manifest.json", estimator, los, config, 9
    )
    scored = pilot.score_rows(geometry_rows)
    _, annotated = pilot.persistence(scored, float(config["analysis"]["partial_f_p_alarm_threshold"]))
    hold = [row for row in annotated if float(row["bin_start_s"]) >= cause.HOLD_START_BIN]
    first = next(
        (float(row["bin_start_s"]) for row in annotated if float(row["bin_start_s"]) >= float(config["carryoff"]["start_seconds"]) and row["persistent_spoof_alarm"]),
        None,
    )
    agreement = truth_agreement_rows(delays, CAMPAIGN_ROOT / "pairs" / PAIR_ID)
    return {
        "phase_offset_deg": phase_deg,
        **receiver_phase_metrics(run_dir),
        "median_truth_direction_r2": float(np.median([row["truth_direction_r2"] for row in agreement])),
        "median_partial_f_p_value": float(np.median([row["partial_f_p_value"] for row in hold])),
        "hold_raw_alarm_rate": float(np.mean([row["raw_spoof_alarm"] for row in hold])),
        "hold_persistent_alarm_rate": float(np.mean([row["persistent_spoof_alarm"] for row in hold])),
        "first_persistent_alarm_s": first,
        "latency_from_onset_s": None if first is None else first - float(config["carryoff"]["start_seconds"]),
        "receiver_manifest": str((run_dir / "manifest.json").resolve()),
        "receiver_manifest_sha256": sha256(run_dir / "manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases-deg", type=float, nargs="+", default=[0.0, 90.0])
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pair = next(item for item in config["pairs"] if item["candidate_id"] == PAIR_ID)
    controlled = json.loads((ROOT / config["inputs"]["controlled_template"]["path"]).read_text(encoding="utf-8"))
    estimator = challenge._estimator(controlled)
    pair_root = CAMPAIGN_ROOT / "pairs" / PAIR_ID
    los = parse_gps_sdr_sim_los_table((pair_root / "components/authentic/simulator.log").read_text(encoding="utf-8"))
    rows = []
    for phase_deg in args.phases_deg:
        print(f"[phase] {phase_deg:g} deg", flush=True)
        run_dir = ensure_phase_receiver(phase_deg, config, pair)
        rows.append(analyze_phase(phase_deg, run_dir, config, estimator, los))
        write_json(
            ANALYSIS_OUTPUT_ROOT / "summary.json",
            {
                "schema": "gnss-doppler-lab.cgc-locked-phase-sweep-development",
                "schema_version": 1,
                "pair_id": PAIR_ID,
                "role": "development-only causal phase intervention; not a released validation set",
                "rows": rows,
                "source_phase_diagnosis": str((ROOT / "artifacts/cgc_locked_phase_root_cause_v1/summary.json").resolve()),
                "claim_boundary": (
                    "One global phase offset on one reused geometry isolates carrier phase but does not establish "
                    "performance over independent per-PRN phase, other geometries, or field spoofers."
                ),
            },
        )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
