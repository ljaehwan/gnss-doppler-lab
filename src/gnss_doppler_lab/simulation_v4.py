"""Paired offline GPS L1 C/A simulation for normal recovery and spoofing studies.

The composer is intentionally file based and never transmits RF.  Every scenario
uses the same authentic GPS-SDR-SIM source, common receiver realization, and
noise samples.  Receiver gain is calibrated once from authentic-only reference
power, then frozen across scenarios; future outage/spoof parameters cannot leak
into a pre-event prefix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import os
import re
import tempfile

import numpy as np
import yaml

from .gps_sdr_sim import GpsSdrSimRunner, simulator_time
from .rf_config import (
    InputConfig,
    OutputConfig,
    RFGenerationConfig,
    Scenario,
    SimulatorConfig,
    StaticPosition,
    TrajectoryPosition,
)
from .rf_impairments import (
    CN0_CAVEAT,
    CompositeChannelProcessor,
    ImpairmentConfig,
    apply_iq_imbalance,
)
from .trajectory import enu_to_llh, llh_to_enu


SCHEMA = "gnss-doppler-lab.simulation-v4"
SCHEMA_VERSION = 1


class SimulationV4Error(ValueError):
    """Invalid or inconsistent paired simulation campaign."""


@dataclass(frozen=True)
class OutageEvent:
    start_seconds: float
    end_seconds: float
    attenuation_db: float
    recovery_ramp_seconds: float


@dataclass(frozen=True)
class SpoofEvent:
    start_seconds: float
    transition_seconds: float
    target_offset_enu_m: tuple[float, float, float]
    initial_advantage_db: float
    final_advantage_db: float
    power_ramp_seconds: float


@dataclass(frozen=True)
class SimulationScenario:
    name: str
    kind: str
    outage: OutageEvent | None = None
    spoofing: SpoofEvent | None = None


@dataclass(frozen=True)
class SimulationCampaign:
    version: int
    name: str
    base_rf_config: RFGenerationConfig
    output_root: Path
    receiver: ImpairmentConfig
    normal_target_rms: float
    keep_component_iq: bool
    scenarios: tuple[SimulationScenario, ...]
    source_config_path: Path


@dataclass
class _OutputState:
    scenario: SimulationScenario
    path: Path
    temporary: Path
    stream: Any
    channel: CompositeChannelProcessor
    digest: Any
    complex_samples: int = 0
    signal_power_sum: float = 0.0
    output_power_sum: float = 0.0
    clipped_complex_samples: int = 0
    clipped_components: int = 0
    peak_magnitude: float = 0.0


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _mapping(value: Any, name: str, allowed: set[str], required: set[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SimulationV4Error(f"{name} must be a mapping")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise SimulationV4Error(f"unknown {name} key(s): {', '.join(sorted(unknown))}")
    if missing:
        raise SimulationV4Error(f"missing {name} key(s): {', '.join(sorted(missing))}")
    return value


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SimulationV4Error(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise SimulationV4Error(f"{name} must be finite")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SimulationV4Error(f"{name} must be an integer")
    return value


def _safe_name(value: Any, name: str) -> str:
    result = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", result):
        raise SimulationV4Error(f"{name} must be a safe identifier")
    return result


def _resolve(config_path: Path, value: Any) -> Path:
    path = Path(str(value))
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _parse_receiver(raw: Any, sample_rate_hz: int) -> tuple[ImpairmentConfig, float]:
    allowed = {
        "seed", "sample_snr_db", "normal_target_rms", "carrier_offset_hz",
        "frequency_drift_hz_per_s", "phase_noise_std_rad_per_sqrt_sample",
        "frontend_cutoff_hz", "frontend_order", "iq_gain_imbalance_db",
        "iq_phase_imbalance_deg", "dc_i", "dc_q", "clip_level", "chunk_samples",
    }
    doc = _mapping(raw, "receiver", allowed, {"seed", "sample_snr_db", "normal_target_rms"})
    seed = _integer(doc["seed"], "receiver.seed")
    if not 0 <= seed < 2**64:
        raise SimulationV4Error("receiver.seed must be in [0, 2^64)")
    target = _finite(doc["normal_target_rms"], "receiver.normal_target_rms")
    if not 1 <= target <= 60:
        raise SimulationV4Error("receiver.normal_target_rms must be in [1, 60]")
    frontend = doc.get("frontend_cutoff_hz")
    frontend_value = None if frontend is None else _finite(frontend, "receiver.frontend_cutoff_hz")
    if frontend_value is not None and frontend_value >= sample_rate_hz / 2:
        raise SimulationV4Error("receiver.frontend_cutoff_hz must be below Nyquist")
    try:
        config = ImpairmentConfig(
            enabled=True,
            profile="explicit",
            seed=seed,
            sample_snr_db=_finite(doc["sample_snr_db"], "receiver.sample_snr_db"),
            carrier_offset_hz=_finite(doc.get("carrier_offset_hz", 0), "receiver.carrier_offset_hz"),
            frequency_drift_hz_per_s=_finite(doc.get("frequency_drift_hz_per_s", 0), "receiver.frequency_drift_hz_per_s"),
            phase_noise_std_rad_per_sqrt_sample=_finite(doc.get("phase_noise_std_rad_per_sqrt_sample", 0), "receiver.phase_noise_std_rad_per_sqrt_sample"),
            frontend_cutoff_hz=frontend_value,
            frontend_order=_integer(doc.get("frontend_order", 4), "receiver.frontend_order"),
            iq_gain_imbalance_db=_finite(doc.get("iq_gain_imbalance_db", 0), "receiver.iq_gain_imbalance_db"),
            iq_phase_imbalance_deg=_finite(doc.get("iq_phase_imbalance_deg", 0), "receiver.iq_phase_imbalance_deg"),
            dc_i=_finite(doc.get("dc_i", 0), "receiver.dc_i"),
            dc_q=_finite(doc.get("dc_q", 0), "receiver.dc_q"),
            gain=1.0,
            agc_target_rms=None,
            clip_level=_finite(doc.get("clip_level", 127), "receiver.clip_level"),
            chunk_samples=_integer(doc.get("chunk_samples", 1_048_576), "receiver.chunk_samples"),
        )
    except ValueError as exc:
        raise SimulationV4Error(f"invalid receiver configuration: {exc}") from exc
    return config, target


def _parse_scenarios(raw: Any, duration_seconds: int) -> tuple[SimulationScenario, ...]:
    if not isinstance(raw, list) or not raw:
        raise SimulationV4Error("scenarios must be a non-empty list")
    scenarios: list[SimulationScenario] = []
    for index, item in enumerate(raw):
        doc = _mapping(item, f"scenarios[{index}]", {"name", "kind", "outage", "spoofing"}, {"name", "kind"})
        name = _safe_name(doc["name"], f"scenarios[{index}].name")
        kind = str(doc["kind"])
        if kind not in {"steady_normal", "recovery_normal", "carryoff_spoof"}:
            raise SimulationV4Error(f"unsupported scenario kind: {kind}")
        outage = spoofing = None
        if kind == "recovery_normal":
            event = _mapping(doc.get("outage"), f"scenarios[{index}].outage", {
                "start_seconds", "end_seconds", "attenuation_db", "recovery_ramp_seconds",
            }, {"start_seconds", "end_seconds", "attenuation_db", "recovery_ramp_seconds"})
            outage = OutageEvent(*(
                _finite(event[key], f"scenarios[{index}].outage.{key}")
                for key in ("start_seconds", "end_seconds", "attenuation_db", "recovery_ramp_seconds")
            ))
            if not 0 < outage.start_seconds < outage.end_seconds < duration_seconds:
                raise SimulationV4Error("outage must occur strictly inside the scenario duration")
            if not -120 <= outage.attenuation_db <= 0:
                raise SimulationV4Error("outage.attenuation_db must be in [-120, 0]")
            if outage.recovery_ramp_seconds < 0 or outage.end_seconds + outage.recovery_ramp_seconds >= duration_seconds:
                raise SimulationV4Error("outage recovery must finish inside the scenario duration")
        elif kind == "carryoff_spoof":
            event = _mapping(doc.get("spoofing"), f"scenarios[{index}].spoofing", {
                "start_seconds", "transition_seconds", "target_offset_enu_m",
                "initial_advantage_db", "final_advantage_db", "power_ramp_seconds",
            }, {
                "start_seconds", "transition_seconds", "target_offset_enu_m",
                "initial_advantage_db", "final_advantage_db", "power_ramp_seconds",
            })
            offsets = event["target_offset_enu_m"]
            if not isinstance(offsets, list) or len(offsets) != 3:
                raise SimulationV4Error("target_offset_enu_m must contain [east, north, up]")
            spoofing = SpoofEvent(
                start_seconds=_finite(event["start_seconds"], "spoofing.start_seconds"),
                transition_seconds=_finite(event["transition_seconds"], "spoofing.transition_seconds"),
                target_offset_enu_m=tuple(_finite(x, "spoofing.target_offset_enu_m") for x in offsets),
                initial_advantage_db=_finite(event["initial_advantage_db"], "spoofing.initial_advantage_db"),
                final_advantage_db=_finite(event["final_advantage_db"], "spoofing.final_advantage_db"),
                power_ramp_seconds=_finite(event["power_ramp_seconds"], "spoofing.power_ramp_seconds"),
            )
            if spoofing.target_offset_enu_m == (0.0, 0.0, 0.0):
                raise SimulationV4Error("target_offset_enu_m must be non-zero")
            if not 0 < spoofing.start_seconds < duration_seconds:
                raise SimulationV4Error("spoofing.start_seconds must occur inside the scenario")
            if spoofing.transition_seconds <= 0 or spoofing.start_seconds + spoofing.transition_seconds >= duration_seconds:
                raise SimulationV4Error("spoofing transition must finish inside the scenario")
            if spoofing.power_ramp_seconds < 0 or spoofing.start_seconds + spoofing.power_ramp_seconds >= duration_seconds:
                raise SimulationV4Error("spoofing power ramp must finish inside the scenario")
            if not all(-40 <= value <= 6 for value in (spoofing.initial_advantage_db, spoofing.final_advantage_db)):
                raise SimulationV4Error("spoofing power advantages must be in [-40, 6] dB")
        if kind == "steady_normal" and ("outage" in doc or "spoofing" in doc):
            raise SimulationV4Error("steady_normal must not define an event")
        scenarios.append(SimulationScenario(name, kind, outage, spoofing))
    names = [item.name for item in scenarios]
    if len(names) != len(set(names)):
        raise SimulationV4Error("scenario names must be unique")
    kinds = {item.kind for item in scenarios}
    required = {"steady_normal", "recovery_normal", "carryoff_spoof"}
    if not required.issubset(kinds):
        raise SimulationV4Error("campaign requires steady_normal, recovery_normal, and carryoff_spoof")
    return tuple(scenarios)


def load_simulation_campaign(path: str | Path) -> SimulationCampaign:
    """Load the strict paired-simulation YAML contract."""
    config_path = Path(path).resolve()
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SimulationV4Error(str(exc)) from exc
    root = _mapping(data, "configuration", {
        "version", "campaign", "input", "output", "receiver", "scenarios", "simulator",
    }, {"version", "campaign", "input", "output", "receiver", "scenarios"})
    if root["version"] != 1:
        raise SimulationV4Error("only simulation-v4 config version 1 is supported")
    campaign = _mapping(root["campaign"], "campaign", {
        "name", "utc", "duration_seconds", "position",
    }, {"name", "utc", "duration_seconds", "position"})
    name = _safe_name(campaign["name"], "campaign.name")
    try:
        utc = datetime.fromisoformat(str(campaign["utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SimulationV4Error("campaign.utc must be ISO-8601") from exc
    if utc.tzinfo is None or utc.microsecond:
        raise SimulationV4Error("campaign.utc must be timezone-aware with integer seconds")
    utc = utc.astimezone(timezone.utc)
    duration = _integer(campaign["duration_seconds"], "campaign.duration_seconds")
    if not 5 <= duration <= 300:
        raise SimulationV4Error("campaign.duration_seconds must be in [5, 300]")
    position_doc = _mapping(campaign["position"], "campaign.position", {
        "type", "latitude_deg", "longitude_deg", "altitude_m",
    }, {"type", "latitude_deg", "longitude_deg", "altitude_m"})
    if position_doc["type"] != "static":
        raise SimulationV4Error("simulation-v4 currently requires a static authentic position")
    position = StaticPosition(
        _finite(position_doc["latitude_deg"], "position.latitude_deg"),
        _finite(position_doc["longitude_deg"], "position.longitude_deg"),
        _finite(position_doc["altitude_m"], "position.altitude_m"),
    )
    if not -90 <= position.latitude_deg <= 90 or not -180 <= position.longitude_deg <= 180:
        raise SimulationV4Error("invalid static coordinates")
    input_doc = _mapping(root["input"], "input", {"rinex_nav"}, {"rinex_nav"})
    output_doc = _mapping(root["output"], "output", {
        "root", "rf_sample_rate_hz", "sample_format", "keep_component_iq",
    }, {"root", "rf_sample_rate_hz"})
    sample_rate = _integer(output_doc["rf_sample_rate_hz"], "output.rf_sample_rate_hz")
    if sample_rate < 1_000_000 or output_doc.get("sample_format", "s8_iq") != "s8_iq":
        raise SimulationV4Error("output requires >=1 MHz signed interleaved IQ8")
    keep_components = output_doc.get("keep_component_iq", False)
    if not isinstance(keep_components, bool):
        raise SimulationV4Error("output.keep_component_iq must be boolean")
    simulator_doc = _mapping(root.get("simulator", {}), "simulator", {"executable"})
    executable = simulator_doc.get("executable")
    resolved_executable = str(_resolve(config_path, executable)) if executable else None
    nav = _resolve(config_path, input_doc["rinex_nav"])
    output_root = _resolve(config_path, output_doc["root"])
    receiver, target = _parse_receiver(root["receiver"], sample_rate)
    scenarios = _parse_scenarios(root["scenarios"], duration)
    base = RFGenerationConfig(
        version=1,
        scenario=Scenario(name, "GPS", "L1CA", utc, duration, position),
        input=InputConfig(nav),
        output=OutputConfig(output_root, sample_rate, "s8_iq"),
        simulator=SimulatorConfig(resolved_executable),
        impairments=replace(receiver, enabled=False, profile="clean"),
    )
    return SimulationCampaign(1, name, base, output_root, receiver, target, keep_components, scenarios, config_path)


def outage_envelope(
    sample_count: int,
    sample_rate_hz: int,
    event: OutageEvent,
    *,
    start_sample: int = 0,
) -> np.ndarray:
    """Authentic amplitude envelope for a normal loss/restoration event."""
    times = (start_sample + np.arange(sample_count, dtype=np.float64)) / sample_rate_hz
    minimum = 10.0 ** (event.attenuation_db / 20.0)
    result = np.ones(sample_count, dtype=np.float32)
    lost = (times >= event.start_seconds) & (times < event.end_seconds)
    result[lost] = minimum
    if event.recovery_ramp_seconds > 0:
        recovering = (times >= event.end_seconds) & (times < event.end_seconds + event.recovery_ramp_seconds)
        x = (times[recovering] - event.end_seconds) / event.recovery_ramp_seconds
        smooth = x * x * (3.0 - 2.0 * x)
        result[recovering] = minimum + (1.0 - minimum) * smooth
    return result


def spoof_power_envelope(
    sample_count: int,
    sample_rate_hz: int,
    event: SpoofEvent,
    *,
    start_sample: int = 0,
) -> np.ndarray:
    """Counterfeit/authentic amplitude ratio, exactly zero before onset."""
    times = (start_sample + np.arange(sample_count, dtype=np.float64)) / sample_rate_hz
    result = np.zeros(sample_count, dtype=np.float32)
    active = times >= event.start_seconds
    initial = 10.0 ** (event.initial_advantage_db / 20.0)
    final = 10.0 ** (event.final_advantage_db / 20.0)
    if event.power_ramp_seconds == 0:
        result[active] = final
    else:
        fraction = np.clip((times[active] - event.start_seconds) / event.power_ramp_seconds, 0.0, 1.0)
        result[active] = initial + fraction * (final - initial)
    return result


def build_carryoff_rows(
    authentic_rows: tuple[tuple[float, float, float, float], ...],
    event: SpoofEvent,
) -> tuple[tuple[float, float, float, float], ...]:
    """Build a C1-smooth false LLH path from an authentic 10 Hz truth path."""
    result = []
    for time_s, latitude, longitude, altitude in authentic_rows:
        if time_s <= event.start_seconds:
            result.append((time_s, latitude, longitude, altitude))
            continue
        x = min(1.0, (time_s - event.start_seconds) / event.transition_seconds)
        progress = x * x * (3.0 - 2.0 * x)
        east, north, up = event.target_offset_enu_m
        false_llh = enu_to_llh(
            east * progress, north * progress, up * progress,
            latitude, longitude, altitude,
        )
        result.append((time_s, *false_llh))
    return tuple(result)


def _write_carryoff_trajectory(path: Path, rows, event: SpoofEvent) -> dict[str, Any]:
    csv_bytes = "".join(
        f"{time_s:.1f},{latitude:.9f},{longitude:.9f},{altitude:.4f}\n"
        for time_s, latitude, longitude, altitude in rows
    ).encode("ascii")
    _atomic_write(path, csv_bytes)
    digest = hashlib.sha256(csv_bytes).hexdigest()
    sidecar = path.with_suffix(".json")
    final_e, final_n, final_u = llh_to_enu(*rows[-1][1:], *rows[0][1:])
    metadata = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "sample_rate_hz": 10,
        "row_count": len(rows),
        "csv_sha256": digest,
        "attack": asdict(event),
        "realized_final_offset_enu_m": [final_e, final_n, final_u],
    }
    _atomic_write(sidecar, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    return {"csv_sha256": digest, "sidecar": sidecar, "sidecar_sha256": _sha256(sidecar), **metadata}


def _measure_authentic_reference(path: Path, sample_rate_hz: int, config: ImpairmentConfig, target_rms: float) -> dict[str, float]:
    channel = CompositeChannelProcessor(sample_rate_hz, config)
    count = 0
    channel_power_sum = 0.0
    noiseless_rx_power_sum = 0.0
    dc = np.complex64(config.dc_i + 1j * config.dc_q)
    with path.open("rb") as stream:
        while raw := stream.read(config.chunk_samples * 2):
            values = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
            signal = channel.process_complex(values[0::2] + 1j * values[1::2])
            channel_power_sum += float(np.vdot(signal, signal).real)
            noiseless = apply_iq_imbalance(signal, config) + dc
            noiseless_rx_power_sum += float(np.vdot(noiseless, noiseless).real)
            count += signal.size
    if not count:
        raise SimulationV4Error("authentic component IQ is empty")
    channel_power = channel_power_sum / count
    noise_power = channel_power / 10.0 ** (float(config.sample_snr_db) / 10.0)
    gi = 10.0 ** (config.iq_gain_imbalance_db / 40.0)
    gq = 1.0 / gi
    transformed_noise_power = noise_power * (gi * gi + gq * gq) / 2.0
    expected_normal_power = noiseless_rx_power_sum / count + transformed_noise_power
    fixed_gain = target_rms / math.sqrt(expected_normal_power)
    return {
        "complex_samples": count,
        "authentic_channel_mean_complex_power": channel_power,
        "frontend_output_awgn_complex_variance": noise_power,
        "expected_normal_pre_gain_mean_complex_power": expected_normal_power,
        "fixed_receiver_gain": fixed_gain,
        "normal_target_rms": target_rms,
    }


def _event_signal(scenario: SimulationScenario, authentic: np.ndarray, counterfeit: np.ndarray, sample_rate_hz: int, start_sample: int) -> np.ndarray:
    if scenario.kind == "steady_normal":
        return authentic
    if scenario.kind == "recovery_normal":
        assert scenario.outage is not None
        return authentic * outage_envelope(authentic.size, sample_rate_hz, scenario.outage, start_sample=start_sample)
    assert scenario.spoofing is not None
    envelope = spoof_power_envelope(authentic.size, sample_rate_hz, scenario.spoofing, start_sample=start_sample)
    return authentic + counterfeit * envelope


def compose_paired_iq(
    authentic_path: str | Path,
    counterfeit_path: str | Path,
    destinations: dict[str, Path],
    scenarios: tuple[SimulationScenario, ...],
    *,
    sample_rate_hz: int,
    receiver: ImpairmentConfig,
    normal_target_rms: float,
    reference_override: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compose all scenarios in one pass with shared noise and frozen gain."""
    authentic_path, counterfeit_path = Path(authentic_path), Path(counterfeit_path)
    if authentic_path.stat().st_size != counterfeit_path.stat().st_size:
        raise SimulationV4Error("authentic and counterfeit component IQ sizes differ")
    if authentic_path.stat().st_size == 0 or authentic_path.stat().st_size % 2:
        raise SimulationV4Error("component IQ must be non-empty signed interleaved IQ8")
    if set(destinations) != {item.name for item in scenarios}:
        raise SimulationV4Error("destination names must match scenario names")
    if reference_override is None:
        reference = _measure_authentic_reference(
            authentic_path, sample_rate_hz, receiver, normal_target_rms
        )
    else:
        required_reference = {
            "complex_samples", "authentic_channel_mean_complex_power",
            "frontend_output_awgn_complex_variance",
            "expected_normal_pre_gain_mean_complex_power", "fixed_receiver_gain",
            "normal_target_rms",
        }
        if set(reference_override) != required_reference or not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in reference_override.values()
        ):
            raise SimulationV4Error("reference_override is incomplete or non-positive")
        reference = {key: float(value) for key, value in reference_override.items()}
        expected_samples = authentic_path.stat().st_size // 2
        if (
            not reference["complex_samples"].is_integer()
            or int(reference["complex_samples"]) != expected_samples
        ):
            raise SimulationV4Error(
                "reference_override complex_samples differs from component IQ"
            )
        if not math.isclose(
            reference["normal_target_rms"], float(normal_target_rms),
            rel_tol=1e-12, abs_tol=0.0,
        ):
            raise SimulationV4Error("reference_override normal_target_rms drifted")
        expected_gain = normal_target_rms / math.sqrt(
            reference["expected_normal_pre_gain_mean_complex_power"]
        )
        if not math.isclose(
            reference["fixed_receiver_gain"], expected_gain,
            rel_tol=1e-12, abs_tol=0.0,
        ):
            raise SimulationV4Error("reference_override fixed_receiver_gain is inconsistent")
    fixed_gain = np.float32(reference["fixed_receiver_gain"])
    noise_std = np.float32(math.sqrt(reference["frontend_output_awgn_complex_variance"] / 2.0))
    states: list[_OutputState] = []
    noise_rng = np.random.default_rng(np.random.SeedSequence(receiver.seed).spawn(2)[1])
    try:
        for scenario in scenarios:
            destination = destinations[scenario.name]
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
            states.append(_OutputState(
                scenario, destination, Path(name), os.fdopen(fd, "wb"),
                CompositeChannelProcessor(sample_rate_hz, receiver), hashlib.sha256(),
            ))
        sample_index = 0
        with authentic_path.open("rb") as authentic_stream, counterfeit_path.open("rb") as counterfeit_stream:
            while True:
                raw_authentic = authentic_stream.read(receiver.chunk_samples * 2)
                raw_counterfeit = counterfeit_stream.read(receiver.chunk_samples * 2)
                if not raw_authentic and not raw_counterfeit:
                    break
                if len(raw_authentic) != len(raw_counterfeit) or len(raw_authentic) % 2:
                    raise SimulationV4Error("component IQ changed or became misaligned during composition")
                a = np.frombuffer(raw_authentic, dtype=np.int8).astype(np.float32)
                c = np.frombuffer(raw_counterfeit, dtype=np.int8).astype(np.float32)
                authentic = (a[0::2] + 1j * a[1::2]).astype(np.complex64)
                counterfeit = (c[0::2] + 1j * c[1::2]).astype(np.complex64)
                draws = noise_rng.standard_normal(authentic.size * 2).reshape(-1, 2).astype(np.float32)
                noise = (noise_std * (draws[:, 0] + 1j * draws[:, 1])).astype(np.complex64)
                rx_noise = apply_iq_imbalance(noise, receiver)
                dc = np.complex64(receiver.dc_i + 1j * receiver.dc_q)
                for state in states:
                    source = _event_signal(state.scenario, authentic, counterfeit, sample_rate_hz, sample_index)
                    signal = state.channel.process_complex(source)
                    rx_signal = apply_iq_imbalance(signal, receiver)
                    post = (rx_signal + rx_noise + dc) * fixed_gain
                    over_i = np.abs(post.real) > receiver.clip_level
                    over_q = np.abs(post.imag) > receiver.clip_level
                    state.clipped_complex_samples += int(np.count_nonzero(over_i | over_q))
                    state.clipped_components += int(np.count_nonzero(over_i) + np.count_nonzero(over_q))
                    i = np.rint(np.clip(post.real, -receiver.clip_level, receiver.clip_level)).astype(np.int8)
                    q = np.rint(np.clip(post.imag, -receiver.clip_level, receiver.clip_level)).astype(np.int8)
                    interleaved = np.empty(signal.size * 2, dtype=np.int8)
                    interleaved[0::2] = i
                    interleaved[1::2] = q
                    payload = interleaved.tobytes()
                    state.stream.write(payload)
                    state.digest.update(payload)
                    state.complex_samples += signal.size
                    state.signal_power_sum += float(np.vdot(rx_signal, rx_signal).real)
                    state.output_power_sum += float(np.sum(i.astype(np.float32) ** 2 + q.astype(np.float32) ** 2))
                    if signal.size:
                        state.peak_magnitude = max(state.peak_magnitude, float(np.max(np.hypot(i.astype(np.float32), q.astype(np.float32)))))
                sample_index += authentic.size
        reports: dict[str, Any] = {}
        for state in states:
            state.stream.flush()
            os.fsync(state.stream.fileno())
            state.stream.close()
            os.replace(state.temporary, state.path)
            reports[state.scenario.name] = {
                "path": state.path.name,
                "sha256": state.digest.hexdigest(),
                "bytes": state.path.stat().st_size,
                "complex_samples": state.complex_samples,
                "actual_duration_seconds": state.complex_samples / sample_rate_hz,
                "mean_quantized_complex_power": state.output_power_sum / state.complex_samples,
                "mean_analog_signal_power": state.signal_power_sum / state.complex_samples,
                "peak_quantized_magnitude": state.peak_magnitude,
                "clipped_complex_samples": state.clipped_complex_samples,
                "clipped_components": state.clipped_components,
                "clipping_fraction": state.clipped_complex_samples / state.complex_samples,
            }
        return {
            "reference": reference,
            "scenarios": reports,
            "processing": {
                "composition": "float32 superposition of IQ8 simulator sources; one final IQ8 quantization",
                "receiver_gain_policy": "authentic-only calibration frozen across every paired scenario",
                "agc": "disabled; no scenario-dependent or future-event gain",
                "noise_pairing": "same seeded frontend-output AWGN samples for every scenario",
                "chunk_samples": receiver.chunk_samples,
            },
            "cn0_caveat": CN0_CAVEAT,
        }
    except Exception:
        for state in states:
            if not state.stream.closed:
                state.stream.close()
            state.temporary.unlink(missing_ok=True)
            state.path.unlink(missing_ok=True)
        raise


def compare_prefix(path_a: str | Path, path_b: str | Path, complex_samples: int) -> dict[str, Any]:
    """Hash and compare an exact IQ prefix without loading it into memory."""
    byte_count = int(complex_samples) * 2
    if byte_count < 0:
        raise SimulationV4Error("prefix sample count must be non-negative")
    digests = []
    payloads_equal = True
    remaining = byte_count
    streams = [Path(path_a).open("rb"), Path(path_b).open("rb")]
    hashes = [hashlib.sha256(), hashlib.sha256()]
    try:
        while remaining:
            count = min(1024 * 1024, remaining)
            chunks = [stream.read(count) for stream in streams]
            if len(chunks[0]) != count or len(chunks[1]) != count:
                raise SimulationV4Error("prefix extends beyond IQ file")
            hashes[0].update(chunks[0])
            hashes[1].update(chunks[1])
            payloads_equal &= chunks[0] == chunks[1]
            remaining -= count
        digests = [value.hexdigest() for value in hashes]
    finally:
        for stream in streams:
            stream.close()
    return {"complex_samples": complex_samples, "bytes": byte_count, "sha256": digests, "byte_identical": payloads_equal}


def _scenario_truth(scenario: SimulationScenario) -> dict[str, Any]:
    if scenario.kind == "steady_normal":
        return {"class": "normal", "event": "steady", "is_spoofing": False}
    if scenario.kind == "recovery_normal":
        return {"class": "normal", "event": "outage_recovery", "is_spoofing": False, "outage": asdict(scenario.outage)}
    return {"class": "spoofing", "event": "carryoff", "is_spoofing": True, "spoofing": asdict(scenario.spoofing)}


def _component_record(path: Path, sample_rate_hz: int, retained: bool) -> dict[str, Any]:
    return {
        "path": str(path) if retained else None,
        "retained": retained,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "complex_samples": path.stat().st_size // 2,
        "rf_sample_rate_hz": sample_rate_hz,
        "sample_format": "s8_iq",
    }


def generate_simulation_campaign(campaign: SimulationCampaign, runner: GpsSdrSimRunner | None = None) -> Path:
    """Generate shared sources, three paired IQ outputs, truth, and manifests."""
    root = campaign.output_root
    if root.exists():
        raise FileExistsError(root)
    components = root / "components"
    rf_root = root / "rf"
    components.mkdir(parents=True)
    rf_root.mkdir()
    config = campaign.base_rf_config
    runner = runner or GpsSdrSimRunner(config.simulator.executable)
    sample_rate = config.output.rf_sample_rate_hz
    authentic = components / "authentic_gps_l1ca_s8_iq.bin"
    counterfeit = components / "counterfeit_gps_l1ca_s8_iq.bin"
    authentic_log = components / "authentic-gps-sdr-sim.log"
    counterfeit_log = components / "counterfeit-gps-sdr-sim.log"
    spoof_scenarios = [item for item in campaign.scenarios if item.kind == "carryoff_spoof"]
    if len(spoof_scenarios) != 1:
        raise SimulationV4Error("pilot campaign requires exactly one carryoff_spoof source")
    spoof = spoof_scenarios[0]
    assert spoof.spoofing is not None
    position = config.scenario.position
    assert isinstance(position, StaticPosition)
    authentic_rows = tuple(
        (index / 10.0, position.latitude_deg, position.longitude_deg, position.altitude_m)
        for index in range(config.scenario.duration_seconds * 10)
    )
    counterfeit_rows = build_carryoff_rows(authentic_rows, spoof.spoofing)
    trajectory_path = components / "counterfeit_trajectory.csv"
    trajectory = _write_carryoff_trajectory(trajectory_path, counterfeit_rows, spoof.spoofing)
    counterfeit_position = TrajectoryPosition(
        trajectory_path,
        "llh",
        counterfeit_rows,
        trajectory["csv_sha256"],
        trajectory["sidecar"],
        trajectory["sidecar_sha256"],
    )
    counterfeit_config = replace(config, scenario=replace(config.scenario, position=counterfeit_position))
    authentic_result = runner.run(config, authentic, authentic_log)
    counterfeit_result = runner.run(counterfeit_config, counterfeit, counterfeit_log)
    expected_bytes = runner.expected_output_bytes(config)
    if authentic.stat().st_size != expected_bytes or counterfeit.stat().st_size != expected_bytes:
        raise SimulationV4Error("GPS-SDR-SIM component size violates the pinned runner contract")
    component_records = {
        "authentic": _component_record(authentic, sample_rate, campaign.keep_component_iq),
        "counterfeit": _component_record(counterfeit, sample_rate, campaign.keep_component_iq),
    }
    run_ids: dict[str, str] = {}
    destinations: dict[str, Path] = {}
    run_dirs: dict[str, Path] = {}
    for scenario in campaign.scenarios:
        run_id = f"{campaign.name}-{scenario.name}_{config.scenario.utc.strftime('%Y%m%dT%H%M%SZ')}"
        run_dir = rf_root / run_id
        run_dir.mkdir()
        run_ids[scenario.name] = run_id
        run_dirs[scenario.name] = run_dir
        destinations[scenario.name] = run_dir / "gps_l1ca_s8_iq.bin"
    composition = compose_paired_iq(
        authentic,
        counterfeit,
        destinations,
        campaign.scenarios,
        sample_rate_hz=sample_rate,
        receiver=campaign.receiver,
        normal_target_rms=campaign.normal_target_rms,
    )
    steady = next(item for item in campaign.scenarios if item.kind == "steady_normal")
    prefix_checks: dict[str, Any] = {}
    for scenario in campaign.scenarios:
        if scenario.kind == "steady_normal":
            continue
        onset = scenario.outage.start_seconds if scenario.outage else scenario.spoofing.start_seconds
        check = compare_prefix(
            destinations[steady.name], destinations[scenario.name], int(round(onset * sample_rate)),
        )
        if not check["byte_identical"]:
            raise SimulationV4Error(f"paired prefix differs before {scenario.name} onset")
        prefix_checks[scenario.name] = {"reference_scenario": steady.name, "onset_seconds": onset, **check}
    time_metadata = simulator_time(config.scenario.utc, config.input.rinex_nav).manifest()
    manifest_paths: dict[str, str] = {}
    for scenario in campaign.scenarios:
        report = composition["scenarios"][scenario.name]
        manifest = {
            "schema_version": 4,
            "run_id": run_ids[scenario.name],
            "scenario": {
                "name": scenario.name,
                "campaign": campaign.name,
                "constellation": "GPS",
                "signal": "L1CA",
                "utc": config.scenario.utc.isoformat().replace("+00:00", "Z"),
                "time": time_metadata,
                "duration_seconds": config.scenario.duration_seconds,
                "position": asdict(position),
            },
            "input": {
                "rinex_nav": str(config.input.rinex_nav),
                "rinex_nav_sha256": _sha256(config.input.rinex_nav),
            },
            "iq": {
                "path": report["path"],
                "sha256": report["sha256"],
                "actual_bytes": report["bytes"],
                "complex_samples": report["complex_samples"],
                "actual_duration_seconds": report["actual_duration_seconds"],
                "rf_sample_rate_hz": sample_rate,
                "sample_format": "s8_iq",
                "channels": 2,
            },
            "simulation_v4": {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "truth": _scenario_truth(scenario),
                "receiver": {
                    "requested": campaign.receiver.manifest(),
                    "reference": composition["reference"],
                    "processing": composition["processing"],
                },
                "measurements": report,
                "paired_prefix_check": prefix_checks.get(scenario.name),
                "sources": component_records,
                "scope": "offline baseband or shielded/cabled receiver evaluation only; no RF transmission",
            },
            "simulator": {
                "identity": runner.identity,
                "executable": runner.executable,
                "provenance": runner.provenance,
                "cli_contract": runner.cli_contract,
                "authentic_command": authentic_result["command"],
                "counterfeit_command": counterfeit_result["command"],
                "authentic_log": str(authentic_log),
                "counterfeit_log": str(counterfeit_log),
            },
        }
        manifest_path = run_dirs[scenario.name] / "manifest.json"
        _atomic_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        manifest_paths[scenario.name] = str(manifest_path)
    campaign_manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "campaign": {
            "name": campaign.name,
            "source_config": str(campaign.source_config_path),
            "source_config_sha256": _sha256(campaign.source_config_path),
            "utc": config.scenario.utc.isoformat().replace("+00:00", "Z"),
            "duration_seconds": config.scenario.duration_seconds,
            "rf_sample_rate_hz": sample_rate,
        },
        "components": component_records,
        "truth": {
            "counterfeit_trajectory": str(trajectory_path),
            "counterfeit_trajectory_sha256": trajectory["csv_sha256"],
            "counterfeit_trajectory_metadata": str(trajectory["sidecar"]),
            "realized_final_offset_enu_m": trajectory["realized_final_offset_enu_m"],
        },
        "composition": composition,
        "paired_prefix_checks": prefix_checks,
        "rf_manifests": manifest_paths,
        "dataset_role": "simulation-only pilot; not external validation evidence",
    }
    manifest_path = root / "campaign_manifest.json"
    _atomic_write(manifest_path, (json.dumps(campaign_manifest, indent=2, sort_keys=True) + "\n").encode())
    if not campaign.keep_component_iq:
        authentic.unlink()
        counterfeit.unlink()
    return manifest_path
