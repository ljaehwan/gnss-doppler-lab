"""Versioned configuration for normal GPS L1 C/A IQ generation."""
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import re
import yaml

class ConfigError(ValueError): pass

@dataclass(frozen=True)
class StaticPosition:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float

@dataclass(frozen=True)
class TrajectoryPosition:
    path: Path
    coordinate_system: str
    rows: tuple
    csv_sha256: str
    metadata_path: Path | None
    metadata_sha256: str | None

@dataclass(frozen=True)
class Scenario:
    name: str
    constellation: str
    signal: str
    utc: datetime
    duration_seconds: int
    position: StaticPosition | TrajectoryPosition

@dataclass(frozen=True)
class InputConfig:
    rinex_nav: Path

@dataclass(frozen=True)
class OutputConfig:
    root: Path
    rf_sample_rate_hz: int
    sample_format: str

@dataclass(frozen=True)
class SimulatorConfig:
    executable: str | None = None

@dataclass(frozen=True)
class RFGenerationConfig:
    version: int
    scenario: Scenario
    input: InputConfig
    output: OutputConfig
    simulator: SimulatorConfig
    impairments: Any


def _load_impairments(raw: Any, sample_rate_hz: int):
    """Parse strict profile semantics into a complete realized configuration."""
    from .rf_impairments import ImpairmentConfig, MultipathTap, clean_impairments, open_sky_normal
    if raw is None:
        return clean_impairments()  # legacy omission remains a zero-cost clean path
    if not isinstance(raw, dict):
        raise ConfigError("impairments must be a mapping")
    if not raw:
        raise ConfigError("impairments mapping must not be empty; omit it for the legacy clean path")
    allowed = {f.name for f in fields(ImpairmentConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown impairment key(s): {', '.join(sorted(unknown))}")
    profile = raw.get("profile")
    if profile is None:
        if raw.get("enabled") is False:
            profile = "clean"
        elif set(raw) - {"enabled", "seed"}:
            profile = "explicit"
        else:
            raise ConfigError("enabled impairments would be an expensive identity pass; select a profile or explicit effects")
    if profile not in {"clean", "explicit", "open_sky_normal"}:
        raise ConfigError("impairments.profile must be clean, explicit, or open_sky_normal")
    default_enabled = profile != "clean"
    enabled = raw.get("enabled", default_enabled)
    if not isinstance(enabled, bool):
        raise ConfigError("impairments.enabled must be boolean")
    if profile == "clean" and enabled:
        raise ConfigError("clean profile requires impairments to be disabled")
    if profile == "open_sky_normal" and not enabled:
        raise ConfigError("open_sky_normal profile requires impairments to be enabled")
    if profile == "explicit" and not enabled:
        raise ConfigError("explicit profile requires impairments to be enabled")
    preset_keys = {"enabled", "profile", "seed"}
    if profile in {"clean", "open_sky_normal"} and set(raw) - preset_keys:
        raise ConfigError(f"{profile} preset accepts only enabled, profile, and seed")
    seed = _strict_integer(raw.get("seed", 0), "impairments.seed")
    if not 0 <= seed < 2**64:
        raise ConfigError("impairments.seed must be an integer in [0, 2^64)")
    if profile == "explicit" and not (set(raw) - preset_keys):
        raise ConfigError("explicit enabled impairments would be an expensive identity pass")
    try:
        if profile == "open_sky_normal":
            return open_sky_normal(seed, sample_rate_hz)
        if profile == "clean":
            return replace(clean_impairments(), seed=seed)
        base = ImpairmentConfig(enabled=True, profile="explicit", seed=seed)
        values = {"enabled": True, "profile": "explicit", "seed": seed}
        integer_fields = {"seed", "frontend_order", "chunk_samples"}
        nullable_fields = {"sample_snr_db", "frontend_cutoff_hz", "agc_target_rms"}
        for key, value in raw.items():
            if key in {"enabled", "profile", "seed"}:
                continue
            if key in integer_fields:
                values[key] = _strict_integer(value, f"impairments.{key}")
            elif key == "multipath":
                if not isinstance(value, list):
                    raise ConfigError("impairments.multipath must be a list")
                taps = []
                for index, tap in enumerate(value):
                    if not isinstance(tap, dict) or set(tap) != {"delay_samples", "attenuation_db", "phase_deg"}:
                        raise ConfigError(f"impairments.multipath[{index}] requires exactly delay_samples, attenuation_db, phase_deg")
                    taps.append(MultipathTap(*(float(tap[k]) for k in ("delay_samples", "attenuation_db", "phase_deg"))))
                values[key] = tuple(taps)
            elif value is None and key in nullable_fields:
                values[key] = None
            else:
                values[key] = float(value)
        cfg = replace(base, **values)
        if cfg.frontend_cutoff_hz is not None and cfg.frontend_cutoff_hz >= sample_rate_hz / 2:
            raise ConfigError("impairments.frontend_cutoff_hz must be below Nyquist")
        return cfg
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"invalid impairments: {exc}") from exc


def _required(obj: dict[str, Any], key: str) -> Any:
    if key not in obj: raise ConfigError(f"missing required key: {key}")
    return obj[key]


def _strict_integer(value: Any, name: str) -> int:
    """Accept exact integers without routing Python/YAML ints through binary64."""
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ConfigError(f"{name} must be an integer")
    raise ConfigError(f"{name} must be an integer")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _trajectory_integrity(trajectory_path: Path) -> tuple[str, Path | None, str | None]:
    """Validate a generated sidecar; external CSVs may omit the sidecar."""
    try:
        csv_digest = _sha256_bytes(trajectory_path.read_bytes())
    except OSError as exc:
        raise ConfigError(f"invalid trajectory: {exc}") from exc
    metadata_path = trajectory_path.with_suffix(".json")
    if not metadata_path.exists():
        return csv_digest, None, None
    try:
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid trajectory sidecar: {exc}") from exc
    generator = metadata.get("generator") if isinstance(metadata, dict) else None
    required = ("csv_sha256", "sample_rate_hz", "actual_row_count")
    if (
        not isinstance(generator, dict)
        or generator.get("schema") != "gnss-doppler-lab.trajectory"
        or generator.get("version") != 2
        or any(key not in metadata for key in required)
    ):
        raise ConfigError("invalid trajectory sidecar: unsupported or incomplete schema")
    claimed_digest = metadata["csv_sha256"]
    if not isinstance(claimed_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed_digest):
        raise ConfigError("invalid trajectory sidecar: csv_sha256 must be a lowercase SHA-256")
    if claimed_digest != csv_digest:
        raise ConfigError("invalid trajectory sidecar: csv_sha256 does not match trajectory CSV")
    return csv_digest, metadata_path, _sha256_bytes(metadata_bytes)


def load_rf_config(path: str | Path) -> RFGenerationConfig:
    path = Path(path).resolve()
    try: data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc: raise ConfigError(str(exc)) from exc
    if not isinstance(data, dict): raise ConfigError("configuration must be a mapping")
    if data.get("version") != 1: raise ConfigError("unsupported config version (only version 1)")
    try:
        s, pos = _required(data, "scenario"), _required(data["scenario"], "position")
        position_type = pos.get("type")
        if position_type not in ("static", "trajectory"): raise ConfigError("position.type must be static or trajectory")
        utc = datetime.fromisoformat(str(_required(s, "utc")).replace("Z", "+00:00"))
        if utc.tzinfo is None: raise ConfigError("scenario.utc must include timezone")
        if utc.microsecond != 0:
            raise ConfigError("scenario.utc must use integer seconds; fractional seconds are not supported")
        utc = utc.astimezone(timezone.utc)
        constellation, signal = s.get("constellation"), s.get("signal")
        if (constellation, signal) != ("GPS", "L1CA"): raise ConfigError("only GPS L1CA is supported")
        name = str(_required(s, "name"))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ConfigError("scenario.name must be a safe identifier")
        duration = _strict_integer(_required(s, "duration_seconds"), "duration_seconds")
        maximum = 300 if position_type == "trajectory" else 86400
        if not 1 <= duration <= maximum: raise ConfigError(f"duration_seconds must be between 1 and {maximum}")
        if position_type == "static":
            position = StaticPosition(float(pos["latitude_deg"]), float(pos["longitude_deg"]), float(pos["altitude_m"]))
            if not all(math.isfinite(v) for v in (position.latitude_deg, position.longitude_deg, position.altitude_m)): raise ConfigError("static coordinates must be finite")
            if not -90 <= position.latitude_deg <= 90 or not -180 <= position.longitude_deg <= 180: raise ConfigError("invalid static coordinates")
        else:
            from .trajectory import read_trajectory
            cs = str(_required(pos, "coordinate_system")).lower()
            if cs not in ("llh", "ecef"): raise ConfigError("coordinate_system must be llh or ecef")
            trajectory_path = (path.parent / _required(pos, "path")).resolve()
            csv_digest, metadata_path, metadata_digest = _trajectory_integrity(trajectory_path)
            try: rows = read_trajectory(trajectory_path, duration, cs)
            except ValueError as exc: raise ConfigError(f"invalid trajectory: {exc}") from exc
            try:
                post_validation_digest = _sha256_bytes(trajectory_path.read_bytes())
            except OSError as exc:
                raise ConfigError(f"invalid trajectory: {exc}") from exc
            if post_validation_digest != csv_digest:
                raise ConfigError("invalid trajectory: CSV changed during validation")
            position = TrajectoryPosition(
                trajectory_path, cs, tuple(rows), csv_digest,
                metadata_path, metadata_digest,
            )
        inp, out = _required(data, "input"), _required(data, "output")
        nav = (path.parent / _required(inp, "rinex_nav")).resolve()
        root = (path.parent / _required(out, "root")).resolve()
        rate = _strict_integer(_required(out, "rf_sample_rate_hz"), "rf_sample_rate_hz")
        fmt = out.get("sample_format", "s8_iq")
        if rate < 1_000_000 or fmt != "s8_iq": raise ConfigError("rf sample rate must be at least 1000000 Hz and sample_format must be s8_iq")
        sim = data.get("simulator", {})
        impairments = _load_impairments(data.get("impairments"), rate)
        return RFGenerationConfig(1, Scenario(name, constellation, signal, utc, duration, position), InputConfig(nav), OutputConfig(root, rate, fmt), SimulatorConfig(sim.get("executable")), impairments)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError): raise
        raise ConfigError(f"invalid configuration: {exc}") from exc
