"""Versioned configuration for normal GPS L1 C/A IQ generation."""
from dataclasses import dataclass
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
class PrnSelection:
    mode: str
    prns: tuple[int, ...]

@dataclass(frozen=True)
class SpoofingPower:
    initial_advantage_db: float
    final_advantage_db: float
    ramp_seconds: float

@dataclass(frozen=True)
class SpoofingConfig:
    attack_type: str
    start_seconds: float
    transition_seconds: float
    target_offset_enu_m: tuple[float, float, float]
    prn_selection: PrnSelection
    power: SpoofingPower
    keep_component_iq: bool

@dataclass(frozen=True)
class RFGenerationConfig:
    version: int
    scenario: Scenario
    input: InputConfig
    output: OutputConfig
    simulator: SimulatorConfig
    spoofing: SpoofingConfig | None = None


def _required(obj: dict[str, Any], key: str) -> Any:
    if key not in obj: raise ConfigError(f"missing required key: {key}")
    return obj[key]


def _strict_integer(value: Any, name: str) -> int:
    """Accept integer-valued YAML numbers, never lossy int() coercion."""
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ConfigError(f"{name} must be an integer")
    return int(number)


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite")
    return number


def _load_spoofing(data: dict[str, Any], duration_seconds: int) -> SpoofingConfig:
    raw = _required(data, "spoofing")
    if not isinstance(raw, dict):
        raise ConfigError("spoofing must be a mapping")
    attack_type = str(_required(raw, "attack_type"))
    if attack_type not in {"abrupt", "carry_off"}:
        raise ConfigError("spoofing.attack_type must be abrupt or carry_off")
    start = _finite_float(_required(raw, "start_seconds"), "spoofing.start_seconds")
    transition = _finite_float(
        _required(raw, "transition_seconds"), "spoofing.transition_seconds"
    )
    if not 0 <= start < duration_seconds:
        raise ConfigError("spoofing.start_seconds must occur within scenario duration")
    if transition < 0 or (attack_type == "carry_off" and transition <= 0):
        raise ConfigError("spoofing.transition_seconds must be positive for carry_off")
    if start + transition > duration_seconds:
        raise ConfigError("spoofing transition must finish within scenario duration")

    offset = _required(raw, "target_offset_enu_m")
    if not isinstance(offset, dict):
        raise ConfigError("spoofing.target_offset_enu_m must be a mapping")
    target_offset = tuple(
        _finite_float(_required(offset, key), f"spoofing.target_offset_enu_m.{key}")
        for key in ("east_m", "north_m", "up_m")
    )
    if target_offset == (0.0, 0.0, 0.0):
        raise ConfigError("spoofing.target_offset_enu_m must be non-zero")

    selection = _required(raw, "prn_selection")
    if not isinstance(selection, dict):
        raise ConfigError("spoofing.prn_selection must be a mapping")
    mode = str(_required(selection, "mode"))
    if mode not in {"all_visible", "explicit"}:
        raise ConfigError("spoofing.prn_selection.mode must be all_visible or explicit")
    raw_prns = selection.get("prns", ())
    if not isinstance(raw_prns, (list, tuple)):
        raise ConfigError("spoofing.prn_selection.prns must be a sequence")
    prns = tuple(_strict_integer(value, "spoofing PRN") for value in raw_prns)
    if len(set(prns)) != len(prns):
        raise ConfigError("spoofing.prn_selection contains duplicate PRNs")
    if any(not 1 <= prn <= 32 for prn in prns):
        raise ConfigError("spoofing PRNs must be in [1, 32]")
    if mode == "explicit" and not prns:
        raise ConfigError("explicit spoofing PRN selection must not be empty")
    if mode == "all_visible" and prns:
        raise ConfigError("all_visible PRN selection must not include an explicit list")

    power = _required(raw, "power")
    if not isinstance(power, dict):
        raise ConfigError("spoofing.power must be a mapping")
    initial_db = _finite_float(
        _required(power, "initial_advantage_db"),
        "spoofing.power.initial_advantage_db",
    )
    final_db = _finite_float(
        _required(power, "final_advantage_db"),
        "spoofing.power.final_advantage_db",
    )
    ramp = _finite_float(_required(power, "ramp_seconds"), "spoofing.power.ramp_seconds")
    if ramp < 0 or start + ramp > duration_seconds:
        raise ConfigError("spoofing.power.ramp_seconds must be non-negative and finish within duration")
    keep = raw.get("keep_component_iq", True)
    if not isinstance(keep, bool):
        raise ConfigError("spoofing.keep_component_iq must be boolean")
    return SpoofingConfig(
        attack_type, start, transition, target_offset,
        PrnSelection(mode, prns), SpoofingPower(initial_db, final_db, ramp), keep,
    )


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
    version = data.get("version")
    if version not in (1, 2): raise ConfigError("unsupported config version (supported: 1 normal, 2 spoofing)")
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
        if not isinstance(sim, dict): raise ConfigError("simulator must be a mapping")
        if version == 1 and "spoofing" in data:
            raise ConfigError("version 1 normal configuration must not contain spoofing")
        spoofing = _load_spoofing(data, duration) if version == 2 else None
        return RFGenerationConfig(version, Scenario(name, constellation, signal, utc, duration, position), InputConfig(nav), OutputConfig(root, rate, fmt), SimulatorConfig(sim.get("executable")), spoofing)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError): raise
        raise ConfigError(f"invalid configuration: {exc}") from exc
