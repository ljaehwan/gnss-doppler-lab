"""Versioned configuration for normal GPS L1 C/A IQ generation."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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
class Scenario:
    name: str
    constellation: str
    signal: str
    utc: datetime
    duration_seconds: int
    position: StaticPosition

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


def _required(obj: dict[str, Any], key: str) -> Any:
    if key not in obj: raise ConfigError(f"missing required key: {key}")
    return obj[key]


def load_rf_config(path: str | Path) -> RFGenerationConfig:
    path = Path(path).resolve()
    try: data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc: raise ConfigError(str(exc)) from exc
    if not isinstance(data, dict): raise ConfigError("configuration must be a mapping")
    if data.get("version") != 1: raise ConfigError("unsupported config version (only version 1)")
    try:
        s, pos = _required(data, "scenario"), _required(data["scenario"], "position")
        if pos.get("type") != "static": raise ConfigError("trajectory position is not supported in milestone 1")
        utc = datetime.fromisoformat(str(_required(s, "utc")).replace("Z", "+00:00"))
        if utc.tzinfo is None: raise ConfigError("scenario.utc must include timezone")
        utc = utc.astimezone(timezone.utc)
        constellation, signal = s.get("constellation"), s.get("signal")
        if (constellation, signal) != ("GPS", "L1CA"): raise ConfigError("only GPS L1CA is supported")
        name = str(_required(s, "name"))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ConfigError("scenario.name must be a safe identifier")
        duration = int(_required(s, "duration_seconds"))
        if not 1 <= duration <= 86400: raise ConfigError("duration_seconds must be between 1 and 86400")
        position = StaticPosition(float(pos["latitude_deg"]), float(pos["longitude_deg"]), float(pos["altitude_m"]))
        if not all(math.isfinite(v) for v in (position.latitude_deg, position.longitude_deg, position.altitude_m)):
            raise ConfigError("static coordinates must be finite")
        if not -90 <= position.latitude_deg <= 90 or not -180 <= position.longitude_deg <= 180: raise ConfigError("invalid static coordinates")
        inp, out = _required(data, "input"), _required(data, "output")
        nav = (path.parent / _required(inp, "rinex_nav")).resolve()
        root = (path.parent / _required(out, "root")).resolve()
        rate = int(_required(out, "rf_sample_rate_hz"))
        fmt = out.get("sample_format", "s8_iq")
        if rate < 1_000_000 or fmt != "s8_iq": raise ConfigError("rf sample rate must be at least 1000000 Hz and sample_format must be s8_iq")
        sim = data.get("simulator", {})
        return RFGenerationConfig(1, Scenario(name, constellation, signal, utc, duration, position), InputConfig(nav), OutputConfig(root, rate, fmt), SimulatorConfig(sim.get("executable")))
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError): raise
        raise ConfigError(f"invalid configuration: {exc}") from exc
