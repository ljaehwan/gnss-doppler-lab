"""Satellite-specific RF multipath controls for the patched gps-sdr-sim."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable

from .gps_sdr_sim import GpsSdrSimRunner, SimulatorError, _sha256


GPS_L1_CA_CHIP_RATE_HZ = 1_023_000.0
PATCH_CONTRACT = "prn-fractional-delay-multipath-v1"


@dataclass(frozen=True)
class SatelliteMultipathEcho:
    """One delayed replica applied to one GPS L1 C/A PRN before RF summation."""

    prn: int
    delay_chips: float
    amplitude_ratio: float
    phase_deg: float

    def __post_init__(self) -> None:
        if not 1 <= self.prn <= 32:
            raise ValueError("prn must be between 1 and 32")
        if not math.isfinite(self.delay_chips) or self.delay_chips <= 0.0:
            raise ValueError("delay_chips must be finite and positive")
        if not math.isfinite(self.amplitude_ratio) or not 0.0 < self.amplitude_ratio <= 1.0:
            raise ValueError("amplitude_ratio must be finite and in (0, 1]")
        if not math.isfinite(self.phase_deg):
            raise ValueError("phase_deg must be finite")

    def delay_samples(self, sample_rate_hz: float) -> float:
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be finite and positive")
        return self.delay_chips * sample_rate_hz / GPS_L1_CA_CHIP_RATE_HZ

    def manifest(self, sample_rate_hz: float) -> dict[str, float | int]:
        return {**asdict(self), "delay_samples": self.delay_samples(sample_rate_hz)}


def validate_echoes(
    echoes: Iterable[SatelliteMultipathEcho], sample_rate_hz: float
) -> tuple[SatelliteMultipathEcho, ...]:
    """Freeze echoes in PRN order and enforce the patched simulator contract."""
    frozen = tuple(sorted(echoes, key=lambda echo: echo.prn))
    if not frozen:
        raise ValueError("at least one satellite multipath echo is required")
    prns = [echo.prn for echo in frozen]
    if len(prns) != len(set(prns)):
        raise ValueError("multipath echoes must contain unique PRNs")
    for echo in frozen:
        delay = echo.delay_samples(sample_rate_hz)
        if delay < 1.0:
            raise ValueError(
                f"PRN {echo.prn} delay resolves to less than one RF sample"
            )
        if delay > sample_rate_hz * 0.001:
            raise ValueError(f"PRN {echo.prn} delay exceeds the 1 ms simulator bound")
    return frozen


def write_multipath_spec(
    path: str | Path,
    echoes: Iterable[SatelliteMultipathEcho],
    *,
    sample_rate_hz: float,
) -> Path:
    """Write the strict C-extension contract without a header or implicit units."""
    frozen = validate_echoes(echoes, sample_rate_hz)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    lines = [
        f"{echo.prn},{echo.delay_samples(sample_rate_hz):.12f},"
        f"{echo.amplitude_ratio:.12f},{echo.phase_deg:.12f}"
        for echo in frozen
    ]
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output


def _unit_interval(seed: int, prn: int, field: str) -> float:
    digest = hashlib.sha256(f"{seed}:{prn}:{field}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def independent_echoes(
    prns: Iterable[int],
    *,
    seed: int,
    delay_chips_range: tuple[float, float] = (0.15, 0.85),
    amplitude_range: tuple[float, float] = (0.15, 0.65),
) -> tuple[SatelliteMultipathEcho, ...]:
    """Derive reproducible PRN-independent delays, amplitudes, and phases."""
    requested_prns = tuple(int(prn) for prn in prns)
    if not requested_prns:
        raise ValueError("prns must not be empty")
    if len(requested_prns) != len(set(requested_prns)):
        raise ValueError("prns must be unique")
    unique_prns = tuple(sorted(requested_prns))
    delay_min, delay_max = delay_chips_range
    amplitude_min, amplitude_max = amplitude_range
    if not (0.0 < delay_min < delay_max):
        raise ValueError("delay_chips_range must be positive and increasing")
    if not (0.0 < amplitude_min < amplitude_max <= 1.0):
        raise ValueError("amplitude_range must lie in (0, 1]")
    return tuple(
        SatelliteMultipathEcho(
            prn=prn,
            delay_chips=delay_min
            + (delay_max - delay_min) * _unit_interval(seed, prn, "delay"),
            amplitude_ratio=amplitude_min
            + (amplitude_max - amplitude_min) * _unit_interval(seed, prn, "amplitude"),
            phase_deg=-180.0 + 360.0 * _unit_interval(seed, prn, "phase"),
        )
        for prn in unique_prns
    )


class PrnMultipathGpsSdrSimRunner(GpsSdrSimRunner):
    """Run the pinned simulator plus the auditable PRN-multipath C patch."""

    def __init__(
        self,
        executable: str | None,
        echoes: Iterable[SatelliteMultipathEcho],
    ):
        super().__init__(executable)
        self.echoes = tuple(echoes)
        self.cli_contract = f"{self.cli_contract}+{PATCH_CONTRACT}"
        self._active_spec_path: Path | None = None

    def build_command(self, config, output: Path, nav=None, time=None) -> list[str]:
        command = super().build_command(config, output, nav, time)
        if self._active_spec_path is None:
            raise SimulatorError("multipath spec is not staged")
        spec_path = self._active_spec_path.resolve()
        output_parent = Path(output).resolve().parent
        if spec_path.parent != output_parent:
            raise SimulatorError(
                "multipath spec must be staged beside the simulator output"
            )
        output_index = command.index("-o")
        command[output_index:output_index] = ["-m", spec_path.name]
        return command

    def run(self, config, output: Path, log: Path) -> dict:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frozen = validate_echoes(self.echoes, config.output.rf_sample_rate_hz)
        spec_path = write_multipath_spec(
            output.parent / "multipath.csv",
            frozen,
            sample_rate_hz=config.output.rf_sample_rate_hz,
        )
        self._active_spec_path = spec_path
        try:
            result = super().run(config, output, log)
        finally:
            self._active_spec_path = None
        result["multipath"] = {
            "patch_contract": PATCH_CONTRACT,
            "spec_path": str(spec_path.resolve()),
            "spec_sha256": _sha256(spec_path),
            "echoes": [
                echo.manifest(config.output.rf_sample_rate_hz) for echo in frozen
            ],
        }
        return result
