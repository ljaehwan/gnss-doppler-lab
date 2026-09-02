"""Reproducible adapter for code/carrier-decoupled gps-sdr-sim experiments."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import os
import shutil
import subprocess
import tempfile
from typing import Any

from .gps_sdr_sim import SimulatorError, simulator_time


UPSTREAM_COMMIT = "28ca29a6719475195e3aabd5930c4ed02d67190f"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _motion_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="ascii", errors="strict") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.strip().split(",")
            if len(fields) != 4:
                raise SimulatorError(f"invalid LLH motion row {number}: {path}")
            try:
                values = tuple(float(value) for value in fields)
            except ValueError as exc:
                raise SimulatorError(f"non-numeric LLH motion row {number}: {path}") from exc
            if abs(values[0] - count / 10.0) > 5e-7:
                raise SimulatorError(f"motion must use an exact 10 Hz time grid at row {number}: {path}")
            if not -90.0 <= values[1] <= 90.0 or not -180.0 <= values[2] <= 180.0:
                raise SimulatorError(f"invalid LLH coordinates at row {number}: {path}")
            count += 1
    if count == 0:
        raise SimulatorError(f"empty LLH motion file: {path}")
    return count


@dataclass(frozen=True)
class DecoupledSimulationRequest:
    nav: Path
    code_motion: Path
    carrier_motion: Path | None
    utc: datetime
    duration_seconds: int
    sample_rate_hz: int
    mode: str
    carrier_phase_seed: int | None = None

    def validate(self) -> dict[str, Any]:
        if self.mode not in {"coupled", "doppler_locked"}:
            raise SimulatorError("mode must be 'coupled' or 'doppler_locked'")
        if self.mode == "doppler_locked" and self.carrier_motion is None:
            raise SimulatorError("doppler_locked mode requires carrier_motion")
        if self.mode == "coupled" and self.carrier_motion is not None:
            raise SimulatorError("coupled mode must omit carrier_motion")
        if self.duration_seconds <= 0 or self.sample_rate_hz < 1_000_000:
            raise SimulatorError("duration and sample rate are outside gps-sdr-sim bounds")
        if self.carrier_phase_seed is not None:
            if isinstance(self.carrier_phase_seed, bool) or not isinstance(self.carrier_phase_seed, int) or self.carrier_phase_seed < 0:
                raise SimulatorError("carrier_phase_seed must be a non-negative integer")
        for path in (self.nav, self.code_motion):
            if not path.is_file():
                raise SimulatorError(f"missing simulator input: {path}")
        code_rows = _motion_rows(self.code_motion)
        required_rows = self.duration_seconds * 10
        if code_rows < required_rows:
            raise SimulatorError("code motion is shorter than the requested duration")
        carrier_rows = None
        if self.carrier_motion is not None:
            if not self.carrier_motion.is_file():
                raise SimulatorError(f"missing carrier motion: {self.carrier_motion}")
            carrier_rows = _motion_rows(self.carrier_motion)
            if carrier_rows < required_rows:
                raise SimulatorError("carrier motion is shorter than the requested duration")
        time = simulator_time(self.utc, self.nav)
        return {
            "code_rows": code_rows,
            "carrier_rows": carrier_rows,
            "time": time,
        }


class CodeCarrierGpsSdrSimRunner:
    """Run the repository's gps-sdr-sim decoupling patch in a short staging path."""

    def __init__(self, executable: str | Path, patch: str | Path):
        self.executable = Path(executable).resolve()
        self.patch = Path(patch).resolve()
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise SimulatorError(f"decoupled simulator is not executable: {self.executable}")
        if not self.patch.is_file():
            raise SimulatorError(f"decoupling patch is missing: {self.patch}")

    @staticmethod
    def expected_output_bytes(request: DecoupledSimulationRequest) -> int:
        blocks = request.duration_seconds * 10 - 1
        return blocks * (request.sample_rate_hz // 10) * 2

    def probe(self) -> dict[str, Any]:
        return {
            "available": True,
            "executable": str(self.executable),
            "executable_sha256": sha256(self.executable),
            "patch": str(self.patch),
            "patch_sha256": sha256(self.patch),
            "upstream_commit": UPSTREAM_COMMIT,
            "cli_extension": {
                "-X": "independent LLH carrier-reference trajectory",
                "-D": "per-channel code/carrier truth CSV",
                "-P": "deterministic independent initial carrier phase per PRN",
            },
        }

    def run(
        self,
        request: DecoupledSimulationRequest,
        output: str | Path,
        truth_csv: str | Path,
        log: str | Path,
    ) -> dict[str, Any]:
        validation = request.validate()
        output, truth_csv, log = Path(output), Path(truth_csv), Path(log)
        for path in (output, truth_csv, log):
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix="ccsim-", dir=output.parent))
        try:
            (stage / "nav.rnx").symlink_to(request.nav.resolve())
            shutil.copyfile(request.code_motion, stage / "code.csv")
            command = [
                str(self.executable), "-e", "nav.rnx", "-x", "code.csv",
            ]
            if request.mode == "doppler_locked":
                assert request.carrier_motion is not None
                shutil.copyfile(request.carrier_motion, stage / "carrier.csv")
                command.extend(["-X", "carrier.csv"])
            if request.carrier_phase_seed is not None:
                command.extend(["-P", str(request.carrier_phase_seed)])
            command.extend([
                "-t", validation["time"].simulator_input_calendar,
                "-d", str(request.duration_seconds),
                "-s", str(request.sample_rate_hz), "-b", "8",
                "-D", "truth.csv", "-o", "iq.bin",
            ])
            result = subprocess.run(
                command, cwd=stage, shell=False, text=True,
                capture_output=True, check=False,
            )
            log.write_text(
                f"time: {validation['time'].manifest()!r}\n"
                f"mode: {request.mode}\ncommand: {command!r}\n"
                f"exit_code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}\n",
                encoding="utf-8",
            )
            staged_iq, staged_truth = stage / "iq.bin", stage / "truth.csv"
            if result.returncode:
                raise SimulatorError(f"decoupled gps-sdr-sim exit code {result.returncode}; see {log}")
            if not staged_iq.is_file() or not staged_truth.is_file():
                raise SimulatorError("decoupled simulator did not emit both IQ and truth outputs")
            expected = self.expected_output_bytes(request)
            if staged_iq.stat().st_size != expected:
                raise SimulatorError(
                    f"decoupled simulator byte contract failed: {staged_iq.stat().st_size} != {expected}"
                )
            staged_iq.replace(output)
            staged_truth.replace(truth_csv)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return {
            "mode": request.mode,
            "carrier_phase_seed": request.carrier_phase_seed,
            "command": command,
            "time": validation["time"].manifest(),
            "inputs": {
                "nav_sha256": sha256(request.nav),
                "code_motion_sha256": sha256(request.code_motion),
                "carrier_motion_sha256": sha256(request.carrier_motion) if request.carrier_motion else None,
            },
            "iq": {"path": str(output.resolve()), "bytes": output.stat().st_size, "sha256": sha256(output)},
            "truth": {"path": str(truth_csv.resolve()), "rows": sum(1 for _ in truth_csv.open()) - 1, "sha256": sha256(truth_csv)},
            "simulator": self.probe(),
        }


def load_truth(path: str | Path) -> dict[tuple[float, int], dict[str, float]]:
    with Path(path).open(newline="", encoding="ascii") as stream:
        return {
            (float(row["time_s"]), int(row["prn"])): {
                key: float(value) for key, value in row.items() if key not in {"time_s", "prn"}
            }
            for row in csv.DictReader(stream)
        }


def summarize_truth_triplet(
    authentic_path: str | Path,
    coupled_path: str | Path,
    locked_path: str | Path,
    *,
    hold_start_seconds: float,
) -> dict[str, Any]:
    authentic = load_truth(authentic_path)
    coupled = load_truth(coupled_path)
    locked = load_truth(locked_path)
    keys = sorted(set(authentic) & set(coupled) & set(locked))
    if not keys:
        raise ValueError("truth triplet has no common PRN-time observations")
    hold = [key for key in keys if key[0] >= hold_start_seconds]
    if not hold:
        raise ValueError("truth triplet has no observations in the hold interval")

    def max_diff(left, right, field, subset=keys):
        return max(abs(left[key][field] - right[key][field]) for key in subset)

    offsets = sorted(abs(locked[key]["relative_code_range_m"]) for key in hold)
    return {
        "common_rows": len(keys),
        "common_prns": len({key[1] for key in keys}),
        "hold_rows": len(hold),
        "locked_vs_coupled_code_range_max_abs_m": max_diff(locked, coupled, "code_range_m"),
        "locked_vs_coupled_code_rate_max_abs_mps": max_diff(locked, coupled, "code_rate_mps"),
        "locked_vs_authentic_carrier_range_max_abs_m": max_diff(locked, authentic, "carrier_range_m"),
        "locked_vs_authentic_carrier_rate_max_abs_mps": max_diff(locked, authentic, "carrier_rate_mps"),
        "coupled_code_vs_carrier_hold_max_abs_m": max(abs(coupled[key]["relative_code_range_m"]) for key in hold),
        "locked_code_vs_carrier_hold_median_abs_m": offsets[len(offsets) // 2],
        "locked_code_vs_carrier_hold_max_abs_m": max(offsets),
        "coupled_vs_locked_carrier_doppler_max_abs_hz": max_diff(coupled, locked, "carrier_doppler_hz"),
    }
