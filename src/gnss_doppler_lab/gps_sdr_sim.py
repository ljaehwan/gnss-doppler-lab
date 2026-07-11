"""Adapter pinned to osqzss/gps-sdr-sim's documented command-line contract."""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib, os, shutil, subprocess, tempfile

class SimulatorError(RuntimeError): pass

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SimulatorTime:
    requested_utc: str
    simulator_input_calendar: str
    simulator_input_time_scale: str
    gps_minus_utc_seconds: int
    gps_week: int
    gps_tow_seconds: int
    leap_seconds_source: dict

    def manifest(self):
        return asdict(self)


def parse_rinex_leap_seconds(nav: str | Path) -> tuple[int, str]:
    """Read the single fixed-column RINEX ``LEAP SECONDS`` header record.

    Fail closed: guessing an offset would silently shift every generated sample.
    """
    path = Path(nav)
    try:
        lines = path.read_text(encoding="ascii", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SimulatorError(f"cannot read ASCII RINEX NAV header: {path}: {exc}") from exc
    records = []
    ended = False
    for number, line in enumerate(lines, 1):
        if len(line) < 60:
            raise SimulatorError(f"malformed RINEX NAV header line {number}: shorter than 60 columns")
        label = line[60:].strip()
        if label == "LEAP SECONDS":
            field = line[:6]
            if not field.strip() or not field.strip().lstrip("+-").isdigit():
                raise SimulatorError(f"malformed LEAP SECONDS value on RINEX header line {number}")
            value = int(field)
            if not 0 <= value <= 99:
                raise SimulatorError(f"invalid LEAP SECONDS value {value} on RINEX header line {number}")
            records.append((value, line))
        if label == "END OF HEADER":
            ended = True
            break
    if not ended:
        raise SimulatorError("RINEX NAV header has no END OF HEADER record")
    if len(records) != 1:
        detail = "missing" if not records else "duplicate"
        raise SimulatorError(f"RINEX NAV header has {detail} LEAP SECONDS record; refusing to guess GPS-UTC")
    return records[0]


def simulator_time(requested_utc: datetime, nav: str | Path) -> SimulatorTime:
    if requested_utc.tzinfo is None:
        raise SimulatorError("requested scenario UTC must be timezone-aware")
    requested_utc = requested_utc.astimezone(timezone.utc)
    leap, header = parse_rinex_leap_seconds(nav)
    gpst = requested_utc + timedelta(seconds=leap)
    if gpst < GPS_EPOCH:
        raise SimulatorError("scenario epoch predates the GPS epoch; LEAP SECONDS validity cannot be established")
    elapsed = int((gpst - GPS_EPOCH).total_seconds())
    week, tow = divmod(elapsed, 7 * 86400)
    path = Path(nav).resolve()
    return SimulatorTime(
        requested_utc.isoformat().replace("+00:00", "Z"),
        gpst.strftime("%Y/%m/%d,%H:%M:%S"), "GPST", leap, week, tow,
        {"nav_path": str(path), "nav_sha256": _sha256(path), "header_record": header},
    )


class GpsSdrSimRunner:
    upstream = "https://github.com/osqzss/gps-sdr-sim"
    pinned_commit = "28ca29a6719475195e3aabd5930c4ed02d67190f"

    def __init__(self, executable: str | None = None):
        candidate = executable or os.environ.get("GPS_SDR_SIM") or shutil.which("gps-sdr-sim")
        if not candidate: raise SimulatorError("gps-sdr-sim not found; use explicit path, GPS_SDR_SIM, or PATH")
        resolved = shutil.which(candidate) if "/" not in candidate else None
        self.executable = str(Path(resolved or candidate).resolve()) if resolved or "/" in candidate else candidate
        self.identity = "unverified gps-sdr-sim executable"; self.provenance = "unverified"
        self.cli_contract = f"osqzss/gps-sdr-sim@{self.pinned_commit}"

    def build_command(self, config, output: Path, nav: str | Path | None = None, time=None) -> list[str]:
        s, p = config.scenario, config.scenario.position
        from .rf_config import TrajectoryPosition
        location = (["-x" if p.coordinate_system == "llh" else "-u", "motion.csv"] if isinstance(p, TrajectoryPosition) else ["-l", f"{p.latitude_deg:.8f},{p.longitude_deg:.8f},{p.altitude_m:.3f}"])
        time = time or simulator_time(s.utc, config.input.rinex_nav)
        return [self.executable, "-e", str(nav if nav is not None else config.input.rinex_nav), *location,
                "-t", time.simulator_input_calendar, "-d", str(s.duration_seconds),
                "-s", str(config.output.rf_sample_rate_hz), "-b", "8", "-o", str(output)]

    def probe(self) -> dict:
        resolved = shutil.which(self.executable) if "/" not in self.executable else self.executable; path = Path(resolved) if resolved else None; digest = None
        if path and path.is_file():
            try: digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError: pass
        return {"available": bool(path and path.is_file() and os.access(path, os.X_OK)), "executable": str(path) if path else self.executable, "executable_sha256": digest, "provenance": self.provenance, "cli_contract": self.cli_contract}

    def run(self, config, output: Path, log: Path) -> dict:
        if not config.input.rinex_nav.is_file(): raise SimulatorError(f"RINEX NAV input does not exist: {config.input.rinex_nav}")
        try: rinex_version = float(config.input.rinex_nav.open("r", encoding="ascii", errors="replace").readline()[:9])
        except (OSError, ValueError) as exc: raise SimulatorError(f"invalid RINEX NAV header: {config.input.rinex_nav}") from exc
        if not 2.0 <= rinex_version < 3.0: raise SimulatorError("the pinned gps-sdr-sim parser requires a GPS RINEX 2 NAV file; " f"received RINEX {rinex_version:.2f}")
        time = simulator_time(config.scenario.utc, config.input.rinex_nav)
        output.parent.mkdir(parents=True, exist_ok=True); log.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent); os.close(fd); tmp = Path(tmp_name); tmp.unlink()
        staged_nav = output.parent / "nav.rnx"; staged_motion = output.parent / "motion.csv"; nav_created = motion_created = False
        try:
            try: staged_nav.symlink_to(config.input.rinex_nav); nav_created = True
            except OSError as exc: raise SimulatorError(f"could not stage RINEX NAV input: {exc}") from exc
            from .rf_config import TrajectoryPosition
            if isinstance(config.scenario.position, TrajectoryPosition):
                position = config.scenario.position
                if os.path.lexists(staged_motion): raise SimulatorError("motion.csv staging collision")
                try:
                    source_digest = _sha256(position.path)
                    if source_digest != position.csv_sha256: raise SimulatorError("trajectory CSV changed after configuration validation")
                    shutil.copyfile(position.path, staged_motion); motion_created = True; staged_digest = _sha256(staged_motion)
                    if staged_digest != position.csv_sha256: raise SimulatorError("staged motion.csv failed integrity verification")
                except SimulatorError: motion_created = os.path.lexists(staged_motion); raise
                except OSError as exc: motion_created = os.path.lexists(staged_motion); raise SimulatorError(f"could not stage motion.csv: {exc}") from exc
            command = self.build_command(config, Path(tmp.name), staged_nav.name, time)
            try:
                result = subprocess.run(command, shell=False, text=True, capture_output=True, check=False, cwd=output.parent)
                log.write_text(f"time: {time.manifest()!r}\ncommand: {command!r}\nexit_code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}\n")
                if result.returncode: raise SimulatorError(f"gps-sdr-sim exit code {result.returncode}; see {log}")
                if not tmp.is_file() or tmp.stat().st_size == 0: raise SimulatorError("gps-sdr-sim produced empty output")
                size = tmp.stat().st_size; tmp.replace(output)
                run_result = {"command": command, "actual_bytes": size, "time": time.manifest()}
                if isinstance(config.scenario.position, TrajectoryPosition): run_result["trajectory_sha256"] = staged_digest
                return run_result
            except OSError as exc: raise SimulatorError(f"could not execute gps-sdr-sim: {exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)
            if nav_created: staged_nav.unlink(missing_ok=True)
            if motion_created: staged_motion.unlink(missing_ok=True)
