"""Adapter pinned to osqzss/gps-sdr-sim's documented command-line contract."""
from pathlib import Path
import hashlib, os, shutil, subprocess, tempfile

class SimulatorError(RuntimeError): pass

class GpsSdrSimRunner:
    upstream = "https://github.com/osqzss/gps-sdr-sim"
    pinned_commit = "28ca29a6719475195e3aabd5930c4ed02d67190f"

    def __init__(self, executable: str | None = None):
        candidate = executable or os.environ.get("GPS_SDR_SIM") or shutil.which("gps-sdr-sim")
        if not candidate: raise SimulatorError("gps-sdr-sim not found; use explicit path, GPS_SDR_SIM, or PATH")
        resolved = shutil.which(candidate) if "/" not in candidate else None
        self.executable = str(Path(resolved or candidate).resolve()) if resolved or "/" in candidate else candidate
        self.identity = "unverified gps-sdr-sim executable"
        self.provenance = "unverified"
        self.cli_contract = f"osqzss/gps-sdr-sim@{self.pinned_commit}"

    def build_command(self, config, output: Path, nav: str | Path | None = None) -> list[str]:
        s, p = config.scenario, config.scenario.position
        return [self.executable, "-e", str(nav if nav is not None else config.input.rinex_nav), "-l", f"{p.latitude_deg:.8f},{p.longitude_deg:.8f},{p.altitude_m:.3f}",
                "-t", s.utc.strftime("%Y/%m/%d,%H:%M:%S"), "-d", str(s.duration_seconds),
                "-s", str(config.output.rf_sample_rate_hz), "-b", "8", "-o", str(output)]

    def probe(self) -> dict:
        resolved = shutil.which(self.executable) if "/" not in self.executable else self.executable
        path = Path(resolved) if resolved else None
        digest = None
        if path and path.is_file():
            try: digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError: pass
        return {"available": bool(path and path.is_file() and os.access(path, os.X_OK)), "executable": str(path) if path else self.executable, "executable_sha256": digest, "provenance": self.provenance, "cli_contract": self.cli_contract}

    def run(self, config, output: Path, log: Path) -> dict:
        if not config.input.rinex_nav.is_file(): raise SimulatorError(f"RINEX NAV input does not exist: {config.input.rinex_nav}")
        try:
            rinex_version = float(config.input.rinex_nav.open("r", encoding="ascii", errors="replace").readline()[:9])
        except (OSError, ValueError) as exc:
            raise SimulatorError(f"invalid RINEX NAV header: {config.input.rinex_nav}") from exc
        if not 2.0 <= rinex_version < 3.0:
            raise SimulatorError(
                "the pinned gps-sdr-sim parser requires a GPS RINEX 2 NAV file; "
                f"received RINEX {rinex_version:.2f}"
            )
        output.parent.mkdir(parents=True, exist_ok=True); log.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent); os.close(fd)
        tmp = Path(tmp_name); tmp.unlink()
        staged_nav = output.parent / "nav.rnx"
        try:
            staged_nav.symlink_to(config.input.rinex_nav)
        except OSError as exc:
            raise SimulatorError(f"could not stage RINEX NAV input: {exc}") from exc
        command = self.build_command(config, Path(tmp.name), staged_nav.name)
        try:
            result = subprocess.run(
                command,
                shell=False,
                text=True,
                capture_output=True,
                check=False,
                cwd=output.parent,
            )
            log.write_text(f"command: {command!r}\nexit_code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}\n")
            if result.returncode: raise SimulatorError(f"gps-sdr-sim exit code {result.returncode}; see {log}")
            if not tmp.is_file() or tmp.stat().st_size == 0: raise SimulatorError("gps-sdr-sim produced empty output")
            size = tmp.stat().st_size
            tmp.replace(output)
            return {"command": command, "actual_bytes": size}
        except OSError as exc: raise SimulatorError(f"could not execute gps-sdr-sim: {exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)
            staged_nav.unlink(missing_ok=True)
