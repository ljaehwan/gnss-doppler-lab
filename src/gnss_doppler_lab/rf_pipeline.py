"""Reproducible IQ generation orchestration and immutable run manifest."""
from pathlib import Path
import hashlib, json


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _position_manifest(position, result):
    from .rf_config import StaticPosition, TrajectoryPosition

    if isinstance(position, TrajectoryPosition):
        snapshot_digest = result.get("trajectory_sha256", position.csv_sha256)
        if snapshot_digest != position.csv_sha256:
            raise ValueError("runner trajectory snapshot does not match validated trajectory")
        return {
            "type": "trajectory",
            "path": str(position.path),
            "coordinate_system": position.coordinate_system,
            "sha256": snapshot_digest,
            "sample_rate_hz": 10,
            "consumed_row_count": len(position.rows),
            "consumed_start_time_s": position.rows[0][0],
            "consumed_end_time_s": position.rows[-1][0],
            "metadata_path": str(position.metadata_path) if position.metadata_path else None,
            "metadata_sha256": position.metadata_sha256,
        }
    if isinstance(position, StaticPosition):
        return {
            "type": "static",
            "latitude_deg": position.latitude_deg,
            "longitude_deg": position.longitude_deg,
            "altitude_m": position.altitude_m,
        }
    raise TypeError(f"unsupported position type: {type(position).__name__}")


def generate_iq(config, runner) -> Path:
    s = config.scenario
    run_id = f"{s.name}_{s.utc.strftime('%Y%m%dT%H%M%SZ')}"
    output_root = config.output.root.resolve()
    run_dir = (output_root / run_id).resolve()
    if run_dir.parent != output_root:
        raise ValueError("run directory must remain directly under output root")
    run_dir.mkdir(parents=True, exist_ok=False)
    iq, log = run_dir / "gps_l1ca_s8_iq.bin", run_dir / "gps-sdr-sim.log"
    try:
        result = runner.run(config, iq, log)
        # Derive independently as well as accepting the runner copy, so every
        # manifest (including alternate runners) has unambiguous time semantics.
        from .gps_sdr_sim import simulator_time
        expected_time = simulator_time(s.utc, config.input.rinex_nav).manifest()
        time_metadata = result.get("time", expected_time)
        if time_metadata != expected_time:
            raise ValueError("runner time metadata does not match scenario UTC and RINEX header")
        actual_bytes = iq.stat().st_size
        if actual_bytes % 2:
            raise ValueError("s8 interleaved IQ byte size must be even")
        complex_samples = actual_bytes // 2
        actual_duration_seconds = complex_samples / config.output.rf_sample_rate_hz
        manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "scenario": {
                "name": s.name,
                "constellation": s.constellation,
                "signal": s.signal,
                "utc": s.utc.isoformat().replace("+00:00", "Z"),
                "time": time_metadata,
                "duration_seconds": s.duration_seconds,
                "position": _position_manifest(s.position, result),
            },
            "input": {
                "rinex_nav": str(config.input.rinex_nav),
                "rinex_nav_sha256": _sha256(config.input.rinex_nav),
            },
            "iq": {
                "path": iq.name,
                "rf_sample_rate_hz": config.output.rf_sample_rate_hz,
                "sample_format": config.output.sample_format,
                "channels": 2,
                "actual_bytes": actual_bytes,
                "complex_samples": complex_samples,
                "actual_duration_seconds": actual_duration_seconds,
                "sha256": _sha256(iq),
            },
            "simulator": {
                "identity": runner.identity,
                "executable": runner.executable,
                "executable_sha256": (
                    _sha256(Path(runner.executable))
                    if Path(runner.executable).is_file() else None
                ),
                "provenance": getattr(runner, "provenance", "unverified"),
                "cli_contract": getattr(runner, "cli_contract", None),
                "command": result["command"],
                "log": log.name,
            },
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest_path
    except Exception:
        # Preserve logs for diagnosis, but never publish a manifest for an incomplete run.
        raise
