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
    clean_iq = run_dir / ".gps_l1ca_clean_s8_iq.bin"
    impairment_report = None
    try:
        simulator_output = clean_iq if config.impairments.enabled else iq
        result = runner.run(config, simulator_output, log)
        requested_bytes = s.duration_seconds * config.output.rf_sample_rate_hz * 2
        expected_bytes_fn = getattr(runner, "expected_output_bytes", None)
        expected_bytes = (
            expected_bytes_fn(config) if callable(expected_bytes_fn) else requested_bytes
        )
        expected_bytes_source = (
            "runner_contract" if callable(expected_bytes_fn) else "requested_duration"
        )
        simulator_bytes = simulator_output.stat().st_size
        reported_bytes = result.get("actual_bytes")
        if simulator_bytes % 2:
            raise ValueError("s8 interleaved IQ byte size must be even")
        if reported_bytes != simulator_bytes:
            raise ValueError(
                f"simulator reported {reported_bytes} bytes but filesystem has {simulator_bytes} bytes"
            )
        if simulator_bytes != expected_bytes:
            raise ValueError(
                f"simulator recording size {simulator_bytes} does not match expected {expected_bytes} bytes"
            )
        if config.impairments.enabled:
            from .rf_impairments import apply_impairments
            impairment_report = apply_impairments(
                clean_iq, iq, config.output.rf_sample_rate_hz, config.impairments
            )
        # Derive independently as well as accepting the runner copy, so every
        # manifest (including alternate runners) has unambiguous time semantics.
        from .gps_sdr_sim import simulator_time
        expected_time = simulator_time(s.utc, config.input.rinex_nav).manifest()
        time_metadata = result.get("time", expected_time)
        if time_metadata != expected_time:
            raise ValueError("runner time metadata does not match scenario UTC and RINEX header")
        actual_bytes = iq.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(f"final IQ size {actual_bytes} does not match expected {expected_bytes} bytes")
        if impairment_report is not None:
            impaired_output = impairment_report["output"]
            if impaired_output["bytes"] != actual_bytes or impaired_output["complex_samples"] * 2 != actual_bytes:
                raise ValueError("impairment output counts do not match final filesystem size")
            iq_sha256 = impaired_output["sha256"]
        else:
            iq_sha256 = _sha256(iq)
        complex_samples = actual_bytes // 2
        actual_duration_seconds = complex_samples / config.output.rf_sample_rate_hz
        manifest = {
            "schema_version": 3 if impairment_report is not None else 2,
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
                "requested_duration_bytes": requested_bytes,
                "expected_bytes": expected_bytes,
                "expected_bytes_source": expected_bytes_source,
                "complex_samples": complex_samples,
                "expected_complex_samples": expected_bytes // 2,
                "actual_duration_seconds": actual_duration_seconds,
                "sha256": iq_sha256,
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
        if impairment_report is not None:
            manifest["impairments"] = impairment_report
        manifest_path = run_dir / "manifest.json"
        manifest_tmp = run_dir / ".manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_tmp.replace(manifest_path)
        return manifest_path
    except Exception:
        # Preserve logs for diagnosis, but never publish a manifest for an incomplete run.
        for incomplete in (run_dir / "manifest.json", run_dir / ".manifest.json.tmp"):
            incomplete.unlink(missing_ok=True)
        raise
    finally:
        # Clean simulator IQ is an implementation detail and can be hundreds of MB.
        clean_iq.unlink(missing_ok=True)
