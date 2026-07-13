"""Defensive, file-based GPS L1 C/A spoofing experiment primitives.

This module never transmits RF.  It builds reproducible baseband scenarios by
combining authentic and counterfeit GPS-SDR-SIM IQ ensembles for offline or
shielded/cabled receiver evaluation.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import hashlib
import json
import math
import os
import tempfile

import numpy as np


class SpoofingError(ValueError):
    """Invalid spoofing scenario or incompatible IQ/RINEX input."""


def _atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def filter_rinex2_nav(source: str | Path, output: str | Path, prns) -> dict:
    """Write a RINEX-2 GPS NAV snapshot containing only ``prns``.

    A GPS-SDR-SIM run over this snapshot naturally emits only selected PRNs
    that are visible at the simulated receiver state.  The fixed eight-line
    GPS NAV record contract is validated rather than heuristically edited.
    """
    source, output = Path(source), Path(output)
    selected = tuple(int(value) for value in prns)
    if not selected:
        raise SpoofingError("explicit PRN selection must not be empty")
    if len(set(selected)) != len(selected):
        raise SpoofingError("explicit PRN selection contains duplicate values")
    if any(value < 1 or value > 32 for value in selected):
        raise SpoofingError("GPS L1 C/A PRNs must be in [1, 32]")
    try:
        lines = source.read_text(encoding="ascii", errors="strict").splitlines(keepends=True)
    except (OSError, UnicodeError) as exc:
        raise SpoofingError(f"cannot read RINEX NAV: {exc}") from exc
    if not lines or not lines[0].lstrip().startswith("2."):
        raise SpoofingError("PRN filtering requires a RINEX-2 GPS NAV file")
    try:
        end = next(i for i, line in enumerate(lines) if len(line) >= 60 and line[60:].strip() == "END OF HEADER")
    except StopIteration as exc:
        raise SpoofingError("RINEX NAV header has no END OF HEADER record") from exc
    body = lines[end + 1 :]
    if len(body) % 8:
        raise SpoofingError("RINEX-2 GPS NAV body must contain complete eight-line records")
    records = []
    available = []
    for index in range(0, len(body), 8):
        record = body[index : index + 8]
        try:
            prn = int(record[0][:2])
        except (ValueError, IndexError) as exc:
            raise SpoofingError(f"invalid GPS PRN at NAV body line {index + 1}") from exc
        if not 1 <= prn <= 32:
            raise SpoofingError(f"invalid GPS PRN {prn} in RINEX-2 NAV")
        available.append(prn)
        if prn in selected:
            records.extend(record)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise SpoofingError(f"selected PRNs not present in RINEX NAV: {missing}")
    _atomic_write(output, "".join(lines[: end + 1] + records).encode("ascii"))
    return {"available_prns": sorted(set(available)), "selected_prns": sorted(selected)}


def _finite(name: str, value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SpoofingError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise SpoofingError(f"{name} must be finite")
    return number


def _progress(time_s: float, attack_type: str, start_s: float, transition_s: float) -> float:
    if time_s < start_s:
        return 0.0
    if attack_type == "abrupt":
        return 1.0
    if time_s <= start_s:
        return 0.0
    x = min(1.0, (time_s - start_s) / transition_s)
    return x * x * (3.0 - 2.0 * x)  # C1 smoothstep carry-off


def build_spoofing_rows(
    authentic_rows,
    *,
    coordinate_system: str,
    attack_type: str,
    start_seconds: float,
    transition_seconds: float,
    target_offset_enu_m,
):
    """Create a phase-continuity-friendly false receiver path.

    Before attack onset the counterfeit source follows the authentic receiver
    exactly.  A carry-off then applies a smooth local-ENU displacement, while
    an abrupt attack applies it at onset.  GPS-SDR-SIM can therefore generate
    both ensembles from the same epoch and ephemeris without an artificial
    trajectory jump before the configured attack.
    """
    from .trajectory import ecef_to_llh, enu_to_llh, llh_to_ecef

    coordinate_system = str(coordinate_system).lower()
    if coordinate_system not in {"llh", "ecef"}:
        raise SpoofingError("coordinate_system must be llh or ecef")
    if attack_type not in {"abrupt", "carry_off"}:
        raise SpoofingError("attack_type must be abrupt or carry_off")
    start = _finite("start_seconds", start_seconds)
    transition = _finite("transition_seconds", transition_seconds)
    if start < 0:
        raise SpoofingError("start_seconds must be non-negative")
    if transition < 0 or (attack_type == "carry_off" and transition <= 0):
        raise SpoofingError("transition_seconds must be positive for carry_off and non-negative otherwise")
    try:
        east, north, up = tuple(_finite("target_offset_enu_m", x) for x in target_offset_enu_m)
    except (TypeError, ValueError) as exc:
        raise SpoofingError("target_offset_enu_m must contain exactly three finite values") from exc
    if len(tuple(target_offset_enu_m)) != 3:
        raise SpoofingError("target_offset_enu_m must contain exactly three finite values")
    if east == north == up == 0:
        raise SpoofingError("target_offset_enu_m must be non-zero")

    result = []
    for row in authentic_rows:
        if len(row) != 4:
            raise SpoofingError("authentic trajectory rows must contain time and three coordinates")
        time_s = _finite("trajectory time", row[0])
        if coordinate_system == "llh":
            lat, lon, altitude = map(float, row[1:])
        else:
            lat, lon, altitude = ecef_to_llh(*map(float, row[1:]))
        fraction = _progress(time_s, attack_type, start, transition)
        if fraction == 0.0:
            result.append(tuple(row))
            continue
        false_llh = enu_to_llh(east * fraction, north * fraction, up * fraction, lat, lon, altitude)
        coordinates = false_llh if coordinate_system == "llh" else llh_to_ecef(*false_llh)
        result.append((time_s, *coordinates))
    if not result:
        raise SpoofingError("authentic trajectory must not be empty")
    if start >= result[-1][0] + 0.1:
        raise SpoofingError("start_seconds must occur within the generated IQ duration")
    return tuple(result)


def power_envelope(
    *,
    sample_count: int,
    sample_rate_hz: int,
    start_seconds: float,
    ramp_seconds: float,
    initial_advantage_db: float,
    final_advantage_db: float,
    start_sample: int = 0,
) -> np.ndarray:
    """Return counterfeit/authentic *amplitude* ratio per complex sample."""
    if sample_count < 0 or sample_rate_hz <= 0 or start_sample < 0:
        raise SpoofingError("sample counts and sample rate must be non-negative with a positive rate")
    start = _finite("start_seconds", start_seconds)
    ramp = _finite("ramp_seconds", ramp_seconds)
    initial = _finite("initial_advantage_db", initial_advantage_db)
    final = _finite("final_advantage_db", final_advantage_db)
    if start < 0 or ramp < 0:
        raise SpoofingError("power timing must be non-negative")
    initial_ratio = 10.0 ** (initial / 20.0)
    final_ratio = 10.0 ** (final / 20.0)
    times = (np.arange(sample_count, dtype=np.float64) + start_sample) / sample_rate_hz
    result = np.zeros(sample_count, dtype=np.float64)
    active = times >= start
    if ramp == 0:
        result[active] = final_ratio
        return result
    fraction = np.clip((times[active] - start) / ramp, 0.0, 1.0)
    result[active] = initial_ratio + fraction * (final_ratio - initial_ratio)
    return result


def mix_iq_files(
    authentic_path: str | Path,
    spoofing_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate_hz: int,
    start_seconds: float,
    ramp_seconds: float,
    initial_advantage_db: float,
    final_advantage_db: float,
    fixed_scale: float | None = None,
    chunk_complex_samples: int = 1_000_000,
) -> dict:
    """Stream two signed-IQ8 ensembles into one composite baseband recording."""
    authentic_path, spoofing_path, output_path = map(Path, (authentic_path, spoofing_path, output_path))
    size_a, size_s = authentic_path.stat().st_size, spoofing_path.stat().st_size
    if size_a != size_s:
        raise SpoofingError("authentic and spoofing IQ files must have the same byte length")
    if size_a % 2:
        raise SpoofingError("signed interleaved IQ source byte count must be even")
    if chunk_complex_samples <= 0:
        raise SpoofingError("chunk_complex_samples must be positive")
    max_ratio = max(10.0 ** (_finite("initial_advantage_db", initial_advantage_db) / 20.0),
                    10.0 ** (_finite("final_advantage_db", final_advantage_db) / 20.0))
    scale = 1.0 / (1.0 + max_ratio) if fixed_scale is None else _finite("fixed_scale", fixed_scale)
    if not 0 < scale <= 1:
        raise SpoofingError("fixed_scale must be in (0, 1]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    clipped = 0
    sample_index = 0
    try:
        with authentic_path.open("rb") as auth, spoofing_path.open("rb") as spoof, os.fdopen(fd, "wb") as out:
            while True:
                raw_a = auth.read(chunk_complex_samples * 2)
                raw_s = spoof.read(chunk_complex_samples * 2)
                if not raw_a and not raw_s:
                    break
                if len(raw_a) != len(raw_s) or len(raw_a) % 2:
                    raise SpoofingError("IQ sources changed or became misaligned during mixing")
                a = np.frombuffer(raw_a, dtype=np.int8).astype(np.float32).reshape(-1, 2)
                s = np.frombuffer(raw_s, dtype=np.int8).astype(np.float32).reshape(-1, 2)
                envelope = power_envelope(
                    sample_count=len(a), sample_rate_hz=sample_rate_hz,
                    start_seconds=start_seconds, ramp_seconds=ramp_seconds,
                    initial_advantage_db=initial_advantage_db,
                    final_advantage_db=final_advantage_db, start_sample=sample_index,
                )
                mixed = scale * (a + s * envelope[:, None])
                clipped += int(np.count_nonzero((mixed < -128.0) | (mixed > 127.0)))
                np.clip(np.rint(mixed), -128, 127).astype(np.int8).tofile(out)
                sample_index += len(a)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, output_path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {
        "complex_samples": sample_index,
        "actual_bytes": output_path.stat().st_size,
        "clipped_components": clipped,
        "fixed_scale": scale,
        "normalization": "fixed_headroom",
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authentic_rows(config):
    """Return the authentic receiver truth as strict 10 Hz rows."""
    from .rf_config import StaticPosition, TrajectoryPosition

    duration = config.scenario.duration_seconds
    position = config.scenario.position
    if isinstance(position, TrajectoryPosition):
        return tuple(position.rows), position.coordinate_system
    if isinstance(position, StaticPosition):
        rows = tuple(
            (index / 10.0, position.latitude_deg, position.longitude_deg, position.altitude_m)
            for index in range(duration * 10)
        )
        return rows, "llh"
    raise SpoofingError(f"unsupported authentic position type: {type(position).__name__}")


def _write_spoofing_trajectory(path: Path, rows, coordinate_system: str, config) -> dict:
    csv_bytes = "".join(
        f"{time_s:.1f},{a:.9f},{b:.9f},{c:.4f}\n" for time_s, a, b, c in rows
    ).encode("ascii")
    _atomic_write(path, csv_bytes)
    digest = hashlib.sha256(csv_bytes).hexdigest()
    sidecar = path.with_suffix(".json")
    metadata = {
        "generator": {"schema": "gnss-doppler-lab.spoofing-trajectory", "version": 1},
        "coordinate_system": coordinate_system,
        "sample_rate_hz": 10,
        "actual_row_count": len(rows),
        "actual_start_time_s": rows[0][0],
        "actual_end_time_s": rows[-1][0],
        "csv_sha256": digest,
        "attack": {
            "type": config.spoofing.attack_type,
            "start_seconds": config.spoofing.start_seconds,
            "transition_seconds": config.spoofing.transition_seconds,
            "target_offset_enu_m": list(config.spoofing.target_offset_enu_m),
        },
    }
    _atomic_write(sidecar, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    return {"csv_sha256": digest, "metadata_path": sidecar, "metadata_sha256": _sha256(sidecar)}


def _iq_record(path: Path, *, sample_rate_hz: int, retained: bool = True) -> dict:
    size = path.stat().st_size
    return {
        "path": path.name if retained else None,
        "retained": retained,
        "sample_format": "s8_iq",
        "rf_sample_rate_hz": sample_rate_hz,
        "actual_bytes": size,
        "complex_samples": size // 2,
        "sha256": _sha256(path),
    }


def generate_spoofing_iq(config, runner) -> Path:
    """Generate authentic and counterfeit ensembles and mix them at baseband.

    The counterfeit receiver path is identical to truth before attack onset and
    then follows an abrupt or smooth carry-off ENU displacement.  ``explicit``
    PRN groups are enforced by a filtered immutable RINEX-2 NAV snapshot;
    ``all_visible`` delegates visibility allocation to the pinned GPS-SDR-SIM.
    """
    if config.version != 2 or config.spoofing is None:
        raise SpoofingError("generate_spoofing_iq requires a version-2 spoofing configuration")
    scenario, attack = config.scenario, config.spoofing
    run_id = f"{scenario.name}_{scenario.utc.strftime('%Y%m%dT%H%M%SZ')}"
    output_root = config.output.root.resolve()
    run_dir = (output_root / run_id).resolve()
    if run_dir.parent != output_root:
        raise SpoofingError("run directory must remain directly under output root")
    run_dir.mkdir(parents=True, exist_ok=False)

    authentic_iq = run_dir / "authentic_gps_l1ca_s8_iq.bin"
    counterfeit_iq = run_dir / "counterfeit_gps_l1ca_s8_iq.bin"
    composite_iq = run_dir / "composite_gps_l1ca_s8_iq.bin"
    authentic_log = run_dir / "authentic-gps-sdr-sim.log"
    counterfeit_log = run_dir / "counterfeit-gps-sdr-sim.log"
    trajectory_path = run_dir / "spoofing_trajectory.csv"

    authentic_rows, coordinate_system = _authentic_rows(config)
    counterfeit_rows = build_spoofing_rows(
        authentic_rows,
        coordinate_system=coordinate_system,
        attack_type=attack.attack_type,
        start_seconds=attack.start_seconds,
        transition_seconds=attack.transition_seconds,
        target_offset_enu_m=attack.target_offset_enu_m,
    )
    trajectory_meta = _write_spoofing_trajectory(
        trajectory_path, counterfeit_rows, coordinate_system, config
    )

    from .rf_config import InputConfig, TrajectoryPosition

    spoof_position = TrajectoryPosition(
        trajectory_path,
        coordinate_system,
        tuple(counterfeit_rows),
        trajectory_meta["csv_sha256"],
        trajectory_meta["metadata_path"],
        trajectory_meta["metadata_sha256"],
    )
    filtered_nav = None
    filter_report = None
    spoof_input = config.input
    if attack.prn_selection.mode == "explicit":
        filtered_nav = run_dir / "counterfeit-selected-prns.nav"
        filter_report = filter_rinex2_nav(
            config.input.rinex_nav, filtered_nav, attack.prn_selection.prns
        )
        spoof_input = InputConfig(filtered_nav)
    counterfeit_config = replace(
        config,
        scenario=replace(scenario, position=spoof_position),
        input=spoof_input,
    )

    authentic_result = runner.run(config, authentic_iq, authentic_log)
    counterfeit_result = runner.run(
        counterfeit_config, counterfeit_iq, counterfeit_log
    )
    if authentic_iq.stat().st_size != counterfeit_iq.stat().st_size:
        raise SpoofingError("simulator ensembles have unequal lengths")
    mix_report = mix_iq_files(
        authentic_iq,
        counterfeit_iq,
        composite_iq,
        sample_rate_hz=config.output.rf_sample_rate_hz,
        start_seconds=attack.start_seconds,
        ramp_seconds=attack.power.ramp_seconds,
        initial_advantage_db=attack.power.initial_advantage_db,
        final_advantage_db=attack.power.final_advantage_db,
    )

    component_retained = attack.keep_component_iq
    authentic_record = _iq_record(
        authentic_iq, sample_rate_hz=config.output.rf_sample_rate_hz,
        retained=component_retained,
    )
    counterfeit_record = _iq_record(
        counterfeit_iq, sample_rate_hz=config.output.rf_sample_rate_hz,
        retained=component_retained,
    )
    composite_record = _iq_record(
        composite_iq, sample_rate_hz=config.output.rf_sample_rate_hz,
    )
    manifest = {
        "schema": {"name": "gnss-doppler-lab.spoofing", "version": 1},
        "run_id": run_id,
        "scenario": {
            "name": scenario.name,
            "utc": scenario.utc.isoformat().replace("+00:00", "Z"),
            "duration_seconds": scenario.duration_seconds,
            "constellation": scenario.constellation,
            "signal": scenario.signal,
        },
        "attack": {
            "type": attack.attack_type,
            "start_seconds": attack.start_seconds,
            "transition_seconds": attack.transition_seconds,
            "target_offset_enu_m": list(attack.target_offset_enu_m),
            "prn_selection": {
                "mode": attack.prn_selection.mode,
                "prns": list(attack.prn_selection.prns),
            },
            "power": {
                "initial_advantage_db": attack.power.initial_advantage_db,
                "final_advantage_db": attack.power.final_advantage_db,
                "ramp_seconds": attack.power.ramp_seconds,
                **mix_report,
            },
        },
        "input": {
            "rinex_nav": str(config.input.rinex_nav),
            "rinex_nav_sha256": _sha256(config.input.rinex_nav),
        },
        "truth": {
            "spoofing_trajectory_path": trajectory_path.name,
            "spoofing_trajectory_sha256": trajectory_meta["csv_sha256"],
            "spoofing_trajectory_metadata_path": trajectory_meta["metadata_path"].name,
            "spoofing_trajectory_metadata_sha256": trajectory_meta["metadata_sha256"],
            "filtered_nav_path": filtered_nav.name if filtered_nav else None,
            "filtered_nav_sha256": _sha256(filtered_nav) if filtered_nav else None,
            "prn_filter": filter_report,
        },
        "iq": {
            "authentic": authentic_record,
            "counterfeit": counterfeit_record,
            "composite": composite_record,
        },
        "simulator": {
            "identity": getattr(runner, "identity", "unknown"),
            "executable": getattr(runner, "executable", None),
            "provenance": getattr(runner, "provenance", "unverified"),
            "cli_contract": getattr(runner, "cli_contract", None),
            "authentic_command": authentic_result["command"],
            "counterfeit_command": counterfeit_result["command"],
            "authentic_log": authentic_log.name,
            "counterfeit_log": counterfeit_log.name,
        },
        "scope": {
            "rf_transmission": False,
            "mode": "offline baseband or shielded/cabled receiver evaluation only",
            "claim": "practical configurable multi-PRN RF/IQ superposition; not a fully receiver-aware perfect spoofer",
        },
    }
    manifest_path = run_dir / "manifest.json"
    _atomic_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    if not component_retained:
        authentic_iq.unlink()
        counterfeit_iq.unlink()
    return manifest_path
