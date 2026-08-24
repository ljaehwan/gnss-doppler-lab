"""Read-only clean-observable audit for the frozen SPLITCLOCK Stage-0A design."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .splitclock_stage0a import GALILEO_E1_WAVELENGTH_M, sha256_file


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def md5_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_output_manifest(root: Path) -> dict[str, Any]:
    """Recreate the QSET R2a aggregate without importing QSET score code."""

    frozen = json.loads((root / "manifest.json").read_text())
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    actual = {"files": rows, "file_count": len(rows), "aggregate_sha256": canonical_sha(rows)}
    return {
        "status": "PASS" if actual == frozen["output_set"] else "FAIL",
        "expected_aggregate_sha256": frozen["output_set"]["aggregate_sha256"],
        "actual_aggregate_sha256": actual["aggregate_sha256"],
        "file_count": len(rows),
        "manifest": frozen,
    }


def _field(text: str) -> float:
    value = text[:14].strip().replace("D", "E")
    return float(value) if value else float("nan")


def parse_galileo_rinex_observations(path: Path) -> list[dict[str, Any]]:
    """Parse only the frozen E:C1B/L1B/D1B/S1B RINEX 3 records."""

    lines = path.read_text(encoding="ascii").splitlines()
    header_end = next(i for i, line in enumerate(lines) if "END OF HEADER" in line)
    obs_header = [line for line in lines[:header_end] if "SYS / # / OBS TYPES" in line]
    if len(obs_header) != 1 or "E    4 C1B L1B D1B S1B" not in obs_header[0]:
        raise ValueError("RINEX observable contract is not E:C1B/L1B/D1B/S1B")
    rows: list[dict[str, Any]] = []
    epoch = None
    receiver_epoch = -1
    for line in lines[header_end + 1 :]:
        if line.startswith(">"):
            parts = line[1:].split()
            epoch = datetime(
                int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]),
                int(float(parts[5])), tzinfo=timezone.utc,
            )
            receiver_epoch += 1
        elif line.startswith("E") and epoch is not None:
            values = [_field(line[3 + 16 * index : 3 + 16 * (index + 1)]) for index in range(4)]
            indicators = [line[3 + 16 * index + 14 : 3 + 16 * index + 16] for index in range(4)]
            rows.append(
                {
                    "epoch": epoch,
                    "receiver_epoch": receiver_epoch,
                    "prn": int(line[1:3]),
                    "pseudorange_m": values[0],
                    "carrier_cycles": values[1],
                    "doppler_hz": values[2],
                    "cn0_db_hz": values[3],
                    "carrier_indicator": indicators[1].strip(),
                }
            )
    if not rows:
        raise ValueError("no Galileo observations")
    return rows


def parse_navigation_inventory(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="ascii").splitlines()
    end = next(i for i, line in enumerate(lines) if "END OF HEADER" in line)
    records = [line for line in lines[end + 1 :] if line.startswith("E")]
    prns = sorted({int(line[1:3]) for line in records})
    numeric_failures = 0
    for line in lines[end + 1 :]:
        if not line.strip():
            continue
        fields = [line[start : start + 19].strip().replace("D", "E") for start in range(23 if line.startswith("E") else 4, len(line), 19)]
        for value in fields:
            if value:
                try:
                    numeric_failures += int(not np.isfinite(float(value)))
                except ValueError:
                    numeric_failures += 1
    return {
        "record_count": len(records),
        "prns": prns,
        "finite_failure_count": numeric_failures,
        "status": "PASS" if records and numeric_failures == 0 else "FAIL",
    }


def panel_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    epochs: dict[datetime, set[int]] = {}
    for row in rows:
        if all(np.isfinite(row[name]) for name in ("pseudorange_m", "carrier_cycles", "doppler_hz")):
            epochs.setdefault(row["epoch"], set()).add(row["prn"])
    qualifying = sorted(epoch for epoch, prns in epochs.items() if len(prns) >= 5)
    longest = current = 0
    previous = None
    for epoch in qualifying:
        current = current + 1 if previous is not None and (epoch - previous).total_seconds() == 1 else 1
        longest = max(longest, current)
        previous = epoch
    return {
        "epoch_count": len(epochs),
        "m_ge_5_epoch_count": len(qualifying),
        "longest_continuous_m_ge_5_seconds": longest,
        "tracked_prns": sorted({row["prn"] for row in rows}),
        "maximum_panel_size": max((len(value) for value in epochs.values()), default=0),
        "finite_coverage": float(sum(len(v) >= 5 for v in epochs.values()) / max(1, len(epochs))),
        "status": "PASS" if longest >= 60 else "FAIL",
    }


def sign_unit_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_prn: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_prn.setdefault(row["prn"], []).append(row)
    delta_rho: list[float] = []
    native_carrier: list[float] = []
    frozen_carrier: list[float] = []
    doppler_range: list[float] = []
    for prn_rows in by_prn.values():
        prn_rows.sort(key=lambda value: value["epoch"])
        for left, right in zip(prn_rows, prn_rows[1:]):
            if (right["epoch"] - left["epoch"]).total_seconds() != 1:
                continue
            values = [left[name] for name in ("pseudorange_m", "carrier_cycles", "doppler_hz")]
            values += [right[name] for name in ("pseudorange_m", "carrier_cycles", "doppler_hz")]
            if not np.isfinite(values).all():
                continue
            delta_rho.append(right["pseudorange_m"] - left["pseudorange_m"])
            increment = GALILEO_E1_WAVELENGTH_M * (right["carrier_cycles"] - left["carrier_cycles"])
            native_carrier.append(increment)
            frozen_carrier.append(-increment)
            doppler_range.append(-GALILEO_E1_WAVELENGTH_M * 0.5 * (left["doppler_hz"] + right["doppler_hz"]))
    arrays = [np.asarray(value, dtype=float) for value in (delta_rho, native_carrier, frozen_carrier, doppler_range)]
    if min(map(len, arrays)) < 2:
        raise ValueError("insufficient consecutive RINEX pairs for sign audit")
    rho, native, frozen, rate = arrays
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    native_pass = corr(native, rate) >= 0.9 and corr(native, rho) >= 0.9
    frozen_pass = corr(frozen, rate) >= 0.9 and corr(frozen, rho) >= 0.9
    return {
        "pair_count": len(rho),
        "wavelength_m": GALILEO_E1_WAVELENGTH_M,
        "rinex_native_formula": "delta_range_m = +wavelength_m * delta(L1B_cycles)",
        "frozen_formula": "delta_range_m = -wavelength_m * delta(cycle_consistent_phase)",
        "correlation": {
            "pseudorange_delta_vs_doppler_range": corr(rho, rate),
            "rinex_native_carrier_vs_doppler_range": corr(native, rate),
            "frozen_carrier_vs_doppler_range": corr(frozen, rate),
            "rinex_native_carrier_vs_pseudorange_delta": corr(native, rho),
            "frozen_carrier_vs_pseudorange_delta": corr(frozen, rho),
        },
        "mean_absolute_error_m": {
            "pseudorange_delta_vs_doppler_range": float(np.mean(np.abs(rho - rate))),
            "rinex_native_carrier_vs_doppler_range": float(np.mean(np.abs(native - rate))),
            "frozen_carrier_vs_doppler_range": float(np.mean(np.abs(frozen - rate))),
        },
        "rinex_native_sign_status": "PASS" if native_pass else "FAIL",
        "frozen_contract_sign_status": "PASS" if frozen_pass else "FAIL",
        "status": "PASS" if frozen_pass else "FAIL",
    }


def first_path(root: Path, pattern: str) -> Path:
    values = sorted(root.glob(pattern))
    if len(values) != 1:
        raise ValueError(f"expected one {pattern} under {root}, got {len(values)}")
    return values[0]


def artifact_manifest(artifact: Path) -> dict[str, Any]:
    excluded = {"artifact_manifest_sha256.json", "test_output.txt", "verifier_output.txt", "fresh_clone_output.txt"}
    files = {
        path.relative_to(artifact).as_posix(): sha256_file(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    return {"algorithm": "sha256", "excluded": sorted(excluded), "files": files}


def verify_artifact(artifact: Path) -> list[str]:
    manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
    errors = []
    for relative, expected in manifest["files"].items():
        path = artifact / relative
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(relative)
    return errors
