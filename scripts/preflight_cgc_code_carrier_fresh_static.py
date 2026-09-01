#!/usr/bin/env python3
"""Select five fresh static code/carrier geometries using LOS support only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from gnss_doppler_lab.gps_sdr_sim import GpsSdrSimRunner  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.rf_config import (  # noqa: E402
    InputConfig, OutputConfig, RFGenerationConfig, Scenario, SimulatorConfig,
    StaticPosition,
)
from gnss_doppler_lab.rf_impairments import clean_impairments  # noqa: E402


CONFIG = ROOT / "configs/experiments/cgc_code_carrier_fresh_static_pool_v1.json"
PROTOCOL = ROOT / "docs/results/cgc_code_carrier_fresh_static_preflight_protocol_v1.md"
OUTPUT = ROOT / "artifacts/cgc_code_carrier_fresh_static_preflight_v1"
RELEASE_INPUTS = (
    "configs/experiments/cgc_code_carrier_fresh_static_pool_v1.json",
    "docs/results/cgc_code_carrier_fresh_static_preflight_protocol_v1.md",
    "scripts/preflight_cgc_code_carrier_fresh_static.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        git("ls-files", "--error-unmatch", relative)
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT).returncode:
            raise ValueError(f"preflight input is not committed and clean: {relative}")
    return {
        "head_commit": git("rev-parse", "HEAD"),
        "input_commits": {relative: git("log", "-1", "--format=%H", "--", relative) for relative in RELEASE_INPUTS},
    }


def validate(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.cgc-code-carrier-fresh-static-pool" or config.get("schema_version") != 1:
        raise ValueError("unsupported candidate-pool schema")
    if config["experiment"].get("score_access_during_preflight") is not False:
        raise ValueError("score access must be forbidden")
    preflight = config["preflight"]
    if (preflight["duration_seconds"], preflight["sample_rate_hz"], preflight["minimum_startup_los_prns"]) != (1, 1_000_000, 10):
        raise ValueError("preflight support contract drifted")
    candidates = config.get("candidates", [])
    if len(candidates) != 10 or len({row["candidate_id"] for row in candidates}) != 10:
        raise ValueError("candidate roster must contain ten unique IDs")
    for slot in preflight["slot_order"]:
        rows = [row for row in candidates if row.get("slot") == slot]
        if len(rows) != 2:
            raise ValueError(f"slot {slot} must contain two ordered candidates")
        for row in rows:
            if abs(float(np_norm(row["target_offset_enu_m"])) - 100.0) > 1e-9:
                raise ValueError("every frozen displacement must be exactly 100 m")
    for key in ("rinex_nav", "simulator"):
        record = config["inputs"][key]
        if sha256(repo_path(record["path"])) != record["sha256"]:
            raise ValueError(f"input hash mismatch: {key}")


def np_norm(values: list[float]) -> float:
    return sum(float(value) ** 2 for value in values) ** 0.5


def probe(config: dict[str, Any], candidate: dict[str, Any], scratch: Path, logs: Path) -> dict[str, Any]:
    position = candidate["position"]
    utc = datetime.fromisoformat(candidate["utc"].replace("Z", "+00:00"))
    rf_config = RFGenerationConfig(
        version=1,
        scenario=Scenario(
            candidate["candidate_id"], "GPS", "L1CA", utc,
            int(config["preflight"]["duration_seconds"]),
            StaticPosition(float(position["latitude_deg"]), float(position["longitude_deg"]), float(position["altitude_m"])),
        ),
        input=InputConfig(repo_path(config["inputs"]["rinex_nav"]["path"])),
        output=OutputConfig(scratch, int(config["preflight"]["sample_rate_hz"]), "s8_iq"),
        simulator=SimulatorConfig(str(repo_path(config["inputs"]["simulator"]["path"]))),
        impairments=clean_impairments(),
    )
    iq = scratch / f"{candidate['candidate_id']}.bin"
    temporary_log = scratch / f"{candidate['candidate_id']}.log"
    GpsSdrSimRunner(rf_config.simulator.executable).run(rf_config, iq, temporary_log)
    los = parse_gps_sdr_sim_los_table(temporary_log.read_text(encoding="utf-8"))
    audit_log = logs / temporary_log.name
    audit_log.write_bytes(temporary_log.read_bytes())
    byte_count = iq.stat().st_size
    iq.unlink()
    return {
        "candidate_id": candidate["candidate_id"],
        "slot": candidate["slot"],
        "startup_los_prn_count": len(los),
        "startup_los_prns": sorted(los),
        "support_eligible": len(los) >= int(config["preflight"]["minimum_startup_los_prns"]),
        "probe_iq_retained": False,
        "discarded_probe_iq_bytes": byte_count,
        "audit_log": {"path": str(audit_log.resolve()), "sha256": sha256(audit_log)},
    }


def run(output: Path) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate(config)
    release = committed_release()
    output.mkdir(parents=True)
    logs = output / "logs"; logs.mkdir()
    with tempfile.TemporaryDirectory(prefix="cgc-cc-fresh-support-") as directory:
        probes = [probe(config, row, Path(directory), logs) for row in config["candidates"]]
    by_id = {row["candidate_id"]: row for row in probes}
    selected = []
    for slot in config["preflight"]["slot_order"]:
        eligible = [row for row in config["candidates"] if row["slot"] == slot and by_id[row["candidate_id"]]["support_eligible"]]
        if not eligible:
            raise RuntimeError(f"no support-eligible candidate in {slot}")
        selected.append(eligible[0])
    summary = {
        "schema": "gnss-doppler-lab.cgc-code-carrier-fresh-static-preflight",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(CONFIG.resolve()), "sha256": sha256(CONFIG)},
        "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
        "release": release,
        "score_accessed": False,
        "probe_iq_retained": False,
        "selection_rule": config["preflight"]["selection_rule"],
        "probes": probes,
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
        "selected_candidates": selected,
        "status": "SUPPORT_ELIGIBLE",
    }
    destination = output / "summary.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(run(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
