#!/usr/bin/env python3
"""Select fresh CGC RF pairs using startup LOS support only."""
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

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from gnss_doppler_lab.gps_sdr_sim import GpsSdrSimRunner  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import (  # noqa: E402
    parse_gps_sdr_sim_los_table,
)
from gnss_doppler_lab.rf_config import (  # noqa: E402
    InputConfig,
    OutputConfig,
    RFGenerationConfig,
    Scenario,
    SimulatorConfig,
    StaticPosition,
)
from gnss_doppler_lab.rf_impairments import clean_impairments  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_rf_fresh_candidate_pool_v2.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/cgc_rf_fresh_candidate_preflight_v2"
PROTOCOL_PATH = REPO_ROOT / "docs/results/cgc_rf_fresh_candidate_preflight_protocol_v2.md"
RELEASE_INPUTS = (
    "configs/experiments/cgc_rf_fresh_candidate_pool_v2.json",
    "scripts/preflight_cgc_rf_fresh_candidates.py",
    "docs/results/cgc_rf_fresh_candidate_preflight_protocol_v2.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def motion_kind(candidate: dict[str, Any]) -> str:
    return "static" if candidate["domain"] == "static" else str(candidate["motion"]["kind"])


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.cgc-rf-fresh-candidate-pool":
        raise ValueError("unsupported candidate-pool schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported candidate-pool version")
    experiment = config.get("experiment", {})
    if experiment.get("candidate_commit") != "60280575737ff70618c5619ff7522c516bbdc67a":
        raise ValueError("frozen candidate commit drifted")
    if experiment.get("score_access_during_preflight") is not False:
        raise ValueError("preflight score access must be forbidden")
    preflight = config.get("preflight", {})
    if preflight.get("duration_seconds") != 1 or preflight.get("sample_rate_hz") != 1000000:
        raise ValueError("preflight must remain a one-second 1 MHz support probe")
    if preflight.get("minimum_startup_los_prns") != 10:
        raise ValueError("startup LOS support threshold drifted")
    required = ["static", "straight", "parallel-sweep"]
    if preflight.get("required_motion_kinds") != required:
        raise ValueError("required motion order drifted")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 9:
        raise ValueError("candidate pool must contain exactly nine rows")
    ids = [str(row.get("paired_group_id")) for row in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    seeds = [row.get("receiver_seed") for row in candidates]
    if len(set(seeds)) != len(seeds):
        raise ValueError("receiver seeds must be unique")
    counts = {kind: 0 for kind in required}
    for row in candidates:
        kind = motion_kind(row)
        if kind not in counts:
            raise ValueError(f"unsupported motion kind: {kind}")
        counts[kind] += 1
        if row.get("split") != "fresh_test" or row.get("duration_seconds") != 30:
            raise ValueError("all candidates must be 30-second fresh-test rows")
        position = row.get("position", {})
        if set(position) != {"latitude_deg", "longitude_deg", "altitude_m"}:
            raise ValueError("candidate position schema drifted")
    if any(count != 3 for count in counts.values()):
        raise ValueError("each required motion kind must have three candidates")


def select_candidates(
    config: dict[str, Any], los_counts: dict[str, int]
) -> list[dict[str, Any]]:
    minimum = int(config["preflight"]["minimum_startup_los_prns"])
    selected: list[dict[str, Any]] = []
    for kind in config["preflight"]["required_motion_kinds"]:
        eligible = [
            row for row in config["candidates"]
            if motion_kind(row) == kind
            and int(los_counts.get(str(row["paired_group_id"]), -1)) >= minimum
        ]
        if not eligible:
            raise RuntimeError(f"no support-eligible candidate for {kind}")
        selected.append(eligible[0])
    return selected


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _committed_release_record() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT)
        if dirty.returncode:
            raise ValueError(f"preflight input is not committed and clean: {relative}")
    candidate = "60280575737ff70618c5619ff7522c516bbdc67a"
    if subprocess.run(["git", "merge-base", "--is-ancestor", candidate, "HEAD"], cwd=REPO_ROOT).returncode:
        raise ValueError("frozen candidate commit is not an ancestor of HEAD")
    return {
        "head_commit": _git("rev-parse", "HEAD"),
        "candidate_commit": candidate,
        "input_commits": {
            relative: _git("log", "-1", "--format=%H", "--", relative)
            for relative in RELEASE_INPUTS
        },
    }


def _probe_candidate(
    config: dict[str, Any], candidate: dict[str, Any], simulator: GpsSdrSimRunner,
    scratch: Path, log_root: Path,
) -> dict[str, Any]:
    candidate_id = str(candidate["paired_group_id"])
    position = candidate["position"]
    utc = datetime.fromisoformat(str(candidate["utc"]).replace("Z", "+00:00"))
    rf_config = RFGenerationConfig(
        version=1,
        scenario=Scenario(
            candidate_id, "GPS", "L1CA", utc, int(config["preflight"]["duration_seconds"]),
            StaticPosition(float(position["latitude_deg"]), float(position["longitude_deg"]), float(position["altitude_m"])),
        ),
        input=InputConfig(_repo_path(config["inputs"]["rinex_nav"])),
        output=OutputConfig(scratch, int(config["preflight"]["sample_rate_hz"]), "s8_iq"),
        simulator=SimulatorConfig(str(_repo_path(config["inputs"]["simulator_executable"]))),
        impairments=clean_impairments(),
    )
    iq_path = scratch / f"{candidate_id}.bin"
    temporary_log = scratch / f"{candidate_id}.log"
    result = simulator.run(rf_config, iq_path, temporary_log)
    los = parse_gps_sdr_sim_los_table(temporary_log.read_text(encoding="utf-8"))
    audit_log = log_root / f"{candidate_id}.log"
    audit_log.write_bytes(temporary_log.read_bytes())
    iq_size = iq_path.stat().st_size
    iq_path.unlink()
    return {
        "candidate_id": candidate_id,
        "motion_kind": motion_kind(candidate),
        "startup_los_prn_count": len(los),
        "startup_los_prns": sorted(los),
        "support_eligible": len(los) >= int(config["preflight"]["minimum_startup_los_prns"]),
        "discarded_probe_iq_bytes": iq_size,
        "audit_log": {"path": str(audit_log.resolve()), "sha256": _sha256(audit_log)},
        "simulator_time": result["time"],
    }


def run(config_path: Path, output_root: Path) -> Path:
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    release = _committed_release_record()
    output_root.mkdir(parents=True)
    log_root = output_root / "logs"
    log_root.mkdir()
    simulator = GpsSdrSimRunner(str(_repo_path(config["inputs"]["simulator_executable"])))
    with tempfile.TemporaryDirectory(prefix="cgc-fresh-los-") as directory:
        scratch = Path(directory)
        probes = [
            _probe_candidate(config, row, simulator, scratch, log_root)
            for row in config["candidates"]
        ]
    counts = {row["candidate_id"]: int(row["startup_los_prn_count"]) for row in probes}
    selected = select_candidates(config, counts)
    document = {
        "schema": "gnss-doppler-lab.cgc-rf-fresh-candidate-preflight",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": _sha256(PROTOCOL_PATH)},
        "release": release,
        "score_accessed": False,
        "probe_iq_retained": False,
        "selection_rule": config["preflight"]["selection_rule"],
        "probes": probes,
        "selected_candidate_ids": [str(row["paired_group_id"]) for row in selected],
        "selected_pairs": selected,
        "status": "SUPPORT_ELIGIBLE",
    }
    summary = output_root / "summary.json"
    summary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(run(args.config, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
