#!/usr/bin/env python3
"""Apply the frozen healthy-ephemeris identity gate to FGI TGD support."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
)


DEFAULT_CONFIG = ROOT / "configs/experiments/fgi_spoofrepo_tgd_support_preflight_v3.json"
PROTOCOL = ROOT / "docs/results/fgi_spoofrepo_tgd_support_preflight_protocol_v3.md"
RELEASE_TOKEN = "RELEASE-FGI-SPOOFREPO-TGD-SUPPORT-PREFLIGHT-V3"
RELEASE_INPUTS = (
    "configs/experiments/fgi_spoofrepo_tgd_support_preflight_v3.json",
    "docs/results/fgi_spoofrepo_tgd_support_preflight_protocol_v3.md",
    "scripts/audit_fgi_spoofrepo_tgd_support_v3.py",
    "src/gnss_doppler_lab/gcmr_geometry.py",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify(record: dict[str, str], label: str) -> Path:
    path = resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{label} SHA-256 mismatch: {observed}")
    return path


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.fgi-spoofrepo-tgd-support-preflight-config":
        raise ValueError("unsupported support-preflight schema")
    if config.get("schema_version") != 3:
        raise ValueError("unsupported support-preflight version")
    for key in ("detector_score_access", "delay_template_access", "threshold_refitting", "receiver_replay"):
        if config["experiment"].get(key) is not False:
            raise ValueError(f"v3 support audit must forbid {key}")
    identity = config["identity_gate"]
    expected = [5, 7, 8, 9, 13, 14, 15, 18, 20, 22, 27, 30]
    if identity.get("expected_healthy_prns") != expected:
        raise ValueError("healthy ephemeris roster drifted")
    if identity.get("minimum_primary_prns") != 8 or identity.get("minimum_primary_bins_each_interval") != 60:
        raise ValueError("identity support gate drifted")
    inherited = config["support_contract_inherited_from_v2"]
    expected_inherited = {
        "bin_seconds": 1.0,
        "minimum_epochs_per_prn_bin": 40,
        "minimum_telemetry_cadence_occupancy": 0.8,
        "require_complex_nine_tap": True,
        "analysis_intervals_seconds": {
            "clean": [40.0, 120.0],
            "post_onset_support": [160.0, 230.0],
        },
    }
    if inherited != expected_inherited:
        raise ValueError("inherited v2 support contract drifted")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT).returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "inputs": {relative: {"sha256": sha256(ROOT / relative)} for relative in RELEASE_INPUTS},
    }


def filter_interval(audit: dict[str, Any], healthy: set[int], minimum_prns: int, minimum_bins: int) -> dict[str, Any]:
    filtered = {
        str(key): sorted(healthy.intersection(int(prn) for prn in prns))
        for key, prns in audit["eligible_prns_by_bin"].items()
    }
    counts = {key: len(prns) for key, prns in filtered.items()}
    primary = sorted(int(key) for key, count in counts.items() if count >= minimum_prns)
    distribution: dict[str, int] = {}
    for count in counts.values():
        distribution[str(count)] = distribution.get(str(count), 0) + 1
    eligible = len(primary) >= minimum_bins
    return {
        "analysis_interval_seconds": audit["rules"]["analysis_interval_seconds"],
        "healthy_ephemeris_prns": [f"G{prn:02d}" for prn in sorted(healthy)],
        "minimum_primary_prns": minimum_prns,
        "minimum_primary_bins": minimum_bins,
        "maximum_identity_valid_prns": max(counts.values(), default=0),
        "eligible_bin_count_by_prn_count": dict(sorted(distribution.items(), key=lambda row: int(row[0]))),
        "primary_bin_count": len(primary),
        "primary_bins": primary,
        "identity_valid_prns_by_bin": filtered,
        "support_eligible": eligible,
        "status": "SUPPORT_ELIGIBLE" if eligible else "INSUFFICIENT_SUPPORT",
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    release = committed_release()
    v2_path = verify(config["inputs"]["cadence_corrected_v2_summary"], "v2 summary")
    ephemeris_path = verify(config["inputs"]["gps_ephemeris"], "GPS ephemeris")
    verify(config["inputs"]["geometry_module"], "geometry module")
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    if v2.get("schema_version") != 2 or v2.get("score_accessed") is not False:
        raise ValueError("pinned v2 summary is not a score-free v2 result")
    if v2.get("delay_template_accessed") is not False or v2.get("detector_loaded") is not False:
        raise ValueError("v2 outcome boundary was not preserved")
    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(ephemeris_path)
    healthy_map, health = ephemeris_health_selection(
        ephemerides, tracked_prns=set(range(1, 33)), min_prns=8
    )
    healthy = set(healthy_map)
    expected = set(config["identity_gate"]["expected_healthy_prns"])
    if healthy != expected:
        raise ValueError("observed healthy ephemeris roster differs from frozen roster")
    minimum_prns = int(config["identity_gate"]["minimum_primary_prns"])
    minimum_bins = int(config["identity_gate"]["minimum_primary_bins_each_interval"])
    audits = {
        name: filter_interval(audit, healthy, minimum_prns, minimum_bins)
        for name, audit in v2["support_audits"].items()
    }
    eligible = all(audit["support_eligible"] for audit in audits.values())
    result = {
        "schema": "gnss-doppler-lab.fgi-spoofrepo-tgd-support-preflight-result",
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": release,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "score_accessed": False,
        "delay_template_accessed": False,
        "detector_loaded": False,
        "receiver_replayed": False,
        "ephemeris_health": health,
        "support_audits": audits,
        "decision": "SUPPORT_ELIGIBLE" if eligible else "INSUFFICIENT_SUPPORT",
        "claim_boundary": config["claim_boundary"],
    }
    output = resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    summary = output / "summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary),
        "decision": result["decision"],
        "healthy_ephemeris_prns": sorted(healthy),
        "clean_maximum_identity_valid_prns": audits["clean"]["maximum_identity_valid_prns"],
        "clean_primary_bins": audits["clean"]["primary_bin_count"],
        "post_onset_maximum_identity_valid_prns": audits["post_onset_support"]["maximum_identity_valid_prns"],
        "post_onset_primary_bins": audits["post_onset_support"]["primary_bin_count"],
        "score_accessed": False,
    }, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-token", required=True)
    args = parser.parse_args()
    if args.release_token != RELEASE_TOKEN:
        raise ValueError("release token mismatch")
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

