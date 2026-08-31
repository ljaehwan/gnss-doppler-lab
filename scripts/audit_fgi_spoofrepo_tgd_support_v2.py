#!/usr/bin/env python3
"""Apply the frozen cadence-corrected support gate to the pinned FGI TGD run."""
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

from gnss_doppler_lab.clean_geometry_support import audit_clean_geometry_support  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/experiments/fgi_spoofrepo_tgd_support_preflight_v2.json"
PROTOCOL = ROOT / "docs/results/fgi_spoofrepo_tgd_support_preflight_protocol_v2.md"
RELEASE_TOKEN = "RELEASE-FGI-SPOOFREPO-TGD-SUPPORT-PREFLIGHT-V2"
RELEASE_INPUTS = (
    "configs/experiments/fgi_spoofrepo_tgd_support_preflight_v2.json",
    "docs/results/fgi_spoofrepo_tgd_support_preflight_protocol_v2.md",
    "scripts/audit_fgi_spoofrepo_tgd_support_v2.py",
    "src/gnss_doppler_lab/clean_geometry_support.py",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.fgi-spoofrepo-tgd-support-preflight-config":
        raise ValueError("unsupported FGI support-preflight schema")
    if config.get("schema_version") != 2:
        raise ValueError("unsupported FGI support-preflight version")
    experiment = config["experiment"]
    for key in ("detector_score_access", "delay_template_access", "threshold_refitting", "receiver_replay"):
        if experiment.get(key) is not False:
            raise ValueError(f"v2 support audit must forbid {key}")
    if config["analysis_intervals_seconds"] != {
        "clean": [40.0, 120.0],
        "post_onset_support": [160.0, 230.0],
    }:
        raise ValueError("analysis intervals drifted")
    expected_gate = {
        "bin_seconds": 1.0,
        "minimum_epochs_per_prn_bin": 40,
        "minimum_telemetry_cadence_occupancy": 0.8,
        "minimum_primary_prns": 8,
        "secondary_boundary_prns": 7,
        "minimum_primary_bins_each_interval": 60,
        "require_complex_nine_tap": True,
    }
    if config["support_gate"] != expected_gate:
        raise ValueError("cadence-corrected support gate drifted")
    receiver = config["receiver_output"]
    if receiver["telemetry_synchronized_dump_rate_hz"] != 50:
        raise ValueError("pinned telemetry cadence drifted")
    expected_minimum = round(
        receiver["telemetry_synchronized_dump_rate_hz"]
        * expected_gate["bin_seconds"]
        * expected_gate["minimum_telemetry_cadence_occupancy"]
    )
    if expected_minimum != expected_gate["minimum_epochs_per_prn_bin"]:
        raise ValueError("minimum epoch count is inconsistent with cadence occupancy")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT)
        if dirty.returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "inputs": {relative: {"sha256": file_hash(ROOT / relative)} for relative in RELEASE_INPUTS},
    }


def validate_pinned_receiver(config: dict[str, Any]) -> Path:
    receiver = config["receiver_output"]
    run_dir = _resolve(receiver["directory"])
    manifest_path = run_dir / "manifest.json"
    receiver_config = run_dir / "receiver.conf"
    if file_hash(manifest_path) != receiver["manifest_sha256"]:
        raise ValueError("pinned receiver manifest hash mismatch")
    if file_hash(receiver_config) != receiver["receiver_config_sha256"]:
        raise ValueError("pinned receiver configuration hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source"]["iq_sha256"] != config["dataset"]["sha256"]:
        raise ValueError("receiver manifest does not identify the pinned FGI RF file")
    paths = sorted((run_dir / "raw").glob("epl_tracking_ch_*.mat"))
    if len(paths) != receiver["mat_file_count"]:
        raise ValueError("tracking MAT file count mismatch")
    return run_dir


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    release = committed_release()
    run_dir = validate_pinned_receiver(config)
    gate = config["support_gate"]
    audits: dict[str, Any] = {}
    for name, interval in config["analysis_intervals_seconds"].items():
        audits[name] = audit_clean_geometry_support(
            run_dir,
            start_s=float(interval[0]),
            end_s=float(interval[1]),
            bin_seconds=float(gate["bin_seconds"]),
            minimum_epochs=int(gate["minimum_epochs_per_prn_bin"]),
            minimum_primary_prns=int(gate["minimum_primary_prns"]),
            secondary_boundary_prns=int(gate["secondary_boundary_prns"]),
            minimum_primary_bins=int(gate["minimum_primary_bins_each_interval"]),
            require_complex_nine_tap=True,
        )
    eligible = all(audit["support_eligible"] for audit in audits.values())
    result = {
        "schema": "gnss-doppler-lab.fgi-spoofrepo-tgd-support-preflight-result",
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": release,
        "config": {"path": str(config_path), "sha256": file_hash(config_path)},
        "protocol": {"path": str(PROTOCOL), "sha256": file_hash(PROTOCOL)},
        "score_accessed": False,
        "delay_template_accessed": False,
        "detector_loaded": False,
        "receiver_replayed": False,
        "support_audits": audits,
        "decision": "SUPPORT_ELIGIBLE" if eligible else "INSUFFICIENT_SUPPORT",
        "claim_boundary": config["claim_boundary"],
    }
    output = _resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    summary = output / "summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary),
        "decision": result["decision"],
        "clean_maximum_eligible_prns": audits["clean"]["maximum_eligible_prns"],
        "clean_primary_bins": audits["clean"]["primary_bin_count"],
        "post_onset_maximum_eligible_prns": audits["post_onset_support"]["maximum_eligible_prns"],
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

