#!/usr/bin/env python3
"""Correct the SS-29 v1 tracking timebase without reprocessing raw IQ."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from gnss_doppler_lab.clean_geometry_support import audit_clean_geometry_support  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/experiments/tuni_gps_ss29_support_preflight_v1_1.json"
PROTOCOL = ROOT / "docs/results/tuni_gps_ss29_support_preflight_protocol_v1_1.md"
RELEASE_TOKEN = "RELEASE-TUNI-GPS-SS29-SUPPORT-PREFLIGHT-V1-1"
RELEASE_INPUTS = (
    "configs/experiments/tuni_gps_ss29_support_preflight_v1_1.json",
    "docs/results/tuni_gps_ss29_support_preflight_protocol_v1_1.md",
    "scripts/recover_tuni_gps_ss29_support_preflight_v1_1.py",
    "src/gnss_doppler_lab/clean_geometry_support.py",
)


def _load_v1() -> Any:
    path = ROOT / "scripts/preflight_tuni_gps_ss29_support.py"
    spec = importlib.util.spec_from_file_location("tuni_gps_ss29_support_v1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V1 = _load_v1()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def canonical_mat_tree(raw: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": V1.file_hash(path),
        }
        for path in sorted(raw.glob("epl_tracking_ch_*.mat"))
    ]
    encoded = (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest(), rows


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.tuni-gps-ss29-support-preflight-correction-config":
        raise ValueError("unsupported SS-29 correction schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported SS-29 correction version")
    experiment = config["experiment"]
    if experiment["base_release_commit"] != "5528711c481fb6a617e2c409d2759f539c97d432":
        raise ValueError("base release commit drifted")
    if experiment["raw_iq_reprocessed"] is not False:
        raise ValueError("v1.1 must not reprocess raw IQ")
    for key in ("detector_score_access", "delay_template_access", "threshold_refitting", "support_rule_change"):
        if experiment[key] is not False:
            raise ValueError("v1.1 correction scope drifted")
    if config["timebase"] != {
        "input_sample_rate_hz": 50_000_000,
        "internal_tracking_sample_rate_hz": 5_000_000,
        "start_offset_seconds": 0.0,
    }:
        raise ValueError("corrected timebase drifted")
    if config["documented_spoofed_prns"] != [1, 2, 21, 32]:
        raise ValueError("documented target PRNs drifted")
    if config["analysis_interval_seconds"] != [5.0, 10.0]:
        raise ValueError("analysis interval drifted")
    if config["support_gate"] != {
        "bin_seconds": 1.0,
        "minimum_epochs_per_prn_bin": 200,
        "minimum_primary_prns": 8,
        "minimum_primary_bins": 3,
        "minimum_target_bins_per_spoofed_prn": 3,
        "require_complex_nine_tap": True,
    }:
        raise ValueError("support gate drifted")


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
        "inputs": {
            relative: {"sha256": V1.file_hash(ROOT / relative)}
            for relative in RELEASE_INPUTS
        },
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    release = committed_release()
    base = config["base_inputs"]
    base_config = _resolve(base["config_path"])
    if V1.file_hash(base_config) != base["config_sha256"]:
        raise ValueError("base v1 config hash mismatch")
    root = _resolve(base["artifact_root"])
    invalid_summary = root / "summary.json"
    invalid_manifest = root / "receiver" / "manifest.json"
    raw = root / "receiver" / "raw"
    prefix = root / "source_prefix" / "SS-29.prefix-10s.bin"
    checks = {
        "invalid_summary_sha256": V1.file_hash(invalid_summary),
        "invalid_receiver_manifest_sha256": V1.file_hash(invalid_manifest),
        "prefix_sha256": V1.file_hash(prefix),
    }
    for key, value in checks.items():
        if value != base[key]:
            raise ValueError(f"base artifact hash mismatch: {key}")
    tree_hash, tree_rows = canonical_mat_tree(raw)
    if (
        tree_hash != base["raw_mat_tree_sha256"]
        or len(tree_rows) != int(base["raw_mat_count"])
        or sum(row["size_bytes"] for row in tree_rows) != int(base["raw_mat_bytes"])
    ):
        raise ValueError("base receiver MAT tree mismatch")
    invalid = json.loads(invalid_summary.read_text(encoding="utf-8"))
    if invalid.get("score_accessed") is not False or invalid.get("decision") != "INSUFFICIENT_SUPPORT":
        raise ValueError("unexpected v1 invalid-summary state")

    output = _resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(output)
    receiver = output / "receiver"
    receiver.mkdir(parents=True)
    (receiver / "raw").symlink_to(raw, target_is_directory=True)
    timebase = config["timebase"]
    corrected_manifest = {
        "schema": "gnss-doppler-lab.tuni-gps-ss29-prefix-receiver.v1_1",
        "correction": {
            "base_manifest": str(invalid_manifest),
            "base_manifest_sha256": checks["invalid_receiver_manifest_sha256"],
            "raw_iq_reprocessed": False,
            "changed_field": "source.sample_rate_hz: 50000000 -> 5000000",
        },
        "source": {
            "iq": str(prefix),
            "iq_bytes": prefix.stat().st_size,
            "input_sample_rate_hz": int(timebase["input_sample_rate_hz"]),
            "sample_rate_hz": int(timebase["internal_tracking_sample_rate_hz"]),
            "start_offset_s": float(timebase["start_offset_seconds"]),
        },
        "tracking": {
            "raw_directory": "raw",
            "tap_count": 9,
            "tap_spacing_chips": 0.125,
            "raw_mat_tree_sha256": tree_hash,
        },
    }
    manifest_path = receiver / "manifest.json"
    manifest_path.write_text(
        json.dumps(corrected_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate = config["support_gate"]
    interval = config["analysis_interval_seconds"]
    audit = audit_clean_geometry_support(
        receiver, start_s=float(interval[0]), end_s=float(interval[1]),
        bin_seconds=float(gate["bin_seconds"]),
        minimum_epochs=int(gate["minimum_epochs_per_prn_bin"]),
        minimum_primary_prns=int(gate["minimum_primary_prns"]),
        secondary_boundary_prns=7,
        minimum_primary_bins=int(gate["minimum_primary_bins"]),
        require_complex_nine_tap=True,
    )
    targets = V1.target_bin_counts(audit, config["documented_spoofed_prns"])
    target_pass = all(
        count >= int(gate["minimum_target_bins_per_spoofed_prn"])
        for count in targets.values()
    )
    support_pass = bool(audit["support_eligible"] and target_pass)
    result = {
        "schema": "gnss-doppler-lab.tuni-gps-ss29-support-preflight-result.v1_1",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": release,
        "config": {"path": str(config_path), "sha256": V1.file_hash(config_path)},
        "protocol": {"path": str(PROTOCOL), "sha256": V1.file_hash(PROTOCOL)},
        "base_v1_invalidated": True,
        "raw_iq_reprocessed": False,
        "score_accessed": False,
        "delay_template_accessed": False,
        "detector_loaded": False,
        "verified_base_artifacts": {**checks, "raw_mat_tree_sha256": tree_hash},
        "corrected_receiver_manifest": {"path": str(manifest_path), "sha256": V1.file_hash(manifest_path)},
        "support_audit": audit,
        "target_eligible_bin_counts": targets,
        "target_coverage_pass": target_pass,
        "decision": "SUPPORT_ELIGIBLE" if support_pass else "INSUFFICIENT_SUPPORT",
    }
    summary = output / "summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary),
        "decision": result["decision"],
        "discovered_prns": audit["discovered_prns"],
        "maximum_eligible_prns": audit["maximum_eligible_prns"],
        "primary_bin_count": audit["primary_bin_count"],
        "target_eligible_bin_counts": targets,
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
