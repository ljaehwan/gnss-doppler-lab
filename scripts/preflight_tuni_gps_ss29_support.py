#!/usr/bin/env python3
"""Download a bounded SS-29 IQ prefix and perform a score-free support gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from gnss_doppler_lab.clean_geometry_support import audit_clean_geometry_support  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/experiments/tuni_gps_ss29_support_preflight_v1.json"
PROTOCOL = ROOT / "docs/results/tuni_gps_ss29_support_preflight_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-TUNI-GPS-SS29-SUPPORT-PREFLIGHT-V1"
RELEASE_INPUTS = (
    "configs/experiments/tuni_gps_ss29_support_preflight_v1.json",
    "docs/results/tuni_gps_ss29_support_preflight_protocol_v1.md",
    "scripts/preflight_tuni_gps_ss29_support.py",
    "src/gnss_doppler_lab/clean_geometry_support.py",
)


def _load_clean_preflight() -> Any:
    path = ROOT / "scripts/preflight_tuni_gps_clean.py"
    spec = importlib.util.spec_from_file_location("tuni_gps_clean_for_ss29", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CLEAN = _load_clean_preflight()


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.tuni-gps-ss29-support-preflight-config":
        raise ValueError("unsupported SS-29 support-preflight schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported SS-29 support-preflight version")
    experiment = config["experiment"]
    forbidden = (
        "detector_score_access", "delay_template_access", "threshold_refitting",
        "post_preflight_rule_change",
    )
    if any(experiment.get(key) is not False for key in forbidden):
        raise ValueError("support preflight must forbid detector access and rule changes")
    dataset = config["dataset"]
    if dataset["scenario"] != "SS-29" or dataset["documented_spoofed_prns"] != [1, 2, 21, 32]:
        raise ValueError("SS-29 target contract drifted")
    if int(dataset["full_file_bytes"]) != 29_999_832_000:
        raise ValueError("SS-29 full-file byte count drifted")
    probe = config["prefix_probe"]
    if (
        int(probe["byte_start"]) != 0
        or int(probe["byte_count"]) != 2_000_000_000
        or float(probe["duration_seconds"]) != 10.0
        or probe["analysis_interval_seconds"] != [5.0, 10.0]
    ):
        raise ValueError("SS-29 bounded prefix contract drifted")
    if int(probe["byte_count"]) != round(
        float(probe["duration_seconds"]) * int(dataset["input_sample_rate_hz"]) * 4
    ):
        raise ValueError("prefix byte count disagrees with big-endian ishort duration")
    gate = config["support_gate"]
    expected = {
        "bin_seconds": 1.0,
        "minimum_epochs_per_prn_bin": 200,
        "minimum_primary_prns": 8,
        "minimum_primary_bins": 3,
        "minimum_target_bins_per_spoofed_prn": 3,
        "require_complex_nine_tap": True,
    }
    if gate != expected:
        raise ValueError("SS-29 support gate drifted")


def parse_content_range(headers: str) -> tuple[int, int, int]:
    matches = re.findall(
        r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", headers
    )
    if not matches:
        raise ValueError("download did not return a byte Content-Range")
    start, end, total = (int(value) for value in matches[-1])
    return start, end, total


def target_bin_counts(audit: dict[str, Any], targets: list[int]) -> dict[str, int]:
    counts = {f"G{target:02d}": 0 for target in targets}
    for values in audit["eligible_prns_by_bin"].values():
        present = {int(value) for value in values}
        for target in targets:
            if target in present:
                counts[f"G{target:02d}"] += 1
    return counts


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT
        )
        if dirty.returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "inputs": {
            relative: {"sha256": file_hash(ROOT / relative)}
            for relative in RELEASE_INPUTS
        },
    }


def _download_prefix(config: dict[str, Any], output: Path) -> tuple[Path, dict[str, Any]]:
    dataset, probe = config["dataset"], config["prefix_probe"]
    source = output / "source_prefix"
    source.mkdir()
    prefix = source / "SS-29.prefix-10s.bin"
    headers = source / "download_headers.txt"
    byte_count = int(probe["byte_count"])
    command = [
        "curl", "--fail", "--location", "--retry", "3",
        "--range", f"0-{byte_count - 1}",
        "--max-filesize", str(byte_count),
        "--dump-header", str(headers),
        "--output", str(prefix), str(dataset["raw_url"]),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"curl prefix download failed with code {completed.returncode}")
    if prefix.stat().st_size != byte_count:
        raise ValueError("bounded SS-29 prefix byte count mismatch")
    content_range = parse_content_range(headers.read_text(encoding="utf-8"))
    expected = (0, byte_count - 1, int(dataset["full_file_bytes"]))
    if content_range != expected:
        raise ValueError(f"unexpected SS-29 Content-Range: {content_range} != {expected}")
    return prefix, {
        "curl_command": command,
        "content_range": list(content_range),
        "bytes": prefix.stat().st_size,
        "sha256": file_hash(prefix),
        "headers_sha256": file_hash(headers),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    release = committed_release()
    dataset = config["dataset"]
    for key in ("readme", "scenario_file"):
        path = _resolve(dataset[f"{key}_path"])
        if file_hash(path, "md5") != dataset[f"{key}_md5"]:
            raise ValueError(f"SS-29 {key} checksum mismatch")
    output = _resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    prefix, download = _download_prefix(config, output)

    executable = _resolve(config["receiver"]["executable"])
    executable_match = shutil.which(str(executable))
    executable = Path(executable_match).resolve() if executable_match else executable
    receiver = output / "receiver"
    raw = receiver / "raw"
    raw.mkdir(parents=True)
    probe = config["prefix_probe"]
    receiver_config = receiver / "receiver.conf"
    receiver_config.write_text(CLEAN.render_config(
        iq_path=prefix, output_dir=receiver,
        duration_s=float(probe["duration_seconds"]),
        channel_count=int(probe["channel_count"]),
    ), encoding="utf-8")
    command = [str(executable), f"--config_file={receiver_config}", "--keyboard=false"]
    completed = subprocess.run(
        command, cwd=receiver, capture_output=True, text=True,
        timeout=1800, check=False,
    )
    (receiver / "receiver.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"GNSS-SDR exited {completed.returncode}")
    manifest = {
        "schema": "gnss-doppler-lab.tuni-gps-ss29-prefix-receiver.v1",
        "source": {
            "sample_rate_hz": 50_000_000,
            "iq": str(prefix),
            "iq_bytes": prefix.stat().st_size,
            "start_offset_s": 0.0,
        },
        "tracking": {
            "raw_directory": "raw",
            "tap_count": 9,
            "tap_spacing_chips": 0.125,
        },
        "receiver": {
            "command": command,
            "return_code": completed.returncode,
            "executable_sha256": file_hash(executable),
            "config_sha256": file_hash(receiver_config),
        },
    }
    (receiver / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate = config["support_gate"]
    interval = probe["analysis_interval_seconds"]
    audit = audit_clean_geometry_support(
        receiver, start_s=float(interval[0]), end_s=float(interval[1]),
        bin_seconds=float(gate["bin_seconds"]),
        minimum_epochs=int(gate["minimum_epochs_per_prn_bin"]),
        minimum_primary_prns=int(gate["minimum_primary_prns"]),
        secondary_boundary_prns=7,
        minimum_primary_bins=int(gate["minimum_primary_bins"]),
        require_complex_nine_tap=True,
    )
    targets = target_bin_counts(audit, dataset["documented_spoofed_prns"])
    target_pass = all(
        count >= int(gate["minimum_target_bins_per_spoofed_prn"])
        for count in targets.values()
    )
    support_pass = bool(audit["support_eligible"] and target_pass)
    result = {
        "schema": "gnss-doppler-lab.tuni-gps-ss29-support-preflight-result",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": release,
        "config": {"path": str(config_path), "sha256": file_hash(config_path)},
        "protocol": {"path": str(PROTOCOL), "sha256": file_hash(PROTOCOL)},
        "score_accessed": False,
        "delay_template_accessed": False,
        "detector_loaded": False,
        "download": download,
        "receiver_manifest": {"path": str(receiver / "manifest.json"), "sha256": file_hash(receiver / "manifest.json")},
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
