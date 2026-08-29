#!/usr/bin/env python3
"""Run a score-free clean satellite-support audit for external receiver data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.clean_geometry_support import audit_clean_geometry_support  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/experiments/clean_geometry_support_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "docs/results/clean_geometry_support_audit_v1.json"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.clean-geometry-support-audit-config":
        raise ValueError("unsupported support-audit config schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported support-audit config version")
    protocol = config.get("protocol", {})
    if protocol.get("score_access") is not False or protocol.get("attack_payload_access") is not False:
        raise ValueError("clean support audit must forbid score and attack-payload access")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("at least one clean candidate is required")
    ids = [str(row.get("id")) for row in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")


def run(config_path: Path, output_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    rules = config["rules"]
    results = []
    for candidate in config["candidates"]:
        audit = audit_clean_geometry_support(
            _resolve(candidate["receiver_run_dir"]),
            start_s=float(candidate["analysis_interval_seconds"][0]),
            end_s=float(candidate["analysis_interval_seconds"][1]),
            bin_seconds=float(rules["bin_seconds"]),
            minimum_epochs=int(rules["minimum_epochs_per_prn_bin"]),
            minimum_primary_prns=int(rules["minimum_primary_prns"]),
            secondary_boundary_prns=int(rules["secondary_boundary_prns"]),
            minimum_primary_bins=int(rules["minimum_primary_bins"]),
            require_complex_nine_tap=bool(rules["require_complex_nine_tap"]),
        )
        audit["candidate"] = {
            key: candidate[key]
            for key in ("id", "dataset", "scenario", "scientific_role", "prior_access")
        }
        results.append(audit)
    document = {
        "schema": "gnss-doppler-lab.clean-geometry-support-audit-batch",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "protocol": config["protocol"],
        "rules": rules,
        "results": results,
        "support_eligible_candidates": [
            row["candidate"]["id"] for row in results if row["support_eligible"]
        ],
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    document = run(args.config, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "support_eligible_candidates": document["support_eligible_candidates"],
        "results": [
            {
                "id": row["candidate"]["id"],
                "maximum_eligible_prns": row["maximum_eligible_prns"],
                "primary_bin_count": row["primary_bin_count"],
                "status": row["status"],
            }
            for row in document["results"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
