#!/usr/bin/env python3
"""Recompute the frozen WCL CGC endpoint from retained receiver outputs.

The original 25 MHz RF files were intentionally removed after successful
GNSS-SDR processing. This replay therefore starts at the retained GNSS-SDR
receiver manifests and tracking outputs, repeats signed-delay extraction and
CGC scoring, and compares both the logical result and emitted CSV bytes with
the sealed result. It never writes inside the frozen data directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_temporal_final_static_v1 as frozen  # noqa: E402


DEFAULT_DATA_ROOT = Path("/home/ubuntu/hdd_data/cgc_temporal_final_static_v1")
DEFAULT_OUTPUT = ROOT / "reproduction-output/wcl-cgc-final-analysis-v1"
REFERENCE_RECORD = ROOT / "docs/results/cgc_temporal_final_static_v1_summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rebased_path(
    recorded_path: str | Path, frozen_root: Path, data_root: Path,
) -> Path:
    """Map an absolute path recorded at release time into a moved data root."""
    source = Path(recorded_path)
    try:
        relative = source.relative_to(frozen_root)
    except ValueError as exc:
        raise ValueError(f"recorded path escapes frozen data root: {source}") from exc
    return data_root / relative


def load_retained_runtime(
    config: dict[str, Any], data_root: Path,
) -> dict[str, dict[str, Any]]:
    frozen_root = Path(config["output_root"])
    runtime: dict[str, dict[str, Any]] = {}
    for pair in config["pairs"]:
        pair_id = pair["candidate_id"]
        pair_root = data_root / "pairs" / pair_id
        complete_path = pair_root / "pair_complete.json"
        if not complete_path.is_file():
            raise FileNotFoundError(f"missing retained pair record: {complete_path}")
        complete = read_json(complete_path)
        if complete.get("pair") != pair:
            raise ValueError(f"pair contract mismatch: {pair_id}")

        log_record = complete["authentic_los_log"]
        log_path = rebased_path(log_record["path"], frozen_root, data_root)
        if not log_path.is_file() or frozen.sha256(log_path) != log_record["sha256"]:
            raise ValueError(f"retained LOS log mismatch: {pair_id}")

        receivers: dict[str, dict[str, str]] = {}
        if set(complete["receivers"]) != set(frozen.CONDITIONS):
            raise ValueError(f"receiver condition roster mismatch: {pair_id}")
        for condition, record in complete["receivers"].items():
            path = rebased_path(record["path"], frozen_root, data_root)
            if not path.is_file() or frozen.sha256(path) != record["sha256"]:
                raise ValueError(f"retained receiver manifest mismatch: {pair_id}/{condition}")
            receivers[condition] = {
                "path": str(path.resolve()),
                "sha256": record["sha256"],
            }
        runtime[pair_id] = {
            **complete,
            "authentic_los_log": {
                "path": str(log_path.resolve()),
                "sha256": log_record["sha256"],
            },
            "receivers": receivers,
        }
    return runtime


def logical_comparison(
    regenerated: dict[str, Any], reference: dict[str, Any],
) -> dict[str, bool]:
    return {
        "decision": regenerated["decision"] == reference["decision"],
        "aggregates": regenerated["aggregates"] == reference["aggregates"],
        "gates": regenerated["gates"] == reference["gates"],
        "pairs": regenerated["pairs"] == reference["pairs"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    config = read_json(frozen.CONFIG)
    compact_reference = read_json(REFERENCE_RECORD)
    full_reference_path = data_root / "summary.json"
    if not full_reference_path.is_file():
        raise FileNotFoundError(f"missing frozen summary: {full_reference_path}")
    if frozen.sha256(full_reference_path) != compact_reference["source_result"]["sha256"]:
        raise ValueError("frozen full-summary hash does not match the committed record")
    full_reference = read_json(full_reference_path)

    print("[replay] validating frozen config, tools, and retained hashes", flush=True)
    # Validate the exact frozen config, inputs, and executable hashes first.
    context = frozen.validate(config)
    context = {**context, "output_root": data_root}
    runtime = load_retained_runtime(config, data_root)
    print(
        "[replay] recomputing 5 geometries x 4 conditions; this can take several minutes",
        flush=True,
    )
    regenerated = frozen.analyze(config, context, runtime)
    print("[replay] comparing logical results and CSV bytes", flush=True)

    delay_rows = regenerated.pop("delay_rows")
    stabilized_rows = regenerated.pop("stabilized_rows")
    score_rows = regenerated.pop("score_rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = {
        "delay_estimates": output_dir / "delay_estimates.csv",
        "stabilized_delay_estimates": output_dir / "stabilized_delay_estimates.csv",
        "geometry_scores": output_dir / "geometry_scores.csv",
    }
    frozen.write_csv(generated_paths["delay_estimates"], delay_rows)
    frozen.write_csv(generated_paths["stabilized_delay_estimates"], stabilized_rows)
    frozen.write_csv(generated_paths["geometry_scores"], score_rows)

    logical = logical_comparison(regenerated, full_reference)
    csv_checks: dict[str, bool] = {}
    generated_files: dict[str, dict[str, Any]] = {}
    for name, path in generated_paths.items():
        digest = frozen.sha256(path)
        expected = full_reference["artifacts"][name]["sha256"]
        csv_checks[name] = digest == expected
        generated_files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "expected_sha256": expected,
        }

    checks = {
        "frozen_summary_hash": True,
        "retained_pair_and_receiver_hashes": True,
        **{f"logical_{key}": value for key, value in logical.items()},
        **{f"csv_{key}": value for key, value in csv_checks.items()},
    }
    exact = all(checks.values())
    report = {
        "schema": "gnss-doppler-lab.wcl-cgc-final-analysis-reproduction",
        "schema_version": 1,
        "status": "EXACT_MATCH" if exact else "MISMATCH",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": frozen.git("rev-parse", "HEAD"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "data_root": str(data_root),
        "frozen_summary": {
            "path": str(full_reference_path),
            "sha256": frozen.sha256(full_reference_path),
        },
        "checks": checks,
        "regenerated": {
            "decision": regenerated["decision"],
            "aggregates": regenerated["aggregates"],
            "gates": regenerated["gates"],
            "files": generated_files,
        },
        "scope": (
            "Analysis replay from retained GNSS-SDR tracking outputs; raw 25 MHz "
            "RF regeneration is outside this command."
        ),
    }
    report_path = output_dir / "reproduction_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
