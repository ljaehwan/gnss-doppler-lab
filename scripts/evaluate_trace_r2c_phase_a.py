#!/usr/bin/env python3
"""Run frozen Phase A and add the preregistered whole-row terminal gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np

import evaluate_trace_r2a_phase_a as evaluator
from gnss_doppler_lab.trace_reproducibility import load_replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair"
SSD = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair"
)
evaluator.ARTIFACT = ARTIFACT
evaluator.SSD = SSD


def row_set(replay) -> np.ndarray:
    records = replay.records
    rows = np.empty(
        len(records),
        dtype=[("prn", "<u4"), ("start", "<u8"), ("end", "<u8")],
    )
    rows["prn"] = records["prn"]
    rows["start"] = records["raw_interval_start_sample"]
    rows["end"] = records["raw_interval_end_sample"]
    return np.sort(rows, order=("prn", "start", "end"))


def row_set_sha256(rows: np.ndarray) -> str:
    digest = hashlib.sha256()
    dataset = b"TEXBAT.cleanStatic"
    for row in rows:
        digest.update(struct.pack("<I", len(dataset)))
        digest.update(dataset)
        digest.update(struct.pack("<IQQ", int(row["prn"]), int(row["start"]), int(row["end"])))
    return digest.hexdigest()


def terminal_summary(replay) -> list[dict[str, int]]:
    result = []
    records = replay.records
    for channel in np.unique(records["channel"]):
        channel_rows = records[records["channel"] == channel]
        for prn in np.unique(channel_rows["prn"]):
            selected = channel_rows[channel_rows["prn"] == prn]
            result.append(
                {
                    "channel": int(channel),
                    "prn": int(prn),
                    "record_count": int(len(selected)),
                    "first_raw_interval_start_sample": int(
                        selected["raw_interval_start_sample"].min()
                    ),
                    "last_raw_interval_start_sample": int(
                        selected["raw_interval_start_sample"].max()
                    ),
                    "last_raw_interval_end_sample": int(
                        selected["raw_interval_end_sample"].max()
                    ),
                }
            )
    return sorted(result, key=lambda row: (row["channel"], row["prn"]))


def main() -> int:
    inherited_code = evaluator.main()
    rep3 = load_replay(
        evaluator.replay_dir("TEXBAT.cleanStatic.rep3"), "TEXBAT.cleanStatic"
    )
    rep4 = load_replay(
        evaluator.replay_dir("TEXBAT.cleanStatic.rep4"), "TEXBAT.cleanStatic"
    )
    rows3, rows4 = row_set(rep3), row_set(rep4)
    summary3, summary4 = terminal_summary(rep3), terminal_summary(rep4)
    row_sets_equal = bool(np.array_equal(rows3, rows4))
    terminal_counts_equal = summary3 == summary4
    audit = {
        "schema": "gnss-doppler-lab.trace-r2c-terminal-row-set-audit.v1",
        "status": "PASS" if row_sets_equal and terminal_counts_equal else "FAIL",
        "whole_replay_row_set_identical": row_sets_equal,
        "terminal_row_counts_per_prn_channel_identical": terminal_counts_equal,
        "rep3": {
            "directory": str(rep3.directory),
            "row_count": int(len(rows3)),
            "row_set_sha256": row_set_sha256(rows3),
            "per_prn_channel": summary3,
        },
        "rep4": {
            "directory": str(rep4.directory),
            "row_count": int(len(rows4)),
            "row_set_sha256": row_set_sha256(rows4),
            "per_prn_channel": summary4,
        },
        "rep3_only_rows": int(len(np.setdiff1d(rows3, rows4))),
        "rep4_only_rows": int(len(np.setdiff1d(rows4, rows3))),
    }
    (ARTIFACT / "terminal_row_set_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    metrics_path = ARTIFACT / "rep3_rep4_reproduction_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    checks = metrics["semantic_reproduction_gate"]["checks"]
    checks["whole_replay_row_set_identical"] = row_sets_equal
    checks["terminal_row_counts_per_prn_channel_identical"] = terminal_counts_equal
    terminal_pass = row_sets_equal and terminal_counts_equal
    if not terminal_pass:
        metrics["semantic_reproduction_gate"]["status"] = "FAIL"
        metrics["phase_a_status"] = "FAIL"
        metrics["phase_b_authorized"] = False
        metrics["failure_verdict_if_any"] = "TERMINAL_ROW_SET_NONREPRODUCIBLE"
    metrics["schema"] = "gnss-doppler-lab.trace-r2c-phase-a-semantic-reproduction.v1"
    metrics["terminal_row_set_gate"] = audit
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    inventory_path = ARTIFACT / "replay_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["schema"] = "gnss-doppler-lab.trace-r2c-replay-inventory.v1"
    inventory["terminal_row_set_audit"] = "terminal_row_set_audit.json"
    inventory["phase_a_decision"] = {
        "status": metrics["phase_a_status"],
        "phase_b_authorized": metrics["phase_b_authorized"],
    }
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    return 0 if inherited_code == 0 and terminal_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
