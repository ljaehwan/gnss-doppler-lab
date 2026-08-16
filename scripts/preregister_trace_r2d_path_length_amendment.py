#!/usr/bin/env python3
"""Preregister the R2d handoff-path mirror after the preserved parser failure."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"


def main() -> int:
    failed = Path(
        "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
        "trace-stage0-r2d-oakbat-clean-support-repair/dumps/phase_a/"
        "texbat_cleanstatic/rep3/manifest.json"
    )
    manifest = json.loads(failed.read_text())
    if manifest["exit_code"] != 0 or manifest["replay_validation"]["observed_dump_file_count"] != 0:
        raise ValueError("preserved failure does not match path-truncation precondition")
    full = manifest["frozen_handoff_path"]
    truncated = full[:-2]
    if len(full) != 166 or len(truncated) != 164 or not full.endswith(".csv"):
        raise ValueError("unexpected handoff path length evidence")
    payload = {
        "schema": "gnss-doppler-lab.trace-r2d-config-path-amendment.v1",
        "status": "PREREGISTERED_BEFORE_PHASE_A_RETRY",
        "failure_run_id": "20260816T135409Z-r2d-phase-a-cleanstatic-rep3",
        "failure_label": "GNSS_SDR_INI_LINE_TRUNCATES_R2D_HANDOFF_PATH",
        "evidence": {
            "full_handoff_path": full,
            "full_handoff_path_length": len(full),
            "configuration_key_length": len("Tracking_1C.trace_handoff_filename="),
            "full_ini_line_length": len("Tracking_1C.trace_handoff_filename=") + len(full),
            "receiver_error_path": truncated,
            "receiver_error_path_length": len(truncated),
            "receiver_exit_code": manifest["exit_code"],
            "observed_dump_file_count": manifest["replay_validation"]["observed_dump_file_count"],
        },
        "authorized_change": "Mirror the already frozen handoff bytes under the shorter R2d SSD root and regenerate only Tracking_1C.trace_handoff_filename values.",
        "required_identity": "Every mirrored handoff SHA-256 must equal its committed R2d artifact counterpart before any replay.",
        "preservation": "Rename and retain the failed rep3 directory and durable run; rerun the exact rep3 role only after the path mirror/config freeze passes.",
        "unchanged": [
            "receiver executable and R2c terminal-drain patch",
            "handoff CSV bytes and state selection",
            "raw IQ sources, hashes, sample ranges, and replay durations",
            "TRACE scoring, gates, windows, tolerances, clean split, controls, and alarm policy",
            "Phase A and Phase B scientific contracts",
        ],
        "attack_data_read_or_scored": False,
    }
    (ARTIFACT / "config_path_length_amendment_preregistered.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "failure_label": payload["failure_label"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
