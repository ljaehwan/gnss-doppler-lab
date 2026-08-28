#!/usr/bin/env python3
"""Audit TUNI Galileo PRN support with an exhaustive fixed-PRN receiver bank.

This is a receiver-input diagnostic, not a detector rerun.  It preserves the
released detector outcome and uses the same acquisition/tracking parameters,
while assigning one channel to every Galileo PRN so the automatic scheduler
cannot spend the short observation only on the strongest spoof component.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/ubuntu/unraid_hdd/tuni2025/galileo")
DEFAULT_EXECUTABLE = ROOT / ".tools" / "gnss-sdr-galileo-complex9-safety-v1"
SCENARIOS = {
    "SS-11": ("SS-11/Galileo_1_Spoofer_Static_MP_TruePosition.bin", {31}),
    "SS-12": ("SS-12/Galileo_2_Spoofer_Static_MP_TruePosition.bin", {9, 31}),
    "SS-13": ("SS-13/Galileo_4_Spoofer_Static_MP_TruePosition.bin", {5, 9, 23, 31}),
}
EXPECTED_RECEIVER_SHA256 = "3deb86bd586fe1db3ff681f8e67d357cbf6f32a3cba26258642d1ab9e79a3bdb"
RELEASE_TOKEN = "RELEASE-TUNI-GALILEO-PRN-SUPPORT-AUDIT-V1"


def _load_preflight() -> Any:
    path = ROOT / "scripts" / "preflight_tuni_galileo_clean.py"
    spec = importlib.util.spec_from_file_location("tuni_clean_preflight_support_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load_preflight()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_fixed_prn_config(*, iq_path: Path, output_dir: Path, duration_s: float) -> str:
    text = PREFLIGHT.render_config(
        iq_path=iq_path,
        output_dir=output_dir,
        input_samples=PREFLIGHT.ishort_source_item_count(duration_s, 50_000_000),
        channel_count=36,
        tracking_tap_count=9,
        input_rate_hz=50_000_000,
        internal_rate_hz=12_500_000,
    )
    assignments = "\n".join(f"Channel{index}.satellite={index + 1}" for index in range(36))
    return text.replace("Channel.signal=1B\n", f"Channel.signal=1B\n{assignments}\n")


def channel_support(mat_paths: list[Path]) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for path in mat_paths:
        with h5py.File(path, "r") as handle:
            if "PRN" not in handle:
                continue
            values = np.asarray(handle["PRN"]).reshape(-1)
            valid = [int(value) for value in values if 1 <= int(value) <= 36]
        if not valid:
            continue
        counts = {prn: valid.count(prn) for prn in sorted(set(valid))}
        for prn, epochs in counts.items():
            rows.append({"channel": int(path.stem.rsplit("_", 1)[-1]), "prn": prn, "valid_epochs": epochs})
    return sorted(rows, key=lambda row: (row["prn"], row["channel"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-token", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "tuni_galileo_prn_support_audit_v1")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--timeout-s", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.release_token != RELEASE_TOKEN:
        raise ValueError("release token mismatch")
    if args.duration_s != 10.0:
        raise ValueError("v1 audit duration is frozen at 10.0 seconds")
    executable_match = shutil.which(str(args.executable))
    executable = Path(executable_match).resolve() if executable_match else args.executable.resolve()
    if sha256(executable) != EXPECTED_RECEIVER_SHA256:
        raise ValueError("receiver SHA-256 mismatch")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    summaries: list[dict[str, Any]] = []
    for scenario_id, (relative, spoofed) in SCENARIOS.items():
        iq_path = (args.data_root / relative).resolve()
        if not iq_path.is_file():
            raise FileNotFoundError(iq_path)
        scenario_dir = output_root / scenario_id.lower()
        raw_dir = scenario_dir / "raw"
        raw_dir.mkdir(parents=True)
        config_path = scenario_dir / "receiver.conf"
        config_path.write_text(
            render_fixed_prn_config(iq_path=iq_path, output_dir=scenario_dir, duration_s=args.duration_s),
            encoding="utf-8",
        )
        command = [str(executable), f"--config_file={config_path}", "--keyboard=false"]
        completed = subprocess.run(
            command, cwd=scenario_dir, capture_output=True, text=True,
            timeout=args.timeout_s, check=False,
        )
        (scenario_dir / "receiver.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        support = channel_support(sorted(raw_dir.glob("epl_tracking_ch_*.mat")))
        sustained = [row for row in support if row["valid_epochs"] >= 1000]
        authentic = sorted({row["prn"] for row in sustained if row["prn"] not in spoofed})
        targets = sorted({row["prn"] for row in sustained if row["prn"] in spoofed})
        summaries.append({
            "scenario": scenario_id,
            "return_code": completed.returncode,
            "spoofed_prns": sorted(spoofed),
            "all_supported_channels": support,
            "sustained_epoch_threshold": 1000,
            "sustained_target_prns": targets,
            "sustained_authentic_prns": authentic,
            "same_stream_control_available": len(authentic) >= 2,
            "receiver_config_sha256": sha256(config_path),
        })
        print(f"[tuni-prn-support] {scenario_id} complete", flush=True)
        if completed.returncode != 0:
            raise RuntimeError(f"receiver failed for {scenario_id}")

    summary = {
        "schema": "gnss-doppler-lab.tuni-galileo-prn-support-audit.v1",
        "scope": "receiver-input support audit; released detector result is not recomputed",
        "duration_seconds": args.duration_s,
        "fixed_prn_roster": list(range(1, 37)),
        "receiver_sha256": sha256(executable),
        "scenarios": summaries,
        "all_scenarios_have_same_stream_controls": all(row["same_stream_control_available"] for row in summaries),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
