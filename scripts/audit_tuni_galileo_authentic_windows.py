#!/usr/bin/env python3
"""Scan preregistered later TUNI windows for trackable authentic PRNs."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/ubuntu/unraid_hdd/tuni2025/galileo")
DEFAULT_EXECUTABLE = ROOT / ".tools" / "gnss-sdr-galileo-complex9-safety-v1"
EXPECTED_RECEIVER_SHA256 = "3deb86bd586fe1db3ff681f8e67d357cbf6f32a3cba26258642d1ab9e79a3bdb"
RELEASE_TOKEN = "RELEASE-TUNI-GALILEO-AUTHENTIC-WINDOW-SCAN-V1"
OFFSETS_SECONDS = (30, 90, 150, 210, 270)
DURATION_SECONDS = 10.0
SUSTAINED_EPOCH_THRESHOLD = 1000
SCENARIOS = {
    "SS-11": ("SS-11/Galileo_1_Spoofer_Static_MP_TruePosition.bin", {31}),
    "SS-12": ("SS-12/Galileo_2_Spoofer_Static_MP_TruePosition.bin", {9, 31}),
    "SS-13": ("SS-13/Galileo_4_Spoofer_Static_MP_TruePosition.bin", {5, 9, 23, 31}),
}


def _load_support_audit() -> Any:
    path = ROOT / "scripts" / "audit_tuni_galileo_prn_support_v1_1.py"
    spec = importlib.util.spec_from_file_location("tuni_prn_support_v1_1_window_scan", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SUPPORT = _load_support_audit()


def render_window_config(
    *, iq_path: Path, output_dir: Path, offset_s: int, authentic_prns: tuple[int, ...]
) -> str:
    if offset_s not in OFFSETS_SECONDS:
        raise ValueError("offset is not in the frozen window roster")
    text = SUPPORT.render_fixed_prn_config(
        iq_path=iq_path,
        output_dir=output_dir,
        duration_s=DURATION_SECONDS,
        prns=authentic_prns,
    )
    return text.replace(
        f"SignalSource.filename={iq_path}\n",
        f"SignalSource.filename={iq_path}\nSignalSource.seconds_to_skip={offset_s}\n",
    )


def git_clean_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    if status.strip():
        raise RuntimeError("release requires a clean git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-token", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "artifacts" / "tuni_galileo_authentic_window_scan_v1",
    )
    parser.add_argument("--timeout-s", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.release_token != RELEASE_TOKEN:
        raise ValueError("release token mismatch")
    release_commit = git_clean_commit()
    executable_match = shutil.which(str(args.executable))
    executable = Path(executable_match).resolve() if executable_match else args.executable.resolve()
    if SUPPORT.sha256(executable) != EXPECTED_RECEIVER_SHA256:
        raise ValueError("receiver SHA-256 mismatch")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    scenario_rows: list[dict[str, Any]] = []
    for scenario_id, (relative, spoofed_prns) in SCENARIOS.items():
        iq_path = (args.data_root / relative).resolve()
        if not iq_path.is_file():
            raise FileNotFoundError(iq_path)
        authentic_prns = tuple(prn for prn in range(1, 37) if prn not in spoofed_prns)
        if len(authentic_prns) > 35:
            raise ValueError("GNSS-SDR v0.0.19 supports at most 35 Galileo E1 channels")
        windows: list[dict[str, Any]] = []
        for offset_s in OFFSETS_SECONDS:
            window_dir = output_root / scenario_id.lower() / f"offset_{offset_s:03d}"
            raw_dir = window_dir / "raw"
            raw_dir.mkdir(parents=True)
            config_path = window_dir / "receiver.conf"
            config_path.write_text(
                render_window_config(
                    iq_path=iq_path, output_dir=window_dir, offset_s=offset_s,
                    authentic_prns=authentic_prns,
                ),
                encoding="utf-8",
            )
            command = [str(executable), f"--config_file={config_path}", "--keyboard=false"]
            completed = subprocess.run(
                command, cwd=window_dir, capture_output=True, text=True,
                timeout=args.timeout_s, check=False,
            )
            (window_dir / "receiver.log").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            if completed.returncode != 0:
                raise RuntimeError(f"receiver failed for {scenario_id} at {offset_s} seconds")
            support = SUPPORT.channel_support(sorted(raw_dir.glob("epl_tracking_ch_*.mat")))
            for row in support:
                channel = row["channel"]
                if channel >= len(authentic_prns) or row["prn"] != authentic_prns[channel]:
                    raise ValueError(f"fixed-PRN channel mismatch for {scenario_id} at {offset_s}")
            sustained = sorted({
                row["prn"] for row in support
                if row["valid_epochs"] >= SUSTAINED_EPOCH_THRESHOLD
            })
            windows.append({
                "offset_seconds": offset_s,
                "receiver_config_sha256": SUPPORT.sha256(config_path),
                "supported_channels": support,
                "sustained_authentic_prns": sustained,
                "control_pair_available": len(sustained) >= 2,
            })
        scenario_rows.append({
            "scenario": scenario_id,
            "excluded_spoof_prns": sorted(spoofed_prns),
            "searched_authentic_prns": list(authentic_prns),
            "windows": windows,
            "any_control_pair_window": any(row["control_pair_available"] for row in windows),
        })
        print(f"[tuni-authentic-window-scan] {scenario_id} complete", flush=True)

    summary = {
        "schema": "gnss-doppler-lab.tuni-galileo-authentic-window-scan.v1",
        "scope": "receiver-input support only; no detector scores recomputed",
        "release_commit": release_commit,
        "receiver_sha256": SUPPORT.sha256(executable),
        "offsets_seconds": list(OFFSETS_SECONDS),
        "duration_seconds": DURATION_SECONDS,
        "sustained_epoch_threshold": SUSTAINED_EPOCH_THRESHOLD,
        "scenarios": scenario_rows,
        "all_scenarios_have_control_pair_window": all(
            row["any_control_pair_window"] for row in scenario_rows
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
