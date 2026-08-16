#!/usr/bin/env python3
"""Authenticate TRACE input fields without computing attack scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_static"
RECEIVER_SOURCE = Path("/home/ubuntu/build-gnss-sdr-complex9")

INPUTS = {
    "TEXBAT": {
        "cleanStatic": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9"),
        "DS1": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds1-complex9"),
        "DS3": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds3-complex9"),
        "DS7": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/receiver/ds7-complex9"),
    },
    "OAKBAT": {
        name: Path(f"/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/{name.lower()}/receiver/{name.lower()}-complex9")
        for name in ("cleanStatic", "OS1", "OS3", "OS4")
    },
}
TAP_FIELDS = [f"{part}_{tap}" for tap in ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4") for part in ("I", "Q")]
CONTRACT_FIELDS = TAP_FIELDS + [
    "Prompt_I", "Prompt_Q", "code_error_chips", "code_error_filt_chips",
    "carr_error_hz", "carr_error_filt_hz", "carrier_doppler_hz",
    "code_freq_chips", "aux1", "PRN_start_sample_count", "PRN",
    "CN0_SNV_dB_Hz", "carrier_lock_test",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, object] = {
        "schema": "gnss-doppler-lab.trace-input-inventory.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_contract": CONTRACT_FIELDS,
        "datasets": {},
    }
    bindings: dict[str, object] = {
        "schema": "gnss-doppler-lab.trace-source-binding.v1",
        "receiver_source": str(RECEIVER_SOURCE),
        "receiver_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RECEIVER_SOURCE, text=True).strip(),
        "receiver_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"], cwd=RECEIVER_SOURCE)).hexdigest(),
        "datasets": {},
    }
    all_valid = True
    for dataset, scenarios in INPUTS.items():
        inventory["datasets"][dataset] = {}
        bindings["datasets"][dataset] = {}
        for scenario, root in scenarios.items():
            manifest = read_manifest(root)
            mats = sorted((root / "raw").glob("epl_tracking_ch_*.mat"))
            if not mats:
                raise FileNotFoundError(f"no receiver MAT files: {root}")
            with h5py.File(mats[0], "r") as handle:
                fields = sorted(handle.keys())
                shapes = {name: list(handle[name].shape) for name in fields}
            missing = sorted(set(CONTRACT_FIELDS) - set(fields))
            same_contract = not missing
            all_valid &= same_contract
            source = manifest.get("source", {})
            auth = manifest.get("authenticated_inputs", {})
            iq_sha = source.get("iq_sha256") or auth.get("iq_before_receiver", {}).get("sha256")
            iq_path = source.get("iq") or auth.get("iq_before_receiver", {}).get("path")
            sample_rate = source.get("sample_rate_hz") or 25_000_000
            inventory["datasets"][dataset][scenario] = {
                "receiver_root": str(root), "mat_count": len(mats), "first_mat_fields": fields,
                "first_mat_shapes": shapes, "missing_contract_fields": missing,
                "full_complex_9tap": all(field in fields for field in TAP_FIELDS),
                "epl_complex": all(field in fields for field in ("I_E", "Q_E", "I_P", "Q_P", "I_L", "Q_L")),
                "sample_rate_hz": sample_rate,
            }
            bindings["datasets"][dataset][scenario] = {
                "raw_iq_path": iq_path, "raw_iq_sha256": iq_sha,
                "receiver_manifest_path": str(root / "manifest.json"),
                "receiver_manifest_sha256": sha256(root / "manifest.json"),
                "receiver_config_path": str(root / "receiver.conf"),
                "receiver_config_sha256": sha256(root / "receiver.conf"),
                "source_mat_count": len(mats),
                "source_mat_sha256": {path.name: sha256(path) for path in mats},
            }
    inventory["same_feature_contract_texbat_oakbat"] = all_valid
    timeline = {
        "schema": "gnss-doppler-lab.trace-timeline-inventory.v1",
        "DS1": {"role": "boundary", "onset_s": 100.0},
        "DS3": {"role": "development", "signal_onset_s": 118.9, "pull_off_s": 195.0},
        "DS7": {"role": "development", "injection_s": 110.0, "transition": [110.0, 130.0], "held": [130.0, 150.0], "time_push_s": 150.0},
        "OS1": {"role": "boundary", "onset_s": 120.0},
        "OS3": {"role": "confirmation", "onset_s": 120.0, "transition": [120.0, 130.0]},
        "OS4": {"role": "confirmation", "onset_s": 120.0, "transition": [120.0, 130.0]},
    }
    for name, payload in (("input_inventory.json", inventory), ("source_binding.json", bindings), ("timeline_inventory.json", timeline)):
        (args.out_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if all_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
